"""
Password reset + password update routes.
All secrets come from environment variables.
Reset codes are DB-backed, hashed, rate-limited, and single-use.
"""
import os
import secrets
import smtplib
import traceback
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from flask import Blueprint, request, jsonify, session, current_app
from werkzeug.security import check_password_hash

from models import User, PasswordReset, db
from extensions import limiter, get_email_from_request 


email_bp = Blueprint('email', __name__)


# ============================================================
# CONFIG  (loaded from environment — never hardcode secrets)
# ============================================================
EMAIL_CONFIG = {
    'smtp_server': os.environ.get('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port':   int(os.environ.get('SMTP_PORT', 587)),
    'email':       os.environ.get('SMTP_USER'),
    'password':    os.environ.get('SMTP_PASSWORD'),
    'from_name':   os.environ.get('SMTP_FROM_NAME', 'MRT-3'),
}

RESET_CODE_TTL_MINUTES   = 5
RESET_CODE_MAX_ATTEMPTS  = 3
MIN_PASSWORD_LENGTH      = 8


# ============================================================
# EMAIL SENDING
# ============================================================
def send_reset_email(to_email: str, code: str) -> bool:
    """Send the password-reset code. Returns True on success."""
    if not EMAIL_CONFIG['email'] or not EMAIL_CONFIG['password']:
        current_app.logger.error("SMTP credentials are not configured.")
        return False

    try:
        msg = MIMEMultipart()
        msg['From']    = f"{EMAIL_CONFIG['from_name']} <{EMAIL_CONFIG['email']}>"
        msg['To']      = to_email
        msg['Subject'] = 'MRT-3 Password Reset Code'

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #f8fafc;">
            <div style="background: #00224D; padding: 20px; text-align: center; border-radius: 12px 12px 0 0;">
                <h1 style="color: white; margin: 0;">MRT-3</h1>
                <p style="color: #94A3B8; margin: 5px 0 0;">Password Reset</p>
            </div>
            <div style="background: white; padding: 30px; border-radius: 0 0 12px 12px; border: 1px solid #E2E8F0; border-top: none;">
                <h2 style="color: #1E293B; margin-top: 0;">Hello,</h2>
                <p style="color: #475569; font-size: 16px; line-height: 1.6;">
                    You requested to reset your password. Use the code below to verify your identity:
                </p>
                <div style="text-align: center; margin: 30px 0; background: #F1F5F9; padding: 20px; border-radius: 12px;">
                    <span style="font-size: 36px; font-weight: 700; color: #3B82F6; letter-spacing: 8px; font-family: monospace;">{code}</span>
                </div>
                <p style="color: #475569; font-size: 14px;">
                    <strong>Note:</strong> This code will expire in {RESET_CODE_TTL_MINUTES} minutes.
                </p>
                <p style="color: #64748B; font-size: 13px; margin-top: 20px; padding-top: 20px; border-top: 1px solid #E2E8F0;">
                    If you didn't request this, please ignore this email or contact support.
                </p>
            </div>
            <div style="text-align: center; padding: 15px; color: #94A3B8; font-size: 12px;">
                &copy; 2024 MRT-3. All rights reserved.
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))

        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'], timeout=15) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
            server.send_message(msg)

        return True

    except Exception as e:
        current_app.logger.error(f"Failed to send reset email: {e}")
        return False


# ============================================================
# RESET-CODE HELPERS  (DB-backed, hashed, single-use)
# ============================================================
def _hash_code(code: str) -> str:
    """Hash a reset code using the same work-factor as passwords."""
    from werkzeug.security import generate_password_hash
    return generate_password_hash(code, method='pbkdf2:sha256')


def create_reset_code(user: User) -> str:
    """
    Generate a fresh 6-digit code for `user`, store it hashed,
    and invalidate any previous codes for the same user.
    Returns the plaintext code (to email it).
    """
    # Invalidate any existing codes for this user
    PasswordReset.query.filter_by(user_id=user.id, used=False).delete()

    code = ''.join(secrets.choice('0123456789') for _ in range(6))

    record = PasswordReset(
        user_id=user.id,
        code_hash=_hash_code(code),
        expires_at=datetime.utcnow() + timedelta(minutes=RESET_CODE_TTL_MINUTES),
        attempts=0,
        used=False,
    )
    db.session.add(record)
    db.session.commit()

    return code


def verify_reset_code(email: str, code: str) -> tuple[bool, str, PasswordReset | None]:
    """
    Verify a reset code WITHOUT consuming it (so /reset-password can
    re-verify). Returns (success, message, record).
    The caller is responsible for marking the record `used=True`
    after the password is actually changed.
    """
    user = User.query.filter_by(username=email).first()
    if not user:
        # Don't leak existence
        return False, 'Invalid or expired code', None

    record = (
        PasswordReset.query
        .filter_by(user_id=user.id, used=False)
        .order_by(PasswordReset.expires_at.desc())
        .first()
    )
    if not record:
        return False, 'Invalid or expired code', None

    if datetime.utcnow() > record.expires_at:
        db.session.delete(record)
        db.session.commit()
        return False, 'Code expired. Request a new one.', None

    if record.attempts >= RESET_CODE_MAX_ATTEMPTS:
        db.session.delete(record)
        db.session.commit()
        return False, 'Too many attempts. Request a new code.', None

    if not check_password_hash(record.code_hash, code):
        record.attempts += 1
        db.session.commit()
        remaining = RESET_CODE_MAX_ATTEMPTS - record.attempts
        return False, f'Invalid code. {remaining} attempts remaining.', None

    return True, 'Code verified', record


# ============================================================
# ROUTES
# ============================================================
@email_bp.route('/request-reset', methods=['POST'])
@limiter.limit("5 per hour")            # per IP
@limiter.limit("3 per hour", key_func=get_email_from_request)  # per email
def request_password_reset():
    """Send a reset code. Always returns the same generic message."""
    try:
        data  = request.json or {}
        email = (data.get('email') or '').strip().lower()

        if not email:
            return jsonify({'success': False, 'error': 'Email is required'}), 400

        user = User.query.filter_by(username=email).first()

        # Generic response regardless of whether the user exists
        if user:
            code = create_reset_code(user)
            send_reset_email(email, code)
            # NOTE: never log the code itself
            current_app.logger.info("Reset code issued for user_id=%s", user.id)

        return jsonify({
            'success': True,
            'message': 'If an account exists, a reset code was sent.',
        })

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error requesting reset: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Something went wrong.'}), 500


@email_bp.route('/verify-code', methods=['POST'])
@limiter.limit("20 per hour")
def verify_code_endpoint():
    """Non-consuming verification (used by the UI between steps)."""
    try:
        data  = request.json or {}
        email = (data.get('email') or '').strip().lower()
        code  = (data.get('code')  or '').strip()

        if not email or not code:
            return jsonify({'success': False, 'error': 'Email and code are required'}), 400

        ok, message, _record = verify_reset_code(email, code)
        return jsonify({'success': ok, 'message': message})

    except Exception as e:
        current_app.logger.error(f"Error verifying code: {e}")
        return jsonify({'success': False, 'error': 'Something went wrong.'}), 500


@email_bp.route('/reset-password', methods=['POST'])
@limiter.limit("10 per hour")
def reset_password():
    """Consume a verified code and set the new password."""
    try:
        data         = request.json or {}
        email        = (data.get('email') or '').strip().lower()
        code         = (data.get('code')  or '').strip()
        new_password = data.get('new_password') or ''

        if not email or not code or not new_password:
            return jsonify({'success': False, 'error': 'Email, code, and new password are required'}), 400

        if len(new_password) < MIN_PASSWORD_LENGTH:
            return jsonify({
                'success': False,
                'error': f'Password must be at least {MIN_PASSWORD_LENGTH} characters'
            }), 400

        ok, message, record = verify_reset_code(email, code)
        if not ok:
            return jsonify({'success': False, 'error': message}), 400

        user = User.query.filter_by(username=email).first()
        if not user:
            return jsonify({'success': False, 'error': 'Invalid or expired code'}), 400

        # Set password via the model helper (single source of truth)
        user.set_password(new_password)

        # Consume the code
        record.used = True
        record.used_at = datetime.utcnow()

        # Invalidate any other outstanding codes for this user
        PasswordReset.query.filter_by(user_id=user.id, used=False).update({'used': True})

        # Invalidate existing sessions (see note below)
        user.session_epoch = (user.session_epoch or 0) + 1

        db.session.commit()

        current_app.logger.info("Password reset completed for user_id=%s", user.id)
        return jsonify({'success': True, 'message': 'Password reset successfully'})

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error resetting password: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Something went wrong.'}), 500


@email_bp.route('/update-password', methods=['POST'])
@limiter.limit("10 per hour")
def update_password():
    """Change password for the currently logged-in user."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401

        data             = request.json or {}
        current_password = data.get('current_password') or ''
        new_password     = data.get('new_password') or ''

        if not current_password or not new_password:
            return jsonify({'success': False, 'error': 'Current and new password are required'}), 400

        if len(new_password) < MIN_PASSWORD_LENGTH:
            return jsonify({
                'success': False,
                'error': f'Password must be at least {MIN_PASSWORD_LENGTH} characters'
            }), 400

        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        if not user.verify_password(current_password):
            return jsonify({'success': False, 'error': 'Current password is incorrect'}), 400

        user.set_password(new_password)
        user.session_epoch = (user.session_epoch or 0) + 1

        db.session.commit()
        current_app.logger.info("Password updated for user_id=%s", user.id)

        return jsonify({'success': True, 'message': 'Password updated successfully'})

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error updating password: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Something went wrong.'}), 500