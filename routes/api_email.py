from flask import Blueprint, request, jsonify, session, current_app
from models import User, db
from datetime import datetime, timedelta
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback

email_bp = Blueprint('email', __name__)

# ========== EMAIL CONFIGURATION ==========
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email': 'junnied1405@gmail.com',
    'password': 'ppkmjgcydeofmejj'
}

# ========== PASSWORD RESET TOKENS ==========
reset_codes = {}

def send_reset_email(email, code):
    """Send reset code via email"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['email']
        msg['To'] = email
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
                    <strong>Note:</strong> This code will expire in 5 minutes.
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
        
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
            server.send_message(msg)
        
        return True
    except Exception as e:
        current_app.logger.error(f"Failed to send email: {e}")
        return False

def generate_reset_code(email):
    """Generate a 6-digit reset code"""
    code = ''.join(secrets.choice('0123456789') for _ in range(6))
    reset_codes[email] = {
        'code': code,
        'expires': datetime.now() + timedelta(minutes=5),
        'attempts': 0
    }
    return code

def verify_reset_code(email, code):
    """Verify reset code"""
    if email not in reset_codes:
        return False, 'No reset request found'
    
    data = reset_codes[email]
    
    if data['attempts'] >= 3:
        del reset_codes[email]
        return False, 'Too many attempts. Request a new code.'
    
    if datetime.now() > data['expires']:
        del reset_codes[email]
        return False, 'Code expired. Request a new one.'
    
    if data['code'] != code:
        data['attempts'] += 1
        return False, f'Invalid code. {3 - data["attempts"]} attempts remaining.'
    
    del reset_codes[email]
    return True, 'Code verified'

@email_bp.route('/update-password', methods=['POST'])
def update_password():
    """Update user password"""
    try:
        data = request.json
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not current_password or not new_password:
            return jsonify({'success': False, 'error': 'Current password and new password are required'}), 400
        
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        if not user.verify_password(current_password):
            return jsonify({'success': False, 'error': 'Current password is incorrect'}), 400
        
        if len(new_password) < 8:
            return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400
        
        from werkzeug.security import generate_password_hash
        user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        
        current_app.logger.info(f"Password updated for user: {user.username}")
        
        return jsonify({'success': True, 'message': 'Password updated successfully'})
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error updating password: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@email_bp.route('/debug-check-user/<email>', methods=['GET'])
def debug_check_user(email):
    """Debug endpoint to check user password hash"""
    try:
        user = User.query.filter_by(username=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'username': user.username,
            'password_hash': user.password_hash[:30] + '...' if user.password_hash else 'None',
            'hash_length': len(user.password_hash) if user.password_hash else 0,
            'has_hash': bool(user.password_hash)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@email_bp.route('/request-reset', methods=['POST'])
def request_password_reset():
    """Request password reset - sends code to email"""
    try:
        data = request.json
        email = data.get('email')
        
        current_app.logger.info(f"Password reset requested for: {email}")
        
        if not email:
            return jsonify({'success': False, 'error': 'Email is required'}), 400
        
        user = User.query.filter_by(username=email).first()
        
        if not user:
            current_app.logger.info(f"No user found with email: {email}")
            return jsonify({'success': True, 'message': 'If an account exists, a reset code was sent.'})
        
        current_app.logger.info(f"User found: {user.username} (ID: {user.id})")
        
        code = generate_reset_code(email)
        current_app.logger.info(f"Generated reset code for {email}: {code}")
        
        if send_reset_email(email, code):
            return jsonify({
                'success': True,
                'message': 'Reset code sent to your email',
                'email': email
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to send email. Please try again.'}), 500
            
    except Exception as e:
        current_app.logger.error(f"Error requesting reset: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@email_bp.route('/verify-code', methods=['POST'])
def verify_reset_code_endpoint():
    """Verify the reset code"""
    try:
        data = request.json
        email = data.get('email')
        code = data.get('code')
        
        if not email or not code:
            return jsonify({'success': False, 'error': 'Email and code are required'}), 400
        
        success, message = verify_reset_code(email, code)
        
        return jsonify({
            'success': success,
            'message': message
        })
        
    except Exception as e:
        current_app.logger.error(f"Error verifying code: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
@email_bp.route('/reset-password', methods=['POST'])
def reset_password():
    """Reset password after code verification"""
    try:
        data = request.json
        email = data.get('email')
        code = data.get('code')
        new_password = data.get('new_password')
        
        print(f"🔐 Reset password attempt for: {email}")
        
        if not email or not code or not new_password:
            return jsonify({'success': False, 'error': 'Email, code, and new password are required'}), 400
        
        success, message = verify_reset_code(email, code)
        if not success:
            return jsonify({'success': False, 'error': message}), 400
        
        user = User.query.filter_by(username=email).first()
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        print(f"👤 User found: {user.username}")
        
        # ========== FIX: Use pbkdf2:sha256 method ==========
        from werkzeug.security import generate_password_hash, check_password_hash
        
        # Use pbkdf2:sha256 method (compatible with your login)
        user.password_hash = generate_password_hash(new_password, method='pbkdf2:sha256')
        db.session.commit()
        
        print(f"✅ Password reset successfully for {user.username}")
        
        # Verify the password works
        db.session.refresh(user)
        test_result = check_password_hash(user.password_hash, new_password)
        print(f"   Password verification test: {'✅ PASS' if test_result else '❌ FAIL'}")
        print(f"   Hash method: {user.password_hash[:10] if user.password_hash else 'None'}...")
        
        if email in reset_codes:
            del reset_codes[email]
        
        if test_result:
            return jsonify({
                'success': True,
                'message': 'Password reset successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Password was updated but verification failed. Please try again.'
            }), 500
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error resetting password: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500