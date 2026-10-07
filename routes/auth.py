from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify, make_response
from models import User, db
from models.activity_log import ActivityLog
from werkzeug.security import check_password_hash
from authlib.integrations.flask_client import OAuth
from datetime import datetime
import os, secrets, string, json
from functools import wraps
from extensions import limiter

auth_bp = Blueprint('auth', __name__)


def log_activity(user_id, user_type, user_email, action, details=None):
    """Log user activity - will be imported from main app or defined here"""
    from flask import request
    try:
        ip_address = request.remote_addr if request else '127.0.0.1'
        log = ActivityLog(
            user_id=user_id,
            user_type=user_type,
            user_email=user_email,
            action=action,
            details=details,
            ip_address=ip_address
        )
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        print(f"Error logging activity: {e}")


def no_cache(f):
    """Decorator to prevent browser caching of protected pages"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        response = make_response(f(*args, **kwargs))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return decorated_function


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        user = User.query.get(session['user_id'])
        if user and not user.is_active:
            session.clear()
            flash('Your account has been deactivated.', 'error')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


def _login_email_key():
    """Per-account rate key: email + IP."""
    email = (request.form.get('email') or '').strip().lower()
    return f"{email}:{request.remote_addr or 'unknown'}"


# ======================================================================
# PUBLIC ENDPOINT ALLOWLIST (checked FIRST in the guard, wins over
# protected prefixes). Password reset must be here — the user is
# locked out and has no session.
# ======================================================================
PUBLIC_API_ENDPOINTS = {
    # Live map & navigation
    'api_other.live_map_directions_v3',
    'api_other.live_map_directions_v2',
    'api_other.live_map_directions',
    'api_other.live_map_directions_now',
    'api_other.travel_prediction',

    # Stations & info
    'api_other.get_stations',
    'api_other.station_info',
    'api_other.get_recommendation',

    # Alerts & broadcasts (public reads)
    'api_other.alerts_count',
    'api_other.alerts_list',
    'api_other.get_public_broadcasts',

    # Predictions
    'api_predict.predict_congestion',
    'api_predict.predict_direction',
    'api_predict.predict_route',
    'api_predict.directional_forecast',
    'api_predict.directional_forecast_all',
    'api_predict.model_evaluation',
    'api_predict.confusion_matrix_endpoint',
    'api_predict.test_rush_hour',

    # Schedules
    'api_schedule.get_headway_route',
    'api_schedule.get_next_trains_route',
    'api_schedule.station_info_route',
    'api_schedule.schedule_with_congestion',
    'api_schedule.compare_stations',
    'api_schedule.test_schedule',
    'api_schedule.test_time_schedule',

    # Public reports
    'api_reports.get_reports',
    'api_reports.get_remaining_reports',
    'api_reports.report_congestion',
    'api_reports.predict_station',

    # Historical
    'api_other.historical_patterns',

    # Health & session
    'model_performance.health_check',
    'api_test',
    'api_other.test_api',
    'auth.check_session',

    'email.request_password_reset',
    'email.verify_code_endpoint',
    'email.reset_password',
}

PUBLIC_PAGE_ENDPOINTS = {
    'auth.login',
    'auth.google_login',
    'auth.google_authorize',
    'auth.google_operator_login',
    'auth.signup',
    'auth.operator_signup',
    'auth.check_session',
    'static',
    'public.home',
    'public.live_map',
    'public.travel_plan',
    'public.alerts',
    'public.report',
    'user.user_dashboard',
}


# ======================================================================
# SESSION VALIDITY GUARD
# ======================================================================
@auth_bp.before_app_request
def check_session_validity():
    """Check if session is valid on every request - prevents back button access"""
    from flask import request, session, flash, redirect, url_for, jsonify

    path = request.path or ''
    endpoint = request.endpoint or ''

    # ------------------------------------------------------------------
    # 0. PUBLIC ALLOWLIST — checked FIRST so it wins over protected prefixes.
    # ------------------------------------------------------------------
    if endpoint in PUBLIC_API_ENDPOINTS:
        return None
    if endpoint in PUBLIC_PAGE_ENDPOINTS:
        return None

    # ------------------------------------------------------------------
    # 1. ALWAYS-PROTECTED PREFIXES
    # ------------------------------------------------------------------
    protected_prefixes = (
        '/api/admin/',
        '/api/operator/',
        '/api/debug/',
        '/debug/',
        '/admin/',
        '/operator/',
        '/api/retrain',
        '/api/model/',
        '/api/profile/',
        '/api/saved-routes',
        '/api/user-reports/',
        '/api/reports/',
        '/uploads/',
    )

    if path.startswith(protected_prefixes):
        # Env-based admin bypass
        if session.get('is_admin') and session.get('role') == 'admin':
            return None

        if 'user_id' not in session:
            return jsonify({'error': 'unauthorized'}), 401

        user = User.query.get(session['user_id'])
        if not user or not user.is_active:
            session.clear()
            return jsonify({'error': 'session expired'}), 401
        return None

    # ------------------------------------------------------------------
    # 2. DEFAULT for non-API pages → require login
    # ------------------------------------------------------------------
    if not path.startswith('/api/'):
        if session.get('is_admin') and session.get('role') == 'admin':
            return None

        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login'))

        user = User.query.get(session['user_id'])
        if not user or not user.is_active:
            session.clear()
            flash('Your session has expired. Please log in again.', 'warning')
            return redirect(url_for('auth.login'))

    return None


@auth_bp.route('/api/check-session')
@limiter.limit("60 per minute")
def check_session():
    """Check if user session is still valid - used by frontend JavaScript"""
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and user.is_active:
            return jsonify({
                'logged_in': True,
                'username': session.get('username'),
                'role': session.get('role')
            })
    return jsonify({'logged_in': False})


@auth_bp.route('/login', methods=['GET', 'POST'])
@no_cache
@limiter.limit("20 per minute", methods=["POST"])
@limiter.limit("100 per hour", methods=["POST"])
@limiter.limit("10 per minute", key_func=_login_email_key, methods=["POST"])
def login():
    """Login page - with no-cache to prevent back button issues"""
    error = None
    error_type = None

    if request.method == 'GET':
        if 'user_id' in session:
            user = User.query.get(session['user_id'])
            if user and user.is_active:
                if user.role == 'admin':
                    return redirect(url_for('admin.admin_dashboard'))
                elif user.role == 'operator':
                    return redirect(url_for('operator.operator_dashboard'))
                else:
                    return redirect(url_for('user.user_dashboard'))
            else:
                session.clear()

    # ========== INVITE HANDLING (GET only) ==========
    invite_token = request.args.get('invite')

    if invite_token:
        operator = User.query.filter_by(invite_token=invite_token, role='operator').first()

        if not operator:
            flash('This invitation is invalid.', 'error')
            return redirect(url_for('auth.login'))

        if operator.is_invite_expired():
            flash('This invitation has expired. Please request a new one from the administrator.', 'error')
            return redirect(url_for('auth.login'))

        return render_template('operator_signup.html',
                               email=operator.username,
                               invite_token=invite_token,
                               assigned_station=operator.favorite_station or 'All Stations')

    # ========== NORMAL LOGIN (POST) ==========
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        ip_address = request.remote_addr

        admin_email = os.getenv('ADMIN_EMAIL')
        admin_password_hash = os.getenv('ADMIN_PASSWORD_HASH')

        from werkzeug.security import check_password_hash, generate_password_hash
        if admin_email and admin_password_hash and email == admin_email and check_password_hash(admin_password_hash, password):
            session.clear()
            session.permanent = True
            session['is_admin'] = True
            session['role'] = 'admin'
            session['username'] = email
            session['user_id'] = None
            log_activity(None, 'admin', email, 'admin_env_login', f'IP: {request.remote_addr}')
            return redirect(url_for('admin.admin_dashboard'))

        user = User.query.filter_by(username=email).first()

        if user is None:
            error = "Account not found."
            error_type = "error"
        elif not user.is_active:
            error = "Account deactivated."
            error_type = "error"
        elif user.google_id and not user.has_password():
            error = "This account uses Google Sign-In. Please click 'Continue with Google'."
            error_type = "info"
        else:
            result = user.verify_password(password)

            if result:
                session.clear()
                session.permanent = True
                session['user_id'] = user.id
                session['username'] = user.username
                session['role'] = user.role
                session['favorite_station'] = user.favorite_station
                session['google_user'] = False

                user.last_login = datetime.now()
                db.session.commit()

                log_activity(user.id, user.role, user.username, 'login_success',
                             f'Logged in from IP: {ip_address}')

                if user.role == 'admin':
                    return redirect(url_for('admin.admin_dashboard'))
                elif user.role == 'operator':
                    return redirect(url_for('operator.operator_dashboard'))
                else:
                    return redirect(url_for('user.user_dashboard'))
            else:
                log_activity(user.id, user.role, user.username, 'login_failed',
                             f'Incorrect password from IP: {ip_address}')
                error = "Incorrect password."
                error_type = "error"

    return render_template('login.html', error=error, error_type=error_type)


@auth_bp.route('/signup', methods=['GET', 'POST'])
@no_cache
@limiter.limit("5 per hour", methods=["POST"])
def signup():
    """Signup page - with no-cache"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        favorite = request.form.get('favorite_station')

        if password != confirm_password:
            return render_template('signup.html', error='Passwords do not match', email=email, favorite=favorite)

        existing_user = User.query.filter_by(username=email).first()
        if existing_user:
            return redirect(url_for('auth.signup', error='email_exists', email=email))

        new_user = User(
            username=email,
            role='commuter',
            favorite_station=favorite if favorite else None,
            created_at=datetime.now(),
            last_login=datetime.now()
        )
        new_user.password = password

        try:
            db.session.add(new_user)
            db.session.commit()

            session.clear()
            session.permanent = True
            session['user_id'] = new_user.id
            session['username'] = new_user.username
            session['role'] = 'commuter'
            session['favorite_station'] = new_user.favorite_station

            flash('Account created successfully!', 'success')
            return redirect(url_for('user.user_dashboard'))
        except Exception as e:
            db.session.rollback()
            return render_template('signup.html', error="Database error. Please try again.")

    error = request.args.get('error')
    email = request.args.get('email')
    return render_template('signup.html', error=error, email=email)


@auth_bp.route('/logout')
@no_cache
def logout():
    """Logout - with no-cache and session clearing"""
    if 'user_id' in session:
        log_activity(session.get('user_id'), session.get('role'),
                     session.get('username'), 'logout', 'User logged out')
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))


@auth_bp.route('/login/google')
@no_cache
@limiter.limit("20 per minute")
def google_login():
    """Initiate Google OAuth login with account selection options"""
    from flask import current_app

    google = current_app.config.get('GOOGLE_CLIENT')

    if google is None:
        flash('Google authentication is not configured. Please contact administrator.', 'error')
        return redirect(url_for('auth.login'))

    redirect_uri = url_for('auth.google_authorize', _external=True)

    client_kwargs = {
        'scope': 'openid email profile',
        'prompt': 'select_account'
    }

    return google.authorize_redirect(redirect_uri, **client_kwargs)


@auth_bp.route('/login/google/operator')
@no_cache
@limiter.limit("20 per minute")
def google_operator_login():
    """Initiate Google OAuth login with operator context"""
    from flask import current_app, request

    invite_token = request.args.get('invite')
    email = request.args.get('email')
    station = request.args.get('station')

    google = current_app.config.get('GOOGLE_CLIENT')

    if google is None:
        flash('Google authentication is not configured.', 'error')
        return redirect(url_for('auth.login'))

    if invite_token:
        session['invite_token'] = invite_token
    if email:
        session['invite_email'] = email
    if station:
        session['invite_station'] = station
    session['is_operator_signup'] = True

    redirect_uri = url_for('auth.google_authorize', _external=True)

    client_kwargs = {
        'scope': 'openid email profile',
        'prompt': 'select_account'
    }

    return google.authorize_redirect(redirect_uri, **client_kwargs)


@auth_bp.route('/login/google/authorize')
@no_cache
@limiter.limit("30 per minute")
def google_authorize():
    """Handle Google OAuth callback - With proper logging"""
    from flask import current_app

    try:
        google = current_app.config.get('GOOGLE_CLIENT')

        if google is None:
            flash('Google authentication is not configured.', 'error')
            return redirect(url_for('auth.login'))

        token = google.authorize_access_token()

        if not token:
            flash('Failed to get access token from Google.', 'error')
            return redirect(url_for('auth.login'))

        user_info = google.parse_id_token(token)

        if not user_info or 'email' not in user_info:
            flash('Failed to get user information from Google.', 'error')
            return redirect(url_for('auth.login'))

        email = user_info.get('email')
        name = user_info.get('name', email.split('@')[0])
        google_id = user_info.get('sub')
        ip_address = request.remote_addr

        user = User.query.filter_by(username=email).first()

        if not user:
            log_activity(None, 'unknown', email, 'login_failed',
                         f'Google login failed - no account found from IP: {ip_address}')
            flash(f'No account found for {email}. Please contact administrator.', 'error')
            return redirect(url_for('auth.login'))

        if not user.is_active:
            log_activity(user.id, user.role, user.username, 'login_failed',
                         f'Google login failed - account deactivated from IP: {ip_address}')
            flash('Your account is deactivated. Please contact administrator.', 'error')
            return redirect(url_for('auth.login'))

        if not user.google_id:
            user.google_id = google_id
            db.session.commit()
            log_activity(user.id, user.role, user.username, 'link_google',
                         f'Google account linked from IP: {ip_address}')

        if session.get('is_operator_signup') and user.role == 'operator':
            user.clear_invite()
            db.session.commit()

        user.last_login = datetime.now()
        db.session.commit()

        session.clear()
        session.permanent = True
        session['user_id'] = user.id
        session['username'] = user.username
        session['role'] = user.role
        session['favorite_station'] = user.favorite_station
        session['google_user'] = True

        log_activity(user.id, user.role, user.username, 'login',
                     f'Google login successful from IP: {ip_address}')

        flash(f'Welcome back, {name}!', 'success')

        if user.role == 'admin':
            return redirect(url_for('admin.admin_dashboard'))
        elif user.role == 'operator':
            return redirect(url_for('operator.operator_dashboard'))
        else:
            return redirect(url_for('user.user_dashboard'))

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f'Google login failed: {str(e)}', 'error')
        return redirect(url_for('auth.login'))


@auth_bp.route('/operator-signup', methods=['POST'])
@no_cache
@limiter.limit("10 per hour")
def operator_signup():
    """Handle operator signup from invitation"""
    try:
        email = request.form.get('email')
        invite_token = request.form.get('invite_token')
        new_password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        name = request.form.get('name')
        station = request.form.get('station')

        if not email:
            return jsonify({'success': False, 'error': 'Email is required'}), 400

        if not new_password or len(new_password) < 8:
            return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400

        if new_password != confirm_password:
            return jsonify({'success': False, 'error': 'Passwords do not match'}), 400

        operator = User.query.filter_by(username=email, role='operator').first()

        if not operator:
            return jsonify({'success': False, 'error': 'Invalid invitation - user not found'}), 401

        if not invite_token or operator.invite_token != invite_token:
            return jsonify({'success': False, 'error': 'Invalid or already-used invitation'}), 401

        if operator.is_invite_expired():
            return jsonify({'success': False, 'error': 'This invitation has expired. Please request a new one.'}), 401

        operator.password = new_password
        operator.is_active = True
        operator.clear_invite()

        if station and station != 'All Stations':
            operator.favorite_station = station

        db.session.commit()

        log_activity(operator.id, 'operator', operator.username, 'signup_complete',
                     f'Operator account activated from invitation')

        session.clear()
        session.permanent = True
        session['user_id'] = operator.id
        session['username'] = operator.username
        session['role'] = 'operator'
        session['favorite_station'] = operator.favorite_station

        operator.last_login = datetime.now()
        db.session.commit()

        return jsonify({
            'success': True,
            'redirect': '/operator-dashboard',
            'message': 'Account created successfully!'
        })

    except Exception as e:
        print(f"❌ Error in operator signup: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500