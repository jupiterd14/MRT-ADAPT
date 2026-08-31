from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify, make_response
from models import User, db
from models.activity_log import ActivityLog
from werkzeug.security import check_password_hash
from authlib.integrations.flask_client import OAuth
from datetime import datetime
import os, secrets, string, json
from functools import wraps

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
@auth_bp.before_app_request
def check_session_validity():
    """Check if session is valid on every request - prevents back button access"""
    from flask import request, session, flash, redirect, url_for
    
    # ========== CRITICAL: Allow ALL /api/ routes FIRST ==========
    # This must be the FIRST check!
    if request.path.startswith('/api/'):
        return None
    
    # Also allow these specific non-API public endpoints
    public_endpoints = [
        'auth.login', 
        'auth.google_login', 
        'auth.google_authorize', 
        'auth.signup',
        'auth.check_session',
        'static',
        'public.home',
        'public.live_map',
        'public.travel_plan',
        'public.alerts',
        'public.report',
        'user.user_dashboard',
    ]
    
    if request.endpoint in public_endpoints:
        return None
    
    # For protected endpoints (admin, operator), require login
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('auth.login'))
    
    # Verify user exists and is active
    user = User.query.get(session['user_id'])
    if not user or not user.is_active:
        session.clear()
        flash('Your session has expired. Please log in again.', 'warning')
        return redirect(url_for('auth.login'))
    
    return None
@auth_bp.route('/api/check-session')
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
def login():
    """Login page - with no-cache to prevent back button issues"""
    error = None
    error_type = None
    
    # On GET request, check if user is actually logged in
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
    
    invite_email = request.args.get('email')
    invite_temp = request.args.get('temp')
    invite_station = request.args.get('station')
    
    if invite_email and invite_temp:
        return render_template('operator_signup.html', 
                             email=invite_email, 
                             temp_password=invite_temp,
                             assigned_station=invite_station or 'All Stations')
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        ip_address = request.remote_addr
        
        print(f"\n{'='*50}")
        print(f"🔑 LOGIN ATTEMPT")
        print(f"   Email: {email}")
        print(f"   Password entered: {password}")
        print(f"{'='*50}\n")
        
        admin_email = os.getenv('ADMIN_EMAIL')
        admin_password = os.getenv('ADMIN_PASSWORD')
        
        if admin_email and admin_password and email == admin_email and password == admin_password:
            session.clear()
            session['admin_logged_in'] = True
            session['is_admin'] = True
            session['role'] = 'admin'
            session['username'] = email
            session['user_id'] = 0
            return redirect(url_for('admin.admin_dashboard'))
        
        user = User.query.filter_by(username=email).first()
        
        if user is None:
            print(f"❌ User not found: {email}")
            error = "Account not found."
            error_type = "error"
        elif not user.is_active:
            print(f"❌ Account deactivated: {email}")
            error = "Account deactivated."
            error_type = "error"
        elif user.google_id and not user.has_password():
            print(f"ℹ️ Google account - no password: {email}")
            error = "This account uses Google Sign-In. Please click 'Continue with Google'."
            error_type = "info"
        else:
            print(f"👤 User found: {user.username}")
            print(f"   Password hash in DB: {user.password_hash[:30] if user.password_hash else 'None'}...")
            
            # Try verification
            result = user.verify_password(password)
            print(f"   Password verification result: {'✅ PASS' if result else '❌ FAIL'}")
            
            if result:
                print(f"✅ Login successful for {email}")
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
                print(f"❌ Incorrect password for {email}")
                error = "Incorrect password."
                error_type = "error"
    
    return render_template('login.html', error=error, error_type=error_type)

@auth_bp.route('/signup', methods=['GET', 'POST'])
@no_cache
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
def google_login():
    """Initiate Google OAuth login with account selection options"""
    from flask import current_app
    
    # Clear any existing session before starting new OAuth flow
    session.clear()
    
    # Get the google client from the oauth instance stored in app config
    google = current_app.config.get('GOOGLE_CLIENT')
    
    if google is None:
        flash('Google authentication is not configured. Please contact administrator.', 'error')
        return redirect(url_for('auth.login'))
    
    # ✅ Build the redirect URI explicitly
    redirect_uri = url_for('auth.google_authorize', _external=True)
    print(f"🔐 Redirect URI: {redirect_uri}")
    
    # Always force account selection to allow switching accounts
    client_kwargs = {
        'scope': 'openid email profile',
        'prompt': 'select_account'
    }
    
    # ✅ Pass the redirect_uri explicitly
    return google.authorize_redirect(redirect_uri, **client_kwargs)

@auth_bp.route('/login/google/authorize')
@no_cache
def google_authorize():
    """Handle Google OAuth callback - With proper logging"""
    from flask import current_app
    
    # Debug logging
    print("\n" + "=" * 60)
    print("🔐 OAUTH CALLBACK RECEIVED")
    print(f"   Full URL: {request.url}")
    print(f"   Args: {dict(request.args)}")
    print(f"   Method: {request.method}")
    print("=" * 60)
    
    try:
        google = current_app.config.get('GOOGLE_CLIENT')
        
        if google is None:
            print("❌ Google client is None!")
            flash('Google authentication is not configured.', 'error')
            return redirect(url_for('auth.login'))
        
        print(f"✅ Google client found: {google.client_id[:20]}...")
        
        # ✅ Get the token - Authlib will validate the redirect_uri
        token = google.authorize_access_token()
        print(f"✅ Token received: {token is not None}")
        
        if not token:
            print("❌ No token received")
            flash('Failed to get access token from Google.', 'error')
            return redirect(url_for('auth.login'))
        
        # Parse user info
        user_info = google.parse_id_token(token, nonce=None)
        print(f"✅ User info: {user_info}")
        
        if not user_info or 'email' not in user_info:
            print("❌ No email in user info")
            flash('Failed to get user information from Google.', 'error')
            return redirect(url_for('auth.login'))
        
        email = user_info.get('email')
        name = user_info.get('name', email.split('@')[0])
        google_id = user_info.get('sub')
        ip_address = request.remote_addr
        
        print(f"🔐 Google OAuth Callback: {email}")
        
        # Find user in database
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
        
        # Link Google ID if not already linked
        if not user.google_id:
            user.google_id = google_id
            db.session.commit()
            log_activity(user.id, user.role, user.username, 'link_google', 
                        f'Google account linked from IP: {ip_address}')
        
        # Successful login
        user.last_login = datetime.now()
        db.session.commit()
        
        # Set session
        session.permanent = True
        session['user_id'] = user.id
        session['username'] = user.username
        session['role'] = user.role
        session['favorite_station'] = user.favorite_station
        session['google_user'] = True
        
        log_activity(user.id, user.role, user.username, 'login', 
                    f'Google login successful from IP: {ip_address}')
        
        flash(f'Welcome back, {name}!', 'success')
        
        # Redirect based on role
        if user.role == 'admin':
            return redirect(url_for('admin.admin_dashboard'))
        elif user.role == 'operator':
            return redirect(url_for('operator.operator_dashboard'))
        else:
            return redirect(url_for('user.user_dashboard'))
            
    except Exception as e:
        print(f"❌ Google login error: {e}")
        import traceback
        traceback.print_exc()
        
        flash(f'Google login failed: {str(e)}', 'error')
        return redirect(url_for('auth.login'))
        return redirect(url_for('auth.login'))

@auth_bp.route('/debug/oauth-config')
def debug_oauth_config():
    """Debug OAuth configuration"""
    from flask import current_app
    google = current_app.config.get('GOOGLE_CLIENT')
    
    return jsonify({
        'google_client_exists': google is not None,
        'client_id': current_app.config.get('GOOGLE_CLIENT_ID', 'NOT SET')[:20] + '...',
        'client_secret_set': bool(current_app.config.get('GOOGLE_CLIENT_SECRET')),
        'redirect_uri_used': 'http://localhost:5000/login/google/authorize',
        'registered_uris_in_console': [
            'http://localhost:5000/login/google/authorize',
            'http://127.0.0.1:5000/login/google/authorize',
            'http://localhost:5000/login/google/invite/callback',
            'http://127.0.0.1:5000/login/google/callback',
            'http://localhost:5000/login/google/callback'
        ]
    })
    
@auth_bp.route('/operator-signup', methods=['POST'])
@no_cache
def operator_signup():
    """Handle operator signup from invitation"""
    try:
        email = request.form.get('email')
        temp_password = request.form.get('temp_password')
        new_password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        name = request.form.get('name')
        station = request.form.get('station')
        
        if new_password != confirm_password:
            return jsonify({'success': False, 'error': 'Passwords do not match'}), 400
        
        operator = User.query.filter_by(username=email, role='operator').first()
        
        if not operator or not operator.verify_password(temp_password):
            return jsonify({'success': False, 'error': 'Invalid invitation'}), 401
        
        operator.password = new_password
        operator.is_active = True
        
        if station and station != 'All Stations':
            operator.favorite_station = station
        
        db.session.commit()
        
        session.clear()
        session.permanent = True
        session['user_id'] = operator.id
        session['username'] = operator.username
        session['role'] = 'operator'
        session['favorite_station'] = operator.favorite_station
        
        operator.last_login = datetime.now()
        db.session.commit()
        
        return jsonify({'success': True, 'redirect': '/operator-dashboard'})
        
    except Exception as e:
        print(f"❌ Error in operator signup: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500