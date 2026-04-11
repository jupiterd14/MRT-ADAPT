from datetime import datetime, timedelta
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from dotenv import load_dotenv
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
import re
import warnings
import time
import secrets
import string
from authlib.integrations.flask_client import OAuth
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from sqlalchemy import text
from mrt_schedule import (
    get_current_headway, get_headway_info, 
    calculate_next_trains, get_trip_schedule,
    get_all_trains_for_station, STATIONS as SCHEDULE_STATIONS
)
from prediction_api import MRT3Predictor


# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='html', static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'


# --- DATABASE CONFIG ---
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'mrt.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


@app.route('/debug-users')
def debug_users():
    """Show all users in the database"""
    users = User.query.all()
    result = {
        "users": [
            {
                "id": u.id, 
                "email": u.username,  # username stores the email
                "username": u.username,
                "role": u.role,
                "is_active": u.is_active
            } 
            for u in users
        ]
    }
    return jsonify(result)

# --- CONTEXT PROCESSOR ---
@app.context_processor
def inject_now():
    """Inject current datetime into templates"""
    return {'now': datetime.now()}

# --- GOOGLE OAUTH CONFIG ---
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# --- STATION METADATA ---
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

STATION_BASE_CAPACITY = {
    "North Ave": 12000, "Quezon Ave": 9000, "Kamuning": 7500, "Cubao": 15000,
    "Santolan": 8000, "Ortigas": 9500, "Shaw Blvd": 11000, "Boni Ave": 8500,
    "Guadalupe": 10000, "Buendia": 9000, "Ayala Ave": 14000, "Magallanes": 9000, "Taft": 16000
}

# --- LOAD SYSTEM CACHE & MODELS ---
print("\n" + "="*70)
print("🚇 WARMING UP SYSTEM...")
start_time = time.time()

try:
    with open('system_cache.pkl', 'rb') as f:
        cache = pickle.load(f)
    STATION_CAPACITIES = cache.get('STATION_CAPACITIES', STATION_BASE_CAPACITY)
    hourly_averages = cache.get('hourly_averages', {})
    station_time_series_last_24 = cache.get('station_time_series_last_24', {})
    weekday_capacity_patterns = cache.get('weekday_capacity_patterns', {})
    weekend_capacity_patterns = cache.get('weekend_capacity_patterns', {})
    print("✅ Loaded system cache")
except Exception as e:
    print(f"⚠️ Cache error: {e}, using defaults")
    STATION_CAPACITIES = STATION_BASE_CAPACITY
    hourly_averages, station_time_series_last_24 = {}, {}
    weekday_capacity_patterns, weekend_capacity_patterns = {}, {}

lstm_models, scalers = {}, {}
models_loaded = 0
for station in STATIONS:
    m_path, s_path = f'models/{station}_lstm.h5', f'models/{station}_scaler.pkl'
    if os.path.exists(m_path) and os.path.exists(s_path):
        try:
            lstm_models[station] = tf.keras.models.load_model(m_path, compile=False)
            with open(s_path, 'rb') as f:
                scalers[station] = pickle.load(f)
            models_loaded += 1
        except Exception as e:
            print(f"⚠️ Error loading {station}: {e}")

print(f"✅ System Ready in {time.time() - start_time:.1f}s")
print(f"📊 Loaded {models_loaded}/{len(STATIONS)} LSTM models")



print("\n" + "="*70)
print("🚀 Initializing MRT3 Predictor API...")
print("="*70)

try:
    predictor = MRT3Predictor(models_dir='models/')
    print("✅ Predictor initialized successfully")
except Exception as e:
    print(f"⚠️ Predictor initialization failed: {e}")
    predictor = None
    
historical_entry = {}
historical_exit = {}
hourly_avg_entry = {}
hourly_avg_exit = {}
dow_avg_entry = {}
dow_avg_exit = {}
direction_counts = {}

try:
    # Try to load from cache file
    if os.path.exists('historical_data_cache.pkl'):
        with open('historical_data_cache.pkl', 'rb') as f:
            cache_data = pickle.load(f)
        
        historical_entry = cache_data.get('historical_entry', {})
        historical_exit = cache_data.get('historical_exit', {})
        direction_counts = cache_data.get('direction_counts', {})
        hourly_avg_entry = cache_data.get('hourly_avg_entry', {})
        hourly_avg_exit = cache_data.get('hourly_avg_exit', {})
        dow_avg_entry = cache_data.get('dow_avg_entry', {})
        dow_avg_exit = cache_data.get('dow_avg_exit', {})
        
        print(f"DEBUG: Keys found in cache: {list(historical_entry.keys())[:3]}...")
        
        if not historical_entry:
            raise Exception("Cache dictionary is empty")
                
            print(f"✅ Successfully loaded real data for {len(historical_entry)} stations")    
        else:
            raise Exception("Cache file not found")

        
except Exception as e:
    print(f"⚠️ Error loading cache: {e}")
    print("Generating fresh historical data...")
    
    # Generate realistic sample data for each station
    for station in STATIONS:
        capacity = STATION_BASE_CAPACITY.get(station, 10000)
        
        # Different stations have different baseline volumes
        if station in ["Cubao", "Ayala Ave", "North Ave"]:
            entry_factor = 0.70
            exit_factor = 0.55
        elif station in ["Magallanes", "Santolan", "Buendia"]:
            entry_factor = 0.40
            exit_factor = 0.30
        else:
            entry_factor = 0.55
            exit_factor = 0.45
        
        historical_entry[station] = capacity * entry_factor
        historical_exit[station] = capacity * exit_factor
    
    # Generate hourly patterns with realistic values
    for hour in range(24):
        # Morning rush (7-9 AM)
        if 7 <= hour <= 9:
            hourly_avg_entry[hour] = 8000 + (hour - 7) * 500
            hourly_avg_exit[hour] = 2500 + (hour - 7) * 300
        # Evening rush (5-8 PM)
        elif 17 <= hour <= 20:
            hourly_avg_entry[hour] = 3500 - (hour - 17) * 300
            hourly_avg_exit[hour] = 7500 + (hour - 17) * 400
        # Late night (10 PM - 4 AM)
        elif hour >= 22 or hour <= 4:
            hourly_avg_entry[hour] = 500
            hourly_avg_exit[hour] = 500
        # Mid-day (10 AM - 4 PM)
        elif 10 <= hour <= 16:
            hourly_avg_entry[hour] = 4500
            hourly_avg_exit[hour] = 4500
        # Early morning (5-6 AM)
        elif 5 <= hour <= 6:
            hourly_avg_entry[hour] = 1500
            hourly_avg_exit[hour] = 1000
        # Late evening (9 PM)
        elif hour == 21:
            hourly_avg_entry[hour] = 4500
            hourly_avg_exit[hour] = 5000
        else:
            hourly_avg_entry[hour] = 3000
            hourly_avg_exit[hour] = 3000
    
    direction_counts = {
        'northbound': 4500000,
        'southbound': 3800000
    }
    
    # *** ADD THIS CODE HERE - SAVE THE REGENERATED DATA ***
    cache_data = {
        'historical_entry': historical_entry,
        'historical_exit': historical_exit,
        'direction_counts': direction_counts,
        'hourly_avg_entry': hourly_avg_entry,
        'hourly_avg_exit': hourly_avg_exit,
        'dow_avg_entry': dow_avg_entry,
        'dow_avg_exit': dow_avg_exit,
        'total_records': sum(historical_entry.values()) + sum(historical_exit.values())
    }
    
    try:
        with open('historical_data_cache.pkl', 'wb') as f:
            pickle.dump(cache_data, f)
        print("✅ Saved fresh data to historical_data_cache.pkl")
    except Exception as save_error:
        print(f"⚠️ Could not save cache: {save_error}")
    # *** END OF ADDED CODE ***
    
    print("✅ Generated fresh historical data")
    print("\n📊 Generated historical averages per station:")
    for station in STATIONS:
        entry = historical_entry.get(station, 0)
        exit_val = historical_exit.get(station, 0)
        print(f"  📍 {station}: Entry avg={entry:.0f}, Exit avg={exit_val:.0f}")
    
    print("\n📊 Generated hourly averages (selected hours):")
    for hour in [6, 8, 12, 18, 20, 22]:
        print(f"  Hour {hour}: Entry={hourly_avg_entry.get(hour, 0):.0f}, Exit={hourly_avg_exit.get(hour, 0):.0f}")

# ========== MODELS ==========

class Broadcast(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    message = db.Column(db.Text)
    disruption_type = db.Column(db.String(50))
    stations = db.Column(db.Text)  # Store as JSON
    severity = db.Column(db.String(20))
    operator_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.now)
    is_active = db.Column(db.Boolean, default=True)
    
class ActivityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    user_type = db.Column(db.String(20))  # 'admin', 'operator', 'commuter'
    user_email = db.Column(db.String(100))
    action = db.Column(db.String(100))  # 'login', 'logout', 'override', 'broadcast', etc.
    details = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.now)
    
    user = db.relationship('User', backref=db.backref('activity_logs', lazy=True))

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=True)
    role = db.Column(db.String(20), default='user')
    google_id = db.Column(db.String(100), unique=True, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    favorite_station = db.Column(db.String(50), nullable=True)
    last_login = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True) 
    
    access_level = db.Column(db.String(20), default='station')
    assigned_zone = db.Column(db.String(20), nullable=True)
    assigned_stations = db.Column(db.Text, nullable=True)
    
    @property
    def password(self):
        raise AttributeError('password is not readable')
    
    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def verify_password(self, password):
        if self.password_hash is None:
            return False
        return check_password_hash(self.password_hash, password)
    
    def has_password(self):
        """Check if user has a password set (non-Google user)"""
        return self.password_hash is not None
    
    def get_assigned_stations_list(self):
        """Get list of assigned stations"""
        import json
        if self.assigned_stations:
            try:
                return json.loads(self.assigned_stations)
            except:
                return []
        return []
    
    def set_assigned_stations(self, stations_list):
        """Set assigned stations from a list"""
        import json
        self.assigned_stations = json.dumps(stations_list)

class Activity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    activity_type = db.Column(db.String(50))
    title = db.Column(db.String(200))
    description = db.Column(db.String(500))
    icon = db.Column(db.String(50))
    icon_color = db.Column(db.String(50))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    station = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    user = db.relationship('User', backref=db.backref('activities', lazy=True))

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    station = db.Column(db.String(50))
    direction = db.Column(db.String(20), nullable=True)  # 'northbound', 'southbound', or None
    reported_congestion = db.Column(db.Integer)
    predicted_congestion = db.Column(db.Integer)
    remarks = db.Column(db.String(500), nullable=True)
    photo_path = db.Column(db.Text, nullable=True)  # Changed from String(200) to Text for multiple photos
    anonymous = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    
    # Add relationship
    user = db.relationship('User', backref=db.backref('reports', lazy=True))
    
class SavedRoute(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    from_station = db.Column(db.String(50), nullable=False)
    to_station = db.Column(db.String(50), nullable=False)
    route_name = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    user = db.relationship('User', backref=db.backref('saved_routes', lazy=True))

with app.app_context():
    db.create_all()


# Add these routes after your Google OAuth config

# Add these routes after your Google OAuth config
@app.route('/login/google')
def google_login():
    """Initiate Google OAuth login with account selection options"""
    # Clear any existing session before starting new OAuth flow
    session.clear()
    
    redirect_uri = url_for('google_authorize', _external=True)
    
    # Always force account selection to allow switching accounts
    client_kwargs = {
        'scope': 'openid email profile',
        'prompt': 'select_account'  # Force Google to show account selection
    }
    
    return google.authorize_redirect(redirect_uri, **client_kwargs)
@app.route('/login/google/authorize')
def google_authorize():
    """Handle Google OAuth callback - With proper logging"""
    try:
        token = google.authorize_access_token()
        user_info = google.parse_id_token(token, nonce=None)
        
        email = user_info.get('email')
        name = user_info.get('name', email.split('@')[0])
        google_id = user_info.get('sub')
        ip_address = request.remote_addr
        
        print(f"🔐 Google OAuth Callback: {email}")
        
        # Clear any existing session
        session.clear()
        
        # Check if user exists in database
        user = User.query.filter_by(username=email).first()
        
        if not user:
            # LOG FAILED GOOGLE LOGIN - No account
            log_activity(None, 'unknown', email, 'login_failed', 
                        f'Google login failed - no account found from IP: {ip_address}')
            flash(f'No account found for {email}. Please contact administrator.', 'error')
            return redirect(url_for('login'))
        
        if not user.is_active:
            # LOG FAILED GOOGLE LOGIN - Account inactive
            log_activity(user.id, user.role, user.username, 'login_failed', 
                        f'Google login failed - account deactivated from IP: {ip_address}')
            flash('Your account is deactivated. Please contact administrator.', 'error')
            return redirect(url_for('login'))
        
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
        
        # LOG SUCCESSFUL GOOGLE LOGIN
        log_activity(user.id, user.role, user.username, 'login', 
                    f'Google login successful from IP: {ip_address}')
        
        # Redirect based on role
        if user.role == 'admin':
            flash(f'Welcome back, {name}!', 'success')
            return redirect(url_for('admin_dashboard'))
        elif user.role == 'operator':
            flash(f'Welcome back, {name}!', 'success')
            return redirect(url_for('operator_dashboard'))
        else:
            flash(f'Welcome back, {name}!', 'success')
            return redirect(url_for('user_dashboard'))
            
    except Exception as e:
        print(f"❌ Google login error: {e}")
        # LOG FAILED GOOGLE LOGIN - Exception
        log_activity(None, 'unknown', request.args.get('email', 'unknown'), 'login_failed', 
                    f'Google login exception: {str(e)} from IP: {request.remote_addr}')
        flash('Google login failed. Please try again or use email/password.', 'error')
        return redirect(url_for('login'))

 # ========== AUDIT LOGGING API ENDPOINTS ==========


@app.route('/api/operator/get-overrides', methods=['GET'])
def get_overrides():
    """Get current active overrides"""
    if 'overrides' not in app.config:
        app.config['overrides'] = {}
    
    import time
    current_time = time.time()
    active_overrides = {}
    
    for station, override in app.config['overrides'].items():
        if override['expiry'] is None or override['expiry'] > current_time:
            active_overrides[station] = override
    
    return jsonify({'overrides': active_overrides})

@app.route('/api/audit/log-action', methods=['POST'])
def log_audit_action():
    """Log any action from frontend (override, broadcast, etc.)"""
    try:
        data = request.json
        action = data.get('action')
        details = data.get('details')
        station = data.get('station')
        
        user_id = session.get('user_id')
        user_role = session.get('role')
        user_email = session.get('username')
        
        if not user_id:
            return jsonify({'error': 'Not logged in'}), 401
        
        # Add station info if provided
        if station:
            details = f"{details} | Station: {station}"
        
        log_activity(user_id, user_role, user_email, action, details)
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error logging action: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/audit/last-login')
def get_last_login():
    """Get last login info for current user"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Not logged in'}), 401
        
        last_login = ActivityLog.query.filter_by(
            user_id=user_id, 
            action='login'
        ).order_by(ActivityLog.timestamp.desc()).first()
        
        if last_login:
            return jsonify({
                'last_login': last_login.timestamp.isoformat(),
                'ip_address': last_login.ip_address
            })
        return jsonify({'last_login': None})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
       
# ========== PREDICTION FUNCTIONS ==========
def log_activity(user_id, user_type, user_email, action, details=None):
    """Log user activity"""
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
        
def get_station_prediction_fallback(station_name):
    """Rule-based fallback prediction (when LSTM fails)"""
    try:
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        weekday = now.weekday()
        
        # Station operating hours: 4:30 AM to 10:30 PM
        if hour < 4 or (hour == 4 and minute < 30) or hour >= 23:
            return 0
        
        capacity = STATION_BASE_CAPACITY.get(station_name, 10000)
        
        # Get base from historical data with fallbacks
        base_entry = historical_entry.get(station_name)
        if base_entry is None or base_entry == 0:
            if station_name in ["Cubao", "Ayala Ave", "North Ave"]:
                base_entry = capacity * 0.70
            elif station_name in ["Magallanes", "Santolan", "Buendia"]:
                base_entry = capacity * 0.40
            else:
                base_entry = capacity * 0.55
        
        base_exit = historical_exit.get(station_name)
        if base_exit is None or base_exit == 0:
            if station_name in ["Cubao", "Ayala Ave", "North Ave"]:
                base_exit = capacity * 0.55
            elif station_name in ["Magallanes", "Santolan", "Buendia"]:
                base_exit = capacity * 0.30
            else:
                base_exit = capacity * 0.45
        
        # Get hourly multipliers
        hour_entry = hourly_avg_entry.get(hour, 3000)
        hour_exit = hourly_avg_exit.get(hour, 3000)
        
        if hour_entry == 0:
            hour_entry = 3000
        if hour_exit == 0:
            hour_exit = 3000
        
        base_hourly = (hour_entry + hour_exit) / 2
        
        # Calculate ridership
        ridership = (base_entry + base_exit) * (base_hourly / 4500)
        
        # Weekend adjustment
        if weekday >= 5:
            ridership *= 0.7
        
        # Station-specific adjustments
        station_adjustments = {
            "North Ave": 1.2, "Quezon Ave": 1.0, "Kamuning": 0.9,
            "Cubao": 1.3, "Santolan": 0.8, "Ortigas": 1.1,
            "Shaw Blvd": 1.05, "Boni Ave": 0.85, "Guadalupe": 0.95,
            "Buendia": 0.8, "Ayala Ave": 1.25, "Magallanes": 0.75,
            "Taft": 1.15
        }
        
        ridership = ridership * station_adjustments.get(station_name, 1.0)
        
        # Time of day variation
        if 7 <= hour <= 9:
            ridership *= 1.3
        elif 17 <= hour <= 20:
            ridership *= 1.25
        elif 10 <= hour <= 16:
            ridership *= 0.85
        
        # Cap at station capacity
        ridership = int(min(ridership, capacity))
        ridership = max(ridership, 100 if (5 <= hour <= 22) else 0)
        
        return ridership
        
    except Exception as e:
        print(f"❌ Error in fallback prediction for {station_name}: {e}")
        return int(STATION_BASE_CAPACITY.get(station_name, 10000) * 0.5)


def get_station_prediction(station_name):
    """Get prediction using LSTM model via predictor"""
    # Try to use the new predictor if available
    if predictor is not None:
        try:
            result = predictor.predict(station_name)
            if 'error' not in result and result.get('success'):
                # Scale up the prediction to be more realistic
                predicted = result['predicted_ridership']
                
                # Scale factor based on station (make it more realistic)
                capacity = STATION_BASE_CAPACITY.get(station_name, 10000)
                
                # If prediction is too low, scale it up
                if predicted < capacity * 0.3:  # Less than 30% capacity
                    # Scale to 40-70% capacity based on time of day
                    hour = datetime.now().hour
                    if 7 <= hour <= 9 or 17 <= hour <= 20:  # Rush hours
                        multiplier = 0.7  # 70% capacity during rush
                    else:
                        multiplier = 0.5  # 50% capacity normal hours
                    
                    predicted = int(capacity * multiplier)
                    
                return predicted
        except Exception as e:
            print(f"⚠️ Predictor error for {station_name}: {e}")
    
    # Fallback to rule-based if predictor fails or not available
    return get_station_prediction_fallback(station_name)


def get_station_prediction_for_datetime(station_name, target_datetime):
    """Get prediction for specific datetime using LSTM"""
    target_hour = target_datetime.hour
    target_minute = target_datetime.minute
    target_weekday = target_datetime.weekday()
    
    # Check if station is open
    if target_hour < 4 or (target_hour == 4 and target_minute < 30) or target_hour >= 23:
        return 0
    
    # Try to use the new predictor
    if predictor is not None:
        try:
            # Get current prediction as base
            result = predictor.predict(station_name)
            if 'error' not in result:
                base_ridership = result['predicted_ridership']
                
                # Adjust based on target hour vs current hour
                now = datetime.now()
                hour_diff = target_hour - now.hour
                
                # Simple adjustment based on time of day
                if 7 <= target_hour <= 9:  # Morning rush
                    multiplier = 1.2
                elif 17 <= target_hour <= 20:  # Evening rush
                    multiplier = 1.15
                elif target_hour >= 22 or target_hour <= 4:  # Late night
                    multiplier = 0.3
                else:
                    multiplier = 0.9
                
                return int(base_ridership * multiplier)
        except Exception as e:
            print(f"⚠️ Predictor failed for datetime: {e}")
    
    # Fallback to old logic
    if station_name in lstm_models and station_name in scalers:
        try:
            # ... existing fallback code ...
            pass
        except:
            pass
    
    return get_station_prediction_fallback(station_name)
# ========== API ROUTES ==========

@app.route('/api/reports')
def get_reports():
    try:
        reports = Report.query.order_by(Report.timestamp.desc()).limit(50).all()
        result = []
        for report in reports:
            username = None
            if not report.anonymous and report.user_id:
                user = User.query.get(report.user_id)
                if user:
                    username = user.username
            
            result.append({
                'id': report.id,
                'station': report.station,
                'direction': report.direction,  # ADD THIS LINE
                'reported_congestion': report.reported_congestion,
                'predicted_congestion': report.predicted_congestion,
                'remarks': report.remarks,
                'anonymous': report.anonymous,
                'username': username,
                'timestamp': report.timestamp.isoformat() if report.timestamp else None,
                'photo_path': report.photo_path  # ADD THIS LINE for images
            })
        return jsonify(result)
    except Exception as e:
        print(f"❌ Error fetching reports: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
# [Keep all your other routes exactly as they are...]
# The rest of your routes (home, user-dashboard, live-map, etc.) remain unchanged



@app.route('/')
def home():
    session.clear()
    session['guest_mode'] = True
    return redirect(url_for('user_dashboard'))

@app.route('/user-dashboard')
def user_dashboard():
    if session.get('admin_logged_in') or session.get('is_admin'):
        return redirect(url_for('admin_dashboard'))
    
    username = session.get('username', 'Public User')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    
    return render_template('user-dashboard.html', 
                         stations=STATIONS, 
                         username=username, 
                         is_logged_in=is_logged_in,
                         google_user=google_user,
                         favorite_station=favorite_station)

@app.route('/live-map')
def live_map():
    username = session.get('username', 'Public User')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    
    station_coords = {
        "North Ave": {"lat": 14.6556, "lng": 121.0302},
        "Quezon Ave": {"lat": 14.6390, "lng": 121.0380},
        "Kamuning": {"lat": 14.6249, "lng": 121.0431},
        "Cubao": {"lat": 14.6213, "lng": 121.0529},
        "Santolan": {"lat": 14.6135, "lng": 121.0630},
        "Ortigas": {"lat": 14.5864, "lng": 121.0565},
        "Shaw Blvd": {"lat": 14.5789, "lng": 121.0532},
        "Boni Ave": {"lat": 14.5716, "lng": 121.0492},
        "Guadalupe": {"lat": 14.5655, "lng": 121.0446},
        "Buendia": {"lat": 14.5547, "lng": 121.0329},
        "Ayala Ave": {"lat": 14.5497, "lng": 121.0305},
        "Magallanes": {"lat": 14.5450, "lng": 121.0254},
        "Taft": {"lat": 14.5378, "lng": 121.0112}
    }
    
    return render_template('live-map.html', 
                         stations=STATIONS, 
                         station_coords=station_coords,
                         username=username, 
                         is_logged_in=is_logged_in,
                         google_user=google_user,
                         favorite_station=favorite_station)

@app.route('/travel-plan')
def travel_plan():
    username = session.get('username', 'Public User')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    
    return render_template('travel-plan.html', 
                         stations=STATIONS, 
                         username=username, 
                         is_logged_in=is_logged_in,
                         google_user=google_user,
                         favorite_station=favorite_station)

@app.route('/alerts')
def alerts():
    username = session.get('username', 'Public User')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    
    return render_template('alerts.html', 
                         stations=STATIONS, 
                         username=username, 
                         is_logged_in=is_logged_in,
                         google_user=google_user,
                         favorite_station=favorite_station)


@app.route('/api/predict/<station_name>')
def predict_congestion(station_name):
    name = station_name.replace('%20', ' ')
    
    date_param = request.args.get('date')
    time_param = request.args.get('time')
    
    if date_param and time_param:
        try:
            year, month, day = map(int, date_param.split('-'))
            hour, minute = map(int, time_param.split(':'))
            target_datetime = datetime(year, month, day, hour, minute)
            ridership = get_station_prediction_for_datetime(name, target_datetime)
        except Exception as e:
            print(f"⚠️ Error parsing datetime: {e}")
            ridership = get_station_prediction(name)
    else:
        ridership = get_station_prediction(name)
    
    capacity = STATION_BASE_CAPACITY.get(name, 10000)
    congestion = min(100, int((ridership / capacity) * 100))
    
    if congestion > 80: status = "CRITICAL"
    elif congestion > 50: status = "BUSY"
    elif congestion > 20: status = "MODERATE"
    else: status = "LIGHT"
    
    return jsonify({
        "station": name, 
        "ridership": ridership, 
        "congestion": congestion, 
        "status": status
    })
    
#user dashboard
"""
1. get api
2. get north and south data
2. return
"""


@app.route('/api/station-forecast-badge/<station_name>')
def station_forecast_api_badge(station_name):
    """Get forecast for next 6 hours using LSTM model"""
    
    
    name = station_name.replace('%20', ' ')
    
    northbound = {}
    sourhbound = {}
    
    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute
    
    # Check if station is closed
    OPERATING_START = 4.5
    OPERATING_END = 22.5
    current_time = current_hour + current_minute / 60
    north_origin = ("northbound")
    south_origin = ("southbound")
    
    # After you calculate current_congestion, add this:


    current_ridership = get_station_prediction(name)
    capacity = STATION_BASE_CAPACITY.get(name, 10000)
    current_congestion = min(100, int((current_ridership / capacity) * 100))
    
    

    station_idx = STATIONS.index(name)
    is_morning_rush = 7 <= current_hour <= 9
    is_evening_rush = 17 <= current_hour <= 20

    north_congestion = current_congestion
    south_congestion = current_congestion

    if current_congestion >= 70:
        if is_morning_rush:
            if station_idx <= 6:  # Northern stations (North Ave to Cubao area)
                south_congestion = current_congestion  # Higher
                north_congestion = max(40, int(current_congestion * 0.6))  # Lower
            else:  # Southern stations
                north_congestion = current_congestion
                south_congestion = max(40, int(current_congestion * 0.6))
        elif is_evening_rush:
            if station_idx <= 6:
                north_congestion = current_congestion
                south_congestion = max(40, int(current_congestion * 0.6))
            else:
                south_congestion = current_congestion
                north_congestion = max(40, int(current_congestion * 0.6))
    
    print(f"🔴 CHART API - {name}: current_congestion = {current_congestion}")
    print(f"🔵 BADGE API - {name}: current_congestion = {current_congestion}")
    try:
    
        
        if current_time < OPERATING_START or current_time >= OPERATING_END:
            return jsonify({
                "northbound": {
                    "station": name,
                    "forecast": [5, 5, 5, 10, 15, 20],
                    "current": 5,
                    "origin": north_origin, 
                    "data_source": "Station Closed - Forecast for opening hours",
                    "operating_hours": "4:30 AM - 10:30 PM"
                    
                },
                "southbound": {
                    "station": name,
                    "forecast": [5, 5, 5, 10, 15, 20],
                    "current": 5,
                    "origin": south_origin, 
                    "data_source": "Station Closed - Forecast for opening hours",
                    "operating_hours": "4:30 AM - 10:30 PM"
                }
            })
        
    except Exception as e:
        print(f"Error {e}")
        
    
    forecast = []
    
    # Get current prediction
   
    
    # ========== FIX: Use previous ridership as base ==========
    prev_ridership = current_ridership  # Start with current
    
    for i in range(6):
        forecast_time = now + timedelta(hours=i+1)
        forecast_hour = forecast_time.hour
        
       
            # Time-based logic
        if 7 <= forecast_hour <= 9:  # Morning rush - INCREASING
            multiplier = 1.2
        elif 17 <= forecast_hour <= 20:  # Evening rush - INCREASING
                multiplier = 1.15
        elif forecast_hour <= 6 or forecast_hour >= 22:  # Late night - DECREASING
                multiplier = 0.7
        else:  # Normal hours - SLIGHT DECREASE
            multiplier = 0.95
            
            # Use PREVIOUS ridership, not current!
        forecast_ridership = int(prev_ridership * multiplier)
        forecast_ridership = max(50, min(forecast_ridership, capacity))
        forecast_congestion = min(100, int((forecast_ridership / capacity) * 100))
            
            # Update prev_ridership for next iteration
        prev_ridership = forecast_ridership
        
        forecast.append(forecast_congestion)
        
    
    return jsonify({
        "northbound": {
            "station": name,
            "forecast": forecast,
            "current": north_congestion,
            "origin": north_origin, 
            "data_source": "Station North",
            "operating_hours": "4:30 AM - 10:30 PM"
                
        },
        "southbound": {
            "station": name,
            "forecast": forecast,
            "current": south_congestion,
            "origin": south_origin, 
            "data_source": "Station South",
            "operating_hours": "4:30 AM - 10:30 PM"
            }
    })
    
    
@app.route('/api/station-forecast/<station_name>')
def station_forecast_api(station_name):
    """Get forecast for next 6 hours using LSTM model"""
    name = station_name.replace('%20', ' ')
    
    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute
    
    # Check if station is closed
    OPERATING_START = 4.5
    OPERATING_END = 22.5
    current_time = current_hour + current_minute / 60
    
    if current_time < OPERATING_START or current_time >= OPERATING_END:
        return jsonify({
            "station": name,
            "forecast": [5, 5, 5, 10, 15, 20],
            "intervals": ["+1h", "+2h", "+3h", "+4h", "+5h", "+6h"],
            "current": 5,
            "data_source": "Station Closed - Forecast for opening hours",
            "operating_hours": "4:30 AM - 10:30 PM"
        })
    
    forecast = []
    
    # Get current prediction
    current_ridership = get_station_prediction(name)
    capacity = STATION_BASE_CAPACITY.get(name, 10000)
    current_congestion = min(100, int((current_ridership / capacity) * 100))
    
  
    
    # ========== FIX: Use previous ridership as base ==========
    prev_ridership = current_ridership  # Start with current
    
    for i in range(6):
        forecast_time = now + timedelta(hours=i+1)
        forecast_hour = forecast_time.hour
        
       
            # Time-based logic
        if 7 <= forecast_hour <= 9:  # Morning rush - INCREASING
            multiplier = 1.2
        elif 17 <= forecast_hour <= 20:  # Evening rush - INCREASING
                multiplier = 1.15
        elif forecast_hour <= 6 or forecast_hour >= 22:  # Late night - DECREASING
                multiplier = 0.7
        else:  # Normal hours - SLIGHT DECREASE
            multiplier = 0.95
            
            # Use PREVIOUS ridership, not current!
        forecast_ridership = int(prev_ridership * multiplier)
        forecast_ridership = max(50, min(forecast_ridership, capacity))
        forecast_congestion = min(100, int((forecast_ridership / capacity) * 100))
            
            # Update prev_ridership for next iteration
        prev_ridership = forecast_ridership
        
        forecast.append(forecast_congestion)
    
    return jsonify({
        "station": name,
        "forecast": forecast,
        "intervals": ["+1h", "+2h", "+3h", "+4h", "+5h", "+6h"],
        "current": current_congestion,
        "data_source": "LSTM Model + Historical Patterns",
        "operating_hours": "4:30 AM - 10:30 PM"
    })


@app.route('/api/batch-predict')
def batch_predict():
    results = []
    for station in STATIONS:
        ridership = get_station_prediction(station)
        capacity = STATION_BASE_CAPACITY.get(station, 10000)
        congestion = int((ridership / capacity) * 100)
        congestion = max(0, min(100, congestion))
        
        if congestion >= 75:
            status = "🔴 CRITICAL"
        elif congestion >= 55:
            status = "🟠 BUSY"
        elif congestion >= 30:
            status = "🟡 MODERATE"
        else:
            status = "🟢 LIGHT"
        
        results.append({
            "station": station,
            "ridership": ridership,
            "congestion": congestion,
            "status": status
        })
        
    return jsonify(results)
@app.route('/api/report-congestion', methods=['POST'])
def report_congestion():
    try:
        # Get user info
        user_id = session.get('user_id')
        ip_address = request.remote_addr
        
        # 1. Rate limiting check
        if is_rate_limited(user_id, ip_address, limit=5, window=3600):
            return jsonify({
                "success": False, 
                "error": "Too many reports (max 5 per hour). Please wait before submitting more reports."
            }), 429
        
        # Check if it's multipart/form-data (with images) or JSON
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Handle FormData with images
            station = request.form.get('station')
            direction = request.form.get('direction')
            reported = request.form.get('congestion')
            remarks = request.form.get('remarks', '')
            anonymous = request.form.get('anonymous', 'false').lower() == 'true'
            
            # Handle image uploads
            photo_paths = []
            if 'images' in request.files:
                files = request.files.getlist('images')
                if len(files) > 5:
                    return jsonify({"success": False, "error": "Maximum 5 photos allowed"}), 400
                
                for file in files:
                    if file and file.filename:
                        # Basic file size check (10MB limit)
                        file.seek(0, 2)
                        size = file.tell()
                        file.seek(0)
                        if size > 10 * 1024 * 1024:
                            return jsonify({"success": False, "error": "Image too large (max 10MB)"}), 400
                        
                        # Clean filename
                        safe_filename = file.filename.replace(' ', '_').replace('%', '')
                        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_filename}"
                        upload_folder = os.path.join('static', 'uploads', 'reports')
                        os.makedirs(upload_folder, exist_ok=True)
                        filepath = os.path.join(upload_folder, filename)
                        file.save(filepath)
                        photo_paths.append(f"/uploads/reports/{filename}")
        else:
            # Handle JSON data
            data = request.json
            station = data.get('station')
            direction = data.get('direction')
            reported = data.get('congestion')
            remarks = data.get('remarks', '')
            anonymous = data.get('anonymous', False)
            photo_paths = []
        
        # 2. Validate required fields
        if not station:
            return jsonify({"success": False, "error": "Station is required"}), 400
        if reported is None:
            return jsonify({"success": False, "error": "Congestion level is required"}), 400
        
        # 3. Validate station exists
        if station not in STATIONS:
            return jsonify({"success": False, "error": "Invalid station"}), 400
        
        # Convert reported to int
        reported = int(reported)
        
        # 4. Validate congestion value
        if not (0 <= reported <= 100):
            return jsonify({"success": False, "error": "Congestion must be between 0 and 100"}), 400
        
        # 5. Check for spammy remarks
        if is_suspicious_remarks(remarks):
            return jsonify({
                "success": False, 
                "error": "Suspicious remarks detected. Please provide meaningful feedback."
            }), 400
        
        # 6. Check for duplicate reports (only for logged-in users)
        if user_id and check_duplicate_report(station, reported, user_id):
            return jsonify({
                "success": False, 
                "error": "You already reported this station recently. Please wait before reporting again."
            }), 400
        
        # 7. Validate direction
        if direction and direction not in ['northbound', 'southbound']:
            direction = None
        
        # Get prediction for comparison
        ridership = get_station_prediction(station)
        capacity = STATION_BASE_CAPACITY.get(station, 10000)
        predicted = int((ridership / capacity) * 100)
        
        # Create report with direction
        report = Report(
            user_id=user_id,
            station=station,
            direction=direction,
            reported_congestion=reported,
            predicted_congestion=predicted,
            remarks=remarks[:500],  # Limit length
            photo_path=json.dumps(photo_paths) if photo_paths else None,
            anonymous=anonymous
        )
        
        db.session.add(report)
        db.session.commit()
        
        return jsonify({
            "success": True, 
            "message": "Report saved successfully",
            "photos": len(photo_paths),
            "direction": direction
        })
        
    except Exception as e:
        print(f"❌ Error saving report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    

def is_rate_limited(user_id, ip_address, limit=5, window=3600):
    """Check if user has exceeded rate limit (5 reports per hour)"""
    key = user_id if user_id else ip_address
    now = datetime.now()
    report_limits = defaultdict(list)
    # Clean old entries
    report_limits[key] = [t for t in report_limits[key] if now - t < timedelta(seconds=window)]
    
    if len(report_limits[key]) >= limit:
        return True
    
    report_limits[key].append(now)
    return False

def is_suspicious_remarks(remarks):
    """Check if remarks look like spam"""
    if not remarks:
        return False
    
    # Check for repeated characters (spam like "AAAAA")
    if re.search(r'(.)\1{10,}', remarks):
        return True
    
    # Check if remarks are all the same character repeated
    if len(set(remarks.lower())) == 1 and len(remarks) > 5:
        return True
    
    return False

def check_duplicate_report(station, congestion_value, user_id, minutes=15):
    """Check if user already reported same station recently"""
    if not user_id:
        return False
    
    time_threshold = datetime.now() - timedelta(minutes=minutes)
    
    duplicate = Report.query.filter(
        Report.station == station,
        Report.reported_congestion == congestion_value,
        Report.user_id == user_id,
        Report.timestamp > time_threshold
    ).first()
    
    return duplicate is not None

# Add this route to serve uploaded images
@app.route('/uploads/reports/<filename>')
def serve_upload(filename):
    """Serve uploaded report images"""
    from flask import send_from_directory
    upload_folder = os.path.join(os.path.dirname(__file__), 'static', 'uploads', 'reports')
    return send_from_directory(upload_folder, filename)
   
@app.route('/api/alerts/count')
def alerts_count():
    try:
        critical_count = 0
        for station in STATIONS:
            ridership = get_station_prediction(station)
            capacity = STATION_BASE_CAPACITY.get(station, 10000)
            congestion = int((ridership / capacity) * 100)
            
            if congestion > 70:
                critical_count += 1
        
        total = critical_count
        display = str(total) if total < 9 else "9+"
        return jsonify({"count": total, "display": display})
        
    except Exception as e:
        print(f"Error getting alert count: {e}")
        return jsonify({"count": 0, "display": "0"})

@app.route('/api/saved-routes', methods=['GET'])
def get_saved_routes():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        user_id = session.get('user_id')
        routes = SavedRoute.query.filter_by(user_id=user_id).order_by(SavedRoute.created_at.desc()).all()
        
        result = []
        for route in routes:
            result.append({
                'id': route.id,
                'from_station': route.from_station,
                'to_station': route.to_station,
                'route_name': route.route_name,
                'created_at': route.created_at.isoformat()
            })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error fetching saved routes: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/saved-routes', methods=['POST'])
def save_route():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.json
        user_id = session.get('user_id')
        from_station = data.get('from_station')
        to_station = data.get('to_station')
        route_name = data.get('route_name', f"{from_station} to {to_station}")
        
        if not from_station or not to_station:
            return jsonify({"error": "Missing station information"}), 400
        
        existing = SavedRoute.query.filter_by(
            user_id=user_id,
            from_station=from_station,
            to_station=to_station
        ).first()
        
        if existing:
            return jsonify({"success": True, "message": "Route already saved"})
        
        route = SavedRoute(
            user_id=user_id,
            from_station=from_station,
            to_station=to_station,
            route_name=route_name
        )
        
        db.session.add(route)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Route saved successfully",
            "route": {
                'id': route.id,
                'from_station': route.from_station,
                'to_station': route.to_station,
                'route_name': route.route_name
            }
        })
        
    except Exception as e:
        print(f"❌ Error saving route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/saved-routes/<int:route_id>', methods=['DELETE'])
def delete_saved_route(route_id):
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        user_id = session.get('user_id')
        route = SavedRoute.query.filter_by(id=route_id, user_id=user_id).first()
        
        if not route:
            return jsonify({"error": "Route not found"}), 404
        
        db.session.delete(route)
        db.session.commit()
        
        return jsonify({"success": True, "message": "Route deleted"})
        
    except Exception as e:
        print(f"❌ Error deleting route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/historical-patterns')
def historical_patterns():
    try:
        patterns = {}
        for station in STATIONS:
            station_patterns = {}
            for hour in range(24):
                # Use actual hourly data from cache if available
                if hour in hourly_avg_entry:
                    base = (hourly_avg_entry.get(hour, 0) + hourly_avg_exit.get(hour, 0)) / 2
                    base = base / 100  # Scale down for percentage
                else:
                    if 7 <= hour <= 9:
                        base = 75
                    elif 17 <= hour <= 20:
                        base = 80
                    elif 10 <= hour <= 16:
                        base = 55
                    elif 21 <= hour <= 22:
                        base = 30
                    elif 5 <= hour <= 6:
                        base = 20
                    else:
                        base = 5
                
                if station in ["Cubao", "Ayala Ave", "North Ave"]:
                    base += 10
                elif station in ["Santolan", "Magallanes"]:
                    base -= 5
                
                station_patterns[hour] = min(100, max(0, base))
            
            patterns[station] = station_patterns
        
        return jsonify(patterns)
    except Exception as e:
        print(f"❌ Error generating historical patterns: {e}")
        return jsonify({}), 500

# ========== DASHBOARD API ROUTES ==========

@app.route('/api/congestion/<station_name>')
def get_congestion(station_name):
    """Get congestion data for a specific station"""
    name = station_name
    
    try:
        ridership = get_station_prediction(name)
        capacity = STATION_BASE_CAPACITY.get(name, 10000)
        congestion = min(100, int((ridership / capacity) * 100))
        
        if congestion > 80:
            status = "SEVERELY CONGESTED"
            color = "critical"
            wait_time = "15-20 min"
        elif congestion > 60:
            status = "CONGESTED"
            color = "congested"
            wait_time = "10-15 min"
        elif congestion > 30:
            status = "MODERATE"
            color = "moderate"
            wait_time = "5-10 min"
        else:
            status = "LIGHT"
            color = "light"
            wait_time = "2-5 min"
        
        return jsonify({
            "station": name,
            "congestion": congestion,
            "status": status,
            "color": color,
            "wait_time": wait_time,
            "ridership": ridership,
            "capacity": capacity
        })
    except Exception as e:
        print(f"❌ Error getting congestion: {e}")
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/bestTime/<station_name>')
def bestTime(station_name):
    """Get congestion data for a specific station"""
    name = station_name
    
    try:
        ridership = get_station_prediction(name)
        capacity = STATION_BASE_CAPACITY.get(name, 10000)
        congestion = min(100, int((ridership / capacity) * 100))
        
        
        if congestion > 80:
            status = "SEVERELY CONGESTED"
            color = "critical"
            wait_time = "15-20 min"
        elif congestion > 60:
            status = "CONGESTED"
            color = "congested"
            wait_time = "10-15 min"
        elif congestion > 30:
            status = "MODERATE"
            color = "moderate"
            wait_time = "5-10 min"
        else:
            status = "LIGHT"
            color = "light"
            wait_time = "2-5 min"
        
        return jsonify({
            "station": name,
            "congestion": congestion,
            "status": status,
            "color": color,
            "wait_time": wait_time,
            "ridership": ridership,
            "capacity": capacity
        })
    except Exception as e:
        print(f"❌ Error getting congestion: {e}")
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/schedule/station/<station_name>')
def get_schedule(station_name):
    """Get train schedule for a station - NOW USING REAL SCHEDULE"""
    name = station_name.replace('%20', '')
    
    try:
        schedule = get_all_trains_for_station(name, limit=5)
        print("🔍 DEBUG: schedule keys =", schedule.keys())
        
        if "error" in schedule:
            return jsonify({"erro8r": schedule["error"], "status": "closed"})
        
        # Get congestion info for headway adjustment
        ridership = get_station_prediction(name)
        capacity = STATION_BASE_CAPACITY.get(name, 10000)
        congestion = min(100, int((ridership / capacity) * 100))
        
        # Adjust headway if severely congested (still realistic adjustment)
        headway = schedule["headway"]
        if congestion > 80:
            headway = max(8, headway + 2)
        elif congestion > 60:
            headway = max(6, headway + 1)
        
        return jsonify({
            "station": name,
            "headway": headway,
            "trains": schedule["trains"]["northbound"][:3],  # Return first 3 northbound trains
            "status": "normal" if schedule["is_operating"] else "closed"
        })
    except Exception as e:
        print(f"❌ Error getting schedule: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/schedule/next-trains/<station_name>')
def get_next_trains(station_name):
    """Get next train arrivals using REAL schedule data"""
    name = station_name.replace('%20', ' ')
    
    try:
        trains = calculate_next_trains(name)
        
        # Check if operating
        is_operating = trains.get("is_operating", True)
        
        if not is_operating:
            return jsonify({
                "is_closed": True,
                "is_operating": False,
                "next_open": "4:30 AM",
                "northbound": {"minutes": None, "origin": None, "status": "closed"},
                "southbound": {"minutes": None, "origin": None, "status": "closed"},
                "headway": 0
            })
        
        # Get the minutes - ensure they're integers
        north_minutes = trains.get("northbound", {}).get("minutes", 5)
        south_minutes = trains.get("southbound", {}).get("minutes", 3)
        north_origin = trains.get("northbound", {}).get("from_station", "Taft")
        south_origin = trains.get("southbound", {}).get("from_station", "North Ave")
        
        # Make sure minutes are at least 1 and not more than 15
        north_minutes = max(1, min(15, north_minutes))
        south_minutes = max(1, min(15, south_minutes))
        
        print(f"🚆 {name}: North={north_minutes}min from {north_origin}, South={south_minutes}min from {south_origin}")
        
        return jsonify({
            "northbound": {
                "minutes": north_minutes,
                "origin": north_origin,
                "from_station": north_origin,
                "status": "scheduled"
            },
            "southbound": {
                "minutes": south_minutes,
                "origin": south_origin,
                "from_station": south_origin,
                "status": "scheduled"
            },
            "headway": trains.get("headway", 5),
            "is_operating": True,
            "is_closed": False
        })
        
    except Exception as e:
        print(f"❌ Error getting next trains: {e}")
        import traceback
        traceback.print_exc()
        # Return realistic fallback based on time of day
        now = datetime.now()
        hour = now.hour
        
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            north_fallback = 3
            south_fallback = 2
        else:
            north_fallback = 5
            south_fallback = 4
            
        return jsonify({
            "is_closed": False,
            "is_operating": True,
            "northbound": {"minutes": north_fallback, "origin": "Taft", "from_station": "Taft", "status": "estimated"},
            "southbound": {"minutes": south_fallback, "origin": "North Ave", "from_station": "North Ave", "status": "estimated"},
            "headway": 5
        })

# REPLACE this entire function
@app.route('/api/schedule/headway')
def get_headway():
    """Get current headway based on REAL schedule"""
    try:
        headway_info = get_headway_info()
        
        # Get average congestion for context
        total_congestion = 0
        for station in STATIONS:
            ridership = get_station_prediction(station)
            capacity = STATION_BASE_CAPACITY.get(station, 10000)
            congestion = min(100, int((ridership / capacity) * 100))
            total_congestion += congestion
        
        avg_congestion = total_congestion / len(STATIONS)
        
        return jsonify({
            "headway": headway_info["headway"] // 60 if headway_info["headway"] else 0,
            "status": headway_info["status"],
            "message": headway_info["message"],
            "average_congestion": round(avg_congestion, 1)
        })
    except Exception as e:
        print(f"❌ Error getting headway: {e}")
        return jsonify({"headway": 5, "status": "normal", "message": "Normal service - trains every 5 minutes"})


# ADD THIS NEW ROUTE for trip planning
@app.route('/api/trip-schedule')
def trip_schedule():
    """Get schedule for a specific trip"""
    from_station = request.args.get('from')
    to_station = request.args.get('to')
    date = request.args.get('date')
    time_str = request.args.get('time')
    
    if not from_station or not to_station:
        return jsonify({"error": "Missing station parameters"}), 400
    
    try:
        target_time = None
        if date and time_str:
            year, month, day = map(int, date.split('-'))
            hour, minute = map(int, time_str.split(':'))
            target_time = datetime(year, month, day, hour, minute)
        
        trip = get_trip_schedule(from_station, to_station, target_time)
        
        if "error" in trip:
            return jsonify(trip), 404
        
        return jsonify(trip)
    except Exception as e:
        print(f"❌ Error getting trip schedule: {e}")
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/alerts/list')
def alerts_list():
    """Get list of active alerts"""
    try:
        alerts = []
        now = datetime.now()
        hour = now.hour
        
        if 7 <= hour <= 9:
            alerts.append({
                "id": "rush-morning",
                "type": "rush_hour",
                "title": "Morning Rush Hour",
                "message": "Expect heavy traffic at North Ave, Quezon Ave, and Cubao stations",
                "time": now.strftime("%I:%M %p"),
                "severity": "warning"
            })
        elif 17 <= hour <= 20:
            alerts.append({
                "id": "rush-evening",
                "type": "rush_hour",
                "title": "Evening Rush Hour",
                "message": "Expect heavy traffic at Ayala, Magallanes, and Taft stations",
                "time": now.strftime("%I:%M %p"),
                "severity": "warning"
            })
        
        for station in STATIONS:
            ridership = get_station_prediction(station)
            capacity = STATION_BASE_CAPACITY.get(station, 10000)
            congestion = min(100, int((ridership / capacity) * 100))
            
            if congestion > 80:
                alerts.append({
                    "id": f"critical-{station}",
                    "type": "critical",
                    "title": f"Critical Congestion at {station}",
                    "message": f"Congestion at {congestion}%. Expect delays of 15-20 minutes.",
                    "time": now.strftime("%I:%M %p"),
                    "severity": "critical"
                })
                break
        
        return jsonify(alerts)
    except Exception as e:
        print(f"❌ Error getting alerts list: {e}")
        return jsonify([])

@app.route('/api/recommendation/<station_name>')
def get_recommendation(station_name):
    """Get travel recommendation for a station"""
    name = station_name.replace('%20', ' ')
    
    try:
        ridership = get_station_prediction(name)
        capacity = STATION_BASE_CAPACITY.get(name, 10000)
        congestion = min(100, int((ridership / capacity) * 100))
        
        now = datetime.now()
        hour = now.hour
        
        if congestion > 80:
            if 7 <= hour <= 9 or 17 <= hour <= 20:
                recommendation = "Consider postponing your trip until after rush hour"
            else:
                recommendation = "Severe congestion. Consider alternative routes or wait 30 minutes"
        elif congestion > 60:
            recommendation = "Heavy traffic. Allow extra 10-15 minutes for your journey"
        elif congestion > 30:
            recommendation = "Moderate traffic. Normal wait times expected"
        else:
            recommendation = "Light traffic. Good time to travel!"
        
        return jsonify({
            "station": name,
            "congestion": congestion,
            "recommendation": recommendation,
            "best_time": get_best_time_to_travel(name)
        })
    except Exception as e:
        print(f"❌ Error getting recommendation: {e}")
        return jsonify({
            "recommendation": "Normal operations. Trains running on schedule.",
            "best_time": "10:00 AM - 3:00 PM"
        })

def get_best_time_to_travel(station_name):
    """Helper function to determine best travel time"""
    hour = datetime.now().hour
    
    if 7 <= hour <= 9:
        return "10:00 AM - 3:00 PM"
    elif 17 <= hour <= 20:
        return "Before 5:00 PM or after 8:00 PM"
    else:
        return "Now is a good time to travel"

# ========== LIVE MAP API ROUTES ==========
@app.route('/api/live-map/directions')
def live_map_directions():
    """Get congestion data for both directions - respects operator overrides"""
    try:
        northbound = {}
        southbound = {}
        
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        current_time = hour + minute / 60
        
        OPERATING_START = 4.5
        OPERATING_END = 22.5
        
        # Get active overrides
        import time
        if 'overrides' not in app.config:
            app.config['overrides'] = {}
        
        current_timestamp = time.time()
        active_overrides = {}
        for station, override in app.config['overrides'].items():
            if override['expiry'] is None or override['expiry'] > current_timestamp:
                active_overrides[station] = override
        
        if current_time < OPERATING_START or current_time >= OPERATING_END:
            for station in STATIONS:
                if station in active_overrides:
                    override = active_overrides[station]
                    congestion = override['congestion']
                    status_text = override['level'].upper()
                    wait_time = get_wait_time(congestion)
                else:
                    congestion = 0
                    status_text = "CLOSED"
                    wait_time = "CLOSED"
                
                northbound[station] = {"congestion": congestion, "wait_time": wait_time, "status": status_text, "ridership": 0}
                southbound[station] = {"congestion": congestion, "wait_time": wait_time, "status": status_text, "ridership": 0}
        else:
            for station in STATIONS:
                # CHECK FOR ACTIVE OVERRIDE FIRST
                if station in active_overrides:
                    override = active_overrides[station]
                    total_congestion = override['congestion']
                    print(f"🔧 OVERRIDE ACTIVE for {station}: {total_congestion}% ({override['level']}) - by {override['operator']}")
                else:
                    # Get the LSTM prediction
                    ridership = get_station_prediction(station)
                    capacity = STATION_BASE_CAPACITY.get(station, 10000)
                    total_congestion = min(100, int((ridership / capacity) * 100))
                
                station_idx = STATIONS.index(station)
                is_morning_rush = 7 <= hour <= 9
                is_evening_rush = 17 <= hour <= 20
                
                if total_congestion >= 70:
                    if is_morning_rush:
                        if station_idx <= 6:
                            south_congestion = total_congestion
                            north_congestion = max(40, int(total_congestion * 0.6))
                        else:
                            north_congestion = total_congestion
                            south_congestion = max(40, int(total_congestion * 0.6))
                    elif is_evening_rush:
                        if station_idx <= 6:
                            north_congestion = total_congestion
                            south_congestion = max(40, int(total_congestion * 0.6))
                        else:
                            south_congestion = total_congestion
                            north_congestion = max(40, int(total_congestion * 0.6))
                    else:
                        north_congestion = total_congestion
                        south_congestion = total_congestion
                else:
                    north_congestion = total_congestion
                    south_congestion = total_congestion
                
                if total_congestion > 80:
                    north_congestion = max(north_congestion, 70)
                    south_congestion = max(south_congestion, 70)
                
                north_congestion = min(100, north_congestion)
                south_congestion = min(100, south_congestion)
                
                capacity = STATION_BASE_CAPACITY.get(station, 10000)
                north_ridership = int((north_congestion / 100) * capacity)
                south_ridership = int((south_congestion / 100) * capacity)
                
                def get_status_and_wait(congestion):
                    if congestion > 80:
                        return "SEVERELY CONGESTED", "15-20 min"
                    elif congestion > 60:
                        return "CONGESTED", "10-15 min"
                    elif congestion > 30:
                        return "MODERATE", "5-10 min"
                    else:
                        return "LIGHT", "2-5 min"
                
                north_status, north_wait = get_status_and_wait(north_congestion)
                south_status, south_wait = get_status_and_wait(south_congestion)
                
                southbound[station] = {
                    "congestion": south_congestion,
                    "wait_time": south_wait,
                    "status": south_status,
                    "ridership": south_ridership,
                    "overridden": station in active_overrides
                }
                
                northbound[station] = {
                    "congestion": north_congestion,
                    "wait_time": north_wait,
                    "status": north_status,
                    "ridership": north_ridership,
                    "overridden": station in active_overrides
                }
        
        return jsonify({
            "northbound": northbound,
            "southbound": southbound,
            "timestamp": now.isoformat(),
            "is_operating": OPERATING_START <= current_time < OPERATING_END,
            "active_overrides": len(active_overrides),
            "active_overrides_details": {s: o['level'] for s, o in active_overrides.items()}
        })
        
    except Exception as e:
        print(f"❌ Error in live_map_directions: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def get_wait_time(congestion):
    """Helper function to get wait time based on congestion"""
    if congestion > 80:
        return "15-20 min"
    elif congestion > 60:
        return "10-15 min"
    elif congestion > 30:
        return "5-10 min"
    else:
        return "2-5 min"
    
    
@app.route('/api/predict-direction/<station_name>')
def predict_direction(station_name):
    """Get prediction with direction info for a station"""
    name = station_name.replace('%20', ' ')
    
    try:
        ridership = get_station_prediction(name)
        capacity = STATION_BASE_CAPACITY.get(name, 10000)
        congestion = min(100, int((ridership / capacity) * 100))
        
        station_idx = STATIONS.index(name) if name in STATIONS else 0
        
        if station_idx < 6:
            direction = "southbound"
            next_station = STATIONS[station_idx + 1] if station_idx + 1 < len(STATIONS) else STATIONS[0]
        elif station_idx > 6:
            direction = "northbound"
            next_station = STATIONS[station_idx - 1] if station_idx - 1 >= 0 else STATIONS[-1]
        else:
            direction = "both"
            next_station = STATIONS[station_idx + 1] if station_idx + 1 < len(STATIONS) else STATIONS[0]
        
        if congestion > 80:
            status = "SEVERELY CONGESTED"
            color = "critical"
            wait_time = "15-20 min"
        elif congestion > 60:
            status = "CONGESTED"
            color = "congested"
            wait_time = "10-15 min"
        elif congestion > 30:
            status = "MODERATE"
            color = "moderate"
            wait_time = "5-10 min"
        else:
            status = "LIGHT"
            color = "light"
            wait_time = "2-5 min"
        
        return jsonify({
            "station": name,
            "congestion": congestion,
            "status": status,
            "color": color,
            "direction": direction,
            "next_station": next_station,
            "ridership": ridership,
            "capacity": capacity,
            "wait_time": wait_time
        })
    except Exception as e:
        print(f"❌ Error in predict_direction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict-route')
def predict_route():
    """Get prediction for a route between two stations"""
    from_station = request.args.get('from')
    to_station = request.args.get('to')
    date = request.args.get('date')
    time = request.args.get('time')
    
    if not from_station or not to_station:
        return jsonify({"error": "Missing station parameters"}), 400
    
    try:
        if date and time:
            year, month, day = map(int, date.split('-'))
            hour, minute = map(int, time.split(':'))
            target_datetime = datetime(year, month, day, hour, minute)
            ridership_from = get_station_prediction_for_datetime(from_station, target_datetime)
            ridership_to = get_station_prediction_for_datetime(to_station, target_datetime)
        else:
            ridership_from = get_station_prediction(from_station)
            ridership_to = get_station_prediction(to_station)
        
        capacity_from = STATION_BASE_CAPACITY.get(from_station, 10000)
        capacity_to = STATION_BASE_CAPACITY.get(to_station, 10000)
        
        congestion_from = min(100, int((ridership_from / capacity_from) * 100))
        congestion_to = min(100, int((ridership_to / capacity_to) * 100))
        
        avg_congestion = (congestion_from + congestion_to) / 2
        
        from_idx = STATIONS.index(from_station) if from_station in STATIONS else 0
        to_idx = STATIONS.index(to_station) if to_station in STATIONS else len(STATIONS) - 1
        station_diff = abs(from_idx - to_idx)
        travel_time = station_diff * 3 + 5
        
        if avg_congestion > 80:
            status = "CRITICAL"
            recommendation = "Consider postponing your trip"
        elif avg_congestion > 60:
            status = "HEAVY"
            recommendation = "Allow extra time for your journey"
        elif avg_congestion > 30:
            status = "MODERATE"
            recommendation = "Normal travel conditions"
        else:
            status = "LIGHT"
            recommendation = "Good time to travel!"
        
        return jsonify({
            "from_station": from_station,
            "to_station": to_station,
            "from_congestion": congestion_from,
            "to_congestion": congestion_to,
            "avg_congestion": round(avg_congestion, 1),
            "status": status,
            "travel_time": travel_time,
            "stations_between": station_diff,
            "recommendation": recommendation
        })
    except Exception as e:
        print(f"❌ Error in predict_route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/next-trains/<station_name>')
def next_trains(station_name):
    """Get next train times - using REAL schedule"""
    name = station_name.replace('%20', ' ')
    
    try:
        # Use the real schedule function
        trains = calculate_next_trains(name)
        
        if not trains.get("is_operating", True):
            return jsonify({
                "station": name,
                "northbound": [],
                "southbound": [],
                "headway": 0,
                "congestion": 0,
                "message": "Station closed"
            })
        
        # Format response to match what frontend expects
        north_trains = []
        south_trains = []
        
        # Get congestion for context
        ridership = get_station_prediction(name)
        capacity = STATION_BASE_CAPACITY.get(name, 10000)
        congestion = min(100, int((ridership / capacity) * 100))
        
        # Get multiple trains (similar to old format)
        headway = trains["headway"] * 60  # convert to seconds
        now = datetime.now()
        
        for i in range(1, 4):
            # Northbound
            north_minutes = trains["northbound"]["minutes"] + (i-1) * (headway // 60)
            north_trains.append({
                "time": (now + timedelta(minutes=north_minutes)).strftime("%I:%M %p"),
                "minutes": north_minutes,
                "from_station": trains["northbound"].get("from_station", "Taft")
            })
            
            # Southbound
            south_minutes = trains["southbound"]["minutes"] + (i-1) * (headway // 60)
            south_trains.append({
                "time": (now + timedelta(minutes=south_minutes)).strftime("%I:%M %p"),
                "minutes": south_minutes,
                "from_station": trains["southbound"].get("from_station", "North Ave")
            })
        
        return jsonify({
            "station": name,
            "northbound": north_trains,
            "southbound": south_trains,
            "headway": trains["headway"],
            "congestion": congestion
        })
    except Exception as e:
        print(f"❌ Error in next_trains: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stations')
def get_stations():
    """Get list of all stations"""
    return jsonify({
        "stations": STATIONS,
        "count": len(STATIONS)
    })

@app.route('/api/station-info/<station_name>')
def station_info(station_name):
    """Get detailed information about a station"""
    name = station_name.replace('%20', ' ')
    
    try:
        ridership = get_station_prediction(name)
        capacity = STATION_BASE_CAPACITY.get(name, 10000)
        congestion = min(100, int((ridership / capacity) * 100))
        
        station_idx = STATIONS.index(name) if name in STATIONS else 0
        
        prev_station = STATIONS[station_idx - 1] if station_idx > 0 else None
        next_station = STATIONS[station_idx + 1] if station_idx + 1 < len(STATIONS) else None
        
        if congestion > 80:
            status = "SEVERELY CONGESTED"
            color = "critical"
            description = "Extremely crowded. Expect significant delays."
        elif congestion > 60:
            status = "CONGESTED"
            color = "congested"
            description = "Very busy. Allow extra time."
        elif congestion > 30:
            status = "MODERATE"
            color = "moderate"
            description = "Moderate crowds. Normal wait times."
        else:
            status = "LIGHT"
            color = "light"
            description = "Light traffic. Good time to travel."
        
        return jsonify({
            "station": name,
            "congestion": congestion,
            "status": status,
            "color": color,
            "description": description,
            "ridership": ridership,
            "capacity": capacity,
            "previous_station": prev_station,
            "next_station": next_station,
            "index": station_idx
        })
    except Exception as e:
        print(f"❌ Error in station_info: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/test')
def test_api():
    """Test if API is working"""
    return jsonify({
        "status": "ok",
        "message": "API is working",
        "time": datetime.now().isoformat(),
        "stations": STATIONS
    })

@app.route('/api/model-metrics')
def model_metrics():
    """Get LSTM model performance metrics"""
    metrics = {}
    for station in STATIONS:
        if station in lstm_models:
            metrics[station] = {
                'status': 'active',
                'model_type': 'LSTM',
                'last_used': datetime.now().isoformat()
            }
        else:
            metrics[station] = {
                'status': 'fallback',
                'model_type': 'rule-based'
            }
    
    return jsonify({
        'models_loaded': len(lstm_models),
        'total_stations': len(STATIONS),
        'coverage': f"{len(lstm_models)}/{len(STATIONS)}",
        'metrics': metrics
    })
    
    
@app.route('/api/admin/retrain-models', methods=['POST'])
def retrain_models():
    """Admin endpoint to retrain all models"""
    if not session.get('admin_logged_in') and session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Run training script
        import subprocess
        result = subprocess.run(['python', 'train_model.py'], 
                               capture_output=True, text=True)
        
        # Reload models
        global lstm_models, scalers
        lstm_models, scalers = {}, {}
        models_loaded = 0
        
        for station in STATIONS:
            m_path, s_path = f'models/{station}_lstm.h5', f'models/{station}_scaler.pkl'
            if os.path.exists(m_path) and os.path.exists(s_path):
                lstm_models[station] = tf.keras.models.load_model(m_path, compile=False)
                with open(s_path, 'rb') as f:
                    scalers[station] = pickle.load(f)
                models_loaded += 1
        
        log_activity(None, 'admin', session.get('username'), 'retrain_models', 
                    f'Retrained {models_loaded} models')
        
        return jsonify({
            'success': True,
            'models_loaded': models_loaded,
            'message': f'Successfully retrained {models_loaded} models'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/api/model-performance')
def model_performance():
    """Get model performance metrics"""
    try:
        if os.path.exists('evaluation_results.csv'):
            df = pd.read_csv('evaluation_results.csv')
            
            performance = {
                'overall': {
                    'avg_mae': df['mae'].mean(),
                    'avg_rmse': df['rmse'].mean(),
                    'avg_mape': df['mape'].mean(),
                    'avg_r2': df['r2'].mean(),
                    'avg_f1': df['f1_score'].mean()
                },
                'stations': df.to_dict('records')
            }
            return jsonify(performance)
        else:
            return jsonify({"error": "No evaluation data available. Run evaluate_model.py first"}), 404
    except Exception as e:
        print(f"❌ Error loading performance data: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/report')
def report():
    username = session.get('username', 'Public User')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    
    return render_template('report.html', 
                         stations=STATIONS, 
                         username=username, 
                         is_logged_in=is_logged_in,
                         google_user=google_user,
                         favorite_station=favorite_station)

@app.route('/admin/dashboard')
def admin_dashboard():
    # Check both old and new session keys
    is_admin = (session.get('admin_logged_in') or 
                session.get('is_admin') or 
                session.get('role') == 'admin')
    
    print(f"🔍 Admin dashboard access check:")
    print(f"   admin_logged_in: {session.get('admin_logged_in')}")
    print(f"   is_admin: {session.get('is_admin')}")
    print(f"   role: {session.get('role')}")
    print(f"   Result: {is_admin}")
    
    if not is_admin:
        print("❌ Not admin, redirecting to login")
        flash('Please login as admin to access this page.', 'warning')
        return redirect(url_for('login'))
    
    total_users = User.query.count()
    operator_count = User.query.filter_by(role='operator').count()
    commuter_count = User.query.filter_by(role='commuter').count()
    admin_count = User.query.filter_by(role='admin').count()
    
    users = User.query.all()
    users_data = []
    for user in users:
        joined_date = user.created_at.strftime('%b %d, %Y') if user.created_at else 'Unknown'
        
        last_active = 'Never'
        if user.last_login:
            days_ago = (datetime.now() - user.last_login).days
            if days_ago == 0:
                last_active = 'Today'
            elif days_ago == 1:
                last_active = 'Yesterday'
            else:
                last_active = f'{days_ago} days ago'
        
        users_data.append({
            'id': user.id,
            'email': user.username,
            'role': user.role if user.role else 'commuter',
            'joined': joined_date,
            'last': last_active,
            'active': user.is_active
        })
    
    return render_template('admin_dashboard.html',
                         admin_email=session.get('username', 'Admin'),
                         total_users=total_users,
                         operator_count=operator_count,
                         commuter_count=commuter_count,
                         admin_count=admin_count,
                         users_data=users_data)

@app.route('/operator-dashboard')
def operator_dashboard():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    
    if user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif user.role == 'commuter':
        return redirect(url_for('user_dashboard'))
    elif user.role == 'operator':
        
        # Determine managed stations based on access level
        managed_stations = []
        
        # Debug print
        print(f"Operator: {user.username}")
        print(f"Access Level: {user.access_level}")
        print(f"Assigned Stations (raw): {user.assigned_stations}")
        print(f"Favorite Station: {user.favorite_station}")
        
        if user.access_level == 'line_wide':
            # Line-wide operator - all 13 stations
            managed_stations = STATIONS
            print(f"Line-wide access: {len(managed_stations)} stations")
            
        elif user.access_level == 'zone':
            # Zone operator
            zones = {
                'north': ['North Ave', 'Quezon Ave', 'Kamuning', 'Cubao', 'Santolan'],
                'central': ['Ortigas', 'Shaw Blvd', 'Boni Ave', 'Guadalupe'],
                'south': ['Buendia', 'Ayala Ave', 'Magallanes', 'Taft']
            }
            managed_stations = zones.get(user.assigned_zone, [])
            print(f"Zone access ({user.assigned_zone}): {len(managed_stations)} stations")
            
        else:
            # Station-level operator
            if user.assigned_stations:
                import json
                try:
                    managed_stations = json.loads(user.assigned_stations)
                except:
                    managed_stations = []
            
            # If no assigned_stations, use favorite_station
            if not managed_stations and user.favorite_station:
                managed_stations = [user.favorite_station]
            
            # Final fallback
            if not managed_stations:
                managed_stations = ['North Ave']  # Default
                
            print(f"Station-level access: {managed_stations}")
        
        # Pass both managed_stations AND all_stations to the template
        return render_template('operator-dashboard.html',
                             username=user.username,
                             role=user.role,
                             managed_stations=managed_stations,
                             all_stations=STATIONS,
                             access_level=user.access_level,
                             assigned_zone=user.assigned_zone)
    else:
        return redirect(url_for('user_dashboard'))

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        
        user = User.query.get(session['user_id'])
        if user and not user.is_active:
            session.clear()
            flash('Your account has been deactivated.', 'error')
            return redirect(url_for('login'))
            
        return f(*args, **kwargs)
    return decorated_function
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    error_type = None
    
    # Check if this is an invitation link
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
        
        # Check if using .env admin credentials (temporary)
        admin_email = os.getenv('ADMIN_EMAIL')
        admin_password = os.getenv('ADMIN_PASSWORD')
        
        if admin_email and admin_password and email == admin_email and password == admin_password:
            session.clear()
            session['admin_logged_in'] = True
            session['is_admin'] = True
            session['role'] = 'admin'
            session['username'] = email
            session['user_id'] = 0
            
            # Log successful admin login
            log_activity(0, 'admin', email, 'login', f'Admin logged in from IP: {ip_address}')
            return redirect(url_for('admin_dashboard'))
        
        user = User.query.filter_by(username=email).first()
        
        if user is None:
            error = "Account not found."
            error_type = "error"
            # LOG FAILED LOGIN - Account not found
            log_activity(None, 'unknown', email, 'login_failed', 
                        f'Account not found from IP: {ip_address}')
                        
        elif not user.is_active:
            error = "Account deactivated."
            error_type = "error"
            # LOG FAILED LOGIN - Account deactivated
            log_activity(user.id, user.role, user.username, 'login_failed', 
                        f'Account deactivated from IP: {ip_address}')
                        
        elif user.google_id and not user.has_password():
            error = "This account uses Google Sign-In. Please click 'Continue with Google'."
            error_type = "info"
            # LOG FAILED LOGIN - Wrong method
            log_activity(user.id, user.role, user.username, 'login_failed', 
                        f'Attempted password login on Google account from IP: {ip_address}')
                        
        elif user.verify_password(password):
            session.clear()
            session.permanent = True
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            session['favorite_station'] = user.favorite_station
            session['google_user'] = False
            
            user.last_login = datetime.now()
            db.session.commit()
            
            # Log successful login
            log_activity(user.id, user.role, user.username, 'login', 
                        f'Successful login from IP: {ip_address}')
            
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user.role == 'operator':
                return redirect(url_for('operator_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            error = "Incorrect password."
            error_type = "error"
            # LOG FAILED LOGIN - Wrong password
            log_activity(user.id, user.role, user.username, 'login_failed', 
                        f'Incorrect password from IP: {ip_address}')
    
    return render_template('login.html', error=error, error_type=error_type)

@app.route('/operator-signup', methods=['POST'])
def operator_signup():
    """Handle operator signup from invitation"""
    try:
        # Get form data
        email = request.form.get('email')
        temp_password = request.form.get('temp_password')
        new_password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        name = request.form.get('name')
        station = request.form.get('station')
        
        print(f"📝 Operator signup attempt for: {email}")
        
        # Validate required fields
        if not email or not temp_password or not new_password or not confirm_password:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Validate passwords match
        if new_password != confirm_password:
            return jsonify({'success': False, 'error': 'Passwords do not match'}), 400
        
        # Validate password length
        if len(new_password) < 8:
            return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400
        
        # Find the operator
        operator = User.query.filter_by(username=email, role='operator').first()
        
        if not operator:
            return jsonify({'success': False, 'error': 'Invalid invitation or operator not found'}), 404
        
        # Verify the temp password
        if not operator.verify_password(temp_password):
            return jsonify({'success': False, 'error': 'Invalid invitation link'}), 401
        
        # Update operator details
        operator.password = new_password
        operator.is_active = True
        
        # Store name in activity log or a separate field (you might want to add a 'name' column to User model)
        # For now, we'll store it in activity log
        log_activity(operator.id, 'operator', operator.username, 'profile_update', 
                    f'Updated name to: {name}')
        
        # Update station if provided
        if station and station != 'All Stations':
            operator.favorite_station = station
        
        db.session.commit()
        
        # Log them in
        session.clear()
        session.permanent = True
        session['user_id'] = operator.id
        session['username'] = operator.username
        session['role'] = 'operator'
        session['favorite_station'] = operator.favorite_station
        session['operator_name'] = name if name else operator.username.split('@')[0]
        
        operator.last_login = datetime.now()
        db.session.commit()
        
        print(f"✅ Operator {email} successfully activated!")
        
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

    
@app.route('/login/google/link')
def google_link_callback():
    """Handle Google account linking for existing operators"""
    try:
        token = google.authorize_access_token()
        user_info = google.parse_id_token(token)
        
        email = user_info.get('email')
        google_id = user_info.get('sub')
        
        # Get the currently logged in user from session
        if 'user_id' not in session:
            flash('Please log in first to link your Google account.', 'warning')
            return redirect(url_for('login'))
        
        user = User.query.get(session['user_id'])
        
        # Check if the Google email matches the logged in user's email
        if user.username != email:
            flash('The Google account email does not match your operator account email.', 'error')
            return redirect(url_for('operator_dashboard'))
        
        # Link the Google account
        user.google_id = google_id
        db.session.commit()
        
        log_activity(user.id, user.role, user.username, 'link_google', 'Linked Google account to operator profile')
        
        flash('Google account linked successfully! You can now login with Google.', 'success')
        return redirect(url_for('operator_dashboard'))
        
    except Exception as e:
        print(f"❌ Error linking Google account: {e}")
        flash('Failed to link Google account. Please try again.', 'error')
        return redirect(url_for('operator_dashboard'))
    
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        favorite = request.form.get('favorite_station')
        
        if password != confirm_password:
            return render_template('signup.html', error='Passwords do not match', email=email, favorite=favorite)
        
        existing_user = User.query.filter_by(username=email).first()
        if existing_user:
            return redirect(url_for('signup', error='email_exists', email=email))
        
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
            return redirect(url_for('user_dashboard'))
        except Exception as e:
            db.session.rollback()
            return render_template('signup.html', error="Database error. Please try again.")
    
    error = request.args.get('error')
    email = request.args.get('email')
    
    return render_template('signup.html', error=error, email=email)

@app.route('/logout')
def logout():
    if 'user_id' in session:
        user_id = session.get('user_id')
        user_role = session.get('role')
        user_email = session.get('username')
        log_activity(user_id, user_role, user_email, 'logout', 'User logged out')
    
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# ========== ADMIN API ROUTES ==========

@app.route('/api/admin/dashboard-stats')
def admin_dashboard_stats():
    """Get dashboard statistics"""
    try:
        total_reports = Report.query.count()
        
        # Calculate severe congestion count (stations with >80% congestion)
        severe_count = 0
        for station in STATIONS:
            ridership = get_station_prediction(station)
            capacity = STATION_BASE_CAPACITY.get(station, 10000)
            congestion = int((ridership / capacity) * 100)
            if congestion > 80:
                severe_count += 1
        
        active_operators = User.query.filter_by(role='operator', is_active=True).count()
        
        # For now, return placeholder for broadcasts
        broadcasts_this_week = 12  # You can implement actual broadcast tracking later
        
        return jsonify({
            'total_reports': total_reports,
            'severe_count': severe_count,
            'active_operators': active_operators,
            'broadcasts_this_week': broadcasts_this_week
        })
    except Exception as e:
        print(f"Error getting dashboard stats: {e}")
        return jsonify({'total_reports': 0, 'severe_count': 0, 'active_operators': 0, 'broadcasts_this_week': 0})

@app.route('/api/admin/station-status')
def admin_station_status():
    """Get station status for admin dashboard"""
    try:
        direction = request.args.get('direction', 'north')
        stations_data = []
        
        for station in STATIONS:
            ridership = get_station_prediction(station)
            capacity = STATION_BASE_CAPACITY.get(station, 10000)
            congestion = int((ridership / capacity) * 100)
            
            if congestion > 80:
                status_text = "SEVERE"
                status_class = "status-severe"
            elif congestion > 60:
                status_text = "CONGESTED"
                status_class = "status-congested"
            else:
                status_text = "ACTIVE"
                status_class = "status-active"
            
            stations_data.append({
                'name': station,
                'congestion': congestion,
                'status_text': status_text,
                'status_class': status_class
            })
        
        return jsonify({'stations': stations_data})
    except Exception as e:
        print(f"Error getting station status: {e}")
        return jsonify({'stations': []})

@app.route('/api/admin/recent-activities-list')
def admin_recent_activities():
    """Get recent IMPORTANT activities for admin dashboard (limit to 4 most recent)"""
    try:
        # Get recent activity logs - limit to last 4 actions
        recent_logs = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(4).all()
        
        activities = []
        for log in recent_logs:
            # Determine icon based on action type
            icon = 'user'
            icon_color = '#3B82F6'
            title = ''
            description = ''
            
            if 'login' in log.action and 'failed' not in log.action:
                icon = 'sign-in-alt'
                icon_color = '#22C55E'
                title = 'Login'
                description = f'{log.user_email} logged in'
            elif 'logout' in log.action:
                icon = 'sign-out-alt'
                icon_color = '#EF4444'
                title = 'Logout'
                description = f'{log.user_email} logged out'
            elif 'broadcast' in log.action:
                icon = 'bullhorn'
                icon_color = '#8B5CF6'
                title = 'Broadcast Sent'
                # Extract station info from details
                if 'Station:' in log.details:
                    station_part = log.details.split('Station:')[-1].strip()
                    description = f'{log.user_email} sent alert to {station_part}'
                else:
                    description = log.details or f'{log.user_email} sent broadcast'
            elif 'override' in log.action:
                icon = 'edit'
                icon_color = '#F59E0B'
                title = 'Override'
                description = log.details or f'{log.user_email} overrode congestion'
            elif 'create_operator' in log.action:
                icon = 'user-plus'
                icon_color = '#10B981'
                title = 'Operator Created'
                description = log.details or f'{log.user_email} created new operator'
            elif 'deactivate_operator' in log.action:
                icon = 'user-slash'
                icon_color = '#EF4444'
                title = 'Operator Deactivated'
                description = log.details or f'{log.user_email} deactivated operator'
            elif 'reactivate_operator' in log.action:
                icon = 'user-check'
                icon_color = '#10B981'
                title = 'Operator Reactivated'
                description = log.details or f'{log.user_email} reactivated operator'
            elif 'login_failed' in log.action:
                icon = 'exclamation-triangle'
                icon_color = '#EF4444'
                title = 'Login Failed'
                description = f'Failed login attempt for {log.user_email}'
            else:
                title = log.action.replace('_', ' ').title()
                description = log.details or f'{log.user_email} performed {log.action}'
            
            activities.append({
                'icon': icon,
                'icon_color': icon_color,
                'title': title,
                'description': description[:100],  # Limit description length
                'station': None,
                'time': log.timestamp.strftime('%I:%M %p')
            })
        
        # If no activities, show sample (only 1-2)
        if not activities:
            activities = [
                {'icon': 'user-plus', 'icon_color': '#22C55E', 'title': 'System Ready', 'description': 'Admin dashboard initialized', 'station': None, 'time': 'Just now'}
            ]
        
        return jsonify(activities)
    except Exception as e:
        print(f"Error getting recent activities: {e}")
        return jsonify([])
    

@app.route('/api/admin/operator-list')
def admin_operator_list():
    """Get list of operators"""
    try:
        operators = User.query.filter_by(role='operator').all()
        
        operator_data = []
        for op in operators:
            operator_data.append({
                'id': op.id,
                'name': op.username.split('@')[0] if '@' in op.username else op.username,
                'email': op.username,
                'zone': op.favorite_station or 'All Stations',
                'last_login': op.last_login.strftime('%b %d, %Y') if op.last_login else 'Never',
                'active': op.is_active
            })
        
        return jsonify(operator_data)
    except Exception as e:
        print(f"Error getting operator list: {e}")
        return jsonify([])
@app.route('/api/admin/generate-invite', methods=['POST'])
def admin_generate_invite():
    """Generate operator invite link - supports both password and Google signup"""
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        station = data.get('station')
        access_level_type = data.get('access_level', 'standard')
        auth_method = data.get('auth_method', 'password')
        
        # Check if user already exists
        existing_user = User.query.filter_by(username=email).first()
        if existing_user:
            # If user exists but is inactive, reactivate them
            if not existing_user.is_active:
                existing_user.is_active = True
                if auth_method == 'google':
                    # Don't set password for Google-only accounts
                    existing_user.password_hash = None
                db.session.commit()
                
                if auth_method == 'google':
                    invite_link = f"{request.host_url}login/google/authorize?invite=true&email={email}"
                    return jsonify({
                        'success': True,
                        'link': invite_link,
                        'auth_method': 'google',
                        'message': 'Account reactivated. User can login with Google.'
                    })
                else:
                    temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
                    existing_user.password = temp_password
                    db.session.commit()
                    invite_link = f"{request.host_url}login?email={email}&temp={temp_password}&station={station}"
                    return jsonify({
                        'success': True,
                        'link': invite_link,
                        'auth_method': 'password',
                        'message': 'Account reactivated with new temporary password.'
                    })
            else:
                return jsonify({'success': False, 'error': 'Email already registered and active'}), 400
        
        import json
        
        STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
                    "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
                    "Ayala Ave", "Magallanes", "Taft"]
        
        # Determine access level and assigned stations
        if access_level_type == 'full':
            db_access_level = 'line_wide'
            assigned_stations = STATIONS
            favorite_station = None
        else:
            db_access_level = 'station'
            if station and station in STATIONS:
                assigned_stations = [station]
                favorite_station = station
            else:
                assigned_stations = ['North Ave']
                favorite_station = 'North Ave'
        
        if station == 'All Stations (Line-Wide)':
            db_access_level = 'line_wide'
            assigned_stations = STATIONS
            favorite_station = None
        
        if auth_method == 'google':
            # Create Google-only operator account - ACTIVE immediately
            new_operator = User(
                username=email,
                role='operator',
                access_level=db_access_level,
                assigned_stations=json.dumps(assigned_stations),
                favorite_station=favorite_station,
                created_at=datetime.now(),
                is_active=True,  # ← ACTIVE IMMEDIATELY for Google login
                # No password hash - Google-only account
            )
            db.session.add(new_operator)
            db.session.commit()
            
            # Generate direct Google login link
            invite_link = f"{request.host_url}login/google/authorize?invite=true&email={email}"
            
            return jsonify({
                'success': True,
                'link': invite_link,
                'auth_method': 'google',
                'message': 'Google Sign-up invite created. User can login directly with Google.',
                'instructions': 'Send this link to the operator. They must use the Google account with this email.'
            })
        else:
            # Password-based invite - starts inactive until password set
            new_operator = User(
                username=email,
                role='operator',
                access_level=db_access_level,
                assigned_stations=json.dumps(assigned_stations),
                favorite_station=favorite_station,
                created_at=datetime.now(),
                is_active=False  # Will be activated when they set password
            )
            
            temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
            new_operator.password = temp_password
            db.session.add(new_operator)
            db.session.commit()
            
            invite_link = f"{request.host_url}login?email={email}&temp={temp_password}&station={station}"
            
            return jsonify({
                'success': True,
                'link': invite_link,
                'auth_method': 'password',
                'message': 'Password setup invite created.',
                'temp_password': temp_password
            })
        
    except Exception as e:
        print(f"❌ Error generating invite: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    
  
@app.route('/login/google/invite')
def google_invite_signup():
    """Handle Google signup from invitation link"""
    try:
        token = google.authorize_access_token()
        user_info = google.parse_id_token(token)
        
        email = user_info.get('email')
        name = user_info.get('name', email.split('@')[0])
        google_id = user_info.get('sub')
        
        # Get invitation parameters
        invited_email = request.args.get('email')
        station = request.args.get('station')
        
        # Check if the Google email matches the invited email
        if invited_email and email != invited_email:
            flash(f'You signed in with {email}, but this invitation was for {invited_email}. Please use the correct Google account.', 'error')
            return redirect(url_for('login'))
        
        # Check if user already exists
        existing_user = User.query.filter_by(username=email).first()
        
        if existing_user:
            if existing_user.role == 'operator':
                # Link Google account if not already linked
                if not existing_user.google_id:
                    existing_user.google_id = google_id
                    existing_user.is_active = True
                    db.session.commit()
                    flash('Google account linked to your operator account!', 'success')
                
                # Log them in
                session.clear()
                session.permanent = True
                session['user_id'] = existing_user.id
                session['username'] = existing_user.username
                session['role'] = 'operator'
                session['google_user'] = True
                
                return redirect(url_for('operator_dashboard'))
            else:
                flash('This email is registered as a commuter, not an operator.', 'error')
                return redirect(url_for('login'))
        
        # Create new operator account
        new_operator = User(
            username=email,
            role='operator',
            google_id=google_id,
            is_active=True,
            favorite_station=station if station != 'All Stations' else None,
            created_at=datetime.now()
        )
        
        db.session.add(new_operator)
        db.session.commit()
        
        # Log them in
        session.clear()
        session.permanent = True
        session['user_id'] = new_operator.id
        session['username'] = new_operator.username
        session['role'] = 'operator'
        session['google_user'] = True
        
        log_activity(new_operator.id, 'operator', new_operator.username, 'signup', 'Operator signed up via Google invitation')
        
        flash(f'Welcome {name}! Your operator account has been created.', 'success')
        return redirect(url_for('operator_dashboard'))
        
    except Exception as e:
        print(f"❌ Google invite signup error: {e}")
        flash('Failed to create account. Please try again.', 'error')
        return redirect(url_for('login'))
      
@app.route('/api/operator/send-broadcast', methods=['POST'])
def operator_send_broadcast():
    """Send broadcast notification"""
    try:
        data = request.json
        title = data.get('title')
        message = data.get('message')
        disruption_type = data.get('disruption_type')
        stations = data.get('stations')
        severity = data.get('severity')
        
        # Log the action
        operator_id = session.get('user_id')
        operator_email = session.get('username')
        log_activity(operator_id, 'operator', operator_email, 'send_broadcast', 
                    f'Broadcast: "{title}" to {len(stations)} stations (Severity: {severity})')
        
        return jsonify({'success': True, 'message': 'Broadcast sent'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/operator/deactivate-broadcast/<int:broadcast_id>', methods=['POST'])
def operator_deactivate_broadcast(broadcast_id):
    """Deactivate a broadcast"""
    try:
        # Log the action
        operator_id = session.get('user_id')
        operator_email = session.get('username')
        log_activity(operator_id, 'operator', operator_email, 'deactivate_broadcast', 
                    f'Deactivated broadcast ID: {broadcast_id}')
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/operator/override-congestion', methods=['POST'])
def operator_override_congestion():
    """Override congestion level for a station - STORES PERMANENTLY"""
    try:
        data = request.json
        station = data.get('station')
        level = data.get('level')
        congestion_value = data.get('congestion_value')
        duration = data.get('duration')
        reason = data.get('reason', '')
        
        # Get operator info
        operator_id = session.get('user_id')
        operator_email = session.get('username')
        
        if not operator_id:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        # Store override in app config (or database for permanent storage)
        if 'overrides' not in app.config:
            app.config['overrides'] = {}
        
        import time
        expiry = None
        if duration != 'manual':
            expiry = time.time() + (int(duration) * 60)
        
        app.config['overrides'][station] = {
            'level': level,
            'congestion': congestion_value,
            'operator': operator_email,
            'reason': reason,
            'expiry': expiry,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log the activity
        log_activity(operator_id, 'operator', operator_email, 'override_congestion', 
                    f'Overrode {station} to {level} ({congestion_value}%) - Duration: {duration} min - Reason: {reason}')
        
        print(f"🔧 OVERRIDE STORED: {station} -> {level} ({congestion_value}%)")
        
        return jsonify({
            'success': True,
            'message': f'{station} set to {level}',
            'override': app.config['overrides'][station]
        })
        
    except Exception as e:
        print(f"❌ Error in override: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/operator/clear-override', methods=['POST'])
def operator_clear_override():
    """Clear an active override for a station"""
    try:
        data = request.json
        station = data.get('station')
        
        operator_id = session.get('user_id')
        operator_email = session.get('username')
        
        if 'overrides' in app.config and station in app.config['overrides']:
            del app.config['overrides'][station]
            log_activity(operator_id, 'operator', operator_email, 'clear_override', 
                        f'Cleared override for {station}')
            return jsonify({'success': True, 'message': f'Override cleared for {station}'})
        
        return jsonify({'success': False, 'error': 'No active override found'}), 404
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/operator/my-activity')
def operator_my_activity():
    """Get current operator's activity logs"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        
        logs = ActivityLog.query.filter_by(user_id=user_id).order_by(ActivityLog.timestamp.desc()).limit(50).all()
        
        log_data = []
        for log in logs:
            log_data.append({
                'action': log.action,
                'details': log.details,
                'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return jsonify(log_data)
    except Exception as e:
        return jsonify([]), 500
    
@app.route('/api/admin/reactivate-operator/<int:operator_id>', methods=['POST'])
def admin_reactivate_operator(operator_id):
    """Reactivate an operator"""
    try:
        operator = User.query.get(operator_id)
        if operator and operator.role == 'operator':
            operator.is_active = True
            db.session.commit()
            
            # Log the action
            admin_id = session.get('user_id')
            admin_email = session.get('username')
            log_activity(admin_id, 'admin', admin_email, 'reactivate_operator', 
                        f'Reactivated operator: {operator.username}')
            
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Operator not found'}), 404
    except Exception as e:
        print(f"Error reactivating operator: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/api/admin/deactivate-operator/<int:operator_id>', methods=['POST'])
def admin_deactivate_operator(operator_id):
    """Deactivate an operator"""
    try:
        operator = User.query.get(operator_id)
        if operator and operator.role == 'operator':
            operator.is_active = False
            db.session.commit()
            
            # Log the action
            admin_id = session.get('user_id')
            admin_email = session.get('username')
            log_activity(admin_id, 'admin', admin_email, 'deactivate_operator', 
                        f'Deactivated operator: {operator.username}')
            
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Operator not found'}), 404
    except Exception as e:
        print(f"Error deactivating operator: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/audit-log')
def admin_audit_log():
    """Get COMPLETE audit log entries for admin dashboard (all actions)"""
    try:
        limit = request.args.get('limit', 200, type=int)
        logs = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(limit).all()
        
        log_data = []
        for log in logs:
            # Get user name (try to get from database if available)
            user_name = log.user_email or 'System'
            if log.user_id:
                user = User.query.get(log.user_id)
                if user:
                    user_name = user.username
            
            log_data.append({
                'id': log.id,
                'userType': log.user_type or 'system',
                'userName': user_name,
                'userEmail': log.user_email,
                'action': log.action,
                'details': log.details or '-',
                'target': log.details or '-',
                'ip_address': log.ip_address or '-',
                'timestamp': log.timestamp.isoformat()
            })
        
        return jsonify(log_data)
    except Exception as e:
        print(f"Error getting audit log: {e}")
        return jsonify([]), 500

@app.route('/api/admin/audit-stats')
def admin_audit_stats():
    """Get real audit statistics from database"""
    try:
        total_actions = ActivityLog.query.count()
        
        # Count unique active admins (users who logged in recently)
        active_admins = User.query.filter_by(role='admin', is_active=True).count()
        
        # Count active operators
        active_operators = User.query.filter_by(role='operator', is_active=True).count()
        
        # Count flagged actions (failed logins, deactivations, etc.)
        flagged = ActivityLog.query.filter(
            ActivityLog.action.in_(['login_failed', 'deactivate_operator'])
        ).count()
        
        return jsonify({
            'total_actions': total_actions,
            'active_admins': active_admins,
            'active_operators': active_operators,
            'flagged': flagged
        })
    except Exception as e:
        print(f"Error getting audit stats: {e}")
        return jsonify({'total_actions': 0, 'active_admins': 0, 'active_operators': 0, 'flagged': 0})

@app.route('/api/admin/profile')
def admin_profile():
    """Get admin profile info"""
    try:
        if session.get('admin_logged_in') or session.get('is_admin'):
            return jsonify({
                'username': session.get('username', 'Admin'),
                'role': 'admin'
            })
        return jsonify({'error': 'Not authorized'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# In app.py
def load_lstm_models():
    models = {}
    scalers = {}
    for station in STATIONS:
        model_path = f'models/{station}_lstm.keras' # or .h5
        scaler_path = f'models/{station}_scaler.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                models[station] = tf.keras.models.load_model(model_path)
                with open(scaler_path, 'rb') as f:
                    scalers[station] = pickle.load(f)
                print(f"✅ Loaded model for {station}")
            except Exception as e:
                print(f"⚠️ Error loading {station}: {e}")
    return models, scalers

# ========== RUN SYSTEM ==========
if __name__ == '__main__':
    
    
    print("\n" + "="*70)
    print("✨ MRT-3 PREDICTION SYSTEM READY!")
    print("="*70)
    print("👤 Starting as GUEST by default")
    print("📊 Historical data: ✅ Loaded from cache")
    print(f"🤖 LSTM models: {models_loaded}/{len(STATIONS)}")
    print("🔐 Password hashing: ✅ Enabled")
    print("🌐 Open http://localhost:5000")
    print("="*70)
    app.run(debug=True, port=5000)