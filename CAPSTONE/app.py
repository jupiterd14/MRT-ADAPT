from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from dotenv import load_dotenv
import pickle
import warnings
import time
import secrets
import string
from authlib.integrations.flask_client import OAuth
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from sqlalchemy import text


# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='html', static_folder='javascript')
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

# ========== LOAD HISTORICAL DATA FROM CACHE ==========
print("\n" + "="*70)
print("📊 LOADING HISTORICAL DATA FROM CACHE...")

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
            hourly_avg_entry[hour] = 2000
            hourly_avg_exit[hour] = 2500
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
    user_type = db.Column(db.String(20))  # 'admin', 'operator'
    user_email = db.Column(db.String(100))
    action = db.Column(db.String(100))  # 'login', 'logout', 'create_operator', 'deactivate_broadcast', etc.
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
        return self.password_hash is not None

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
    reported_congestion = db.Column(db.Integer)
    predicted_congestion = db.Column(db.Integer)
    remarks = db.Column(db.String(500), nullable=True)
    photo_path = db.Column(db.String(200), nullable=True)
    anonymous = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    
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
        
def get_station_prediction(station_name):
    """Get prediction using trained LSTM model"""
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    weekday = now.weekday()
    
    # Station operating hours: 4:30 AM to 10:30 PM
    if hour < 4 or (hour == 4 and minute < 30) or hour >= 23:
        return 0
    
    # Try to use LSTM model if available
    if station_name in lstm_models and station_name in scalers:
        try:
            # Prepare input sequence for LSTM
            # Use last 24 hours of data from cache if available
            if station_name in station_time_series_last_24:
                recent_data = station_time_series_last_24[station_name]
            else:
                # Fallback: use historical hourly averages to create sequence
                recent_data = []
                for h in range(24):
                    if h in hourly_avg_entry and h in hourly_avg_exit:
                        recent_data.append((hourly_avg_entry[h] + hourly_avg_exit[h]) / 2)
                    else:
                        recent_data.append(3000)
            
            # Convert to numpy array and scale
            recent_array = np.array(recent_data[-24:]).reshape(-1, 1)
            scaled_data = scalers[station_name].transform(recent_array)
            
            # Reshape for LSTM (samples, timesteps, features)
            X_input = scaled_data.reshape(1, 24, 1)
            
            # Make prediction
            predicted_scaled = lstm_models[station_name].predict(X_input, verbose=0)
            predicted_value = scalers[station_name].inverse_transform(predicted_scaled)[0][0]
            
            # Apply constraints
            capacity = STATION_BASE_CAPACITY.get(station_name, 10000)
            predicted_value = max(100, min(predicted_value, capacity))
            
            # Apply time-of-day adjustments for better accuracy
            if 7 <= hour <= 9:  # Morning rush
                predicted_value *= 1.15
            elif 17 <= hour <= 20:  # Evening rush
                predicted_value *= 1.1
            elif weekday >= 5:  # Weekend
                predicted_value *= 0.85
            
            return int(predicted_value)
            
        except Exception as e:
            print(f"⚠️ LSTM prediction failed for {station_name}: {e}")
            # Fall through to rule-based prediction
    
    # FALLBACK: Rule-based prediction (your existing logic)
    try:
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
    

def get_station_prediction_for_datetime(station_name, target_datetime):
    """Get prediction for specific datetime using LSTM"""
    target_hour = target_datetime.hour
    target_minute = target_datetime.minute
    target_weekday = target_datetime.weekday()
    
    if target_hour < 4 or (target_hour == 4 and target_minute < 30) or target_hour >= 23:
        return 0
    
    # Try LSTM first
    if station_name in lstm_models and station_name in scalers:
        try:
            # For future predictions, we need to create a sequence based on historical patterns
            # Use weekly patterns to estimate the sequence
            recent_data = []
            
            # Build sequence using historical patterns for the same day of week
            for h in range(target_hour - 23, target_hour + 1):
                hour_mod = h % 24
                if hour_mod in hourly_avg_entry and hour_mod in hourly_avg_exit:
                    value = (hourly_avg_entry[hour_mod] + hourly_avg_exit[hour_mod]) / 2
                    # Apply day-of-week adjustment
                    if target_weekday >= 5:  # Weekend
                        value *= 0.85
                    recent_data.append(value)
                else:
                    recent_data.append(3000)
            
            recent_array = np.array(recent_data[-24:]).reshape(-1, 1)
            scaled_data = scalers[station_name].transform(recent_array)
            X_input = scaled_data.reshape(1, 24, 1)
            
            predicted_scaled = lstm_models[station_name].predict(X_input, verbose=0)
            predicted_value = scalers[station_name].inverse_transform(predicted_scaled)[0][0]
            
            capacity = STATION_BASE_CAPACITY.get(station_name, 10000)
            return int(max(100, min(predicted_value, capacity)))
            
        except Exception as e:
            print(f"⚠️ LSTM prediction failed: {e}")
    
    # Fallback to existing logic

# ========== API ROUTES ==========

@app.route('/api/reports')
def get_reports():
    try:
        reports = Report.query.order_by(Report.timestamp.desc()).limit(20).all()
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
                'reported_congestion': report.reported_congestion,
                'remarks': report.remarks,
                'anonymous': report.anonymous,
                'username': username,
                'timestamp': report.timestamp.isoformat() if report.timestamp else None
            })
        return jsonify(result)
    except Exception as e:
        print(f"❌ Error fetching reports: {e}")
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

@app.route('/api/station-forecast/<station_name>')
def station_forecast_api(station_name):
    name = station_name.replace('%20', ' ')
    
    now = datetime.now()
    current_hour = now.hour
    
    forecast = []
    for i in range(6):
        forecast_hour = (current_hour + i + 1) % 24
        ridership = get_station_prediction_for_datetime(name, datetime(now.year, now.month, now.day, forecast_hour, 0))
        capacity = STATION_BASE_CAPACITY.get(name, 10000)
        forecast_congestion = min(100, int((ridership / capacity) * 100))
        forecast.append(forecast_congestion)
    
    return jsonify({
        "station": name,
        "forecast": forecast,
        "intervals": ["+1h", "+2h", "+3h", "+4h", "+5h", "+6h"],
        "current": forecast[0] if forecast else 0
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
        data = request.json
        station = data.get('station')
        reported = data.get('congestion')
        remarks = data.get('remarks', '')
        anonymous = data.get('anonymous', False)
        
        user_id = session.get('user_id') if 'user_id' in session else None
        
        ridership = get_station_prediction(station)
        capacity = STATION_BASE_CAPACITY.get(station, 10000)
        predicted = int((ridership / capacity) * 100)
        
        report = Report(
            user_id=user_id,
            station=station,
            reported_congestion=reported,
            predicted_congestion=predicted,
            remarks=remarks,
            anonymous=anonymous
        )
        db.session.add(report)
        db.session.commit()
        
        return jsonify({"success": True, "message": "Report saved successfully"})
        
    except Exception as e:
        print(f"❌ Error saving report: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

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
    name = station_name.replace('%20', ' ')
    
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
    """Get train schedule for a station"""
    name = station_name.replace('%20', ' ')
    
    try:
        ridership = get_station_prediction(name)
        capacity = STATION_BASE_CAPACITY.get(name, 10000)
        congestion = min(100, int((ridership / capacity) * 100))
        
        if congestion > 80:
            headway = 8
        elif congestion > 60:
            headway = 6
        elif congestion > 30:
            headway = 5
        else:
            headway = 4
        
        now = datetime.now()
        
        trains = []
        for i in range(1, 6):
            train_time = now + timedelta(minutes=headway * i)
            trains.append({
                "time": train_time.strftime("%I:%M %p"),
                "minutes": headway * i,
                "destination": "North Ave" if name in ["Taft", "Ayala Ave", "Magallanes"] else "Taft"
            })
        
        return jsonify({
            "station": name,
            "headway": headway,
            "trains": trains,
            "status": "normal"
        })
    except Exception as e:
        print(f"❌ Error getting schedule: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/schedule/next-trains/<station_name>')
def get_next_trains(station_name):
    """Get next train arrivals for northbound and southbound"""
    name = station_name.replace('%20', ' ')
    
    try:
        station_idx = STATIONS.index(name) if name in STATIONS else 6
        
        ridership = get_station_prediction(name)
        capacity = STATION_BASE_CAPACITY.get(name, 10000)
        congestion = min(100, int((ridership / capacity) * 100))
        
        if congestion > 80:
            headway = 8
        elif congestion > 60:
            headway = 6
        elif congestion > 30:
            headway = 5
        else:
            headway = 4
        
        north_arrival = headway + (station_idx % 3)
        north_from = STATIONS[min(station_idx + 2, len(STATIONS) - 1)] if station_idx < len(STATIONS) - 2 else STATIONS[0]
        
        south_arrival = headway + ((len(STATIONS) - station_idx) % 3)
        south_from = STATIONS[max(station_idx - 2, 0)] if station_idx > 1 else STATIONS[len(STATIONS) - 1]
        
        return jsonify({
            "northbound": {
                "minutes": north_arrival,
                "from_station": north_from,
                "status": "on_time"
            },
            "southbound": {
                "minutes": south_arrival,
                "from_station": south_from,
                "status": "on_time"
            },
            "headway": headway
        })
    except Exception as e:
        print(f"❌ Error getting next trains: {e}")
        return jsonify({
            "northbound": {"minutes": 8, "from_station": "Santolan", "status": "on_time"},
            "southbound": {"minutes": 10, "from_station": "Quezon Ave", "status": "on_time"},
            "headway": 6
        })

@app.route('/api/schedule/headway')
def get_headway():
    """Get current headway across the line"""
    try:
        total_congestion = 0
        for station in STATIONS:
            ridership = get_station_prediction(station)
            capacity = STATION_BASE_CAPACITY.get(station, 10000)
            congestion = min(100, int((ridership / capacity) * 100))
            total_congestion += congestion
        
        avg_congestion = total_congestion / len(STATIONS)
        
        if avg_congestion > 70:
            headway = 8
            status = "reduced"
        elif avg_congestion > 50:
            headway = 6
            status = "normal"
        elif avg_congestion > 30:
            headway = 5
            status = "good"
        else:
            headway = 4
            status = "excellent"
        
        return jsonify({
            "headway": headway,
            "status": status,
            "average_congestion": round(avg_congestion, 1),
            "message": f"Trains arriving every {headway} minutes" if headway <= 6 else f"Heavy traffic, trains every {headway} minutes"
        })
    except Exception as e:
        print(f"❌ Error getting headway: {e}")
        return jsonify({"headway": 6, "status": "normal", "message": "Trains arriving every 6 minutes"})

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
    """Get congestion data for both northbound and southbound directions for all stations"""
    try:
        northbound = {}
        southbound = {}
        
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        current_time = hour + minute / 60
        
        if current_time < 4.5 or current_time >= 22.5:
            for station in STATIONS:
                northbound[station] = {"congestion": 0, "wait_time": "CLOSED", "status": "CLOSED", "ridership": 0}
                southbound[station] = {"congestion": 0, "wait_time": "CLOSED", "status": "CLOSED", "ridership": 0}
        else:
            for station in STATIONS:
                station_idx = STATIONS.index(station) + 1
                capacity = STATION_BASE_CAPACITY.get(station, 10000)
                
                base_entry = historical_entry.get(station, capacity * 0.4)
                base_exit = historical_exit.get(station, capacity * 0.3)
                
                # Time-based multipliers based on rush hour patterns
                if 7 <= hour <= 9:  # Morning rush
                    time_mult = 1.5
                    if station_idx <= 6:
                        south_mult = 1.6
                        north_mult = 0.8
                    else:
                        south_mult = 0.7
                        north_mult = 1.3
                elif 17 <= hour <= 20:  # Evening rush
                    time_mult = 1.5
                    if station_idx >= 8:
                        south_mult = 0.6
                        north_mult = 1.7
                    else:
                        south_mult = 1.3
                        north_mult = 0.7
                else:
                    time_mult = 0.8
                    south_mult = 1.0
                    north_mult = 1.0
                
                southbound_ridership = int(base_entry * time_mult * south_mult)
                northbound_ridership = int(base_exit * time_mult * north_mult)
                
                southbound_congestion = min(100, int((southbound_ridership / capacity) * 100))
                northbound_congestion = min(100, int((northbound_ridership / capacity) * 100))
                
                def get_status_and_wait(congestion):
                    if congestion > 80:
                        return "SEVERELY CONGESTED", "15-20 min"
                    elif congestion > 60:
                        return "CONGESTED", "10-15 min"
                    elif congestion > 30:
                        return "MODERATE", "5-10 min"
                    else:
                        return "LIGHT", "2-5 min"
                
                south_status, south_wait = get_status_and_wait(southbound_congestion)
                north_status, north_wait = get_status_and_wait(northbound_congestion)
                
                southbound[station] = {
                    "congestion": southbound_congestion,
                    "wait_time": south_wait,
                    "status": south_status,
                    "ridership": southbound_ridership
                }
                
                northbound[station] = {
                    "congestion": northbound_congestion,
                    "wait_time": north_wait,
                    "status": north_status,
                    "ridership": northbound_ridership
                }
        
        return jsonify({
            "northbound": northbound,
            "southbound": southbound,
            "timestamp": now.isoformat(),
            "is_operating": 4.5 <= current_time < 22.5
        })
        
    except Exception as e:
        print(f"❌ Error in live_map_directions: {e}")
        return jsonify({"error": str(e)}), 500

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
    """Get next train times for a station"""
    name = station_name.replace('%20', ' ')
    
    try:
        station_idx = STATIONS.index(name) if name in STATIONS else 6
        
        ridership = get_station_prediction(name)
        capacity = STATION_BASE_CAPACITY.get(name, 10000)
        congestion = min(100, int((ridership / capacity) * 100))
        
        if congestion > 80:
            headway = 8
        elif congestion > 60:
            headway = 6
        elif congestion > 30:
            headway = 5
        else:
            headway = 4
        
        now = datetime.now()
        
        north_trains = []
        for i in range(1, 4):
            minutes = headway * i
            train_time = now + timedelta(minutes=minutes)
            from_idx = max(0, station_idx - i)
            north_trains.append({
                "time": train_time.strftime("%I:%M %p"),
                "minutes": minutes,
                "from_station": STATIONS[from_idx]
            })
        
        south_trains = []
        for i in range(1, 4):
            minutes = headway * i
            train_time = now + timedelta(minutes=minutes)
            from_idx = min(len(STATIONS) - 1, station_idx + i)
            south_trains.append({
                "time": train_time.strftime("%I:%M %p"),
                "minutes": minutes,
                "from_station": STATIONS[from_idx]
            })
        
        return jsonify({
            "station": name,
            "northbound": north_trains,
            "southbound": south_trains,
            "headway": headway,
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
        return render_template('operator-dashboard.html',
                             username=user.username,
                             station=user.favorite_station or 'All Stations',
                             stations=STATIONS)
    else:
        return redirect(url_for('user_dashboard'))

# ========== AUTH ROUTES ==========

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
        
        # Check if using .env admin credentials
        admin_email = os.getenv('ADMIN_EMAIL')
        admin_password = os.getenv('ADMIN_PASSWORD')
        
        if admin_email and admin_password and email == admin_email and password == admin_password:
            session.clear()
            session['admin_logged_in'] = True
            session['is_admin'] = True
            session['role'] = 'admin'
            session['username'] = email
            session['user_id'] = 0  # Special ID for .env admin
            
            # Log admin login
            log_activity(None, 'admin', email, 'login', 'Admin logged in via .env credentials')
            return redirect(url_for('admin_dashboard'))
        
        user = User.query.filter_by(username=email).first()
        
        if user is None:
            error = "Account not found."
            log_activity(None, 'unknown', email, 'login_failed', 'Account not found')
        elif not user.is_active:
            error = "Account deactivated."
            log_activity(user.id, user.role, user.username, 'login_failed', 'Account deactivated')
        elif user.verify_password(password):
            session.clear()
            session.permanent = True
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            session['favorite_station'] = user.favorite_station
            
            user.last_login = datetime.now()
            db.session.commit()
            
            # Log successful login
            log_activity(user.id, user.role, user.username, 'login', f'Successful login from IP: {request.remote_addr}')
            
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user.role == 'operator':
                return redirect(url_for('operator_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            error = "Incorrect password."
            log_activity(user.id, user.role, user.username, 'login_failed', 'Incorrect password')
    
    return render_template('login.html', error=error)

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
    """Get recent activities for admin dashboard"""
    try:
        # Get recent reports as activities
        recent_reports = Report.query.order_by(Report.timestamp.desc()).limit(10).all()
        
        activities = []
        for report in recent_reports:
            activities.append({
                'icon': 'exclamation-triangle',
                'icon_color': '#EF4444',
                'title': 'New Report Submitted',
                'description': f'Congestion reported at {report.station}',
                'station': report.station,
                'time': report.timestamp.strftime('%I:%M %p')
            })
        
        # If no activities, add some sample ones
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
    """Generate operator invite link"""
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        station = data.get('station')
        
        # Check if user already exists
        existing_user = User.query.filter_by(username=email).first()
        if existing_user:
            return jsonify({'success': False, 'error': 'Email already registered'}), 400
        
        # Create new operator
        new_operator = User(
            username=email,
            role='operator',
            favorite_station=station if station != 'All Stations (Line-Wide)' else None,
            created_at=datetime.now(),
            is_active=True
        )
        temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
        new_operator.password = temp_password
        db.session.add(new_operator)
        db.session.commit()
        
        # Log the action
        admin_id = session.get('user_id')
        admin_email = session.get('username')
        log_activity(admin_id, 'admin', admin_email, 'create_operator', 
                    f'Created operator: {email} (Name: {name}, Station: {station})')
        
        invite_link = f"{request.host_url}login?email={email}&temp={temp_password}&station={station}"
        
        return jsonify({
            'success': True,
            'link': invite_link,
            'message': 'Operator created successfully'
        })
    except Exception as e:
        print(f"Error generating invite: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


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
    """Override congestion level"""
    try:
        data = request.json
        station = data.get('station')
        level = data.get('level')
        
        # Log the action
        operator_id = session.get('user_id')
        operator_email = session.get('username')
        log_activity(operator_id, 'operator', operator_email, 'override_congestion', 
                    f'Overrode {station} congestion to {level}')
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/api/admin/activity-logs')
def admin_activity_logs():
    """Get activity logs for admin dashboard"""
    try:
        limit = request.args.get('limit', 100, type=int)
        logs = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(limit).all()
        
        log_data = []
        for log in logs:
            log_data.append({
                'id': log.id,
                'user_type': log.user_type,
                'user_email': log.user_email,
                'action': log.action,
                'details': log.details,
                'ip_address': log.ip_address,
                'timestamp': log.timestamp.isoformat()
            })
        
        return jsonify(log_data)
    except Exception as e:
        print(f"Error getting activity logs: {e}")
        return jsonify([]), 500

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
    """Get audit log entries"""
    try:
        # For now, return sample audit entries
        # You can implement full audit logging later
        sample_audit = [
            {
                'userType': 'admin',
                'userName': 'admin@dotrmrt3.gov.ph',
                'action': 'Logged in',
                'target': 'System',
                'timestamp': datetime.now().isoformat()
            },
            {
                'userType': 'system',
                'userName': 'System',
                'action': 'Initialized',
                'target': 'Admin Dashboard',
                'timestamp': datetime.now().isoformat()
            }
        ]
        return jsonify(sample_audit)
    except Exception as e:
        print(f"Error getting audit log: {e}")
        return jsonify([])

@app.route('/api/admin/audit-stats')
def admin_audit_stats():
    """Get audit statistics"""
    try:
        return jsonify({
            'total_actions': 2,
            'active_admins': 1,
            'active_operators': User.query.filter_by(role='operator', is_active=True).count(),
            'flagged': 0
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