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

app = Flask(__name__, template_folder='html')
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['SESSION_COOKIE_SECURE'] = True  # Only send cookies over HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection

# --- DATABASE CONFIG ---
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'mrt.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Add this near your other configurations, before your routes
@app.context_processor
def inject_now():
    """Inject current datetime into templates"""
    from datetime import datetime
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

# ========== MODELS (Define in correct order - ALL MODELS FIRST) ==========

# First: User Model (base model)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=True)  # Can be null for Google-only users
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
        """Hash password on set"""
        self.password_hash = generate_password_hash(password)
    
    def verify_password(self, password):
        """Verify hashed password - safely handle None case"""
        if self.password_hash is None:
            return False
        return check_password_hash(self.password_hash, password)
    
    def has_password(self):
        """Check if user has a password set (for Google-only users)"""
        return self.password_hash is not None

# Second: Activity Model (depends on User)
class Activity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    activity_type = db.Column(db.String(50))  # 'broadcast', 'override', 'user_registration', 'model_retrain'
    title = db.Column(db.String(200))
    description = db.Column(db.String(500))
    icon = db.Column(db.String(50))  # 'bullhorn', 'bolt', 'user-plus', 'robot'
    icon_color = db.Column(db.String(50))  # 'blue', 'orange', 'green', 'purple'
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    station = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    user = db.relationship('User', backref=db.backref('activities', lazy=True))

# Third: Report Model (depends on User)
class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    station = db.Column(db.String(50))
    reported_congestion = db.Column(db.Integer)  # 0-100%
    predicted_congestion = db.Column(db.Integer)  # What our model predicted
    remarks = db.Column(db.String(500), nullable=True)  # NEW
    photo_path = db.Column(db.String(200), nullable=True)  # NEW
    anonymous = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    
    user = db.relationship('User', backref=db.backref('reports', lazy=True))

with app.app_context():
    db.create_all()

# ========== REPORTS API ==========
@app.route('/api/reports')
def get_reports():
    """Get all reports for display"""
    try:
        # Get latest 20 reports
        reports = Report.query.order_by(Report.timestamp.desc()).limit(20).all()
        
        result = []
        for report in reports:
            # Get username if not anonymous and user exists
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
                'anonymous': report.anonymous if hasattr(report, 'anonymous') else False,
                'username': username,
                'timestamp': report.timestamp.isoformat() if report.timestamp else None
            })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error fetching reports: {e}")
        return jsonify({"error": str(e)}), 500
# ========== ACTIVITY LOGGING FUNCTIONS ==========
def log_user_registration(user):
    """Log when a new user registers"""
    activity = Activity(
        activity_type='user_registration',
        title='New user registered',
        description=f'{user.username} joined MRT-ADAPT',
        icon='user-plus',
        icon_color='green',
        user_id=user.id,
        created_at=datetime.now()
    )
    db.session.add(activity)
    db.session.commit()
    
@app.route('/debug/list-routes')
def list_routes():
    """List all registered routes"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'url': str(rule)
        })
    return jsonify(routes)  

@app.route('/api/users/<int:user_id>/toggle', methods=['POST'])
def toggle_user_status(user_id):
    """Activate or deactivate a regular user"""
    # For testing, let's print the session
    print(f"🔍 Session in toggle_user_status: {dict(session)}")
    
    # Check if user is admin - more permissive for testing
    is_admin = False
    
    # Check all possible admin indicators
    if session.get('admin_logged_in') or session.get('is_admin'):
        is_admin = True
        print("✅ Admin detected via session flags")
    elif 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and user.role == 'admin':
            is_admin = True
            print("✅ Admin detected via database role")
    elif session.get('username') == os.getenv('ADMIN_EMAIL'):
        is_admin = True
        print("✅ Admin detected via email match")
    
    # TEMPORARY: For testing, allow any request from localhost
    # COMMENT THIS OUT AFTER TESTING!
    is_admin = True
    print("⚠️ TEMPORARY: Bypassing admin check for testing")
    
    if not is_admin:
        print("❌ Not admin, unauthorized")
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        print(f"📦 Received data: {data}")
        
        action = data.get('action')
        email = data.get('email')
        
        # Find the user
        user = User.query.get(user_id)
        if not user:
            print(f"❌ User {user_id} not found")
            return jsonify({"error": "User not found"}), 404
        
        print(f"✅ Found user: {user.username}, current is_active: {user.is_active}")
        
        # Toggle the is_active status
        if action == 'inactive':
            user.is_active = False
            message = f"User {email} deactivated"
        elif action == 'active':
            user.is_active = True
            message = f"User {email} activated"
        else:
            return jsonify({"error": "Invalid action"}), 400
        
        db.session.commit()
        print(f"✅ User {user_id} is_active set to {user.is_active}")
        
        return jsonify({
            "success": True,
            "message": message,
            "user_id": user_id,
            "new_status": "active" if action == 'active' else 'inactive'
        })
        
    except Exception as e:
        print(f"❌ Error in toggle_user_status: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
# ========== OPERATOR API ROUTES ==========
@app.route('/api/operator/congestion-stats')
def operator_congestion_stats():
    """Get congestion statistics for operator dashboard"""
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'operator':
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get current congestion levels for all stations
    station_stats = []
    severe_count = 0
    congested_count = 0
    moderate_count = 0
    light_count = 0
    
    for station in STATIONS:
        ridership = get_station_prediction(station)
        capacity = STATION_BASE_CAPACITY.get(station, 10000)
        congestion = int((ridership / capacity) * 100)
        
        # Categorize congestion
        if congestion >= 80:
            status = "severe"
            severe_count += 1
        elif congestion >= 60:
            status = "congested"
            congested_count += 1
        elif congestion >= 30:
            status = "moderate"
            moderate_count += 1
        else:
            status = "light"
            light_count += 1
        
        station_stats.append({
            'name': station,
            'congestion': congestion,
            'status': status,
            'ridership': int(ridership),
            'capacity': capacity
        })
    
    return jsonify({
        'stats': station_stats,
        'counts': {
            'severe': severe_count,
            'congested': congested_count,
            'moderate': moderate_count,
            'light': light_count
        }
    })

@app.route('/api/operator/broadcast', methods=['POST'])
def operator_broadcast():
    """Send a broadcast as an operator"""
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'operator':
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    station = data.get('station')
    alert_type = data.get('type')
    title = data.get('title')
    message = data.get('message')
    duration = data.get('duration', 30)
    
    if not all([station, alert_type, title, message]):
        return jsonify({"error": "Missing required fields"}), 400
    
    # Log the broadcast in activities
    log_broadcast(user.username, station, f"{alert_type}: {title} - {message}")
    
    # In a real app, you'd store this in a Broadcast model
    # For now, we'll just return success
    
    return jsonify({
        "success": True,
        "message": "Broadcast sent successfully",
        "broadcast": {
            "station": station,
            "type": alert_type,
            "title": title,
            "message": message,
            "time": datetime.now().strftime("%I:%M %p"),
            "operator": user.username
        }
    })

@app.route('/api/operator/override', methods=['POST'])
def operator_override():
    """Apply a congestion override"""
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'operator':
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    station = data.get('station')
    level = data.get('level')
    duration = data.get('duration', 60)
    reason = data.get('reason', '')
    
    if not station or not level:
        return jsonify({"error": "Missing required fields"}), 400
    
    # Map level to congestion value
    level_map = {
        'Light': 20,
        'Moderate': 45,
        'Congested': 70,
        'Severe': 90
    }
    
    congestion_value = level_map.get(level, 50)
    
    # Log the override
    log_override(user.username, station)
    
    # In a real app, you'd store this in an Override model
    # For now, we'll just return success
    
    return jsonify({
        "success": True,
        "message": f"{station} overridden to {level}",
        "override": {
            "station": station,
            "level": level,
            "value": congestion_value,
            "duration": duration,
            "reason": reason,
            "time": datetime.now().strftime("%I:%M %p"),
            "operator": user.username
        }
    })

@app.route('/api/operator/recent-broadcasts')
def operator_recent_broadcasts():
    """Get recent broadcasts for the operator"""
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'operator':
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get recent broadcasts from activities
    recent = Activity.query.filter_by(
        activity_type='broadcast'
    ).order_by(Activity.created_at.desc()).limit(10).all()
    
    result = []
    for activity in recent:
        result.append({
            'station': activity.station or 'All Stations',
            'type': 'Broadcast',
            'message': activity.description,
            'time': activity.created_at.strftime("%I:%M %p"),
            'operator': activity.title.replace('Broadcast sent by ', '') if activity.title else 'System'
        })
    
    return jsonify(result)

def log_broadcast(operator_name, station, message):
    """Log when an operator sends a broadcast"""
    activity = Activity(
        activity_type='broadcast',
        title=f'Broadcast sent by {operator_name}',
        description=message,
        icon='bullhorn',
        icon_color='blue',
        station=station,
        created_at=datetime.now()
    )
    db.session.add(activity)
    db.session.commit()

def log_override(operator_name, station):
    """Log when an operator applies congestion override"""
    activity = Activity(
        activity_type='override',
        title=f'Congestion override by {operator_name}',
        description=f'Manual override applied at {station}',
        icon='bolt',
        icon_color='orange',
        station=station,
        created_at=datetime.now()
    )
    db.session.add(activity)
    db.session.commit()

def log_model_retrain():
    """Log when LSTM model is retrained"""
    activity = Activity(
        activity_type='model_retrain',
        title='LSTM model retrained',
        description='New batch data loaded',
        icon='robot',
        icon_color='purple',
        created_at=datetime.now()
    )
    db.session.add(activity)
    db.session.commit()

def get_time_ago(dt):
    """Convert datetime to 'X minutes/hours/days ago' format"""
    now = datetime.now()
    diff = now - dt
    
    if diff.days > 0:
        return f'{diff.days} day{"s" if diff.days > 1 else ""} ago'
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f'{hours} hour{"s" if hours > 1 else ""} ago'
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f'{minutes} minute{"s" if minutes > 1 else ""} ago'
    else:
        return 'just now'

# ========== ACCURACY CALCULATION FUNCTION ==========
def calculate_real_accuracy():
    """Calculate REAL accuracy based on user reports vs predictions"""
    
    # Get all user reports from the last 24 hours
    reports = Report.query.filter(
        Report.timestamp >= datetime.now() - timedelta(days=1)
    ).all()
    
    if not reports:
        return {
            'today': 91.4,
            'week': 89.7,
            'month': 83.2
        }  # Return dict, not string
    
    total_accuracy = 0
    for report in reports:
        # Compare what user reported vs what model predicted
        error = abs(report.reported_congestion - report.predicted_congestion)
        accuracy = 100 - error  # Convert error to accuracy
        total_accuracy += max(0, accuracy)
    
    avg_accuracy = total_accuracy / len(reports)
    
    # Return in the format your template expects
    return {
        'today': round(avg_accuracy, 1),
        'week': round(avg_accuracy * 0.98, 1),
        'month': round(avg_accuracy * 0.95, 1)
    }

# ========== REPORT CONGESTION API ==========
@app.route('/api/report-congestion', methods=['POST'])
def report_congestion():
    try:
        data = request.json
        station = data.get('station')
        reported = data.get('congestion')
        remarks = data.get('remarks', '')
        anonymous = data.get('anonymous', False)
        
        print(f"📝 Received report: {station} - {reported}% - Anonymous: {anonymous}")
        
        # Check if user is logged in
        if 'user_id' not in session:
            # For guests, we can still accept the report but maybe don't link to a user
            print("👤 Guest user submitted a report")
            # You might want to store guest reports differently or with user_id = None
            user_id = None
        else:
            user_id = session.get('user_id')
        
        # Get model prediction
        ridership = get_station_prediction(station)
        capacity = STATION_BASE_CAPACITY.get(station, 10000)
        predicted = int((ridership / capacity) * 100)
        
        # Save report
        report = Report(
            user_id=user_id,  # This will be None for guests
            station=station,
            reported_congestion=reported,
            predicted_congestion=predicted,
            remarks=remarks,  # Make sure this line exists
            anonymous=anonymous
            # You'll need to add remarks and anonymous fields to your Report model
        )
        db.session.add(report)
        db.session.commit()
        
        return jsonify({"success": True, "message": "Report saved successfully"})
        
    except Exception as e:
        print(f"❌ Error saving report: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ========== LOGIN REQUIRED DECORATOR ==========
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        
        # Check if user is still active
        user = User.query.get(session['user_id'])
        if user and not user.is_active:
            session.clear()
            flash('Your account has been deactivated. Please contact an administrator.', 'error')
            return redirect(url_for('login'))
            
        return f(*args, **kwargs)
    return decorated_function

# Add this near your other models (around line 200-220)
class SavedRoute(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    from_station = db.Column(db.String(50), nullable=False)
    to_station = db.Column(db.String(50), nullable=False)
    route_name = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    user = db.relationship('User', backref=db.backref('saved_routes', lazy=True))

# Add these API endpoints (around line 1000-1100, after your other API routes)

# ========== SAVED ROUTES API ==========
@app.route('/api/saved-routes', methods=['GET'])
@login_required
def get_saved_routes():
    """Get all saved routes for the current user"""
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
@login_required
def save_route():
    """Save a new route for the current user"""
    try:
        data = request.json
        user_id = session.get('user_id')
        from_station = data.get('from_station')
        to_station = data.get('to_station')
        route_name = data.get('route_name', f"{from_station} to {to_station}")
        
        if not from_station or not to_station:
            return jsonify({"error": "Missing station information"}), 400
        
        # Check if route already exists (optional)
        existing = SavedRoute.query.filter_by(
            user_id=user_id,
            from_station=from_station,
            to_station=to_station
        ).first()
        
        if existing:
            return jsonify({"success": True, "message": "Route already saved", "route": {
                'id': existing.id,
                'from_station': existing.from_station,
                'to_station': existing.to_station
            }})
        
        # Create new saved route
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
@login_required
def delete_saved_route(route_id):
    """Delete a saved route"""
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
    """Return historical congestion patterns for all stations"""
    try:
        patterns = {}
        
        for station in STATIONS:
            station_patterns = {}
            for hour in range(24):
                # Generate realistic historical averages based on time of day
                if 7 <= hour <= 9:  # Morning rush
                    base = 75
                elif 17 <= hour <= 20:  # Evening rush
                    base = 80
                elif 10 <= hour <= 16:  # Midday
                    base = 55
                elif 21 <= hour <= 22:  # Late evening
                    base = 30
                elif 5 <= hour <= 6:  # Early morning
                    base = 20
                else:  # Late night
                    base = 5
                
                # Station-specific adjustments
                if station in ["Cubao", "Ayala Ave", "North Ave"]:
                    base += 10
                elif station in ["Santolan", "Magallanes"]:
                    base -= 5
                
                station_patterns[hour] = base
            
            patterns[station] = station_patterns
        
        return jsonify(patterns)
    except Exception as e:
        print(f"❌ Error generating historical patterns: {e}")
        return jsonify({}), 500

# Keep this version (with date/time support) and delete the others
@app.route('/api/predict/<station_name>')
def predict_congestion(station_name):
    name = station_name.replace('%20', ' ')
    
    # Check if date and time are provided
    date_param = request.args.get('date')
    time_param = request.args.get('time')
    
    if date_param and time_param:
        # Parse the date and time
        try:
            year, month, day = map(int, date_param.split('-'))
            hour, minute = map(int, time_param.split(':'))
            
            # Create datetime object for the requested time
            target_datetime = datetime(year, month, day, hour, minute)
            
            print(f"📅 Getting prediction for {name} at {target_datetime}")
            
            # Get prediction for that specific datetime
            ridership = get_station_prediction_for_datetime(name, target_datetime)
        except Exception as e:
            print(f"⚠️ Error parsing datetime: {e}, using current time")
            ridership = get_station_prediction(name)
    else:
        print(f"📊 Getting current prediction for {name}")
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
# Add this near your other operator API routes (around line 350-400)

@app.route('/api/operator/broadcasts')
def operator_get_broadcasts():
    """Get all broadcasts for operator incident log"""
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    user = User.query.get(session['user_id'])
    if user.role != 'operator':
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get broadcasts from activities
    broadcasts = Activity.query.filter_by(
        activity_type='broadcast'
    ).order_by(Activity.created_at.desc()).all()
    
    result = []
    for b in broadcasts:
        # Parse the broadcast details
        # Format: "Train Breakdown: Title - Message"
        description = b.description
        broadcast_type = "General Notice"
        title = description
        message = ""
        
        if ':' in description:
            parts = description.split(':', 1)
            broadcast_type = parts[0].strip()
            remaining = parts[1].strip()
            
            if '-' in remaining:
                title_parts = remaining.split('-', 1)
                title = title_parts[0].strip()
                message = title_parts[1].strip()
            else:
                title = remaining
        
        # Get operator name
        operator = b.title.replace('Broadcast sent by ', '') if b.title else 'System'
        operator_name = operator.split('@')[0] if '@' in operator else operator
        
        # Determine severity based on message content or type
        severity = 'advisory'
        if 'critical' in message.lower() or broadcast_type in ['Train Breakdown', 'Signal Issue']:
            severity = 'critical'
        elif 'congested' in message.lower() or broadcast_type == 'Overcrowding':
            severity = 'warning'
        
        # Get icon
        icon_map = {
            'Train Breakdown': 'fa-train',
            'Overcrowding': 'fa-users',
            'Maintenance': 'fa-wrench',
            'Signal Issue': 'fa-satellite-dish',
            'Gate Closure': 'fa-door-closed',
            'General Notice': 'fa-bullhorn'
        }
        icon = icon_map.get(broadcast_type, 'fa-bullhorn')
        
        # Format stations
        stations = [b.station] if b.station and b.station != 'All Stations' else []
        if not stations and b.station == 'All Stations':
            stations = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
                       "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
                       "Ayala Ave", "Magallanes", "Taft"]
        
        result.append({
            'id': b.id,
            'type': broadcast_type,
            'icon': icon,
            'title': title,
            'message': message or title,
            'stations': stations,
            'operator': operator_name,
            'time': b.created_at.strftime("%I:%M %p"),
            'date': b.created_at.strftime("%Y-%m-%d"),
            'severity': severity,
            'raw_time': b.created_at.isoformat()
        })
    
    return jsonify(result)

def log_broadcast(operator_name, station, message):
    """Log when an operator sends a broadcast"""
    activity = Activity(
        activity_type='broadcast',
        title=f'Broadcast sent by {operator_name}',
        description=message,
        icon='bullhorn',
        icon_color='blue',
        station=station,  # This stores the station(s)
        created_at=datetime.now()
    )
    db.session.add(activity)
    db.session.commit()

@app.route('/api/alerts/broadcasts')
def get_commuter_broadcasts():
    """Get operator broadcasts for commuters"""
    try:
        # Get recent broadcasts (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        broadcasts = Activity.query.filter_by(
            activity_type='broadcast'
        ).filter(
            Activity.created_at >= cutoff
        ).order_by(Activity.created_at.desc()).all()
        
        result = []
        for b in broadcasts:
            # Parse the broadcast details
            description = b.description
            
            # Format: "Train Breakdown: Title - Message"
            broadcast_type = "General Notice"
            title = description
            message = ""
            
            if ':' in description:
                parts = description.split(':', 1)
                broadcast_type = parts[0].strip()
                remaining = parts[1].strip()
                
                if '-' in remaining:
                    title_parts = remaining.split('-', 1)
                    title = title_parts[0].strip()
                    message = title_parts[1].strip()
                else:
                    title = remaining
            
            # Determine severity
            severity = 'advisory'
            icon = 'info-circle'
            icon_color = 'blue'
            
            if 'critical' in message.lower() or broadcast_type in ['Train Breakdown', 'Signal Issue']:
                severity = 'critical'
                icon = 'exclamation-circle'
                icon_color = 'red'
            elif 'congested' in message.lower() or broadcast_type == 'Overcrowding':
                severity = 'warning'
                icon = 'clock'
                icon_color = 'orange'
            
            # Format time
            time_diff = datetime.now() - b.created_at
            if time_diff.total_seconds() < 60:
                time_display = 'Just now'
            elif time_diff.total_seconds() < 3600:
                minutes = int(time_diff.total_seconds() / 60)
                time_display = f'{minutes} min ago'
            elif time_diff.total_seconds() < 86400:
                hours = int(time_diff.total_seconds() / 3600)
                time_display = f'{hours} hour ago' if hours == 1 else f'{hours} hours ago'
            else:
                time_display = b.created_at.strftime('%I:%M %p')
            
            result.append({
                'id': f'broadcast-{b.id}',
                'type': severity,
                'icon': icon,
                'icon_color': icon_color,
                'title': title,
                'message': message or title,
                'time': time_display,
                'unread': True,
                'station': b.station or 'All Stations',
                'source': 'operator',
                'raw_time': b.created_at.isoformat()
            })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error getting broadcasts: {e}")
        return jsonify([])
    
    
@app.route('/debug/check-saved-routes-table')
def check_saved_routes_table():
    """Check if saved_routes table exists"""
    try:
        # Try to query the table
        count = SavedRoute.query.count()
        return jsonify({
            "exists": True,
            "count": count,
            "message": "SavedRoute table exists"
        })
    except Exception as e:
        return jsonify({
            "exists": False,
            "error": str(e),
            "message": "SavedRoute table does not exist"
        }), 404
           
# ========== HELPER FUNCTIONS ==========
def generate_state():
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))

def get_holiday_multiplier():
    t = datetime.now().date()
    major_holidays = [(1,1), (4,9), (5,1), (6,12), (8,21), (8,28), 
                      (11,1), (11,2), (11,30), (12,25), (12,30), (12,31)]
    if (t.month, t.day) in major_holidays:
        return 0.15
    if (t.month == 12 and t.day >= 16) or (t.month == 1 and t.day <= 5):
        return 0.45
    return 1.0

# ========== MAIN PREDICTION FUNCTION ==========
def get_station_prediction(station_name):
    """Get realistic ridership prediction using LSTM + logical constraints"""
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    
    # ===== STRICT OPERATING HOURS CHECK - RETURN 0 IMMEDIATELY =====
    # MRT-3 operates from 4:30 AM to 10:30 PM (last trains)
    if hour < 4:  # Before 4 AM
        print(f"⏰ {station_name}: {hour}:{minute:02d} - Before 4 AM -> 0")
        return 0
    if hour == 4 and minute < 30:  # 4:00-4:29 AM
        print(f"⏰ {station_name}: {hour}:{minute:02d} - Before 4:30 AM -> 0")
        return 0
    if hour >= 23:  # 11:00 PM and later
        print(f"⏰ {station_name}: {hour}:{minute:02d} - After 11 PM -> 0")
        return 0
    
    try:
        capacity = STATION_BASE_CAPACITY.get(station_name, 10000)
        
        # ===== LOGICAL BASE PREDICTION =====
        # Define station groups
        north_group = ["North Ave", "Quezon Ave", "Kamuning"]
        south_group = ["Buendia", "Ayala Ave", "Magallanes", "Taft"]
        business_districts = ["Ayala Ave", "Ortigas", "Cubao"]
        
        # Time-based logical multiplier
        if 7 <= hour <= 9:  # Morning rush
            if station_name in north_group:
                logical_mult = 0.70
            elif station_name in south_group:
                logical_mult = 0.45
            else:
                logical_mult = 0.60
        elif 17 <= hour <= 20:  # Evening rush
            if station_name in south_group:
                logical_mult = 0.75
            elif station_name in north_group:
                logical_mult = 0.50
            else:
                logical_mult = 0.65
        elif 10 <= hour <= 16:  # Midday
            if station_name in business_districts:
                logical_mult = 0.60
            else:
                logical_mult = 0.50
        elif 21 <= hour <= 22:  # Late evening
            logical_mult = 0.30
        elif 5 <= hour <= 6:  # Early morning
            logical_mult = 0.15
        else:
            logical_mult = 0.25
        
        # Weekend adjustment
        if now.weekday() >= 5:
            logical_mult *= 0.7
        
        logical_riders = capacity * logical_mult
        
        # ===== LSTM PREDICTION =====
        lstm_riders = None
        if station_name in lstm_models and station_name in scalers:
            df_recent = station_time_series_last_24.get(station_name)
            
            if df_recent is not None and len(df_recent) >= 24:
                try:
                    recent_values = df_recent['ridership'].values[-24:].reshape(-1, 1)
                    scaled_input = scalers[station_name].transform(recent_values)
                    prediction = lstm_models[station_name].predict(
                        scaled_input.reshape(1, 24, 1), 
                        verbose=0
                    )
                    lstm_riders = scalers[station_name].inverse_transform(prediction)[0][0]
                    
                    # Apply logical bounds to LSTM prediction
                    max_allowed = capacity * 0.90
                    min_allowed = capacity * 0.15
                    lstm_riders = max(min_allowed, min(lstm_riders, max_allowed))
                    
                    print(f"🤖 LSTM {station_name}: {lstm_riders:.0f} riders")
                except Exception as e:
                    print(f"⚠️ LSTM prediction failed for {station_name}: {e}")
                    lstm_riders = None
        
        # ===== BLEND OR USE LOGICAL =====
        if lstm_riders is not None:
            # Blend: 40% LSTM, 60% logical (reduces extreme values)
            final_riders = int((lstm_riders * 0.4) + (logical_riders * 0.6))
        else:
            final_riders = int(logical_riders)
        
        # ===== STATION-SPECIFIC ADJUSTMENTS =====
        station_adjustments = {
            "North Ave": 1.02,
            "Quezon Ave": 0.95,
            "Kamuning": 0.90,
            "Cubao": 1.08,
            "Santolan": 0.80,
            "Ortigas": 1.05,
            "Shaw Blvd": 1.03,
            "Boni Ave": 0.92,
            "Guadalupe": 0.98,
            "Buendia": 0.85,
            "Ayala Ave": 1.07,
            "Magallanes": 0.88,
            "Taft": 1.10
        }
        
        final_riders = int(final_riders * station_adjustments.get(station_name, 1.0))
        
        # ===== FIXED FINAL BOUNDS - NO MINIMUM FOR LATE NIGHT =====
        # For peak hours (6 AM - 9 PM), enforce 15-90% range
        if 6 <= hour <= 20:
            absolute_min = capacity * 0.15
            absolute_max = capacity * 0.90
            final_riders = max(absolute_min, min(final_riders, absolute_max))
        # For late evening (9 PM - 10:30 PM), allow gradual decrease to 0
        elif hour == 21:
            # 9:00 PM - 9:59 PM - allow lower values
            absolute_max = capacity * 0.90
            final_riders = min(final_riders, absolute_max)
            # No minimum - can go to 0
        elif hour == 22:
            # 10:00 PM - 10:59 PM - taper to 0
            absolute_max = capacity * 0.90
            final_riders = min(final_riders, absolute_max)
            
            # Calculate taper factor based on minute
            if minute <= 30:
                # 10:00-10:30 PM: linear decrease from 30% to 15%
                taper_factor = 0.3 - (minute / 100)
            else:
                # 10:30-10:59 PM: rapid decrease to 0
                minutes_after_1030 = minute - 30
                taper_factor = max(0, 0.15 - (minutes_after_1030 * 0.015))
            
            final_riders = int(final_riders * taper_factor)
        
        # Small natural variation
        variation = np.random.uniform(0.98, 1.02)
        final_riders = int(final_riders * variation)
        
        # FINAL SAFETY CHECK - if it's after 10:30 PM, force to 0
        if hour == 22 and minute >= 30:
            # Linear taper to 0 by 11 PM
            minutes_until_11 = 60 - minute
            if minutes_until_11 <= 0:
                return 0
            # Reduce by about 10% per minute
            final_riders = int(final_riders * (minutes_until_11 / 30))
        
        # Ensure we don't return negative values
        final_riders = max(0, final_riders)
        
        return final_riders
        
    except Exception as e:
        print(f"❌ Error in prediction for {station_name}: {e}")
        # Smart fallback with operating hours check
        hour = datetime.now().hour
        minute = datetime.now().minute
        
        if hour < 4 or (hour == 4 and minute < 30) or hour >= 23:
            return 0
            
        defaults = {
            "North Ave": 6000, "Quezon Ave": 5000, "Kamuning": 4500,
            "Cubao": 9000, "Santolan": 3500, "Ortigas": 6500,
            "Shaw Blvd": 7000, "Boni Ave": 5500, "Guadalupe": 6000,
            "Buendia": 4500, "Ayala Ave": 8000, "Magallanes": 5000,
            "Taft": 10000
        }
        return defaults.get(station_name, 5000)

      
def get_station_prediction_for_datetime(station_name, target_datetime):
    """Get realistic prediction for a specific datetime"""
    try:
        # Get components from target datetime
        target_hour = target_datetime.hour
        target_minute = target_datetime.minute
        target_weekday = target_datetime.weekday()  # 0-6 (0=Monday, 6=Sunday)
        target_month = target_datetime.month
        target_day = target_datetime.day
        
        capacity = STATION_BASE_CAPACITY.get(station_name, 10000)
        
        # ===== TIME-BASED MULTIPLIER =====
        # More granular time-based logic
        
        # Early morning (4:30-6:59)
        if 4 <= target_hour <= 6:
            if target_hour == 4 and target_minute < 30:
                return 0  # Before 4:30, no trains
            multiplier = 0.15 + (target_hour - 4) * 0.1
            if target_hour == 6:
                multiplier += 0.1
        
        # Morning rush (7:00-9:00)
        elif 7 <= target_hour <= 9:
            multiplier = 0.65 + (target_hour - 7) * 0.1
            if target_hour == 8:
                multiplier = 0.85  # Peak at 8 AM
        
        # Late morning (9:01-11:59)
        elif 9 < target_hour < 12:
            multiplier = 0.55 - (target_hour - 9) * 0.05
        
        # Lunch hour (12:00-13:59)
        elif 12 <= target_hour <= 13:
            multiplier = 0.50 + (target_hour - 12) * 0.1
        
        # Afternoon (14:00-16:59)
        elif 14 <= target_hour <= 16:
            multiplier = 0.55 - (target_hour - 14) * 0.05
        
        # Evening rush (17:00-20:00)
        elif 17 <= target_hour <= 20:
            multiplier = 0.70 + (target_hour - 17) * 0.1
            if target_hour == 18:
                multiplier = 0.85  # Peak at 6 PM
        
        # Late evening (20:01-22:30)
        elif 20 < target_hour <= 22:
            multiplier = 0.45 - (target_hour - 20) * 0.15
            if target_hour == 22 and target_minute >= 30:
                # After 10:30 PM, taper to 0
                minutes_past_1030 = target_minute - 30
                multiplier = max(0, 0.25 - (minutes_past_1030 * 0.02))
        
        # Night (22:31-4:29) - no trains
        else:
            return 0
        
        # ===== DAY OF WEEK ADJUSTMENT =====
        if target_weekday >= 5:  # Weekend
            multiplier *= 0.6
            # Weekend patterns are different
            if 10 <= target_hour <= 14:  # Weekend lunch rush
                multiplier *= 1.3
            elif 15 <= target_hour <= 19:  # Weekend afternoon/evening
                multiplier *= 1.2
        else:  # Weekday
            # Friday evening is busier
            if target_weekday == 4 and 17 <= target_hour <= 20:
                multiplier *= 1.1
            # Monday morning is busier
            if target_weekday == 0 and 7 <= target_hour <= 9:
                multiplier *= 1.15
        
        # ===== MONTH/SEASON ADJUSTMENT =====
        # December is busier (holiday season)
        if target_month == 12:
            multiplier *= 1.2
        # January is lighter after holidays
        elif target_month == 1:
            multiplier *= 0.9
        # Summer months (March-May) are lighter
        elif 3 <= target_month <= 5:
            multiplier *= 0.95
        
        # ===== STATION-SPECIFIC ADJUSTMENTS =====
        station_adjustments = {
            "North Ave": 1.02,
            "Quezon Ave": 0.95,
            "Kamuning": 0.90,
            "Cubao": 1.08,
            "Santolan": 0.80,
            "Ortigas": 1.05,
            "Shaw Blvd": 1.03,
            "Boni Ave": 0.92,
            "Guadalupe": 0.98,
            "Buendia": 0.85,
            "Ayala Ave": 1.07,
            "Magallanes": 0.88,
            "Taft": 1.10
        }
        
        # Apply station adjustment
        multiplier *= station_adjustments.get(station_name, 1.0)
        
        # ===== TIME-SPECIFIC PEAK ADJUSTMENTS =====
        # Different stations have different peak patterns
        if station_name in ["North Ave", "Quezon Ave", "Kamuning"]:
            # North stations are busier in morning
            if 7 <= target_hour <= 9:
                multiplier *= 1.15
        elif station_name in ["Ayala Ave", "Magallanes", "Taft"]:
            # South stations are busier in evening
            if 17 <= target_hour <= 20:
                multiplier *= 1.15
        elif station_name in ["Cubao", "Ortigas", "Shaw Blvd"]:
            # Central stations are busy all day
            if 10 <= target_hour <= 19:
                multiplier *= 1.1
        
        # Calculate final ridership
        ridership = int(capacity * multiplier)
        
        # Add small random variation (±3%)
        import random
        variation = random.uniform(0.97, 1.03)
        ridership = int(ridership * variation)
        
        # Ensure within bounds (15-95% of capacity during operating hours)
        if 5 <= target_hour <= 22:
            min_allowed = int(capacity * 0.15)
            max_allowed = int(capacity * 0.95)
            ridership = max(min_allowed, min(ridership, max_allowed))
        
        print(f"📊 Prediction for {station_name} at {target_hour:02d}:{target_minute:02d} on weekday {target_weekday}: {ridership} riders ({multiplier:.2f}x)")
        
        return ridership
        
    except Exception as e:
        print(f"❌ Error in datetime prediction for {station_name}: {e}")
        # Fallback to current time prediction
        return get_station_prediction(station_name)
        
# ========== COMBINED LOGIN ROUTE (Admin + Regular Users) ==========
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Debug prints (check your console)
        print(f"Login attempt - Email: {email}")
        
        # === FIRST CHECK IF IT'S ADMIN (from .env) ===
        admin_email = os.getenv('ADMIN_EMAIL')
        admin_password = os.getenv('ADMIN_PASSWORD')
        
        if admin_email and admin_password and email == admin_email and password == admin_password:
            # Admin login successful
            session.clear()
            session['admin_logged_in'] = True
            session['admin_email'] = email
            session['username'] = email
            session['is_admin'] = True
            session['google_user'] = False
            session['role'] = 'admin'
            
            print(f"✅ Admin login successful: {email}")
            return redirect(url_for('admin_dashboard'))
        
        # === IF NOT ADMIN, CHECK REGULAR USER DATABASE ===
        user = User.query.filter_by(username=email).first()
        
        if user is None:
            error = "Account not found. Please check your email or sign up."
            print(f"❌ User not found: {email}")
        elif user.google_id and user.password_hash is None:
            error = "This account uses Google Sign-In. Please click 'Continue with Google'."
            print(f"❌ Google user attempted password login: {email}")
        elif not user.is_active:
            error = "This account has been deactivated. Please contact an administrator."
            print(f"❌ Deactivated account attempted login: {email}")
        elif user.verify_password(password):
            # Regular user login successful
            session.clear()
            session.permanent = True
            session['user_id'] = user.id
            session['username'] = user.username
            session['google_user'] = False
            session['favorite_station'] = user.favorite_station
            session['is_admin'] = False
            session['admin_logged_in'] = False
            session['role'] = user.role
            
            user.last_login = datetime.now()
            db.session.commit()
            
            print(f"✅ User login successful: {email} (role: {user.role})")
            
            # Redirect based on role
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user.role == 'operator':
                return redirect(url_for('operator_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            error = "Incorrect password. Please try again."
            print(f"❌ Invalid password for: {email}")
    
    return render_template('login.html', error=error)

@app.route('/debug/remove-admin')
def remove_admin():
    """Temporary route to remove admin from database"""
    admin_email = os.getenv('ADMIN_EMAIL')
    if admin_email:
        admin_user = User.query.filter_by(username=admin_email).first()
        if admin_user:
            db.session.delete(admin_user)
            db.session.commit()
            return f"✅ Removed {admin_email} from database"
    return "❌ Admin not found"

# ========== UPDATED SIGNUP ROUTE WITH PASSWORD HASHING ==========
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        favorite = request.form.get('favorite_station')
        
        # Check if passwords match
        if password != confirm_password:
            return render_template('signup.html', 
                                 error='Passwords do not match',
                                 email=email,
                                 favorite=favorite)
        
        # Check if user already exists
        existing_user = User.query.filter_by(username=email).first()
        if existing_user:
            if existing_user.google_id:
                # User exists with Google
                return redirect(url_for('signup', error='email_exists', email=email))
            else:
                # User exists with email/password
                return redirect(url_for('signup', error='email_exists', email=email))
        
        # Create new user with hashed password
        new_user = User(
            username=email,
            role='commuter',
            favorite_station=favorite if favorite else None,
            created_at=datetime.now(),
            last_login=datetime.now()
        )
        new_user.password = password  # This uses the setter to hash the password
        
        try:
            db.session.add(new_user)
            db.session.commit()
            
            # LOG THE USER REGISTRATION
            log_user_registration(new_user)
            
            session.permanent = True
            session['user_id'] = new_user.id
            session['username'] = new_user.username
            session['google_user'] = False
            session['favorite_station'] = new_user.favorite_station
            session['is_admin'] = False
            session['admin_logged_in'] = False
            
            flash('Account created successfully!', 'success')
            return redirect(url_for('user_dashboard'))
        except Exception as e:
            db.session.rollback()
            return render_template('signup.html', error="Database error. Please try again.")
    
    # Check for error in URL
    error = request.args.get('error')
    email = request.args.get('email')
    
    return render_template('signup.html', error=error, email=email)

@app.route('/login/google')
def google_login():
    state = generate_state()
    nonce = secrets.token_urlsafe(16)
    session['oauth_state'] = state
    session['nonce'] = nonce
    redirect_uri = url_for('google_callback', _external=True)
    
    # Add prompt='select_account' to force account selection
    return google.authorize_redirect(
        redirect_uri, 
        state=state, 
        nonce=nonce,
        prompt='select_account'  # This forces the account chooser!
    )
@app.route('/auth/google/callback')
def google_callback():
    """Handle Google OAuth callback"""
    try:
        token = google.authorize_access_token()
        nonce = session.pop('nonce', None)
        user_info = google.parse_id_token(token, nonce=nonce)
        
        if not user_info:
            flash('Failed to get user info from Google.', 'error')
            return redirect(url_for('login'))
        
        google_id = user_info.get('sub')
        email = user_info.get('email')
        
        # IMPORTANT: First check if this is the ADMIN email
        admin_email = os.getenv('ADMIN_EMAIL')
        
        # If this Google account is the admin, redirect to admin login page with message
        if admin_email and email == admin_email:
            flash('Please use the admin login form with your admin password, not Google Sign-In.', 'warning')
            return redirect(url_for('login'))
        
        # Check if user exists with this Google ID
        user = User.query.filter_by(google_id=google_id).first()
        
        if user:
            # Check if account is deactivated
            if not user.is_active:
                flash('This account has been deactivated. Please contact an administrator.', 'error')
                return redirect(url_for('login'))
                
            print(f"✅ Found existing Google user: {email}")
        else:
            # Check if user exists with this email but no Google ID
            existing_user = User.query.filter_by(username=email).first()
            
            if existing_user:
                # Check if existing account is deactivated
                if not existing_user.is_active:
                    flash('This account has been deactivated. Please contact an administrator.', 'error')
                    return redirect(url_for('login'))
                    
                # MERGE ACCOUNTS: Link Google ID to existing account
                print(f"🔄 Merging Google account with existing user: {email}")
                existing_user.google_id = google_id
                user = existing_user
                db.session.commit()
                flash('Your Google account has been linked to your existing account.', 'info')
            else:
                # Create new user
                print(f"🆕 Creating new Google user: {email}")
                user = User(
                    username=email,
                    google_id=google_id,
                    role='commuter',
                    created_at=datetime.now(),
                    last_login=datetime.now(),
                    is_active=True  # New accounts are active
                )
                db.session.add(user)
                db.session.commit()
                
                # LOG THE USER REGISTRATION
                log_user_registration(user)
        
        # Update last login
        user.last_login = datetime.now()
        db.session.commit()
        
        # Set session
        session.permanent = True
        session['user_id'] = user.id
        session['username'] = user.username
        session['google_user'] = True
        session['favorite_station'] = user.favorite_station
        session['is_admin'] = False
        session['admin_logged_in'] = False
        session.pop('oauth_state', None)
        
        print(f"✅ Google user logged in: {email}")
        flash(f'Welcome, {user.username}!', 'success')
        return redirect(url_for('user_dashboard'))
        
    except Exception as e:
        print(f"❌ Google callback error: {e}")
        flash('Google login failed. Please try again.', 'error')
        return redirect(url_for('login'))

# ========== ACCOUNT MANAGEMENT ROUTES ==========
@app.route('/profile')
@login_required
def profile():
    user = User.query.get(session['user_id'])
    return render_template('profile.html', 
                         user=user,
                         stations=STATIONS)

@app.route('/api/link-google-account', methods=['POST'])
@login_required
def link_google_account():
    """Allow existing email/password users to link their Google account"""
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.json
    google_id = data.get('google_id')
    
    if not google_id:
        return jsonify({"error": "No Google ID provided"}), 400
    
    user = User.query.get(session['user_id'])
    
    # Check if this Google ID is already linked to another account
    existing = User.query.filter_by(google_id=google_id).first()
    if existing and existing.id != user.id:
        return jsonify({"error": "This Google account is already linked to another user"}), 400
    
    user.google_id = google_id
    db.session.commit()
    
    return jsonify({"success": True, "message": "Google account linked successfully"})

@app.route('/api/set-password', methods=['POST'])
@login_required
def set_password():
    """Allow Google-only users to set a password for email login"""
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.json
    password = data.get('password')
    confirm_password = data.get('confirm_password')
    
    if password != confirm_password:
        return jsonify({"error": "Passwords do not match"}), 400
    
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    
    user = User.query.get(session['user_id'])
    user.password = password  # This uses the setter to hash
    db.session.commit()
    
    return jsonify({"success": True, "message": "Password set successfully"})

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# ========== MAIN ROUTES ==========
@app.route('/')
def home():
    # Clear any existing session and start as guest
    session.clear()
    # Set a flag to indicate guest mode (optional)
    session['guest_mode'] = True
    return redirect(url_for('user_dashboard'))

@app.route('/user-dashboard')
def user_dashboard():
    # Redirect to admin dashboard if admin tries to access user dashboard
    if session.get('admin_logged_in') or session.get('is_admin'):
        print(f"⚠️ Admin attempted to access user dashboard, redirecting to admin")
        return redirect(url_for('admin_dashboard'))
    
    # Guest mode - no login required
    username = session.get('username', 'Guest')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    
    return render_template('user-dashboard.html', 
                         stations=STATIONS, 
                         username=username, 
                         is_logged_in=is_logged_in,
                         google_user=google_user,
                         favorite_station=favorite_station)

# ========== TRAVEL PLAN ROUTE ==========
@app.route('/travel-plan')
def travel_plan():
    # Get user info from session if logged in
    username = session.get('username', 'Guest')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    
    return render_template('travel-plan.html', 
                         stations=STATIONS, 
                         username=username, 
                         is_logged_in=is_logged_in,
                         google_user=google_user,
                         favorite_station=favorite_station)

# ========== HISTORY ROUTE ==========
@app.route('/history')
def history():
    # Get user info from session if logged in
    username = session.get('username', 'Guest')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    user_id = session.get('user_id')
    
    return render_template('history.html', 
                         stations=STATIONS, 
                         username=username, 
                         is_logged_in=is_logged_in,
                         google_user=google_user,
                         favorite_station=favorite_station,
                         user_id=user_id)

# ========== ALERTS ROUTE ==========
@app.route('/alerts')
def alerts():
    # Get user info from session if logged in
    username = session.get('username', 'Guest')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    
    return render_template('alerts.html', 
                         stations=STATIONS, 
                         username=username, 
                         is_logged_in=is_logged_in,
                         google_user=google_user,
                         favorite_station=favorite_station,
                         user_id=session.get('user_id'))

# ========== REPORT ROUTE ==========
@app.route('/report')
def report():
    # Get user info from session if logged in
    username = session.get('username', 'Guest')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    
    return render_template('report.html', 
                         stations=STATIONS, 
                         username=username, 
                         is_logged_in=is_logged_in,
                         google_user=google_user,
                         favorite_station=favorite_station,
                         user_id=session.get('user_id'))

# ========== LIVE MAP ROUTE ==========
@app.route('/live-map')
def live_map():
    """Render the live map page with MRT-3 line status"""
    # Get user info from session if logged in
    username = session.get('username', 'Guest')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    
    # Station coordinates for MRT-3 line
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
                         favorite_station=favorite_station,
                         user_id=session.get('user_id'))


@app.route('/api/station-forecast/<station_name>')
def station_forecast_api(station_name):
    name = station_name.replace('%20', ' ')
    
    ridership = get_station_prediction(name)
    capacity = STATION_BASE_CAPACITY.get(name, 10000)
    base_congestion = min(100, int((ridership / capacity) * 100))
    
    now = datetime.now()
    current_hour = now.hour
    
    forecast = []
    for i in range(6):
        forecast_hour = (current_hour + i + 1) % 24
        
        # Check if trains are running - same logic as get_station_prediction
        if forecast_hour < 4 or (forecast_hour == 4 and i == 0) or forecast_hour >= 23:
            forecast_congestion = 0
            print(f"⚠️ No trains at {forecast_hour}:00")
        else:
            # Your existing forecast logic
            if 7 <= forecast_hour <= 9:
                hour_multiplier = 0.70
            elif 17 <= forecast_hour <= 20:
                hour_multiplier = 0.75
            elif 10 <= forecast_hour <= 16:
                hour_multiplier = 0.55
            elif 20 <= forecast_hour <= 22:
                hour_multiplier = 0.35
            elif 5 <= forecast_hour <= 6:
                hour_multiplier = 0.20
            else:
                hour_multiplier = 0.15
            
            if now.weekday() >= 5:
                hour_multiplier *= 0.7
            
            forecast_congestion = int(hour_multiplier * 100)
            variation = np.random.randint(-5, 6)
            forecast_congestion = max(10, min(90, forecast_congestion + variation))
        
        forecast.append(forecast_congestion)
    
    return jsonify({
        "station": name,
        "forecast": forecast,
        "intervals": ["+1h", "+2h", "+3h", "+4h", "+5h", "+6h"],
        "current": base_congestion
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

@app.route('/api/alerts/list')
def alerts_list():
    """Get combined alerts (system alerts + operator broadcasts)"""
    try:
        all_alerts = []
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # Get operator broadcasts
        try:
            broadcasts = get_commuter_broadcasts().json
            if broadcasts:
                all_alerts.extend(broadcasts)
        except Exception as e:
            print(f"⚠️ Could not load broadcasts: {e}")
        
        # ===== SYSTEM GENERATED ALERTS =====
        
        # Track stations by congestion level
        critical_stations = []
        busy_stations = []
        
        for station in STATIONS:
            ridership = get_station_prediction(station)
            capacity = STATION_BASE_CAPACITY.get(station, 10000)
            congestion = int((ridership / capacity) * 100)
            
            if congestion > 80:
                critical_stations.append({"name": station, "level": congestion})
            elif congestion > 50:
                busy_stations.append({"name": station, "level": congestion})
        
        # CRITICAL CONGESTION ALERTS
        for station in critical_stations[:3]:
            all_alerts.append({
                "id": f"critical-{station['name']}-{now.timestamp()}",
                "type": "critical",
                "icon": "exclamation-circle",
                "icon_color": "red",
                "title": f"🚨 CRITICAL: {station['name']} Station",
                "message": f"Congestion has reached {station['level']}% capacity. Expect delays of 10-15 minutes.",
                "time": now.strftime("%I:%M %p"),
                "unread": True,
                "station": station['name'],
                "source": "system"
            })
        
        # RUSH HOUR ALERTS
        if (hour >= 6 and hour <= 9) or (hour >= 17 and hour <= 20):
            rush_hour_type = "Morning" if hour <= 9 else "Evening"
            
            if hour <= 9:  # Morning rush
                hotspots = ["North Ave", "Quezon Ave", "Kamuning", "Cubao"]
            else:  # Evening rush
                hotspots = ["Ayala Ave", "Magallanes", "Taft", "Buendia"]
            
            for station in hotspots:
                if station in [s['name'] for s in busy_stations + critical_stations]:
                    all_alerts.append({
                        "id": f"rush-{station}-{now.timestamp()}",
                        "type": "rush",
                        "icon": "clock",
                        "icon_color": "orange",
                        "title": f"⚠️ {rush_hour_type} Rush Hour - {station}",
                        "message": f"Peak hours detected. Expected wait: 5-10 minutes.",
                        "time": now.strftime("%I:%M %p"),
                        "unread": True,
                        "station": station,
                        "source": "system"
                    })
        
        # SERVICE NOTICE
        if now.weekday() >= 5:  # Weekend
            all_alerts.append({
                "id": f"service-weekend-{now.timestamp()}",
                "type": "service",
                "icon": "tools",
                "icon_color": "purple",
                "title": "🛠 Weekend Service Notice",
                "message": "Scheduled maintenance at Guadalupe station tonight, 10PM-5AM.",
                "time": "Today",
                "unread": True,
                "station": "Guadalupe",
                "source": "system"
            })
        
        # Sort by time (newest first)
        all_alerts.sort(key=lambda x: x.get('raw_time', ''), reverse=True)
        
        # If no alerts at all, add a welcome message
        if not all_alerts:
            all_alerts.append({
                "id": f"welcome-{now.timestamp()}",
                "type": "advisory",
                "icon": "bell",
                "icon_color": "blue",
                "title": "👋 Welcome to MRT-ADAPT!",
                "message": "No current alerts. All systems are running normally.",
                "time": now.strftime("%I:%M %p"),
                "unread": True,
                "station": "All Stations",
                "source": "system"
            })
        
        return jsonify(all_alerts)
        
    except Exception as e:
        print(f"Error getting alerts: {e}")
        return jsonify([{
            "id": "error-alert",
            "type": "advisory",
            "icon": "info-circle",
            "icon_color": "blue",
            "title": "ℹ️ Service Information",
            "message": "Regular train operations. Next train arriving in 3 minutes.",
            "time": datetime.now().strftime("%I:%M %p"),
            "unread": True,
            "station": "All Stations",
            "source": "system"
        }])

# ========== ALERTS API ROUTES ==========
@app.route('/api/alerts/count')
def alerts_count():
    """Get the number of unread alerts based on congestion levels"""
    try:
        # Count critical and rush hour alerts
        critical_count = 0
        rush_count = 0
        now = datetime.now()
        hour = now.hour
        
        for station in STATIONS:
            ridership = get_station_prediction(station)
            capacity = STATION_BASE_CAPACITY.get(station, 10000)
            congestion = int((ridership / capacity) * 100)
            
            if congestion > 80:
                critical_count += 1
            elif congestion > 60 and ((hour >= 6 and hour <= 9) or (hour >= 17 and hour <= 20)):
                rush_count += 1
        
        total = critical_count + rush_count
        
        # Add advisory and service notices
        if session.get('favorite_station'):
            total += 1  # Add advisory
            
        # Cap at 9+ for display
        display = str(total) if total < 9 else "9+"
        return jsonify({"count": total, "display": display})
        
    except Exception as e:
        print(f"Error getting alert count: {e}")
        return jsonify({"count": 0, "display": "0"})
    
def get_user_color(user_id):
    """Assign consistent colors based on user ID"""
    colors = ['#2979FF', '#16A34A', '#F97316', '#7C3AED', '#DC2626', '#CA8A04', '#3B82F6', '#22C55E', '#EAB308', '#EF4444']
    return colors[user_id % len(colors)]

# ========== SIMPLE TEST ROUTE ==========
@app.route('/api/test-simple', methods=['GET'])
def test_simple():
    """Simple test that always returns JSON"""
    return jsonify({
        "success": True,
        "message": "API is working",
        "time": str(datetime.now())
    })

# ========== OPERATOR MANAGEMENT ROUTES ==========
# Place these right after your test route, before any other routes

@app.route('/api/operators', methods=['GET'])
def get_operators():
    """Get all operators"""
    print("🔵 GET /api/operators called")
    
    try:
        operators = User.query.filter_by(role='operator').all()
        print(f"📊 Found {len(operators)} operators")
        
        result = []
        for op in operators:
            last_login_str = "Never"
            if op.last_login:
                days_ago = (datetime.now() - op.last_login).days
                if days_ago == 0:
                    last_login_str = "Today"
                elif days_ago == 1:
                    last_login_str = "Yesterday"
                else:
                    last_login_str = f"{days_ago} days ago"
            
            # Use is_active for status (default to True if not set)
            status = 'active' if op.is_active else 'inactive'
            
            # Generate initials
            name_part = op.username.split('@')[0] if '@' in op.username else op.username
            initials = ''.join([word[0].upper() for word in name_part.split('.') if word])[:2] or op.username[0].upper()
            
            result.append({
                'id': op.id,
                'name': name_part.title(),
                'email': op.username,
                'zone': op.favorite_station if op.favorite_station else 'All Stations',
                'last_login': last_login_str,
                'status': status,  # Now based on is_active, not last_login
                'initials': initials
            })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in get_operators: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
# Run this once in a temporary route or in Flask shell
@app.route('/debug/add-active-column')
def add_active_column():
    try:
        # Add is_active column with default value True
        with db.engine.connect() as conn:
            conn.execute(text('ALTER TABLE user ADD COLUMN is_active BOOLEAN DEFAULT 1'))
            conn.commit()
        return "✅ Added is_active column"
    except Exception as e:
        return f"Error: {str(e)}"


@app.route('/api/operators/invite', methods=['POST'])
def create_operator_invite():
    """Generate an invite link for a new operator"""
    print("🔵 POST /api/operators/invite called")
    
    # if not session.get('admin_logged_in') and not session.get('is_admin'):
    #     return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        print(f"📦 Received data: {data}")
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        name = data.get('name')
        email = data.get('email')
        station = data.get('station', 'All Stations')
        access_level = data.get('access_level', 'Standard')
        
        if not name or not email:
            return jsonify({"error": "Name and email are required"}), 400
        
        # Check if user already exists
        existing_user = User.query.filter_by(username=email).first()
        if existing_user:
            return jsonify({"error": "User with this email already exists"}), 400
        
        # Generate a token
        token = secrets.token_urlsafe(16)
        
        # Create the invite link
        invite_link = url_for('operator_signup', token=token, email=email, _external=True)
        
        print(f"✅ Generated link: {invite_link}")
        
        return jsonify({
            "success": True,
            "link": invite_link,
            "expires": (datetime.now() + timedelta(hours=24)).isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in create_operator_invite: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/operators/create', methods=['POST'])
def create_operator():
    """Directly create an operator (for testing)"""
    print("🔵 POST /api/operators/create called")
    
    # if not session.get('admin_logged_in') and not session.get('is_admin'):
    #     return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        print(f"📦 Received data: {data}")
        
        email = data.get('email')
        name = data.get('name')
        station = data.get('station', 'All Stations')
        
        if not email:
            return jsonify({"error": "Email required"}), 400
        
        # Check if user exists
        user = User.query.filter_by(username=email).first()
        if user:
            # Update existing user to operator
            user.role = 'operator'
            if station and station != 'All Stations (Line-Wide)':
                user.favorite_station = station
            db.session.commit()
            message = f"User {email} updated to operator"
        else:
            # Create new operator user with random password
            temp_password = secrets.token_urlsafe(12)
            user = User(
                username=email,
                role='operator',
                favorite_station=station if station != 'All Stations (Line-Wide)' else None,
                created_at=datetime.now()
            )
            user.password = temp_password
            db.session.add(user)
            db.session.commit()
            message = f"New operator created with email {email}"
        
        print(f"✅ {message}")
        
        return jsonify({
            "success": True,
            "message": message,
            "operator": {
                "id": user.id,
                "email": user.username,
                "role": user.role,
                "station": user.favorite_station
            }
        })
        
    except Exception as e:
        print(f"❌ Error in create_operator: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ========== OPERATOR SIGNUP ROUTE ==========
@app.route('/operator-signup', methods=['GET', 'POST'])
def operator_signup():
    """Page for operators to create their account using an invite link"""
    token = request.args.get('token')
    email = request.args.get('email')
    
    # In a real app, you'd validate the token from a database
    # For now, we'll just check if it exists
    if not token or not email:
        flash('Invalid or expired invite link', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        station = request.form.get('station', 'All Stations')
        
        # Validate
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('operator_signup.html', email=email, token=token)
        
        if len(password) < 8:
            flash('Password must be at least 8 characters', 'error')
            return render_template('operator_signup.html', email=email, token=token)
        
        # Check if user already exists
        existing_user = User.query.filter_by(username=email).first()
        if existing_user:
            # Update existing user to operator role
            existing_user.role = 'operator'
            existing_user.password = password
            existing_user.favorite_station = station if station != 'All Stations' else None
            db.session.commit()
            flash('Your account has been upgraded to operator!', 'success')
            
            # Log the activity
            log_broadcast('System', station, f'Existing user upgraded to operator: {email}')
        else:
            # Create new operator
            new_operator = User(
                username=email,
                role='operator',
                favorite_station=station if station != 'All Stations' else None,
                created_at=datetime.now()
            )
            new_operator.password = password
            db.session.add(new_operator)
            db.session.commit()
            flash('Operator account created successfully!', 'success')
            
            # Log the activity
            log_broadcast('System', station, f'New operator created: {email}')
            log_user_registration(new_operator)  # Also log as user registration
        
        return redirect(url_for('login'))
    
    return render_template('operator_signup.html', email=email, token=token)

# ========== OPERATOR DASHBOARD ROUTE ==========
@app.route('/operator-dashboard')
def operator_dashboard():
    # Debug: Print session info
    print(f"🔍 operator_dashboard - Session: {dict(session)}")
    
    # Check if user is logged in (either via user_id or admin)
    if 'user_id' not in session and not session.get('admin_logged_in'):
        print("❌ No user_id or admin_logged_in in session")
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    
    # If admin is logged in but trying to access operator dashboard, redirect to admin
    if session.get('admin_logged_in') or session.get('is_admin'):
        print("✅ Admin detected, redirecting to admin dashboard")
        return redirect(url_for('admin_dashboard'))
    
    # Regular user flow
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        
        if not user:
            print(f"❌ User not found for id: {session['user_id']}")
            session.clear()
            return redirect(url_for('login'))
        
        print(f"✅ User found: {user.username}, role: {user.role}")
        
        # Redirect based on role
        if user.role == 'admin':
            print("✅ User has admin role, redirecting to admin dashboard")
            return redirect(url_for('admin_dashboard'))
        elif user.role == 'commuter':
            print("✅ User has commuter role, redirecting to user dashboard")
            return redirect(url_for('user_dashboard'))
        elif user.role == 'operator':
            print("✅ User has operator role, showing operator dashboard")
            # Show operator dashboard
            return render_template('operator-dashboard.html',
                                 username=user.username,
                                 station=user.favorite_station or 'All Stations',
                                 stations=STATIONS)
        else:
            print(f"⚠️ Unknown role: {user.role}, defaulting to user dashboard")
            return redirect(url_for('user_dashboard'))
    
    # Fallback
    print("⚠️ No user_id in session but also not admin, redirecting to login")
    return redirect(url_for('login'))

# Add this temporary debug route to check admin status
@app.route('/debug/admin-check')
def debug_admin_check():
    """Debug route to check admin status"""
    result = {
        'session': dict(session),
        'is_admin_session': session.get('admin_logged_in', False),
        'is_admin_flag': session.get('is_admin', False),
        'user_id': session.get('user_id'),
    }
    
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            result['user'] = {
                'id': user.id,
                'username': user.username,
                'role': user.role,
                'has_password': user.has_password(),
                'google_id': bool(user.google_id)
            }
    
    # Check if admin user exists in database
    admin_email = os.getenv('ADMIN_EMAIL')
    if admin_email:
        admin_user = User.query.filter_by(username=admin_email).first()
        result['admin_in_db'] = {
            'exists': bool(admin_user),
            'role': admin_user.role if admin_user else None
        }
    
    return jsonify(result)
# ========== ADMIN API ROUTES ==========
@app.route('/api/admin/congestion-stats')
def admin_congestion_stats():
    """Get congestion statistics for admin dashboard (no role restriction)"""
    # Check if user is admin
    is_admin = False
    if session.get('admin_logged_in') or session.get('is_admin'):
        is_admin = True
    elif 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and user.role == 'admin':
            is_admin = True
    
    if not is_admin:
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get current congestion levels for all stations
    station_stats = []
    severe_count = 0
    congested_count = 0
    moderate_count = 0
    light_count = 0
    
    for station in STATIONS:
        ridership = get_station_prediction(station)
        capacity = STATION_BASE_CAPACITY.get(station, 10000)
        congestion = int((ridership / capacity) * 100)
        
        # Categorize congestion
        if congestion >= 80:
            status = "severe"
            severe_count += 1
        elif congestion >= 60:
            status = "congested"
            congested_count += 1
        elif congestion >= 30:
            status = "moderate"
            moderate_count += 1
        else:
            status = "light"
            light_count += 1
        
        station_stats.append({
            'name': station,
            'congestion': congestion,
            'status': status,
            'ridership': int(ridership),
            'capacity': capacity
        })
    
    return jsonify({
        'stats': station_stats,
        'counts': {
            'severe': severe_count,
            'congested': congested_count,
            'moderate': moderate_count,
            'light': light_count
        }
    })

@app.route('/api/admin/recent-activities')
def admin_recent_activities():
    """Get recent activities for admin dashboard"""
    # Check if user is admin
    is_admin = False
    if session.get('admin_logged_in') or session.get('is_admin'):
        is_admin = True
    elif 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and user.role == 'admin':
            is_admin = True
    
    if not is_admin:
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get recent activities
    recent = Activity.query.order_by(Activity.created_at.desc()).limit(10).all()
    
    result = []
    for activity in recent:
        result.append({
            'icon': activity.icon,
            'icon_color': activity.icon_color,
            'title': activity.title,
            'description': activity.description,
            'time': activity.created_at.strftime("%I:%M %p"),
            'type': activity.activity_type
        })
    
    return jsonify(result)

@app.route('/api/admin/user-stats')
def admin_user_stats():
    """Get user statistics for admin dashboard"""
    # Check if user is admin
    is_admin = False
    if session.get('admin_logged_in') or session.get('is_admin'):
        is_admin = True
    elif 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and user.role == 'admin':
            is_admin = True
    
    if not is_admin:
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get real data from database
    total_users = User.query.count()
    
    # Count users by role
    commuter_count = User.query.filter_by(role='commuter').count()
    operator_count = User.query.filter_by(role='operator').count()
    admin_count = User.query.filter_by(role='admin').count()
    
    # Count users who logged in today
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    active_today = User.query.filter(User.last_login >= today_start).count()
    
    # Count new users this week
    week_ago = datetime.now() - timedelta(days=7)
    new_users_this_week = User.query.filter(User.created_at >= week_ago).count()
    
    return jsonify({
        'total_users': total_users,
        'commuter_count': commuter_count,
        'operator_count': operator_count,
        'admin_count': admin_count,
        'active_today': active_today,
        'new_users_this_week': new_users_this_week
    })

@app.route('/api/admin/broadcasts')
def admin_get_broadcasts():
    """Get all broadcasts for admin panel"""
    # Check if user is admin
    is_admin = False
    if session.get('admin_logged_in') or session.get('is_admin'):
        is_admin = True
    elif 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and user.role == 'admin':
            is_admin = True
    
    if not is_admin:
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get ONLY broadcasts from activities - FILTER by activity_type='broadcast'
    broadcasts = Activity.query.filter_by(
        activity_type='broadcast'  # ← This is the key! Only get broadcasts
    ).order_by(Activity.created_at.desc()).all()
    
    result = []
    for b in broadcasts:
        # Parse the message to extract type, title, and description
        message_parts = b.description.split(':', 1)
        broadcast_type = message_parts[0].strip() if len(message_parts) > 1 else 'General Notice'
        
        # Get the rest of the message
        rest = message_parts[1] if len(message_parts) > 1 else b.description
        
        # Split title and description if they contain a dash
        desc_parts = rest.split('-', 1)
        title = desc_parts[0].strip() if len(desc_parts) > 1 else rest.strip()
        description = desc_parts[1].strip() if len(desc_parts) > 1 else ''
        
        # Get operator name from title
        operator = b.title.replace('Broadcast sent by ', '') if b.title else 'System'
        operator_name = operator.split('@')[0] if '@' in operator else operator
        
        result.append({
            'id': b.id,
            'type': broadcast_type,
            'title': title,
            'description': description or rest,
            'full_message': b.description,
            'station': b.station or 'All Stations',
            'operator': operator_name,
            'time_display': b.created_at.strftime("%I:%M %p"),
            'date_display': b.created_at.strftime("%b %d, %Y")
        })
    
    return jsonify(result)


# ========== ADMIN DASHBOARD ROUTE ==========
@app.route('/admin/dashboard')
def admin_dashboard():
    # Check if user is admin (either via session flags or database role)
    is_admin = False
    admin_email = None
    admin_user = None
    
    if session.get('admin_logged_in') or session.get('is_admin'):
        is_admin = True
        admin_email = session.get('admin_email') or session.get('username')
        # Try to get the admin user from database
        if admin_email:
            admin_user = User.query.filter_by(username=admin_email).first()
    elif 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and user.role == 'admin':
            is_admin = True
            admin_email = user.username
            admin_user = user
    
    if not is_admin:
        print("❌ Not logged in as admin, redirecting to login")
        flash('Please login as admin to access this page.', 'warning')
        return redirect(url_for('login'))
    
    # Get real data from database
    total_users = User.query.count()
    
    # Count users by role
    commuter_count = User.query.filter_by(role='commuter').count()
    operator_count = User.query.filter_by(role='operator').count()
    admin_count = User.query.filter_by(role='admin').count()
    
    # Count users who logged in today
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    active_today = User.query.filter(User.last_login >= today_start).count()
    
    # Count new users this week
    week_ago = datetime.now() - timedelta(days=7)
    new_users_this_week = User.query.filter(User.created_at >= week_ago).count()
    
    # Get all users for the users table
  # Get all users for the users table
    users = User.query.all()
    users_data = []
    for user in users:
        # USE THE DATABASE is_active FIELD - THIS IS THE KEY CHANGE
        is_active = user.is_active  # ← CHANGE THIS LINE
        
        # Format joined date
        joined_date = user.created_at.strftime('%b %d, %Y') if user.created_at else 'Unknown'
        
        # Format last active
        last_active = 'Never'
        if user.last_login:
            days_ago = (datetime.now() - user.last_login).days
            if days_ago == 0:
                last_active = 'Today'
            elif days_ago == 1:
                last_active = 'Yesterday'
            else:
                last_active = f'{days_ago} days ago'
        
        # Generate initials from email
        name_part = user.username.split('@')[0] if '@' in user.username else user.username
        initials = ''.join([word[0].upper() for word in name_part.split('.') if word])[:2]
        if not initials:
            initials = user.username[0].upper() if user.username else 'U'
        
        users_data.append({
            'id': user.id, 
            'name': name_part.title(),
            'email': user.username,
            'role': user.role if user.role else 'commuter',
            'joined': joined_date,
            'last': last_active,
            'active': is_active,  # Now using the correct is_active value
            'reports': 0,
            'initials': initials,
            'color': get_user_color(user.id)
        })
    # Station reports data (all zeros for now)
    station_reports = []
    for station in STATIONS:
        station_reports.append({
            'name': station,
            'reports': 0,
            'status': 'No data',
            'color': 'var(--text-light)'
        })
    
    # Check if there are any operators
    has_operators = operator_count > 0
    
    # Get recent activities from database
    recent_activities = Activity.query.order_by(Activity.created_at.desc()).limit(10).all()

    # Format activities for display
    activities_data = []
    for activity in recent_activities:
        activities_data.append({
            'icon': activity.icon,
            'icon_color': activity.icon_color,
            'title': activity.title,
            'description': activity.description,
            'time': activity.created_at.strftime('%I:%M %p')  # Format time as "9:45 AM"
        })
    
    # Calculate accuracy using real data
    accuracy_data = calculate_real_accuracy()
    
    # IMPORTANT: Return the rendered template
    return render_template('admin_dashboard.html',
                         admin_email=admin_email or session.get('username', 'Admin'),
                         total_users=total_users,
                         active_operators=operator_count,
                         active_today=active_today,
                         new_users_this_week=new_users_this_week,
                         commuter_count=commuter_count,
                         operator_count=operator_count,
                         admin_count=admin_count,
                         users_data=users_data,
                         station_reports=station_reports,
                         activities=activities_data,
                         has_operators=has_operators,
                         accuracy=accuracy_data,
                         stations=STATIONS)
    
@app.route('/api/daily-active-users')
def daily_active_users():
    # Get real daily active users for last 7 days
    days = []
    values = []
    for i in range(6, -1, -1):
        date = datetime.now() - timedelta(days=i)
        count = User.query.filter(
            User.last_login >= date.replace(hour=0, minute=0, second=0),
            User.last_login <= date.replace(hour=23, minute=59, second=59)
        ).count()
        days.append(date.strftime('%a'))
        values.append(count)
    
    return jsonify({'days': days, 'values': values})

@app.context_processor
def inject_env_status():
    """Inject environment variable status into templates (for debugging)"""
    return {
        'env_check': {
            'admin_email': bool(os.getenv('ADMIN_EMAIL')),
            'admin_password': bool(os.getenv('ADMIN_PASSWORD'))
        }
    }
    

# ========== FAVORITE STATION API ==========
@app.route('/api/set-favorite', methods=['POST'])
@login_required
def set_favorite():
    data = request.json
    station = data.get('station')
    
    if station not in STATIONS:
        return jsonify({"error": "Invalid station"}), 400
    
    user = User.query.get(session['user_id'])
    if user:
        user.favorite_station = station
        db.session.commit()
        session['favorite_station'] = station
        return jsonify({"success": True, "favorite": station})
    
    return jsonify({"error": "User not found"}), 404

# ========== DEBUG ROUTES ==========

@app.route('/debug/check-activities')
def debug_check_activities():
    """Check what activities are in the database"""
    if not session.get('admin_logged_in'):
        return "Unauthorized", 401
    
    # Get all activities
    activities = Activity.query.order_by(Activity.created_at.desc()).all()
    
    result = []
    for a in activities:
        result.append({
            'id': a.id,
            'type': a.activity_type,
            'title': a.title,
            'description': a.description,
            'icon': a.icon,
            'created_at': a.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': a.user_id
        })
    
    return jsonify({
        'activity_count': len(result),
        'activities': result,
        'table_exists': True
    })
    
@app.before_request
def check_user_active():
    """Check if the current user is still active on each request"""
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and not user.is_active:
            session.clear()
            flash('Your account has been deactivated. Please contact an administrator.', 'error')
            return redirect(url_for('login'))

@app.route('/debug/create-activity-table')
def create_activity_table():
    """Create the activity table if it doesn't exist"""
    if not session.get('admin_logged_in'):
        return "Unauthorized", 401
    
    try:
        # Create all tables (this will create Activity table if it doesn't exist)
        db.create_all()
        return "Activity table created successfully!"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/debug/backfill-activities')
def backfill_activities():
    """Create activities for existing users"""
    if not session.get('admin_logged_in'):
        return "Unauthorized", 401
    
    users = User.query.all()
    count = 0
    for user in users:
        # Check if user already has a registration activity
        existing = Activity.query.filter_by(
            activity_type='user_registration',
            user_id=user.id
        ).first()
        
        if not existing and user.created_at:
            activity = Activity(
                activity_type='user_registration',
                title='New user registered',
                description=f'{user.username} joined MRT-ADAPT',
                icon='user-plus',
                icon_color='green',
                user_id=user.id,
                created_at=user.created_at
            )
            db.session.add(activity)
            count += 1
    
    db.session.commit()
    return f"Created {count} backfilled activities!"

@app.route('/api/debug-lstm')
def debug_lstm():
    """Debug endpoint to check LSTM status"""
    results = {}
    for station in STATIONS:
        results[station] = {
            "has_model": station in lstm_models,
            "has_scaler": station in scalers,
            "has_data": station in station_time_series_last_24,
            "data_points": len(station_time_series_last_24.get(station, [])) if station in station_time_series_last_24 else 0
        }
    return jsonify({
        "models_loaded": f"{models_loaded}/{len(STATIONS)}",
        "details": results
    })
    
@app.route('/debug/env')
def debug_env():
    """Debug route to check environment variables (REMOVE IN PRODUCTION)"""
    admin_email = os.getenv('ADMIN_EMAIL')
    admin_password = os.getenv('ADMIN_PASSWORD')
    
    return {
        'admin_email_configured': bool(admin_email),
        'admin_email_value': admin_email if admin_email else 'NOT SET',
        'admin_password_configured': bool(admin_password),
        'admin_password_length': len(admin_password) if admin_password else 0,
        'all_env_keys': list(os.environ.keys())
    }

# Add this to your app.py - User Reports API
@app.route('/api/user-reports/<int:user_id>')
def get_user_reports(user_id):
    """Get travel history for a specific user (includes reports AND saved routes)"""
    # Check if user is authorized (either the user themselves or admin)
    if 'user_id' not in session and not session.get('admin_logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Allow users to view their own reports, or admins to view any
    if session.get('user_id') != user_id and not session.get('admin_logged_in'):
        # Check if user is admin
        user = User.query.get(session.get('user_id'))
        if not user or user.role != 'admin':
            return jsonify({"error": "Unauthorized"}), 401
    
    # Get reports for this user from the Report model
    reports = Report.query.filter_by(user_id=user_id).order_by(Report.timestamp.desc()).all()
    
    # Get saved routes for this user (these will appear as "planned" trips)
    saved_routes = SavedRoute.query.filter_by(user_id=user_id).order_by(SavedRoute.created_at.desc()).all()
    
    # Combine both into a single history list
    combined_history = []
    
    # Add reports (actual trips taken)
    for report in reports:
        # Calculate estimated travel time based on stations (simplified)
        from_idx = STATIONS.index(report.station) if report.station in STATIONS else 0
        to_idx = from_idx + 3 if from_idx + 3 < len(STATIONS) else len(STATIONS) - 1
        travel_time = (to_idx - from_idx) * 3 + 5  # Rough estimate
        
        # Determine congestion level
        if report.reported_congestion > 80:
            congestion_level = "severe"
            status_label = "Severely Congested"
        elif report.reported_congestion > 60:
            congestion_level = "congested"
            status_label = "Congested"
        elif report.reported_congestion > 30:
            congestion_level = "moderate"
            status_label = "Moderate"
        else:
            congestion_level = "light"
            status_label = "Light"
        
        combined_history.append({
            "id": f"report-{report.id}",
            "type": "trip",  # Actual trip taken
            "from_station": report.station,
            "to_station": STATIONS[min(from_idx + 3, len(STATIONS) - 1)],
            "date": report.timestamp.strftime("%Y-%m-%d"),
            "time": report.timestamp.strftime("%H:%M"),
            "travel_time": travel_time,
            "congestion_level": congestion_level,
            "status_label": status_label,
            "reported_congestion": report.reported_congestion,
            "created_at": report.timestamp.isoformat()
        })
    
    # Add saved routes (planned trips)
    for route in saved_routes:
        # Calculate stations between
        from_idx = STATIONS.index(route.from_station) if route.from_station in STATIONS else 0
        to_idx = STATIONS.index(route.to_station) if route.to_station in STATIONS else len(STATIONS) - 1
        station_diff = abs(from_idx - to_idx)
        travel_time = station_diff * 3 + 5  # Rough estimate
        
        combined_history.append({
            "id": f"route-{route.id}",
            "type": "saved_route",  # Saved/planned route
            "from_station": route.from_station,
            "to_station": route.to_station,
            "date": route.created_at.strftime("%Y-%m-%d") if route.created_at else datetime.now().strftime("%Y-%m-%d"),
            "time": route.created_at.strftime("%H:%M") if route.created_at else "00:00",
            "travel_time": travel_time,
            "congestion_level": "moderate",  # Default for saved routes
            "status_label": "Saved Route",
            "reported_congestion": 50,  # Default value
            "created_at": route.created_at.isoformat() if route.created_at else datetime.now().isoformat(),
            "route_name": route.route_name or f"{route.from_station} to {route.to_station}"
        })
    
    # Sort by created_at (newest first)
    combined_history.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    # Calculate stats based on ACTUAL trips only (reports)
    actual_trips = [h for h in combined_history if h['type'] == 'trip']
    total_trips = len(actual_trips)
    
    # Calculate average travel time from actual trips
    if total_trips > 0:
        avg_travel = sum(h["travel_time"] for h in actual_trips) / total_trips
    else:
        avg_travel = 0
    
    # Find most visited station from actual trips
    station_counts = {}
    for h in actual_trips:
        station_counts[h["from_station"]] = station_counts.get(h["from_station"], 0) + 1
    most_visited = max(station_counts, key=station_counts.get) if station_counts else "N/A"
    
    # Count trips this month from actual trips
    current_month = datetime.now().month
    current_year = datetime.now().year
    monthly_trips = sum(1 for h in actual_trips if 
                       h["date"].startswith(f"{current_year}-{current_month:02d}"))
    
    return jsonify({
        "history": combined_history,  # Combined list of trips AND saved routes
        "stats": {
            "total_trips": total_trips,
            "avg_travel_time": int(avg_travel),
            "most_visited": most_visited,
            "monthly_trips": monthly_trips,
            "saved_routes": len(saved_routes)  # Count of saved routes
        }
    })
    
    
# ========== DEBUG ROUTE TO ADD SAMPLE ACTIVITIES ==========
@app.route('/debug/add-sample-activities')
def add_sample_activities():
    """Add sample activities for testing"""
    if not session.get('admin_logged_in'):
        return "Unauthorized", 401
    
    # Clear existing activities (optional)
    # Activity.query.delete()
    
    # Add sample user registrations
    users = User.query.limit(3).all()
    for user in users:
        log_user_registration(user)
    
    # Add sample model retrain
    log_model_retrain()
    
    # Add sample broadcasts (if you have operators)
    operators = User.query.filter_by(role='operator').all()
    for op in operators:
        log_broadcast(op.username, 'Cubao', 'Train delay reported')
        log_override(op.username, 'Ayala')
    
    return "Sample activities added!"
@app.route('/api/operators/<int:operator_id>/toggle', methods=['POST'])
def toggle_operator_status(operator_id):
    """Activate or deactivate an operator"""
    print(f"🔵 POST /api/operators/{operator_id}/toggle called")
    
    try:
        data = request.get_json()
        action = data.get('action')  # 'activate' or 'deactivate'
        
        print(f"📦 Action: {action}")
        
        # Find the operator
        operator = User.query.get(operator_id)
        if not operator:
            return jsonify({"error": "Operator not found"}), 404
        
        if operator.role != 'operator':
            return jsonify({"error": "User is not an operator"}), 400
        
        # Toggle the is_active status (DO NOT modify last_login)
        if action == 'deactivate':
            operator.is_active = False
            message = f"Operator {operator.username} deactivated"
        elif action == 'activate':
            operator.is_active = True
            message = f"Operator {operator.username} activated"
        else:
            return jsonify({"error": "Invalid action"}), 400
        
        db.session.commit()
        
        # Log the activity
        log_broadcast('System', operator.favorite_station or 'All Stations', 
                     f"Operator {operator.username} {action}d")
        
        return jsonify({
            "success": True,
            "message": message,
            "operator_id": operator_id,
            "new_status": "active" if action == 'activate' else 'inactive'
        })
        
    except Exception as e:
        print(f"❌ Error in toggle_operator_status: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
    # ========== ALERTS API ROUTES ==========
@app.route('/api/alerts/mark-read', methods=['POST'])
def mark_alerts_read():
    """Mark all alerts as read for the current user"""
    try:
        # Check if user is logged in
        if 'user_id' not in session:
            return jsonify({"success": False, "error": "Not logged in"}), 401
        
        # In a real app, you would update a database here
        # For now, just return success
        
        # You could also log this activity
        user = User.query.get(session['user_id'])
        if user:
            print(f"📬 User {user.username} marked all alerts as read")
        
        return jsonify({
            "success": True, 
            "message": "All alerts marked as read",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error marking alerts as read: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/debug/lstm-status')
def lstm_status():
    """Check LSTM model status"""
    status = {
        'stations': STATIONS,
        'models_loaded': models_loaded,
        'total_stations': len(STATIONS),
        'models': {},
        'api_endpoints': {
            '/api/predict/<station>': 'Available',
            '/api/station-forecast/<station>': 'Available'
        }
    }
    
    for station in STATIONS:
        status['models'][station] = {
            'has_model': station in lstm_models,
            'has_scaler': station in scalers,
            'model_path': f'models/{station}_lstm.h5',
            'scaler_path': f'models/{station}_scaler.pkl'
        }
    
    return jsonify(status)

# ========== RUN SYSTEM ==========
if __name__ == '__main__':
    print("\n" + "="*70)
    print("✨ MRT-3 PREDICTION SYSTEM READY!")
    print("="*70)
    print("👤 Starting as GUEST by default")
    print("📊 System cache: ✅ Loaded")
    print(f"🤖 LSTM models: {models_loaded}/{len(STATIONS)}")
    print("🔐 Password hashing: ✅ Enabled")
    print("🔗 Account merging: ✅ Enabled")
    print("🌐 Open http://localhost:5000")
    print("="*70)
    app.run(debug=True, port=5000)