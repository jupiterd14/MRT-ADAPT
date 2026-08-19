import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, session, flash, redirect, url_for, jsonify
from extensions import cache
from dotenv import load_dotenv
import warnings
from authlib.integrations.flask_client import OAuth
import pickle
import tempfile
from datetime import datetime, timedelta



load_dotenv()
warnings.filterwarnings('ignore')

from config import Config

from models import db
from models.user import User
from models.report import Report
from models.broadcast import Broadcast
from models.activity_log import ActivityLog
from models.saved_route import SavedRoute
from models.station_data import StationData
# Add after your existing imports
from services.lstm_integration import MRT3LSTMPredictor, init_lstm_predictor, schedule_weekly_retraining

from utils import (
    STATIONS, STATION_BASE_CAPACITY, STATION_COORDINATES,
    get_operator_stations, get_station_list, get_capacity,
    track_report_submission, is_rate_limited, is_suspicious_remarks, check_duplicate_report,
    log_activity as utils_log_activity,
    report_tracker
)

from services import (
    load_directional_models, load_real_historical_data,
    get_directional_prediction, get_station_prediction,
    get_feature_sequence_for_station,
    directional_models, directional_scalers,
    historical_entry, historical_exit, hourly_avg_entry, hourly_avg_exit
)

from routes import (
    auth_bp, user_bp, admin_bp, operator_bp, public_bp,
    api_predict_bp, api_schedule_bp, api_reports_bp, api_other_bp,
    model_perf_bp, email_bp  # ADD email_bp HERE
)
# ============ CACHE SETUP FOR FAST RELOADS ============
_MODELS_CACHE = {}
_MODELS_CACHE_FILE = None



def get_models_cache_path():
    """Get a persistent cache file path for models"""
    cache_dir = tempfile.gettempdir()
    return os.path.join(cache_dir, 'mrt3_models_cache.pkl')

def get_historical_cache_path():
    """Get cache file path for historical data"""
    cache_dir = tempfile.gettempdir()
    return os.path.join(cache_dir, 'mrt3_historical_cache.pkl')

def load_models_with_cache(stations, models_path):
    """Load models with disk caching to speed up reloads"""
    global _MODELS_CACHE
    
    cache_file = get_models_cache_path()
    
    # Check if models are already in memory
    if _MODELS_CACHE.get('loaded'):
        print("✓ Using models from memory cache")
        return _MODELS_CACHE['directional_models'], _MODELS_CACHE['directional_scalers']
    
    # Check disk cache
    if os.path.exists(cache_file):
        try:
            print("📦 Loading models from disk cache (fast)...")
            with open(cache_file, 'rb') as f:
                _MODELS_CACHE = pickle.load(f)
            print(f"✓ Loaded {len(_MODELS_CACHE['directional_models'])} models from cache")
            return _MODELS_CACHE['directional_models'], _MODELS_CACHE['directional_scalers']
        except Exception as e:
            print(f"Cache load failed: {e}, reloading from source...")
    
    # Load fresh from source
    print("🔄 Loading models from source (first time only - this may take a while)...")
    directional_models, directional_scalers = load_directional_models(stations, models_path)
    
    # Save to cache
    _MODELS_CACHE = {
        'directional_models': directional_models,
        'directional_scalers': directional_scalers,
        'loaded': True
    }
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(_MODELS_CACHE, f)
        print(f"✓ Models cached to {cache_file}")
        print("  Next reload will be much faster!")
    except Exception as e:
        print(f"Cache save failed: {e}")
    
    return directional_models, directional_scalers



def load_historical_with_cache(stations, base_capacity):
    """Load historical data with disk caching"""
    cache_file = get_historical_cache_path()
    
    # Check disk cache
    if os.path.exists(cache_file):
        try:
            print("📦 Loading historical data from cache...")
            with open(cache_file, 'rb') as f:
                historical_data = pickle.load(f)
            print(f"✓ Loaded historical data for {len(historical_data['historical_entry'])} stations from cache")
            return historical_data
        except Exception as e:
            print(f"Historical cache load failed: {e}, reloading from source...")
    
    # Load fresh from source
    print("🔄 Loading historical data from source (first time only)...")
    historical_data = load_real_historical_data(stations, base_capacity)
    
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(historical_data, f)
        print(f"✓ Historical data cached")
    except Exception as e:
        print(f"Historical cache save failed: {e}")
    
    return historical_data

# ============ APP INITIALIZATION ============
app = Flask(__name__, template_folder='html', static_folder='static')
app.config.from_object(Config)
cache.init_app(app, config={'CACHE_TYPE': 'SimpleCache'})

@app.route('/api/test')
def api_test():
    return jsonify({"status": "ok", "message": "API is working", "time": datetime.now().isoformat()})

db.init_app(app)

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=app.config.get('GOOGLE_CLIENT_ID'),
    client_secret=app.config.get('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

app.config['GOOGLE_CLIENT'] = google

@app.route('/uploads/reports/<filename>')
def serve_upload(filename):
    from flask import send_from_directory, abort, current_app
    import os
    
    # Use app.root_path which is the directory where app.py is located
    upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', 'reports')
    
    # Check if file exists
    file_path = os.path.join(upload_folder, filename)
    if not os.path.exists(file_path):
        abort(404)
    
    return send_from_directory(upload_folder, filename)

# Add this route as well
@app.route('/uploads/reports/<path:filename>')
def serve_upload_with_path(filename):
    """Serve uploaded images with path support"""
    from flask import send_from_directory, abort, current_app
    import os
    
    upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', 'reports')
    
    # Security: Prevent directory traversal
    safe_path = os.path.normpath(filename)
    if safe_path.startswith('..'):
        abort(403)
    
    file_path = os.path.join(upload_folder, safe_path)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        abort(404)
    
    return send_from_directory(upload_folder, safe_path)

app.register_blueprint(auth_bp, url_prefix='/')
app.register_blueprint(user_bp, url_prefix='/')
app.register_blueprint(admin_bp, url_prefix='/')
app.register_blueprint(operator_bp, url_prefix='/')
app.register_blueprint(public_bp, url_prefix='/')
app.register_blueprint(api_predict_bp, url_prefix='/api')
app.register_blueprint(api_schedule_bp, url_prefix='/api')
app.register_blueprint(api_reports_bp, url_prefix='/api')
app.register_blueprint(api_other_bp, url_prefix='/api')
app.register_blueprint(model_perf_bp, url_prefix='/api')
app.register_blueprint(email_bp, url_prefix='/api/profile')

@app.context_processor
def inject_now():
    return {'now': datetime.now()}
# ============ LAZY LOADING - Only load models when needed ============
print("\n" + "="*50)
print("MRT-3 PREDICTION SYSTEM - LAZY LOADING MODE")
print("="*50)
print("⏳ Models will load on first request (not at startup)")
print("💡 First request may take 10-30 seconds")
print("="*50 + "\n")

# Don't load models yet - just set up the config
DIRECTIONAL_MODELS_PATH = 'models_2022-2024_v8'

# Initialize empty placeholders - will be filled on first request
directional_models_cached = None
directional_scalers_cached = None
historical_data = None

# Create a function to lazy load models
def ensure_models_loaded():
    """Load models only when needed (on first API call)"""
    global directional_models_cached, directional_scalers_cached, historical_data
    
    if directional_models_cached is not None:
        return  # Already loaded
    
    print("🔄 Loading models on-demand (first request)...")
    print("⏳ This may take 10-30 seconds...")
    
    # Load models with caching
    directional_models_cached, directional_scalers_cached = load_models_with_cache(STATIONS, DIRECTIONAL_MODELS_PATH)
    
    # Load historical data with caching
    historical_data = load_historical_with_cache(STATIONS, STATION_BASE_CAPACITY)
    
    # Update the services module with loaded data
    import services
    services.directional_models = directional_models_cached
    services.directional_scalers = directional_scalers_cached
    services.historical_entry = historical_data.get('historical_entry', {})
    services.historical_exit = historical_data.get('historical_exit', {})
    services.hourly_avg_entry = historical_data.get('hourly_avg_entry', {})
    services.hourly_avg_exit = historical_data.get('hourly_avg_exit', {})
    
    print(f"✓ System ready with {len(directional_models_cached)} directional models")
    print(f"✓ Historical data loaded for {len(historical_data['historical_entry'])} stations")
    
    app.config['DIRECTIONAL_MODELS'] = directional_models_cached
    app.config['DIRECTIONAL_SCALERS'] = directional_scalers_cached

# Store function in app config for use in routes
app.config['ENSURE_MODELS_LOADED'] = ensure_models_loaded
# ============ LSTM MODELS - LAZY LOADING ============
print("\n" + "="*50)
print("LSTM MODELS - LAZY LOADING")
print("="*50)
print("⏳ LSTM models will load on first prediction request")
print("="*50 + "\n")

LSTM_MODEL_PATH = 'models_2022-2024_v8'
lstm_predictor = None

def ensure_lstm_loaded():
    """Lazy load LSTM models on first request"""
    global lstm_predictor
    
    if lstm_predictor is not None:
        return
    
    print("🔄 Loading LSTM models on-demand...")
    lstm_predictor = MRT3LSTMPredictor(model_path=LSTM_MODEL_PATH)
    
    if lstm_predictor.load_models():
        print(f"✅ LSTM models loaded successfully!")
        print(f"   📊 Loaded {len(lstm_predictor.models)} station-direction models")
        app.config['LSTM_PREDICTOR'] = lstm_predictor
        
        # Start weekly retraining in background (won't block startup)
        try:
            schedule_weekly_retraining(app)
        except Exception as e:
            print(f"⚠️ Weekly retraining not started: {e}")
    else:
        print("⚠️ LSTM models not loaded - using fallback predictions")
        app.config['LSTM_PREDICTOR'] = None

app.config['ENSURE_LSTM_LOADED'] = ensure_lstm_loaded


# ============ SINGLE MODEL LAZY LOADING ============
# Load ONLY the model needed for a specific station/direction
# This prevents memory issues on Render's free tier

def ensure_single_model_loaded(station_name, direction):
    """Load ONLY ONE model for a specific station-direction"""
    global directional_models_cached, directional_scalers_cached, historical_data
    
    # Initialize if not done yet
    if directional_models_cached is None:
        directional_models_cached = {}
        directional_scalers_cached = {}
        
        # Load historical data (small, always needed)
        historical_data = load_historical_with_cache(STATIONS, STATION_BASE_CAPACITY)
        
        # Update services with historical data
        import services
        services.historical_entry = historical_data.get('historical_entry', {})
        services.historical_exit = historical_data.get('historical_exit', {})
        services.hourly_avg_entry = historical_data.get('hourly_avg_entry', {})
        services.hourly_avg_exit = historical_data.get('hourly_avg_exit', {})
    
    # Check if THIS SPECIFIC model is already loaded
    model_key = f"{station_name}_{direction}"
    if model_key in directional_models_cached:
        return  # Already loaded
    
    print(f"🔄 Loading single model: {model_key}")
    
    # Load ONLY this one model
    try:
        from services.model_loader import load_single_model
        model, scalers = load_single_model(station_name, direction, DIRECTIONAL_MODELS_PATH)
        
        if model:
            directional_models_cached[model_key] = model
            directional_scalers_cached[f"{model_key}_feature"] = scalers.get('feature')
            directional_scalers_cached[f"{model_key}_target"] = scalers.get('target')
            print(f"✅ Loaded model for {model_key}")
        else:
            print(f"❌ Failed to load model for {model_key}")
            
    except Exception as e:
        print(f"❌ Error loading {model_key}: {e}")
    
    # Update app config with whatever models we have
    app.config['DIRECTIONAL_MODELS'] = directional_models_cached
    app.config['DIRECTIONAL_SCALERS'] = directional_scalers_cached
app.config['ENSURE_SINGLE_MODEL_LOADED'] = ensure_single_model_loaded

# ============ WRAPPER FUNCTIONS ============
def get_directional_prediction_wrapper(station_name, direction, target_datetime=None):
    """Get directional prediction with LSTM enhancement"""
    
    # First try LSTM if available
    lstm_predictor = app.config.get('LSTM_PREDICTOR')
    if lstm_predictor and hasattr(lstm_predictor, 'models') and len(lstm_predictor.models) > 0:
        try:
            from models import db
            prediction = lstm_predictor.predict_congestion(
                station_name, 
                direction,
                db.session
            )
            if prediction is not None:
                print(f"📊 Using LSTM for {station_name} {direction}: {prediction:.1f}%")
                return prediction
        except Exception as e:
            print(f"⚠️ LSTM prediction failed: {e}, falling back...")
    
    # Fallback to original prediction system
    return get_directional_prediction(
        station_name, direction, target_datetime,
        directional_models_cached, directional_scalers_cached,
        get_feature_sequence_for_station
    )

def get_station_prediction_wrapper(station_name):
    """Get station prediction with LSTM enhancement"""
    
    # First try LSTM if available
    lstm_predictor = app.config.get('LSTM_PREDICTOR')
    if lstm_predictor and hasattr(lstm_predictor, 'models') and len(lstm_predictor.models) > 0:
        try:
            from models import db
            # Try Northbound first, then Southbound
            for direction in ['Northbound', 'Southbound']:
                prediction = lstm_predictor.predict_congestion(
                    station_name, 
                    direction,
                    db.session
                )
                if prediction is not None:
                    print(f"📊 Using LSTM for {station_name}: {prediction:.1f}%")
                    return prediction
        except Exception as e:
            print(f"⚠️ LSTM prediction failed: {e}, falling back...")
    
    # Fallback to original prediction system
    return get_station_prediction(
        station_name, None,
        directional_models_cached, directional_scalers_cached,
        get_feature_sequence_for_station
    )

def log_activity_wrapper(user_id, user_type, user_email, action, details=None):
    return utils_log_activity(
        user_id, user_type, user_email, action, details,
        ActivityLog, db.session, request=None
    )

app.config['GET_DIRECTIONAL_PREDICTION'] = get_directional_prediction_wrapper
app.config['GET_STATION_PREDICTION'] = get_station_prediction_wrapper
app.config['LOG_ACTIVITY'] = log_activity_wrapper
app.config['STATIONS'] = STATIONS
app.config['STATION_BASE_CAPACITY'] = STATION_BASE_CAPACITY
app.config['STATION_COORDINATES'] = STATION_COORDINATES

typeIcons = {
    "Train Breakdown": "fa-train",
    "Overcrowding": "fa-users", 
    "Maintenance": "fa-wrench",
    "Signal Issue": "fa-satellite-dish",
    "Gate Closure": "fa-door-closed",
    "General Notice": "fa-bullhorn"
}
app.config['TYPE_ICONS'] = typeIcons

# ============ DATABASE SETUP ============
with app.app_context():
    db.create_all()
    
    try:
        from sqlalchemy import inspect, text
        
        inspector = inspect(db.engine)
        
        # ========== REPORT TABLE MIGRATIONS ==========
        columns = [col['name'] for col in inspector.get_columns('report')]
        
        if 'direction' not in columns:
            print("Adding 'direction' column to report table...")
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE report ADD COLUMN direction VARCHAR(20)'))
                conn.commit()
            print("Direction column added successfully")
        else:
            print("Direction column already exists")
        
        if 'flag_count' not in columns:
            print("Adding 'flag_count' column to report table...")
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE report ADD COLUMN flag_count INTEGER DEFAULT 0'))
                conn.commit()
            print("flag_count column added successfully")
        
        if 'reviewed' not in columns:
            print("Adding 'reviewed' column to report table...")
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE report ADD COLUMN reviewed BOOLEAN DEFAULT 0'))
                conn.commit()
            print("reviewed column added successfully")
        
        # ========== BROADCAST TABLE MIGRATIONS ==========
        try:
            broadcast_columns = [col['name'] for col in inspector.get_columns('broadcast')]
            if 'direction' not in broadcast_columns:
                print("Adding 'direction' column to broadcast table...")
                with db.engine.connect() as conn:
                    conn.execute(text('ALTER TABLE broadcast ADD COLUMN direction VARCHAR(20) DEFAULT "both"'))
                    conn.commit()
                print("Direction column added to broadcast table")
        except Exception as broadcast_error:
            print(f"Broadcast table note: {broadcast_error}")
        
        # ========== ACTIVITY_LOG TABLE MIGRATIONS (ADD THIS) ==========
        try:
            activity_columns = [col['name'] for col in inspector.get_columns('activity_log')]
            
            if 'is_flagged' not in activity_columns:
                print("Adding flag columns to activity_log table...")
                with db.engine.connect() as conn:
                    conn.execute(text('ALTER TABLE activity_log ADD COLUMN is_flagged BOOLEAN DEFAULT 0'))
                    conn.execute(text('ALTER TABLE activity_log ADD COLUMN flag_reason VARCHAR(500)'))
                    conn.execute(text('ALTER TABLE activity_log ADD COLUMN flagged_at DATETIME'))
                    conn.execute(text('ALTER TABLE activity_log ADD COLUMN admin_review_notes TEXT'))
                    conn.execute(text('ALTER TABLE activity_log ADD COLUMN reviewed_by VARCHAR(100)'))
                    conn.execute(text('ALTER TABLE activity_log ADD COLUMN reviewed_at DATETIME'))
                    conn.commit()
                print("✅ Flag columns added to activity_log table")
            else:
                print("Flag columns already exist in activity_log")
        except Exception as flag_error:
            print(f"Activity log migration note: {flag_error}")
            
    except Exception as e:
        print(f"Note: {e}")

# ============ DEBUG ROUTES ============
@app.route('/debug/raw-prediction/<station_name>/<direction>')
def raw_prediction(station_name, direction):
    """Get raw prediction without any wrapper logic"""
    from services import get_feature_sequence_for_station
    import numpy as np
    
    model_key = f"{station_name}_{direction}"
    
    if model_key not in directional_models_cached:
        return jsonify({'error': f'Model {model_key} not found'})
    
    try:
        now = datetime.now()
        # Get the sequence
        sequence = get_feature_sequence_for_station(station_name, direction, now)
        
        if sequence is None:
            return jsonify({'error': 'Could not generate feature sequence'})
        
        # Get scalers
        feature_scaler = directional_scalers_cached.get(f'{model_key}_feature')
        target_scaler = directional_scalers_cached.get(f'{model_key}_target')
        
        if not feature_scaler or not target_scaler:
            return jsonify({'error': 'Scalers not found'})
        
        # Correct reshaping
        print(f"Original sequence shape: {sequence.shape}")
        
        # Reshape to 2D for scaler: (24, 29) -> (24, 29) actually stays same
        scaled_sequence = feature_scaler.transform(sequence)
        
        # Reshape to 3D for LSTM: (24, 29) -> (1, 24, 29)
        scaled_sequence = scaled_sequence.reshape(1, 24, -1)
        
        # Predict
        pred_scaled = directional_models_cached[model_key].predict(scaled_sequence, verbose=0)
        pred_real = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
        
        return jsonify({
            'station': station_name,
            'direction': direction,
            'prediction': round(pred_real, 1),
            'sequence_shape': sequence.shape,
            'timestamp': now.isoformat(),
            'success': True
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@app.route('/debug/feature-sequence-test/<station>/<direction>')
def test_feature_sequence(station, direction):
    """Test if feature sequence generation is working"""
    from services.feature_engineering import get_feature_sequence_for_station
    from datetime import datetime
    from urllib.parse import unquote
    
    # Decode the URL-encoded station name (convert %20 back to space)
    station = unquote(station)
    direction = unquote(direction)
    
    now = datetime.now()
    sequence = get_feature_sequence_for_station(station, direction, now)
    
    result = {
        "station": station,
        "direction": direction,
        "target_time": now.isoformat(),
        "sequence_is_none": sequence is None,
        "sequence_shape": sequence.shape if sequence is not None else None,
        "current_working_directory": os.getcwd(),
        "data_file_exists": os.path.exists('data (2022-2024)/2025.csv'),
        "alternative_path_exists": os.path.exists('../data (2022-2024)/2025.csv'),
        "models_loaded": len(directional_models_cached)
    }
    
    return jsonify(result)

@app.route('/debug/test-real-model/<station_name>')
def test_real_model(station_name):
    """Test if real models are being used"""
    from services import get_feature_sequence_for_station
    import numpy as np
    
    results = {}
    now = datetime.now()
    
    for direction in ['Northbound', 'Southbound']:
        model_key = f"{station_name}_{direction}"
        
        if model_key in directional_models_cached:
            try:
                # Get a real sequence - this returns (24, 29) shape
                sequence = get_feature_sequence_for_station(station_name, direction, now)
                
                if sequence is not None and sequence.shape == (24, 29):
                    feature_scaler = directional_scalers_cached.get(f'{model_key}_feature')
                    target_scaler = directional_scalers_cached.get(f'{model_key}_target')
                    
                    if feature_scaler and target_scaler:
                        # CORRECT RESHAPING: sequence is already (24, 29)
                        # Scale the 2D array directly
                        scaled_sequence = feature_scaler.transform(sequence)  # This works with (24, 29)
                        # Reshape to (1, 24, 29) for LSTM
                        scaled_sequence = scaled_sequence.reshape(1, 24, -1)
                        
                        pred_scaled = directional_models_cached[model_key].predict(scaled_sequence, verbose=0)
                        pred_real = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                        
                        results[direction] = {
                            'prediction': round(pred_real, 1),
                            'model_key': model_key,
                            'sequence_shape': sequence.shape,
                            'using_real_model': True
                        }
                    else:
                        results[direction] = {'error': 'Missing scaler'}
                else:
                    results[direction] = {'error': f'Invalid sequence shape: {sequence.shape if sequence is not None else None}'}
            except Exception as e:
                results[direction] = {'error': str(e)}
                import traceback
                results[direction]['traceback'] = traceback.format_exc()
        else:
            results[direction] = {'error': f'Model {model_key} not found'}
    
    return jsonify({
        'station': station_name,
        'time': now.isoformat(),
        'predictions': results,
        'total_models_loaded': len(directional_models_cached)
    })
    
@app.route('/debug/test-forecast/<station_name>')
def test_forecast(station_name):
    """Test forecast predictions for different hours"""
    from services import get_feature_sequence_for_station
    from datetime import datetime, timedelta
    
    results = []
    now = datetime.now()
    
    # Test current hour and next 5 hours
    for i in range(6):
        forecast_time = now + timedelta(hours=i)
        
        # Get predictions for both directions
        north_pred = None
        south_pred = None
        
        model_key_north = f"{station_name}_Northbound"
        model_key_south = f"{station_name}_Southbound"
        
        if model_key_north in directional_models_cached:
            try:
                sequence = get_feature_sequence_for_station(station_name, 'Northbound', forecast_time)
                if sequence is not None:
                    feature_scaler = directional_scalers_cached.get(f'{model_key_north}_feature')
                    target_scaler = directional_scalers_cached.get(f'{model_key_north}_target')
                    if feature_scaler and target_scaler:
                        scaled_sequence = feature_scaler.transform(sequence)
                        scaled_sequence = scaled_sequence.reshape(1, 24, -1)
                        pred_scaled = directional_models_cached[model_key_north].predict(scaled_sequence, verbose=0)
                        north_pred = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
            except Exception as e:
                north_pred = f"Error: {e}"
        
        if model_key_south in directional_models_cached:
            try:
                sequence = get_feature_sequence_for_station(station_name, 'Southbound', forecast_time)
                if sequence is not None:
                    feature_scaler = directional_scalers_cached.get(f'{model_key_south}_feature')
                    target_scaler = directional_scalers_cached.get(f'{model_key_south}_target')
                    if feature_scaler and target_scaler:
                        scaled_sequence = feature_scaler.transform(sequence)
                        scaled_sequence = scaled_sequence.reshape(1, 24, -1)
                        pred_scaled = directional_models_cached[model_key_south].predict(scaled_sequence, verbose=0)
                        south_pred = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
            except Exception as e:
                south_pred = f"Error: {e}"
        
        results.append({
            'hour': forecast_time.hour,
            'time': forecast_time.strftime('%Y-%m-%d %H:%M:%S'),
            'northbound': round(north_pred, 1) if north_pred else None,
            'southbound': round(south_pred, 1) if south_pred else None
        })
    
    return jsonify({
        'station': station_name,
        'current_time': now.strftime('%Y-%m-%d %H:%M:%S'),
        'forecasts': results
    })
    
@app.route('/debug/routes')
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(str(rule))
    return jsonify(sorted(routes))

@app.route('/debug/historical-all')
def debug_historical_all():
    return jsonify({
        'historical_entry': historical_entry,
        'historical_exit': historical_exit,
        'hourly_avg_entry': hourly_avg_entry,
        'hourly_avg_exit': hourly_avg_exit
    })

@app.route('/debug/db-schema')
def debug_db_schema():
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    
    tables = inspector.get_table_names()
    result = {}
    for table in tables:
        columns = []
        for col in inspector.get_columns(table):
            columns.append({
                'name': col['name'],
                'type': str(col['type']),
                'nullable': col['nullable'],
                'default': str(col['default']) if col['default'] else None
            })
        result[table] = columns
    
    return jsonify(result)

@app.route('/debug/latest-report-images')
def debug_latest_report_images():
    """Check the image paths in the latest report"""
    from models import Report
    import json
    
    # Get the latest report with images
    latest_report = Report.query.filter(
        Report.photo_path.isnot(None),
        Report.photo_path != 'null',
        Report.photo_path != ''
    ).order_by(Report.id.desc()).first()
    
    if not latest_report:
        return jsonify({'error': 'No reports with images found'})
    
    # Parse photo paths
    photo_paths = []
    if latest_report.photo_path:
        try:
            if isinstance(latest_report.photo_path, str) and latest_report.photo_path.startswith('['):
                photo_paths = json.loads(latest_report.photo_path)
            elif isinstance(latest_report.photo_path, str):
                photo_paths = [latest_report.photo_path]
        except:
            photo_paths = [latest_report.photo_path]
    
    # Generate full URLs
    full_urls = []
    for path in photo_paths:
        if path.startswith('/'):
            full_url = f"http://localhost:5000{path}"
        else:
            full_url = f"http://localhost:5000/uploads/reports/{path}"
        full_urls.append(full_url)
    
    return jsonify({
        'report_id': latest_report.id,
        'station': latest_report.station,
        'timestamp': latest_report.timestamp.isoformat(),
        'photo_paths_stored': photo_paths,
        'full_urls': full_urls,
        'raw_photo_path': latest_report.photo_path
    })
   
   
@app.route('/debug/image-test')
def debug_image_test():
    import os
    from flask import current_app
    
    # Use the same path as the upload route
    upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', 'reports')
    
    # Check if folder exists
    folder_exists = os.path.exists(upload_folder)
    
    # List files in the folder
    files = []
    if folder_exists:
        try:
            files = os.listdir(upload_folder)
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({
        'upload_folder': upload_folder,
        'folder_exists': folder_exists,
        'files': files[:20],  # Show first 20 files
        'file_count': len(files),
        'current_directory': os.getcwd(),
        'root_path': current_app.root_path,
        'route_working': True
    })
    
@app.route('/debug/folder-structure')
def debug_folder_structure():
    import os
    from flask import current_app
    
    root = current_app.root_path
    results = {
        'root_path': root,
        'current_directory': os.getcwd(),
        'static_exists': os.path.exists(os.path.join(root, 'static')),
        'static_uploads_exists': os.path.exists(os.path.join(root, 'static', 'uploads')),
        'static_uploads_reports_exists': os.path.exists(os.path.join(root, 'static', 'uploads', 'reports')),
        'uploads_exists': os.path.exists(os.path.join(root, 'uploads')),
        'uploads_reports_exists': os.path.exists(os.path.join(root, 'uploads', 'reports')),
    }
    
    # If reports folder exists, list files
    reports_folder = os.path.join(root, 'static', 'uploads', 'reports')
    if results['static_uploads_reports_exists']:
        results['files_in_static_reports'] = os.listdir(reports_folder)[:20]
        results['static_file_count'] = len(os.listdir(reports_folder))
    
    # Also check alternate location
    alt_reports = os.path.join(root, 'uploads', 'reports')
    if results['uploads_reports_exists']:
        results['files_in_uploads_reports'] = os.listdir(alt_reports)[:20]
        results['alt_file_count'] = len(os.listdir(alt_reports))
    
    # Check if the specific file from your report exists
    test_file = '20260528_160258_screencapture-localhost-5000-operator-dashboard-2026-05-27-13_49_02.png'
    test_paths = [
        os.path.join(root, 'static', 'uploads', 'reports', test_file),
        os.path.join(root, 'uploads', 'reports', test_file),
        os.path.join(os.getcwd(), 'static', 'uploads', 'reports', test_file),
        os.path.join(os.getcwd(), 'uploads', 'reports', test_file),
    ]
    
    results['test_file_search'] = {}
    for path in test_paths:
        results['test_file_search'][path] = os.path.exists(path)
    
    return jsonify(results)


 

@app.route('/debug/google-config')
def debug_google_config():
    return jsonify({
        'GOOGLE_CLIENT_ID': app.config.get('GOOGLE_CLIENT_ID', 'NOT SET'),
        'GOOGLE_CLIENT_SECRET': 'SET' if app.config.get('GOOGLE_CLIENT_SECRET') else 'NOT SET',
        'GOOGLE_CLIENT_IN_CONFIG': 'GOOGLE_CLIENT' in app.config,
    })
    
@app.route('/debug/model-status')
def debug_model_status():
    return jsonify({
        'directional_models_loaded': len(directional_models_cached),
        'stations': STATIONS,
        'models': list(directional_models_cached.keys())[:10]
    })

@app.route('/debug/clear-cache')
def debug_clear_cache():
    """Clear all caches to force fresh loading"""
    cache_files = [get_models_cache_path(), get_historical_cache_path()]
    results = {}
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            results[os.path.basename(cache_file)] = "deleted"
        else:
            results[os.path.basename(cache_file)] = "not found"
    
    return jsonify({
        "status": "Cache cleared",
        "files": results,
        "message": "Restart the app to reload models from source"
    })
    
@app.route('/debug/raw-model-output/<station_name>/<direction>')
def debug_raw_model_output(station_name, direction):
    from services import get_feature_sequence_for_station
    import numpy as np
    from datetime import datetime
    
    now = datetime.now()
    model_key = f"{station_name}_{direction}"
    
    result = {
        'station': station_name,
        'direction': direction,
        'model_key': model_key,
        'model_exists': model_key in directional_models_cached,
        'total_models_loaded': len(directional_models_cached)
    }
    
    if model_key in directional_models_cached:
        try:
            sequence = get_feature_sequence_for_station(station_name, direction, now)
            
            if sequence is not None and len(sequence) == 24:
                feature_scaler = directional_scalers_cached.get(f'{model_key}_feature')
                target_scaler = directional_scalers_cached.get(f'{model_key}_target')
                
                if feature_scaler:
                    scaled_sequence = feature_scaler.transform(sequence)
                    input_sequence = scaled_sequence.reshape(1, 24, -1)
                    
                    pred_scaled = directional_models_cached[model_key].predict(input_sequence, verbose=0)
                    result['raw_scaled_output'] = float(pred_scaled[0][0])
                    
                    if target_scaler:
                        pred_original = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                        result['after_inverse_transform'] = pred_original
                        
                        result['target_scaler_min'] = float(target_scaler.data_min_[0])
                        result['target_scaler_max'] = float(target_scaler.data_max_[0])
                        result['target_scaler_range'] = result['target_scaler_max'] - result['target_scaler_min']
                        
                        calculated = pred_scaled[0][0] * (result['target_scaler_max'] - result['target_scaler_min']) + result['target_scaler_min']
                        result['calculated_from_scaled'] = float(calculated)
                    else:
                        result['direct_model_output'] = float(pred_scaled[0][0])
                    
                    result['feature_sequence_shape'] = sequence.shape
                    result['feature_scaler_exists'] = feature_scaler is not None
                else:
                    result['error'] = 'No feature scaler found'
            else:
                result['error'] = f'Invalid sequence length: {len(sequence) if sequence is not None else None}'
        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
    else:
        available = [k for k in directional_models_cached.keys() if station_name in k]
        result['available_models_for_station'] = available
        result['all_model_keys_sample'] = list(directional_models_cached.keys())[:10]
    
    return jsonify(result)

@app.route('/debug/test-no-clamp/<station_name>')
def debug_test_no_clamp(station_name):
    from services import get_directional_prediction, get_feature_sequence_for_station
    from datetime import datetime
    
    now = datetime.now()
    results = {}
    
    for direction in ['Northbound', 'Southbound']:
        model_key = f"{station_name}_{direction}"
        
        if model_key in directional_models_cached:
            try:
                sequence = get_feature_sequence_for_station(station_name, direction, now)
                if sequence is not None and len(sequence) == 24:
                    feature_scaler = directional_scalers_cached.get(f'{model_key}_feature')
                    target_scaler = directional_scalers_cached.get(f'{model_key}_target')
                    
                    if feature_scaler:
                        scaled_sequence = feature_scaler.transform(sequence)
                        input_sequence = scaled_sequence.reshape(1, 24, -1)
                        pred_scaled = directional_models_cached[model_key].predict(input_sequence, verbose=0)
                        
                        if target_scaler:
                            raw = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                        else:
                            raw = float(pred_scaled[0][0])
                        
                        results[direction] = {
                            'raw_prediction': round(raw, 2),
                            'model_key': model_key
                        }
                    else:
                        results[direction] = {'error': 'No feature scaler'}
                else:
                    results[direction] = {'error': 'Invalid sequence'}
            except Exception as e:
                results[direction] = {'error': str(e)}
        else:
            results[direction] = {'error': 'Model not found'}
    
    return jsonify({
        'station': station_name,
        'time': now.isoformat(),
        'predictions': results
    })

@app.route('/debug/live-map-test')
def debug_live_map_test():
    from services import get_feature_sequence_for_station
    
    results = {}
    now = datetime.now()
    
    for station in STATIONS[:5]:
        station_results = {}
        for direction in ['Northbound', 'Southbound']:
            model_key = f"{station}_{direction}"
            if model_key in directional_models_cached:
                try:
                    sequence = get_feature_sequence_for_station(station, direction, now)
                    if sequence is not None and len(sequence) == 24:
                        feature_scaler = directional_scalers_cached.get(f'{model_key}_feature')
                        target_scaler = directional_scalers_cached.get(f'{model_key}_target')
                        if feature_scaler and target_scaler:
                            scaled_sequence = feature_scaler.transform(sequence)
                            input_sequence = scaled_sequence.reshape(1, 24, -1)
                            pred_scaled = directional_models_cached[model_key].predict(input_sequence, verbose=0)
                            pred = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                            station_results[direction] = round(pred, 1)
                except Exception as e:
                    station_results[direction] = f"Error: {e}"
            else:
                station_results[direction] = "No model"
        results[station] = station_results
    
    return jsonify({
        'time': now.isoformat(),
        'predictions': results,
        'models_loaded': len(directional_models_cached)
    })
    
@app.route('/debug/simple-prediction/<station_name>')
def debug_simple_prediction(station_name):
    from datetime import datetime
    import numpy as np
    
    now = datetime.now()
    results = {}
    
    for direction in ['Northbound', 'Southbound']:
        model_key = f"{station_name}_{direction}"
        
        if model_key not in directional_models_cached:
            results[direction] = "Model not found"
            continue
        
        try:
            model = directional_models_cached[model_key]
            feature_scaler = directional_scalers_cached.get(f'{model_key}_feature')
            target_scaler = directional_scalers_cached.get(f'{model_key}_target')
            
            if feature_scaler is None or target_scaler is None:
                results[direction] = "Missing scaler"
                continue
            
            input_shape = model.input_shape
            print(f"Model input shape for {model_key}: {input_shape}")
            
            dummy_input = np.zeros((1, 24, 29))
            dummy_scaled = feature_scaler.transform(dummy_input.reshape(-1, 29)).reshape(1, 24, 29)
            dummy_pred = model.predict(dummy_scaled, verbose=0)
            dummy_result = float(target_scaler.inverse_transform(dummy_pred.reshape(-1, 1))[0][0])
            
            results[direction] = {
                "dummy_prediction": round(dummy_result, 1),
                "model_works": True
            }
            
        except Exception as e:
            results[direction] = {"error": str(e)}
    
    return jsonify({
        'station': station_name,
        'time': now.isoformat(),
        'results': results
    })

# ============ LSTM DEBUG ROUTES ============
@app.route('/debug/lstm-status')
def debug_lstm_status():
    """Check LSTM model status"""
    lstm_predictor = app.config.get('LSTM_PREDICTOR')
    
    if not lstm_predictor:
        return jsonify({
            'status': 'not_initialized',
            'message': 'LSTM predictor not initialized'
        })
    
    return jsonify({
        'status': 'ready',
        'models_loaded': len(lstm_predictor.models),
        'station_directions': lstm_predictor.station_directions[:10],  # Show first 10
        'model_path': lstm_predictor.model_path,
        'feature_cols_count': len(lstm_predictor.feature_cols) if lstm_predictor.feature_cols else 0,
        'capacities_loaded': bool(lstm_predictor.capacities)
    })
    
@app.route('/admin/import-csvs', methods=['GET', 'POST'])
def admin_import_csvs():
    """Import CSV files from Google Drive (Render workaround)"""
    import requests
    import os
    import re
    import json
    from datetime import datetime

    SECRET_TOKEN = "mrt3_import_2024"
    
    token = request.args.get('secret') or request.json.get('secret') if request.is_json else None
    if token != SECRET_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'services', 'data (2022-2024)')
    os.makedirs(data_dir, exist_ok=True)

    # Try Google Drive first, fallback to file.io URLs
    # If Google Drive fails, upload your files to https://file.io and put URLs here
    file_sources = {
        '2022.csv': {
            'google': 'https://drive.google.com/uc?export=download&id=1IFMhSnvU6Tps-9AAEmRL3Jn7oDbhdlVA',
            'backup': None  # Add file.io URL here if needed
        },
        '2023.csv': {
            'google': 'https://drive.google.com/uc?export=download&id=14H6zXJxXHMX4kt3-1tkXc0gUH066cuF_',
            'backup': None
        },
        '2024.csv': {
            'google': 'https://drive.google.com/uc?export=download&id=1xDbrMdTomXkrGQ5i54FBDE6yEWrXAO1N',
            'backup': None
        },
    }

    results = {}

    for filename, sources in file_sources.items():
        filepath = os.path.join(data_dir, filename)
        downloaded = False
        
        # Try Google Drive
        try:
            print(f"📥 Downloading {filename} from Google Drive...")
            
            session = requests.Session()
            response = session.get(sources['google'], stream=True, timeout=120)
            
            # Check if we got HTML
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                print(f"⚠️ Google Drive returned HTML for {filename}, trying backup...")
                # Try with confirm parameter
                response = session.get(sources['google'] + '&confirm=1', stream=True, timeout=120)
                content_type = response.headers.get('Content-Type', '')
            
            # If still HTML or not CSV, try backup
            if 'text/html' in content_type or 'text/csv' not in content_type:
                if sources['backup']:
                    print(f"🔄 Trying backup URL for {filename}...")
                    response = requests.get(sources['backup'], stream=True, timeout=120)
                    content_type = response.headers.get('Content-Type', '')
            
            # Save if we have a valid file
            if response.status_code == 200 and 'text/html' not in content_type:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                results[filename] = {
                    'status': 'success',
                    'size_mb': round(file_size, 2),
                    'path': filepath
                }
                downloaded = True
                print(f"✅ Downloaded {filename} ({file_size:.2f} MB)")
            else:
                results[filename] = {
                    'status': 'failed',
                    'error': 'Received HTML or invalid response',
                    'content_type': content_type
                }
                print(f"❌ Failed to download {filename}: HTML response")
                
        except Exception as e:
            results[filename] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"❌ Error downloading {filename}: {e}")

    # If all failed, create synthetic data
    if all(r.get('status') != 'success' for r in results.values()):
        print("⚠️ All imports failed, generating synthetic data...")
        try:
            from services.model_loader import _generate_synthetic_historical_data
            from utils import STATIONS, STATION_BASE_CAPACITY
            _generate_synthetic_historical_data(STATIONS, STATION_BASE_CAPACITY)
            return jsonify({
                'success': True,
                'message': 'Generated synthetic data (imports failed)',
                'results': results
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Import failed and synthetic generation failed: {e}',
                'results': results
            })

    # Clear cache so app reloads data
    try:
        cache_files = ['mrt3_historical_cache.pkl', 'historical_data_cache_2023_2024.pkl']
        for cache_file in cache_files:
            cache_path = os.path.join('/tmp', cache_file)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"🗑️ Removed cache: {cache_file}")
    except Exception as e:
        print(f"⚠️ Could not clear cache: {e}")

    return jsonify({
        'success': True,
        'message': 'CSV import completed',
        'results': results,
        'data_directory': data_dir
    })
    
@app.route('/debug/csv-status')
def debug_csv_status():
    """Check if CSV files are loaded"""
    import os
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'services', 'data (2022-2024)')
    
    results = {
        'directory': data_dir,
        'exists': os.path.exists(data_dir),
        'files': {}
    }
    
    if os.path.exists(data_dir):
        for filename in ['2022.csv', '2023.csv', '2024.csv']:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / (1024 * 1024)
                results['files'][filename] = {
                    'exists': True,
                    'size_mb': round(size, 2)
                }
            else:
                results['files'][filename] = {'exists': False}
    
    return jsonify(results)

@app.route('/debug/lstm-predict/<station>/<direction>')
def debug_lstm_predict(station, direction):
    """Test LSTM prediction for a station-direction"""
    from urllib.parse import unquote
    from models import db
    
    station = unquote(station)
    direction = unquote(direction)
    
    lstm_predictor = app.config.get('LSTM_PREDICTOR')
    
    if not lstm_predictor:
        return jsonify({'error': 'LSTM predictor not available'})
    
    try:
        prediction = lstm_predictor.predict_congestion(station, direction, db.session)
        
        return jsonify({
            'station': station,
            'direction': direction,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat(),
            'models_loaded': len(lstm_predictor.models)
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'station': station,
            'direction': direction
        })


@app.route('/debug/find-uploads')
def debug_find_uploads():
    """Find where uploads are actually stored"""
    import os
    from flask import current_app
    
    root = current_app.root_path
    results = {
        'root': root,
        'search_results': {}
    }
    
    # Search for uploads folder
    search_paths = [
        os.path.join(root, 'uploads'),
        os.path.join(root, 'static', 'uploads'),
        os.path.join(root, 'static', 'uploads', 'reports'),
        os.path.join(root, 'uploads', 'reports'),
        os.path.join(os.path.dirname(root), 'uploads'),  # One level up
        os.path.join(os.path.dirname(root), 'static', 'uploads'),
    ]
    
    for path in search_paths:
        exists = os.path.exists(path)
        results['search_results'][path] = {
            'exists': exists,
            'contents': os.listdir(path) if exists else []
        }
    
    return jsonify(results)
# ============ MAIN ============
if __name__ == '__main__':
    print("\n" + "="*50)
    print("MRT-3 PREDICTION SYSTEM READY!")
    print("="*50)
    print(f"✓ {len(directional_models_cached)} directional models loaded")
    print(f"✓ {len(historical_data['historical_entry'])} stations with historical data")
    print(f"✓ {len(STATIONS)} total stations configured")
    
    # Check if LSTM models are loaded
    lstm_predictor = app.config.get('LSTM_PREDICTOR')
    if lstm_predictor:
        print(f"✓ {len(lstm_predictor.models)} LSTM models loaded for enhanced predictions")
    else:
        print("⚠️ LSTM models not loaded - using fallback predictions")
    
    print("\n💡 TIP: First load may take 10-30 seconds (loading models)")
    print("💡 Subsequent reloads will take only 1-2 seconds (using cache)")
    print("\n🗑️  To force fresh model loading: visit /debug/clear-cache")
    print("\n🌐 Open http://localhost:5000")
    print("="*50 + "\n")
    
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)