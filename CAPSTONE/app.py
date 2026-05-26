from datetime import datetime, timedelta
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, session, flash, redirect, url_for, jsonify
from dotenv import load_dotenv
import warnings
from authlib.integrations.flask_client import OAuth
import pickle
import tempfile



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
    model_perf_bp 
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

@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# ============ LOAD MODELS AND HISTORICAL DATA WITH CACHING ============
print("\n" + "="*50)
print("LOADING MRT-3 PREDICTION SYSTEM")
print("="*50)

# Load models with caching
DIRECTIONAL_MODELS_PATH = 'models_2022-2024_v4'
directional_models_cached, directional_scalers_cached = load_models_with_cache(STATIONS, DIRECTIONAL_MODELS_PATH)

# Load historical data with caching
historical_data = load_historical_with_cache(STATIONS, STATION_BASE_CAPACITY)

# Update the services module with loaded data so other modules can access them
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

print("="*50 + "\n")

# ============ WRAPPER FUNCTIONS ============
def get_directional_prediction_wrapper(station_name, direction, target_datetime=None):
    return get_directional_prediction(
        station_name, direction, target_datetime,
        directional_models_cached, directional_scalers_cached,
        get_feature_sequence_for_station
    )

def get_station_prediction_wrapper(station_name):
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
        })@app.route('/debug/test-real-model/<station_name>')
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

@app.route('/uploads/reports/<filename>')
def serve_upload(filename):
    from flask import send_from_directory
    upload_folder = os.path.join(os.path.dirname(__file__), 'static', 'uploads', 'reports')
    return send_from_directory(upload_folder, filename)

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

# ============ MAIN ============
if __name__ == '__main__':
    print("\n" + "="*50)
    print("MRT-3 PREDICTION SYSTEM READY!")
    print("="*50)
    print(f"✓ {len(directional_models_cached)} directional models loaded")
    print(f"✓ {len(historical_data['historical_entry'])} stations with historical data")
    print(f"✓ {len(STATIONS)} total stations configured")
    print("\n💡 TIP: First load may take 10-30 seconds (loading models)")
    print("💡 Subsequent reloads will take only 1-2 seconds (using cache)")
    print("\n🗑️  To force fresh model loading: visit /debug/clear-cache")
    print("\n🌐 Open http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5000)