import os
import gc

# Reduce Python memory
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['PYTHONMALLOC'] = 'malloc'

# Limit TensorFlow memorya
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Aggressive garbage collection
gc.set_threshold(50, 3, 3)

import tensorflow as tf
tf.config.run_functions_eagerly(False)
tf.keras.backend.clear_session()

print("✅ Extreme memory optimization applied!")

from flask import Flask, session, flash, redirect, url_for, jsonify, request
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

# ✅ KEEP LSTM IMPORT (needed for retraining)
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
    model_perf_bp, email_bp
)

# ============ CACHE SETUP FOR FAST RELOADS ============
_MODELS_CACHE = {}
_MODELS_CACHE_FILE = None
_MODELS_LOADED = False  # Track if models are loaded
_WARMUP_COMPLETE = False  # Track if models are warmed up


def get_models_cache_path():
    cache_dir = tempfile.gettempdir()
    return os.path.join(cache_dir, 'mrt3_models_cache.pkl')


def get_historical_cache_path():
    cache_dir = tempfile.gettempdir()
    return os.path.join(cache_dir, 'mrt3_historical_cache.pkl')


def load_models_with_cache(stations, models_path):
    global _MODELS_CACHE, _MODELS_LOADED
    
    # If already in memory, return instantly
    if _MODELS_LOADED and _MODELS_CACHE.get('directional_models'):
        print("✓ Using models from RAM")
        return _MODELS_CACHE['directional_models'], _MODELS_CACHE['directional_scalers']
    
    # Load directly from .keras files (no pickle)
    print("🔄 Loading models from .keras files...")
    directional_models, directional_scalers = load_directional_models(stations, models_path)
    
    # Store in memory cache (not pickle)
    _MODELS_CACHE = {
        'directional_models': directional_models,
        'directional_scalers': directional_scalers
    }
    _MODELS_LOADED = True
    
    return directional_models, directional_scalers


def load_historical_with_cache(stations, base_capacity):
    cache_file = get_historical_cache_path()
    
    if os.path.exists(cache_file):
        try:
            print("📦 Loading historical data from cache...")
            with open(cache_file, 'rb') as f:
                historical_data = pickle.load(f)
            print(f"✓ Loaded historical data for {len(historical_data['historical_entry'])} stations from cache")
            return historical_data
        except Exception as e:
            print(f"Historical cache load failed: {e}, reloading from source...")
    
    print("🔄 Loading historical data from source (first time only)...")
    historical_data = load_real_historical_data(stations, base_capacity)
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(historical_data, f)
        print(f"✓ Historical data cached")
    except Exception as e:
        print(f"Historical cache save failed: {e}")
    
    return historical_data


# ============ MODEL WARMUP (ELIMINATE COLD-START LATENCY) ============
# ============ MODEL WARMUP (ELIMINATE COLD-START LATENCY) ============
def warmup_all_models():
    """
    🔥 CRITICAL: Warms up all 26 models to eliminate cold-start latency.
    Forces TensorFlow graph compilation at startup.
    """
    global directional_models_cached, _MODELS_LOADED, _WARMUP_COMPLETE
    
    if not _MODELS_LOADED or not directional_models_cached:
        print("⚠️ Models not loaded yet! Call preload_all_models() first.")
        return False
    
    if _WARMUP_COMPLETE:
        print("✅ Models already warmed up!")
        return True
    
    print("\n" + "="*60)
    print("🔥 WARMING UP ALL 26 MODELS (Building TensorFlow graphs)...")
    print("="*60)
    
    import time
    import numpy as np
    start_time = time.time()
    
    dummy_input = np.zeros((1, 24, 16), dtype=np.float32)

    successful = 0
    failed = 0
    total = len(directional_models_cached)

    for idx, (model_key, model) in enumerate(directional_models_cached.items(), 1):
        try:
            # Fast tensor call
            _ = model(dummy_input, training=False).numpy()
            successful += 1
        except Exception:
            # Fallback call
            try:
                _ = model.predict(dummy_input, verbose=0)
                successful += 1
            except Exception as e2:
                failed += 1
                print(f"  ⚠️ Failed to warmup {model_key}: {e2}")
        
        if idx % 5 == 0 or idx == total:
            print(f"  ⏳ Warmup progress: {idx}/{total} models")
    
    elapsed = time.time() - start_time
    gc.collect()
    
    print("="*60)
    print(f"✅ WARMUP COMPLETE in {elapsed:.2f} seconds")
    print(f"   ✅ {successful} models warmed up successfully")
    if failed > 0:
        print(f"   ⚠️ {failed} models failed to warmup")
    print("="*60 + "\n")
    
    app.config['WARMUP_STATS'] = {
        'successful': successful,
        'failed': failed,
        'duration_seconds': round(elapsed, 2),
        'models_warmed': successful,
        'total_models': total
    }
    
    _WARMUP_COMPLETE = True
    return True


# ============ CSV IMPORT FUNCTION ============
def import_csv_files():
    import requests
    import os
    import json
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'services', 'data (2022-2024)')
    os.makedirs(data_dir, exist_ok=True)
    
    file_sources = {
        '2022.csv': {
            'google': 'https://drive.google.com/uc?export=download&id=1IFMhSnvU6Tps-9AAEmRL3Jn7oDbhdlVA',
            'backup': None
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
        try:
            print(f"📥 Downloading {filename} from Google Drive...")
            session = requests.Session()
            response = session.get(sources['google'], stream=True, timeout=120)
            
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                response = session.get(sources['google'] + '&confirm=1', stream=True, timeout=120)
                content_type = response.headers.get('Content-Type', '')
            
            if response.status_code == 200 and 'text/html' not in content_type:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                results[filename] = {'status': 'success', 'size_mb': round(file_size, 2)}
                print(f"✅ Downloaded {filename} ({file_size:.2f} MB)")
            else:
                results[filename] = {'status': 'failed', 'error': 'Invalid response'}
                print(f"❌ Failed to download {filename}")
        except Exception as e:
            results[filename] = {'status': 'failed', 'error': str(e)}
            print(f"❌ Error downloading {filename}: {e}")
    
    return results


# ============ APP INITIALIZATION ============
# ============ APP INITIALIZATION ============
app = Flask(__name__, template_folder='html', static_folder='static')
app.config.from_object(Config)

# ✅ Configure cache properly
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300
app.config['CACHE_THRESHOLD'] = 1000

# Initialize cache
cache.init_app(app)

@app.route('/warmup')
def warmup():
    """Warm up the app by loading ALL models and data at startup"""
    import time
    start = time.time()
    
    try:
        # Load ALL models at once
        preload_all_models()
        
        elapsed = time.time() - start
        
        return jsonify({
            "status": "warmup complete",
            "models_loaded": len(directional_models_cached) if directional_models_cached else 0,
            "elapsed_seconds": round(elapsed, 2),
            "memory_mb": get_memory_usage(),
            "warmed_up": _WARMUP_COMPLETE
        })
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500


@app.route('/debug/warmup-status')
def debug_warmup_status():
    """Check if models are warmed up and ready for instant predictions"""
    warmup_stats = app.config.get('WARMUP_STATS', {})
    models_loaded = len(directional_models_cached) if directional_models_cached else 0
    
    return jsonify({
        'warmup_complete': _WARMUP_COMPLETE,
        'total_models': models_loaded,
        'models_warmed': warmup_stats.get('successful', 0),
        'warmup_failures': warmup_stats.get('failed', 0),
        'warmup_duration_seconds': warmup_stats.get('duration_seconds', 0),
        'status': 'ready' if (_WARMUP_COMPLETE and models_loaded > 0) else 'warming_up',
        'memory_mb': get_memory_usage(),
        'message': 'All models are warmed up and ready for instant predictions!' if _WARMUP_COMPLETE else 'Models are loading...'
    })


def get_memory_usage():
    """Helper to get memory usage"""
    try:
        import psutil
        process = psutil.Process()
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except:
        return 0
    
    
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
    
    upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', 'reports')
    file_path = os.path.join(upload_folder, filename)
    if not os.path.exists(file_path):
        abort(404)
    return send_from_directory(upload_folder, filename)

@app.route('/uploads/reports/<path:filename>')
def serve_upload_with_path(filename):
    from flask import send_from_directory, abort, current_app
    import os
    
    upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', 'reports')
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

# ============ PRELOAD MODELS AT STARTUP ============
print("\n" + "="*50)
print("🚀 MRT-3 PREDICTION SYSTEM - PRELOADING MODELS")
print("="*50)
print("⏳ Loading models at startup (this is the slow part)...")
print("💡 This takes 15-30 seconds ONCE, then all predictions are instant")
print("💡 Your professor will see FAST predictions on first click")
print("="*50 + "\n")

DIRECTIONAL_MODELS_PATH = 'models_2022-2024_v10'

# Global variables - will be loaded at startup
directional_models_cached = {}
directional_scalers_cached = {}
_MODELS_LOADED = False
_WARMUP_COMPLETE = False
historical_data = None

import threading
_MODEL_LOAD_LOCK = threading.Lock()

def ensure_models_loaded(station_name=None, direction=None):
    """Ensure models are loaded - now just returns since we preload at startup"""
    global directional_models_cached, directional_scalers_cached, _MODELS_LOADED, historical_data
    
    # Should already be loaded from startup, but check just in case
    if _MODELS_LOADED and directional_models_cached:
        return
    
    # Fallback: load if not loaded at startup (shouldn't happen)
    print("\n" + "="*60)
    print("⚠️ FALLBACK: Loading models on first request...")
    print("⏳ This should NOT happen if startup preload worked")
    print("="*60)
    
    import time
    start = time.time()
    
    # Load models from cache or disk
    directional_models_cached, directional_scalers_cached = load_models_with_cache(
        STATIONS, DIRECTIONAL_MODELS_PATH
    )
    
    # Load historical data
    historical_data = load_historical_with_cache(STATIONS, STATION_BASE_CAPACITY)
    
    # Update services module
    import services
    services.directional_models = directional_models_cached
    services.directional_scalers = directional_scalers_cached
    services.historical_entry = historical_data.get('historical_entry', {})
    services.historical_exit = historical_data.get('historical_exit', {})
    services.hourly_avg_entry = historical_data.get('hourly_avg_entry', {})
    services.hourly_avg_exit = historical_data.get('hourly_avg_exit', {})
    
    # Store in app config
    app.config['DIRECTIONAL_MODELS'] = directional_models_cached
    app.config['DIRECTIONAL_SCALERS'] = directional_scalers_cached
    app.config['HISTORICAL_DATA'] = historical_data
    
    _MODELS_LOADED = True
    elapsed = time.time() - start
    
    print("="*60)
    print(f"✅ FALLBACK: MODELS LOADED in {elapsed:.1f} seconds")
    print(f"✅ {len(directional_models_cached)} directional models ready")
    print("="*60 + "\n")

# Register the loader
app.config['ENSURE_MODELS_LOADED'] = ensure_models_loaded

# ============ SINGLE MODEL LOADER ============
def ensure_single_model_loaded(station_name, direction):
    """Ensure models are loaded - calls the main loader"""
    ensure_models_loaded()

app.config['ENSURE_SINGLE_MODEL_LOADED'] = ensure_single_model_loaded

# ============ ACTUALLY LOAD MODELS AT STARTUP ============
print("🔄 Starting model preload...")
with app.app_context():
    try:
        # Load models at startup
        directional_models_cached, directional_scalers_cached = load_models_with_cache(
            STATIONS, DIRECTIONAL_MODELS_PATH
        )
        
        # Load historical data
        historical_data = load_historical_with_cache(STATIONS, STATION_BASE_CAPACITY)
        
        # Update services module
        import services
        services.directional_models = directional_models_cached
        services.directional_scalers = directional_scalers_cached
        services.historical_entry = historical_data.get('historical_entry', {})
        services.historical_exit = historical_data.get('historical_exit', {})
        services.hourly_avg_entry = historical_data.get('hourly_avg_entry', {})
        services.hourly_avg_exit = historical_data.get('hourly_avg_exit', {})
        
        # Store in app config
        app.config['DIRECTIONAL_MODELS'] = directional_models_cached
        app.config['DIRECTIONAL_SCALERS'] = directional_scalers_cached
        app.config['HISTORICAL_DATA'] = historical_data
        
        _MODELS_LOADED = True
        
        print(f"\n✅ MODELS LOADED at startup!")
        print(f"   📊 {len(directional_models_cached)} directional models loaded")
        print(f"   📊 {len(historical_data['historical_entry'])} stations historical data")
        
        # NOW WARM THEM UP FOR INSTANT PREDICTIONS
        warmup_all_models()
       
      
        # ✅ ADD THIS: Preload all station patterns (LOAD ALL 26 PARQUET FILES AND PATTERNS)
        try:
            from services.feature_engineering import preload_all_station_patterns
            preload_all_station_patterns()
            print("   📊 All station patterns preloaded")
        except Exception as e:
            print(f"   ⚠️ Pattern preload skipped: {e}")
        
        # ✅ Load correction factors
        try:
            from routes.api_predict import load_correction_factors
            load_correction_factors()
            print("   📊 Correction factors loaded")
        except Exception as e:
            print(f"   ⚠️ Correction factors load skipped: {e}")
        
        # ✅ Preload P95 cache from disk
        try:
            from routes.api_predict import preload_p95_cache
            preload_p95_cache()
            print("   📊 P95 cache preloaded from disk")
        except Exception as e:
            print(f"   ⚠️ P95 preload skipped: {e}")
         
    except Exception as e:
        print(f"⚠️ Startup load failed: {e}")
        print("   Models will load on first request instead (fallback)")
        import traceback
        traceback.print_exc()
# ============ LSTM MODELS - LAZY LOADING (ONLY FOR RETRAINING) ============
print("\n" + "="*50)
print("LSTM MODELS - LAZY LOADING (Retraining Only)")
print("="*50)
print("⏳ LSTM models will load only when retraining is triggered")
print("💡 Visit /admin/retrain to trigger retraining")
print("="*50 + "\n")

LSTM_MODEL_PATH = 'models_2022-2024_v10'
lstm_predictor = None

def ensure_lstm_for_retraining():
    """Load LSTM models ONLY when retraining is triggered"""
    global lstm_predictor
    
    if lstm_predictor is not None:
        return
    
    print("🔄 Loading LSTM models for retraining...")
    lstm_predictor = MRT3LSTMPredictor(model_path=LSTM_MODEL_PATH)
    
    if lstm_predictor.load_models():
        print(f"✅ LSTM models loaded successfully!")
        print(f"   📊 Loaded {len(lstm_predictor.models)} station-direction models")
        app.config['LSTM_PREDICTOR'] = lstm_predictor
    else:
        print("⚠️ LSTM models not loaded - retraining will be disabled")
        app.config['LSTM_PREDICTOR'] = None

def preload_all_models():
    """Preload ALL 26 models - ONLY call this if you want to force-load."""
    global directional_models_cached, directional_scalers_cached, _MODELS_LOADED
    
    if _MODELS_LOADED and directional_models_cached is not None:
        print(f"✅ All models already loaded! ({len(directional_models_cached)}/26)")
        return directional_models_cached, directional_scalers_cached
    
    print("\n" + "="*60)
    print("🔄 PRELOADING ALL 26 MODELS...")
    print("="*60)
    
    directional_models_cached, directional_scalers_cached = load_models_with_cache(STATIONS, DIRECTIONAL_MODELS_PATH)
    
    global historical_data
    historical_data = load_historical_with_cache(STATIONS, STATION_BASE_CAPACITY)
    
    import services
    services.directional_models = directional_models_cached
    services.directional_scalers = directional_scalers_cached
    services.historical_entry = historical_data.get('historical_entry', {})
    services.historical_exit = historical_data.get('historical_exit', {})
    services.hourly_avg_entry = historical_data.get('hourly_avg_entry', {})
    services.hourly_avg_exit = historical_data.get('hourly_avg_exit', {})
    
    app.config['DIRECTIONAL_MODELS'] = directional_models_cached
    app.config['DIRECTIONAL_SCALERS'] = directional_scalers_cached
    app.config['HISTORICAL_DATA'] = historical_data
    
    # ✅ Add pattern preload here too
    try:
        from services.feature_engineering import preload_all_station_patterns
        preload_all_station_patterns()
        print("   📊 All station patterns preloaded")
    except Exception as e:
        print(f"   ⚠️ Pattern preload skipped: {e}")
    
    try:
        from routes.api_predict import preload_p95_cache, preload_typical_patterns, load_correction_factors
        print("\n📊 Preloading P95 cache and typical patterns...")
        preload_p95_cache()
        preload_typical_patterns()
        load_correction_factors()
        print("   📊 Correction factors loaded")
    except Exception as e:
        print(f"⚠️ Error preloading P95/typical patterns: {e}")
    
    _MODELS_LOADED = True
    
    print("="*60)
    print(f"✅ All models loaded! ({len(directional_models_cached)}/26 models)")
    print(f"✅ Historical data loaded for {len(historical_data['historical_entry'])} stations")
    print("="*60 + "\n")
    
    return directional_models_cached, directional_scalers_cached

@app.route('/debug/app-config')
def debug_app_config():
    return jsonify({
        'ENSURE_SINGLE_MODEL_LOADED': app.config.get('ENSURE_SINGLE_MODEL_LOADED') is not None,
        'DIRECTIONAL_MODELS': app.config.get('DIRECTIONAL_MODELS') is not None,
        'DIRECTIONAL_SCALERS': app.config.get('DIRECTIONAL_SCALERS') is not None,
        'models_loaded': len(directional_models_cached) if directional_models_cached else 0
    })
    
# ============ WRAPPER FUNCTIONS ============
def get_directional_prediction_wrapper(station_name, direction, target_datetime=None):
    """
    Wrapper that uses the main prediction API only.
    This ensures consistency with the live map and all other endpoints.
    """
    from routes.api_predict import get_directional_prediction
    return get_directional_prediction(station_name, direction, target_datetime)

    

def get_station_prediction_wrapper(station_name):
    """
    Wrapper that uses the main prediction API only.
    """
    from routes.api_predict import get_directional_prediction
    from config import Config
    
    now = Config.get_current_time()
    north = get_directional_prediction(station_name, 'Northbound', now)
    south = get_directional_prediction(station_name, 'Southbound', now)
    
    if north is None and south is None:
        return 50
    
    if north is None:
        return south
    if south is None:
        return north
    
    return (north + south) / 2

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

    # ========== AUTO-IMPORT CSV FILES (RENDER FIX) ==========
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'services', 'data (2022-2024)')
    csv_files = ['2022.csv', '2023.csv', '2024.csv']

    all_present = all(os.path.exists(os.path.join(data_dir, f)) for f in csv_files)

    if all_present:
        print("✅ All CSV files present! (not loading into memory)")
    else:
        print(f"⚠️ Some CSV files missing: {[f for f in csv_files if not os.path.exists(os.path.join(data_dir, f))]}")
        print("🔄 Auto-importing CSV files from Google Drive...")
        results = import_csv_files()  # ✅ CALL the import function!
        
        # Verify after import
        all_present_now = all(os.path.exists(os.path.join(data_dir, f)) for f in csv_files)
        if all_present_now:
            print("✅ CSV files imported successfully!")
        else:
            print("⚠️ Some CSV files still missing!")
    # ===========================================

# ============ ADMIN RETRAINING ENDPOINT ============
@app.route('/admin/retrain', methods=['POST'])
def admin_retrain():
    """Manually trigger weekly retraining - loads LSTM models temporarily"""
    try:
        print("🔄 Starting weekly retraining...")
        
        # Load LSTM models only when retraining
        ensure_lstm_for_retraining()
        
        lstm_predictor = app.config.get('LSTM_PREDICTOR')
        if not lstm_predictor:
            return jsonify({"status": "failed", "error": "LSTM models not loaded"}), 400
        
        # Start retraining in background
        try:
            schedule_weekly_retraining(app)
            return jsonify({
                "status": "success",
                "message": "Weekly retraining started! LSTM models loaded."
            })
        except Exception as e:
            return jsonify({"status": "failed", "error": str(e)}), 500
            
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

# ============ LSTM STATUS DEBUG ROUTE ============
@app.route('/debug/lstm-status')
def debug_lstm_status():
    """Check if LSTM models are loaded"""
    lstm_predictor = app.config.get('LSTM_PREDICTOR')
    
    if not lstm_predictor:
        return jsonify({
            'status': 'not_loaded',
            'message': 'LSTM models not loaded. They load only when retraining is triggered.',
            'models_loaded': 0
        })
    
    return jsonify({
        'status': 'loaded',
        'models_loaded': len(lstm_predictor.models) if hasattr(lstm_predictor, 'models') else 0,
        'station_directions': lstm_predictor.station_directions[:10] if hasattr(lstm_predictor, 'station_directions') else [],
        'model_path': lstm_predictor.model_path if hasattr(lstm_predictor, 'model_path') else None
    })

@app.route('/debug/raw-prediction/<station_name>/<direction>')
def raw_prediction(station_name, direction):
    from services import get_feature_sequence_for_station
    from routes.api_predict import get_p95_percentile
    import numpy as np
    
    ensure_single_model_loaded(station_name, direction)
    
    model_key = f"{station_name}_{direction}"
    
    if model_key not in directional_models_cached:
        return jsonify({'error': f'Model {model_key} not found'})
    
    try:
        now = Config.get_current_time()
        sequence = get_feature_sequence_for_station(station_name, direction, now)
        
        if sequence is None:
            return jsonify({'error': 'Could not generate feature sequence'})
        
        # ✅ FIX: get_feature_sequence_for_station() already returns scaled features
        # Do NOT transform again!
        target_scaler = directional_scalers_cached.get(f'{model_key}_target')
        
        if not target_scaler:
            return jsonify({'error': 'Target scaler not found'})
        
        # sequence is already scaled (24, 16)
        scaled_sequence = sequence.reshape(1, 24, -1)
        
        input_tensor = tf.convert_to_tensor(scaled_sequence, dtype=tf.float32)
        pred_scaled = directional_models_cached[model_key](input_tensor, training=False).numpy()
        pred_passengers = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
        
        # ✅ Get P95 for congestion
        p95 = get_p95_percentile(station_name, direction)
        pred_congestion = (pred_passengers / p95) * 100
        pred_congestion = max(0, min(100, pred_congestion))
        
        return jsonify({
            'station': station_name,
            'direction': direction,
            'predicted_passengers': round(pred_passengers, 1),
            'predicted_congestion': round(pred_congestion, 1),
            'p95_percentile': round(p95, 0),
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
    from services.feature_engineering import get_feature_sequence_for_station
    from urllib.parse import unquote
    
    station = unquote(station)
    direction = unquote(direction)
    
    now = Config.get_current_time()
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
        "models_loaded": len(directional_models_cached) if directional_models_cached else 0
    }
    
    return jsonify(result)
@app.route('/debug/test-real-model/<station_name>')
def test_real_model(station_name):
    from services import get_feature_sequence_for_station
    import numpy as np
    
    results = {}
    now = Config.get_current_time()
    
    for direction in ['Northbound', 'Southbound']:
        ensure_single_model_loaded(station_name, direction)
        model_key = f"{station_name}_{direction}"
        
        if model_key in directional_models_cached:
            try:
                sequence = get_feature_sequence_for_station(station_name, direction, now)
                
                # ✅ FIX: V10 models use 24x16, not 29
                if sequence is not None and sequence.shape == (24, 16):
                    target_scaler = directional_scalers_cached.get(f'{model_key}_target')
                    
                    if target_scaler:
                        # sequence is already scaled
                        scaled_sequence = sequence.reshape(1, 24, -1)
                        
                        input_tensor = tf.convert_to_tensor(scaled_sequence, dtype=tf.float32)
                        pred_scaled = directional_models_cached[model_key](input_tensor, training=False).numpy()
                        pred_real = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                        results[direction] = {
                            'prediction': round(pred_real, 1),
                            'model_key': model_key,
                            'sequence_shape': sequence.shape,
                            'using_real_model': True
                        }
                    else:
                        results[direction] = {'error': 'Missing target scaler'}
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
        'total_models_loaded': len(directional_models_cached) if directional_models_cached else 0
    })

@app.route('/debug/routes')
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(str(rule))
    return jsonify(sorted(routes))

@app.route('/debug/model-status')
def debug_model_status():
    return jsonify({
        'directional_models_loaded': len(directional_models_cached) if directional_models_cached else 0,
        'stations': STATIONS,
        'models': list(directional_models_cached.keys())[:10] if directional_models_cached else []
    })

@app.route('/debug/clear-cache')
def debug_clear_cache():
    """Clear all caches - useful for forcing a fresh load"""
    cache_files = [get_models_cache_path(), get_historical_cache_path()]
    results = {}
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            results[os.path.basename(cache_file)] = "deleted"
        else:
            results[os.path.basename(cache_file)] = "not found"
    
    # Also clear memory cache flag
    global _MODELS_LOADED, _WARMUP_COMPLETE
    _MODELS_LOADED = False
    _WARMUP_COMPLETE = False
    
    return jsonify({
        "status": "Cache cleared",
        "files": results,
        "message": "Restart the app or call /warmup to reload models from source"
    })
@app.route('/debug/raw-model-output/<station_name>/<direction>')
def debug_raw_model_output(station_name, direction):
    from services import get_feature_sequence_for_station
    import numpy as np
    from datetime import datetime
    
    ensure_single_model_loaded(station_name, direction)
    now = Config.get_current_time()
    model_key = f"{station_name}_{direction}"
    
    result = {
        'station': station_name,
        'direction': direction,
        'model_key': model_key,
        'model_exists': model_key in directional_models_cached if directional_models_cached else False,
        'total_models_loaded': len(directional_models_cached) if directional_models_cached else 0
    }
    
    if model_key in directional_models_cached:
        try:
            sequence = get_feature_sequence_for_station(station_name, direction, now)
            
            if sequence is not None and len(sequence) == 24:
                target_scaler = directional_scalers_cached.get(f'{model_key}_target')
                
                if target_scaler:
                    # sequence is already scaled
                    scaled_sequence = sequence.reshape(1, 24, -1)
                    
                    input_tensor = tf.convert_to_tensor(scaled_sequence, dtype=tf.float32)
                    pred_scaled = directional_models_cached[model_key](input_tensor, training=False).numpy()
                    result['raw_scaled_output'] = float(pred_scaled[0][0])
                    
                    # ✅ FIX: Use StandardScaler inverse_transform
                    pred_original = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                    result['after_inverse_transform'] = pred_original
                    
                    # ✅ FIX: StandardScaler uses mean_ and scale_, not data_min_/data_max_
                    if hasattr(target_scaler, 'mean_') and hasattr(target_scaler, 'scale_'):
                        result['target_scaler_type'] = 'StandardScaler'
                        result['target_scaler_mean'] = float(target_scaler.mean_[0])
                        result['target_scaler_scale'] = float(target_scaler.scale_[0])
                        
                        # Manual calculation for verification
                        calculated = pred_scaled[0][0] * target_scaler.scale_[0] + target_scaler.mean_[0]
                        result['calculated_from_scaled'] = float(calculated)
                    elif hasattr(target_scaler, 'data_min_') and hasattr(target_scaler, 'data_max_'):
                        # Fallback for MinMaxScaler (backward compatibility)
                        result['target_scaler_type'] = 'MinMaxScaler'
                        result['target_scaler_min'] = float(target_scaler.data_min_[0])
                        result['target_scaler_max'] = float(target_scaler.data_max_[0])
                        calculated = pred_scaled[0][0] * (target_scaler.data_max_[0] - target_scaler.data_min_[0]) + target_scaler.data_min_[0]
                        result['calculated_from_scaled'] = float(calculated)
                    else:
                        result['warning'] = 'Unknown scaler type'
                        result['direct_model_output'] = float(pred_scaled[0][0])
                    
                    result['feature_sequence_shape'] = sequence.shape
                    result['feature_scaler_exists'] = True
                else:
                    result['error'] = 'No target scaler found'
            else:
                result['error'] = f'Invalid sequence length: {len(sequence) if sequence is not None else None}'
        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
    else:
        available = [k for k in directional_models_cached.keys() if station_name in k] if directional_models_cached else []
        result['available_models_for_station'] = available
        result['all_model_keys_sample'] = list(directional_models_cached.keys())[:10] if directional_models_cached else []
    
    return jsonify(result)

@app.route('/debug/list-models')
def debug_list_models():
    import os
    import glob
    
    model_dir = 'models_2022-2024_v10'
    if not os.path.exists(model_dir):
        return jsonify({'error': f'Directory {model_dir} not found'})
    
    files = glob.glob(os.path.join(model_dir, '*.keras'))
    file_names = [os.path.basename(f) for f in files]
    
    return jsonify({
        'directory': model_dir,
        'exists': os.path.exists(model_dir),
        'model_files': file_names,
        'count': len(file_names)
    })

@app.route('/debug/csv-status')
def debug_csv_status():
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

@app.route('/debug/memory')
def debug_memory():
    import psutil
    import gc
    gc.collect()
    
    memory = psutil.virtual_memory()
    process = psutil.Process()
    
    return jsonify({
        'system_total_mb': memory.total / (1024 * 1024),
        'system_available_mb': memory.available / (1024 * 1024),
        'system_used_mb': memory.used / (1024 * 1024),
        'process_memory_mb': process.memory_info().rss / (1024 * 1024),
        'models_loaded': len(directional_models_cached) if directional_models_cached else 0,
        'lstm_loaded': app.config.get('LSTM_PREDICTOR') is not None,
        'warmup_complete': _WARMUP_COMPLETE
    })

@app.route('/admin/import-csvs', methods=['GET', 'POST'])
def admin_import_csvs():
    import requests
    import os

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'services', 'data (2022-2024)')
    os.makedirs(data_dir, exist_ok=True)

    file_sources = {
        '2022.csv': {
            'google': 'https://drive.google.com/uc?export=download&id=1IFMhSnvU6Tps-9AAEmRL3Jn7oDbhdlVA',
            'backup': None
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
        
        try:
            print(f"📥 Downloading {filename} from Google Drive...")
            
            session = requests.Session()
            response = session.get(sources['google'], stream=True, timeout=120)
            
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                response = session.get(sources['google'] + '&confirm=1', stream=True, timeout=120)
                content_type = response.headers.get('Content-Type', '')
            
            if 'text/html' in content_type or 'text/csv' not in content_type:
                if sources['backup']:
                    response = requests.get(sources['backup'], stream=True, timeout=120)
                    content_type = response.headers.get('Content-Type', '')
            
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
                print(f"✅ Downloaded {filename} ({file_size:.2f} MB)")
            else:
                results[filename] = {
                    'status': 'failed',
                    'error': 'Received HTML or invalid response',
                    'content_type': content_type
                }
                print(f"❌ Failed to download {filename}")
                
        except Exception as e:
            results[filename] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"❌ Error downloading {filename}: {e}")

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


# ========== MEMORY TRACING ==========
import tracemalloc
tracemalloc.start()

# At the end of startup, add:
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("\n🔍 TOP 10 MEMORY USERS:")
for stat in top_stats[:10]:
    print(stat)

# ============ MAIN ============
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 MRT-3 PREDICTION SYSTEM - STARTUP COMPLETE")
    print("="*50)
    print(f"✅ {len(directional_models_cached)} directional models loaded")
    print(f"✅ Warmup: {'COMPLETE' if _WARMUP_COMPLETE else 'PENDING'}")
    print("💡 All predictions will be INSTANT from the first click!")
    print("="*50 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

application = app