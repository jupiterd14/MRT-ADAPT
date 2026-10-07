from dotenv import load_dotenv
load_dotenv()
import os


os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.matplotlib')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
# Lazy TensorFlow loader - only imports when actually used
class _LazyTF:
    _mod = None
    def __getattr__(self, name):
        if self._mod is None:
            import tensorflow as _t
            _t.config.run_functions_eagerly(False)
            try:
                _t.keras.backend.clear_session()
            except Exception:
                pass
            self._mod = _t
        return getattr(self._mod, name)

tf = _LazyTF()
# Don't try to build the cache during import
from zoneinfo import ZoneInfo
import gc
import time
import sys
if sys.platform != 'win32':
    os.environ['TZ'] = 'Asia/Manila'
    time.tzset()
    print(f"✅ Timezone set to: {time.tzname}")
else:
    print("ℹ️ Windows detected – using system timezone (Asia/Manila assumed in Config)")

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
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


from flask import Flask, session, flash, redirect, url_for, jsonify, request, render_template
from extensions import cache, limiter

import warnings
from authlib.integrations.flask_client import OAuth
import pickle
import tempfile
from datetime import datetime, timedelta


warnings.filterwarnings('ignore')

from config import Config

from models import db
from models.user import User
from models.report import Report
from models.broadcast import Broadcast
from models.activity_log import ActivityLog
from models.saved_route import SavedRoute
from models.station_data import StationData

from services.lstm_integration import (
    MRT3LSTMPredictor,
    init_lstm_predictor,
    schedule_weekly_retraining,
    register_admin_retrain,
    retrain_and_reload,
    update_global_models
)
from utils import (
    STATIONS, STATION_BASE_CAPACITY, STATION_COORDINATES,
    get_operator_stations, get_station_list, get_capacity,
    log_activity as utils_log_activity,
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
_STARTUP_IN_PROGRESS = False  # Track if deferred startup is running


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
def warmup_all_models():
    """
    🔥 CRITICAL: Warms up all 26 models to eliminate cold-start latency.
    Forces full graph compilation using a realistic random input and predict().
    """
    global directional_models_cached, _MODELS_LOADED, _WARMUP_COMPLETE
    
    if not _MODELS_LOADED or not directional_models_cached:
        print("⚠️ Models not loaded yet! Call preload_all_models() first.")
        return False
    
    if _WARMUP_COMPLETE:
        print("✅ Models already warmed up!")
        return True
    
    print("\n" + "="*60)
    print("🔥 WARMING UP ALL 26 MODELS (Full inference compilation)...")
    print("="*60)
    
    import time
    import numpy as np
    start_time = time.time()
    
    # Use a realistic random input (not zeros) to trigger full graph optimization
    # Get a real scaled sequence for a representative station and time
    from services.feature_engineering import get_scaled_feature_sequence
    from config import Config
    real_dt = Config.get_current_time().replace(hour=8, minute=0, second=0)
    real_features = get_scaled_feature_sequence("North Ave", "Northbound", real_dt)
    dummy_input = real_features.reshape(1, 24, -1)

    successful = 0
    failed = 0
    total = len(directional_models_cached)

    for idx, (model_key, model) in enumerate(directional_models_cached.items(), 1):
        try:
            # Force a full inference pass using .predict() (compiles the graph)
            _ = model.predict(dummy_input, verbose=0)
            successful += 1
        except Exception as e:
            failed += 1
            print(f"  ⚠️ Failed to warmup {model_key}: {e}")
        
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
app = Flask(__name__, template_folder='html', static_folder='static')
app.config.from_object(Config)

# ✅ Configure cache properly
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300
app.config['CACHE_THRESHOLD'] = 1000


# Initialize cache
# Initialize cache
cache.init_app(app)
app.extensions.setdefault('cache', {})[cache] = cache

# Initialize rate limiter
limiter.init_app(app)


@app.errorhandler(429)
def ratelimit_handler(e):
    """Return JSON for API routes, HTML page for others."""
    if request.path.startswith('/api/'):
        resp = jsonify({
            'error': 'rate_limited',
            'message': 'Too many requests. Please slow down.',
            'retry_after_seconds': 60,
        })
        resp.status_code = 429
        resp.headers['Retry-After'] = '60'
        return resp
    return render_template('429.html'), 429


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
    client_kwargs={'scope': 'openid email profile'},
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
register_admin_retrain(app)


@app.context_processor
def inject_now():
    return {'now': datetime.now(ZoneInfo('Asia/Manila'))}

# ============ PRELOAD MODELS AT STARTUP ============
# (The actual loading block is moved to the end of the file, after all route definitions)

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
    """
    Cache-only mode: nothing to load.
    Prediction routes use _PREDICTION_CACHE directly.
    """
    return
# Register the loader
app.config['ENSURE_MODELS_LOADED'] = ensure_models_loaded

# ============ SINGLE MODEL LOADER ============
def ensure_single_model_loaded(station_name, direction):
    """Ensure models are loaded - calls the main loader"""
    ensure_models_loaded()

app.config['ENSURE_SINGLE_MODEL_LOADED'] = ensure_single_model_loaded

def warm_cache(app):
    """Pre‑warm the live‑map cache by making an internal request."""
    with app.test_client() as client:
        try:
            # v2 has a fixed cache key -> perfect for warming
            response = client.get('/api/live-map/directions/v2')
            if response.status_code == 200:
                print("✅ Live‑map cache warmed successfully.")
            else:
                print(f"⚠️ Cache warming failed with status {response.status_code}")
        except Exception as e:
            print(f"⚠️ Cache warming error: {e}")




LSTM_MODEL_PATH = 'models_2022-2024_v10'


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
    # ✅ Preload all station patterns (typical profiles) AND all scaled sequences
    try:
        from services.feature_engineering import (
            preload_all_station_patterns,
            preload_all_data,                # new
            precompute_all_scaled_sequences  # new
        )
        # 1. Preload all DataFrames and scalers
        preload_all_data()
        # 2. Preload day-of-week typical patterns (already in your code)
        preload_all_station_patterns()
        # 3. Precompute ALL scaled sequences (7×24 per station/direction)
        precompute_all_scaled_sequences()
        print("   📊 All station patterns AND scaled sequences preloaded")
    except Exception as e:
        print(f"   ⚠️ Preload skipped: {e}")
        import traceback
        traceback.print_exc()
        
    try:
        from routes.api_predict import preload_p90_cache, preload_typical_patterns, load_correction_factors  # Changed from P95
        print("\n📊 Preloading P90 cache and typical patterns...")
        preload_p90_cache()  # Changed from P95
        preload_typical_patterns()
        load_correction_factors()
        print("   📊 Correction factors loaded")
    except Exception as e:
        print(f"⚠️ Error preloading P90/typical patterns: {e}")  # Changed from P95
    
    _MODELS_LOADED = True
    
    print("="*60)
    print(f"✅ All models loaded! ({len(directional_models_cached)}/26 models)")
    print(f"✅ Historical data loaded for {len(historical_data['historical_entry'])} stations")
    print("="*60 + "\n")
    
    return directional_models_cached, directional_scalers_cached

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
# NOTE: db.create_all() is called inside _deferred_startup() via _init_db_safe()
# so it never blocks Gunicorn from binding to $PORT.

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

# ================================================================
#  🔥 DEFERRED STARTUP — runs in background AFTER Gunicorn binds
#  This is the KEY FIX for Render's "no open ports detected" error.
# ================================================================

cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(cache_dir, exist_ok=True)

pred_cache_file = os.path.join(cache_dir, 'cached_predictions.pkl')
p90_file = os.path.join(cache_dir, 'p90_cache.pkl')
corr_file = os.path.join(cache_dir, 'correction_factors.pkl')

_STARTUP_LOCK = threading.Lock()


def _init_db_safe():
    """Create tables without blocking startup if DB is unreachable."""
    try:
        with app.app_context():
            db.create_all()
        print("✅ Database tables ensured.")
        return True
    except Exception as e:
        print(f"⚠️ db.create_all() failed — app will still start: {e}")
        return False


def _deferred_startup():
    """
    Cache-only startup — loads prediction cache from disk and nothing else.
    Models, CSVs, and historical data are NOT loaded.
    """
    global directional_models_cached, directional_scalers_cached
    global historical_data, _MODELS_LOADED, _WARMUP_COMPLETE, _STARTUP_IN_PROGRESS

    if not _STARTUP_LOCK.acquire(blocking=False):
        print("⚠️ Deferred startup already running, skipping duplicate.")
        return

    _STARTUP_IN_PROGRESS = True
    try:
        _init_db_safe()

        with app.app_context():
            try:
                # ---------- STEP 1: CSV check (no download) ----------
                data_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'services', 'data (2022-2024)'
                )
                os.makedirs(data_dir, exist_ok=True)
                csv_files = ['2022.csv', '2023.csv', '2024.csv']
                missing = [f for f in csv_files if not os.path.exists(os.path.join(data_dir, f))]
                if missing:
                    print(f"ℹ️ CSVs not on disk (cache-only mode, not downloading): {missing}")
                else:
                    print("✅ CSVs present on disk (not used).")

                # ---------- STEP 2: Load cache files ----------
                loaded_any = False

                if os.path.exists(pred_cache_file):
                    try:
                        from routes.api_predict import _PREDICTION_CACHE
                        with open(pred_cache_file, 'rb') as f:
                            data = pickle.load(f)
                            _PREDICTION_CACHE.update(data)
                        print(f"✅ Prediction cache loaded: {len(data)} entries")
                        loaded_any = True
                    except Exception as e:
                        print(f"⚠️ Prediction cache load failed: {e}")

                if os.path.exists(p90_file):
                    try:
                        from routes.api_predict import _P90_CACHE
                        with open(p90_file, 'rb') as f:
                            p90_data = pickle.load(f)
                            _P90_CACHE.update(p90_data)
                            app.config['P90_CACHE'] = p90_data
                        print(f"✅ P90 cache loaded: {len(p90_data)} entries")
                        loaded_any = True
                    except Exception as e:
                        print(f"⚠️ P90 cache load failed: {e}")

                if os.path.exists(corr_file):
                    try:
                        from routes.api_predict import _PENDING_CORRECTION_FACTORS
                        with open(corr_file, 'rb') as f:
                            corr_data = pickle.load(f)
                            _PENDING_CORRECTION_FACTORS.update(corr_data)
                        print(f"✅ Correction factors loaded: {len(corr_data)} entries")
                    except Exception as e:
                        print(f"⚠️ Correction factors load failed: {e}")
                else:
                    print("ℹ️ No correction_factors.pkl — running without correction.")

                if not loaded_any:
                    print("⚠️ No prediction cache found. Predictions will use fallback values.")
                else:
                    print("✅ Deferred startup complete — cache-only mode ready.")

                # Mark ready so status endpoints don't hang
                _WARMUP_COMPLETE = True
                _MODELS_LOADED = False  # intentionally False — models are not loaded

            except Exception as e:
                print(f"⚠️ Deferred startup failed: {e}")
                import traceback
                traceback.print_exc()
    finally:
        _STARTUP_IN_PROGRESS = False
        _STARTUP_LOCK.release()
        
# Kick off deferred startup in a daemon thread.
# This must be AFTER all @app.route decorators and blueprint registrations.
_started_once = False

@app.before_request
def _lazy_start_deferred_startup():
    """Start the deferred startup thread in the worker process, on first request."""
    global _started_once
    if not _started_once:
        _started_once = True
        t = threading.Thread(
            target=_deferred_startup,
            daemon=True,
            name="deferred-startup",
        )
        t.start()
        print(f"🚀 Started deferred startup in worker PID={os.getpid()}")

# ========== MEMORY TRACING ==========
import tracemalloc
tracemalloc.start()

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)

# ============ MAIN ============
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

application = app