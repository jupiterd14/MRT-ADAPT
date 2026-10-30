from dotenv import load_dotenv
load_dotenv()
import os


os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.matplotlib')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)


# ============================================================
# LAZY TENSORFLOW LOADER (only imports when actually used)
# ============================================================
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
import json
import threading

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

# Limit TensorFlow memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Aggressive garbage collection
gc.set_threshold(50, 3, 3)


from flask import Flask, session, flash, redirect, url_for, jsonify, request, render_template
from werkzeug.middleware.proxy_fix import ProxyFix
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


# ============================================================
# GLOBAL STATE (declared EARLY so functions can reference them)
# ============================================================
_MODELS_CACHE = {}
_MODELS_CACHE_FILE = None
_MODELS_LOADED = False
_WARMUP_COMPLETE = False
_STARTUP_IN_PROGRESS = False

directional_models_cached = {}
directional_scalers_cached = {}
historical_data = None

_MODEL_LOAD_LOCK = threading.Lock()
_STARTUP_LOCK = threading.Lock()
_started_once = False

DIRECTIONAL_MODELS_PATH = 'models_2023-2025_v14'
LSTM_MODEL_PATH = 'models_2023-2025_v14'


# ============================================================
# CACHE PATHS
# ============================================================
def get_models_cache_path():
    cache_dir = tempfile.gettempdir()
    return os.path.join(cache_dir, 'mrt3_models_cache.pkl')


def get_historical_cache_path():
    cache_dir = tempfile.gettempdir()
    return os.path.join(cache_dir, 'mrt3_historical_cache.pkl')


# ✅ ALIGN WITH api_predict.py (JSON for P90, PKL for correction)
cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(cache_dir, exist_ok=True)

pred_cache_file = os.path.join(cache_dir, 'cached_predictions.pkl')
p90_file = 'p90_percentiles.json'          # JSON — matches api_predict.py
corr_file = 'correction_factors.pkl'        # PKL — matches api_predict.py


# ============================================================
# MODEL LOADING HELPERS
# ============================================================
def load_models_with_cache(stations, models_path):
    """Load models from .keras files, with in-RAM caching."""
    global _MODELS_CACHE, _MODELS_LOADED

    if _MODELS_LOADED and _MODELS_CACHE.get('directional_models'):
        print("✓ Using models from RAM")
        return _MODELS_CACHE['directional_models'], _MODELS_CACHE['directional_scalers']

    print("🔄 Loading models from .keras files...")
    loaded_models, loaded_scalers = load_directional_models(stations, models_path)

    # ✅ Guard against empty load
    if not loaded_models:
        print("❌ No models loaded from disk")
        return {}, {}

    _MODELS_CACHE = {
        'directional_models': loaded_models,
        'directional_scalers': loaded_scalers
    }
    _MODELS_LOADED = True

    return loaded_models, loaded_scalers


def load_historical_with_cache(stations, base_capacity):
    cache_file = get_historical_cache_path()

    if os.path.exists(cache_file):
        try:
            print("📦 Loading historical data from cache...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Loaded historical data for {len(data['historical_entry'])} stations from cache")
            return data
        except Exception as e:
            print(f"Historical cache load failed: {e}, reloading from source...")

    print("🔄 Loading historical data from source (first time only)...")
    data = load_real_historical_data(stations, base_capacity)

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Historical data cached")
    except Exception as e:
        print(f"Historical cache save failed: {e}")

    return data


# ============================================================
# MODEL WARMUP (optional, not used in cache-only mode)
# ============================================================
def warmup_all_models():
    """
    Warms up all loaded models to eliminate cold-start latency.
    Skipped automatically in cache-only mode (_MODELS_LOADED = False).
    """
    global _MODELS_LOADED, _WARMUP_COMPLETE

    if not _MODELS_LOADED or not directional_models_cached:
        print("⚠️ Models not loaded yet! Call preload_all_models() first.")
        return False

    if _WARMUP_COMPLETE:
        print("✅ Models already warmed up!")
        return True

    print("\n" + "=" * 60)
    print("🔥 WARMING UP ALL MODELS (Full inference compilation)...")
    print("=" * 60)

    import numpy as np
    start_time = time.time()

    # Realistic input triggers full graph optimization
    from services.feature_engineering import get_scaled_feature_sequence
    real_dt = Config.get_current_time().replace(hour=8, minute=0, second=0)
    real_features = get_scaled_feature_sequence("North Ave", "Northbound", real_dt)
    dummy_input = real_features.reshape(1, 24, -1)

    successful = 0
    failed = 0
    total = len(directional_models_cached)

    for idx, (model_key, model) in enumerate(directional_models_cached.items(), 1):
        try:
            _ = model.predict(dummy_input, verbose=0)
            successful += 1
        except Exception as e:
            failed += 1
            print(f"  ⚠️ Failed to warmup {model_key}: {e}")

        if idx % 5 == 0 or idx == total:
            print(f"  ⏳ Warmup progress: {idx}/{total} models")

    elapsed = time.time() - start_time
    gc.collect()

    print("=" * 60)
    print(f"✅ WARMUP COMPLETE in {elapsed:.2f} seconds")
    print(f"   ✅ {successful} models warmed up successfully")
    if failed > 0:
        print(f"   ⚠️ {failed} models failed to warmup")
    print("=" * 60 + "\n")

    app.config['WARMUP_STATS'] = {
        'successful': successful,
        'failed': failed,
        'duration_seconds': round(elapsed, 2),
        'models_warmed': successful,
        'total_models': total
    }

    _WARMUP_COMPLETE = True
    return True


# ============================================================
# CSV IMPORT FUNCTION
# ============================================================
def import_csv_files():
    import requests

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'services', 'data (2022-2024)')
    os.makedirs(data_dir, exist_ok=True)

    file_sources = {
        '2022.csv': {'google': 'https://drive.google.com/uc?export=download&id=1IFMhSnvU6Tps-9AAEmRL3Jn7oDbhdlVA', 'backup': None},
        '2023.csv': {'google': 'https://drive.google.com/uc?export=download&id=14H6zXJxXHMX4kt3-1tkXc0gUH066cuF_', 'backup': None},
        '2024.csv': {'google': 'https://drive.google.com/uc?export=download&id=1xDbrMdTomXkrGQ5i54FBDE6yEWrXAO1N', 'backup': None},
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


# ============================================================
# APP INITIALIZATION
# ============================================================
app = Flask(__name__, template_folder='html', static_folder='static')
app.config.from_object(Config)

# ✅ ProxyFix for real client IP behind Render's reverse proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# ✅ Cache config
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300
app.config['CACHE_THRESHOLD'] = 1000

cache.init_app(app)
app.extensions.setdefault('cache', {})[cache] = cache

limiter.init_app(app)


@app.errorhandler(429)
def ratelimit_handler(e):
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


@app.after_request
def add_security_headers(response):
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(self), microphone=(), camera=()'
    return response


# ============================================================
# BASIC ROUTES
# ============================================================
@app.route('/warmup')
def warmup():
    """Warm up the app by loading ALL models and data at startup."""
    start = time.time()

    try:
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
    try:
        import psutil
        process = psutil.Process()
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
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


# ============================================================
# FILE SERVING ROUTES
# ============================================================
@app.route('/uploads/reports/<filename>')
def serve_upload(filename):
    from flask import send_from_directory, abort, current_app

    upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', 'reports')
    file_path = os.path.join(upload_folder, filename)
    if not os.path.exists(file_path):
        abort(404)
    return send_from_directory(upload_folder, filename)


@app.route('/uploads/reports/<path:filename>')
def serve_upload_with_path(filename):
    from flask import send_from_directory, abort, current_app

    upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', 'reports')
    safe_path = os.path.normpath(filename)
    if safe_path.startswith('..'):
        abort(403)
    file_path = os.path.join(upload_folder, safe_path)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        abort(404)
    return send_from_directory(upload_folder, safe_path)


# ============================================================
# BLUEPRINT REGISTRATION
# ============================================================
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
app.register_blueprint(email_bp, url_prefix='/api/auth')
register_admin_retrain(app)


@app.context_processor
def inject_now():
    return {'now': datetime.now(ZoneInfo('Asia/Manila'))}


# ============================================================
# MODEL LOADING (lazy trigger, cache-only mode by default)
# ============================================================
def ensure_models_loaded(station_name=None, direction=None):
    """
    Cache-only mode: no models loaded eagerly.
    If prediction cache misses, this triggers preload_all_models() lazily.
    """
    global _MODELS_LOADED
    if not _MODELS_LOADED:
        models_dir = os.path.join(os.path.dirname(__file__), DIRECTIONAL_MODELS_PATH)
        if os.path.exists(models_dir):
            print("⚠️ ensure_models_loaded: triggering preload_all_models()...")
            try:
                preload_all_models()
            except Exception as e:
                print(f"⚠️ preload_all_models failed: {e}")
        else:
            print(f"ℹ️ Models directory not found: {models_dir} — using cache-only mode")

app.config['ENSURE_MODELS_LOADED'] = ensure_models_loaded


def ensure_single_model_loaded(station_name, direction):
    """Ensure models are loaded - delegates to main loader."""
    ensure_models_loaded()

app.config['ENSURE_SINGLE_MODEL_LOADED'] = ensure_single_model_loaded


def warm_cache(app):
    """Pre-warm the live-map cache by making an internal request."""
    with app.test_client() as client:
        try:
            response = client.get('/api/live-map/directions/v2')
            if response.status_code == 200:
                print("✅ Live-map cache warmed successfully.")
            else:
                print(f"⚠️ Cache warming failed with status {response.status_code}")
        except Exception as e:
            print(f"⚠️ Cache warming error: {e}")


# ============================================================
# PRELOAD ALL MODELS
# ============================================================
def preload_all_models():
    """Preload ALL 26 models. Only called if you want to force-load."""
    global directional_models_cached, directional_scalers_cached, _MODELS_LOADED, historical_data

    if _MODELS_LOADED and directional_models_cached:
        print(f"✅ All models already loaded! ({len(directional_models_cached)}/26)")
        return directional_models_cached, directional_scalers_cached

    print("\n" + "=" * 60)
    print("🔄 PRELOADING ALL 26 MODELS...")
    print("=" * 60)

    loaded_models, loaded_scalers = load_models_with_cache(STATIONS, DIRECTIONAL_MODELS_PATH)

    # ✅ Guard: abort if models didn't load
    if not loaded_models:
        print("❌ preload_all_models: no models loaded, aborting.")
        return {}, {}

    directional_models_cached = loaded_models
    directional_scalers_cached = loaded_scalers

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

    # Preload patterns + scaled sequences
    try:
        from services.feature_engineering import (
            preload_all_station_patterns,
            preload_all_data,
            precompute_all_scaled_sequences
        )
        preload_all_data()
        preload_all_station_patterns()
        precompute_all_scaled_sequences()
        print("   📊 All station patterns AND scaled sequences preloaded")
    except Exception as e:
        print(f"   ⚠️ Preload skipped: {e}")
        import traceback
        traceback.print_exc()

    # Preload P90 + correction factors
    try:
        from routes.api_predict import (
            preload_p90_cache,
            preload_typical_patterns,
            load_correction_factors,
        )
        print("\n📊 Preloading P90 cache and typical patterns...")
        preload_p90_cache()
        preload_typical_patterns()
        load_correction_factors()
        print("   📊 Correction factors loaded")
    except Exception as e:
        print(f"⚠️ Error preloading P90/typical patterns: {e}")

    _MODELS_LOADED = True

    print("=" * 60)
    print(f"✅ All models loaded! ({len(directional_models_cached)}/26 models)")
    print(f"✅ Historical data loaded for {len(historical_data['historical_entry'])} stations")
    print("=" * 60 + "\n")

    return directional_models_cached, directional_scalers_cached


# ============================================================
# WRAPPER FUNCTIONS
# ============================================================
def get_directional_prediction_wrapper(station_name, direction, target_datetime=None):
    """Wrapper around the main prediction API (ensures consistency)."""
    from routes.api_predict import get_directional_prediction as _gdp
    return _gdp(station_name, direction, target_datetime)


def get_station_prediction_wrapper(station_name):
    """Wrapper around the main prediction API for a station (avg of both directions)."""
    from routes.api_predict import get_directional_prediction as _gdp

    now = Config.get_current_time()
    north = _gdp(station_name, 'Northbound', now)
    south = _gdp(station_name, 'Southbound', now)

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


# ============================================================
# ADMIN CSV IMPORT
# ============================================================
@app.route('/admin/import-csvs', methods=['GET', 'POST'])
def admin_import_csvs():
    import requests

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'services', 'data (2022-2024)')
    os.makedirs(data_dir, exist_ok=True)

    file_sources = {
        '2022.csv': {'google': 'https://drive.google.com/uc?export=download&id=1IFMhSnvU6Tps-9AAEmRL3Jn7oDbhdlVA', 'backup': None},
        '2023.csv': {'google': 'https://drive.google.com/uc?export=download&id=14H6zXJxXHMX4kt3-1tkXc0gUH066cuF_', 'backup': None},
        '2024.csv': {'google': 'https://drive.google.com/uc?export=download&id=1xDbrMdTomXkrGQ5i54FBDE6yEWrXAO1N', 'backup': None},
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


# ============================================================
# DEFERRED STARTUP — runs in background AFTER Gunicorn binds
# KEY FIX for Render's "no open ports detected" error.
# ============================================================
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
    Cache-only startup. Loads persisted caches from disk.
    Falls back to precompute_all_predictions() if no cache exists and models are loaded.
    """
    global _MODELS_LOADED, _WARMUP_COMPLETE, _STARTUP_IN_PROGRESS

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

                # 2a. Prediction cache
                if os.path.exists(pred_cache_file):
                    try:
                        from routes.api_predict import _PREDICTION_CACHE
                        with open(pred_cache_file, 'rb') as f:
                            data = pickle.load(f)
                        _PREDICTION_CACHE.clear()
                        _PREDICTION_CACHE.update(data)
                        # Mirror into app.config so any module can reach it
                        app.config['PREDICTION_CACHE'] = dict(data)
                        print(f"✅ Prediction cache loaded: {len(data)} entries")
                        loaded_any = True
                    except Exception as e:
                        print(f"⚠️ Prediction cache load failed: {e}")

                # 2b. P90 cache (JSON — matches api_predict.py)
                if os.path.exists(p90_file):
                    try:
                        with open(p90_file, 'r') as f:
                            p90_data = json.load(f)
                        if 'P90_CACHE' not in app.config:
                            app.config['P90_CACHE'] = {}
                        app.config['P90_CACHE'].update(p90_data)
                        print(f"✅ P90 cache loaded: {len(p90_data)} entries")
                        loaded_any = True
                    except Exception as e:
                        print(f"⚠️ P90 cache load failed: {e}")

                # 2c. Correction factors (PKL — matches api_predict.py)
                if os.path.exists(corr_file):
                    try:
                        with open(corr_file, 'rb') as f:
                            corr_data = pickle.load(f)
                        if 'CORRECTION_FACTORS' not in app.config:
                            app.config['CORRECTION_FACTORS'] = {}
                        app.config['CORRECTION_FACTORS'].update(corr_data)
                        print(f"✅ Correction factors loaded: {len(corr_data)} entries")
                    except Exception as e:
                        print(f"⚠️ Correction factors load failed: {e}")
                else:
                    print("ℹ️ No correction_factors.pkl — running without correction.")

                # ---------- STEP 3: Fallback precompute if nothing loaded ----------
                if not loaded_any:
                    print("⚠️ No prediction cache found. Attempting fallback...")
                    try:
                        if _MODELS_LOADED:
                            from routes.api_predict import (
                                precompute_all_predictions,
                                _PREDICTION_CACHE,
                            )
                            precompute_all_predictions()
                            try:
                                with open(pred_cache_file, 'wb') as f:
                                    pickle.dump(dict(_PREDICTION_CACHE), f)
                                print(f"✅ Precomputed and cached {len(_PREDICTION_CACHE)} predictions")
                            except Exception as e:
                                print(f"⚠️ Could not persist prediction cache: {e}")
                        else:
                            print("ℹ️ Models not loaded — predictions will use fallback values.")
                    except Exception as e:
                        print(f"⚠️ Fallback precompute failed: {e}")
                        import traceback
                        traceback.print_exc()

                # ---------- STEP 4: Clear stale request-level cache ----------
                try:
                    from routes.api_predict import _REQUEST_CACHE
                    _REQUEST_CACHE.clear()
                    print("✅ Cleared stale request-level cache")
                except Exception as e:
                    print(f"⚠️ Could not clear request cache: {e}")

                print("✅ Deferred startup complete — cache-only mode ready.")
                _WARMUP_COMPLETE = True
                # _MODELS_LOADED remains False in cache-only mode (intentional)

            except Exception as e:
                print(f"⚠️ Deferred startup failed: {e}")
                import traceback
                traceback.print_exc()
    finally:
        _STARTUP_IN_PROGRESS = False
        _STARTUP_LOCK.release()



# Run deferred startup synchronously on the first request in each worker.
# Blocks the very first request (~1s) so the prediction cache is guaranteed
# to be populated BEFORE any handler runs. No race condition.
@app.before_request
def _lazy_start_deferred_startup():
    """Run deferred startup synchronously on first request in this worker."""
    global _started_once
    if not _started_once:
        _started_once = True
        print(f"🚀 Running deferred startup synchronously in worker PID={os.getpid()}")
        _deferred_startup()
# ============================================================
# MEMORY TRACING (opt-in only — off by default to save RAM)
# ============================================================
if os.environ.get('ENABLE_TRACEMALLOC') == '1':
    import tracemalloc
    tracemalloc.start()
    snapshot = tracemalloc.take_snapshot()
    for stat in snapshot.statistics('lineno')[:10]:
        print(stat)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

application = app
