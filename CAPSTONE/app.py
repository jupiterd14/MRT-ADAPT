# app.py
from datetime import datetime, timedelta
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, session, flash, redirect, url_for, jsonify
from dotenv import load_dotenv
import warnings
from authlib.integrations.flask_client import OAuth

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

# ========== IMPORT MODULES ==========
# Configuration
from config import Config

# Database
from models import db
from models.user import User
from models.report import Report
from models.broadcast import Broadcast
from models.activity_log import ActivityLog
from models.saved_route import SavedRoute
from models.station_data import StationData

# Utils (pure helpers)
from utils import (
    STATIONS, STATION_BASE_CAPACITY, STATION_COORDINATES,
    get_operator_stations, get_station_list, get_capacity,
    track_report_submission, is_rate_limited, is_suspicious_remarks, check_duplicate_report,
    log_activity as utils_log_activity,
    report_tracker
)

# Services (business logic)
from services import (
    load_directional_models, load_real_historical_data,
    get_directional_prediction, get_station_prediction,
    get_feature_sequence_for_station,
    directional_models, directional_scalers,
    historical_entry, historical_exit, hourly_avg_entry, hourly_avg_exit
)

# Routes (blueprints)
from routes import (
    auth_bp, user_bp, admin_bp, operator_bp, public_bp,
    api_predict_bp, api_schedule_bp, api_reports_bp, api_other_bp
)

# ========== CREATE FLASK APP ==========
app = Flask(__name__, template_folder='html', static_folder='static')
app.config.from_object(Config)

# ========== INITIALIZE DATABASE ==========
db.init_app(app)

# ========== SETUP GOOGLE OAUTH ==========
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=app.config.get('GOOGLE_CLIENT_ID'),
    client_secret=app.config.get('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)
# REMOVED the line that was causing the error:
# app.extensions['authlib.client']['google'] = google

app.config['GOOGLE_CLIENT'] = google

# ========== REGISTER BLUEPRINTS ==========
app.register_blueprint(auth_bp, url_prefix='/')
app.register_blueprint(user_bp, url_prefix='/')
app.register_blueprint(admin_bp, url_prefix='/')
app.register_blueprint(operator_bp, url_prefix='/')
app.register_blueprint(public_bp, url_prefix='/')
app.register_blueprint(api_predict_bp, url_prefix='/api')
app.register_blueprint(api_schedule_bp, url_prefix='/api')
app.register_blueprint(api_reports_bp, url_prefix='/api')
app.register_blueprint(api_other_bp, url_prefix='/api')

# ========== CONTEXT PROCESSOR ==========
@app.context_processor
def inject_now():
    """Inject current datetime into templates"""
    return {'now': datetime.now()}

# ========== GLOBAL FUNCTIONS INJECTION FOR BLUEPRINTS ==========
def get_directional_prediction_wrapper(station_name, direction, target_datetime=None):
    return get_directional_prediction(
        station_name, direction, target_datetime,
        directional_models, directional_scalers,
        get_feature_sequence_for_station
    )

def get_station_prediction_wrapper(station_name):
    return get_station_prediction(
        station_name, None,
        directional_models, directional_scalers,
        get_feature_sequence_for_station
    )

def log_activity_wrapper(user_id, user_type, user_email, action, details=None):
    return utils_log_activity(
        user_id, user_type, user_email, action, details,
        ActivityLog, db.session, request=None
    )

# Store functions in app config for blueprints to access
app.config['GET_DIRECTIONAL_PREDICTION'] = get_directional_prediction_wrapper
app.config['GET_STATION_PREDICTION'] = get_station_prediction_wrapper
app.config['LOG_ACTIVITY'] = log_activity_wrapper
app.config['STATIONS'] = STATIONS
app.config['STATION_BASE_CAPACITY'] = STATION_BASE_CAPACITY
app.config['STATION_COORDINATES'] = STATION_COORDINATES

# ========== TYPE ICONS ==========
typeIcons = {
    "Train Breakdown": "fa-train",
    "Overcrowding": "fa-users", 
    "Maintenance": "fa-wrench",
    "Signal Issue": "fa-satellite-dish",
    "Gate Closure": "fa-door-closed",
    "General Notice": "fa-bullhorn"
}
app.config['TYPE_ICONS'] = typeIcons

# ========== LOAD DIRECTIONAL MODELS (2023-2024 REAL DATA) ==========
print("\n" + "="*70)
print("🚇 LOADING DIRECTIONAL MODELS (2023-2024 REAL DATA)...")
print("="*70)

DIRECTIONAL_MODELS_PATH = 'models_2023-2024'
load_directional_models(STATIONS, DIRECTIONAL_MODELS_PATH)

# ========== LOAD HISTORICAL DATA ==========
print("\n" + "="*70)
print("📊 LOADING HISTORICAL DATA...")
print("="*70)

historical_data = load_real_historical_data(STATIONS, STATION_BASE_CAPACITY)
print(f"✅ Loaded historical data for {len(historical_data['historical_entry'])} stations")

# ========== CREATE DATABASE TABLES ==========
with app.app_context():
    db.create_all()
    
    try:
        from sqlalchemy import inspect, text
        
        # Check and add missing columns to report table
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('report')]
        
        if 'direction' not in columns:
            print("🔧 Adding 'direction' column to report table...")
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE report ADD COLUMN direction VARCHAR(20)'))
                conn.commit()
            print("✅ Direction column added successfully")
        else:
            print("✅ Direction column already exists")
        
        if 'flag_count' not in columns:
            print("🔧 Adding 'flag_count' column to report table...")
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE report ADD COLUMN flag_count INTEGER DEFAULT 0'))
                conn.commit()
            print("✅ flag_count column added successfully")
        
        if 'reviewed' not in columns:
            print("🔧 Adding 'reviewed' column to report table...")
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE report ADD COLUMN reviewed BOOLEAN DEFAULT 0'))
                conn.commit()
            print("✅ reviewed column added successfully")
        
        # Check broadcast table for direction column
        try:
            broadcast_columns = [col['name'] for col in inspector.get_columns('broadcast')]
            if 'direction' not in broadcast_columns:
                print("🔧 Adding 'direction' column to broadcast table...")
                with db.engine.connect() as conn:
                    conn.execute(text('ALTER TABLE broadcast ADD COLUMN direction VARCHAR(20) DEFAULT "both"'))
                    conn.commit()
                print("✅ Direction column added to broadcast table")
        except Exception as broadcast_error:
            print(f"⚠️ Broadcast table note: {broadcast_error}")
            
    except Exception as e:
        print(f"⚠️ Note: {e}")

# ========== DEBUG ROUTES ==========
@app.route('/debug/routes')
def list_routes():
    """List all registered routes"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(str(rule))
    return jsonify(sorted(routes))

@app.route('/debug/historical-all')
def debug_historical_all():
    """Show all historical data"""
    return jsonify({
        'historical_entry': historical_entry,
        'historical_exit': historical_exit,
        'hourly_avg_entry': hourly_avg_entry,
        'hourly_avg_exit': hourly_avg_exit
    })

@app.route('/debug/db-schema')
def debug_db_schema():
    """Debug database schema"""
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

# ========== UPLOADS ROUTE ==========
@app.route('/uploads/reports/<filename>')
def serve_upload(filename):
    """Serve uploaded report images"""
    from flask import send_from_directory
    upload_folder = os.path.join(os.path.dirname(__file__), 'static', 'uploads', 'reports')
    return send_from_directory(upload_folder, filename)

# Add to app.py temporarily to debug
@app.route('/debug/google-config')
def debug_google_config():
    return jsonify({
        'GOOGLE_CLIENT_ID': app.config.get('GOOGLE_CLIENT_ID', 'NOT SET'),
        'GOOGLE_CLIENT_SECRET': 'SET' if app.config.get('GOOGLE_CLIENT_SECRET') else 'NOT SET',
        'GOOGLE_CLIENT_IN_CONFIG': 'GOOGLE_CLIENT' in app.config,
    })
    
@app.route('/debug/model-status')
def debug_model_status():
    """Check if models are loaded"""
    from services.model_loader import directional_models
    
    return jsonify({
        'directional_models_loaded': len(directional_models),
        'stations': STATIONS,
        'models': list(directional_models.keys())[:10]
    })
    
# ========== DEBUG: RAW MODEL OUTPUT ENDPOINT ==========
@app.route('/debug/raw-model-output/<station_name>/<direction>')
def debug_raw_model_output(station_name, direction):
    """Get raw model output before any scaling or processing"""
    from services.model_loader import directional_models, directional_scalers
    from services import get_feature_sequence_for_station
    import numpy as np
    from datetime import datetime
    
    now = datetime.now()
    model_key = f"{station_name}_{direction}"
    
    result = {
        'station': station_name,
        'direction': direction,
        'model_key': model_key,
        'model_exists': model_key in directional_models,
        'total_models_loaded': len(directional_models)
    }
    
    if model_key in directional_models:
        try:
            # Get feature sequence
            sequence = get_feature_sequence_for_station(station_name, direction, now)
            
            if sequence is not None and len(sequence) == 24:
                feature_scaler = directional_scalers.get(f'{model_key}_feature')
                target_scaler = directional_scalers.get(f'{model_key}_target')
                
                if feature_scaler:
                    scaled_sequence = feature_scaler.transform(sequence)
                    input_sequence = scaled_sequence.reshape(1, 24, -1)
                    
                    # RAW model output (scaled)
                    pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
                    result['raw_scaled_output'] = float(pred_scaled[0][0])
                    
                    # After inverse transform
                    if target_scaler:
                        pred_original = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                        result['after_inverse_transform'] = pred_original
                        
                        # Check what values the target_scaler knows
                        result['target_scaler_min'] = float(target_scaler.data_min_[0])
                        result['target_scaler_max'] = float(target_scaler.data_max_[0])
                        result['target_scaler_range'] = result['target_scaler_max'] - result['target_scaler_min']
                        
                        # Calculate what the original scale was
                        calculated = pred_scaled[0][0] * (result['target_scaler_max'] - result['target_scaler_min']) + result['target_scaler_min']
                        result['calculated_from_scaled'] = float(calculated)
                    else:
                        # No target scaler - model might output directly in 0-100
                        result['direct_model_output'] = float(pred_scaled[0][0])
                    
                    # Also check what features were used
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
        # List available models for this station
        available = [k for k in directional_models.keys() if station_name in k]
        result['available_models_for_station'] = available
        result['all_model_keys_sample'] = list(directional_models.keys())[:10]
    
    return jsonify(result)


# ========== DEBUG: TEST PREDICTION WITHOUT CLAMPING ==========
@app.route('/debug/test-no-clamp/<station_name>')
def debug_test_no_clamp(station_name):
    """Test prediction without any clamping"""
    from services import get_directional_prediction, get_feature_sequence_for_station
    from services.model_loader import directional_models, directional_scalers
    from datetime import datetime
    
    now = datetime.now()
    results = {}
    
    for direction in ['Northbound', 'Southbound']:
        model_key = f"{station_name}_{direction}"
        
        if model_key in directional_models:
            try:
                sequence = get_feature_sequence_for_station(station_name, direction, now)
                if sequence is not None and len(sequence) == 24:
                    feature_scaler = directional_scalers.get(f'{model_key}_feature')
                    target_scaler = directional_scalers.get(f'{model_key}_target')
                    
                    if feature_scaler:
                        scaled_sequence = feature_scaler.transform(sequence)
                        input_sequence = scaled_sequence.reshape(1, 24, -1)
                        pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
                        
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
    """Test what the live map API returns"""
    from services.model_loader import directional_models
    from services import get_directional_prediction, get_feature_sequence_for_station
    
    results = {}
    now = datetime.now()
    
    for station in STATIONS[:5]:  # First 5 stations
        station_results = {}
        for direction in ['Northbound', 'Southbound']:
            model_key = f"{station}_{direction}"
            if model_key in directional_models:
                try:
                    sequence = get_feature_sequence_for_station(station, direction, now)
                    if sequence is not None and len(sequence) == 24:
                        feature_scaler = directional_scalers.get(f'{model_key}_feature')
                        target_scaler = directional_scalers.get(f'{model_key}_target')
                        if feature_scaler and target_scaler:
                            scaled_sequence = feature_scaler.transform(sequence)
                            input_sequence = scaled_sequence.reshape(1, 24, -1)
                            pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
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
        'models_loaded': len(directional_models)
    })
    
@app.route('/debug/simple-prediction/<station_name>')
def debug_simple_prediction(station_name):
    """Simplified prediction without complex feature engineering"""
    from services.model_loader import directional_models, directional_scalers
    from datetime import datetime
    import numpy as np
    
    now = datetime.now()
    results = {}
    
    for direction in ['Northbound', 'Southbound']:
        model_key = f"{station_name}_{direction}"
        
        if model_key not in directional_models:
            results[direction] = "Model not found"
            continue
        
        try:
            # Get the model and scalers
            model = directional_models[model_key]
            feature_scaler = directional_scalers.get(f'{model_key}_feature')
            target_scaler = directional_scalers.get(f'{model_key}_target')
            
            if feature_scaler is None or target_scaler is None:
                results[direction] = "Missing scaler"
                continue
            
            # Create a simple test input (just use zeros or random values)
            # This tests if the model itself works
            input_shape = model.input_shape
            print(f"Model input shape for {model_key}: {input_shape}")
            
            # Create dummy input with correct shape
            dummy_input = np.zeros((1, 24, 29))  # 24 time steps, 29 features
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
   
    
# ========== RUN SYSTEM ==========
if __name__ == '__main__':
    print("\n" + "="*70)
    print("✨ MRT-3 PREDICTION SYSTEM READY! (2023-2024 Real Data)")
    print("="*70)
    from services import directional_models as svc_directional_models
    print(f"🚇 Directional Models Loaded: {len(svc_directional_models)}")
    print(f"📊 Historical Data: {len(historical_entry)} stations")
    print(f"📍 Stations: {len(STATIONS)}")
    print("👤 Starting as GUEST by default")
    print("🔐 Password hashing: Enabled")
    print("🌐 Open http://localhost:5000")
    print("="*70)
    app.run(debug=True, port=5000)