import os
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

gc.set_threshold(50, 3, 3)

import tensorflow as tf
tf.config.run_functions_eagerly(False)
tf.keras.backend.clear_session()
print("✅ TensorFlow memory optimized in model_loader.py")

# ============================================================
# REST OF IMPORTS
# ============================================================
import pickle
from tensorflow.keras.saving import register_keras_serializable
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ============================================================
# GLOBAL VARIABLES - EMPTY, NO DATA LOADED
# ============================================================
directional_models = {}
directional_scalers = {}
historical_entry = {}
historical_exit = {}
hourly_avg_entry = {}
hourly_avg_exit = {}
dow_avg_entry = {}
dow_avg_exit = {}
direction_counts = {}
station_time_series = {}

# Station name mapping (display name → file name)
STATION_FILE_MAP = {
    "North Avenue": "North Ave",
    "Quezon Avenue": "Quezon Ave",
    "Shaw Boulevard": "Shaw Blvd",
    "Ayala Avenue": "Ayala Ave",
    "Boni Avenue": "Boni Ave",
}

# Register the rmse function so it can be loaded
@register_keras_serializable()
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


# ============================================================
# HELPER: Get Model File Path (Supports Multiple Suffixes)
# ============================================================
def get_model_file_path(model_path, model_key):
    """
    Get the correct model file path, checking all possible suffixes.
    Supports: _lstm_v10_plus.keras, _lstm_enhanced.keras, _best.keras
    """
    suffixes = ['_lstm_v10_plus.keras', '_lstm_enhanced.keras', '_best.keras']
    for suffix in suffixes:
        path = os.path.join(model_path, f'{model_key}{suffix}')
        if os.path.exists(path):
            return path
    return None


def get_scaler_paths(model_path, model_key):
    """Get feature and target scaler paths for a model"""
    feature_path = os.path.join(model_path, f'{model_key}_feature_scaler.pkl')
    target_path = os.path.join(model_path, f'{model_key}_target_scaler.pkl')
    return feature_path, target_path


# ============================================================
# LOAD ALL MODELS
# ============================================================
def load_directional_models(STATIONS, DIRECTIONAL_MODELS_PATH='models_2022-2024_v10'):
    """
    Load ALL directional models - USE WITH CAUTION!
    This loads all 26 models and uses ~300MB memory.
    Only use if you have enough memory.
    """
    global directional_models, directional_scalers
    
    directional_models = {}
    directional_scalers = {}
    
    print(f"📂 Loading models from: {DIRECTIONAL_MODELS_PATH}")
    
    # Check if directory exists
    if not os.path.exists(DIRECTIONAL_MODELS_PATH):
        print(f"❌ Directory not found: {DIRECTIONAL_MODELS_PATH}")
        print(f"   Current working directory: {os.getcwd()}")
        return directional_models, directional_scalers
    
    # First, check what model files exist
    all_files = os.listdir(DIRECTIONAL_MODELS_PATH)
    model_files = [f for f in all_files if f.endswith('.keras')]
    print(f"📋 Found {len(model_files)} .keras files in directory")
    
    loaded_count = 0
    failed_count = 0
    
    for station in STATIONS:
        for direction in ['Northbound', 'Southbound']:
            station_file = STATION_FILE_MAP.get(station, station)
            model_key = f"{station_file}_{direction}"
            
            # ✅ FIX: Use the helper to find the model file
            model_path = get_model_file_path(DIRECTIONAL_MODELS_PATH, model_key)
            
            if model_path is None:
                print(f"  ⚠️ No model file for {model_key}")
                failed_count += 1
                continue
            
            feature_scaler_path, target_scaler_path = get_scaler_paths(DIRECTIONAL_MODELS_PATH, model_key)
            
            try:
                print(f"  Loading {model_key}...")
                directional_models[model_key] = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'rmse': rmse}
                )
                
                # Load feature scaler
                if os.path.exists(feature_scaler_path):
                    with open(feature_scaler_path, 'rb') as f:
                        directional_scalers[f'{model_key}_feature'] = pickle.load(f)
                else:
                    print(f"    ⚠️ Feature scaler not found: {feature_scaler_path}")
                
                # Load target scaler
                if os.path.exists(target_scaler_path):
                    with open(target_scaler_path, 'rb') as f:
                        directional_scalers[f'{model_key}_target'] = pickle.load(f)
                else:
                    print(f"    ⚠️ Target scaler not found: {target_scaler_path}")
                
                loaded_count += 1
                gc.collect()
                
            except Exception as e:
                failed_count += 1
                print(f"  ❌ Error loading {model_key}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n✅ Loaded {loaded_count} directional models")
    if failed_count > 0:
        print(f"   ⚠️ Failed to load {failed_count} models")
    print(f"✅ Loaded {len(directional_scalers)} scalers")
    
    # Show target scaler info
    target_scalers = [k for k in directional_scalers.keys() if k.endswith('_target')]
    print(f"✅ Target scalers loaded: {len(target_scalers)}")
    if target_scalers:
        print(f"   Examples: {target_scalers[:3]}")
        first_scaler = directional_scalers[target_scalers[0]]
        if hasattr(first_scaler, 'data_min_') and hasattr(first_scaler, 'data_max_'):
            print(f"   First scaler range: {float(first_scaler.data_min_[0]):.1f} to {float(first_scaler.data_max_[0]):.1f}")
    
    return directional_models, directional_scalers


# ============================================================
# LOAD SINGLE MODEL (Memory Efficient)
# ============================================================
def load_single_model(station, direction, model_path='models_2022-2024_v10'):
    """
    Load a SINGLE model for one station-direction (memory efficient)
    This is the recommended way to load models.
    Only ~20-30MB per model.
    """
    station_file = STATION_FILE_MAP.get(station, station)
    model_key = f"{station_file}_{direction}"
    result = {'model': None, 'feature': None, 'target': None}
    
    # ✅ FIX: Use the helper to find the model file
    model_file = get_model_file_path(model_path, model_key)
    
    if model_file is None:
        print(f"⚠️ No model file for {model_key}")
        return None, result
    
    try:
        # Load the model
        model = tf.keras.models.load_model(model_file, custom_objects={'rmse': rmse})
        result['model'] = model
        
        # Load feature scaler
        feature_scaler_path, target_scaler_path = get_scaler_paths(model_path, model_key)
        
        if os.path.exists(feature_scaler_path):
            with open(feature_scaler_path, 'rb') as f:
                result['feature'] = pickle.load(f)
        
        if os.path.exists(target_scaler_path):
            with open(target_scaler_path, 'rb') as f:
                result['target'] = pickle.load(f)
        
        print(f"✅ Loaded single model: {model_key}")
        return model, result
        
    except Exception as e:
        print(f"❌ Error loading {model_key}: {e}")
        return None, result


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================
def load_real_historical_data(STATIONS, STATION_BASE_CAPACITY, 
                               historical_cache='historical_data_cache_2023_2024.pkl'):
    """Load historical data - USE WITH CAUTION! This loads data into memory."""
    global historical_entry, historical_exit, hourly_avg_entry, hourly_avg_exit
    global dow_avg_entry, dow_avg_exit, direction_counts
    
    print(f"\nLOADING HISTORICAL DATA...")
    
    if os.path.exists(historical_cache):
        try:
            with open(historical_cache, 'rb') as f:
                cache_data = pickle.load(f)
            
            historical_entry = cache_data.get('historical_entry', {})
            historical_exit = cache_data.get('historical_exit', {})
            direction_counts = cache_data.get('direction_counts', {})
            hourly_avg_entry = cache_data.get('hourly_avg_entry', {})
            hourly_avg_exit = cache_data.get('hourly_avg_exit', {})
            dow_avg_entry = cache_data.get('dow_avg_entry', {})
            dow_avg_exit = cache_data.get('dow_avg_exit', {})
            
            print(f"Loaded cached historical data: {len(historical_entry)} stations")
            
            if historical_entry:
                return {
                    'historical_entry': historical_entry,
                    'historical_exit': historical_exit,
                    'hourly_avg_entry': hourly_avg_entry,
                    'hourly_avg_exit': hourly_avg_exit,
                    'dow_avg_entry': dow_avg_entry,
                    'dow_avg_exit': dow_avg_exit,
                    'direction_counts': direction_counts
                }
        except Exception as e:
            print(f"Cache error: {e}")
    
    print("No cache found, generating synthetic historical data...")
    _generate_synthetic_historical_data(STATIONS, STATION_BASE_CAPACITY)
    
    return {
        'historical_entry': historical_entry,
        'historical_exit': historical_exit,
        'hourly_avg_entry': hourly_avg_entry,
        'hourly_avg_exit': hourly_avg_exit,
        'dow_avg_entry': dow_avg_entry,
        'dow_avg_exit': dow_avg_exit,
        'direction_counts': direction_counts
    }


# ============================================================
# UNLOAD MODEL (Free Memory)
# ============================================================
def unload_model(model_key):
    """Unload a model to free memory"""
    global directional_models, directional_scalers
    
    if model_key in directional_models:
        del directional_models[model_key]
        print(f"🗑️ Unloaded model: {model_key}")
    
    # Remove scalers
    for key in list(directional_scalers.keys()):
        if model_key in key:
            del directional_scalers[key]
    
    gc.collect()


# ============================================================
# SYNTHETIC HISTORICAL DATA (Fallback)
# ============================================================
def _generate_synthetic_historical_data(STATIONS, STATION_BASE_CAPACITY):
    """Generate synthetic historical data (fallback)"""
    global historical_entry, historical_exit, hourly_avg_entry, hourly_avg_exit, direction_counts
    
    for station in STATIONS:
        capacity = STATION_BASE_CAPACITY.get(station, 10000)
        
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
    
    for hour in range(24):
        if 7 <= hour <= 9:
            hourly_avg_entry[hour] = 8000 + (hour - 7) * 500
            hourly_avg_exit[hour] = 2500 + (hour - 7) * 300
        elif 17 <= hour <= 20:
            hourly_avg_entry[hour] = 3500 - (hour - 17) * 300
            hourly_avg_exit[hour] = 7500 + (hour - 17) * 400
        elif hour >= 22 or hour <= 4:
            hourly_avg_entry[hour] = 500
            hourly_avg_exit[hour] = 500
        elif 10 <= hour <= 16:
            hourly_avg_entry[hour] = 4500
            hourly_avg_exit[hour] = 4500
        elif 5 <= hour <= 6:
            hourly_avg_entry[hour] = 1500
            hourly_avg_exit[hour] = 1000
        elif hour == 21:
            hourly_avg_entry[hour] = 4500
            hourly_avg_exit[hour] = 5000
        else:
            hourly_avg_entry[hour] = 3000
            hourly_avg_exit[hour] = 3000
    
    direction_counts = {'northbound': 4500000, 'southbound': 3800000}
    print("Generated synthetic historical data")


# ============================================================
# FAST PREDICTION (Direct Tensor Execution)
# ============================================================
def debug_prediction(station, direction, features):
    """
    Fast prediction using direct tensor execution.
    Features must already be prepared and scaled.
    """
    station_file = STATION_FILE_MAP.get(station, station)
    model_key = f"{station_file}_{direction}"
    
    if model_key not in directional_models:
        print(f"⚠️ Model {model_key} not loaded")
        return None
    
    feature_key = f'{model_key}_feature'
    target_key = f'{model_key}_target'
    
    model = directional_models[model_key]
    feature_scaler = directional_scalers.get(feature_key)
    target_scaler = directional_scalers.get(target_key)
    
    if feature_scaler is None:
        print(f"⚠️ Feature scaler missing")
        return None
    
    # Scale
    features_scaled = feature_scaler.transform(features)
    scaled_sequence = features_scaled.reshape(1, 24, -1)
    
    # FAST: Direct tensor call instead of model.predict()
    input_tensor = tf.convert_to_tensor(scaled_sequence, dtype=tf.float32)
    pred_scaled = model(input_tensor, training=False).numpy()
    
    # Inverse transform
    if target_scaler is not None:
        pred_congestion = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
        pred_congestion = pred_congestion[0][0]
    else:
        pred_congestion = pred_scaled[0][0] * 100
    
    return max(0, min(100, float(pred_congestion)))


# ============================================================
# AUTO-LOAD ALL MODELS AT STARTUP
# ============================================================
def auto_load_all_models(limit=None):
    """Automatically load all models at startup, with optional limit"""
    global directional_models, directional_scalers
    
    print("=" * 50)
    print("🔄 Auto-loading all models...")
    
    correct_dir = "models_2022-2024_v10"
    
    if not os.path.exists(correct_dir):
        print(f"⚠️ Models directory not found: {correct_dir}")
        return 0
    
    files = os.listdir(correct_dir)
    # ✅ FIX: Look for ALL .keras files (including _v10_plus)
    model_files = [f for f in files if f.endswith('.keras')]
    
    if not model_files:
        print(f"⚠️ No model files found in {correct_dir}")
        return 0
    
    if limit:
        model_files = model_files[:limit]
    
    print(f"📦 Loading {len(model_files)} models from {correct_dir}...")
    
    loaded = 0
    failed = 0
    
    for f in model_files:
        # ✅ FIX: Handle all filename patterns
        name = f.replace('_lstm_v10_plus.keras', '').replace('_lstm_enhanced.keras', '').replace('_best.keras', '')
        parts = name.split('_')
        
        if len(parts) >= 2:
            station = ' '.join(parts[:-1])
            direction = parts[-1]
            
            station_file = STATION_FILE_MAP.get(station, station)
            model_key = f"{station_file}_{direction}"
            
            try:
                model_path = os.path.join(correct_dir, f)
                model = tf.keras.models.load_model(model_path, custom_objects={'rmse': rmse})
                
                directional_models[model_key] = model
                
                # Load scalers
                feature_scaler_path = os.path.join(correct_dir, f'{name}_feature_scaler.pkl')
                target_scaler_path = os.path.join(correct_dir, f'{name}_target_scaler.pkl')
                
                if os.path.exists(feature_scaler_path):
                    with open(feature_scaler_path, 'rb') as fs:
                        directional_scalers[f'{model_key}_feature'] = pickle.load(fs)
                
                if os.path.exists(target_scaler_path):
                    with open(target_scaler_path, 'rb') as ts:
                        directional_scalers[f'{model_key}_target'] = pickle.load(ts)
                
                loaded += 1
                if loaded % 5 == 0:
                    print(f"   Loaded {loaded}/{len(model_files)} models...")
                    
            except Exception as e:
                failed += 1
                print(f"   ⚠️ Could not load {model_key}: {e}")
    
    print(f"\n✅ Auto-loaded {loaded}/{len(model_files)} models")
    if failed > 0:
        print(f"   ⚠️ Failed to load {failed} models")
    print(f"   directional_models now has: {len(directional_models)} models")
    print("=" * 50)
    
    return len(directional_models)


# ============================================================
# STATIONS LIST
# ============================================================
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]


# ============================================================
# MODULE LOADED MESSAGE
# ============================================================
print("\n" + "="*50)
print("✅ Using original target scalers from model files (passenger counts)")
print("="*50)

print("=" * 50)
print("✅ model_loader.py loaded successfully!")
print(f"✅ directional_models count: {len(directional_models)} (loaded on demand)")
print(f"✅ directional_scalers count: {len(directional_scalers)} (loaded on demand)")
print("=" * 50)

# DO NOT auto-load here - let the app call when needed
# auto_load_all_models() is available but NOT called