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

def debug_prediction(station, direction, features):
    """Debug why predictions are stuck at 38%"""
    model_key = f"{station}_{direction}"
    
    print(f"\n{'='*50}")
    print(f"DEBUG: {model_key}")
    print(f"{'='*50}")
    
    # 1. Check if model exists
    print(f"1. Model in directional_models: {model_key in directional_models}")
    if model_key not in directional_models:
        print(f"   Available models: {list(directional_models.keys())[:5]}...")
        return None
    
    # 2. Check scalers
    feature_key = f'{model_key}_feature'
    target_key = f'{model_key}_target'
    
    print(f"2. Feature scaler exists: {feature_key in directional_scalers}")
    print(f"3. Target scaler exists: {target_key in directional_scalers}")
    
    if feature_key not in directional_scalers:
        print(f"   Available scalers: {list(directional_scalers.keys())[:5]}...")
        return None
    
    # 4. Get the actual model and scaler
    model = directional_models[model_key]
    feature_scaler = directional_scalers[feature_key]
    target_scaler = directional_scalers.get(target_key)
    
    # 5. Check input features BEFORE scaling
    print(f"\n4. Input features shape: {features.shape}")
    print(f"   First row, first 5 values: {features[0, :5]}")
    print(f"   Min/Max values in features: {features.min():.3f} / {features.max():.3f}")
    
    # 6. Scale features
    features_scaled = feature_scaler.transform(features)
    print(f"\n5. After scaling:")
    print(f"   First row, first 5 values: {features_scaled[0, :5]}")
    print(f"   Min/Max scaled: {features_scaled.min():.3f} / {features_scaled.max():.3f}")
    
    # 7. Check if scaling worked
    if features_scaled.max() > 5 or features_scaled.min() < -5:
        print(f"   ⚠️ WARNING: Scaling seems off! Range too large")
    
    # 8. Make prediction
    print(f"\n6. Making prediction...")
    pred_scaled = model.predict(features_scaled, verbose=0)
    print(f"   Raw model output: {pred_scaled[0]}")
    print(f"   Shape: {pred_scaled.shape}")
    
    # 9. Inverse transform
    if target_scaler is not None:
        print(f"\n7. Using target scaler to inverse transform...")
        pred_congestion = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
        pred_congestion = pred_congestion[0][0]
        print(f"   After inverse transform: {pred_congestion}")
    else:
        print(f"\n7. No target scaler! Converting directly...")
        pred_congestion = pred_scaled[0][0] * 100
        print(f"   Raw output * 100: {pred_congestion}")
    
    # 10. Clip to valid range
    pred_congestion = max(0, min(100, pred_congestion))
    
    print(f"\n8. FINAL PREDICTION: {pred_congestion:.1f}%")
    
    return pred_congestion

def load_directional_models(STATIONS, DIRECTIONAL_MODELS_PATH='models_2022-2024_v8'):
    """
    Load ALL directional models - USE WITH CAUTION!
    This loads all 26 models and uses ~300MB memory.
    Only use if you have enough memory.
    """
    global directional_models, directional_scalers
    
    directional_models = {}
    directional_scalers = {}
    
    station_map = {
        "North Avenue": "North Ave",
        "Quezon Avenue": "Quezon Ave",
        "Shaw Boulevard": "Shaw Blvd",
        "Ayala Avenue": "Ayala Ave",
        "Boni Avenue": "Boni Ave",
    }
    
    print(f"📂 Loading models from: {DIRECTIONAL_MODELS_PATH}")
    
    # Check if directory exists
    if not os.path.exists(DIRECTIONAL_MODELS_PATH):
        print(f"❌ Directory not found: {DIRECTIONAL_MODELS_PATH}")
        print(f"   Current working directory: {os.getcwd()}")
        return directional_models, directional_scalers
    
    for station in STATIONS:
        for direction in ['Northbound', 'Southbound']:
            
            station_file = STATION_FILE_MAP.get(station, station)
            model_key = f"{station_file}_{direction}"
            
            # Try both naming conventions
            model_path_v1 = os.path.join(DIRECTIONAL_MODELS_PATH, f'{model_key}_lstm_enhanced.keras')
            model_path_v2 = os.path.join(DIRECTIONAL_MODELS_PATH, f'{model_key}_best.keras')
            
            # Use whichever exists
            model_path = None
            if os.path.exists(model_path_v1):
                model_path = model_path_v1
            elif os.path.exists(model_path_v2):
                model_path = model_path_v2
            
            if model_path is None:
                print(f"  ⚠️ No model file for {model_key}")
                continue
            
            feature_scaler_path = os.path.join(DIRECTIONAL_MODELS_PATH, f'{model_key}_feature_scaler.pkl')
            target_scaler_path = os.path.join(DIRECTIONAL_MODELS_PATH, f'{model_key}_target_scaler.pkl')
            
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
                    print(f"    Will use fallback: multiply by 100")
                
                # Force garbage collection after each model
                gc.collect()
                
            except Exception as e:
                print(f"  ❌ Error loading {model_key}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n✅ Loaded {len(directional_models)} directional models")
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

def load_single_model(station, direction, model_path='models_2022-2024_v8'):
    """
    Load a SINGLE model for one station-direction (memory efficient)
    This is the recommended way to load models.
    Only ~20-30MB per model.
    """
    station_file = STATION_FILE_MAP.get(station, station)
    model_key = f"{station_file}_{direction}"
    result = {'model': None, 'feature': None, 'target': None}
    
    # Try both naming conventions
    model_path_v1 = os.path.join(model_path, f'{model_key}_lstm_enhanced.keras')
    model_path_v2 = os.path.join(model_path, f'{model_key}_best.keras')
    
    model_file = None
    if os.path.exists(model_path_v1):
        model_file = model_path_v1
    elif os.path.exists(model_path_v2):
        model_file = model_path_v2
    
    if model_file is None:
        print(f"⚠️ No model file for {model_key}")
        return None, result
    
    try:
        # Load the model
        model = tf.keras.models.load_model(model_file, custom_objects={'rmse': rmse})
        result['model'] = model
        
        # Load feature scaler
        feature_scaler_path = os.path.join(model_path, f'{model_key}_feature_scaler.pkl')
        target_scaler_path = os.path.join(model_path, f'{model_key}_target_scaler.pkl')
        
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
    tf.keras.backend.clear_session()

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

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

# ==================================================
# NO AUTO-LOADING - Everything loads on demand
# ==================================================
print("=" * 50)
print("⚠️ AUTO-LOADING DISABLED - Models load on first request")
print("=" * 50)

print("\n" + "="*50)
print("✅ Using original target scalers from model files (passenger counts)")
print("="*50)

print("=" * 50)
print("✅ model_loader.py loaded successfully!")
print(f"✅ directional_models count: {len(directional_models)} (loaded on demand)")
print(f"✅ directional_scalers count: {len(directional_scalers)} (loaded on demand)")
print("=" * 50)

# ==================================================
# AUTO-LOAD MODELS AT STARTUP
# ==================================================
# At the bottom of model_loader.py, replace the auto-load section with:

def auto_load_all_models(limit=None):
    """Automatically load all models at startup, with optional limit"""
    global directional_models, directional_scalers
    
    print("=" * 50)
    print("🔄 Auto-loading all models...")
    
    correct_dir = "models_2022-2024_v8"
    
    if not os.path.exists(correct_dir):
        print(f"⚠️ Models directory not found: {correct_dir}")
        return 0
    
    files = os.listdir(correct_dir)
    model_files = [f for f in files if f.endswith('.keras') and ('lstm_enhanced' in f or 'best' in f)]
    
    if not model_files:
        print(f"⚠️ No model files found in {correct_dir}")
        return 0
    
    # Limit if specified (for testing)
    if limit:
        model_files = model_files[:limit]
    
    print(f"📦 Loading {len(model_files)} models from {correct_dir}...")
    
    loaded = 0
    failed = 0
    
    for f in model_files:
        # Extract station and direction from filename
        name = f.replace('_lstm_enhanced.keras', '').replace('_best.keras', '')
        parts = name.split('_')
        
        if len(parts) >= 2:
            station = ' '.join(parts[:-1])
            direction = parts[-1]
            
            # Get the station file name
            station_file = STATION_FILE_MAP.get(station, station)
            model_key = f"{station_file}_{direction}"
            
            try:
                # Load the model
                model_path = os.path.join(correct_dir, f)
                model = tf.keras.models.load_model(model_path, custom_objects={'rmse': rmse})
                
                # Store in global dictionary
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

# ==================================================
# NO AUTO-LOAD - Everything loads on demand
# ==================================================
print("=" * 50)
print("✅ model_loader.py loaded - NO models loaded yet")
print(f"✅ directional_models: {len(directional_models)} (empty - lazy loading)")
print(f"✅ directional_scalers: {len(directional_scalers)} (empty - lazy loading)")
print("=" * 50)

# DO NOT auto-load here - let the app call when needed
# auto_load_all_models() is available but NOT called