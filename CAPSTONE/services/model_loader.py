# services/model_loader.py - COMPLETE FIXED VERSION

import os
import pickle
import tensorflow as tf

# Global state - these are the actual variables that get imported
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


def load_directional_models(STATIONS, DIRECTIONAL_MODELS_PATH='models_2022-2024_NEW_v3'):
    """
    Load directional LSTM models for all stations
    Updates the GLOBAL directional_models and directional_scalers
    """
    global directional_models, directional_scalers  # ← CRITICAL: Must be here
    
    # Clear existing
    directional_models = {}
    directional_scalers = {}
    
    print(f"\n🚇 LOADING DIRECTIONAL MODELS (2023-2024 REAL DATA)...")
    print(f"📁 Looking in: {DIRECTIONAL_MODELS_PATH}")
    
    if not os.path.exists(DIRECTIONAL_MODELS_PATH):
        print(f"❌ Directory {DIRECTIONAL_MODELS_PATH} not found!")
        print(f"   Current working directory: {os.getcwd()}")
        return directional_models, directional_scalers
    
    print(f"📂 Found directory. Looking for model files...")
    
    models_loaded = 0
    
    for station in STATIONS:
        for direction in ['Northbound', 'Southbound']:
            model_key = f"{station}_{direction}"
            station_underscore = station.replace(' ', '_')
            model_key_underscore = f"{station_underscore}_{direction}"
            
            # Try different possible model paths
            possible_model_paths = [
                f'{DIRECTIONAL_MODELS_PATH}/{model_key}_lstm_enhanced.keras',
                f'{DIRECTIONAL_MODELS_PATH}/{model_key}_best.keras',
                f'{DIRECTIONAL_MODELS_PATH}/{model_key}.keras',
                f'{DIRECTIONAL_MODELS_PATH}/{model_key_underscore}_lstm_enhanced.keras',
                f'{DIRECTIONAL_MODELS_PATH}/{model_key_underscore}_best.keras',
            ]
            
            model_path = None
            for path in possible_model_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"  ✅ Found model: {os.path.basename(path)}")
                    break
            
            if model_path is None:
                continue
            
            # Look for feature scaler
            possible_scaler_paths = [
                f'{DIRECTIONAL_MODELS_PATH}/{model_key}_feature_scaler.pkl',
                f'{DIRECTIONAL_MODELS_PATH}/{model_key_underscore}_feature_scaler.pkl',
            ]
            
            feature_scaler_path = None
            for path in possible_scaler_paths:
                if os.path.exists(path):
                    feature_scaler_path = path
                    print(f"  ✅ Found feature scaler: {os.path.basename(path)}")
                    break
            
            # Look for target scaler
            target_scaler_path = f'{DIRECTIONAL_MODELS_PATH}/{model_key}_target_scaler.pkl'
            if not os.path.exists(target_scaler_path):
                target_scaler_path = None
            
            if feature_scaler_path:
                try:
                    print(f"  🔧 Loading {model_key}...")
                    directional_models[model_key] = tf.keras.models.load_model(
                        model_path, 
                        compile=False
                    )
                    with open(feature_scaler_path, 'rb') as f:
                        directional_scalers[f'{model_key}_feature'] = pickle.load(f)
                    
                    if target_scaler_path:
                        with open(target_scaler_path, 'rb') as f:
                            directional_scalers[f'{model_key}_target'] = pickle.load(f)
                    
                    models_loaded += 1
                    print(f"  ✅ Loaded: {model_key}")
                except Exception as e:
                    print(f"  ⚠️ Error loading {model_key}: {e}")
            else:
                print(f"  ⚠️ Missing scaler for {model_key}, skipping")
    
    print(f"\n📊 Loaded {models_loaded} directional models")
    print(f"📊 Global directional_models now has {len(directional_models)} models")
    
    return directional_models, directional_scalers


def load_real_historical_data(STATIONS, STATION_BASE_CAPACITY, 
                               historical_cache='historical_data_cache_2023_2024.pkl'):
    """
    Load real historical ridership data
    """
    global historical_entry, historical_exit, hourly_avg_entry, hourly_avg_exit
    global dow_avg_entry, dow_avg_exit, direction_counts
    
    print(f"\n📊 LOADING HISTORICAL DATA...")
    
    # Try to load from cache
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
            
            print(f"✅ Loaded cached historical data: {len(historical_entry)} stations")
            
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
            print(f"⚠️ Cache error: {e}")
    
    # Generate synthetic data as fallback
    print("⚠️ No cache found, generating synthetic historical data...")
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


def _generate_synthetic_historical_data(STATIONS, STATION_BASE_CAPACITY):
    """Generate synthetic historical data as fallback"""
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
    
    # Generate hourly patterns
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
    print("✅ Generated synthetic historical data")