import os
import pickle
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

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
    target_scaler = directional_scalers.get(target_key)  # Might be None
    
    # 5. Check input features BEFORE scaling
    print(f"\n4. Input features shape: {features.shape}")
    print(f"   First row, first 5 values: {features[0, :5]}")
    print(f"   Min/Max values in features: {features.min():.3f} / {features.max():.3f}")
    
    # 6. Scale features
    features_scaled = feature_scaler.transform(features)
    print(f"\n5. After scaling:")
    print(f"   First row, first 5 values: {features_scaled[0, :5]}")
    print(f"   Min/Max scaled: {features_scaled.min():.3f} / {features_scaled.max():.3f}")
    
    # 7. Check if scaling worked (should be roughly 0-1 range)
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
        # This is likely your problem - the model outputs scaled values (0-1)
        # But you're treating it as percentage
        pred_congestion = pred_scaled[0][0]
        print(f"   Raw output as percentage: {pred_congestion}")
        print(f"   Raw output * 100: {pred_congestion * 100}")
        
        # If raw output is 0.38, then *100 = 38% (matches your issue!)
        if 0.35 < pred_scaled[0][0] < 0.41:
            print(f"\n   🔴 PROBLEM IDENTIFIED!")
            print(f"   Model output is {pred_scaled[0][0]:.3f} (0-1 scale)")
            print(f"   But you're missing target_scaler to convert to 0-100%")
            print(f"   Expected: multiply by 100 = {pred_scaled[0][0]*100:.1f}%")
            print(f"   But you're not doing this conversion!")
    
    # 10. Clip to valid range
    pred_congestion = max(0, min(100, pred_congestion))
    
    print(f"\n8. FINAL PREDICTION: {pred_congestion:.1f}%")
    
    return pred_congestion

def load_directional_models(STATIONS, DIRECTIONAL_MODELS_PATH='models_2022-2024_NEW_v2w/openclose'):
    global directional_models, directional_scalers
    
    directional_models = {}
    directional_scalers = {}
    
    for station in STATIONS:
        for direction in ['Northbound', 'Southbound']:
            model_key = f"{station}_{direction}"
            
            # Check for BOTH model and both scalers
            model_path = f'{DIRECTIONAL_MODELS_PATH}/{model_key}_lstm_enhanced.keras'
            feature_scaler_path = f'{DIRECTIONAL_MODELS_PATH}/{model_key}_feature_scaler.pkl'
            target_scaler_path = f'{DIRECTIONAL_MODELS_PATH}/{model_key}_target_scaler.pkl'
            
            # Verify all three files exist
            if not all([os.path.exists(model_path), 
                       os.path.exists(feature_scaler_path),
                       os.path.exists(target_scaler_path)]):
                print(f"  ⚠️ Missing files for {model_key}, skipping")
                continue
            
            try:
                print(f"  Loading {model_key}...")
                directional_models[model_key] = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'rmse': rmse}  
                )
                
                with open(feature_scaler_path, 'rb') as f:
                    directional_scalers[f'{model_key}_feature'] = pickle.load(f)
                
                with open(target_scaler_path, 'rb') as f:
                    directional_scalers[f'{model_key}_target'] = pickle.load(f)
                
                print(f"  ✅ Loaded {model_key}")
                
            except Exception as e:
                print(f"  ❌ Error loading {model_key}: {e}")
    
    print(f"\n✅ Loaded {len(directional_models)} directional models")
    return directional_models, directional_scalers

def load_real_historical_data(STATIONS, STATION_BASE_CAPACITY, 
                               historical_cache='historical_data_cache_2023_2024.pkl'):
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

def _generate_synthetic_historical_data(STATIONS, STATION_BASE_CAPACITY):
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