from flask import Blueprint, request, jsonify, current_app
from extensions import cache
from datetime import datetime, timedelta
from services.feature_engineering import get_feature_sequence_for_station
from config import Config
import numpy as np
import math
from constants import MRT3_PLATFORM_CAPACITY
import tensorflow as tf 


api_predict_bp = Blueprint('api_predict', __name__)

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guaadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

import json
import os
import pickle
# Add at the top with other imports
import time

# ========== REQUEST-LEVEL CACHE ==========
_REQUEST_CACHE = {}
_REQUEST_CACHE_TTL = 300  # 10 seconds
_P90_CACHE = {}  # Add this global variable
_P90_FILE = 'p90_percentiles.json'
_PENDING_CORRECTION_FACTORS = {}

def get_cached_prediction(station_name, direction, target_datetime):
    """Get prediction with request-level caching to prevent duplicate calls"""
    if target_datetime is None:
        target_datetime = Config.get_current_time()
    
    # Create cache key
    cache_key = f"{station_name}_{direction}_{target_datetime.strftime('%Y%m%d%H')}_{target_datetime.minute // 5}"
    
    # Clean old cache entries
    current_time = time.time()
    for key in list(_REQUEST_CACHE.keys()):
        if current_time - _REQUEST_CACHE[key]['timestamp'] > _REQUEST_CACHE_TTL:
            del _REQUEST_CACHE[key]
    
    # Return cached value if exists
    if cache_key in _REQUEST_CACHE:
        return _REQUEST_CACHE[cache_key]['value']
    
    return None

def set_cached_prediction(station_name, direction, target_datetime, value):
    """Store prediction in request cache"""
    if target_datetime is None:
        target_datetime = Config.get_current_time()
    
    cache_key = f"{station_name}_{direction}_{target_datetime.strftime('%Y%m%d%H')}_{target_datetime.minute // 5}"
    
    _REQUEST_CACHE[cache_key] = {
        'value': value,
        'timestamp': time.time()
    }

# ========== P90 CACHE - USING APP CONFIG ==========
_P90_FILE = 'p90_percentiles.json'
CORRECTION_FILE = 'correction_factors.pkl'

def get_p90_cache():
    """Return the P90 cache dictionary (was get_p95_cache)"""
    return _P90_CACHE


_PENDING_CORRECTION_FACTORS = {}

def get_correction_factors():
    """Get correction factors from app config"""
    try:
        if 'CORRECTION_FACTORS' not in current_app.config:
            # Check if we have pending factors
            global _PENDING_CORRECTION_FACTORS
            if _PENDING_CORRECTION_FACTORS:
                current_app.config['CORRECTION_FACTORS'] = _PENDING_CORRECTION_FACTORS
                _PENDING_CORRECTION_FACTORS = {}
            else:
                current_app.config['CORRECTION_FACTORS'] = {}
        return current_app.config['CORRECTION_FACTORS']
    except RuntimeError:
        # Working outside of application context
        return _PENDING_CORRECTION_FACTORS
# In api_predict.py - replace get_p95_percentile with get_p90_percentile

def get_p90_percentile(station_name, direction):
    """Get P90 using app config cache (persistent across requests)"""
    key = f"{station_name}_{direction}"
    
    # Get cache from app config
    p90_cache = get_p90_cache()
    
    # Check if already cached
    if key in p90_cache:
        return p90_cache[key]
    
    # Compute P90
    from services.feature_engineering import get_station_dataframe_cached
    hourly = get_station_dataframe_cached(station_name, direction)
    
    if hourly is not None and len(hourly) > 0:
        # Use TotalPassenger directly
        passengers = hourly['TotalPassenger'].values
        non_zero = passengers[passengers > 0]
        
        if len(non_zero) > 0:
            # Use the 90th percentile of passenger counts (changed from 95)
            p90 = np.percentile(non_zero, 90)
            
            # Ensure minimum value
            p90 = max(p90, 100)  # At least 100 passengers
            
            # Cap at a reasonable maximum
            p99 = np.percentile(non_zero, 99)
            max_reasonable = p99 * 1.5
            if p90 > max_reasonable:
                p90 = max_reasonable
        else:
            p90 = 1000  # Fallback
        
        # Store in cache
        p90_cache[key] = float(p90)
        
        # Save to disk
        try:
            p90_file = 'p90_percentiles.json'  # Changed from P95_FILE
            if os.path.exists(p90_file):
                with open(p90_file, 'r') as f:
                    all_p90 = json.load(f)
            else:
                all_p90 = {}
            all_p90[key] = float(p90)
            with open(p90_file, 'w') as f:
                json.dump(all_p90, f, indent=2)
        except:
            pass
        
        return p90_cache[key]
    
    # Fallback
    fallback = 1000
    p90_cache[key] = fallback
    return fallback

# Add cache function
def get_p90_cache():
    """Get P90 cache from app config (persistent across requests)"""
    if 'P90_CACHE' not in current_app.config:
        current_app.config['P90_CACHE'] = {}
    return current_app.config['P90_CACHE']

# Preload P90 cache
def preload_p90_cache():
    """
    Preload all P90 values from file or calculate them
    Renamed from preload_p95_cache
    """
    
    from services.feature_engineering import STATION_NUMBERS
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p90_file = os.path.join(script_dir, _P90_FILE)
    
    if os.path.exists(p90_file):
        try:
            with open(p90_file, 'r') as f:
                p90_data = json.load(f)
            for key, value in p90_data.items():
                _P90_CACHE[key] = value
            print(f"✅ Loaded {len(_P90_CACHE)} P90 values from cache")
            return True
        except Exception as e:
            print(f"⚠️ Error loading P90 cache: {e}")
    
    # Calculate if file doesn't exist
    print("🔄 Calculating P90 values from data...")
    stations = list(STATION_NUMBERS.keys())
    directions = ['Northbound', 'Southbound']
    
    for station in stations:
        for direction in directions:
            get_p90_percentile(station, direction)
    
    print(f"✅ Preloaded {len(_P90_CACHE)} P90 values")
    return True
def safe_cache_key(prefix):
    """Generate cache key safely handling None request"""
    try:
        from flask import request
        if request is not None:
            if hasattr(request, 'view_args') and request.view_args:
                station = request.view_args.get('station_name', 'unknown')
            else:
                station = 'unknown'
        else:
            station = 'unknown'
    except:
        station = 'unknown'
    
    return f"{prefix}_{station}_{datetime.now().hour}_{datetime.now().minute // 5}"

def load_correction_factors():
    """Load correction factors into app config"""
    correction_factors = get_correction_factors()
    
    if os.path.exists(CORRECTION_FILE):
        try:
            with open(CORRECTION_FILE, 'rb') as f:
                factors = pickle.load(f)
                correction_factors.update(factors)
            print(f"✅ Loaded {len(factors)} correction factors into app config")
        except Exception as e:
            print(f"⚠️ Could not load correction factors: {e}")

# Call at module load
#load_correction_factors()

# ========== PRELOAD TYPICAL PATTERNS ==========
def preload_typical_patterns():
    """Pre-compute typical patterns for all station-direction pairs to avoid cold-start overhead."""
    from services.feature_engineering import get_typical_pattern
    from flask import current_app

    stations = current_app.config.get('STATIONS', STATIONS)
    directions = ['Northbound', 'Southbound']

    print("🔄 Preloading typical patterns for all stations...")
    for station in stations:
        for direction in directions:
            try:
                # This function builds and caches the typical pattern internally
                pattern = get_typical_pattern(station, direction)
                if pattern is not None:
                    print(f"✅ Preloaded {station} {direction} (length {len(pattern)})")
            except Exception as e:
                print(f"⚠️ Failed to preload {station} {direction}: {e}")
    print("✅ Typical patterns preloaded.")

# ========== MODEL CACHE ==========
_models_cache = None
_scalers_cache = None

def get_models():
    """Get models from cache or app config"""
    global _models_cache, _scalers_cache
    
    if _models_cache is not None:
        return _models_cache, _scalers_cache
    
    # Get from app config only
    directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    
    if directional_models:
        _models_cache = directional_models
        _scalers_cache = directional_scalers
        return _models_cache, _scalers_cache
    
    return None, None

def set_models(models, scalers):
    """Set models in cache"""
    global _models_cache, _scalers_cache
    _models_cache = models
    _scalers_cache = scalers

# Add this debug endpoint to see the last 24 hours
@api_predict_bp.route('/debug/last-24-hours/<station_name>/<direction>')
def debug_last_24_hours(station_name, direction):
    from services.feature_engineering import get_station_dataframe
    
    station = station_name.replace('%20', ' ')
    df = get_station_dataframe(station, direction)
    
    if df is None:
        return jsonify({"error": "No data"})
    
    last_24 = df.tail(24)
    
    return jsonify({
        "station": station,
        "direction": direction,
        "last_24_hours": {
            "passengers": last_24['TotalPassenger'].tolist(),
            "min": float(last_24['TotalPassenger'].min()),
            "max": float(last_24['TotalPassenger'].max()),
            "mean": float(last_24['TotalPassenger'].mean()),
            "dates": [idx.isoformat() for idx in last_24.index]
        }
    })
  
def ensure_models_loaded(station_name=None, direction=None):
    """Ensure models are loaded - calls the lazy loader from app"""
    ensure_fn = current_app.config.get('ENSURE_MODELS_LOADED')
    
    if not ensure_fn:
        print("⚠️ No ensure function found in app config")
        return
    
    # This will load models on first call, then return fast
    ensure_fn()
    
    # Update local cache from app config
    models = current_app.config.get('DIRECTIONAL_MODELS', {})
    scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    
    if models:
        set_models(models, scalers)
    
    if station_name and direction:
        ensure_fn(station_name, direction)
    elif station_name:
        for d in ['Northbound', 'Southbound']:
            ensure_fn(station_name, d)
    else:
        ensure_fn("North Ave", "Northbound")
        ensure_fn("North Ave", "Southbound")
def get_raw_prediction(station_name, direction, target_datetime):
    """Returns the raw passenger count from the model."""
    
    ensure_models_loaded(station_name, direction)
    directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    
    model_key = f"{station_name}_{direction}"
    if model_key not in directional_models:
        return None

    try:
        # ✅ FIX: get_feature_sequence_for_station returns SCALED features
        from services.feature_engineering import get_scaled_feature_sequence
        features_scaled = get_scaled_feature_sequence(station_name, direction, target_datetime)
        if features_scaled is None:
            return None

        # ✅ FIX: Get target scaler only
        target_scaler = directional_scalers.get(f'{model_key}_target')
        if target_scaler is None:
            return None

        # ✅ FIX: features_scaled is already scaled, just reshape
        input_sequence = features_scaled.reshape(1, 24, -1)
        raw_scaled = directional_models[model_key].predict(input_sequence, verbose=0)[0][0]
        passenger_count = float(target_scaler.inverse_transform([[raw_scaled]])[0][0])
        
        if passenger_count < 0:
            passenger_count = 0
            
        return passenger_count
    except Exception as e:
        print(f"⚠️ get_raw_prediction error: {e}")
        return None

def compute_and_save_correction_factors(test_days=30, end_date=None):
    """
    Computes correction factors for all station-directions by comparing
    model predictions to actual historical passenger counts.
    Saves to correction_factors.pkl.
    """
    
    from services.feature_engineering import get_station_dataframe
    import numpy as np
    import pickle
    from datetime import timedelta

    factors = {}
    if end_date is None:
        end_date = Config.get_current_time()
    start_date = end_date - timedelta(days=test_days)

    for station in STATIONS:
        for direction in ['Northbound', 'Southbound']:
            model_key = f"{station}_{direction}"
            # Only compute if model exists
            if model_key not in current_app.config.get('DIRECTIONAL_MODELS', {}):
                continue

            # Get actual historical hourly data for this station-direction
            df = get_station_dataframe(station, direction)
            if df is None or len(df) == 0:
                continue

            # Filter to test period
            mask = (df.index >= start_date) & (df.index < end_date)
            test_df = df[mask]
            if len(test_df) == 0:
                continue

            ratios = []
            for timestamp, row in test_df.iterrows():
                actual = row['TotalPassenger']
                if actual <= 0:
                    continue

                pred = get_raw_prediction(station, direction, timestamp)
                if pred is None or pred <= 0:
                    continue

                ratio = actual / pred
                # Avoid extreme outliers
                if 0.1 < ratio < 10:
                    ratios.append(ratio)

            if ratios:
                factor = np.median(ratios)
                # Clamp to [0.5, 1.5] to avoid over‑/under‑scaling
                factor = min(1.5, factor)
                factors[model_key] = factor
                print(f"✅ {station} {direction}: factor = {factor:.3f} (based on {len(ratios)} samples)")

    # Save
    with open(CORRECTION_FILE, 'wb') as f:
        pickle.dump(factors, f)
    print(f"✅ Saved {len(factors)} correction factors to {CORRECTION_FILE}")
    return factors

# ========== LAZY LOAD HISTORICAL PEAKS - ONLY WHEN NEEDED ==========
# Use app config for historical peaks too
def get_historical_peak(station_name, direction):
    """Lazy load historical peak for just one station-direction"""
    key = f"{station_name}_{direction}"
    
    # Get from app config
    if 'HISTORICAL_PEAKS' not in current_app.config:
        current_app.config['HISTORICAL_PEAKS'] = {}
    historical_peaks = current_app.config['HISTORICAL_PEAKS']
    
    # Check if already loaded
    if key in historical_peaks:
        return historical_peaks[key]
    
    # Compute just this one
    from services.feature_engineering import get_station_dataframe_cached  
    import numpy as np
    
    hourly = get_station_dataframe_cached(station_name, direction)
    if hourly is not None and len(hourly) > 0:
        passengers = hourly['TotalPassenger'].values
        peak_abs = float(passengers.max())
        historical_peaks[key] = {
            "peak": peak_abs,
            "absolute_max": peak_abs,
            "percentile": 100
        }
        return historical_peaks[key]
    
    return None

def get_active_overrides():
    """Get active overrides - uses Config time for expiry check"""
    overrides_file = 'overrides.json'
    
    if os.path.exists(overrides_file):
        try:
            with open(overrides_file, 'r') as f:
                all_overrides = json.load(f)
            
            # Get current time from Config (not system time)
            config_time = Config.get_current_time()
            now_timestamp = config_time.timestamp()
            
            active = {}
            for key, override in all_overrides.items():
                expiry = override.get('expiry')
                
                # Use Config time for expiry check
                if expiry is None or expiry > now_timestamp:
                    active[key] = override
                else:
                    print(f"⏰ Override expired: {key} (expiry: {expiry}, config_time: {now_timestamp})")
            
            return active
        except Exception as e:
            print(f"Error loading overrides: {e}")
            return {}
    
    return {}
 
@api_predict_bp.route('/debug/quick-check/<station_name>')
def debug_quick_check(station_name):
    """Quick debug showing what's happening with the data"""
    from services.feature_engineering import get_station_dataframe, FEATURE_COLS
    import numpy as np
    
    station = station_name.replace('%20', ' ')
    now = Config.get_current_time()
    results = {}
    
    for direction in ['Northbound', 'Southbound']:
        try:
            df = get_station_dataframe(station, direction)
            
            if df is None or len(df) == 0:
                results[direction] = {'error': 'No data'}
                continue
            
            # Last 24 hours
            last_24 = df.tail(24)
            zero_count = int((last_24['TotalPassenger'] == 0).sum())
            zero_ratio = float(zero_count / 24)
            
            # Historical averages (operating hours only)
            operating_df = df[(df.index.hour >= 5) & (df.index.hour < 23)]
            hourly_avg = operating_df.groupby(operating_df.index.hour)['TotalPassenger'].mean()
            
            # Get P90 (changed from P95)
            try:
                p90 = get_p90_percentile(station, direction)
            except:
                p90 = 0
            
            results[direction] = {
                'last_24_hours': {
                    'min': float(last_24['TotalPassenger'].min()),
                    'max': float(last_24['TotalPassenger'].max()),
                    'mean': float(last_24['TotalPassenger'].mean()),
                    'zero_count': zero_count,
                    'zero_ratio': zero_ratio,
                    'is_anomalous': zero_ratio > 0.5
                },
                'historical_averages': {
                    f"{hour:02d}:00": round(float(hourly_avg.get(hour, 0)), 0)
                    for hour in [6, 7, 8, 9, 12, 17, 18, 19, 20, 21]
                    if hour in hourly_avg.index
                },
                'p90': float(p90) if p90 else 0,  # Changed from p95
                'capacity': MRT3_PLATFORM_CAPACITY.get(station, 1000),
                'data_stats': {
                    'total_hours': int(len(df)),
                    'non_zero_hours': int((df['TotalPassenger'] > 0).sum()),
                    'max_passenger': float(df['TotalPassenger'].max())
                }
            }
            
        except Exception as e:
            results[direction] = {'error': str(e)}
    
    return jsonify({
        'station': station,
        'time': now.isoformat(),
        'results': results,
        'recommendation': 'If last_24_hours.is_anomalous is True, use historical averages'
    })
    
@api_predict_bp.route('/debug/cache-keys/<station_name>')
def debug_cache_keys(station_name):
    """Debug what cache keys exist for a station."""
    cache_instance = current_app.extensions.get('cache')
    
    if not cache_instance:
        return jsonify({'error': 'Cache not found'}), 500
    
    keys = []
    for hour in range(24):
        key = f"forecast_{station_name}_{hour}"
        value = cache_instance.get(key)
        if value is not None:
            keys.append({
                'key': key,
                'exists': True,
                'has_value': value is not None
            })
    
    return jsonify({
        'station': station_name,
        'cache_keys_found': keys,
        'total_found': len(keys)
    })
 
 
@api_predict_bp.route('/debug/historical-patterns/<station_name>')
def historical_patterns(station_name):
    """Check actual historical patterns for comparison"""
    from services.feature_engineering import get_station_dataframe
    import numpy as np
    
    station = station_name.replace('%20', ' ')
    results = {}
    
    # Get actual predictions for this station
    north_preds = {}
    south_preds = {}
    now = Config.get_current_time()
    
    for direction in ['Northbound', 'Southbound']:
        df = get_station_dataframe(station, direction)
        if df is not None:
            # Get average by hour from 2022-2024
            hourly_avg = df.groupby(df.index.hour)['TotalPassenger'].mean()
            hourly_std = df.groupby(df.index.hour)['TotalPassenger'].std()
            
            # Convert to congestion using p90 (changed from p95)
            p90 = get_p90_percentile(station, direction)
            
            results[direction] = {
                "hourly_avg_congestion": {
                    f"{hour}:00": round((hourly_avg[hour] / p90) * 100, 1)
                    for hour in range(6, 22) if hour in hourly_avg.index
                },
                "expected_rush_hour": "7-9 AM" if direction == "Southbound" else "5-7 PM"
            }
            
            # Get model predictions for comparison
            for hour in range(6, 22):
                test_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                if direction == 'Northbound':
                    north_preds[f"{hour}:00"] = get_directional_prediction(station, 'Northbound', test_time)
                else:
                    south_preds[f"{hour}:00"] = get_directional_prediction(station, 'Southbound', test_time)
    
    return jsonify({
        "station": station,
        "historical_patterns": results,
        "your_predictions": {
            "northbound": {h: round(v,1) for h,v in north_preds.items()},
            "southbound": {h: round(v,1) for h,v in south_preds.items()}
        }
    })
    
@api_predict_bp.route('/debug/model-weights/<station_name>')
def debug_model_weights(station_name):
    """Check if model weights are balanced"""
    station = station_name.replace('%20', ' ')
    directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
    
    results = {}
    for direction in ['Northbound', 'Southbound']:
        model_key = f"{station}_{direction}"
        if model_key in directional_models:
            model = directional_models[model_key]
            # Get weights of the last layer
            weights = model.get_weights()
            results[direction] = {
                "layer_count": len(weights),
                "last_layer_shape": weights[-1].shape if weights else None,
                "weight_stats": {
                    "mean": float(weights[-1].mean()) if weights else None,
                    "std": float(weights[-1].std()) if weights else None,
                    "min": float(weights[-1].min()) if weights else None,
                    "max": float(weights[-1].max()) if weights else None
                } if weights else None
            }
    
    return jsonify(results)

@api_predict_bp.route('/debug/override-status')
def debug_override_status():
    """Debug what overrides are active and where they're coming from."""
    from config import Config
    
    # Get from config
    config_overrides = current_app.config.get('overrides', {})
    
    # Get from file
    file_overrides = {}
    if os.path.exists('overrides.json'):
        try:
            with open('overrides.json', 'r') as f:
                file_overrides = json.load(f)
        except:
            pass
    
    # Get active (uses the function)
    active = get_active_overrides()
    
    return jsonify({
        'config_overrides': config_overrides,
        'file_overrides': file_overrides,
        'active_overrides': active,
        'config_has_overrides': 'overrides' in current_app.config,
        'current_time': Config.get_current_time().isoformat()
    })
def get_directional_prediction(station_name, direction, target_datetime=None):
    """Get directional prediction with request-level caching."""
    
    if target_datetime is None:
        target_datetime = Config.get_current_time()
    
    # Check request cache first
    cached = get_cached_prediction(station_name, direction, target_datetime)
    if cached is not None:
        return cached
    
    ensure_models_loaded(station_name, direction)
    
    # Check if MRT is closed
    hour = target_datetime.hour
    minute = target_datetime.minute
    current_time_decimal = hour + minute / 60
    
    OPERATING_START = 4.5
    OPERATING_END = 22.5
    
    if current_time_decimal < OPERATING_START or current_time_decimal >= OPERATING_END:
        return 0
    
    directional_models, directional_scalers = get_models()
    
    if directional_models is None:
        return _get_operating_hours_fallback(target_datetime)
    
    model_key = f"{station_name}_{direction}"
    
    if model_key not in directional_models:
        return _get_operating_hours_fallback(target_datetime)
    
    try:
        # ========== ✅ FIX: Use get_scaled_feature_sequence directly ==========
        # This returns ALREADY SCALED features - DO NOT apply feature scaler again!
        from services.feature_engineering import get_scaled_feature_sequence
        features_scaled = get_scaled_feature_sequence(station_name, direction, target_datetime)
        
        if features_scaled is None:
            return _get_operating_hours_fallback(target_datetime)
        
        # ========== ✅ FIX: Get target scaler ONLY ==========
        target_scaler = directional_scalers.get(f'{model_key}_target')
        if target_scaler is None:
            return _get_operating_hours_fallback(target_datetime)
        
        # ========== ✅ FIX: features_scaled is already scaled ==========
        input_sequence = features_scaled.reshape(1, 24, -1)
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.float32)
        
        prediction_scaled = directional_models[model_key](input_tensor, training=False).numpy()
        raw_output = float(prediction_scaled[0][0])
        
        passenger_count = float(target_scaler.inverse_transform([[raw_output]])[0][0])
        
        # Get P90 (changed from P95)
        p90 = get_p90_percentile(station_name, direction)
        if p90 <= 0:
            p90 = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
        
        # Convert to congestion
        congestion = (passenger_count / p90) * 100
        congestion = max(0, min(congestion, 100))
        
        # Apply correction factor
        correction_factors = get_correction_factors()
        factor = correction_factors.get(model_key, 1.0)
        congestion = congestion * factor
        congestion = max(0, min(congestion, 100))
        
        # Day of week adjustment
        dow = target_datetime.weekday()
        if dow >= 5:
            dow_factor = 0.7
        elif dow == 4:
            dow_factor = 1.1
        elif dow == 0:
            dow_factor = 1.05
        else:
            dow_factor = 1.0
        
        congestion = congestion * dow_factor
        congestion = max(0, min(congestion, 100))
        
        # Store in request cache
        set_cached_prediction(station_name, direction, target_datetime, congestion)
        
        return congestion
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return _get_operating_hours_fallback(target_datetime)
    
def get_fallback_directional_prediction(station_name, direction, target_datetime=None):
    """Fallback prediction when models aren't available"""
    from config import Config
    target_datetime = target_datetime or Config.get_current_time()
    hour = target_datetime.hour
    
    # Check operating hours
    current_time = hour + target_datetime.minute / 60
    if current_time < 4.5 or current_time >= 22.5:
        return 0
    
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        return 65
    elif 10 <= hour <= 16:
        return 45
    else:
        return 25
@api_predict_bp.route('/debug/verify-features/<station_name>/<direction>')
def debug_verify_features(station_name, direction):
    """Verify what features are being fed to the model."""
    # ✅ FIX: Use get_scaled_feature_sequence
    from services.feature_engineering import get_scaled_feature_sequence
    
    station = station_name.replace('%20', ' ')
    now = Config.get_current_time()
    
    ensure_models_loaded(station, direction)
    directional_models, directional_scalers = get_models()
    
    model_key = f"{station}_{direction}"
    
    result = {
        "station": station,
        "direction": direction,
        "time": now.isoformat(),
        "model_loaded": model_key in directional_models
    }
    
    if model_key not in directional_models:
        result["error"] = f"Model {model_key} not loaded"
        return jsonify(result)
    
    try:
        # ✅ FIX: Already-scaled features
        features_scaled = get_scaled_feature_sequence(station, direction, now)
        
        if features_scaled is None:
            result["error"] = "No features returned"
            return jsonify(result)
        
        result["feature_stats"] = {
            "shape": features_scaled.shape,
            "min": float(features_scaled.min()),
            "max": float(features_scaled.max()),
            "mean": float(features_scaled.mean()),
            "std": float(features_scaled.std()),
            "sample_first_10": features_scaled[0, :10].tolist() if features_scaled.ndim > 1 else features_scaled[:10].tolist()
        }
        
        # Check if features are already scaled (should be between ~-2 and 2 for StandardScaler)
        feature_min = float(features_scaled.min())
        feature_max = float(features_scaled.max())
        
        result["scaling_check"] = {
            "likely_scaled": -3 < feature_min and feature_max < 3,
            "explanation": "Features appear to be already scaled if min/max are within [-3, 3] range"
        }
        
        # Try prediction
        target_scaler = directional_scalers.get(f'{model_key}_target')
        if target_scaler:
            input_sequence = features_scaled.reshape(1, 24, -1)
            input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.float32)
            
            prediction_scaled = directional_models[model_key](input_tensor, training=False).numpy()
            raw_output = float(prediction_scaled[0][0])
            
            passenger_count = float(target_scaler.inverse_transform([[raw_output]])[0][0])
            
            result["prediction"] = {
                "raw_scaled_output": raw_output,
                "passenger_count": passenger_count
            }
            
    except Exception as e:
        import traceback
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    
    return jsonify(result)

def clamp_prediction_by_time(congestion, target_datetime):
    """Clamp prediction based on time of day"""
    from config import Config
    target_datetime = target_datetime or Config.get_current_time()
    hour = target_datetime.hour
    
    # If MRT is closed, return 0
    current_time = hour + target_datetime.minute / 60
    if current_time < 4.5 or current_time >= 22.5:
        return 0
    
    # Clamp to reasonable ranges based on time
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        return max(30, min(congestion, 100))
    elif 10 <= hour <= 16:
        return max(10, min(congestion, 80))
    else:
        return max(0, min(congestion, 60))

def get_best_time_to_travel(station_name=None):
    """Get the best time to travel recommendation"""
    from config import Config
    now = Config.get_current_time()
    hour = now.hour
    
    if 7 <= hour <= 9:
        return "10:00 AM - 3:00 PM (Avoid morning rush hour)"
    elif 17 <= hour <= 20:
        return "Before 5:00 PM or after 8:00 PM (Avoid evening rush hour)"
    return "Now is a good time to travel!"

def get_wait_time(congestion):
    """Get wait time based on congestion percentage"""
    if congestion > 80:
        return "15-20 min"
    elif congestion > 60:
        return "10-15 min"
    elif congestion > 30:
        return "5-10 min"
    else:
        return "2-5 min"
    
@api_predict_bp.route('/debug/simulate-all-stations-day')
def debug_simulate_all_stations_day():
    """
    Simulate predictions for ALL stations for the full day.
    Returns a summary per hour for all stations.
    """
    results = {
        "date": "2025-06-26",
        "hours": {}
    }
    
    for hour in range(4, 24):
        results["hours"][f"{hour:02d}:00"] = {}
        test_time = datetime(2025, 6, 26, hour, 0, 0)
        is_operating = 0 <= (hour + 0/60) < 24
        
        if not is_operating:
            for station in STATIONS:
                results["hours"][f"{hour:02d}:00"][station] = {
                    "avg": 0,
                    "status": "CLOSED"
                }
            continue
        
        for station in STATIONS:
            north_result = _get_directional_prediction_with_details(station, 'Northbound', test_time)
            south_result = _get_directional_prediction_with_details(station, 'Southbound', test_time)
            
            avg = (north_result["congestion"] + south_result["congestion"]) / 2
            
            if avg > 80:
                status = "SEVERE"
            elif avg > 50:
                status = "CONGESTED"
            elif avg > 25:
                status = "MODERATE"
            else:
                status = "LIGHT"
            
            results["hours"][f"{hour:02d}:00"][station] = {
                "avg": round(avg, 1),
                "status": status,
                "northbound": round(north_result["congestion"], 1),
                "southbound": round(south_result["congestion"], 1)
            }
    
    return jsonify(results)
@api_predict_bp.route('/debug/pipeline-details/<station_name>')
def debug_pipeline_details(station_name):
    """Debug the entire prediction pipeline step by step"""
    from services.feature_engineering import get_scaled_feature_sequence
    
    station = station_name.replace('%20', ' ')
    directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    correction_factors = get_correction_factors()
    
    results = {
        "station": station,
        "timestamp": Config.get_current_time().isoformat(),
        "details": {}
    }
    
    now = Config.get_current_time()
    test_hours = [6, 8, 10, 12, 14, 16, 18, 20]
    
    for direction in ['Northbound', 'Southbound']:
        model_key = f"{station}_{direction}"
        if model_key not in directional_models:
            continue
            
        results["details"][direction] = {}
        target_scaler = directional_scalers.get(f'{model_key}_target')
        
        p90 = get_p90_percentile(station, direction)  # Changed from p95
        
        for hour in test_hours:
            test_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            try:
                # ✅ FIX: Get scaled features directly
                features_scaled = get_feature_sequence_for_station(station, direction, test_time)
                if features_scaled is None:
                    continue
                
                # ✅ FIX: Show target scaler info (StandardScaler)
                target_scaler_info = {}
                if target_scaler:
                    if hasattr(target_scaler, 'mean_') and hasattr(target_scaler, 'scale_'):
                        target_scaler_info = {
                            "type": "StandardScaler",
                            "mean": float(target_scaler.mean_[0]),
                            "scale": float(target_scaler.scale_[0])
                        }
                    elif hasattr(target_scaler, 'data_min_') and hasattr(target_scaler, 'data_max_'):
                        target_scaler_info = {
                            "type": "MinMaxScaler",
                            "data_min": float(target_scaler.data_min_[0]),
                            "data_max": float(target_scaler.data_max_[0])
                        }
                
                # ✅ FIX: Make prediction
                input_sequence = features_scaled.reshape(1, 24, -1)
                pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
                raw_output = float(pred_scaled[0][0])
                
                passenger_count = float(target_scaler.inverse_transform([[raw_output]])[0][0])
                factor = correction_factors.get(model_key, 1.0)
                passenger_count = passenger_count * factor
                
                congestion = (passenger_count / p90) * 100  # Changed from p95
                congestion = max(0, min(congestion, 100))
                
                results["details"][direction][f"{hour}:00"] = {
                    "target_scaler": target_scaler_info,
                    "p90_percentile": round(float(p90), 2),  # Changed from p95
                    "correction_factor": round(float(factor), 3),
                    "model_output": {
                        "raw_scaled": round(raw_output, 4),
                        "inverse_transformed_passengers": round(passenger_count, 0),
                        "congestion_percentage": round(congestion, 1)
                    }
                }
                
            except Exception as e:
                results["details"][direction][f"{hour}:00"] = {"error": str(e)}
    
    return jsonify(results)
@api_predict_bp.route('/debug/check-raw-output/<station_name>')
def debug_check_raw_output(station_name):
    """Check raw model output and what it means - USING P90 PERCENTILE"""
    # ✅ FIX: Use get_scaled_feature_sequence
    from services.feature_engineering import get_scaled_feature_sequence
    
    station = station_name.replace('%20', ' ')
    directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    correction_factors = get_correction_factors()
    
    results = []
    now = Config.get_current_time()
    
    for direction in ['Northbound', 'Southbound']:
        model_key = f"{station}_{direction}"
        if model_key not in directional_models:
            continue
            
        target_scaler = directional_scalers.get(f'{model_key}_target')
        
        p90 = get_p90_percentile(station, direction)  # Changed from p95
        factor = correction_factors.get(model_key, 1.0)
        
        try:
            # ✅ FIX: Already-scaled features
            features_scaled = get_scaled_feature_sequence(station, direction, now)
            if features_scaled is not None:
                input_sequence = features_scaled.reshape(1, 24, -1)
                pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
                raw_output = float(pred_scaled[0][0])
                
                passenger_count = float(target_scaler.inverse_transform([[raw_output]])[0][0])
                passenger_count = passenger_count * factor
                
                congestion = (passenger_count / p90) * 100  # Changed from p95
                congestion = max(0, min(congestion, 100))
                
                scaler_info = {}
                if hasattr(target_scaler, 'mean_') and hasattr(target_scaler, 'scale_'):
                    scaler_info = {
                        "type": "StandardScaler",
                        "mean": float(target_scaler.mean_[0]),
                        "scale": float(target_scaler.scale_[0])
                    }
                elif hasattr(target_scaler, 'data_min_') and hasattr(target_scaler, 'data_max_'):
                    scaler_info = {
                        "type": "MinMaxScaler",
                        "data_min": float(target_scaler.data_min_[0]),
                        "data_max": float(target_scaler.data_max_[0])
                    }
                
                results.append({
                    "direction": direction,
                    "raw_model_output": round(raw_output, 4),
                    "scaler_info": scaler_info,
                    "p90_percentile": round(float(p90), 2),  # Changed from p95
                    "correction_factor": round(float(factor), 3),
                    "passenger_count": round(passenger_count, 0),
                    "congestion_percentage": round(congestion, 1)
                })
        except Exception as e:
            results.append({
                "direction": direction,
                "error": str(e)
            })
    
    return jsonify({
        "station": station,
        "time": now.isoformat(),
        "results": results
    })
    
@api_predict_bp.route('/debug/raw-values/<station_name>/<direction>')
def debug_raw_values(station_name, direction):
    """Debug raw model output values"""
    from services.feature_engineering import get_scaled_feature_sequence
    
    station = station_name.replace('%20', ' ')
    now = Config.get_current_time()
    
    ensure_models_loaded(station, direction)
    directional_models, directional_scalers = get_models()
    correction_factors = get_correction_factors()
    
    model_key = f"{station}_{direction}"
    
    result = {
        'station': station,
        'direction': direction,
        'time': now.isoformat(),
        'model_loaded': model_key in directional_models if directional_models else False,
    }
    
    if not directional_models or model_key not in directional_models:
        result['error'] = f'Model {model_key} not loaded'
        return jsonify(result)
    
    try:
        # ✅ FIX: get_feature_sequence_for_station returns SCALED features
        features_scaled = get_feature_sequence_for_station(station, direction, now)
        
        if features_scaled is None:
            result['error'] = 'No features returned'
            return jsonify(result)
        
        # ✅ FIX: Get target scaler ONLY (features are already scaled)
        target_scaler = directional_scalers.get(f'{model_key}_target')
        
        if target_scaler is None:
            result['error'] = 'Missing target scaler'
            return jsonify(result)
        
        # ✅ FIX: features_scaled is already scaled, just reshape
        input_sequence = features_scaled.reshape(1, 24, -1)
        
        # Get prediction
        pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
        raw_output = float(pred_scaled[0][0])
        result['raw_scaled_output'] = raw_output
        
        # Inverse transform
        passenger_count = float(target_scaler.inverse_transform([[raw_output]])[0][0])
        result['passenger_count'] = passenger_count
        
        # Get P90 (changed from P95)
        p90 = get_p90_percentile(station, direction)
        result['p90'] = p90  # Changed from p95
        
        # Calculate congestion
        congestion = (passenger_count / p90) * 100  # Changed from p95
        result['congestion_raw'] = congestion
        result['congestion_clamped'] = max(0, min(congestion, 100))
        
        # Correction factor
        factor = correction_factors.get(model_key, 1.0)
        result['correction_factor'] = factor
        result['passenger_count_corrected'] = passenger_count * factor
        
        # Show feature sample
        result['feature_sample'] = features_scaled[0, :10].tolist()
        
        # ✅ FIX: Target scaler info (StandardScaler)
        if hasattr(target_scaler, 'mean_') and hasattr(target_scaler, 'scale_'):
            result['target_scaler_type'] = 'StandardScaler'
            result['target_scaler_mean'] = float(target_scaler.mean_[0])
            result['target_scaler_scale'] = float(target_scaler.scale_[0])
        elif hasattr(target_scaler, 'data_min_') and hasattr(target_scaler, 'data_max_'):
            result['target_scaler_type'] = 'MinMaxScaler'
            result['target_scaler_min'] = float(target_scaler.data_min_[0])
            result['target_scaler_max'] = float(target_scaler.data_max_[0])
        
    except Exception as e:
        import traceback
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
    
    return jsonify(result)

@api_predict_bp.route('/debug/raw-prediction/<station_name>/<direction>')
def debug_raw_prediction(station_name, direction):
    """Debug raw model prediction"""
    from services.feature_engineering import get_feature_sequence_for_station
    
    station = station_name.replace('%20', ' ')
    now = Config.get_current_time()
    
    ensure_models_loaded(station, direction)
    
    directional_models, directional_scalers = get_models()
    correction_factors = get_correction_factors()
    
    model_key = f"{station}_{direction}"
    
    result = {
        'station': station,
        'direction': direction,
        'time': now.isoformat(),
        'model_loaded': model_key in directional_models if directional_models else False,
    }
    
    if not directional_models or model_key not in directional_models:
        result['error'] = f'Model {model_key} not loaded'
        return jsonify(result)
    
    try:
        # ✅ FIX: get_feature_sequence_for_station returns SCALED features
        features_scaled = get_feature_sequence_for_station(station, direction, now)
        result['features_shape'] = features_scaled.shape if features_scaled is not None else None
        
        if features_scaled is None:
            result['error'] = 'No features returned'
            return jsonify(result)
        
        # ✅ FIX: Get target scaler ONLY (features are already scaled)
        target_scaler = directional_scalers.get(f'{model_key}_target')
        
        result['has_target_scaler'] = target_scaler is not None
        
        if target_scaler is None:
            result['error'] = 'Missing target scaler'
            return jsonify(result)
        
        # ✅ FIX: features_scaled is already scaled, just reshape
        input_sequence = features_scaled.reshape(1, 24, -1)
        
        # Get prediction
        pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
        raw_output = float(pred_scaled[0][0])
        result['raw_scaled_output'] = raw_output
        
        # Inverse transform
        passenger_count = float(target_scaler.inverse_transform([[raw_output]])[0][0])
        result['passenger_count'] = passenger_count
        
        # Get P90 (changed from P95)
        p90 = get_p90_percentile(station, direction)
        result['p90'] = p90  # Changed from p95
        
        # Calculate congestion
        congestion = (passenger_count / p90) * 100  # Changed from p95
        congestion = max(0, min(congestion, 100))
        result['congestion'] = congestion
        
        # Correction factor
        factor = correction_factors.get(model_key, 1.0)
        result['correction_factor'] = factor
        result['passenger_count_corrected'] = passenger_count * factor
        
        # ✅ FIX: Show scaler info (StandardScaler)
        if hasattr(target_scaler, 'mean_') and hasattr(target_scaler, 'scale_'):
            result['target_scaler_type'] = 'StandardScaler'
            result['target_scaler_mean'] = float(target_scaler.mean_[0])
            result['target_scaler_scale'] = float(target_scaler.scale_[0])
        
        # Show feature sample
        result['feature_sample'] = features_scaled[0, :10].tolist()
        
    except Exception as e:
        import traceback
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
    
    return jsonify(result)

@api_predict_bp.route('/debug/test-time/<station_name>/<int:hour>')
def debug_test_time(station_name, hour):
    """Test predictions at a specific hour"""
    from services.feature_engineering import get_feature_sequence_for_station
    
    station = station_name.replace('%20', ' ')
    directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    
    # Create test time
    now = Config.get_current_time()
    test_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    
    results = {
        "station": station,
        "test_time": test_time.strftime("%Y-%m-%d %H:%M"),
        "hour": hour,
        "is_rush_hour": "YES" if (7 <= hour <= 9 or 17 <= hour <= 19) else "NO",
        "is_operating": "YES" if (0 <= hour + 0/60 <= 24) else "NO",
        "predictions": {}
    }
    
    for direction in ['Northbound', 'Southbound']:
        model_key = f"{station}_{direction}"
        if model_key in directional_models:
            congestion = get_directional_prediction(station, direction, test_time)
            results["predictions"][direction] = round(congestion, 1)
    
    return jsonify(results)

@api_predict_bp.route('/debug/simulate-day/<station_name>')
def simulate_day(station_name):
    """
    Simulate predictions for a full day (24 hours) for a specific station.
    Includes raw passenger counts from inverse transform and individual statuses per direction.
    """
    from services.feature_engineering import get_feature_sequence_for_station
    
    station = station_name.replace('%20', ' ')
    directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    
    # Use a specific date (June 26, 2025)
    test_date = datetime(2025, 6, 26)
    
    results = {
        "station": station,
        "date": test_date.strftime("%Y-%m-%d"),
        "capacity": MRT3_PLATFORM_CAPACITY.get(station, 1000),
        "predictions": []
    }
    
    # Helper function to get status
    def get_status(cong):
        if cong > 80:
            return "SEVERE"
        elif cong > 50:
            return "CONGESTED"
        elif cong > 25:
            return "MODERATE"
        else:
            return "LIGHT"
    
    # Test every hour from 4 AM to 11 PM (operating hours)
    for hour in range(4, 24):
        test_time = test_date.replace(hour=hour, minute=0, second=0, microsecond=0)
        
        # Check if operating
        time_decimal = hour + 0 / 60
        is_operating = 0 <= time_decimal < 24
        
        hour_data = {
            "hour": hour,
            "time": f"{hour:02d}:00",
            "is_operating": is_operating,
            "northbound": {
                "congestion": None,
                "passengers": None,
                "raw_model_output": None,
                "status": None
            },
            "southbound": {
                "congestion": None,
                "passengers": None,
                "raw_model_output": None,
                "status": None
            },
            "avg": None,
            "status": None
        }
        
        if is_operating:
            # Get predictions with passenger counts
            north_result = _get_directional_prediction_with_details(station, 'Northbound', test_time)
            south_result = _get_directional_prediction_with_details(station, 'Southbound', test_time)
            
            north_cong = north_result["congestion"]
            south_cong = south_result["congestion"]
            
            # Set northbound data with status
            hour_data["northbound"] = {
                "congestion": round(north_cong, 1),
                "passengers": round(north_result["passengers"], 0),
                "raw_model_output": round(north_result["raw_output"], 4),
                "status": get_status(north_cong)
            }
            
            # Set southbound data with status
            hour_data["southbound"] = {
                "congestion": round(south_cong, 1),
                "passengers": round(south_result["passengers"], 0),
                "raw_model_output": round(south_result["raw_output"], 4),
                "status": get_status(south_cong)
            }
            
            # Calculate average
            avg_cong = (north_cong + south_cong) / 2
            hour_data["avg"] = round(avg_cong, 1)
            
            # Set average status (for reference)
            hour_data["status"] = get_status(avg_cong)
        else:
            hour_data["northbound"] = {
                "congestion": 0,
                "passengers": 0,
                "raw_model_output": 0,
                "status": "CLOSED"
            }
            hour_data["southbound"] = {
                "congestion": 0,
                "passengers": 0,
                "raw_model_output": 0,
                "status": "CLOSED"
            }
            hour_data["avg"] = 0
            hour_data["status"] = "CLOSED"
        
        results["predictions"].append(hour_data)
    
    # Add summary statistics
    operating_hours = [h for h in results["predictions"] if h["is_operating"] and h["avg"] is not None and h["avg"] > 0]
    if operating_hours:
        avgs = [h["avg"] for h in operating_hours]
        
        # Find peak hour
        peak_idx = avgs.index(max(avgs))
        peak_hour_data = operating_hours[peak_idx]
        
        results["summary"] = {
            "peak_hour": peak_hour_data["time"],
            "peak_congestion": max(avgs),
            "peak_passengers": max([
                h["northbound"]["passengers"] or 0 for h in operating_hours
            ] + [
                h["southbound"]["passengers"] or 0 for h in operating_hours
            ]),
            "average_congestion": round(sum(avgs) / len(avgs), 1),
            "min_congestion": min(avgs),
            "rush_hours": {
                "morning_peak": [
                    {
                        "hour": h["hour"],
                        "time": h["time"],
                        "avg": h["avg"],
                        "northbound": h["northbound"],
                        "southbound": h["southbound"],
                        "status": h["status"]
                    }
                    for h in operating_hours if 7 <= h["hour"] <= 9
                ],
                "evening_peak": [
                    {
                        "hour": h["hour"],
                        "time": h["time"],
                        "avg": h["avg"],
                        "northbound": h["northbound"],
                        "southbound": h["southbound"],
                        "status": h["status"]
                    }
                    for h in operating_hours if 17 <= h["hour"] <= 19
                ]
            }
        }
    
    return jsonify(results)

def _get_directional_prediction_with_details(station_name, direction, target_datetime):
    """Helper function that returns both congestion and passenger count"""
    
    ensure_models_loaded(station_name, direction)
    
    directional_models, directional_scalers = get_models()
    correction_factors = get_correction_factors()
    
    model_key = f"{station_name}_{direction}"
    
    result = {
        "congestion": 0,
        "passengers": 0,
        "raw_output": 0
    }
    
    if directional_models is None or model_key not in directional_models:
        return result
    
    try:
        # ========== ✅ FIX: Use get_scaled_feature_sequence ==========
        from services.feature_engineering import get_scaled_feature_sequence
        features_scaled = get_scaled_feature_sequence(station_name, direction, target_datetime)
        
        if features_scaled is None:
            return result
        
        # ========== ✅ FIX: Get target scaler ONLY ==========
        target_scaler = directional_scalers.get(f'{model_key}_target')
        if target_scaler is None:
            return result
        
        # ========== ✅ FIX: features_scaled is already scaled ==========
        input_sequence = features_scaled.reshape(1, 24, -1)
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.float32)
        prediction_scaled = directional_models[model_key](input_tensor, training=False).numpy()
        raw_output = float(prediction_scaled[0][0])
        
        passenger_count = float(target_scaler.inverse_transform([[raw_output]])[0][0])
        
        # Get P90 (changed from P95)
        p90 = get_p90_percentile(station_name, direction)
        if p90 <= 0:
            p90 = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
        
        # Calculate congestion
        congestion = (passenger_count / p90) * 100  # Changed from p95
        congestion = max(0, min(congestion, 100))
        
        # Apply correction factor
        factor = correction_factors.get(model_key, 1.0)
        congestion = congestion * factor
        congestion = max(0, min(congestion, 100))
        
        result["congestion"] = congestion
        result["passengers"] = passenger_count
        result["raw_output"] = raw_output
        
    except Exception as e:
        print(f"Error in _get_directional_prediction_with_details: {e}")
        import traceback
        traceback.print_exc()
    
    return result
@api_predict_bp.route('/debug/feature-scaler/<station_name>/<direction>')
def debug_feature_scaler(station_name, direction):
    """Debug feature scaler transformation"""
    from services.feature_engineering import get_feature_sequence_for_station
    
    station = station_name.replace('%20', ' ')
    now = Config.get_current_time()
    
    ensure_models_loaded(station, direction)
    directional_models, directional_scalers = get_models()
    
    model_key = f"{station}_{direction}"
    
    result = {
        'station': station,
        'direction': direction,
        'time': now.isoformat()
    }
    
    if model_key not in directional_models:
        result['error'] = f'Model {model_key} not loaded'
        return jsonify(result)
    
    try:
        # ✅ FIX: get_feature_sequence_for_station returns SCALED features
        features_scaled = get_feature_sequence_for_station(station, direction, now)
        result['features_shape'] = features_scaled.shape if features_scaled is not None else None
        
        if features_scaled is None:
            result['error'] = 'No features returned'
            return jsonify(result)
        
        # ✅ FIX: Show the scaled features directly
        result['scaled_features_sample'] = features_scaled[0, :10].tolist()
        result['scaled_features_min'] = float(features_scaled.min())
        result['scaled_features_max'] = float(features_scaled.max())
        result['scaled_features_mean'] = float(features_scaled.mean())
        
        # Show feature scaler parameters (MinMaxScaler)
        feature_scaler = directional_scalers.get(f'{model_key}_feature')
        result['has_feature_scaler'] = feature_scaler is not None
        
        if feature_scaler:
            if hasattr(feature_scaler, 'data_min_'):
                result['scaler_data_min'] = feature_scaler.data_min_.tolist()[:10]
            if hasattr(feature_scaler, 'data_max_'):
                result['scaler_data_max'] = feature_scaler.data_max_.tolist()[:10]
            if hasattr(feature_scaler, 'scale_'):
                result['scaler_scale'] = feature_scaler.scale_.tolist()[:10]
                
    except Exception as e:
        import traceback
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
    
    return jsonify(result)

@api_predict_bp.route('/debug/target-scaler-file/<station_name>/<direction>')
def debug_target_scaler_file(station_name, direction):
    """Debug what's in the target scaler file"""
    import pickle
    import os
    
    station = station_name.replace('%20', ' ')
    model_key = f"{station}_{direction}"
    
    model_folder = 'models_2022-2024_v10'
    target_scaler_path = os.path.join(model_folder, f'{model_key}_target_scaler.pkl')
    
    result = {
        'station': station,
        'direction': direction,
        'scaler_path': target_scaler_path,
        'file_exists': os.path.exists(target_scaler_path)
    }
    
    if os.path.exists(target_scaler_path):
        try:
            with open(target_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            if hasattr(scaler, 'data_min_'):
                result['data_min'] = float(scaler.data_min_[0])
                result['data_max'] = float(scaler.data_max_[0])
                result['range'] = result['data_max'] - result['data_min']
            
            if hasattr(scaler, 'scale_'):
                result['scale'] = float(scaler.scale_[0])
                
            result['scaler_type'] = str(type(scaler))
            
        except Exception as e:
            result['error'] = str(e)
    
    return jsonify(result)

def _get_operating_hours_fallback(target_datetime):
    """Get realistic fallback based on time of day"""
    hour = target_datetime.hour
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        return 65
    elif 10 <= hour <= 16:
        return 45
    elif 5 <= hour <= 6 or 20 <= hour <= 21:
        return 25
    else:
        return 10

def get_station_prediction(station_name):
    """Get average congestion for a station"""
    north = get_directional_prediction(station_name, 'Northbound')
    south = get_directional_prediction(station_name, 'Southbound')
    return (north + south) / 2

# ========== MAIN PREDICTION ENDPOINTS ==========
@api_predict_bp.route('/debug/raw-model-test/<station_name>')
def debug_raw_model_test(station_name):
    """Test raw model prediction without any fallback"""
    # ✅ FIX: Use get_scaled_feature_sequence instead
    from services.feature_engineering import get_scaled_feature_sequence
    station = station_name.replace('%20', ' ')
    directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    
    now = Config.get_current_time()
    results = {}
    
    for direction in ['Northbound', 'Southbound']:
        model_key = f"{station}_{direction}"
        
        if model_key not in directional_models:
            results[direction] = {"error": f"Model {model_key} not loaded"}
            continue
        
        try:
            # ✅ FIX: Get already-scaled features directly
            features_scaled = get_scaled_feature_sequence(station, direction, now)
            if features_scaled is None:
                results[direction] = {"error": "No features returned"}
                continue
            
            # ✅ FIX: Only get target scaler (features already scaled)
            target_scaler = directional_scalers.get(f'{model_key}_target')
            
            if target_scaler is None:
                results[direction] = {"error": "Missing target scaler"}
                continue
            
            # ✅ FIX: features_scaled is already scaled, just reshape
            input_sequence = features_scaled.reshape(1, 24, -1)
            
            # Get prediction
            pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
            raw_output = float(pred_scaled[0][0])
            
            # Inverse transform
            passenger_count = float(target_scaler.inverse_transform([[raw_output]])[0][0])
            
            results[direction] = {
                "raw_scaled_output": raw_output,
                "passenger_count": passenger_count,
                "feature_shape": features_scaled.shape,
                "features_sample": features_scaled[0, :5].tolist(),
                "target_scaler": {
                    "mean": float(target_scaler.mean_[0]) if hasattr(target_scaler, 'mean_') else None,
                    "scale": float(target_scaler.scale_[0]) if hasattr(target_scaler, 'scale_') else None
                }
            }
            
        except Exception as e:
            import traceback
            results[direction] = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    return jsonify({
        "station": station,
        "time": now.isoformat(),
        "results": results
    })
    
@api_predict_bp.route('/debug/raw-csv-data/<station_name>')
def debug_raw_csv_data(station_name):
    """Check raw CSV data for a station"""
    from services.feature_engineering import get_station_dataframe
    import pandas as pd
    
    station = station_name.replace('%20', ' ')
    results = {}
    
    for direction in ['Northbound', 'Southbound']:
        df = get_station_dataframe(station, direction)
        
        if df is not None:
            # Check raw passenger stats
            results[direction] = {
                'total_rows': len(df),
                'min_passengers': float(df['TotalPassenger'].min()),
                'max_passengers': float(df['TotalPassenger'].max()),
                'mean_passengers': float(df['TotalPassenger'].mean()),
                'hourly_stats': {
                    f"{hour}:00": {
                        'mean': float(df[df.index.hour == hour]['TotalPassenger'].mean() or 0),
                        'max': float(df[df.index.hour == hour]['TotalPassenger'].max() or 0)
                    }
                    for hour in range(6, 22)
                }
            }
    
    return jsonify(results)
    
@api_predict_bp.route('/debug/check-data-availability')
def debug_check_data_availability():
    """Check what data is available for lookback"""
    from services.feature_engineering import get_station_dataframe, load_data_fast
    
    station = request.args.get('station', 'North Ave')
    direction = request.args.get('direction', 'Northbound')
    
    df = get_station_dataframe(station, direction)
    if df is None:
        return jsonify({"error": "No data loaded"})
    
    hourly = get_station_dataframe(station, direction)
    
    result = {
        "station": station,
        "direction": direction,
        "raw_data_stats": {
            "total_rows": len(df),
            "date_range": {
                "min": df['datetime'].min().isoformat() if 'datetime' in df.columns else None,
                "max": df['datetime'].max().isoformat() if 'datetime' in df.columns else None
            },
            "available_years": df['datetime'].dt.year.unique().tolist() if 'datetime' in df.columns else []
        },
        "hourly_data_stats": {
            "total_rows": len(hourly) if hourly is not None else 0,
            "date_range": {
                "min": hourly.index.min().isoformat() if hourly is not None and len(hourly) > 0 else None,
                "max": hourly.index.max().isoformat() if hourly is not None and len(hourly) > 0 else None
            } if hourly is not None else None
        }
    }
    
    from datetime import datetime
    test_date = datetime(2024, 6, 20, 10, 55)
    
    if hourly is not None and len(hourly) > 0:
        date_exists = test_date in hourly.index
        before = hourly[hourly.index < test_date].tail(5) if len(hourly[hourly.index < test_date]) > 0 else None
        after = hourly[hourly.index >= test_date].head(5) if len(hourly[hourly.index >= test_date]) > 0 else None
        
        result["test_date_check"] = {
            "test_date": test_date.isoformat(),
            "exists_in_data": date_exists,
            "before_date": [
                {"timestamp": idx.isoformat(), "passengers": row['TotalPassenger']}
                for idx, row in before.iterrows()
            ] if before is not None else "No data before",
            "after_date": [
                {"timestamp": idx.isoformat(), "passengers": row['TotalPassenger']}
                for idx, row in after.iterrows()
            ] if after is not None else "No data after"
        }
    
    return jsonify(result)
@api_predict_bp.route('/debug/target-scaler-test/<station_name>')
def debug_target_scaler_test(station_name):
    """Test what the target scaler does - StandardScaler version"""
    station = station_name.replace('%20', ' ')
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    
    results = {}
    
    for direction in ['Northbound', 'Southbound']:
        model_key = f"{station}_{direction}"
        target_scaler = directional_scalers.get(f'{model_key}_target')
        
        if target_scaler:
            test_values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
            
            # ✅ FIX: StandardScaler uses mean_ and scale_
            if hasattr(target_scaler, 'mean_') and hasattr(target_scaler, 'scale_'):
                results[direction] = {
                    "scaler_type": "StandardScaler",
                    "mean": float(target_scaler.mean_[0]),
                    "scale": float(target_scaler.scale_[0]),
                    "conversions": {
                        f"input_{v:.1f}": float(target_scaler.inverse_transform(np.array([[v]]))[0][0])
                        for v in test_values
                    }
                }
            elif hasattr(target_scaler, 'data_min_') and hasattr(target_scaler, 'data_max_'):
                results[direction] = {
                    "scaler_type": "MinMaxScaler",
                    "data_min": float(target_scaler.data_min_[0]),
                    "data_max": float(target_scaler.data_max_[0]),
                    "conversions": {
                        f"input_{v:.1f}": float(target_scaler.inverse_transform(np.array([[v]]))[0][0])
                        for v in test_values
                    }
                }
            else:
                results[direction] = {"error": "Unknown scaler type"}
        else:
            results[direction] = {"error": "Target scaler not found"}
    
    return jsonify(results)
@api_predict_bp.route('/debug/passenger-prediction/<station_name>')
def debug_passenger_prediction(station_name):
    """Debug raw passenger predictions vs p90-based congestion"""
    # ✅ FIX: Use get_scaled_feature_sequence
    from services.feature_engineering import get_scaled_feature_sequence
    
    station = station_name.replace('%20', ' ')
    directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    correction_factors = get_correction_factors()
    
    results = {}
    test_times = [8, 12, 18, 21]
    
    for hour in test_times:
        test_time = Config.get_current_time().replace(hour=hour, minute=0, second=0)
        
        for direction in ['Northbound', 'Southbound']:
            model_key = f"{station}_{direction}"
            
            if model_key not in directional_models:
                continue
            
            try:
                # ✅ FIX: Already-scaled features
                features_scaled = get_scaled_feature_sequence(station, direction, test_time)
                target_scaler = directional_scalers.get(f'{model_key}_target')
                
                input_sequence = features_scaled.reshape(1, 24, -1)
                
                raw_output = directional_models[model_key].predict(input_sequence, verbose=0)
                raw_value = float(raw_output[0][0])
                
                passenger_count = float(target_scaler.inverse_transform([[raw_value]])[0][0])
                
                # Apply correction factor
                factor = correction_factors.get(model_key, 1.0)
                passenger_count = passenger_count * factor
                
                p90 = get_p90_percentile(station, direction)  # Changed from p95
                
                congestion = (passenger_count / p90) * 100  # Changed from p95
                congestion = max(0, min(congestion, 100))
                
                results[f"{hour}:00_{direction}"] = {
                    "raw_scaled_output": round(raw_value, 4),
                    "p90_percentile": round(float(p90), 2),  # Changed from p95
                    "correction_factor": round(float(factor), 3),
                    "predicted_passengers": round(passenger_count, 0),
                    "congestion_percentage": round(congestion, 1),
                    "time_of_day": "Rush" if (7 <= hour <= 9 or 17 <= hour <= 19) else "Normal"
                }
                
            except Exception as e:
                results[f"{hour}:00_{direction}"] = {"error": str(e)}
    
    return jsonify(results)
from config import Config
def is_override_active(override, target_time):
    """
    Check if an override is active for the given target time.
    Uses Config time for consistency.
    """
    expiry = override.get('expiry')
    timestamp = override.get('timestamp')
    duration_minutes = override.get('duration_minutes', 60)
    
    if not expiry or not timestamp:
        return False
    
    try:
        # Get Config time for comparison
        config_time = Config.get_current_time()
        now_timestamp = config_time.timestamp()
        
        # First check: Is the override expired?
        if expiry <= now_timestamp:
            print(f"⏰ Override expired: {override}")
            return False
        
        # Second check: Is the target time within the override window?
        # Parse the start time (it might be in 2026, but we only care about the time-of-day)
        override_start = datetime.fromisoformat(timestamp)
        
        # Get the target time's date (from Config)
        target_date = target_time.date()
        
        # Create a datetime with the target date and the override's time
        override_start_in_target_date = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            override_start.hour,
            override_start.minute,
            override_start.second,
            override_start.microsecond
        )
        
        # Calculate the end time
        override_end = override_start_in_target_date + timedelta(minutes=duration_minutes)
        
        # Check if target_time is within the window
        is_active = override_start_in_target_date <= target_time <= override_end
        
        print(f"🔍 Override check: start={override_start_in_target_date}, end={override_end}, target={target_time}, active={is_active}")
        return is_active
        
    except Exception as e:
        print(f"⚠️ Error in is_override_active: {e}")
        return False

@api_predict_bp.route('/directional-forecast/<station_name>')
def directional_forecast(station_name):
    
    from services.feature_engineering import _TYPICAL_PATTERN_CACHE,  _BASELINE_FEATURES_CACHE
    
    
    name = station_name.replace('%20', ' ')
    
    date_param = request.args.get('date')
    time_param = request.args.get('time')
    
    if date_param and time_param:
        try:
            year, month, day = map(int, date_param.split('-'))
            hour, minute = map(int, time_param.split(':'))
            base_time = datetime(year, month, day, hour, minute)
        except Exception as e:
            print(f"⚠️ Invalid date/time: {e}, using current time")
            base_time = Config.get_current_time()
    else:
        base_time = Config.get_current_time()
    
    # ========== GET ACTIVE OVERRIDES FROM FILE ==========
    active_overrides = get_active_overrides()
    
    # Debug: Print override status
    print(f"\n🔍 DIRECTIONAL FORECAST for {name}")
    print(f"   Active overrides: {list(active_overrides.keys())}")
    north_key = f"{name}_northbound"
    south_key = f"{name}_southbound"
    print(f"   North override exists: {north_key in active_overrides}")
    print(f"   South override exists: {south_key in active_overrides}")
    
    forecasts = []
    
    for i in range(6):
        target_time = base_time + timedelta(hours=i)
        
        is_north_overridden = False
        is_south_overridden = False
        north_cong = None
        south_cong = None
        
        # ========== CHECK NORTHBOUND OVERRIDE ==========
        if north_key in active_overrides:
            override = active_overrides[north_key]
            override_congestion = override.get('congestion', 50)
            
            # Check if override is active
            if is_override_active(override, target_time):
                north_cong = override_congestion
                is_north_overridden = True
                print(f"🔧 OVERRIDE ACTIVE: {name} Northbound at {target_time} -> {north_cong}%")
            else:
                print(f"⏰ Override NOT active for {target_time}")
        
        # ========== CHECK SOUTHBOUND OVERRIDE ==========
        if south_key in active_overrides:
            override = active_overrides[south_key]
            override_congestion = override.get('congestion', 50)
            
            # Check if override is active
            if is_override_active(override, target_time):
                south_cong = override_congestion
                is_south_overridden = True
                print(f"🔧 OVERRIDE ACTIVE: {name} Southbound at {target_time} -> {south_cong}%")
            else:
                print(f"⏰ Override NOT active for {target_time}")
        
        # Use model predictions if no active override for this hour
        if north_cong is None:
            north_cong = get_directional_prediction(name, 'Northbound', target_time)
            if north_cong is None:
                north_cong = 0
                
        if south_cong is None:
            south_cong = get_directional_prediction(name, 'Southbound', target_time)
            if south_cong is None:
                south_cong = 0
        
        # Handle None values
        if north_cong is None:
            north_cong = 0
        if south_cong is None:
            south_cong = 0
        
        ampm = target_time.strftime('%I:%M %p')
        if i == 0:
            ampm = f"NOW ({ampm})"
        
        forecasts.append({
            "hour": target_time.hour,
            "time": ampm,
            "northbound": round(north_cong, 1),
            "southbound": round(south_cong, 1),
            "northbound_overridden": is_north_overridden,
            "southbound_overridden": is_south_overridden
        })
    
    return jsonify({
        "station": name,
        "timestamp": base_time.isoformat(),
        "active_overrides": len(active_overrides),
        "override_details": {
            north_key: active_overrides.get(north_key) for north_key in [north_key] if north_key in active_overrides
        },
        "current": {
            "northbound": forecasts[0]["northbound"],
            "southbound": forecasts[0]["southbound"],
            "northbound_overridden": forecasts[0]["northbound_overridden"],
            "southbound_overridden": forecasts[0]["southbound_overridden"]
        },
        "forecasts": forecasts
    })

@api_predict_bp.route('/debug-only')
def test():
   directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
   return jsonify({
       'models_in_config': len(directional_models),
       'model_keys': list(directional_models.keys()),
       'config_keys': list(current_app.config.keys())[:20]
   })

@api_predict_bp.route('/directional-forecast/all')
@cache.cached(
    timeout=300,
    key_prefix=lambda: f"all_stations_{datetime.now().hour}"
)
def directional_forecast_all():
    """Get current congestion for ALL stations at once"""
    result = {"northbound": {}, "southbound": {}}
    now = Config.get_current_time()
    
    print(f"\n[PREDICTION API] Getting all stations at {now.strftime('%H:%M:%S')}")
    
    for station in STATIONS:
        north_cong = get_directional_prediction(station, 'Northbound', now)
        south_cong = get_directional_prediction(station, 'Southbound', now)
        
        def get_status(cong):
            if cong > 80: return "SEVERE"
            if cong > 50: return "CONGESTED"
            if cong > 25: return "MODERATE"
            return "LIGHT"
        
        result['northbound'][station] = {
            "congestion": round(float(north_cong), 1),
            "status": get_status(north_cong)
        }
        result['southbound'][station] = {
            "congestion": round(float(south_cong), 1),
            "status": get_status(south_cong)
        }
        import gc
        gc.collect()
    
    return jsonify(result)
@api_predict_bp.route('/debug-model-output/<station_name>')
def debug_model_output(station_name):
    """Check raw model output for different times - FIXED with capacity"""
    # ✅ FIX: Use get_scaled_feature_sequence
    from services.feature_engineering import get_scaled_feature_sequence
    
    station = station_name.replace('%20', ' ')
    directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    correction_factors = get_correction_factors()
    
    results = {}
    test_times = [8, 12, 18, 21]
    
    for hour in test_times:
        test_time = Config.get_current_time().replace(hour=hour, minute=0, second=0)
        
        for direction in ['Northbound', 'Southbound']:
            model_key = f"{station}_{direction}"
            
            if model_key not in directional_models:
                continue
            
            try:
                # ✅ FIX: Already-scaled features
                features_scaled = get_scaled_feature_sequence(station, direction, test_time)
                target_scaler = directional_scalers.get(f'{model_key}_target')
                
                input_sequence = features_scaled.reshape(1, 24, -1)
                
                raw_output = directional_models[model_key].predict(input_sequence, verbose=0)
                raw_value = float(raw_output[0][0])
                
                passenger_count = float(target_scaler.inverse_transform([[raw_value]])[0][0])
                
                # Apply correction factor
                factor = correction_factors.get(model_key, 1.0)
                passenger_count = passenger_count * factor
                
                p90 = get_p90_percentile(station, direction)  # Changed from p95
                        
                congestion = (passenger_count / p90) * 100  # Changed from p95
                congestion = max(0, min(congestion, 100))
                
                results[f"{hour}:00_{direction}"] = {
                    "raw_model_output": round(raw_value, 4),
                    "p90_percentile": round(float(p90), 2),  # Changed from p95
                    "correction_factor": round(float(factor), 3),
                    "predicted_passengers": round(passenger_count, 0),
                    "predicted_congestion": round(congestion, 1)
                }
                
            except Exception as e:
                results[f"{hour}:00_{direction}"] = {"error": str(e)}
    
    return jsonify(results)

@api_predict_bp.route('/model-evaluation')
def model_evaluation():
    """Evaluate model performance using the same 90th percentile as the API."""
    from services.feature_engineering import get_station_dataframe, get_feature_sequence_for_station
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
    import numpy as np

    station = request.args.get('station', 'North Ave')
    direction = request.args.get('direction', 'Northbound')
    test_days = int(request.args.get('days', 30))

    df = get_station_dataframe(station, direction)
    if df is None or len(df) == 0:
        return jsonify({"error": "No data available"})

    directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    correction_factors = get_correction_factors()

    model_key = f"{station}_{direction}"
    if model_key not in directional_models:
        return jsonify({"error": f"Model {model_key} not found"})

    p90 = get_p90_percentile(station, direction)  # Changed from p95
    print(f"🔍 Using p90 = {p90:.0f} for {station} {direction}")

    def get_congestion_category(cong):
        if cong > 80:
            return "Severe"
        elif cong > 50:
            return "Heavy"
        elif cong > 25:
            return "Moderate"
        else:
            return "Light"

    predictions = []
    actuals = []

    end_date = df.index.max()
    start_date = end_date - timedelta(days=test_days)
    test_data = df[(df.index >= start_date) & (df.index < end_date)]

    print(f"Testing on {len(test_data)} hours from {start_date} to {end_date}")

    for timestamp in test_data.index:
        actual_passengers = test_data.loc[timestamp, 'TotalPassenger']
        actual_congestion = (actual_passengers / p90) * 100  # Changed from p95
        actual_congestion = min(actual_congestion, 100)

        try:
            # ✅ FIX: get_feature_sequence_for_station returns SCALED features
            features_scaled = get_feature_sequence_for_station(station, direction, timestamp)
            if features_scaled is None:
                continue

            target_scaler = directional_scalers.get(f'{model_key}_target')
            if target_scaler is None:
                continue

            # ✅ FIX: features_scaled is already scaled, just reshape
            input_sequence = features_scaled.reshape(1, 24, -1)

            pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
            raw_value = float(pred_scaled[0][0])

            passenger_count = float(target_scaler.inverse_transform([[raw_value]])[0][0])
            factor = correction_factors.get(model_key, 1.0)
            passenger_count = passenger_count * factor

            predicted_congestion = (passenger_count / p90) * 100  # Changed from p95
            predicted_congestion = min(predicted_congestion, 100)

            predictions.append(predicted_congestion)
            actuals.append(actual_congestion)

        except Exception as e:
            print(f"Error at {timestamp}: {e}")
            continue

    if len(predictions) == 0:
        return jsonify({"error": "No valid predictions"})

    pred_categories = [get_congestion_category(p) for p in predictions]
    actual_categories = [get_congestion_category(a) for a in actuals]

    categories = ["Light", "Moderate", "Heavy", "Severe"]

    cm = confusion_matrix(actual_categories, pred_categories, labels=categories)
    class_report = classification_report(actual_categories, pred_categories, labels=categories, output_dict=True)

    accuracy = accuracy_score(actual_categories, pred_categories)
    macro_f1 = f1_score(actual_categories, pred_categories, labels=categories, average='macro')
    weighted_f1 = f1_score(actual_categories, pred_categories, labels=categories, average='weighted')

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)

    mae_by_category = {}
    for category in categories:
        indices = [i for i, a in enumerate(actual_categories) if a == category]
        if indices:
            cat_mae = np.mean([abs(actuals[i] - predictions[i]) for i in indices])
            mae_by_category[category] = round(cat_mae, 2)

    return jsonify({
        "station": station,
        "direction": direction,
        "p90_percentile": round(p90, 2),  # Changed from p95
        "test_period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "total_hours_tested": len(predictions)
        },
        "confusion_matrix": {
            "labels": categories,
            "matrix": cm.tolist()
        },
        "classification_report": class_report,
        "accuracy": round(accuracy * 100, 2),
        "f1_scores": {
            "macro": round(macro_f1 * 100, 2),
            "weighted": round(weighted_f1 * 100, 2)
        },
        "regression_metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "mae_by_category": mae_by_category
        },
        "sample_predictions": [
            {
                "timestamp": test_data.index[i].isoformat(),
                "actual": round(actuals[i], 1),
                "predicted": round(predictions[i], 1),
                "error": round(abs(actuals[i] - predictions[i]), 1)
            }
            for i in range(min(10, len(predictions)))
        ]
    })

@api_predict_bp.route('/confusion-matrix')
def confusion_matrix_endpoint():
    """Generate confusion matrix visualization data"""
    from services.feature_engineering import get_station_dataframe
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import io
    import base64
    
    station = request.args.get('station', 'North Ave')
    direction = request.args.get('direction', 'Northbound')
    
    # Placeholder response
    return jsonify({
        "image": None,
        "matrix": [],
        "labels": ["Light", "Moderate", "Heavy", "Severe"],
        "message": "Run model-evaluation first to generate confusion matrix"
    })

@api_predict_bp.route('/test-rush-hour')
def test_rush_hour():
    """Test predictions for rush hour times"""
    results = {}
    now = Config.get_current_time()
    
    test_times = [
        now.replace(hour=8, minute=0),
        now.replace(hour=12, minute=0),
        now.replace(hour=18, minute=0),
        now.replace(hour=21, minute=0),
    ]
    
    for test_time in test_times:
        north = get_directional_prediction("North Ave", "Northbound", test_time)
        south = get_directional_prediction("North Ave", "Southbound", test_time)
        
        results[test_time.strftime("%H:%M")] = {
            "northbound": round(north, 1),
            "southbound": round(south, 1),
            "avg": round((north + south) / 2, 1)
        }
    
    return jsonify(results)

@api_predict_bp.route('/debug/scalers/<station_name>')
def debug_scalers(station_name):
    """Debug target scaler parameters - StandardScaler version"""
    station = station_name.replace('%20', ' ')
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    
    results = {}
    for direction in ['Northbound', 'Southbound']:
        model_key = f"{station}_{direction}"
        target_scaler = directional_scalers.get(f'{model_key}_target')
        
        if target_scaler:
            test_values = [0.0, 0.25, 0.5, 0.75, 1.0]
            inverse_values = target_scaler.inverse_transform(np.array(test_values).reshape(-1, 1))
            
            # ✅ FIX: Check scaler type
            if hasattr(target_scaler, 'mean_') and hasattr(target_scaler, 'scale_'):
                results[direction] = {
                    "scaler_exists": True,
                    "scaler_type": "StandardScaler",
                    "mean": float(target_scaler.mean_[0]),
                    "scale": float(target_scaler.scale_[0]),
                    "test_conversion": {
                        f"input_{v}": float(inverse_values[i][0]) 
                        for i, v in enumerate(test_values)
                    }
                }
            elif hasattr(target_scaler, 'data_min_') and hasattr(target_scaler, 'data_max_'):
                results[direction] = {
                    "scaler_exists": True,
                    "scaler_type": "MinMaxScaler",
                    "data_min": float(target_scaler.data_min_[0]),
                    "data_max": float(target_scaler.data_max_[0]),
                    "test_conversion": {
                        f"input_{v}": float(inverse_values[i][0]) 
                        for i, v in enumerate(test_values)
                    }
                }
            else:
                results[direction] = {"scaler_exists": True, "scaler_type": "Unknown"}
        else:
            results[direction] = {"scaler_exists": False}
    
    return jsonify(results)

@api_predict_bp.route('/debug/check-lookback')
def debug_check_lookback():
    """Check what values are in the lookback column"""
    from services.feature_engineering import get_station_dataframe
    
    station = "North Ave"
    direction = "Northbound"
    
    df = get_station_dataframe(station, direction)
    
    if df is not None:
        last_24 = df.tail(24)
        
        return jsonify({
            "station": station,
            "direction": direction,
            "lookback_column_name": "congestion" if 'congestion' in df.columns else "NOT FOUND",
            "actual_values_last_24_hours": last_24['congestion'].tolist() if 'congestion' in df.columns else [],
            "values_range": {
                "min": float(df['congestion'].min()),
                "max": float(df['congestion'].max()),
                "mean": float(df['congestion'].mean())
            } if 'congestion' in df.columns else None,
            "total_passenger_range": {
                "min": float(df['TotalPassenger'].min()),
                "max": float(df['TotalPassenger'].max()),
                "mean": float(df['TotalPassenger'].mean())
            }
        })
    
    return jsonify({"error": "No data"})

@api_predict_bp.route('/debug/lookback-values/<station_name>')
def debug_lookback_values(station_name):
    """Check what lookback values are being passed to the model"""
    from services.feature_engineering import get_feature_sequence_for_station
    
    station = station_name.replace('%20', ' ')
    test_time = Config.get_current_time().replace(hour=8, minute=0, second=0)
    
    results = {}
    
    for direction in ['Northbound', 'Southbound']:
        features = get_feature_sequence_for_station(station, direction, test_time)
        
        if features is not None:
            lookback_values = features[:, -1]
            
            results[direction] = {
                "lookback_values_scaled": lookback_values.tolist()[:10],
                "min_scaled": float(lookback_values.min()),
                "max_scaled": float(lookback_values.max()),
                "mean_scaled": float(lookback_values.mean()),
            }
    
    return jsonify(results)

@api_predict_bp.route('/debug/scaler-test/<station_name>')
def debug_scaler_test(station_name):
    """Test what the scaler does with different inputs"""
    station = station_name.replace('%20', ' ')
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    
    results = {}
    for direction in ['Northbound', 'Southbound']:
        model_key = f"{station}_{direction}"
        target_scaler = directional_scalers.get(f'{model_key}_target')
        
        if target_scaler:
            test_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            results[direction] = {
                "scaler_type": str(type(target_scaler)),
                "data_min": float(target_scaler.data_min_[0]) if hasattr(target_scaler, 'data_min_') else None,
                "data_max": float(target_scaler.data_max_[0]) if hasattr(target_scaler, 'data_max_') else None,
                "conversions": {
                    f"input_{v:.1f}": float(target_scaler.inverse_transform(np.array([[v]]))[0][0])
                    for v in test_values
                }
            }
    
    return jsonify(results)

@api_predict_bp.route('/debug/training-data-distribution')
def debug_training_data_distribution():
    """Check what data the model was trained on"""
    from services.feature_engineering import get_station_dataframe
    
    station = "North Ave"
    results = {}
    
    for direction in ['Northbound', 'Southbound']:
        df = get_station_dataframe(station, direction)
        
        if df is not None:
            hourly_avg = df.groupby(df.index.hour)['TotalPassenger'].mean()
            
            results[direction] = {
                "hourly_average_passengers": {
                    f"{hour:02d}:00": round(float(hourly_avg[hour]), 0) 
                    for hour in range(24) if hour in hourly_avg.index
                },
                "peak_hour": int(hourly_avg.idxmax()),
                "peak_passengers": float(hourly_avg.max()),
                "morning_rush_8am": float(hourly_avg[8]) if 8 in hourly_avg.index else 0,
                "evening_rush_6pm": float(hourly_avg[18]) if 18 in hourly_avg.index else 0,
            }
    
    return jsonify(results)

@api_predict_bp.route('/debug/fix-scaler/<station_name>')
def debug_fix_scaler(station_name):
    """Check what the target scaler should be based on data"""
    from services.feature_engineering import get_station_dataframe
    
    station = station_name.replace('%20', ' ')
    results = {}
    
    for direction in ['Northbound', 'Southbound']:
        hourly = get_station_dataframe(station, direction)
        
        if hourly is not None and len(hourly) > 0:
            passengers = hourly['TotalPassenger'].values
            data_min = float(passengers.min())
            data_max = float(passengers.max())
            
            directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
            model_key = f"{station}_{direction}"
            target_scaler = directional_scalers.get(f'{model_key}_target')
            
            current_min = float(target_scaler.data_min_[0]) if target_scaler and hasattr(target_scaler, 'data_min_') else None
            current_max = float(target_scaler.data_max_[0]) if target_scaler and hasattr(target_scaler, 'data_max_') else None
            
            results[direction] = {
                "actual_data": {
                    "min_passengers": data_min,
                    "max_passengers": data_max,
                    "mean_passengers": float(passengers.mean()),
                    "total_data_points": len(passengers)
                },
                "current_scaler": {
                    "min": current_min,
                    "max": current_max
                },
                "capacity_based_fix": f"Use capacity {MRT3_PLATFORM_CAPACITY.get(station, 1000)} instead of scaler_max"
            }
    
    return jsonify(results)

@api_predict_bp.route('/debug/check-scaler-values')
def debug_check_scaler_values():
    """Check what values the target scalers are actually using"""
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    
    results = {}
    
    for key, scaler in directional_scalers.items():
        if '_target' in key:
            if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
                results[key] = {
                    "data_min": float(scaler.data_min_[0]),
                    "data_max": float(scaler.data_max_[0]),
                    "scale": float(scaler.scale_[0]) if hasattr(scaler, 'scale_') else None
                }
            else:
                results[key] = {"error": "No data_min_ or data_max_ attribute"}
    
    return jsonify(results)

@api_predict_bp.route('/predict/<station_name>')
@cache.cached(
    timeout=300,
    key_prefix=lambda: safe_cache_key('predict')
)
def predict_congestion(station_name):
    """Get current snapshot congestion metrics for a single station"""
    name = station_name.replace('%20', ' ')
    
    date_param = request.args.get('date')
    time_param = request.args.get('time')
    
    target_datetime = None
    if date_param and time_param:
        try:
            year, month, day = map(int, date_param.split('-'))
            hour, minute = map(int, time_param.split(':'))
            target_datetime = datetime(year, month, day, hour, minute)
        except:
            target_datetime = None

    north_congestion = get_directional_prediction(name, 'Northbound', target_datetime)
    south_congestion = get_directional_prediction(name, 'Southbound', target_datetime)
    congestion = (north_congestion + south_congestion) / 2
    
    if congestion > 80: status = "CRITICAL"
    elif congestion > 50: status = "BUSY"
    elif congestion > 20: status = "MODERATE"
    else: status = "LIGHT"
    
    return jsonify({
        "station": name,
        "congestion": round(congestion, 1),
        "northbound": round(north_congestion, 1),
        "southbound": round(south_congestion, 1),
        "status": status
    })

@api_predict_bp.route('/predict-direction/<station_name>')
@cache.cached(
    timeout=300,
    key_prefix=lambda: safe_cache_key('pred_dir')
)
def predict_direction(station_name):
    name = station_name.replace('%20', ' ')
    
    north_congestion = get_directional_prediction(name, 'Northbound')
    south_congestion = get_directional_prediction(name, 'Southbound')
    congestion = (north_congestion + south_congestion) / 2
    
    station_idx = STATIONS.index(name) if name in STATIONS else 0
    
    if station_idx < 6:
        direction = "southbound"
        next_station = STATIONS[station_idx + 1] if station_idx + 1 < len(STATIONS) else STATIONS[0]
    elif station_idx > 6:
        direction = "northbound"
        next_station = STATIONS[station_idx - 1] if station_idx - 1 >= 0 else STATIONS[-1]
    else:
        direction = "both"
        next_station = STATIONS[station_idx + 1] if station_idx + 1 < len(STATIONS) else STATIONS[0]
    
    if congestion > 80: 
        status = "SEVERELY CONGESTED"
        color = "critical"
        wait_time = "15-20 min"
    elif congestion > 50: 
        status = "CONGESTED"
        color = "congested"
        wait_time = "10-15 min"
    elif congestion > 25: 
        status = "MODERATE"
        color = "moderate"
        wait_time = "5-10 min"
    else: 
        status = "LIGHT"
        color = "light"
        wait_time = "2-5 min"
    
    return jsonify({
        "station": name,
        "congestion": round(congestion, 1),
        "northbound": round(north_congestion, 1),
        "southbound": round(south_congestion, 1),
        "status": status,
        "color": color,
        "direction": direction,
        "next_station": next_station,
        "wait_time": wait_time
    })

@api_predict_bp.route('/predict-route')
@cache.cached(
    timeout=300,
    key_prefix=lambda: f"route_{request.args.get('from')}_{request.args.get('to')}_{datetime.now().hour}"
)
def predict_route():
    from_station = request.args.get('from')
    to_station = request.args.get('to')
    date = request.args.get('date')
    time = request.args.get('time')
    
    if not from_station or not to_station:
        return jsonify({"error": "Missing station parameters"}), 400
    
    if date and time:
        try:
            year, month, day = map(int, date.split('-'))
            hour, minute = map(int, time.split(':'))
            target_datetime = datetime(year, month, day, hour, minute)
            north_from = get_directional_prediction(from_station, 'Northbound', target_datetime)
            south_from = get_directional_prediction(from_station, 'Southbound', target_datetime)
            north_to = get_directional_prediction(to_station, 'Northbound', target_datetime)
            south_to = get_directional_prediction(to_station, 'Southbound', target_datetime)
            congestion_from = (north_from + south_from) / 2
            congestion_to = (north_to + south_to) / 2
        except:
            congestion_from = get_station_prediction(from_station)
            congestion_to = get_station_prediction(to_station)
    else:
        congestion_from = get_station_prediction(from_station)
        congestion_to = get_station_prediction(to_station)
    
    avg_congestion = (congestion_from + congestion_to) / 2
    
    from_idx = STATIONS.index(from_station) if from_station in STATIONS else 0
    to_idx = STATIONS.index(to_station) if to_station in STATIONS else len(STATIONS) - 1
    station_diff = abs(from_idx - to_idx)
    travel_time = station_diff * 3 + 5
    
    if avg_congestion > 80: 
        status = "CRITICAL"
        recommendation = "Consider postponing your trip"
    elif avg_congestion > 50: 
        status = "HEAVY"
        recommendation = "Allow extra time for your journey"
    elif avg_congestion > 25: 
        status = "MODERATE"
        recommendation = "Normal travel conditions"
    else: 
        status = "LIGHT"
        recommendation = "Good time to travel!"
    
    return jsonify({
        "from_station": from_station,
        "to_station": to_station,
        "from_congestion": round(congestion_from, 1),
        "to_congestion": round(congestion_to, 1),
        "avg_congestion": round(avg_congestion, 1),
        "status": status,
        "travel_time": travel_time,
        "stations_between": station_diff,
        "recommendation": recommendation
    })

@api_predict_bp.route('/debug/simulate-all-stations-full-day')
def simulate_all_stations_full_day():
    """
    Simulate predictions for ALL stations for the full day (4 AM - 11 PM).
    Returns comprehensive data for debugging.
    """
    from datetime import datetime, timedelta
    
    test_date = Config.get_current_time().replace(hour=0, minute=0, second=0, microsecond=0)
    
    results = {
        "date": test_date.strftime("%Y-%m-%d"),
        "simulation_time": Config.get_current_time().isoformat(),
        "stations": {},
        "summary": {}
    }
    
    station_summaries = {}
    
    for station in STATIONS:
        station_data = []
        north_avg = 0
        south_avg = 0
        peak_hour = None
        peak_congestion = 0
        
        for hour in range(4, 24):
            test_time = test_date.replace(hour=hour, minute=0, second=0, microsecond=0)
            time_decimal = hour + 0 / 60
            is_operating = 0 <= time_decimal < 24
            
            hour_data = {
                "hour": hour,
                "time": f"{hour:02d}:00",
                "is_operating": is_operating,
                "northbound": 0,
                "southbound": 0,
                "avg": 0,
                "status": "CLOSED"
            }
            
            if is_operating:
                try:
                    north_cong = get_directional_prediction(station, 'Northbound', test_time)
                    south_cong = get_directional_prediction(station, 'Southbound', test_time)
                    
                    north_cong = north_cong if north_cong is not None else 0
                    south_cong = south_cong if south_cong is not None else 0
                    
                    avg_cong = (north_cong + south_cong) / 2
                    
                    hour_data["northbound"] = round(north_cong, 1)
                    hour_data["southbound"] = round(south_cong, 1)
                    hour_data["avg"] = round(avg_cong, 1)
                    
                    north_avg += north_cong
                    south_avg += south_cong
                    
                    if avg_cong > peak_congestion:
                        peak_congestion = avg_cong
                        peak_hour = hour
                    
                    if avg_cong > 80:
                        hour_data["status"] = "SEVERE"
                    elif avg_cong > 50:
                        hour_data["status"] = "CONGESTED"
                    elif avg_cong > 25:
                        hour_data["status"] = "MODERATE"
                    else:
                        hour_data["status"] = "LIGHT"
                        
                except Exception as e:
                    hour_data["error"] = str(e)
            
            station_data.append(hour_data)
        
        operating_hours = [h for h in station_data if h["is_operating"] and h["avg"] > 0]
        if operating_hours:
            avg_cong = sum(h["avg"] for h in operating_hours) / len(operating_hours)
            avg_north = sum(h["northbound"] for h in operating_hours) / len(operating_hours)
            avg_south = sum(h["southbound"] for h in operating_hours) / len(operating_hours)
            
            status_counts = {
                "SEVERE": sum(1 for h in operating_hours if h["status"] == "SEVERE"),
                "CONGESTED": sum(1 for h in operating_hours if h["status"] == "CONGESTED"),
                "MODERATE": sum(1 for h in operating_hours if h["status"] == "MODERATE"),
                "LIGHT": sum(1 for h in operating_hours if h["status"] == "LIGHT")
            }
        else:
            avg_cong = 0
            avg_north = 0
            avg_south = 0
            status_counts = {}
        
        station_summaries[station] = {
            "avg_congestion": round(avg_cong, 1),
            "avg_northbound": round(avg_north, 1),
            "avg_southbound": round(avg_south, 1),
            "peak_hour": f"{peak_hour:02d}:00" if peak_hour is not None else "N/A",
            "peak_congestion": round(peak_congestion, 1) if peak_hour is not None else 0,
            "status_counts": status_counts
        }
        
        results["stations"][station] = station_data
    
    total_avg = sum(s["avg_congestion"] for s in station_summaries.values()) / len(station_summaries)
    busiest = max(station_summaries.items(), key=lambda x: x[1]["peak_congestion"])
    
    results["summary"] = {
        "total_stations": len(STATIONS),
        "overall_average_congestion": round(total_avg, 1),
        "busiest_station": busiest[0],
        "busiest_peak": busiest[1]["peak_congestion"],
        "busiest_hour": busiest[1]["peak_hour"],
        "station_summaries": station_summaries
    }
    
    return jsonify(results)

@api_predict_bp.route('/admin/generate-factors', methods=['POST'])
def generate_factors():
    """Admin endpoint to recompute correction factors from historical data."""
    try:
        test_days = request.json.get('test_days', 30) if request.is_json else 30
        factors = compute_and_save_correction_factors(test_days=test_days)
        # Reload factors so they take effect immediately
        load_correction_factors()
        return jsonify({
            "success": True,
            "message": f"Generated {len(factors)} correction factors",
            "factors": factors
        })
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
        
@api_predict_bp.route('/debug/feature-alignment/<station_name>/<direction>')
def debug_feature_alignment(station_name, direction):
    from services.feature_engineering import get_feature_sequence_for_station
    import pickle
    
    station = station_name.replace('%20', ' ')
    now = Config.get_current_time()
    
    # Get features from API
    features = get_feature_sequence_for_station(station, direction, now)
    
    # Load training feature scaler
    model_key = f"{station}_{direction}"
    scaler_path = f'models_2022-2024_v10/{model_key}_feature_scaler.pkl'
    
    with open(scaler_path, 'rb') as f:
        train_scaler = pickle.load(f)
    
    # Get raw features (need to implement this)
    # raw_features = get_raw_features_for_station(station, direction, now)
    
    return jsonify({
        'scaled_from_api': features[0, :10].tolist(),
        'train_scaler_min': train_scaler.data_min_.tolist()[:10],
        'train_scaler_max': train_scaler.data_max_.tolist()[:10],
        'feature_columns': 'check against training'
    })