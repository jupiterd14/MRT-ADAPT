from flask import Blueprint, request, jsonify, current_app
from extensions import cache
from datetime import datetime, timedelta
from services.feature_engineering import get_feature_sequence_for_station, get_scaled_feature_sequence
from config import Config
import numpy as np
import math
from constants import MRT3_PLATFORM_CAPACITY
class _LazyTF:
    _mod = None
    def __getattr__(self, name):
        if self._mod is None:
            import tensorflow as _t
            self._mod = _t
        return getattr(self._mod, name)

tf = _LazyTF()


api_predict_bp = Blueprint('api_predict', __name__)

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

import json
import os
import pickle
# Add at the top with other imports
import time

# ========== REQUEST-LEVEL CACHE ==========
_PREDICTION_CACHE = {}
_REQUEST_CACHE = {}
_REQUEST_CACHE_TTL = 300  # 10 seconds
_P90_CACHE = {}  # Add this global variable
_P90_FILE = 'p90_percentiles.json'
_PENDING_CORRECTION_FACTORS = {}
def get_cached_prediction(station_name, direction, target_datetime):
    if target_datetime is None:
        target_datetime = Config.get_current_time()
    # NEW: hour only, no minute slot
    cache_key = f"{station_name}_{direction}_{target_datetime.strftime('%Y%m%d%H')}"
    # Clean old entries (optional)
    current_time = time.time()
    for key in list(_REQUEST_CACHE.keys()):
        if current_time - _REQUEST_CACHE[key]['timestamp'] > _REQUEST_CACHE_TTL:
            del _REQUEST_CACHE[key]
    if cache_key in _REQUEST_CACHE:
        return _REQUEST_CACHE[cache_key]['value']
    return None

def set_cached_prediction(station_name, direction, target_datetime, value):
    if target_datetime is None:
        target_datetime = Config.get_current_time()
    # NEW: hour only
    cache_key = f"{station_name}_{direction}_{target_datetime.strftime('%Y%m%d%H')}"
    _REQUEST_CACHE[cache_key] = {'value': value, 'timestamp': time.time()}

# ========== P90 CACHE - USING APP CONFIG ==========
_P90_FILE = 'p90_percentiles.json'
CORRECTION_FILE = 'correction_factors.pkl'

def get_p90_cache():
    """Return the P90 cache dictionary (was get_p95_cache)"""
    return _P90_CACHE


_PENDING_CORRECTION_FACTORS = {}
def get_all_stations_predictions():
    result = {"northbound": {}, "southbound": {}}
    now = Config.get_current_time()
    
    # Operating hours check
    current_time = now.hour + now.minute / 60
    is_closed = current_time < 4.5 or current_time >= 22.5
    
    for station in STATIONS:
        if is_closed:
            result['northbound'][station] = {"congestion": 0, "status": "CLOSED"}
            result['southbound'][station] = {"congestion": 0, "status": "CLOSED"}
            continue
        
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
    
    return result

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
 
    
from services.feature_engineering import STATION_NUMBERS

def precompute_all_predictions():
    """
    Precompute congestion predictions for all (station, direction, dow, hour)
    using the cached scaled feature sequences and the trained models.
    Stores results in _PREDICTION_CACHE for instant lookup.
    """
    stations = list(STATION_NUMBERS.keys())
    directions = ['Northbound', 'Southbound']
    dows = range(7)
    hours = range(24)
    
    # Ensure models are loaded
    ensure_models_loaded()
    directional_models, directional_scalers = get_models()
    
    total = 0
    for station in stations:
        for direction in directions:
            model_key = f"{station}_{direction}"
            model = directional_models.get(model_key)
            target_scaler = directional_scalers.get(f'{model_key}_target')
            if model is None or target_scaler is None:
                print(f"⚠️ Skipping {model_key} – model or scaler missing")
                continue
            
            # Get P90 once per station-direction
            p90 = get_p90_percentile(station, direction)
            if p90 <= 0:
                p90 = MRT3_PLATFORM_CAPACITY.get(station, 1000)
            correction_factors = get_correction_factors()
            factor = correction_factors.get(model_key, 1.0)
            
            for dow in dows:
                # Build a batch of 24 hours for this station-direction-dow
                features_list = []
                for hour in hours:
                    # Create a dummy datetime with the right dow and hour (any date)
                    base_dt = datetime(2025, 1, 6) + timedelta(days=dow, hours=hour)
                    features = get_scaled_feature_sequence(station, direction, base_dt)
                    if features is None:
                        # Should never happen after precompute_all_scaled_sequences
                        features = np.zeros((24, 16), dtype=np.float32)
                    features_list.append(features)
                
                # Stack into batch (24, 24, 16)
                batch_features = np.stack(features_list, axis=0).astype(np.float32)  # (24, 24, 16)
                
                # Batch prediction (shape: (24, 1))
                pred_scaled = model.predict(batch_features, verbose=0)
                pred_passengers = target_scaler.inverse_transform(pred_scaled).flatten()
                
                # Convert each hour to congestion and store
                for hour, passenger in enumerate(pred_passengers):
                    congestion = (passenger / p90) * 100
                    congestion = max(0, min(congestion, 100))
                    congestion = congestion * factor
                    congestion = max(0, min(congestion, 100))
                    
                    # Apply day-of-week adjustment (same as in get_directional_prediction)
                    if dow >= 5:  # weekend
                        dow_factor = 0.7
                    elif dow == 4:  # Friday
                        dow_factor = 1.1
                    elif dow == 0:  # Monday
                        dow_factor = 1.05
                    else:
                        dow_factor = 1.0
                    congestion = congestion * dow_factor
                    congestion = max(0, min(congestion, 100))
                    
                    cache_key = f"{station}_{direction}_{dow}_{hour}"
                    _PREDICTION_CACHE[cache_key] = congestion
                    total += 1
                
                if total % 100 == 0:
                    print(f"   Precomputed {total} predictions...")
    
    print(f"✅ Precomputed {total} predictions (expected: {len(stations)*len(directions)*len(dows)*len(hours)})")
    
def get_batch_directional_predictions(station_name, direction, base_time, num_hours=6):
    """
    Get predictions for the next `num_hours` hours using the global cache.
    No model loading – pure cache lookup.
    """
    results = []
    for i in range(num_hours):
        dt = base_time + timedelta(hours=i)
        results.append(get_directional_prediction(station_name, direction, dt))
    return results
def get_directional_prediction(station_name, direction, target_datetime=None):
    if target_datetime is None:
        target_datetime = Config.get_current_time()
    
    # ========== CLOSED CHECK FIRST — BEFORE ANY CACHE ==========
    # This must happen before request cache and precomputed cache lookups,
    # otherwise cached values for non-operating hours will be returned.
    hour = target_datetime.hour
    minute = target_datetime.minute
    current_time_decimal = hour + minute / 60
    
    OPERATING_START = 4.5   # 4:30 AM
    OPERATING_END = 22.5    # 10:30 PM
    
    if current_time_decimal < OPERATING_START or current_time_decimal >= OPERATING_END:
        return 0
    # ============================================================
    
    # Check request cache first
    cached = get_cached_prediction(station_name, direction, target_datetime)
    if cached is not None:
        return cached
    
    # Check precomputed prediction cache
    dow = target_datetime.weekday()
    cache_key = f"{station_name}_{direction}_{dow}_{hour}"
    print(f"🔍 CACHE LOOKUP: key={cache_key} in_cache={cache_key in _PREDICTION_CACHE} cache_size={len(_PREDICTION_CACHE)}")
    if cache_key in _PREDICTION_CACHE:
        congestion = _PREDICTION_CACHE[cache_key]
        # Store in request-level cache for this specific time
        set_cached_prediction(station_name, direction, target_datetime, congestion)
        return congestion
    
    # ========== START TIMING (fallback path — models must be loaded) ==========
    total_start = time.time()
    
    ensure_models_loaded(station_name, direction)
    
    directional_models, directional_scalers = get_models()
    
    if directional_models is None:
        return _get_operating_hours_fallback(target_datetime)
    
    model_key = f"{station_name}_{direction}"
    
    if model_key not in directional_models:
        return _get_operating_hours_fallback(target_datetime)
    
    try:
        # ========== TIMING: Feature extraction ==========
        t0 = time.time()
        from services.feature_engineering import get_scaled_feature_sequence
        features_scaled = get_scaled_feature_sequence(station_name, direction, target_datetime)
        print(f"⏱️ [get_directional_prediction] get_scaled_feature_sequence: {time.time()-t0:.4f}s")
        
        if features_scaled is None:
            return _get_operating_hours_fallback(target_datetime)
        
        # ========== TIMING: Get target scaler ==========
        t1 = time.time()
        target_scaler = directional_scalers.get(f'{model_key}_target')
        print(f"⏱️ [get_directional_prediction] get_target_scaler: {time.time()-t1:.4f}s")
        
        if target_scaler is None:
            return _get_operating_hours_fallback(target_datetime)
        
        # ========== TIMING: Model prediction ==========
        t2 = time.time()
        input_sequence = features_scaled.reshape(1, 24, -1)
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.float32)
        
        prediction_scaled = directional_models[model_key](input_tensor, training=False).numpy()
        raw_output = float(prediction_scaled[0][0])
        print(f"⏱️ [get_directional_prediction] model_predict: {time.time()-t2:.4f}s")
        
        # ========== TIMING: Inverse transform ==========
        t3 = time.time()
        passenger_count = float(target_scaler.inverse_transform([[raw_output]])[0][0])
        print(f"⏱️ [get_directional_prediction] inverse_transform: {time.time()-t3:.4f}s")
        
        # ========== TIMING: P90 calculation ==========
        t4 = time.time()
        p90 = get_p90_percentile(station_name, direction)
        if p90 <= 0:
            p90 = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
        print(f"⏱️ [get_directional_prediction] get_p90: {time.time()-t4:.4f}s")
        
        # Convert to congestion
        congestion = (passenger_count / p90) * 100
        congestion = max(0, min(congestion, 100))
        
        # ========== TIMING: Correction factors ==========
        t5 = time.time()
        correction_factors = get_correction_factors()
        factor = correction_factors.get(model_key, 1.0)
        congestion = congestion * factor
        congestion = max(0, min(congestion, 100))
        print(f"⏱️ [get_directional_prediction] correction_factors: {time.time()-t5:.4f}s")
        
        # ========== TIMING: Day of week adjustment ==========
        t6 = time.time()
        dow = target_datetime.weekday()
        if dow >= 5:      # weekend
            dow_factor = 0.7
        elif dow == 4:    # Friday
            dow_factor = 1.1
        elif dow == 0:    # Monday
            dow_factor = 1.05
        else:
            dow_factor = 1.0
        
        congestion = congestion * dow_factor
        congestion = max(0, min(congestion, 100))
        print(f"⏱️ [get_directional_prediction] dow_adjustment: {time.time()-t6:.4f}s")
        
        # Store in request cache
        set_cached_prediction(station_name, direction, target_datetime, congestion)
        
        # ========== TOTAL TIME ==========
        total_elapsed = time.time() - total_start
        if total_elapsed > 0.5:
            print(f"⏱️ [get_directional_prediction] TOTAL: {total_elapsed:.4f}s ⚠️ SLOW")
        else:
            print(f"⏱️ [get_directional_prediction] TOTAL: {total_elapsed:.4f}s")
        
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
    start_total = time.time()  # define at the start
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

    # ========== GET ACTIVE OVERRIDES (ONCE) ==========
    t0 = time.time()
    active_overrides = get_active_overrides()
    print(f"⏱️ get_active_overrides: {time.time()-t0:.3f}s")

    print(f"\n🔍 DIRECTIONAL FORECAST for {name}")
    print(f"   Active overrides: {list(active_overrides.keys())}")
    north_key = f"{name}_northbound"
    south_key = f"{name}_southbound"
    print(f"   North override exists: {north_key in active_overrides}")
    print(f"   South override exists: {south_key in active_overrides}")

    # ========== BATCH PREDICTIONS (6 hours per direction) ==========
    t1 = time.time()
    north_preds = get_batch_directional_predictions(name, 'Northbound', base_time, num_hours=6)
    south_preds = get_batch_directional_predictions(name, 'Southbound', base_time, num_hours=6)
    print(f"⏱️ Batch predictions (both dirs): {time.time()-t1:.3f}s")

    # ========== BUILD FORECASTS WITH OVERRIDES ==========
    forecasts = []
    for i in range(6):
        target_time = base_time + timedelta(hours=i)
        # Start with batch predictions
        north_cong = north_preds[i]
        south_cong = south_preds[i]

        is_north_overridden = False
        is_south_overridden = False

        # ========== CHECK NORTHBOUND OVERRIDE ==========
        if north_key in active_overrides:
            override = active_overrides[north_key]
            override_congestion = override.get('congestion', 50)
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
            if is_override_active(override, target_time):
                south_cong = override_congestion
                is_south_overridden = True
                print(f"🔧 OVERRIDE ACTIVE: {name} Southbound at {target_time} -> {south_cong}%")
            else:
                print(f"⏰ Override NOT active for {target_time}")

        # Ensure values are numbers (should already be)
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

    # ========== TOTAL TIME ==========
    print(f"⏱️ TOTAL directional_forecast: {time.time()-start_total:.3f}s")

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

@api_predict_bp.route('/directional-forecast/all')
def directional_forecast_all():
    """Get current congestion for ALL stations at once."""
    print(f"\n[PREDICTION API] Getting all stations at {Config.get_current_time().strftime('%H:%M:%S')}")
    data = get_all_stations_predictions()
    return jsonify(data)

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


@api_predict_bp.route('/predict/<station_name>')
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
     