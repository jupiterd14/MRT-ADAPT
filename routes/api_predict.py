"""
API Predict - Core prediction endpoints
Single source of truth for P90, congestion calculation, and directional predictions.
Optimized for Render free tier (lazy TF, minimal disk I/O, in-memory caches).
"""

from flask import Blueprint, request, jsonify, current_app
from extensions import cache
from datetime import datetime, timedelta
from services.feature_engineering import (
    get_feature_sequence_for_station,
    get_scaled_feature_sequence,
    get_station_dataframe,
    get_station_dataframe_cached,
    get_typical_pattern,
    STATION_NUMBERS,
)
from config import Config
from constants import MRT3_PLATFORM_CAPACITY
import numpy as np
import json
import os
import pickle
import time


# ============================================================
# LAZY TENSORFLOW IMPORT (Render-friendly)
# ============================================================
class _LazyTF:
    _mod = None
    def __getattr__(self, name):
        if self._mod is None:
            import tensorflow as _t
            self._mod = _t
        return getattr(self._mod, name)


tf = _LazyTF()


api_predict_bp = Blueprint('api_predict', __name__)

STATIONS = [
    "North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan",
    "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia",
    "Ayala Ave", "Magallanes", "Taft"
]


# ============================================================
# MODULE-LEVEL CACHES
# ============================================================
_PREDICTION_CACHE = {}          # Precomputed (station, dir, dow, hour) → congestion
_REQUEST_CACHE = {}             # Per-request cache (hour granularity)
_REQUEST_CACHE_TTL = 300        # 5 minutes
_LAST_CLEANUP = [0.0]           # Timestamp of last periodic cleanup
_PENDING_CORRECTION_FACTORS = {}
_P90_FILE = 'p90_percentiles.json'
CORRECTION_FILE = 'correction_factors.pkl'

# Model cache (references to app.config dicts)
_models_cache = None
_scalers_cache = None


# ============================================================
# REQUEST-LEVEL CACHE
# ============================================================
def get_cached_prediction(station_name, direction, target_datetime):
    if target_datetime is None:
        target_datetime = Config.get_current_time()

    cache_key = f"{station_name}_{direction}_{target_datetime.strftime('%Y%m%d%H')}"

    # Periodic cleanup (every 60s, not every call)
    now = time.time()
    if now - _LAST_CLEANUP[0] > 60:
        _LAST_CLEANUP[0] = now
        expired = [
            k for k, v in _REQUEST_CACHE.items()
            if now - v['timestamp'] > _REQUEST_CACHE_TTL
        ]
        for k in expired:
            del _REQUEST_CACHE[k]

    entry = _REQUEST_CACHE.get(cache_key)
    return entry['value'] if entry else None


def set_cached_prediction(station_name, direction, target_datetime, value):
    if target_datetime is None:
        target_datetime = Config.get_current_time()

    cache_key = f"{station_name}_{direction}_{target_datetime.strftime('%Y%m%d%H')}"
    # Cast to native float to prevent numpy types from leaking into JSON
    _REQUEST_CACHE[cache_key] = {'value': float(value), 'timestamp': time.time()}


# ============================================================
# CORRECTION FACTORS
# ============================================================
def get_correction_factors():
    """Get correction factors from app config (persistent across requests)."""
    try:
        if 'CORRECTION_FACTORS' not in current_app.config:
            global _PENDING_CORRECTION_FACTORS
            if _PENDING_CORRECTION_FACTORS:
                current_app.config['CORRECTION_FACTORS'] = _PENDING_CORRECTION_FACTORS
                _PENDING_CORRECTION_FACTORS = {}
            else:
                current_app.config['CORRECTION_FACTORS'] = {}
        return current_app.config['CORRECTION_FACTORS']
    except RuntimeError:
        # Outside app context (e.g., during startup)
        return _PENDING_CORRECTION_FACTORS


def load_correction_factors():
    """Load correction factors into app config (call at startup)."""
    correction_factors = get_correction_factors()

    if os.path.exists(CORRECTION_FILE):
        try:
            with open(CORRECTION_FILE, 'rb') as f:
                factors = pickle.load(f)
                correction_factors.update(factors)
            print(f"✅ Loaded {len(factors)} correction factors")
        except Exception as e:
            print(f"⚠️ Could not load correction factors: {e}")


# ============================================================
# P90 CACHE - SINGLE SOURCE OF TRUTH
# ============================================================
def get_p90_cache():
    """Get P90 cache from app config (persistent across requests)."""
    if 'P90_CACHE' not in current_app.config:
        current_app.config['P90_CACHE'] = {}
    return current_app.config['P90_CACHE']

def get_p90_percentile(station_name, direction):
    """
    Get P90 percentile for a station-direction.
    Single source of truth for all congestion calculations.

    Uses MAX over per-hour P90s (peak-hour capacity) so congestion
    is meaningful across the day. Prevents evening rush from clamping
    at 100% when the global P90 is dragged down by off-peak hours.

    Priority:
      1. App config cache (fastest)
      2. Compute from historical data
      3. Fallback to platform capacity
    """
    key = f"{station_name}_{direction}"
    p90_cache = get_p90_cache()

    # 1. Check cache
    if key in p90_cache:
        return p90_cache[key]

    # 2. Compute from data
    hourly = get_station_dataframe_cached(station_name, direction)

    if hourly is not None and len(hourly) > 0:
        # ── Compute P90 for EACH hour of day, then take the max ──
        hour_p90s = []
        for hr in range(24):
            hr_data = hourly[hourly.index.hour == hr]
            if len(hr_data) > 0:
                hr_passengers = hr_data['TotalPassenger'].values
                hr_non_zero = hr_passengers[hr_passengers > 0]
                if len(hr_non_zero) > 0:
                    hr_p90 = float(np.percentile(hr_non_zero, 90))
                    hour_p90s.append(hr_p90)

        if hour_p90s:
            # ✅ MAX of per-hour P90s = peak-hour capacity
            p90 = max(hour_p90s)
            p90 = max(p90, 100)  # Minimum sanity floor

            # Cap at platform capacity × 10 to prevent absurd values
            capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
            max_reasonable = capacity * 10
            if p90 > max_reasonable:
                p90 = max_reasonable
        else:
            # No non-zero data → use platform capacity
            p90 = float(MRT3_PLATFORM_CAPACITY.get(station_name, 1000))
    else:
        # No data at all → use platform capacity (NOT flat 1000)
        p90 = float(MRT3_PLATFORM_CAPACITY.get(station_name, 1000))

    # Store in app config cache
    p90_cache[key] = p90

    # 3. Persist to disk (atomic write to avoid race conditions)
    _persist_p90_to_disk(key, p90)

    return p90


def _persist_p90_to_disk(key, value):
    """Atomic write of a single P90 value to disk. Swallows errors silently."""
    try:
        p90_file = _P90_FILE
        all_p90 = {}
        if os.path.exists(p90_file):
            try:
                with open(p90_file, 'r') as f:
                    all_p90 = json.load(f)
            except (json.JSONDecodeError, IOError):
                all_p90 = {}

        all_p90[key] = float(value)   # ✓ already cast

        tmp_file = p90_file + '.tmp'
        with open(tmp_file, 'w') as f:
            json.dump(all_p90, f, indent=2)
        os.replace(tmp_file, p90_file)
    except Exception as e:
        print(f"⚠️ Could not persist P90 to disk: {e}")
        
def preload_p90_cache():
    """Preload all P90 values from file, or compute them if file is missing."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p90_file = os.path.join(script_dir, _P90_FILE)

    if os.path.exists(p90_file):
        try:
            with open(p90_file, 'r') as f:
                p90_data = json.load(f)

            if 'P90_CACHE' not in current_app.config:
                current_app.config['P90_CACHE'] = {}
            current_app.config['P90_CACHE'].update(p90_data)
            print(f"✅ Loaded {len(p90_data)} P90 values from cache")
            return True
        except Exception as e:
            print(f"⚠️ Error loading P90 cache: {e}")

    # File missing → compute all
    print("🔄 Calculating P90 values from data...")
    for station in STATION_NUMBERS.keys():
        for direction in ['Northbound', 'Southbound']:
            get_p90_percentile(station, direction)

    print(f"✅ Preloaded {len(current_app.config.get('P90_CACHE', {}))} P90 values")
    return True


# ============================================================
# TYPICAL PATTERN PRELOAD
# ============================================================
def preload_typical_patterns():
    """Pre-compute typical patterns for all station-direction pairs."""
    stations = current_app.config.get('STATIONS', STATIONS)
    directions = ['Northbound', 'Southbound']

    print("🔄 Preloading typical patterns for all stations...")
    for station in stations:
        for direction in directions:
            try:
                pattern = get_typical_pattern(station, direction)
                if pattern is not None:
                    print(f"✅ Preloaded {station} {direction} (length {len(pattern)})")
            except Exception as e:
                print(f"⚠️ Failed to preload {station} {direction}: {e}")
    print("✅ Typical patterns preloaded.")


# ============================================================
# MODEL CACHE
# ============================================================
def get_models():
    """Get models from app config (with local cache)."""
    global _models_cache, _scalers_cache

    if _models_cache is not None:
        return _models_cache, _scalers_cache

    directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})

    if directional_models:
        _models_cache = directional_models
        _scalers_cache = directional_scalers
        return _models_cache, _scalers_cache

    return None, None


def set_models(models, scalers):
    """Update the local model cache."""
    global _models_cache, _scalers_cache
    _models_cache = models
    _scalers_cache = scalers


def clear_models_cache():
    """Clear the local model cache (useful for hot reloads)."""
    global _models_cache, _scalers_cache
    _models_cache = None
    _scalers_cache = None
# 1. app.py — make ensure_models_loaded a strict no-op
def ensure_models_loaded(station_name=None, direction=None):
    """Cache-only mode: never auto-load models."""
    return

# ============================================================
# RAW PREDICTION (passengers, not congestion)
# ============================================================
def get_raw_prediction(station_name, direction, target_datetime):
    """Returns the raw passenger count from the model."""
    ensure_models_loaded(station_name, direction)

    directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})

    model_key = f"{station_name}_{direction}"
    if model_key not in directional_models:
        return None

    try:
        features_scaled = get_scaled_feature_sequence(station_name, direction, target_datetime)
        if features_scaled is None:
            return None

        target_scaler = directional_scalers.get(f'{model_key}_target')
        if target_scaler is None:
            return None

        input_sequence = features_scaled.reshape(1, 24, -1)
        raw_scaled = directional_models[model_key].predict(input_sequence, verbose=0)[0][0]
        passenger_count = float(target_scaler.inverse_transform([[raw_scaled]])[0][0])

        return max(0.0, passenger_count)
    except Exception as e:
        print(f"⚠️ get_raw_prediction error: {e}")
        return None


def compute_and_save_correction_factors(test_days=30, end_date=None):
    """Compute correction factors by comparing model vs actual for all station-directions."""
    factors = {}
    if end_date is None:
        end_date = Config.get_current_time()
    start_date = end_date - timedelta(days=test_days)

    for station in STATIONS:
        for direction in ['Northbound', 'Southbound']:
            model_key = f"{station}_{direction}"
            if model_key not in current_app.config.get('DIRECTIONAL_MODELS', {}):
                continue

            df = get_station_dataframe(station, direction)
            if df is None or len(df) == 0:
                continue

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
                if 0.1 < ratio < 10:
                    ratios.append(ratio)

            if ratios:
                factor = float(np.median(ratios))
                factor = min(1.5, factor)  # Clamp
                factors[model_key] = factor
                print(f"✅ {station} {direction}: factor = {factor:.3f} ({len(ratios)} samples)")

    with open(CORRECTION_FILE, 'wb') as f:
        pickle.dump(factors, f)
    print(f"✅ Saved {len(factors)} correction factors to {CORRECTION_FILE}")
    return factors


# ============================================================
# HISTORICAL PEAKS (reference only)
# ============================================================
def get_historical_peak(station_name, direction):
    """Lazy-load historical peak for one station-direction."""
    key = f"{station_name}_{direction}"

    if 'HISTORICAL_PEAKS' not in current_app.config:
        current_app.config['HISTORICAL_PEAKS'] = {}
    historical_peaks = current_app.config['HISTORICAL_PEAKS']

    if key in historical_peaks:
        return historical_peaks[key]

    hourly = get_station_dataframe_cached(station_name, direction)
    if hourly is not None and len(hourly) > 0:
        peak_abs = float(hourly['TotalPassenger'].values.max())
        historical_peaks[key] = {
            "peak": peak_abs,
            "absolute_max": peak_abs,
            "percentile": 100
        }
        return historical_peaks[key]

    return None


# ============================================================
# OVERRIDES
# ============================================================
def get_active_overrides():
    """
    Get active overrides using Config time for consistency across modules.
    Handles expiry in UTC epoch seconds.
    """
    overrides_file = 'overrides.json'

    if not os.path.exists(overrides_file):
        return {}

    try:
        with open(overrides_file, 'r') as f:
            all_overrides = json.load(f)

        config_time = Config.get_current_time()
        now_timestamp = config_time.timestamp()

        active = {}
        for key, override in all_overrides.items():
            expiry = override.get('expiry')
            if expiry is None or expiry > now_timestamp:
                active[key] = override
            else:
                print(f"⏰ Override expired: {key}")
        return active
    except Exception as e:
        print(f"Error loading overrides: {e}")
        return {}


def is_override_active(override, target_time):
    """
    Check if an override is active for the given target time.
    Handles overnight windows correctly.
    """
    expiry = override.get('expiry')
    timestamp = override.get('timestamp')
    duration_minutes = override.get('duration_minutes', 60)

    if not expiry or not timestamp:
        return False

    try:
        config_time = Config.get_current_time()
        now_timestamp = config_time.timestamp()

        # First check: has the override already expired?
        if expiry <= now_timestamp:
            return False

        override_start = datetime.fromisoformat(timestamp)
        target_date = target_time.date()

        # Overlay the override's time-of-day onto the target date
        override_start_in_target_date = datetime(
            target_date.year, target_date.month, target_date.day,
            override_start.hour, override_start.minute,
            override_start.second, override_start.microsecond
        )
        override_end = override_start_in_target_date + timedelta(minutes=duration_minutes)

        # Handle overnight windows (end time crosses midnight)
        if override_end.day != override_start_in_target_date.day:
            return (
                target_time >= override_start_in_target_date
                or target_time <= override_end
            )
        return override_start_in_target_date <= target_time <= override_end

    except Exception as e:
        print(f"⚠️ Error in is_override_active: {e}")
        return False


# ============================================================
# PRECOMPUTE ALL PREDICTIONS
# ============================================================
def precompute_all_predictions():
    """
    Precompute congestion for all (station, direction, dow, hour) combos.
    Stores in _PREDICTION_CACHE for O(1) lookups.

    All cached values are native Python floats to keep jsonify happy.
    """
    stations = list(STATION_NUMBERS.keys())
    directions = ['Northbound', 'Southbound']
    dows = list(range(7))
    hours = list(range(24))

    ensure_models_loaded()
    directional_models, directional_scalers = get_models()

    if directional_models is None:
        print("❌ precompute_all_predictions: models not available")
        return

    total = 0
    for station in stations:
        for direction in directions:
            model_key = f"{station}_{direction}"
            model = directional_models.get(model_key)
            target_scaler = directional_scalers.get(f'{model_key}_target')
            if model is None or target_scaler is None:
                print(f"⚠️ Skipping {model_key} – model or scaler missing")
                continue

            p90 = float(get_p90_percentile(station, direction))
            if p90 <= 0:
                p90 = float(MRT3_PLATFORM_CAPACITY.get(station, 1000))

            correction_factors = get_correction_factors()
            factor = float(correction_factors.get(model_key, 1.0))

            for dow in dows:
                features_list = []
                for hour in hours:
                    base_dt = datetime(2025, 1, 6) + timedelta(days=dow, hours=hour)
                    features = get_scaled_feature_sequence(station, direction, base_dt)
                    if features is None:
                        features = np.zeros((24, 16), dtype=np.float32)
                    features_list.append(features)

                batch_features = np.stack(features_list, axis=0).astype(np.float32)

                pred_scaled = model.predict(batch_features, verbose=0)
                pred_passengers = target_scaler.inverse_transform(pred_scaled).flatten()

                for hour, passenger in enumerate(pred_passengers):
                    # ── All arithmetic on native floats ──
                    passenger = float(passenger)
                    congestion = (passenger / p90) * 100.0
                    congestion = max(0.0, min(congestion, 100.0))
                    congestion = congestion * factor
                    congestion = max(0.0, min(congestion, 100.0))

                    if dow >= 5:
                        dow_factor = 0.7
                    elif dow == 4:
                        dow_factor = 1.1
                    elif dow == 0:
                        dow_factor = 1.05
                    else:
                        dow_factor = 1.0
                    congestion = congestion * dow_factor
                    congestion = max(0.0, min(congestion, 100.0))

                    cache_key = f"{station}_{direction}_{dow}_{hour}"
                    _PREDICTION_CACHE[cache_key] = float(congestion)   # ← CAST
                    total += 1

                if total % 100 == 0:
                    print(f"   Precomputed {total} predictions...")

    expected = len(stations) * len(directions) * len(dows) * len(hours)
    print(f"✅ Precomputed {total} predictions (expected: {expected})")
# ============================================================
# BATCH PREDICTIONS
# ============================================================
def get_batch_directional_predictions(station_name, direction, base_time, num_hours=6):
    """Get predictions for the next `num_hours` hours via cache."""
    return [
        get_directional_prediction(station_name, direction, base_time + timedelta(hours=i))
        for i in range(num_hours)
    ]


# ============================================================
# MAIN PREDICTION FUNCTION (SINGLE SOURCE OF TRUTH)
# ============================================================
def get_directional_prediction(station_name, direction, target_datetime=None):
    if target_datetime is None:
        target_datetime = Config.get_current_time()

    hour = target_datetime.hour
    minute = target_datetime.minute
    current_time_decimal = hour + minute / 60

    OPERATING_START = 4.5
    OPERATING_END = 22.5

    if current_time_decimal < OPERATING_START or current_time_decimal >= OPERATING_END:
        return 0

    cached = get_cached_prediction(station_name, direction, target_datetime)
    if cached is not None:
        return float(cached)   # ← CAST

    dow = target_datetime.weekday()
    cache_key = f"{station_name}_{direction}_{dow}_{hour}"
    if cache_key in _PREDICTION_CACHE:
        congestion = float(_PREDICTION_CACHE[cache_key])   # ← CAST
        set_cached_prediction(station_name, direction, target_datetime, congestion)
        return congestion

    return float(_live_model_prediction(station_name, direction, target_datetime) or 0)   # ← CAST


def _live_model_prediction(station_name, direction, target_datetime):
    """Slow path: run model inference for a single datetime."""
    total_start = time.time()

    ensure_models_loaded(station_name, direction)
    directional_models, directional_scalers = get_models()

    if directional_models is None:
        return _get_operating_hours_fallback(target_datetime)

    model_key = f"{station_name}_{direction}"
    if model_key not in directional_models:
        return _get_operating_hours_fallback(target_datetime)

    try:
        features_scaled = get_scaled_feature_sequence(station_name, direction, target_datetime)
        if features_scaled is None:
            return _get_operating_hours_fallback(target_datetime)

        target_scaler = directional_scalers.get(f'{model_key}_target')
        if target_scaler is None:
            return _get_operating_hours_fallback(target_datetime)

        # Model inference
        input_sequence = features_scaled.reshape(1, 24, -1)
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.float32)
        prediction_scaled = directional_models[model_key](input_tensor, training=False).numpy()
        raw_output = float(prediction_scaled[0][0])

        # Inverse transform
        passenger_count = float(target_scaler.inverse_transform([[raw_output]])[0][0])

        # P90 normalization
        p90 = get_p90_percentile(station_name, direction)
        if p90 <= 0:
            p90 = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)

        congestion = (passenger_count / p90) * 100
        congestion = max(0.0, min(congestion, 100.0))

        # Correction factor
        correction_factors = get_correction_factors()
        factor = correction_factors.get(model_key, 1.0)
        congestion = congestion * factor
        congestion = max(0.0, min(congestion, 100.0))

        # DOW adjustment
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
        congestion = max(0.0, min(congestion, 100.0))

        congestion = float(congestion)
        set_cached_prediction(station_name, direction, target_datetime, congestion)

        total_elapsed = time.time() - total_start
        if total_elapsed > 0.5:
            print(f"⏱️ [live] {station_name} {direction}: {total_elapsed:.3f}s ⚠️ SLOW")

        return congestion

    except Exception as e:
        print(f"⚠️ _live_model_prediction error: {e}")
        import traceback
        traceback.print_exc()
        return _get_operating_hours_fallback(target_datetime)


# ============================================================
# FALLBACKS
# ============================================================
def _get_operating_hours_fallback(target_datetime):
    """Realistic fallback based on hour of day."""
    hour = target_datetime.hour
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        return 65
    elif 10 <= hour <= 16:
        return 45
    elif 5 <= hour <= 6 or 20 <= hour <= 21:
        return 25
    else:
        return 10


def get_fallback_directional_prediction(station_name, direction, target_datetime=None):
    """Fallback when models aren't available."""
    target_datetime = target_datetime or Config.get_current_time()
    hour = target_datetime.hour
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
    """Clamp prediction to reasonable ranges based on time of day."""
    target_datetime = target_datetime or Config.get_current_time()
    hour = target_datetime.hour
    current_time = hour + target_datetime.minute / 60

    if current_time < 4.5 or current_time >= 22.5:
        return 0

    if 7 <= hour <= 9 or 17 <= hour <= 19:
        return max(30, min(congestion, 100))
    elif 10 <= hour <= 16:
        return max(10, min(congestion, 80))
    else:
        return max(0, min(congestion, 60))


# ============================================================
# HELPERS
# ============================================================
def get_best_time_to_travel(station_name=None):
    now = Config.get_current_time()
    hour = now.hour
    if 7 <= hour <= 9:
        return "10:00 AM - 3:00 PM (Avoid morning rush hour)"
    elif 17 <= hour <= 20:
        return "Before 5:00 PM or after 8:00 PM (Avoid evening rush hour)"
    return "Now is a good time to travel!"


def get_wait_time(congestion):
    if congestion > 80:
        return "15-20 min"
    elif congestion > 60:
        return "10-15 min"
    elif congestion > 30:
        return "5-10 min"
    else:
        return "2-5 min"

def get_all_stations_predictions():
    result = {"northbound": {}, "southbound": {}}
    now = Config.get_current_time()

    current_time = now.hour + now.minute / 60
    is_closed = current_time < 4.5 or current_time >= 22.5

    def get_status(cong):
        if cong > 80:
            return "SEVERE"
        if cong > 50:
            return "CONGESTED"
        if cong > 25:
            return "MODERATE"
        return "LIGHT"

    for station in STATIONS:
        if is_closed:
            result['northbound'][station] = {"congestion": 0.0, "status": "CLOSED"}
            result['southbound'][station] = {"congestion": 0.0, "status": "CLOSED"}
            continue

        north_cong = float(get_directional_prediction(station, 'Northbound', now) or 0)
        south_cong = float(get_directional_prediction(station, 'Southbound', now) or 0)

        result['northbound'][station] = {
            "congestion": round(north_cong, 1),
            "status": get_status(north_cong)
        }
        result['southbound'][station] = {
            "congestion": round(south_cong, 1),
            "status": get_status(south_cong)
        }

    return result


def _get_directional_prediction_with_details(station_name, direction, target_datetime):
    """Returns both congestion and passenger count (for debug endpoints)."""
    ensure_models_loaded(station_name, direction)

    directional_models, directional_scalers = get_models()
    correction_factors = get_correction_factors()
    model_key = f"{station_name}_{direction}"

    result = {"congestion": 0, "passengers": 0, "raw_output": 0}

    if directional_models is None or model_key not in directional_models:
        return result

    try:
        features_scaled = get_scaled_feature_sequence(station_name, direction, target_datetime)
        if features_scaled is None:
            return result

        target_scaler = directional_scalers.get(f'{model_key}_target')
        if target_scaler is None:
            return result

        input_sequence = features_scaled.reshape(1, 24, -1)
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.float32)
        prediction_scaled = directional_models[model_key](input_tensor, training=False).numpy()
        raw_output = float(prediction_scaled[0][0])

        passenger_count = float(target_scaler.inverse_transform([[raw_output]])[0][0])

        p90 = get_p90_percentile(station_name, direction)
        if p90 <= 0:
            p90 = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)

        congestion = (passenger_count / p90) * 100
        congestion = max(0.0, min(congestion, 100.0))

        factor = correction_factors.get(model_key, 1.0)
        congestion = congestion * factor
        congestion = max(0.0, min(congestion, 100.0))

        result["congestion"] = congestion
        result["passengers"] = passenger_count
        result["raw_output"] = raw_output

    except Exception as e:
        print(f"Error in _get_directional_prediction_with_details: {e}")

    return result


def get_station_prediction(station_name):
    """Average congestion for a station (both directions)."""
    north = get_directional_prediction(station_name, 'Northbound') or 0
    south = get_directional_prediction(station_name, 'Southbound') or 0
    return (north + south) / 2


# ============================================================
# ENDPOINTS
# ============================================================
@api_predict_bp.route('/directional-forecast/<station_name>')
def directional_forecast(station_name):
    start_total = time.time()
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

    active_overrides = get_active_overrides()

    north_preds = get_batch_directional_predictions(name, 'Northbound', base_time, num_hours=6)
    south_preds = get_batch_directional_predictions(name, 'Southbound', base_time, num_hours=6)

    north_key = f"{name}_northbound"
    south_key = f"{name}_southbound"

    forecasts = []
    for i in range(6):
        target_time = base_time + timedelta(hours=i)
        north_cong = north_preds[i]
        south_cong = south_preds[i]
        is_north_overridden = False
        is_south_overridden = False

        if north_key in active_overrides:
            override = active_overrides[north_key]
            if is_override_active(override, target_time):
                north_cong = override.get('congestion', 50)
                is_north_overridden = True

        if south_key in active_overrides:
            override = active_overrides[south_key]
            if is_override_active(override, target_time):
                south_cong = override.get('congestion', 50)
                is_south_overridden = True

        north_cong = north_cong if north_cong is not None else 0
        south_cong = south_cong if south_cong is not None else 0

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

    print(f"⏱️ TOTAL directional_forecast: {time.time()-start_total:.3f}s")

    return jsonify({
        "station": name,
        "timestamp": base_time.isoformat(),
        "active_overrides": len(active_overrides),
        "override_details": {
            k: active_overrides.get(k)
            for k in [north_key, south_key] if k in active_overrides
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
    """Get current congestion for ALL stations."""
    print(f"\n[PREDICTION API] Getting all stations at {Config.get_current_time().strftime('%H:%M:%S')}")
    return jsonify(get_all_stations_predictions())


@api_predict_bp.route('/model-evaluation')
def model_evaluation():
    """Evaluate model performance using the same P90 as the API."""
    from sklearn.metrics import (
        confusion_matrix, classification_report, accuracy_score, f1_score,
        mean_absolute_error, mean_squared_error, r2_score
    )

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

    p90 = get_p90_percentile(station, direction)

    def get_congestion_category(cong):
        if cong > 80:
            return "Severe"
        elif cong > 50:
            return "Heavy"
        elif cong > 25:
            return "Moderate"
        return "Light"

    predictions = []
    actuals = []

    end_date = df.index.max()
    start_date = end_date - timedelta(days=test_days)
    test_data = df[(df.index >= start_date) & (df.index < end_date)]

    for timestamp in test_data.index:
        actual_passengers = test_data.loc[timestamp, 'TotalPassenger']
        actual_congestion = min((actual_passengers / p90) * 100, 100)

        try:
            features_scaled = get_scaled_feature_sequence(station, direction, timestamp)
            if features_scaled is None:
                continue

            target_scaler = directional_scalers.get(f'{model_key}_target')
            if target_scaler is None:
                continue

            input_sequence = features_scaled.reshape(1, 24, -1)
            pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
            raw_value = float(pred_scaled[0][0])

            passenger_count = float(target_scaler.inverse_transform([[raw_value]])[0][0])
            factor = correction_factors.get(model_key, 1.0)
            passenger_count = passenger_count * factor

            predicted_congestion = min((passenger_count / p90) * 100, 100)
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
        "p90_percentile": round(p90, 2),
        "test_period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "total_hours_tested": len(predictions)
        },
        "confusion_matrix": {"labels": categories, "matrix": cm.tolist()},
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
    """Deprecated. Use POST /api/model/evaluate."""
    station = request.args.get('station', 'North Ave')
    direction = request.args.get('direction', 'Northbound')

    return jsonify({
        "success": False,
        "message": "Use POST /api/model/evaluate instead. This endpoint is deprecated.",
        "station": station,
        "direction": direction
    })


@api_predict_bp.route('/test-rush-hour')
def test_rush_hour():
    """Test predictions for rush hour times."""
    now = Config.get_current_time()
    results = {}

    for hour in [8, 12, 18, 21]:
        test_time = now.replace(hour=hour, minute=0)
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
    """Current snapshot for a single station."""
    name = station_name.replace('%20', ' ')

    date_param = request.args.get('date')
    time_param = request.args.get('time')
    target_datetime = None

    if date_param and time_param:
        try:
            year, month, day = map(int, date_param.split('-'))
            hour, minute = map(int, time_param.split(':'))
            target_datetime = datetime(year, month, day, hour, minute)
        except Exception:
            target_datetime = None

    north_congestion = get_directional_prediction(name, 'Northbound', target_datetime) or 0
    south_congestion = get_directional_prediction(name, 'Southbound', target_datetime) or 0
    congestion = (north_congestion + south_congestion) / 2

    if congestion > 80:
        status = "CRITICAL"
    elif congestion > 50:
        status = "BUSY"
    elif congestion > 20:
        status = "MODERATE"
    else:
        status = "LIGHT"

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

    north_congestion = get_directional_prediction(name, 'Northbound') or 0
    south_congestion = get_directional_prediction(name, 'Southbound') or 0
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
        status, color, wait_time = "SEVERELY CONGESTED", "critical", "15-20 min"
    elif congestion > 50:
        status, color, wait_time = "CONGESTED", "congested", "10-15 min"
    elif congestion > 25:
        status, color, wait_time = "MODERATE", "moderate", "5-10 min"
    else:
        status, color, wait_time = "LIGHT", "light", "2-5 min"

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
            north_from = get_directional_prediction(from_station, 'Northbound', target_datetime) or 0
            south_from = get_directional_prediction(from_station, 'Southbound', target_datetime) or 0
            north_to = get_directional_prediction(to_station, 'Northbound', target_datetime) or 0
            south_to = get_directional_prediction(to_station, 'Southbound', target_datetime) or 0
            congestion_from = (north_from + south_from) / 2
            congestion_to = (north_to + south_to) / 2
        except Exception:
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
    """Recompute correction factors from historical data."""
    try:
        test_days = 30
        if request.is_json:
            data = request.get_json(silent=True) or {}
            test_days = data.get('test_days', 30)

        factors = compute_and_save_correction_factors(test_days=test_days)
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