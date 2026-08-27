"""
Feature engineering for LSTM predictions - OPTIMIZED FOR SPEED & MEMORY
"""

import os 
import gc
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import time
from functools import wraps

# ========== TIMING DECORATOR ==========
def timer(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"⏱️ {func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper

# ========== GLOBAL RAM CACHES ==========
_STATION_DATA_CACHE = {}      # Holds all loaded station Parquet DataFrames in RAM
_FEATURE_SCALER_CACHE = {}    # Cache for feature scalers
_TARGET_SCALER_CACHE = {}     # Cache for target scalers
_BASELINE_FEATURES_CACHE = {} # Cache for baseline features
_TYPICAL_PATTERN_CACHE = {}   # Cache for pre-calculated typical day patterns
_TYPICAL_PROFILE_CACHE = {}   # Cache for day-of-week passenger profiles
_SCALED_SEQUENCE_CACHE = {}   # Key: station_direction_dow_hour -> scaled array

# ========== IMPORT CONSTANTS (avoid duplication) ==========
from constants import MRT3_PLATFORM_CAPACITY

# Station numbers
STATION_NUMBERS = {
    "North Ave": 1, "Quezon Ave": 2, "Kamuning": 3, "Cubao": 4,
    "Santolan": 5, "Ortigas": 6, "Shaw Blvd": 7, "Boni Ave": 8,
    "Guadalupe": 9, "Buendia": 10, "Ayala Ave": 11, "Magallanes": 12,
    "Taft": 13
}

# FEATURE_COLS must match the training order
FEATURE_COLS = [
    'TotalPassenger',
    'hour', 'weekday', 'month',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'is_operating_hour', 'is_morning_rush', 'is_evening_rush',
    'is_holiday', 'is_christmas_season', 'is_payday'
]

# ========== HOLIDAYS (match training) ==========
HOLIDAYS = [
    '2022-01-01', '2022-04-09', '2022-04-14', '2022-04-15', '2022-04-16',
    '2022-05-01', '2022-06-12', '2022-08-21', '2022-08-29', '2022-11-30',
    '2022-12-08', '2022-12-25', '2022-12-30', '2022-12-31',
    '2023-01-01', '2023-04-06', '2023-04-07', '2023-05-01', '2023-06-12',
    '2023-08-28', '2023-11-27', '2023-12-08', '2023-12-25', '2023-12-30',
    '2024-01-01', '2024-03-28', '2024-03-29', '2024-05-01', '2024-06-12',
    '2024-08-26', '2024-11-30', '2024-12-08', '2024-12-25', '2024-12-30', '2024-12-31',
    '2025-01-01', '2025-04-09', '2025-04-17', '2025-04-18', '2025-05-01',
    '2025-06-12', '2025-08-21', '2025-08-25', '2025-11-30', '2025-12-08',
    '2025-12-25', '2025-12-30', '2025-12-31'
]

def is_holiday(date):
    """Check if a date is a Philippine holiday (matches training)"""
    return date.strftime('%Y-%m-%d') in HOLIDAYS

def is_christmas_season(date):
    month_day = date.strftime('%m-%d')
    return (month_day >= '12-15') or (month_day <= '01-05')

def is_payday(date):
    return date.day in [15, 30, 31]

def is_friday(date):
    return date.weekday() == 4


# ============================================================
# HELPER / SCALER FUNCTIONS
# ============================================================

def get_synthetic_congestion(hour):
    """Generate synthetic congestion for a given hour (0-23)"""
    if 0 <= hour <= 4:
        return 2.0
    elif 5 <= hour <= 6:
        return 5.0 + (hour - 5) * 5.0
    elif 7 <= hour <= 9:
        return 20.0 + (hour - 7) * 15.0
    elif 10 <= hour <= 11:
        return 35.0 + (hour - 10) * 5.0
    elif 12 <= hour <= 13:
        return 40.0
    elif 14 <= hour <= 16:
        return 35.0 + (hour - 14) * 5.0
    elif 17 <= hour <= 19:
        return 50.0 + (hour - 17) * 15.0
    elif 20 <= hour <= 21:
        return 60.0 - (hour - 20) * 15.0
    elif 22 <= hour <= 23:
        return 25.0 - (hour - 22) * 10.0
    else:
        return 10.0

def get_feature_scaler(station_name, direction):
    cache_key = f"{station_name}_{direction}"
    if cache_key in _FEATURE_SCALER_CACHE:
        return _FEATURE_SCALER_CACHE[cache_key]
    
    model_folder = 'models_2022-2024_v10'
    scaler_path = f'{model_folder}/{cache_key}_feature_scaler.pkl'
    
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            _FEATURE_SCALER_CACHE[cache_key] = scaler
            return scaler
        except Exception:
            return None
    return None

def get_target_scaler(station_name, direction):
    cache_key = f"{station_name}_{direction}"
    if cache_key in _TARGET_SCALER_CACHE:
        return _TARGET_SCALER_CACHE[cache_key]
    
    model_folder = 'models_2022-2024_v10'
    scaler_path = f'{model_folder}/{cache_key}_target_scaler.pkl'
    if not os.path.exists(scaler_path):
        scaler_path = f'{model_folder}/{cache_key.replace(" ", "_")}_target_scaler.pkl'
    
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            _TARGET_SCALER_CACHE[cache_key] = scaler
            return scaler
        except Exception:
            _TARGET_SCALER_CACHE[cache_key] = None
    else:
        _TARGET_SCALER_CACHE[cache_key] = None
        
    return _TARGET_SCALER_CACHE[cache_key]

# ============================================================
# PARQUET & DATAFRAME LOADING (PERSISTENT IN RAM)
# ============================================================

def get_station_dataframe_cached(station_name, direction):
    """
    Loads Parquet dataset ONCE into RAM and keeps all 26 datasets available 
    to avoid repeated disk reading on live map polling.
    """
    global _STATION_DATA_CACHE
    cache_key = f"{station_name}_{direction}"
    
    # Return instantly if already in memory
    if cache_key in _STATION_DATA_CACHE:
        return _STATION_DATA_CACHE[cache_key]
    
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
    os.makedirs(cache_dir, exist_ok=True)
    parquet_file = os.path.join(cache_dir, f'{cache_key}.parquet')
    
    if os.path.exists(parquet_file):
        try:
            print(f"📦 Loading Parquet to RAM (One-Time): {cache_key}")
            df = pd.read_parquet(parquet_file)
            _STATION_DATA_CACHE[cache_key] = df
            return df
        except Exception as e:
            print(f"⚠️ Error loading Parquet for {cache_key}: {e}")
    
    # Generate from CSV if Parquet doesn't exist
    print(f"🔄 Generating Parquet cache for {cache_key} from CSV...")
    df = get_station_dataframe(station_name, direction)
    
    if df is not None and len(df) > 0:
        try:
            df.to_parquet(parquet_file, compression='gzip')
            print(f"✅ Saved cache to {parquet_file} ({len(df)} rows)")
            _STATION_DATA_CACHE[cache_key] = df
        except Exception as e:
            print(f"⚠️ Could not save Parquet: {e}")
            _STATION_DATA_CACHE[cache_key] = df
    
    return df

def get_station_dataframe(station_name, direction):
    """
    Memory-optimized - streams CSV files and only keeps the needed data
    CORRECTED directional filtering
    """
    print(f"📊 Loading real data for {station_name}_{direction} from CSV...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data (2022-2024)')
    
    csv_files = ['2022.csv', '2023.csv', '2024.csv']
    station_num = STATION_NUMBERS.get(station_name)
    
    if not station_num:
        print(f"❌ Unknown station: {station_name}")
        return None
    
    is_north_terminal = (station_name == "North Ave")
    is_south_terminal = (station_name == "Taft")
    
    all_filtered = []
    CHUNK_SIZE = 50000
    
    for csv_file in csv_files:
        filepath = os.path.join(data_dir, csv_file)
        if not os.path.exists(filepath):
            continue
            
        try:
            needed_columns = ['StationEntry', 'StationExit', 'Date', 'Time', 'TotalPassenger']
            
            for chunk in pd.read_csv(filepath, 
                                    chunksize=CHUNK_SIZE,
                                    usecols=needed_columns,
                                    dtype={'StationEntry': 'int16', 'StationExit': 'int16', 'TotalPassenger': 'float32'}):
                
                # ========== CORRECT DIRECTIONAL FILTERING ==========
                if direction == 'Northbound':
                    if is_north_terminal:
                        # North Ave: passengers EXIT (end of line)
                        filtered = chunk[chunk['StationExit'] == station_num]
                    elif is_south_terminal:
                        # Taft: passengers ENTER to go north
                        filtered = chunk[chunk['StationEntry'] == station_num]
                    else:
                        # All other stations: passengers ENTER to go north
                        filtered = chunk[chunk['StationEntry'] == station_num]
                else:  # Southbound
                    if is_north_terminal:
                        # North Ave: passengers ENTER to go south
                        filtered = chunk[chunk['StationEntry'] == station_num]
                    elif is_south_terminal:
                        # Taft: passengers EXIT (end of line)
                        filtered = chunk[chunk['StationExit'] == station_num]
                    else:
                        # All other stations: passengers EXIT to go south
                        filtered = chunk[chunk['StationExit'] == station_num]
                
                if len(filtered) > 0:
                    all_filtered.append(filtered)
                
                del chunk, filtered
                gc.collect()
                
        except Exception as e:
            print(f"   ⚠️ Error processing {csv_file}: {e}")
            continue
    
    if not all_filtered:
        return None
    
    combined = pd.concat(all_filtered)
    del all_filtered
    gc.collect()
        
    # ========== Create datetime index ==========
    print(f"   📅 Creating datetime index...")
    combined['datetime'] = pd.to_datetime(combined['Date'] + ' ' + combined['Time'])
    combined = combined.set_index('datetime')
    combined = combined[['TotalPassenger']]
    
    # ========== Resample with min_count=1 to preserve missing hours ==========
    print(f"   ⏰ Resampling to hourly...")
    combined = combined.resample('h').sum(min_count=1)
    
    # ========== FILL MISSING HOURLY PASSENGER COUNTS ==========
    missing_before = combined['TotalPassenger'].isna().sum()
    if missing_before > 0:
        combined['TotalPassenger'] = combined['TotalPassenger'].fillna(0)
        print(f"🔧 Filled {missing_before} missing hourly passenger records with 0")
    
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()
    
    print(f"   ✅ Loaded {len(combined)} hours of real data")
    print(f"   📊 Passenger stats - Min: {combined['TotalPassenger'].min():.0f}, Max: {combined['TotalPassenger'].max():.0f}")
    
    # ========== ADD TIME FEATURES ==========
    combined = add_cyclical_time_features(combined)
    combined = add_smart_operating_flags(combined)
    combined = smart_data_cleaner(combined)
    
    combined['hour'] = combined.index.hour
    combined['weekday'] = combined.index.weekday
    combined['month'] = combined.index.month
    
    # ========== CALCULATE CONGESTION ==========
    capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
    percentage = (combined['TotalPassenger'] / capacity * 100).clip(0, 100)
 
    combined['congestion'] = percentage
    combined['congestion_percentage'] = percentage
    combined['raw_passengers'] = combined['TotalPassenger']
    
    # Add date-based features
    combined['is_weekend'] = (combined.index.weekday >= 5).astype(np.int8)
    combined['is_christmas_season'] = np.array([is_christmas_season(d) for d in combined.index], dtype=np.int8)
    combined['is_payday'] = combined.index.day.isin([15, 30, 31]).astype(np.int8)
    combined['is_friday'] = (combined.index.weekday == 4).astype(np.int8)
    combined['is_rush_hour'] = ((combined['hour'].between(7, 9)) | (combined['hour'].between(17, 19))).astype(np.int8)
    combined['is_holiday'] = np.array([is_holiday(d) for d in combined.index], dtype=np.int8)
    combined['is_special_event'] = 0
    
    return combined

# ============================================================
# TYPICAL PATTERN BUILDING WITH DISK PERSISTENCE
# ============================================================

def build_typical_day_pattern_fast(station_name, direction, target_datetime, seq_length=24):
    """
    SUPER FAST version - uses day-of-week pattern cache
    No Pandas filtering, no median calculation during requests!
    """
    dow = target_datetime.weekday()
    cache_key = f"{station_name}_{direction}_{dow}"
    
    # Check if we have this day's pattern cached
    if cache_key in _TYPICAL_PROFILE_CACHE:
        hourly_passengers = _TYPICAL_PROFILE_CACHE[cache_key]
    else:
        # Build it once and cache it (should happen at startup)
        df = get_station_dataframe_cached(station_name, direction)
        if df is None:
            return get_baseline_features(target_datetime, seq_length)
        
        # Build the 24-hour profile for this day of week
        from routes.api_predict import get_p95_percentile
        p95 = get_p95_percentile(station_name, direction)
        hourly_passengers = []
        
        for hour in range(24):
            # Get data for this specific hour AND day of week
            hour_data = df[
                (df.index.hour == hour) & 
                (df.index.weekday == dow)
            ]
            
            if len(hour_data) > 0:
                passenger = float(hour_data['TotalPassenger'].median())
            else:
                # Fallback: same hour regardless of day
                fallback = df[df.index.hour == hour]
                if len(fallback) > 0:
                    passenger = float(fallback['TotalPassenger'].median())
                else:
                    passenger = get_synthetic_congestion(hour) / 100.0 * p95
            
            hourly_passengers.append(passenger)
        
        # Cache it for next time
        _TYPICAL_PROFILE_CACHE[cache_key] = hourly_passengers
    
    # ============================================================
    # BUILD THE 24-HOUR SEQUENCE (FAST - no Pandas filtering!)
    # ============================================================
    
    # Calculate the 24-hour window ending at target - 1 hour
    end_time = target_datetime.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    start_time = end_time - timedelta(hours=seq_length - 1)
    
    # Build sequence using the cached profile
    typical_passengers = []
    for i in range(seq_length):
        timestamp = start_time + timedelta(hours=i)
        hour = timestamp.hour
        typical_passengers.append(hourly_passengers[hour])
    
    # Create DataFrame (minimal overhead)
    index = pd.date_range(start=start_time, end=end_time, freq='h')
    typical_df = pd.DataFrame(index=index)
    typical_df['TotalPassenger'] = typical_passengers
    
    # Calculate congestion using cached P95
    from routes.api_predict import get_p95_percentile
    p95 = get_p95_percentile(station_name, direction)
    typical_df['congestion'] = (typical_df['TotalPassenger'] / p95 * 100).clip(0, 100)
    typical_df['congestion_percentage'] = typical_df['congestion']
    typical_df['raw_passengers'] = typical_df['TotalPassenger']
    
    # ========== Add all features (fast vectorized operations) ==========
    hours = typical_df.index.hour
    weekdays = typical_df.index.weekday
    months = typical_df.index.month
    
    typical_df['hour'] = hours
    typical_df['weekday'] = weekdays
    typical_df['month'] = months
    
    typical_df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    typical_df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    typical_df['dow_sin'] = np.sin(2 * np.pi * weekdays / 7)
    typical_df['dow_cos'] = np.cos(2 * np.pi * weekdays / 7)
    typical_df['month_sin'] = np.sin(2 * np.pi * (months - 1) / 12)
    typical_df['month_cos'] = np.cos(2 * np.pi * (months - 1) / 12)
    
    typical_df['is_operating_hour'] = ((hours >= 5) & (hours < 23)).astype(np.int8)
    typical_df['is_morning_rush'] = ((hours >= 7) & (hours <= 9)).astype(np.int8)
    typical_df['is_evening_rush'] = ((hours >= 17) & (hours <= 19)).astype(np.int8)
    
    typical_df['is_holiday'] = np.array([is_holiday(d) for d in typical_df.index], dtype=np.int8)
    typical_df['is_christmas_season'] = np.array([is_christmas_season(d) for d in typical_df.index], dtype=np.int8)
    typical_df['is_payday'] = typical_df.index.day.isin([15, 30, 31]).astype(np.int8)
    typical_df['is_weekend'] = (weekdays >= 5).astype(np.int8)
    
    return typical_df

def build_typical_day_pattern(df, target_datetime, seq_length=24, station_name=None, direction=None):
    """
    Builds and caches typical day patterns with disk persistence
    FIXED: Sequence ends at target - 1 hour (matches training alignment)
    """
    # ========== Get P95 ==========
    from routes.api_predict import get_p95_percentile
    p95 = get_p95_percentile(station_name, direction)
    if p95 is None or p95 <= 0:
        p95 = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
    
    # ============================================================
    # BUILD ACTUAL 24-HOUR TIMELINE
    # ENDS AT target - 1 hour (matches training alignment)
    # ============================================================
    
    end_time = target_datetime.replace(
        minute=0,
        second=0,
        microsecond=0
    ) - timedelta(hours=1)
    
    start_time = end_time - timedelta(hours=seq_length - 1)
    
    index = pd.date_range(
        start=start_time,
        end=end_time,
        freq='h'
    )
    
    # ============================================================
    # Cache the FULL 24-hour profile per station/direction
    # ============================================================
    cache_key = f"{station_name}_{direction}_{end_time.strftime('%Y%m%d')}"
    
    if cache_key in _TYPICAL_PATTERN_CACHE:
        hourly_passengers = _TYPICAL_PATTERN_CACHE[cache_key].copy()
    else:
        # Build typical passenger counts for ALL 24 hours
        hourly_passengers = []
        for hour in range(24):
            hour_data = df[(df.index.hour == hour)] if df is not None else []
            
            if len(hour_data) > 0:
                values = hour_data['TotalPassenger'].dropna()
                if len(values) > 0:
                    typical_passenger = float(values.median())
                else:
                    typical_passenger = get_synthetic_congestion(hour) / 100.0 * p95
            else:
                typical_passenger = get_synthetic_congestion(hour) / 100.0 * p95
            
            if np.isnan(typical_passenger) or typical_passenger < 0:
                typical_passenger = get_synthetic_congestion(hour) / 100.0 * p95
            
            hourly_passengers.append(typical_passenger)
        
        _TYPICAL_PATTERN_CACHE[cache_key] = hourly_passengers.copy()
    
    # ============================================================
    # BUILD SEQUENCE FOR THIS SPECIFIC TARGET HOUR
    # ============================================================
    
    typical_passengers = []
    for timestamp in index:
        hour = timestamp.hour
        dow = timestamp.weekday()
        
        # Get data for this specific hour and day of week
        hour_data = df[
            (df.index.hour == hour) &
            (df.index.weekday == dow)
        ]
        
        if len(hour_data) > 0:
            values = hour_data['TotalPassenger'].dropna()
            if len(values) > 0:
                passenger = float(values.median())
            else:
                passenger = get_synthetic_congestion(hour) / 100.0 * p95
        else:
            # Fallback: same hour regardless of weekday
            fallback = df[df.index.hour == hour]['TotalPassenger'].dropna()
            if len(fallback) > 0:
                passenger = float(fallback.median())
            else:
                passenger = get_synthetic_congestion(hour) / 100.0 * p95
        
        if np.isnan(passenger) or passenger < 0:
            passenger = get_synthetic_congestion(hour) / 100.0 * p95
        
        typical_passengers.append(passenger)
    
    # ========== Create DataFrame ==========
    typical_df = pd.DataFrame(index=index)
    typical_df['TotalPassenger'] = typical_passengers
    
    # ========== Calculate congestion using P95 ==========
    typical_df['congestion'] = (typical_df['TotalPassenger'] / p95 * 100).clip(0, 100)
    typical_df['congestion_percentage'] = typical_df['congestion']
    typical_df['raw_passengers'] = typical_df['TotalPassenger']
    
    # ========== Manually calculate ALL features ==========
    hours = typical_df.index.hour
    weekdays = typical_df.index.weekday
    months = typical_df.index.month
    minutes = typical_df.index.minute
    
    typical_df['hour'] = hours
    typical_df['weekday'] = weekdays
    typical_df['month'] = months
    
    typical_df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    typical_df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    typical_df['dow_sin'] = np.sin(2 * np.pi * weekdays / 7)
    typical_df['dow_cos'] = np.cos(2 * np.pi * weekdays / 7)
    typical_df['month_sin'] = np.sin(2 * np.pi * (months - 1) / 12)
    typical_df['month_cos'] = np.cos(2 * np.pi * (months - 1) / 12)
    
    time_decimal = hours + minutes / 60
    typical_df['time_decimal'] = time_decimal
    typical_df['is_operating_hour'] = ((time_decimal >= 4.5) & (time_decimal < 23.0)).astype(np.int8)
    typical_df['is_morning_rush'] = ((time_decimal >= 7.0) & (time_decimal <= 9.0)).astype(np.int8)
    typical_df['is_evening_rush'] = ((time_decimal >= 17.0) & (time_decimal <= 19.0)).astype(np.int8)
    typical_df['minute_normalized'] = minutes / 60.0
    
    typical_df['is_holiday'] = np.array([is_holiday(d) for d in typical_df.index], dtype=np.int8)
    typical_df['is_christmas_season'] = np.array([is_christmas_season(d) for d in typical_df.index], dtype=np.int8)
    typical_df['is_payday'] = typical_df.index.day.isin([15, 30, 31]).astype(np.int8)
    typical_df['is_weekend'] = (weekdays >= 5).astype(np.int8)
    typical_df['is_friday'] = (weekdays == 4).astype(np.int8)
    
    typical_df['is_rush_hour'] = (
        ((hours >= 7) & (hours <= 9)) |
        ((hours >= 17) & (hours <= 19))
    ).astype(np.int8)
    
    typical_df['is_special_event'] = 0
    typical_df['is_maintenance_record'] = 0
    typical_df['is_extended_hours'] = 0
    
    return typical_df

# ============================================================
# SCALED FEATURE SEQUENCE (CACHED)
# ============================================================
def get_scaled_feature_sequence(station_name, direction, target_datetime, seq_length=24):
    """Returns already-scaled feature sequence - ZERO computation during requests!"""
    
    # Round to hour for caching
    hour_key = target_datetime.replace(minute=0, second=0, microsecond=0)
    dow = target_datetime.weekday()
    cache_key = f"{station_name}_{direction}_{dow}_{hour_key.strftime('%Y%m%d%H')}_{seq_length}"
    
    if cache_key in _SCALED_SEQUENCE_CACHE:
        print(f"⚡ Cache hit for {station_name}_{direction} at {hour_key}")
        return _SCALED_SEQUENCE_CACHE[cache_key].copy()
    
    # Build the sequence using fast pattern
    typical_df = build_typical_day_pattern_fast(station_name, direction, target_datetime, seq_length)
    
    if typical_df is None:
        return get_baseline_features(target_datetime, seq_length)
    
    # Extract features
    feature_values = typical_df[FEATURE_COLS].values.astype(np.float32)
    
    # Scale using the feature scaler (once, at build time)
    feature_scaler = get_feature_scaler(station_name, direction)
    if feature_scaler is not None:
        try:
            scaled = feature_scaler.transform(feature_values).astype(np.float32)
        except Exception as e:
            print(f"⚠️ Scaling failed: {e}")
            scaled = feature_values
    else:
        scaled = feature_values
    
    _SCALED_SEQUENCE_CACHE[cache_key] = scaled.copy()
    return scaled

# ============================================================
# FEATURE SEQUENCE FOR STATION (HISTORICAL + SCALED)
# ============================================================
def get_feature_sequence_for_station(station_name, direction, target_datetime, seq_length=24):
    """Retrieve feature sequence for the target time - uses actual data when available."""
    
    df = get_station_dataframe_cached(station_name, direction)
    
    if df is None or len(df) == 0:
        return get_baseline_features(target_datetime, seq_length)
    
    latest_date = df.index.max()
    
    if target_datetime > latest_date:
        print(f"📊 Future date {target_datetime} - using typical pattern")
        # Use fast typical pattern for future dates (already scaled and cached)
        return get_scaled_feature_sequence(station_name, direction, target_datetime, seq_length)
    else:
        # Historical dates - use actual data
        mask = df.index < target_datetime
        if mask.sum() >= seq_length:
            lookback_df = df[mask].tail(seq_length).copy()
        else:
            print(f"📊 Not enough data for {target_datetime}, using typical pattern")
            return get_scaled_feature_sequence(station_name, direction, target_datetime, seq_length)
    
    # ========== Ensure all FEATURE_COLS exist ==========
    for col in FEATURE_COLS:
        if col not in lookback_df.columns:
            if col == 'is_holiday':
                lookback_df[col] = 0
            elif col == 'is_christmas_season':
                lookback_df[col] = np.array([is_christmas_season(d) for d in lookback_df.index], dtype=np.int8)
            elif col == 'is_payday':
                lookback_df[col] = lookback_df.index.day.isin([15, 30, 31]).astype(np.int8)
            elif col == 'TotalPassenger' and col not in lookback_df.columns:
                if 'congestion' in lookback_df.columns:
                    capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
                    lookback_df[col] = (lookback_df['congestion'] / 100.0 * capacity)
                else:
                    lookback_df[col] = 0
    
    # ========== Extract features ==========
    try:
        feature_values = lookback_df[FEATURE_COLS].values.astype(np.float32)
    except KeyError as e:
        print(f"⚠️ Missing columns in lookback_df: {e}")
        return get_baseline_features(target_datetime, seq_length)
    
    # ========== Apply scaling ==========
    feature_scaler = get_feature_scaler(station_name, direction)
    if feature_scaler is not None:
        try:
            feature_values_scaled = feature_scaler.transform(feature_values)
            print(f"✅ Applied feature scaling for {station_name}_{direction}")
            return feature_values_scaled.astype(np.float32)
        except Exception as e:
            print(f"⚠️ Feature scaling failed: {e}")
            return feature_values
    else:
        print(f"⚠️ No feature scaler found for {station_name}_{direction}")
        return feature_values

# ============================================================
# BASELINE FEATURES (FALLBACK)
# ============================================================
def get_baseline_features(target_datetime, seq_length=24):
    """Baseline features with CORRECT column mapping and holiday awareness."""
    cache_key = f"{target_datetime.strftime('%Y%m%d')}_{target_datetime.weekday()}_{seq_length}"
    
    if cache_key not in _BASELINE_FEATURES_CACHE:
        default_features = np.zeros((seq_length, len(FEATURE_COLS)), dtype=np.float32)
        
        for i in range(seq_length):
            loop_time = target_datetime - timedelta(hours=(seq_length - i))
            h_val = loop_time.hour
            dow = loop_time.weekday()
            month = loop_time.month
            
            # Estimate congestion
            if dow >= 5:  # Weekend
                if 10 <= h_val <= 16:
                    congestion = 25
                elif 17 <= h_val <= 19:
                    congestion = 30
                else:
                    congestion = 10
            else:  # Weekday
                if 17 <= h_val <= 19:
                    congestion = 50
                elif 7 <= h_val <= 9:
                    congestion = 20
                else:
                    congestion = 10
            
            # Fill all features (order must match FEATURE_COLS)
            default_features[i, 0] = congestion * 10  # TotalPassenger estimate
            default_features[i, 1] = h_val
            default_features[i, 2] = dow
            default_features[i, 3] = month
            default_features[i, 4] = np.sin(2 * np.pi * h_val / 24)
            default_features[i, 5] = np.cos(2 * np.pi * h_val / 24)
            default_features[i, 6] = np.sin(2 * np.pi * dow / 7)
            default_features[i, 7] = np.cos(2 * np.pi * dow / 7)
            default_features[i, 8] = np.sin(2 * np.pi * (month - 1) / 12)
            default_features[i, 9] = np.cos(2 * np.pi * (month - 1) / 12)
            
            time_decimal = h_val
            default_features[i, 10] = 1 if (4.5 <= time_decimal < 23.0) else 0
            default_features[i, 11] = 1 if (7.0 <= time_decimal <= 9.0) else 0
            default_features[i, 12] = 1 if (17.0 <= time_decimal <= 19.0) else 0
            
            default_features[i, 13] = 1 if is_holiday(loop_time) else 0
            default_features[i, 14] = 1 if is_christmas_season(loop_time) else 0
            default_features[i, 15] = 1 if loop_time.day in [15, 30, 31] else 0
            
        _BASELINE_FEATURES_CACHE[cache_key] = default_features
        
    return _BASELINE_FEATURES_CACHE[cache_key].copy()

# ============================================================
# PREDICTION HELPER (USES TARGET SCALER)
# ============================================================
def predict_congestion_with_model(station_name, direction, feature_sequence, model):
    """
    Helper function to make predictions using the trained model
    Handles scaling and inverse transformation
    """
    target_scaler = get_target_scaler(station_name, direction)
    
    if target_scaler is None:
        print(f"⚠️ No target scaler found for {station_name}_{direction}")
        return None
    
    if len(feature_sequence.shape) == 2:
        feature_sequence = feature_sequence.reshape(1, feature_sequence.shape[0], feature_sequence.shape[1])
    
    pred_scaled = model.predict(feature_sequence, verbose=0)
    pred_passengers = target_scaler.inverse_transform(pred_scaled)
    
    return pred_passengers

# ============================================================
# PRELOAD / WARMUP ALL PATTERNS (CALL ONCE AT STARTUP)
# ============================================================
def preload_all_station_patterns():
    """
    Preloads all 7 day-of-week patterns (NOT calendar dates!)
    This runs ONCE at startup.
    """
    print("\n🚀 Preloading all day-of-week typical patterns...")
    
    stations = list(STATION_NUMBERS.keys())
    directions = ['Northbound', 'Southbound']
    
    # Preload P95 values (populates _P95_LOCAL_CACHE)
    for station in stations:
        for direction in directions:
            from routes.api_predict import get_p95_percentile
            get_p95_percentile(station, direction)
    
    # Build ALL 7 day-of-week patterns
    for station in stations:
        for direction in directions:
            for dow in range(7):
                days_ahead = (dow - datetime.now().weekday()) % 7
                sample_dt = datetime.now() + timedelta(days=days_ahead)
                sample_dt = sample_dt.replace(hour=12, minute=0, second=0, microsecond=0)
                build_typical_day_pattern_fast(station, direction, sample_dt)
    
    print(f"✅ Preloaded {len(_TYPICAL_PROFILE_CACHE)} day-of-week patterns")
    print(f"   Expected: {len(stations)} × 2 directions × 7 days = 182")

# ============================================================
# TIME & FEATURE UTILITIES
# ============================================================
def add_cyclical_time_features(df):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    hours = df.index.hour
    weekdays = df.index.weekday
    months = df.index.month
    minutes = df.index.minute
    
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    df['dow_sin'] = np.sin(2 * np.pi * weekdays / 7)
    df['dow_cos'] = np.cos(2 * np.pi * weekdays / 7)
    df['month_sin'] = np.sin(2 * np.pi * (months - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (months - 1) / 12)
    df['time_decimal'] = hours + minutes / 60
    df['is_operating_hour'] = ((df['time_decimal'] >= 4.5) & (df['time_decimal'] < 23.0)).astype(np.int8)
    df['minute_normalized'] = minutes / 60.0
    return df

def add_smart_operating_flags(df):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    time_decimal = df.index.hour + df.index.minute / 60
    df['is_morning_rush'] = ((time_decimal >= 7.0) & (time_decimal <= 9.0)).astype(np.int8)
    df['is_evening_rush'] = ((time_decimal >= 17.0) & (time_decimal <= 19.0)).astype(np.int8)
    return df

def smart_data_cleaner(df):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    time_decimal = df.index.hour + df.index.minute / 60
    passenger_count = df.get('TotalPassenger', pd.Series(0, index=df.index))
    df['is_maintenance_record'] = ((time_decimal < 5.0) & (passenger_count < 10)).astype(np.int8)
    df['is_extended_hours'] = ((time_decimal >= 22.0) & (time_decimal < 23.0) & (passenger_count >= 10)).astype(np.int8)
    return df

# ============================================================
# LEGACY / COMPATIBILITY FUNCTIONS
# ============================================================
def load_data_fast():
    print("📊 Using memory-optimized data loading (no full data cache)")
    return None

def load_data():
    return load_data_fast()

def categorize_congestion(congestion_value, capacity=None, station_name=None):
    if capacity is not None and capacity > 0:
        percentage = (congestion_value / capacity) * 100
    elif station_name is not None:
        capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
        percentage = (congestion_value / capacity) * 100
    else:
        percentage = congestion_value
    
    if percentage < 30:
        return 0
    elif percentage < 60:
        return 1
    elif percentage < 90:
        return 2
    else:
        return 3

def infer_direction(row):
    entry = row['StationEntry']
    exit_st = row['StationExit']
    if entry < exit_st:
        return 'Southbound'
    elif entry > exit_st:
        return 'Northbound'
    else:
        return 'Unknown'

def get_congestion_category_name(category):
    names = {0: "Light", 1: "Moderate", 2: "Congested", 3: "Severely Congested"}
    return names.get(category, "Unknown")

def get_hourly_window_from_csv(station_name, direction, target_datetime, seq_length=24):
    df = get_station_dataframe_cached(station_name, direction)
    if df is None:
        return None
    return df.tail(seq_length)

print("=" * 50)
print("✅ feature_engineering.py loaded successfully!")
print("=" * 50)