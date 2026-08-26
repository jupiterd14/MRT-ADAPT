"""
Feature engineering for LSTM predictions - OPTIMIZED FOR SPEED & MEMORY
"""

import os 
import gc
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ========== GLOBAL RAM CACHES ==========
_STATION_DATA_CACHE = {}      # Holds all loaded station Parquet DataFrames in RAM
_FEATURE_SCALER_CACHE = {}    # Cache for feature scalers
_TARGET_SCALER_CACHE = {}     # Cache for target scalers
_BASELINE_FEATURES_CACHE = {} # Cache for baseline features
_TYPICAL_PATTERN_CACHE = {}   # Cache for pre-calculated typical day patterns

# Station numbers
STATION_NUMBERS = {
    "North Ave": 1, "Quezon Ave": 2, "Kamuning": 3, "Cubao": 4,
    "Santolan": 5, "Ortigas": 6, "Shaw Blvd": 7, "Boni Ave": 8,
    "Guadalupe": 9, "Buendia": 10, "Ayala Ave": 11, "Magallanes": 12,
    "Taft": 13
}


# In feature_engineering.py, update FEATURE_COLS to match training
FEATURE_COLS = [
    'TotalPassenger',
    'hour', 'weekday', 'month',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'is_operating_hour', 'is_morning_rush', 'is_evening_rush',
    'is_holiday', 'is_christmas_season', 'is_payday'
]
MRT3_PLATFORM_CAPACITY = {
    "North Ave": 1142, "Quezon Ave": 1195, "Kamuning": 1364, "Cubao": 1747,
    "Santolan": 1306, "Ortigas": 1331, "Shaw Blvd": 1619, "Boni Ave": 1417,
    "Guadalupe": 1301, "Buendia": 1645, "Ayala Ave": 1222, "Magallanes": 1202,
    "Taft": 720
}


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
    FIXED: Directional filtering matches training
    """
    print(f"📊 Loading real data for {station_name}_{direction} from CSV (streaming mode)...")
    
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
            print(f"   ⚠️ File not found: {filepath}")
            continue
            
        print(f"   📄 Processing: {csv_file}")
        try:
            needed_columns = ['StationEntry', 'StationExit', 'Date', 'Time', 'TotalPassenger']
            
            for chunk in pd.read_csv(filepath, 
                                    chunksize=CHUNK_SIZE,
                                    usecols=needed_columns,
                                    dtype={'StationEntry': 'int16', 'StationExit': 'int16', 'TotalPassenger': 'float32'}):
                
                # ========== DIRECTIONAL FILTERING - MATCHES TRAINING ==========
                if direction == 'Northbound':
                    if is_north_terminal:
                        # North Ave: passengers EXIT (end of line)
                        filtered = chunk[chunk['StationExit'] == station_num]
                    elif is_south_terminal:
                        # Taft: passengers ENTER to go north
                        filtered = chunk[chunk['StationEntry'] == station_num]
                    else:
                        # All other stations: passengers EXIT to go north
                        filtered = chunk[chunk['StationExit'] == station_num]  # ← FIXED
                else:  # Southbound
                    if is_north_terminal:
                        # North Ave: passengers ENTER to go south
                        filtered = chunk[chunk['StationEntry'] == station_num]
                    elif is_south_terminal:
                        # Taft: passengers EXIT (end of line)
                        filtered = chunk[chunk['StationExit'] == station_num]
                    else:
                        # All other stations: passengers ENTER to go south
                        filtered = chunk[chunk['StationEntry'] == station_num]  # ← CORRECT
                
                if len(filtered) > 0:
                    all_filtered.append(filtered)
                
                del chunk, filtered
                gc.collect()
                
        except Exception as e:
            print(f"   ⚠️ Error processing {csv_file}: {e}")
            continue
    
    if not all_filtered:
        print(f"⚠️ No real data found for {station_name}_{direction}")
        return None
    
    print(f"   🔄 Combining {len(all_filtered)} chunks...")
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
    combined['is_holiday'] = np.array([is_holiday(d) for d in combined.index], dtype=np.int8)  # ← FIXED
    combined['is_special_event'] = 0
    
    return combined
#==========================================
# TYPICAL PATTERN BUILDING WITH DISK PERSISTENCE
# ============================================================
def build_typical_day_pattern(df, target_datetime, seq_length=24, station_name=None, direction=None):
    """
    Builds and caches typical day patterns with disk persistence
    FIXED: Uses P95 for congestion calculation, not capacity
    """
    target_dow = target_datetime.weekday()
    cache_key = f"{station_name}_{direction}_dow_{target_dow}"
    
    # Check in-memory cache first
    if cache_key in _TYPICAL_PATTERN_CACHE:
        return _TYPICAL_PATTERN_CACHE[cache_key].copy()
    
    # ========== Get P95 for congestion denominator ==========
    # Import here to avoid circular imports
    try:
        from routes.api_predict import get_p95_percentile
        p95 = get_p95_percentile(station_name, direction)
    except:
        # Fallback to capacity if P95 not available
        p95 = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
    
    if p95 is None or p95 <= 0:
        p95 = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
    
    # ========== Build typical passenger counts directly ==========
    hourly_passengers = []
    
    for hour in range(24):
        hour_data = df[(df.index.hour == hour) & (df.index.weekday == target_dow)] if df is not None else []
        
        if len(hour_data) > 0:
            # Get TotalPassenger values directly
            values = hour_data['TotalPassenger'].dropna()
            
            if len(values) > 0:
                typical_passenger = float(values.median())
            else:
                # No valid data - use synthetic fallback
                typical_passenger = get_synthetic_congestion(hour) / 100.0 * p95
        else:
            # No data - use synthetic fallback
            typical_passenger = get_synthetic_congestion(hour) / 100.0 * p95
        
        # Ensure we never have NaN or negative
        if np.isnan(typical_passenger) or typical_passenger < 0:
            typical_passenger = get_synthetic_congestion(hour) / 100.0 * p95
        
        hourly_passengers.append(typical_passenger)
    
    # ========== Build the 24-hour sequence ==========
    target_hour = target_datetime.hour
    
    typical_data = []
    for i in range(seq_length):
        hour = (target_hour - seq_length + 1 + i) % 24
        typical_data.append(hourly_passengers[hour])
    
    # ========== Create timestamps ==========
    base_date = target_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    start_hour = (target_hour - seq_length + 1) % 24
    
    index = []
    current_time = base_date + timedelta(hours=start_hour)
    for i in range(seq_length):
        index.append(current_time)
        current_time += timedelta(hours=1)
    
    # ========== Create DataFrame ==========
    typical_df = pd.DataFrame(index=index)
    
    # Add TotalPassenger (directly from typical passenger counts)
    typical_df['TotalPassenger'] = typical_data
    
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
    typical_df['is_rush_hour'] = ((hours >= 7) & (hours <= 9)) | ((hours >= 17) & (hours <= 19)).astype(np.int8)
    typical_df['is_special_event'] = 0
    
    typical_df['is_maintenance_record'] = 0
    typical_df['is_extended_hours'] = 0
    
    # Store in memory cache
    _TYPICAL_PATTERN_CACHE[cache_key] = typical_df.copy()
    
    return typical_df
def get_feature_sequence_for_station(station_name, direction, target_datetime, seq_length=24):
    """Retrieve feature sequence for the target time"""
    
    df = get_station_dataframe_cached(station_name, direction)
    
    if df is None or len(df) == 0:
        return get_baseline_features(target_datetime, seq_length)
    
    latest_date = df.index.max()
    
    if target_datetime > latest_date:
        print(f"📊 Future date {target_datetime} - using typical pattern + recent trend")
        
        recent_df = df.tail(168)
        typical_df = build_typical_day_pattern(df, target_datetime, seq_length, station_name, direction)
        
        if typical_df is not None:
            # ========== ✅ FIX: USE THE TYPICAL PATTERN AS-IS ==========
            lookback_df = typical_df.copy()
            
            # Apply trend adjustment if needed
            target_hour = target_datetime.hour
            recent_avg = recent_df[recent_df.index.hour == target_hour]['TotalPassenger'].mean() if len(recent_df) > 0 else None
            typical_value = lookback_df[lookback_df.index.hour == target_hour]['TotalPassenger'].iloc[0] if len(lookback_df) > 0 else None
            
            if recent_avg is not None and typical_value is not None and typical_value > 0:
                trend_factor = recent_avg / typical_value
                trend_factor = max(0.5, min(2.0, trend_factor))
                lookback_df['TotalPassenger'] = lookback_df['TotalPassenger'] * trend_factor
                print(f"📊 Applied trend factor: {trend_factor:.2f}")
            
            # ========== ✅ FIX: DON'T RECALCULATE FEATURES ==========
            # The typical_df already has ALL features correctly calculated
            # Just update the passenger-dependent columns
            capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
            lookback_df['congestion'] = (lookback_df['TotalPassenger'] / capacity * 100).clip(0, 100)
            lookback_df['congestion_percentage'] = lookback_df['congestion']
            lookback_df['raw_passengers'] = lookback_df['TotalPassenger']
            
            print(f"📊 Using typical pattern for {target_datetime} (Day {target_datetime.weekday()})")
        else:
            return get_baseline_features(target_datetime, seq_length)
    else:
        # Historical dates - use actual data
        mask = df.index < target_datetime
        if mask.sum() >= seq_length:
            lookback_df = df[mask].tail(seq_length).copy()
        else:
            print(f"📊 Not enough data for {target_datetime}, using typical pattern")
            lookback_df = build_typical_day_pattern(df, target_datetime, seq_length, station_name, direction)
    
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
def get_baseline_features(target_datetime, seq_length=24):
    """Baseline features with CORRECT column mapping"""
    cache_key = f"{target_datetime.strftime('%Y%m%d')}_{target_datetime.weekday()}_{seq_length}"
    
    if cache_key not in _BASELINE_FEATURES_CACHE:
        # Initialize with zeros - shape (seq_length, len(FEATURE_COLS))
        default_features = np.zeros((seq_length, len(FEATURE_COLS)), dtype=np.float32)
        
        for i in range(seq_length):
            loop_time = target_datetime - timedelta(hours=(seq_length - i))
            h_val = loop_time.hour
            dow = loop_time.weekday()
            month = loop_time.month
            
            # ========== ✅ FIX: Correct column mapping ==========
            # FEATURE_COLS order:
            # 0: TotalPassenger, 1: hour, 2: weekday, 3: month,
            # 4: hour_sin, 5: hour_cos, 6: dow_sin, 7: dow_cos,
            # 8: month_sin, 9: month_cos,
            # 10: is_operating_hour, 11: is_morning_rush, 12: is_evening_rush,
            # 13: is_holiday, 14: is_christmas_season, 15: is_payday
            
            # Calculate congestion-based passenger estimate
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
            
            # Fill in all features
            default_features[i, 0] = congestion * 10  # TotalPassenger estimate
            default_features[i, 1] = h_val  # hour
            default_features[i, 2] = dow  # weekday
            default_features[i, 3] = month  # month
            default_features[i, 4] = np.sin(2 * np.pi * h_val / 24)  # hour_sin
            default_features[i, 5] = np.cos(2 * np.pi * h_val / 24)  # hour_cos
            default_features[i, 6] = np.sin(2 * np.pi * dow / 7)  # dow_sin
            default_features[i, 7] = np.cos(2 * np.pi * dow / 7)  # dow_cos
            default_features[i, 8] = np.sin(2 * np.pi * (month - 1) / 12)  # month_sin
            default_features[i, 9] = np.cos(2 * np.pi * (month - 1) / 12)  # month_cos
            
            # Operating hours (4.5 to 23.0)
            time_decimal = h_val
            default_features[i, 10] = 1 if (4.5 <= time_decimal < 23.0) else 0  # is_operating_hour
            default_features[i, 11] = 1 if (7.0 <= time_decimal <= 9.0) else 0  # is_morning_rush
            default_features[i, 12] = 1 if (17.0 <= time_decimal <= 19.0) else 0  # is_evening_rush
            
            default_features[i, 13] = 0  # is_holiday (default)
            default_features[i, 14] = 1 if is_christmas_season(loop_time) else 0  # is_christmas_season
            default_features[i, 15] = 1 if loop_time.day in [15, 30, 31] else 0  # is_payday
            
        _BASELINE_FEATURES_CACHE[cache_key] = default_features
        
    return _BASELINE_FEATURES_CACHE[cache_key].copy()

def predict_congestion_with_model(station_name, direction, feature_sequence, model):
    """
    Helper function to make predictions using the trained model
    Handles scaling and inverse transformation
    """
    # Get target scaler
    target_scaler = get_target_scaler(station_name, direction)
    
    if target_scaler is None:
        print(f"⚠️ No target scaler found for {station_name}_{direction}")
        return None
    
    # Ensure feature_sequence has the right shape
    if len(feature_sequence.shape) == 2:
        # Add batch dimension if needed
        feature_sequence = feature_sequence.reshape(1, feature_sequence.shape[0], feature_sequence.shape[1])
    
    # Make prediction
    pred_scaled = model.predict(feature_sequence, verbose=0)
    
    # Inverse transform to get actual passenger counts
    pred_passengers = target_scaler.inverse_transform(pred_scaled)
    
    return pred_passengers

# ============================================================
# PRELOAD / WARMUP ALL PATTERNS (CALL ONCE AT STARTUP)
# ============================================================

def preload_all_station_patterns():
    """
    Preloads all 26 Parquet files and computes patterns for all Days of Week (0-6)
    so ZERO file access or calculation logs occur during runtime.
    """
    print("\n🚀 Preloading all Parquet datasets & typical day patterns into RAM...")
    now = datetime.now()
    
    stations = list(STATION_NUMBERS.keys())
    total_stations = len(stations)
    total_directions = 2
    total_dows = 7
    
    print(f"   📊 Loading {total_stations} stations × {total_directions} directions × {total_dows} day patterns...")
    
    for station in stations:
        for direction in ['Northbound', 'Southbound']:
            # Load the dataframe first
            df = get_station_dataframe_cached(station, direction)
            if df is not None:
                # Pre-build all 7 day-of-week patterns
                for dow in range(7):
                    # Create a sample datetime for this day of week
                    # Find the next occurrence of this day of week
                    days_ahead = (dow - now.weekday()) % 7
                    sample_dt = now + timedelta(days=days_ahead)
                    build_typical_day_pattern(df, sample_dt, 24, station, direction)
            
            # Force garbage collection periodically
            gc.collect()
    
    print(f"✅ All station patterns preloaded into RAM successfully!")
    print(f"   📦 {len(_STATION_DATA_CACHE)} station DataFrames cached")
    print(f"   📊 {len(_TYPICAL_PATTERN_CACHE)} typical patterns cached")
    print("   💡 Zero disk reads will occur during live map requests!\n")

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

def is_christmas_season(date):
    month_day = date.strftime('%m-%d')
    return (month_day >= '12-15') or (month_day <= '01-05')

# ============================================================
# LEGACY / COMPATIBILITY FUNCTIONS
# ============================================================

def load_data_fast():
    """Legacy function - now uses on-demand loading"""
    print("📊 Using memory-optimized data loading (no full data cache)")
    return None

def load_data():
    """Legacy function - now uses fast loading"""
    return load_data_fast()

def categorize_congestion(congestion_value, capacity=None, station_name=None):
    """Categorize congestion into 4 levels"""
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

# Add this function to services/feature_engineering.py
# Place it near the end of the file, before the print statement

def infer_direction(row):
    """Infer direction from station entry/exit numbers"""
    entry = row['StationEntry']
    exit_st = row['StationExit']
    if entry < exit_st:
        return 'Southbound'
    elif entry > exit_st:
        return 'Northbound'
    else:
        return 'Unknown'
    
def is_payday(date):
    """Check if a date is a payday (15th, 30th, or 31st)"""
    return date.day in [15, 30, 31]

def is_friday(date):
    """Check if a date is a Friday"""
    return date.weekday() == 4

def get_congestion_category_name(category):
    names = {0: "Light", 1: "Moderate", 2: "Congested", 3: "Severely Congested"}
    return names.get(category, "Unknown")

def get_hourly_window_from_csv(station_name, direction, target_datetime, seq_length=24):
    """Legacy function - now uses get_station_dataframe_cached"""
    df = get_station_dataframe_cached(station_name, direction)
    if df is None:
        return None
    return df.tail(seq_length)

print("=" * 50)
print("✅ feature_engineering.py loaded successfully!")
print("=" * 50)