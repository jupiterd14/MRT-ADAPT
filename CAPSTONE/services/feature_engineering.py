"""
Feature engineering for LSTM predictions - OPTIMIZED FOR SPEED
"""

# Add memory monitoring
import psutil
import tracemalloc

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def log_memory_usage(stage=""):
    """Log memory usage at a specific stage"""
    mem = get_memory_usage()
    print(f"   📊 Memory ({stage}): {mem:.1f} MB")
    

import os 
import gc
import pandas as pd
import numpy as np

import pickle
from datetime import datetime, timedelta
from urllib.parse import unquote


from config import Config

# ========== GLOBAL CACHE ==========
_DATA_CACHE = None
_PER_DIRECTION_MAX = None
_STATION_DATA_CACHE = {}  # Cache for station-specific dataframes
_FEATURE_SCALER_CACHE = {}  # Cache for scalers
_BASELINE_FEATURES_CACHE = {}  # Cache for baseline features

# Station numbers (matching training)
STATION_NUMBERS = {
    "North Ave": 1, "Quezon Ave": 2, "Kamuning": 3, "Cubao": 4,
    "Santolan": 5, "Ortigas": 6, "Shaw Blvd": 7, "Boni Ave": 8,
    "Guadalupe": 9, "Buendia": 10, "Ayala Ave": 11, "Magallanes": 12,
    "Taft": 13
}

FEATURE_COLS = [
    'hour', 'weekday', 'month',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'is_operating_hour', 'is_morning_rush', 'is_evening_rush', 'is_noon',
    'is_pre_opening', 'is_post_closing',
    'minutes_until_closing', 'minutes_since_opening', 'time_normalized', 'minute_normalized',
    'is_weekend', 'is_holiday', 'is_special_event', 'is_christmas_season', 'is_payday', 'is_friday',
    'is_rush_hour', 'is_maintenance_record', 'is_extended_hours', 'congestion'
]

MRT3_PLATFORM_CAPACITY = {
    "North Ave": 1142, "Quezon Ave": 1195, "Kamuning": 1364, "Cubao": 1747,
    "Santolan": 1306, "Ortigas": 1331, "Shaw Blvd": 1619, "Boni Ave": 1417,
    "Guadalupe": 1301, "Buendia": 1645, "Ayala Ave": 1222, "Magallanes": 1202,
    "Taft": 720
}

import joblib
def process_raw_data():
    """Legacy function - no longer used"""
    print("⚠️ process_raw_data() is deprecated - using on-demand loading")
    return None

def load_data_fast():
    """
    Memory-optimized data loading - ONLY loads data when needed,
    and doesn't keep ALL data in memory.
    """
    global _DATA_CACHE
    
    # ❌ REMOVE the global cache - it's causing memory issues!
    # We'll load data on-demand instead
    
    # Read only the needed data for a specific station
    # This is handled by get_station_dataframe now
    
    print("📊 Using memory-optimized data loading (no full data cache)")
    
    # Return None - we'll load data on demand in get_station_dataframe
    return None

def get_hourly_window_from_csv(station_name, direction, target_datetime, seq_length=24):
    """
    Directly read only the needed 24-hour window from CSV files.
    Returns a DataFrame with hourly data AND all feature columns.
    """
    station_num = STATION_NUMBERS.get(station_name)
    if not station_num:
        return None
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data (2022-2024)')
    
    end_time = target_datetime
    start_time = target_datetime - timedelta(hours=seq_length)
    
    hourly_frames = []
    CHUNK_SIZE = 50000
    
    csv_files = ['2022.csv', '2023.csv', '2024.csv']
    
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
                
                if direction == 'Northbound':
                    mask = chunk['StationExit'] == station_num
                else:
                    mask = chunk['StationEntry'] == station_num
                
                filtered = chunk[mask]
                if len(filtered) == 0:
                    del chunk, filtered
                    gc.collect()
                    continue
                
                filtered['datetime'] = pd.to_datetime(filtered['Date'] + ' ' + filtered['Time'])
                filtered = filtered.set_index('datetime')
                filtered = filtered[['TotalPassenger']]
                
                window_data = filtered[(filtered.index >= start_time) & (filtered.index < end_time)]
                if len(window_data) > 0:
                    hourly = window_data.resample('H').sum()
                    hourly_frames.append(hourly)
                
                del chunk, filtered, window_data
                gc.collect()
                
        except Exception as e:
            print(f"⚠️ Error reading {csv_file}: {e}")
            continue
    
    if not hourly_frames:
        return None
    
    combined = pd.concat(hourly_frames)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    
    # ========== ADD FEATURE ENGINEERING ==========
    capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
    
    # Calculate congestion
    feature_scaler = get_feature_scaler(station_name, direction)
    if feature_scaler is not None:
        training_max = feature_scaler.data_max_[-1] if hasattr(feature_scaler, 'data_max_') else 50
    else:
        training_max = min(capacity * 0.3, 50)
    
    percentage = (combined['TotalPassenger'] / capacity * 100).clip(0, 100)
    combined['congestion'] = (percentage / 100) * training_max
    combined['congestion_percentage'] = percentage
    combined['raw_passengers'] = combined['TotalPassenger']
    
    # Add time features
    combined = add_cyclical_time_features(combined)
    combined = add_smart_operating_flags(combined)
    combined = smart_data_cleaner(combined)
    
    # Add date-based features
    combined['is_weekend'] = (combined.index.weekday >= 5).astype(np.int8)
    combined['is_christmas_season'] = np.array([is_christmas_season(d) for d in combined.index], dtype=np.int8)
    combined['is_payday'] = combined.index.day.isin([15, 30, 31]).astype(np.int8)
    combined['is_friday'] = (combined.index.weekday == 4).astype(np.int8)
    combined['is_rush_hour'] = ((combined['hour'].between(7, 9)) | (combined['hour'].between(17, 19))).astype(np.int8)
    combined['is_holiday'] = 0
    combined['is_special_event'] = 0
    
    print(f"   ✅ Window has {len(combined)} hours with features")
    return combined

def categorize_congestion(congestion_value, capacity=None, station_name=None):
    """
    Categorize congestion into 4 levels based on passenger count or percentage.
    
    Args:
        congestion_value: Raw passenger count or percentage
        capacity: Station capacity (if using percentage)
        station_name: Station name to get capacity from
        
    Returns:
        int: Category 0-3
            0 = Light (0-30% or 0-30% of capacity)
            1 = Moderate (30-60% of capacity)
            2 = Congested (60-90% of capacity)
            3 = Severely Congested (90-100%+ of capacity)
    """
    # If capacity is provided, convert to percentage
    if capacity is not None and capacity > 0:
        percentage = (congestion_value / capacity) * 100
    elif station_name is not None:
        capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
        percentage = (congestion_value / capacity) * 100
    else:
        # Assume congestion_value is already a percentage (0-100)
        percentage = congestion_value
    
    # Categorize based on percentage
    if percentage < 30:
        return 0  # Light
    elif percentage < 60:
        return 1  # Moderate
    elif percentage < 90:
        return 2  # Congested
    else:
        return 3  # Severely Congested

def get_congestion_category_name(category):
    """Get the human-readable name for a congestion category"""
    names = {
        0: "Light",
        1: "Moderate", 
        2: "Congested",
        3: "Severely Congested"
    }
    return names.get(category, "Unknown")


def get_feature_scaler(station_name, direction):
    """Get the feature scaler for a station-direction pair"""
    cache_key = f"{station_name}_{direction}"
    
    # Check cache first
    if cache_key in _FEATURE_SCALER_CACHE:
        return _FEATURE_SCALER_CACHE[cache_key]
    
    # Try to load from disk
    model_folder = 'models_2022-2024_v8'
    scaler_path = f'{model_folder}/{cache_key}_feature_scaler.pkl'
    
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            _FEATURE_SCALER_CACHE[cache_key] = scaler
            return scaler
        except Exception as e:
            print(f"   ⚠️ Error loading feature scaler: {e}")
            return None
    else:
        print(f"   ⚠️ No feature scaler found at: {scaler_path}")
        return None
def get_station_dataframe(station_name, direction):
    """
    Memory-optimized - streams CSV files and only keeps the needed data
    """
    global _STATION_DATA_CACHE
    
    cache_key = f"{station_name}_{direction}"
    
    # Return from cache if available
    if cache_key in _STATION_DATA_CACHE:
        print(f"📦 Using cached data for {cache_key}")
        return _STATION_DATA_CACHE[cache_key]
    
    print(f"📊 Loading real data for {cache_key} from CSV (streaming mode)...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data (2022-2024)')
    
    csv_files = ['2022.csv', '2023.csv', '2024.csv']
    station_num = STATION_NUMBERS.get(station_name)
    
    if not station_num:
        print(f"❌ Unknown station: {station_name}")
        return None
    
    all_hourly_data = []
    CHUNK_SIZE = 20000  # Balanced chunk size
    
    for csv_file in csv_files:
        filepath = os.path.join(data_dir, csv_file)
        if not os.path.exists(filepath):
            print(f"   ⚠️ File not found: {filepath}")
            continue
            
        print(f"   📄 Processing: {csv_file}")
        try:
            # Read only needed columns
            needed_columns = ['StationEntry', 'StationExit', 'Date', 'Time', 'TotalPassenger']
            
            # Process in chunks
            for chunk in pd.read_csv(filepath, 
                                    chunksize=CHUNK_SIZE,
                                    usecols=needed_columns,
                                    dtype={'StationEntry': 'int16', 'StationExit': 'int16', 'TotalPassenger': 'float32'}):
                
                # Filter for this station
                if direction == 'Northbound':
                    filtered = chunk[chunk['StationExit'] == station_num]
                else:
                    filtered = chunk[chunk['StationEntry'] == station_num]
                
                if len(filtered) > 0:
                    # Convert to datetime and resample immediately
                    filtered['datetime'] = pd.to_datetime(filtered['Date'] + ' ' + filtered['Time'])
                    filtered = filtered.set_index('datetime')
                    filtered = filtered[['TotalPassenger']]
                    
                    # Resample to hourly (aggregates and reduces size)
                    hourly = filtered.resample('H').sum()
                    all_hourly_data.append(hourly)
                
                # Free memory
                del chunk, filtered
                gc.collect()
                
        except Exception as e:
            print(f"   ⚠️ Error processing {csv_file}: {e}")
            continue
    
    if not all_hourly_data:
        print(f"⚠️ No real data found for {cache_key}")
        return None
    
    # Combine all hourly data
    print(f"   🔄 Combining {len(all_hourly_data)} chunks...")
    combined = pd.concat(all_hourly_data)
    del all_hourly_data
    gc.collect()
    
    # Sort by datetime
    combined = combined.sort_index()
    
    # Remove duplicates (if any)
    combined = combined[~combined.index.duplicated(keep='first')]
    
    print(f"   ✅ Loaded {len(combined)} hours of real data")
    
    # ========== CALCULATE CONGESTION ==========
    capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
    
    feature_scaler = get_feature_scaler(station_name, direction)
    if feature_scaler is not None:
        training_max = feature_scaler.data_max_[-1] if hasattr(feature_scaler, 'data_max_') else 50
    else:
        training_max = min(capacity * 0.3, 50)
    
    percentage = (combined['TotalPassenger'] / capacity * 100).clip(0, 100)
    combined['congestion'] = (percentage / 100) * training_max
    combined['congestion_percentage'] = percentage
    combined['raw_passengers'] = combined['TotalPassenger']
    
    # Add time features
    combined = add_cyclical_time_features(combined)
    combined = add_smart_operating_flags(combined)
    combined = smart_data_cleaner(combined)
    
    # Add date-based features
    combined['is_weekend'] = (combined.index.weekday >= 5).astype(np.int8)
    combined['is_christmas_season'] = np.array([is_christmas_season(d) for d in combined.index], dtype=np.int8)
    combined['is_payday'] = combined.index.day.isin([15, 30, 31]).astype(np.int8)
    combined['is_friday'] = (combined.index.weekday == 4).astype(np.int8)
    combined['is_rush_hour'] = ((combined['hour'].between(7, 9)) | (combined['hour'].between(17, 19))).astype(np.int8)
    combined['is_holiday'] = 0
    combined['is_special_event'] = 0
    
    # ⭐ Cache only the hourly data (not the full raw data)
    if len(_STATION_DATA_CACHE) >= 2:
        oldest_key = next(iter(_STATION_DATA_CACHE))
        print(f"🗑️ Removing oldest from cache: {oldest_key}")
        del _STATION_DATA_CACHE[oldest_key]
        gc.collect()
    
    _STATION_DATA_CACHE[cache_key] = combined
    print(f"✅ Cached {len(combined)} hours for {cache_key}")
    
    return combined
def get_baseline_features(target_datetime, seq_length=24):
    """Cache baseline features for repeated lookups"""
    cache_key = f"{target_datetime.strftime('%Y%m%d%H')}_{seq_length}"
    
    if cache_key not in _BASELINE_FEATURES_CACHE:
        default_features = np.zeros((seq_length, len(FEATURE_COLS)), dtype=np.float32)
        for i in range(seq_length):
            loop_time = target_datetime - timedelta(hours=(seq_length - i))
            h_val = loop_time.hour
            
            default_features[i, 0] = h_val
            default_features[i, 1] = loop_time.weekday()
            default_features[i, 2] = loop_time.month
            default_features[i, 3] = np.sin(2 * np.pi * h_val / 24)
            default_features[i, 4] = np.cos(2 * np.pi * h_val / 24)
            default_features[i, 5] = np.sin(2 * np.pi * loop_time.weekday() / 7)
            default_features[i, 6] = np.cos(2 * np.pi * loop_time.weekday() / 7)
            default_features[i, 7] = np.sin(2 * np.pi * (loop_time.month - 1) / 12)
            default_features[i, 8] = np.cos(2 * np.pi * (loop_time.month - 1) / 12)
            
            # ========== USE RAW PASSENGER COUNTS FOR BASELINE ==========
            if 7 <= h_val <= 9:
                default_features[i, -1] = 3500.0  # Morning rush - raw passengers
            elif 17 <= h_val <= 19:
                default_features[i, -1] = 4000.0  # Evening rush - raw passengers
            elif 10 <= h_val <= 16:
                default_features[i, -1] = 2000.0  # Midday - raw passengers
            elif 5 <= h_val <= 6:
                default_features[i, -1] = 800.0   # Early morning - raw passengers
            elif 20 <= h_val <= 21:
                default_features[i, -1] = 1500.0  # Late evening - raw passengers
            else:
                default_features[i, -1] = 100.0   # Very early/late - raw passengers
        
        _BASELINE_FEATURES_CACHE[cache_key] = default_features
    
    return _BASELINE_FEATURES_CACHE[cache_key].copy()

def get_station_dataframe_cached(station_name, direction):
    """
    Load full dataset once and cache as Parquet for faster future loads.
    Uses less memory than CSV and loads faster.
    """
    global _STATION_DATA_CACHE
    cache_key = f"{station_name}_{direction}"
    
    if cache_key in _STATION_DATA_CACHE:
        print(f"📦 Using memory cache for {cache_key}")
        return _STATION_DATA_CACHE[cache_key]
    
    # Check for Parquet cache
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
    os.makedirs(cache_dir, exist_ok=True)
    parquet_file = os.path.join(cache_dir, f'{cache_key}.parquet')
    
    if os.path.exists(parquet_file):
        try:
            print(f"📦 Loading cached Parquet: {parquet_file}")
            df = pd.read_parquet(parquet_file)
            _STATION_DATA_CACHE[cache_key] = df
            return df
        except Exception as e:
            print(f"⚠️ Error loading Parquet: {e}")
            # Continue to regenerate
    
    # Generate from CSV using get_station_dataframe
    print(f"🔄 Generating Parquet cache for {cache_key} from CSV...")
    df = get_station_dataframe(station_name, direction)
    
    if df is not None and len(df) > 0:
        try:
            # Save as Parquet (compressed)
            df.to_parquet(parquet_file, compression='gzip')
            print(f"✅ Saved cache to {parquet_file} ({len(df)} rows)")
            _STATION_DATA_CACHE[cache_key] = df
        except Exception as e:
            print(f"⚠️ Could not save Parquet: {e}")
            # Still keep in memory
            _STATION_DATA_CACHE[cache_key] = df
    
    return df

def get_feature_sequence_for_station(station_name, direction, target_datetime, seq_length=24):
    """Get feature sequence - uses windowed CSV loading for memory efficiency"""
    station_name = unquote(station_name)
    direction = unquote(direction)
    
    # Try to load only the needed window from CSV
    window_df = get_hourly_window_from_csv(station_name, direction, target_datetime, seq_length)
    
    if window_df is None or len(window_df) == 0:
        print(f"⚠️ No real data in window for {station_name} {direction}, using baseline")
        return get_baseline_features(target_datetime, seq_length)
    
    # Ensure we have all feature columns
    missing_cols = [col for col in FEATURE_COLS if col not in window_df.columns]
    if missing_cols:
        print(f"   ⚠️ Missing columns: {missing_cols[:5]}... using baseline")
        return get_baseline_features(target_datetime, seq_length)
    
    # If we have less than seq_length hours, pad with baseline
    if len(window_df) < seq_length:
        print(f"   ⚠️ Only {len(window_df)} hours in window, padding with baseline")
        missing_count = seq_length - len(window_df)
        baseline = get_baseline_features(target_datetime, missing_count)
        window_features = window_df[FEATURE_COLS].values.astype(np.float32)
        features = np.vstack([baseline, window_features])
    else:
        # Take the last seq_length hours
        window_df = window_df.tail(seq_length)
        features = window_df[FEATURE_COLS].values.astype(np.float32)
    
    # Ensure exactly seq_length rows
    if len(features) != seq_length:
        if len(features) > seq_length:
            features = features[:seq_length]
        else:
            while len(features) < seq_length:
                features = np.vstack([features, features[-1:]])
    
    # Apply feature scaler
    feature_scaler = get_feature_scaler(station_name, direction)
    if feature_scaler is not None:
        try:
            features = feature_scaler.transform(features)
        except Exception as e:
            print(f"⚠️ Feature scaling failed: {e}")
    
    return features

# Add this global cache
_TARGET_SCALER_CACHE = {}

def get_target_scaler(station_name, direction):
    """Load the target scaler for a station-direction"""
    cache_key = f"{station_name}_{direction}"
    
    # Check cache
    if cache_key in _TARGET_SCALER_CACHE:
        return _TARGET_SCALER_CACHE[cache_key]
    
    # Load from disk
    model_folder = 'models_2022-2024_v8'
    scaler_path = f'{model_folder}/{cache_key}_target_scaler.pkl'
    
    # Try alternative naming
    if not os.path.exists(scaler_path):
        alt_path = f'{model_folder}/{cache_key.replace(" ", "_")}_target_scaler.pkl'
        if os.path.exists(alt_path):
            scaler_path = alt_path
    
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                _TARGET_SCALER_CACHE[cache_key] = pickle.load(f)
            print(f"   ✅ Loaded target scaler for {cache_key}")
            return _TARGET_SCALER_CACHE[cache_key]
        except Exception as e:
            print(f"   ⚠️ Error loading target scaler: {e}")
            _TARGET_SCALER_CACHE[cache_key] = None
    else:
        print(f"   ⚠️ No target scaler found at: {scaler_path}")
        _TARGET_SCALER_CACHE[cache_key] = None
    
    return _TARGET_SCALER_CACHE[cache_key]

# Keep the original load_data function for backward compatibility
def load_data():
    """Legacy function - now uses fast loading"""
    return load_data_fast()
def add_cyclical_time_features(df):
    """Add cyclical time features using the DataFrame index"""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    
    # Use numpy for all calculations to avoid Index vs Series issues
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
    """Add operating flags using the DataFrame index"""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    
    # Convert index time to numpy array for operations
    time_decimal = df.index.hour + df.index.minute / 60
    
    # Use numpy for operations
    df['is_morning_rush'] = ((time_decimal >= 7.0) & (time_decimal <= 9.0)).astype(np.int8)
    df['is_evening_rush'] = ((time_decimal >= 17.0) & (time_decimal <= 19.0)).astype(np.int8)
    df['is_noon'] = ((time_decimal >= 12.0) & (time_decimal <= 13.0)).astype(np.int8)
    df['is_pre_opening'] = ((time_decimal >= 4.5) & (time_decimal < 5.0)).astype(np.int8)
    df['is_post_closing'] = ((time_decimal >= 22.5) & (time_decimal < 23.0)).astype(np.int8)
    
    # Use numpy for clip operations
    minutes_until = (23.0 - time_decimal) * 60
    df['minutes_until_closing'] = np.clip(minutes_until, 0, None).astype(np.float32)
    
    minutes_since = (time_decimal - 4.5) * 60
    df['minutes_since_opening'] = np.clip(minutes_since, 0, None).astype(np.float32)
    
    df['time_normalized'] = np.clip((time_decimal - 4.5) / (23.0 - 4.5), 0, 1)
    
    return df
def smart_data_cleaner(df):
    """Clean data and add maintenance flags using the DataFrame index"""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    
    time_decimal = df.index.hour + df.index.minute / 60
    passenger_count = df['TotalPassenger']
    
    df['is_maintenance_record'] = ((time_decimal < 5.0) & (passenger_count < 10)).astype(np.int8)
    df['is_extended_hours'] = ((time_decimal >= 22.0) & (time_decimal < 23.0) & (passenger_count >= 10)).astype(np.int8)
    
    # DON'T touch congestion here - let get_station_dataframe handle it
    return df
    
def is_christmas_season(date):
    month_day = date.strftime('%m-%d')
    return (month_day >= '12-15') or (month_day <= '01-05')

def is_payday(date):
    return date.day in [15, 30, 31]

def is_friday(date):
    return date.weekday() == 4

def infer_direction(row):
    entry = row['StationEntry']
    exit_st = row['StationExit']
    if entry < exit_st:
        return 'Southbound'
    elif entry > exit_st:
        return 'Northbound'
    else:
        return 'Unknown'

print("=" * 50)
print("✅ feature_engineering.py loaded successfully!")
print(f"✅ get_feature_sequence_for_station is defined: {get_feature_sequence_for_station is not None}")
print("=" * 50)
