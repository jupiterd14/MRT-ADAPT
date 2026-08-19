"""
Feature engineering for LSTM predictions - OPTIMIZED FOR SPEED
"""

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
    Memory-optimized - reads ONLY the needed station data from CSV
    Uses StationEntry and StationExit columns (NO 'Station' column!)
    """
    global _STATION_DATA_CACHE
    
    cache_key = f"{station_name}_{direction}"
    
    # Return from cache if available
    if cache_key in _STATION_DATA_CACHE:
        print(f"📦 Using cached data for {cache_key}")
        return _STATION_DATA_CACHE[cache_key]
    
    print(f"📊 Loading data for {cache_key} from CSV...")
    
    data_dir = 'services/data (2022-2024)'
    csv_files = ['2022.csv', '2023.csv', '2024.csv']
    
    all_data = []
    station_num = STATION_NUMBERS.get(station_name)
    
    if not station_num:
        print(f"❌ Unknown station: {station_name}")
        return None
    
    for csv_file in csv_files:
        filepath = os.path.join(data_dir, csv_file)
        if os.path.exists(filepath):
            try:
                chunk_size = 10000
                for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                    if direction == 'Northbound':
                        filtered = chunk[chunk['StationExit'] == station_num]
                    else:
                        filtered = chunk[chunk['StationEntry'] == station_num]
                    
                    if len(filtered) > 0:
                        all_data.append(filtered)
                    del chunk
                    gc.collect()
            except Exception as e:
                print(f"⚠️ Error loading {csv_file}: {e}")
    
    if not all_data:
        print(f"⚠️ No data found for {cache_key}")
        _STATION_DATA_CACHE[cache_key] = None
        return None
    
    # Combine filtered data
    combined = pd.concat(all_data)
    combined['datetime'] = pd.to_datetime(combined['Date'] + ' ' + combined['Time'])
    combined = combined.set_index('datetime')
    combined = combined.sort_index()
    
    # Resample to hourly
    hourly = combined.resample('H').sum()
    
    # ========== CALCULATE CONGESTION ==========
    capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
    
    feature_scaler = get_feature_scaler(station_name, direction)
    if feature_scaler is not None:
        training_max = feature_scaler.data_max_[-1] if hasattr(feature_scaler, 'data_max_') else 50
    else:
        training_max = min(capacity * 0.3, 50)
    
    percentage = (hourly['TotalPassenger'] / capacity * 100).clip(0, 100)
    hourly['congestion'] = (percentage / 100) * training_max
    hourly['congestion_percentage'] = percentage
    hourly['raw_passengers'] = hourly['TotalPassenger']
    # ========================================
    
    # Add time features
    hourly = add_cyclical_time_features(hourly)
    hourly = add_smart_operating_flags(hourly)
    hourly = smart_data_cleaner(hourly)
    
    # Add date-based features
    hourly['is_weekend'] = (hourly.index.weekday >= 5).astype(np.int8)
    hourly['is_christmas_season'] = np.array([is_christmas_season(d) for d in hourly.index], dtype=np.int8)
    hourly['is_payday'] = hourly.index.day.isin([15, 30, 31]).astype(np.int8)
    hourly['is_friday'] = (hourly.index.weekday == 4).astype(np.int8)
    hourly['is_rush_hour'] = ((hourly['hour'].between(7, 9)) | (hourly['hour'].between(17, 19))).astype(np.int8)
    hourly['is_holiday'] = 0
    hourly['is_special_event'] = 0
    
    # Free memory
    del combined
    del all_data
    gc.collect()
    
    # SMALL CACHE - Keep only 2 stations in memory
    if len(_STATION_DATA_CACHE) >= 2:
        oldest_key = next(iter(_STATION_DATA_CACHE))
        print(f"🗑️ Removing oldest from cache: {oldest_key}")
        del _STATION_DATA_CACHE[oldest_key]
        gc.collect()
    
    _STATION_DATA_CACHE[cache_key] = hourly
    
    print(f"✅ Loaded {len(hourly)} hours for {cache_key}")
    return hourly
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

def get_feature_sequence_for_station(station_name, direction, target_datetime, seq_length=24):
    """Get feature sequence - congestion is scaled, then ALL 29 features go through scaler"""
    station_name = unquote(station_name)
    direction = unquote(direction)
    
    hourly = get_station_dataframe(station_name, direction)
    if hourly is None:
        print(f"⚠️ No hourly data for {station_name} {direction}, using baseline")
        return get_baseline_features(target_datetime, seq_length)
    
    # Handle 2025 predictions
    if target_datetime.year >= 2025:
        try:
            lookback_end = target_datetime.replace(year=2024)
        except:
            lookback_end = target_datetime.replace(year=2024, month=2, day=28)
        start_lookback = lookback_end - timedelta(hours=seq_length)
    else:
        start_lookback = target_datetime - timedelta(hours=seq_length)
        lookback_end = target_datetime
    
    # Get available data in the window
    sequence_df = hourly[(hourly.index >= start_lookback) & (hourly.index < lookback_end)]
    
    available_rows = len(sequence_df)
    
    if available_rows == 0:
        return get_baseline_features(target_datetime, seq_length)
    
    # Get features with raw passenger counts
    if available_rows == seq_length:
        features_df = sequence_df[FEATURE_COLS].copy()
        features = features_df.values.astype(np.float32)
    elif available_rows >= 10:
        features_df = sequence_df[FEATURE_COLS].copy()
        features = features_df.values.astype(np.float32)
        if available_rows < seq_length:
            missing_count = seq_length - available_rows
            baseline_features = get_baseline_features(target_datetime, missing_count)
            features = np.vstack([baseline_features, features])
    else:
        return get_baseline_features(target_datetime, seq_length)
    
    # Ensure exactly seq_length rows
    if len(features) != seq_length:
        if len(features) > seq_length:
            features = features[:seq_length]
        else:
            while len(features) < seq_length:
                features = np.vstack([features, features[-1:]])
    
    
    # ========== APPLY FEATURE SCALER TO ALL 29 FEATURES ==========
    feature_scaler = get_feature_scaler(station_name, direction)
    
    if feature_scaler is not None:
        try:
            #print("\n========== BEFORE SCALING ==========")
            #print("First row:", features[0])
            #print("Congestion range:",
                #features[:, -1].min(),
                #features[:, -1].max())

            # Scale all 29 features
            features = feature_scaler.transform(features)

            #print("\n========== AFTER SCALING ==========")
            #print("First row:", features[0])
            #print("Congestion range:",
                #features[:, -1].min(),
                #features[:, -1].max())

            #print(f"✅ Scaled all {features.shape[1]} features")
            #print(f"✅ Final shape: {features.shape}")

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
