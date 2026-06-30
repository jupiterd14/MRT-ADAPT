"""
Feature engineering for LSTM predictions - OPTIMIZED FOR SPEED
"""

import pandas as pd
import numpy as np
import os
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
    """Load the training data (2022-2024) that the models were trained on"""
    # ========== FIX: Explicitly load only 2022-2024 data ==========
    data_files = []
    possible_years = ['2022', '2023', '2024']
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for year in possible_years:
        # Try multiple paths with explicit priority
        paths = [
            os.path.join(script_dir, f'data (2022-2024)/{year}.csv'),
            os.path.join(script_dir, f'../data (2022-2024)/{year}.csv'),
            os.path.join(script_dir, f'{year}.csv'),
            os.path.join(script_dir, f'data/{year}.csv'),
            f'data (2022-2024)/{year}.csv',
            f'../data (2022-2024)/{year}.csv',
            f'{year}.csv',
            f'data/{year}.csv'
        ]
        
        found = False
        for path in paths:
            if os.path.exists(path):
                print(f"📊 Loading {year} data from: {path}")
                try:
                    df = pd.read_csv(path)
                    data_files.append(df)
                    found = True
                    break
                except Exception as e:
                    print(f"   ❌ Error reading {path}: {e}")
        
        if not found:
            print(f"⚠️ {year}.csv not found, skipping...")
    
    if not data_files:
        print("❌ No training data files found (2022, 2023, 2024)")
        print("   Looking for: 2022.csv, 2023.csv, 2024.csv in:")
        print(f"   - {os.path.join(script_dir, 'data (2022-2024)/')}")
        print("   - ./")
        print("   - data/")
        return None
    
    # Combine all data
    df = pd.concat(data_files, ignore_index=True)
    print(f"✅ Loaded {len(df)} rows from {len(data_files)} years of training data")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Check passenger counts
    if 'TotalPassenger' in df.columns:
        print(f"   Passenger count - min: {df['TotalPassenger'].min()}, max: {df['TotalPassenger'].max()}, mean: {df['TotalPassenger'].mean():.1f}")
    
    # Process the data
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['minute'] = df['datetime'].dt.minute
    
    # Add features
    df = add_cyclical_time_features(df)
    df = add_smart_operating_flags(df)
    df = smart_data_cleaner(df)
    
    df['is_weekend'] = (df['datetime'].dt.weekday >= 5).astype(np.int8)
    df['is_holiday'] = 0
    df['is_special_event'] = 0
    df['is_christmas_season'] = df['datetime'].apply(is_christmas_season).astype(np.int8)
    df['is_payday'] = df['datetime'].apply(is_payday).astype(np.int8)
    df['is_friday'] = df['datetime'].apply(is_friday).astype(np.int8)
    df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(np.int8)
    df['Direction'] = df.apply(infer_direction, axis=1)
    
    return df

def load_data_fast():
    """Optimized data loading - creates cache on first run"""
    global _DATA_CACHE
    
    if _DATA_CACHE is not None:
        return _DATA_CACHE
    
    preprocessed_path = 'data/preprocessed.parquet'
    
    # Try loading preprocessed data first
    if os.path.exists(preprocessed_path):
        try:
            print("📊 Loading preprocessed data from cache...")
            _DATA_CACHE = pd.read_parquet(preprocessed_path)
            
            # Verify we have 2022-2024 data
            years_loaded = _DATA_CACHE['datetime'].dt.year.unique().tolist()
            print(f"   Years in data: {years_loaded}")
            print(f"   Date range: {_DATA_CACHE['datetime'].min()} to {_DATA_CACHE['datetime'].max()}")
            
            # If only 2025 data, force reload from CSV
            if years_loaded == [2025] or (len(years_loaded) == 1 and years_loaded[0] >= 2025):
                print("⚠️ WARNING: Only 2025 data found! Forcing reload from CSV files...")
                os.remove(preprocessed_path)
                _DATA_CACHE = None
                return load_data_fast()  # Recursively reload
            
            return _DATA_CACHE
            
        except Exception as e:
            print(f"⚠️ Error loading cache: {e}")
            _DATA_CACHE = None
    
    # Process raw data and save cache
    print("📊 Processing raw data from CSV files...")
    _DATA_CACHE = process_raw_data()
    if _DATA_CACHE is not None:
        # Create directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        _DATA_CACHE.to_parquet(preprocessed_path, compression='snappy')
        print(f"💾 Saved preprocessed data to {preprocessed_path} for future runs")
    else:
        return None
    
    return _DATA_CACHE

def get_station_dataframe(station_name, direction):
    """Cache station-specific dataframes - USE RAW PASSENGER COUNTS FOR LOOKBACK"""
    cache_key = f"{station_name}_{direction}"
    
    # Return from memory cache if available
    if cache_key in _STATION_DATA_CACHE:
        return _STATION_DATA_CACHE[cache_key]
    
    df = load_data_fast()
    if df is None:
        return None
    
    station_num = STATION_NUMBERS.get(station_name)
    if not station_num:
        print(f"❌ Unknown station: {station_name}")
        return None
    
    # Filter by station metrics and tracking direction
    if station_name == "North Ave":
        if direction == 'Northbound':
            station_df = df[df['StationExit'] == station_num].copy()
        else:
            station_df = df[df['StationEntry'] == station_num].copy()
    elif station_name == "Taft":
        if direction == 'Northbound':
            station_df = df[df['StationEntry'] == station_num].copy()
        else:
            station_df = df[df['StationExit'] == station_num].copy()
    else:
        if direction == 'Northbound':
            station_df = df[df['StationExit'] == station_num].copy()
        else:
            station_df = df[df['StationEntry'] == station_num].copy()
    
    # Filter by direction and sort
    station_df = station_df[station_df['Direction'] == direction].sort_values('datetime')
    
    if len(station_df) < 100:
        print(f"⚠️ Not enough data for {station_name} {direction}: {len(station_df)} rows")
        _STATION_DATA_CACHE[cache_key] = None
        return None
    
    # Get current time to exclude incomplete hour
    current_time = Config.get_current_time()
    current_hour_floor = current_time.replace(minute=0, second=0, microsecond=0)
    
    # Exclude current incomplete hour
    station_df = station_df[station_df['datetime'] < current_hour_floor]
    
    # Aggregate to hourly
    station_df['hour_timestamp'] = station_df['datetime'].dt.floor('h')
    hourly = station_df.groupby('hour_timestamp').agg({
        'TotalPassenger': 'sum',
        'hour': 'first', 'weekday': 'first', 'month': 'first',
        'minute': 'first',
        'hour_sin': 'first', 'hour_cos': 'first',
        'dow_sin': 'first', 'dow_cos': 'first',
        'month_sin': 'first', 'month_cos': 'first',
        'time_decimal': 'first',
        'is_operating_hour': 'first',
        'minute_normalized': 'first',
        'is_morning_rush': 'first', 'is_evening_rush': 'first', 'is_noon': 'first',
        'is_pre_opening': 'first', 'is_post_closing': 'first',
        'minutes_until_closing': 'first', 'minutes_since_opening': 'first',
        'time_normalized': 'first',
        'is_weekend': 'first', 'is_holiday': 'first', 'is_special_event': 'first',
        'is_christmas_season': 'first', 'is_payday': 'first', 'is_friday': 'first',
        'is_rush_hour': 'first',
        'is_maintenance_record': 'first', 'is_extended_hours': 'first'
    }).reset_index()
    
    # Create complete hour range
    if len(hourly) > 0:
        min_date = hourly['hour_timestamp'].min().floor('D')
        max_date = hourly['hour_timestamp'].max().ceil('D')
        all_hours = pd.date_range(start=min_date, end=max_date, freq='h')
        
        hourly.set_index('hour_timestamp', inplace=True)
        hourly = hourly.reindex(all_hours)
        
        # Fill closed hours with zeros
        closed_hour_mask = (hourly.index.hour >= 22) | (hourly.index.hour <= 4)
        if 'TotalPassenger' in hourly.columns:
            hourly.loc[closed_hour_mask, 'TotalPassenger'] = 0
        
        # Forward fill remaining NaN values
        hourly = hourly.ffill().bfill()
        
        # ========== CRITICAL FIX ==========
        # The model expects RAW PASSENGER COUNTS in the 'congestion' column
        # because it was trained on raw passenger counts
        # DO NOT convert to percentage - keep as raw passenger counts!
        hourly['congestion'] = hourly['TotalPassenger'].copy()
        
        # Also calculate actual congestion percentage for display (if needed elsewhere)
        capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
        hourly['congestion_percentage'] = (hourly['TotalPassenger'] / capacity * 100).clip(0, 100)
        
        hourly = hourly.sort_index()
    
    # Store in memory cache
    _STATION_DATA_CACHE[cache_key] = hourly
    
    print(f"✅ Created hourly data for {station_name} {direction}")
    print(f"   Raw passenger counts (lookback) - min: {hourly['congestion'].min():.0f}, max: {hourly['congestion'].max():.0f}, mean: {hourly['congestion'].mean():.0f}")
    
    return hourly

def get_feature_scaler(station_name, direction):
    """Cache feature scalers to avoid repeated disk I/O"""
    cache_key = f"{station_name}_{direction}"
    
    if cache_key not in _FEATURE_SCALER_CACHE:
        # ========== FIX: Use the correct model folder ==========
        model_folder = 'models_2022-2024_v8'
        scaler_path = f'{model_folder}/{cache_key}_feature_scaler.pkl'
        
        # Try alternative naming
        if not os.path.exists(scaler_path):
            # Check if file exists with different naming
            alt_path = f'{model_folder}/{cache_key.replace(" ", "_")}_feature_scaler.pkl'
            if os.path.exists(alt_path):
                scaler_path = alt_path
        
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    _FEATURE_SCALER_CACHE[cache_key] = pickle.load(f)
                print(f"   ✅ Loaded feature scaler for {cache_key}")
            except Exception as e:
                print(f"   ⚠️ Error loading scaler: {e}")
                _FEATURE_SCALER_CACHE[cache_key] = None
        else:
            _FEATURE_SCALER_CACHE[cache_key] = None
            print(f"   ⚠️ No feature scaler found at: {scaler_path}")
    
    return _FEATURE_SCALER_CACHE[cache_key]

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
            default_features[i, 3] = np.sin(2 * np.pi * h_val / 24)  # hour_sin
            default_features[i, 4] = np.cos(2 * np.pi * h_val / 24)  # hour_cos
            default_features[i, 5] = np.sin(2 * np.pi * loop_time.weekday() / 7)  # dow_sin
            default_features[i, 6] = np.cos(2 * np.pi * loop_time.weekday() / 7)  # dow_cos
            default_features[i, 7] = np.sin(2 * np.pi * (loop_time.month - 1) / 12)  # month_sin
            default_features[i, 8] = np.cos(2 * np.pi * (loop_time.month - 1) / 12)  # month_cos
            
            # Set congestion based on rush hours
            if 7 <= h_val <= 9:
                default_features[i, -1] = 3500.0  # Morning rush passenger count
            elif 17 <= h_val <= 19:
                default_features[i, -1] = 4000.0  # Evening rush passenger count
            elif 10 <= h_val <= 16:
                default_features[i, -1] = 2000.0  # Midday
            elif 5 <= h_val <= 6:
                default_features[i, -1] = 800.0   # Early morning
            elif 20 <= h_val <= 21:
                default_features[i, -1] = 1500.0  # Late evening
            else:
                default_features[i, -1] = 100.0   # Very early/late
        
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
    
    # ========== SCALE CONGESTION USING TARGET SCALER ==========
    target_scaler = get_target_scaler(station_name, direction)
    congestion_idx = -1
    
    if target_scaler is not None:
        try:
            # Extract raw congestion and scale using target scaler
            raw_congestion = features[:, congestion_idx].reshape(-1, 1)
            scaled_congestion = target_scaler.transform(raw_congestion)
            
            # Replace the congestion column with the scaled version
            features[:, congestion_idx] = scaled_congestion.flatten()
            
            print(f"   📊 {station_name} {direction}: Using target scaler max={target_scaler.data_max_[0]:.0f}")
            print(f"   📊 Raw congestion: min={raw_congestion.min():.1f}, max={raw_congestion.max():.1f}")
            print(f"   📊 Scaled congestion: min={features[:, congestion_idx].min():.4f}, max={features[:, congestion_idx].max():.4f}, mean={features[:, congestion_idx].mean():.4f}")
            
        except Exception as e:
            print(f"   ⚠️ Target scaler failed: {e}")
            # Fallback: use capacity-based scaling
            capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
            max_scale = capacity * 2.0
            features[:, congestion_idx] = features[:, congestion_idx] / max_scale
            features[:, congestion_idx] = np.clip(features[:, congestion_idx], 0, 1)
    else:
        # Fallback: use capacity-based scaling
        print(f"   ⚠️ No target scaler, using capacity fallback")
        capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
        max_scale = capacity * 2.0
        features[:, congestion_idx] = features[:, congestion_idx] / max_scale
        features[:, congestion_idx] = np.clip(features[:, congestion_idx], 0, 1)
    
    # ========== APPLY FEATURE SCALER TO ALL 29 FEATURES ==========
    feature_scaler = get_feature_scaler(station_name, direction)
    
    if feature_scaler is not None:
        try:
            # Scale ALL 29 features (including the now-scaled congestion column)
            features = feature_scaler.transform(features)
            print(f"   ✅ Scaled all 29 features with feature scaler")
            print(f"   ✅ Final shape: {features.shape}")
            
        except Exception as e:
            print(f"   ⚠️ Feature scaling failed: {e}")
    
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
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    df['time_decimal'] = df['hour'] + df['minute'] / 60
    df['is_operating_hour'] = ((df['time_decimal'] >= 4.5) & (df['time_decimal'] < 23.0)).astype(np.int8)
    df['minute_normalized'] = df['minute'] / 60.0
    return df

def add_smart_operating_flags(df):
    time_decimal = df['time_decimal']
    df['is_morning_rush'] = ((time_decimal >= 7.0) & (time_decimal <= 9.0)).astype(np.int8)
    df['is_evening_rush'] = ((time_decimal >= 17.0) & (time_decimal <= 19.0)).astype(np.int8)
    df['is_noon'] = ((time_decimal >= 12.0) & (time_decimal <= 13.0)).astype(np.int8)
    df['is_pre_opening'] = ((time_decimal >= 4.5) & (time_decimal < 5.0)).astype(np.int8)
    df['is_post_closing'] = ((time_decimal >= 22.5) & (time_decimal < 23.0)).astype(np.int8)
    minutes_until = (23.0 - time_decimal) * 60
    df['minutes_until_closing'] = minutes_until.clip(lower=0).astype(np.float32)
    minutes_since = (time_decimal - 4.5) * 60
    df['minutes_since_opening'] = minutes_since.clip(lower=0).astype(np.float32)
    df['time_normalized'] = ((time_decimal - 4.5) / (23.0 - 4.5)).clip(0, 1)
    return df

def smart_data_cleaner(df):
    time_decimal = df['time_decimal']
    passenger_count = df['TotalPassenger']
    df['is_maintenance_record'] = ((time_decimal < 5.0) & (passenger_count < 10)).astype(np.int8)
    df['is_extended_hours'] = ((time_decimal >= 22.0) & (time_decimal < 23.0) & (passenger_count >= 10)).astype(np.int8)
    df.loc[df['is_maintenance_record'] == 1, 'congestion'] = 0
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
