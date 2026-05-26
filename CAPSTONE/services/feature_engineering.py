# services/feature_engineering.py
"""
Feature engineering for LSTM predictions - MATCHES DIAGNOSIS SCRIPT
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta

# ========== GLOBAL CACHE ==========
_DATA_CACHE = None
_PER_DIRECTION_MAX = None

# Station numbers (matching training)
STATION_NUMBERS = {
    "North Ave": 1, "Quezon Ave": 2, "Kamuning": 3, "Cubao": 4,
    "Santolan": 5, "Ortigas": 6, "Shaw Blvd": 7, "Boni Ave": 8,
    "Guadalupe": 9, "Buendia": 10, "Ayala Ave": 11, "Magallanes": 12,
    "Taft": 13
}

def load_data():
    """Load data once and cache it (like diagnosis script)"""
    global _DATA_CACHE, _PER_DIRECTION_MAX
    
    if _DATA_CACHE is not None:
        return _DATA_CACHE
    
    # Find data file
    possible_paths = [
        'data (2022-2024)/2025.csv',
        '../data (2022-2024)/2025.csv',
        '2025.csv',
        'data/2025.csv'
    ]
    
    data_file = None
    for path in possible_paths:
        if os.path.exists(path):
            data_file = path
            break
    
    if not data_file:
        print("❌ No data file found")
        return None
    
    print(f"📊 Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['minute'] = df['datetime'].dt.minute
    
    # Add features (matching diagnosis script)
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
    
    # Load per-direction max
    max_path = 'models_2022-2024_v5/per_direction_max_passengers.pkl'
    if os.path.exists(max_path):
        with open(max_path, 'rb') as f:
            _PER_DIRECTION_MAX = pickle.load(f)
        print(f"✅ Loaded {len(_PER_DIRECTION_MAX)} per-direction max values")
    
    _DATA_CACHE = df
    return _DATA_CACHE

# ========== FEATURE ENGINEERING FUNCTIONS (COPY FROM DIAGNOSIS) ==========
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

# ========== MAIN FUNCTION (MATCHES DIAGNOSIS) ==========
def get_feature_sequence_for_station(station_name, direction, target_datetime, seq_length=24):
    """
    Generate feature sequence - EXACTLY like diagnosis script
    """
    df = load_data()
    if df is None:
        return None
    
    station_num = STATION_NUMBERS.get(station_name)
    if not station_num:
        print(f"❌ Unknown station: {station_name}")
        return None
    
    # Filter for the correct station and direction
    # For Northbound: use exits, for Southbound: use entries
    if direction == 'Northbound':
        station_df = df[df['StationExit'] == station_num].copy()
    else:
        station_df = df[df['StationEntry'] == station_num].copy()
    
    # Filter by direction
    station_df = station_df[station_df['Direction'] == direction].sort_values('datetime')
    
    if len(station_df) < 100:
        print(f"⚠️ Not enough data for {station_name} {direction}: {len(station_df)} rows")
        return None
    
    # Aggregate by hour
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
    
    # Calculate congestion using training max
    key = f"{station_name}_{direction}"
    if _PER_DIRECTION_MAX and key in _PER_DIRECTION_MAX:
        station_max = _PER_DIRECTION_MAX[key]
    else:
        station_max = hourly['TotalPassenger'].quantile(0.99)
        if station_max == 0:
            station_max = 1
    
    hourly['congestion'] = (hourly['TotalPassenger'] / station_max * 100).clip(0, 100)
    hourly = hourly.sort_values('hour_timestamp')
    
    # Find target time
    target_hour = pd.to_datetime(target_datetime).floor('h')
    if target_hour not in hourly['hour_timestamp'].values:
        available = hourly[hourly['hour_timestamp'] <= target_hour]['hour_timestamp'].values
        if len(available) == 0:
            return None
        target_hour = available[-1]
    
    idx = hourly[hourly['hour_timestamp'] == target_hour].index[0]
    if idx < seq_length:
        return None
    
    # Get sequence
    history_df = hourly.iloc[idx-seq_length:idx]
    
    FEATURE_COLS = [
        'hour', 'weekday', 'month',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'is_operating_hour', 'is_morning_rush', 'is_evening_rush', 'is_noon',
        'is_pre_opening', 'is_post_closing',
        'minutes_until_closing', 'minutes_since_opening', 'time_normalized', 'minute_normalized',
        'is_weekend', 'is_holiday', 'is_special_event', 'is_christmas_season', 'is_payday', 'is_friday',
        'is_rush_hour', 'is_maintenance_record', 'is_extended_hours', 'congestion'
    ]
    
    features = history_df[FEATURE_COLS].values.astype(np.float32)
    
    scaler_path = f'models_2022-2024_v5/{station_name}_{direction}_feature_scaler.pkl'
    
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)
        features = feature_scaler.transform(features)
        print(f"  DEBUG: Used saved scaler - features range: {features.min():.3f} - {features.max():.3f}")
    else:
        print(f"  WARNING: No scaler found at {scaler_path}")
        # Fallback to manual normalization
        features[:, 0] = features[:, 0] / 24.0
        features[:, 1] = features[:, 1] / 7.0
        features[:, 2] = (features[:, 2] - 1) / 12.0
        features[:, -1] = features[:, -1] / 100.0
    
    return features


# At the end of services/feature_engineering.py
print("=" * 50)
print("✅ feature_engineering.py loaded successfully!")
print(f"✅ get_feature_sequence_for_station is defined: {get_feature_sequence_for_station is not None}")
print("=" * 50)