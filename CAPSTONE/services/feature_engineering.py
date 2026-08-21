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
_TYPICAL_PATTERN_CACHE = {}  # Cache for typical day patterns (NEW)

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
                    hourly = filtered.resample('h').sum()
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

    # FIX: Use SAME congestion calculation as training
    percentage = (combined['TotalPassenger'] / capacity * 100).clip(0, 100)
    combined['congestion'] = percentage  # ← Directly use percentage!
    combined['congestion_percentage'] = percentage
    combined['raw_passengers'] = combined['TotalPassenger']
    
    # Add time features
    combined = add_cyclical_time_features(combined)
    combined = add_smart_operating_flags(combined)
    combined = smart_data_cleaner(combined)
    
    
    combined['hour'] = combined.index.hour
    combined['weekday'] = combined.index.weekday
    combined['month'] = combined.index.month
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
    
    # ========== DETERMINE CORRECT FILTERING BASED ON STATION TYPE ==========
    # North Ave (Station 1) = Northern Terminal
    # Taft (Station 13) = Southern Terminal
    # All others = Middle Stations
    
    is_north_terminal = (station_name == "North Ave")
    is_south_terminal = (station_name == "Taft")
    
    all_hourly_data = []
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
                
                # ========== CORRECT DIRECTION FILTERING ==========
                if direction == 'Northbound':
                    # Northbound = people going NORTH
                    if is_north_terminal:
                        # North Ave: Northbound = people EXITING (arriving from south)
                        filtered = chunk[chunk['StationExit'] == station_num]
                    elif is_south_terminal:
                        # Taft: Northbound = people ENTERING (going north)
                        filtered = chunk[chunk['StationEntry'] == station_num]
                    else:
                        # Middle stations: Northbound = people EXITING (arriving from south)
                        filtered = chunk[chunk['StationExit'] == station_num]
                else:
                    # Southbound = people going SOUTH
                    if is_north_terminal:
                        # North Ave: Southbound = people ENTERING (going south)
                        filtered = chunk[chunk['StationEntry'] == station_num]
                    elif is_south_terminal:
                        # Taft: Southbound = people EXITING (arriving from north)
                        filtered = chunk[chunk['StationExit'] == station_num]
                    else:
                        # Middle stations: Southbound = people ENTERING (going south)
                        filtered = chunk[chunk['StationEntry'] == station_num]
                
                if len(filtered) > 0:
                    filtered['datetime'] = pd.to_datetime(filtered['Date'] + ' ' + filtered['Time'])
                    filtered = filtered.set_index('datetime')
                    filtered = filtered[['TotalPassenger']]
                    
                    try:
                        hourly = filtered.resample('h').sum()
                    except Exception as e:
                        try:
                            hourly = filtered.resample('H').sum()
                        except Exception as e2:
                            try:
                                hourly = filtered.resample('1h').sum()
                            except Exception as e3:
                                hourly = filtered.groupby(filtered.index.floor('h')).sum()
                    
                    all_hourly_data.append(hourly)
                
                del chunk, filtered
                gc.collect()
                
        except Exception as e:
            print(f"   ⚠️ Error processing {csv_file}: {e}")
            continue
    
    if not all_hourly_data:
        print(f"⚠️ No real data found for {cache_key}")
        return None
    
    print(f"   🔄 Combining {len(all_hourly_data)} chunks...")
    combined = pd.concat(all_hourly_data)
    del all_hourly_data
    gc.collect()
    
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    
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
    # ========== CALCULATE CONGESTION ==========
    capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)

    # FIX: Use SAME congestion calculation as training
    # Training uses: congestion = (TotalPassenger / capacity * 100).clip(0, 100)
    # This gives values 0-100 (percentage of capacity)
    percentage = (combined['TotalPassenger'] / capacity * 100).clip(0, 100)
    combined['congestion'] = percentage  # ← Directly use percentage, NO extra scaling!
    combined['congestion_percentage'] = percentage
    combined['raw_passengers'] = combined['TotalPassenger']
    
    # Add date-based features
    combined['is_weekend'] = (combined.index.weekday >= 5).astype(np.int8)
    combined['is_christmas_season'] = np.array([is_christmas_season(d) for d in combined.index], dtype=np.int8)
    combined['is_payday'] = combined.index.day.isin([15, 30, 31]).astype(np.int8)
    combined['is_friday'] = (combined.index.weekday == 4).astype(np.int8)
    combined['is_rush_hour'] = ((combined['hour'].between(7, 9)) | (combined['hour'].between(17, 19))).astype(np.int8)
    combined['is_holiday'] = 0
    combined['is_special_event'] = 0
    
    # Cache
    if len(_STATION_DATA_CACHE) >= 5:
        oldest_key = next(iter(_STATION_DATA_CACHE))
        print(f"🗑️ Removing oldest from cache: {oldest_key}")
        del _STATION_DATA_CACHE[oldest_key]
        gc.collect()
    
    _STATION_DATA_CACHE[cache_key] = combined
    print(f"✅ Cached {len(combined)} hours for {cache_key}")
    
    return combined

def get_baseline_features(target_datetime, seq_length=24):
    """Cache baseline features for repeated lookups with better defaults"""
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
            
            # ========== MORE REALISTIC BASELINE CONGESTION ==========
            if 7 <= h_val <= 9:
                # Morning rush
                default_features[i, -1] = 60 + (h_val - 7) * 10  # 60% at 7am, 80% at 9am
            elif 17 <= h_val <= 19:
                # Evening rush
                default_features[i, -1] = 65 + (h_val - 17) * 10  # 65% at 5pm, 85% at 7pm
            elif 10 <= h_val <= 16:
                # Midday
                default_features[i, -1] = 40 + (h_val - 10) * 3  # 40-58%
            elif 5 <= h_val <= 6:
                # Early morning - increased from 20%
                default_features[i, -1] = 25 + (h_val - 5) * 10  # 25% at 5am, 35% at 6am
            elif 20 <= h_val <= 22:
                # Late evening
                default_features[i, -1] = 35 - (h_val - 20) * 5  # 35% at 8pm, 25% at 10pm
            else:
                # Very early/late
                default_features[i, -1] = 10
        
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
    """Get feature sequence - uses actual recent data when available, else typical pattern"""
    
    df = get_station_dataframe_cached(station_name, direction)
    
    if df is None or len(df) == 0:
        return get_baseline_features(target_datetime, seq_length)
    
    latest_date = df.index.max()
    
    # ========== FIX: Use typical pattern for future dates, but with hour variation ==========
    # Check if target is in the future (no data available)
    if target_datetime > latest_date:
        # Use typical pattern with hour variation
        lookback_df = build_typical_day_pattern(df, target_datetime, seq_length, station_name, direction)
        print(f"🔮 Using typical pattern for {station_name} {direction} at hour {target_datetime.hour}")
    else:
        # Try to get actual data for the lookback window
        mask = df.index < target_datetime
        if mask.sum() >= seq_length:
            # Use actual historical data
            lookback_df = df[mask].tail(seq_length)
            print(f"📊 Using actual historical data for {station_name} {direction}")
        elif mask.sum() > 0:
            # Use what's available
            lookback_df = df[mask].tail(seq_length)
            print(f"📊 Using partial actual data for {station_name} {direction}")
        else:
            # No data available - use typical pattern
            lookback_df = build_typical_day_pattern(df, target_datetime, seq_length, station_name, direction)
            print(f"🔮 Using typical pattern for {station_name} {direction}")
    
    # Make sure congestion column is in the right range (0-100)
    if 'congestion' in lookback_df.columns:
        lookback_df['congestion'] = lookback_df['congestion'].clip(0, 100)
    
    features = lookback_df[FEATURE_COLS].values.astype(np.float32)
    return features
def build_typical_day_pattern(df, target_datetime, seq_length=24, station_name=None, direction=None):
    """
    Build a typical day pattern from historical averages with HOUR-SPECIFIC patterns.
    """
    date_key = target_datetime.strftime('%Y%m%d_%H')
    cache_key = f"{station_name}_{direction}_{date_key}"
    
    if cache_key in _TYPICAL_PATTERN_CACHE:
        print(f"📦 Using cached typical pattern for {station_name} {direction} at hour {target_datetime.hour}")
        return _TYPICAL_PATTERN_CACHE[cache_key].copy()
    
    capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000) if station_name else 1000
    
    # ========== GET HOUR-SPECIFIC CONGESTION FROM REAL DATA ==========
    hourly_congestion = {}
    
    for hour in range(24):
        hour_data = df[df.index.hour == hour]
        non_zero = hour_data[hour_data['TotalPassenger'] > 0]
        
        if len(non_zero) > 50:
            # Use P75 for realistic peak patterns
            hourly_congestion[hour] = non_zero['congestion'].quantile(0.75)
        elif len(non_zero) > 20:
            # Use median if less data
            hourly_congestion[hour] = non_zero['congestion'].median()
        else:
            hourly_congestion[hour] = 0
    
    # ========== APPLY REALISTIC RUSH HOUR PATTERNS ==========
    # Override with realistic values based on station and direction
    
    # Check if this is a terminal station
    is_terminal = station_name in ["North Ave", "Taft"]
    
    for hour in range(24):
        # ========== EARLY MORNING (5-6 AM) ==========
        if 5 <= hour <= 6:
            if direction == 'Northbound':
                # Northbound early morning - commuters heading to work
                if is_terminal:
                    # Terminals have higher early morning traffic
                    base = 30 + (hour - 5) * 15  # 30% at 5am, 45% at 6am
                else:
                    base = 25 + (hour - 5) * 10  # 25% at 5am, 35% at 6am
            else:
                # Southbound early morning - lighter
                if is_terminal:
                    base = 20 + (hour - 5) * 10  # 20% at 5am, 30% at 6am
                else:
                    base = 15 + (hour - 5) * 10  # 15% at 5am, 25% at 6am
            hourly_congestion[hour] = max(hourly_congestion.get(hour, 0), base)
        
        # ========== MORNING RUSH (7-9 AM) ==========
        elif 7 <= hour <= 9:
            if direction == 'Northbound':
                # Northbound morning rush - people heading to work
                if is_terminal:
                    base = 50 + (hour - 7) * 20  # 50% at 7am, 70% at 8am, 90% at 9am
                else:
                    base = 40 + (hour - 7) * 15  # 40% at 7am, 55% at 8am, 70% at 9am
            else:
                # Southbound morning rush - lighter (reverse commute)
                if is_terminal:
                    base = 30 + (hour - 7) * 15  # 30% at 7am, 45% at 8am, 60% at 9am
                else:
                    base = 25 + (hour - 7) * 10  # 25% at 7am, 35% at 8am, 45% at 9am
            hourly_congestion[hour] = max(hourly_congestion.get(hour, 0), base)
        
        # ========== MIDDAY (10 AM - 4 PM) ==========
        elif 10 <= hour <= 16:
            # Moderate congestion
            base = 30 + (hour - 10) * 2  # 30-42% throughout midday
            hourly_congestion[hour] = max(hourly_congestion.get(hour, 0), base)
        
        # ========== EVENING RUSH (5-7 PM) ==========
        elif 17 <= hour <= 19:
            if direction == 'Northbound':
                # Northbound evening rush - people heading home
                if is_terminal:
                    base = 60 + (hour - 17) * 15  # 60% at 5pm, 75% at 6pm, 90% at 7pm
                else:
                    base = 50 + (hour - 17) * 15  # 50% at 5pm, 65% at 6pm, 80% at 7pm
            else:
                # Southbound evening rush - reverse commute
                if is_terminal:
                    base = 40 + (hour - 17) * 15  # 40% at 5pm, 55% at 6pm, 70% at 7pm
                else:
                    base = 35 + (hour - 17) * 10  # 35% at 5pm, 45% at 6pm, 55% at 7pm
            hourly_congestion[hour] = max(hourly_congestion.get(hour, 0), base)
        
        # ========== LATE EVENING (8-10 PM) ==========
        elif 20 <= hour <= 22:
            # Decreasing congestion
            base = 40 - (hour - 20) * 10  # 40% at 8pm, 30% at 9pm, 20% at 10pm
            hourly_congestion[hour] = max(hourly_congestion.get(hour, 0), base)
    
    # ========== ENSURE TERMINAL STATIONS HAVE HIGHER BASE ==========
    if is_terminal:
        for hour in [5, 6, 7, 8, 17, 18, 19]:
            if direction == 'Northbound':
                hourly_congestion[hour] = max(hourly_congestion.get(hour, 0), 35)
            else:
                hourly_congestion[hour] = max(hourly_congestion.get(hour, 0), 25)
    
    # ========== CREATE THE SEQUENCE ==========
    typical_data = []
    target_hour = target_datetime.hour
    
    for i in range(seq_length):
        hour = (target_hour - (seq_length - i)) % 24
        congestion_val = hourly_congestion.get(hour, 20)
        typical_data.append(congestion_val)
    
    # Create DataFrame
    base_date = target_datetime.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    index = [base_date + timedelta(hours=i) for i in range(seq_length)]
    
    typical_df = pd.DataFrame({
        'congestion': typical_data,
        'TotalPassenger': [c / 100 * capacity for c in typical_data]
    }, index=index)
    
    # Add all feature columns
    typical_df = add_cyclical_time_features(typical_df)
    typical_df = add_smart_operating_flags(typical_df)
    typical_df = smart_data_cleaner(typical_df)
    
    typical_df['hour'] = typical_df.index.hour
    typical_df['weekday'] = typical_df.index.weekday
    typical_df['month'] = typical_df.index.month
    
    typical_df['is_weekend'] = (typical_df.index.weekday >= 5).astype(np.int8)
    typical_df['is_christmas_season'] = np.array([is_christmas_season(d) for d in typical_df.index], dtype=np.int8)
    typical_df['is_payday'] = typical_df.index.day.isin([15, 30, 31]).astype(np.int8)
    typical_df['is_friday'] = (typical_df.index.weekday == 4).astype(np.int8)
    typical_df['is_rush_hour'] = ((typical_df['hour'].between(7, 9)) | (typical_df['hour'].between(17, 19))).astype(np.int8)
    typical_df['is_holiday'] = 0
    typical_df['is_special_event'] = 0
    
    print(f"   📊 Built typical day pattern for {station_name} {direction} at hour {target_hour}:")
    print(f"      Avg congestion: {typical_df['congestion'].mean():.1f}%")
    print(f"      Max congestion: {typical_df['congestion'].max():.1f}%")
    print(f"      Min congestion: {typical_df['congestion'].min():.1f}%")
    print(f"      Hour {target_hour} value: {hourly_congestion.get(target_hour, 20):.1f}%")
    
    _TYPICAL_PATTERN_CACHE[cache_key] = typical_df.copy()
    
    return typical_df
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