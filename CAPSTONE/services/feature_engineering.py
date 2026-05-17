"""
Feature engineering for LSTM predictions - No Flask dependencies
"""
from datetime import datetime, timedelta
import numpy as np

# Station data
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

STATION_BASE_CAPACITY = {
    "North Ave": 12000, "Quezon Ave": 9000, "Kamuning": 7500, "Cubao": 15000,
    "Santolan": 8000, "Ortigas": 9500, "Shaw Blvd": 11000, "Boni Ave": 8500,
    "Guadalupe": 10000, "Buendia": 9000, "Ayala Ave": 14000, "Magallanes": 9000, "Taft": 16000
}


def add_cyclical_time_features_for_prediction(df):
    """Add cyclical time features for a DataFrame (for batch predictions)"""
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


def add_smart_operating_flags_for_prediction(df):
    """Add smart operating flags for a DataFrame"""
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


def get_fallback_directional_prediction_for_features(station_name, direction, target_datetime,
                                                      historical_entry=None):
    """Get fallback prediction for feature generation"""
    hour = target_datetime.hour
    minute = target_datetime.minute
    time_decimal = hour + minute / 60
    
    OPERATING_START = 4.5
    OPERATING_END = 22.5
    
    if time_decimal < OPERATING_START or time_decimal >= OPERATING_END:
        return 0
    
    if 5 <= hour < 6:
        return 5 + (hour - 5) * 3
    
    if 21 <= hour < 22.5:
        return max(10, 40 - (hour - 21) * 15)
    
    station_idx = STATIONS.index(station_name) if station_name in STATIONS else 0
    capacity = STATION_BASE_CAPACITY.get(station_name, 10000)
    
    hist_entry = historical_entry or {}
    base_ridership = hist_entry.get(station_name, capacity * 0.55)
    
    is_morning_rush = 7 <= hour <= 9
    is_evening_rush = 17 <= hour <= 20
    
    if direction == 'Southbound':
        if is_morning_rush:
            multiplier = 1.65 if station_idx <= 5 else 0.85
        elif is_evening_rush:
            multiplier = 0.45 if station_idx <= 5 else 1.45
        else:
            multiplier = 0.7
    else:
        if is_morning_rush:
            multiplier = 0.35 if station_idx <= 5 else 1.15
        elif is_evening_rush:
            multiplier = 1.55 if station_idx <= 5 else 0.55
        else:
            multiplier = 0.7
    
    ridership = int(base_ridership * multiplier)
    ridership = min(ridership, capacity)
    return min(100, int((ridership / capacity) * 100))


# Add these functions at the top of feature_engineering.py

def is_holiday(date, holidays_list):
    """Check if a date is a holiday"""
    date_str = date.strftime('%Y-%m-%d')
    return date_str in holidays_list

def is_special_event(date, special_events_dict):
    """Check if a date has a special event"""
    date_str = date.strftime('%Y-%m-%d')
    return date_str in special_events_dict

def is_christmas_season(date):
    """Check if date is in Christmas season"""
    month_day = date.strftime('%m-%d')
    return (month_day >= '12-15') or (month_day <= '01-05')

def is_payday(date):
    """Check if date is a payday (15th, 30th, 31st)"""
    return date.day in [15, 30, 31]

def is_friday(date):
    """Check if date is Friday"""
    return date.weekday() == 4
def get_feature_sequence_for_station(station_name, direction, target_datetime,
                                      historical_entry=None):
    """
    Generate the 24-hour feature sequence needed for LSTM prediction
    VERSION 2 - WITH CYCLICAL FEATURES AND DATE-SPECIFIC FLAGS
    """
    from datetime import timedelta
    import numpy as np
    
    # Define holidays and events (you'll need to pass these or import them)
    # For now, we'll create empty dictionaries - you should import from your training config
    holidays_list = [
        '2023-01-01', '2023-04-06', '2023-04-07', '2023-05-01', '2023-06-12',
        '2023-08-28', '2023-11-27', '2023-12-08', '2023-12-25', '2023-12-30',
        '2024-01-01', '2024-03-28', '2024-03-29', '2024-05-01', '2024-06-12',
        '2024-08-26', '2024-11-30', '2024-12-08', '2024-12-25', '2024-12-30', '2024-12-31'
    ]
    
    special_events_dict = {
        '2023-01-09': 'Feast of Black Nazarene',
        '2023-04-21': 'Eid al-Fitr',
        '2023-06-28': 'Eid al-Adha',
        '2023-10-30': 'Barangay Elections',
        '2023-11-02': 'All Souls Day',
        '2024-02-10': 'Chinese New Year',
        '2024-03-11': 'Eid al-Fitr',
        '2024-08-08': 'Technical issue Boni-Guadalupe',
        '2024-08-21': 'Ninoy Aquino Day',
    }
    
    sequence = []
    
    for h in range(24, 0, -1):  # 24 hours back
        past_time = target_datetime - timedelta(hours=h)
        
        # Use fallback for missing historical data
        congestion = get_fallback_directional_prediction_for_features(
            station_name, direction, past_time, historical_entry
        )
        
        # V2 Feature extraction
        hour = past_time.hour
        minute = past_time.minute
        weekday = past_time.weekday()
        month = past_time.month
        time_decimal = hour + minute / 60
        
        # Cyclical features
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * weekday / 7)
        dow_cos = np.cos(2 * np.pi * weekday / 7)
        month_sin = np.sin(2 * np.pi * (month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12)
        
        # Operating flags
        is_operating_hour = 1 if (4.5 <= time_decimal < 23.0) else 0
        is_morning_rush = 1 if (7.0 <= time_decimal <= 9.0) else 0
        is_evening_rush = 1 if (17.0 <= time_decimal <= 19.0) else 0
        is_noon = 1 if (12.0 <= time_decimal <= 13.0) else 0
        is_pre_opening = 1 if (4.5 <= time_decimal < 5.0) else 0
        is_post_closing = 1 if (22.5 <= time_decimal < 23.0) else 0
        
        minutes_until_closing = max(0, (23.0 - time_decimal) * 60)
        minutes_since_opening = max(0, (time_decimal - 4.5) * 60)
        time_normalized = max(0, min(1, (time_decimal - 4.5) / (23.0 - 4.5)))
        minute_normalized = minute / 60.0
        
        # ========== DATE-SPECIFIC CALENDAR FEATURES (USING ACTUAL DATE) ==========
        is_weekend = 1 if weekday >= 5 else 0
        
        # Check if the actual date is a holiday
        date_str = past_time.strftime('%Y-%m-%d')
        is_holiday_val = 1 if date_str in holidays_list else 0
        
        # Check if the actual date has a special event
        is_special_event_val = 1 if date_str in special_events_dict else 0
        
        # Christmas season based on actual date
        month_day = past_time.strftime('%m-%d')
        is_christmas_season_val = 1 if (month_day >= '12-15') or (month_day <= '01-05') else 0
        
        # Payday based on actual date
        is_payday_val = 1 if past_time.day in [15, 30, 31] else 0
        
        # Friday based on actual date
        is_friday_val = 1 if weekday == 4 else 0
        
        is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
        
        # Data quality flags (0 for prediction)
        is_maintenance_record = 0
        is_extended_hours = 0
        
        features = [
            hour, weekday, month,
            hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos,
            is_operating_hour, is_morning_rush, is_evening_rush, is_noon,
            is_pre_opening, is_post_closing,
            minutes_until_closing, minutes_since_opening, time_normalized, minute_normalized,
            is_weekend, is_holiday_val, is_special_event_val, is_christmas_season_val, is_payday_val, is_friday_val,
            is_rush_hour,
            is_maintenance_record, is_extended_hours,
            congestion
        ]
        sequence.append(features)
    
    # Convert to numpy array with float32 - ONLY ONE RETURN
    result = np.array(sequence, dtype=np.float32)
    print(f"✅ Generated sequence for {station_name} {direction}: shape={result.shape}")
    return result