"""
Prediction business logic using real 2023-2024 data - No Flask dependencies
"""
from datetime import datetime
import numpy as np

# Station data (should match your actual MRT stations)
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

STATION_BASE_CAPACITY = {
    "North Ave": 12000, "Quezon Ave": 9000, "Kamuning": 7500, "Cubao": 15000,
    "Santolan": 8000, "Ortigas": 9500, "Shaw Blvd": 11000, "Boni Ave": 8500,
    "Guadalupe": 10000, "Buendia": 9000, "Ayala Ave": 14000, "Magallanes": 9000, "Taft": 16000
}

# Global state (will be set by model_loader using real 2023-2024 data)
directional_models = {}
directional_scalers = {}
historical_entry = {}
historical_exit = {}
hourly_avg_entry = {}
hourly_avg_exit = {}


def get_wait_time(congestion):
    """Helper function to get wait time based on congestion"""
    if congestion > 80:
        return "15-20 min"
    elif congestion > 60:
        return "10-15 min"
    elif congestion > 30:
        return "5-10 min"
    else:
        return "2-5 min"


def get_best_time_to_travel(station_name=None):
    """Helper function to determine best travel time based on real patterns"""
    hour = datetime.now().hour
    # Use real historical patterns for better recommendations
    if 7 <= hour <= 9:
        return "10:00 AM - 3:00 PM (Avoid morning rush hour)"
    elif 17 <= hour <= 20:
        return "Before 5:00 PM or after 8:00 PM (Avoid evening rush hour)"
    else:
        # Check if current time is good based on historical data
        current_congestion = hourly_avg_entry.get(hour, 50)
        if current_congestion < 40:
            return "Now is a good time to travel!"
        else:
            return "Consider traveling in 1-2 hours for lighter traffic"

# services/predictor.py

def clamp_prediction_by_time(congestion, target_datetime):
    """
    MINIMAL clamping - only enforce absolute bounds.
    The model already learned operating hours from training data.
    """
    # Only enforce 0-100% range (absolute bounds)
    # DO NOT force values to 0 during operating hours
    # DO NOT cap to 15/30% during certain times
    
    # Just ensure it's within 0-100
    return min(100, max(0, congestion))


# Or even better - remove the clamp entirely:
def clamp_prediction_by_time(congestion, target_datetime):
    """No clamping - trust the model's predictions"""
    return congestion


def get_fallback_directional_prediction(station_name, direction, target_datetime=None,
                                         historical_entry=None, STATIONS=None, 
                                         STATION_BASE_CAPACITY=None,
                                         hourly_avg_entry=None):
    """
    Fallback when directional model isn't available
    Uses real 2023-2024 historical patterns
    """
    if target_datetime is None:
        target_datetime = datetime.now()
    
    hour = target_datetime.hour
    minute = target_datetime.minute
    time_decimal = hour + minute / 60
    
    OPERATING_START = 4.5
    OPERATING_END = 22.5
    
    stations = STATIONS or globals().get('STATIONS', [])
    station_base_capacity = STATION_BASE_CAPACITY or globals().get('STATION_BASE_CAPACITY', {})
    hist_entry = historical_entry or globals().get('historical_entry', {})
    hourly_avg = hourly_avg_entry or globals().get('hourly_avg_entry', {})
    
    if time_decimal < OPERATING_START or time_decimal >= OPERATING_END:
        return 0
    
    if 5 <= hour < 6:
        return 5 + (hour - 5) * 3
    
    if 21 <= hour < 22.5:
        return max(10, 40 - (hour - 21) * 15)
    
    station_idx = stations.index(station_name) if station_name in stations else 0
    capacity = station_base_capacity.get(station_name, 10000)
    
    # Use real historical data if available
    if hist_entry and station_name in hist_entry:
        base_ridership = hist_entry.get(station_name, capacity * 0.55)
    else:
        base_ridership = capacity * 0.55
    
    # Use real hourly patterns
    hour_factor = hourly_avg.get(hour, 4500) / 4500 if hourly_avg else 1.0
    
    # Direction-specific logic based on real MRT passenger flow
    is_morning_rush = 7 <= hour <= 9
    is_evening_rush = 17 <= hour <= 20
    
    if direction == 'Southbound':
        if is_morning_rush:
            # More passengers going southbound to CBD in morning
            if station_idx <= 5:  # Northern stations (North Ave to Cubao)
                multiplier = 1.65
            else:
                multiplier = hour_factor * 0.85
        elif is_evening_rush:
            # Less passengers going southbound in evening
            if station_idx <= 5:
                multiplier = 0.45
            else:
                multiplier = hour_factor * 1.45
        else:
            multiplier = hour_factor * 0.7
    else:  # Northbound
        if is_morning_rush:
            # Less passengers going northbound in morning
            if station_idx <= 5:
                multiplier = 0.35
            else:
                multiplier = hour_factor * 1.15
        elif is_evening_rush:
            # More passengers going northbound in evening (leaving CBD)
            if station_idx <= 5:
                multiplier = hour_factor * 1.55
            else:
                multiplier = 0.55
        else:
            multiplier = hour_factor * 0.7
    
    ridership = int(base_ridership * multiplier)
    ridership = min(ridership, capacity)
    congestion = min(100, int((ridership / capacity) * 100))
    
    return congestion


def get_directional_prediction(station_name, direction, target_datetime=None,
                                directional_models=None, directional_scalers=None,
                                get_feature_sequence_func=None,
                                fallback_func=None):
    """
    Get directional prediction using LSTM model trained on 2023-2024 real data
    """
    if target_datetime is None:
        target_datetime = datetime.now()
    
    model_key = f"{station_name}_{direction}"
    
    models = directional_models or globals().get('directional_models', {})
    scalers = directional_scalers or globals().get('directional_scalers', {})
    
    prediction = None
    
    if model_key in models:
        try:
            if get_feature_sequence_func:
                sequence = get_feature_sequence_func(station_name, direction, target_datetime)
            else:
                from .feature_engineering import get_feature_sequence_for_station
                sequence = get_feature_sequence_for_station(station_name, direction, target_datetime)
            
            if sequence is not None and len(sequence) == 24:
                feature_scaler = scalers.get(f'{model_key}_feature')
                target_scaler = scalers.get(f'{model_key}_target')
                
                if feature_scaler is not None and target_scaler is not None:
                    scaled_sequence = feature_scaler.transform(sequence)
                    input_sequence = scaled_sequence.reshape(1, 24, -1)
                    
                    pred_scaled = models[model_key].predict(input_sequence, verbose=0)
                    prediction = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
        except Exception as e:
            print(f"⚠️ Directional model error for {model_key}: {e}")
    
    if prediction is None:
        if fallback_func:
            prediction = fallback_func(station_name, direction, target_datetime)
        else:
            from .predictor import get_fallback_directional_prediction
            prediction = get_fallback_directional_prediction(station_name, direction, target_datetime)
    
    
    return min(100, max(0, prediction))


def get_station_prediction(station_name, target_datetime=None,
                           directional_models=None, directional_scalers=None,
                           get_feature_sequence_func=None):
    """
    Get congestion prediction for a station (averages both directions)
    Uses real 2023-2024 data and LSTM models
    """
    if target_datetime is None:
        target_datetime = datetime.now()
    
    try:
        north = get_directional_prediction(station_name, 'Northbound', target_datetime,
                                           directional_models, directional_scalers,
                                           get_feature_sequence_func)
        south = get_directional_prediction(station_name, 'Southbound', target_datetime,
                                           directional_models, directional_scalers,
                                           get_feature_sequence_func)
        
        avg_congestion = (north + south) / 2
        
        return min(100, max(0, avg_congestion))
        
    except Exception as e:
        print(f"⚠️ Error in get_station_prediction for {station_name}: {e}")
        hist_entry = globals().get('historical_entry', {})
        capacity = STATION_BASE_CAPACITY.get(station_name, 10000)
        base_ridership = hist_entry.get(station_name, capacity * 0.55)
        congestion = min(100, int((base_ridership / capacity) * 100))
        return congestion


def calculate_directional_congestion(station_name, direction, target_datetime=None):
    """
    Public function to calculate directional congestion
    """
    return get_directional_prediction(station_name, direction, target_datetime)