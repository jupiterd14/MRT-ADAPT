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
                               get_sequence_func=None):
    """Get directional prediction (0-100%) for a station"""
    
    # If no models provided, return fallback
    if directional_models is None or directional_scalers is None:
        print(f"[DEBUG] No models provided, using fallback")
        return get_fallback_directional_prediction(station_name, direction, target_datetime)
    
    model_key = f"{station_name}_{direction}"
    
    if model_key not in directional_models:
        print(f"[DEBUG] Model {model_key} not found, using fallback")
        return get_fallback_directional_prediction(station_name, direction, target_datetime)
    
    try:
        if target_datetime is None:
            target_datetime = datetime.now()
        
        # ========== ADD OPERATING HOURS CHECK ==========
        hour = target_datetime.hour
        minute = target_datetime.minute
        time_decimal = hour + minute / 60
        
        OPERATING_START = 4.5   # 4:30 AM
        OPERATING_END = 22.5    # 10:30 PM
        
        # MRT-3 is effectively closed outside these hours
        if time_decimal < OPERATING_START or time_decimal >= OPERATING_END:
            print(f"[DEBUG] {station_name} {direction} at {target_datetime} - OUTSIDE OPERATING HOURS ({time_decimal:.1f}), returning 0%")
            return 0
        
        # Also handle late evening (10:30 PM - 11:30 PM) - trains are rare
        if time_decimal >= 22.0 and time_decimal < 23.0:
            print(f"[DEBUG] {station_name} {direction} at {target_datetime} - LATE EVENING, low congestion expected")
            # Return a low value but let the model decide
            pass
        
        # If no sequence function, return fallback
        if get_sequence_func is None:
            print(f"[DEBUG] No sequence function provided, using fallback")
            return get_fallback_directional_prediction(station_name, direction, target_datetime)
        
        sequence = get_sequence_func(station_name, direction, target_datetime)
        
        if sequence is None:
            print(f"[DEBUG] Sequence is None, using fallback")
            return get_fallback_directional_prediction(station_name, direction, target_datetime)
        
        # Ensure sequence has correct shape
        if len(sequence.shape) == 2:
            input_sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        else:
            input_sequence = sequence.reshape(1, 24, -1)
        
        # Make prediction directly on the normalized sequence
        pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
        
        # Convert to percentage
        target_scaler = directional_scalers.get(f'{model_key}_target')
        
        if target_scaler is not None:
            # Use the target scaler to inverse transform
            pred_reshaped = pred_scaled.reshape(-1, 1)
            prediction_raw = target_scaler.inverse_transform(pred_reshaped)
            prediction = float(prediction_raw[0][0])
            print(f"[DEBUG] Using target scaler: {pred_scaled[0][0]:.3f} -> {prediction:.1f}%")
        else:
            # If no target scaler, assume model outputs 0-1 (normalized congestion)
            # Convert to percentage by multiplying by 100
            prediction = float(pred_scaled[0][0]) * 100
            print(f"[DEBUG] No target scaler: {pred_scaled[0][0]:.3f} -> {prediction:.1f}%")
        
        # Apply operating hours post-processing for late evening
        if time_decimal >= 22.0 and time_decimal < 23.0:
            # Reduce prediction by 50% for last hour of operation
            prediction = prediction * 0.5
            print(f"[DEBUG] Late evening adjustment: {prediction:.1f}%")
        
        # Clamp and return
        prediction = max(0, min(100, prediction))
        return prediction
        
    except Exception as e:
        print(f"Error predicting {station_name} {direction}: {e}")
        import traceback
        traceback.print_exc()
        return get_fallback_directional_prediction(station_name, direction, target_datetime)
    
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


def calculate_directional_congestion(station_name, direction, target_datetime=None,
                                      directional_models=None, directional_scalers=None,
                                      get_sequence_func=None):
    """
    Public function to calculate directional congestion
    """
    return get_directional_prediction(station_name, direction, target_datetime,
                                       directional_models, directional_scalers,
                                       get_sequence_func)


# Add debug print at the end
print("=" * 50)
print("✅ predictor.py loaded successfully!")
print("✅ get_directional_prediction is defined")
print("=" * 50)