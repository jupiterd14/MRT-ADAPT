"""
Prediction business logic - Minimal version for lstm_performance
"""

from config import Config
from flask import current_app

# Global state (will be set by model_loader)
directional_models = {}
directional_scalers = {}

def get_directional_prediction(station_name, direction, target_datetime=None):
    """Get directional prediction - works with app config or direct models"""
    
    if target_datetime is None:
        target_datetime = Config.get_current_time()
    
    # First, try to get from app config (set by api_predict.py)
    if hasattr(current_app, 'config') and 'GET_DIRECTIONAL_PREDICTION' in current_app.config:
        return current_app.config['GET_DIRECTIONAL_PREDICTION'](station_name, direction, target_datetime)
    
    # Otherwise, use direct models if available
    model_key = f"{station_name}_{direction}"
    
    # Check operating hours
    hour = target_datetime.hour
    minute = target_datetime.minute
    time_decimal = hour + minute / 60
    
    OPERATING_START = 4.5
    OPERATING_END = 22.5
    
    if time_decimal < OPERATING_START or time_decimal >= OPERATING_END:
        return 0
    
    # If no models, return time-based fallback
    if model_key not in directional_models:
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            return 65
        elif 10 <= hour <= 16:
            return 45
        else:
            return 25
    
    try:
        from services.feature_engineering import get_feature_sequence_for_station, MRT3_PLATFORM_CAPACITY
        
        features = get_feature_sequence_for_station(station_name, direction, target_datetime)
        
        if features is None:
            raise ValueError("No features")
        
        feature_scaler = directional_scalers.get(f'{model_key}_feature')
        target_scaler = directional_scalers.get(f'{model_key}_target')
        
        if feature_scaler is None or target_scaler is None:
            raise ValueError("No scalers")
        
        # Features are already scaled, just reshape
        if features.ndim == 2:
            input_sequence = features.reshape(1, 24, -1)
        else:
            input_sequence = features
        
        pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
        predicted_passengers = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
        
        # Convert to congestion
        capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
        congestion = (predicted_passengers / capacity * 100)
        congestion = max(0, min(100, congestion))
        
        return congestion
        
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        # Fallback
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            return 65
        return 35

def get_station_prediction(station_name, target_datetime=None):
    """Get average congestion for a station"""
    north = get_directional_prediction(station_name, 'Northbound', target_datetime)
    south = get_directional_prediction(station_name, 'Southbound', target_datetime)
    return (north + south) / 2

def get_wait_time(congestion):
    if congestion > 80:
        return "15-20 min"
    elif congestion > 60:
        return "10-15 min"
    elif congestion > 30:
        return "5-10 min"
    return "2-5 min"

def get_best_time_to_travel(station_name=None):
    now = Config.get_current_time()
    hour = now.hour
    if 7 <= hour <= 9:
        return "10:00 AM - 3:00 PM (Avoid morning rush hour)"
    elif 17 <= hour <= 20:
        return "Before 5:00 PM or after 8:00 PM (Avoid evening rush hour)"
    return "Now is a good time to travel!"

def get_fallback_directional_prediction(station_name, direction, target_datetime=None):
    return get_directional_prediction(station_name, direction, target_datetime)

def clamp_prediction_by_time(congestion, target_datetime):
    return congestion

print("=" * 50)
print("✅ predictor.py loaded (minimal version for lstm_performance)")
print("=" * 50)