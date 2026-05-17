# services/__init__.py
"""
Services module - Business logic with no Flask dependencies
"""
from .predictor import (
    get_directional_prediction, 
    get_station_prediction, 
    get_fallback_directional_prediction, 
    clamp_prediction_by_time, 
    get_wait_time, 
    get_best_time_to_travel
)
from .feature_engineering import (
    get_feature_sequence_for_station, 
    add_cyclical_time_features_for_prediction, 
    add_smart_operating_flags_for_prediction
)
from .model_loader import (
    load_directional_models, 
    load_real_historical_data,
    directional_models, 
    directional_scalers, 
    historical_entry, 
    historical_exit,
    hourly_avg_entry, 
    hourly_avg_exit
)

# Remove these if they don't exist in model_loader.py:
# get_station_time_series, get_real_time_prediction

__all__ = [
    'get_directional_prediction', 
    'get_station_prediction', 
    'get_fallback_directional_prediction', 
    'clamp_prediction_by_time', 
    'get_wait_time', 
    'get_best_time_to_travel',
    'get_feature_sequence_for_station', 
    'add_cyclical_time_features_for_prediction', 
    'add_smart_operating_flags_for_prediction',
    'load_directional_models', 
    'load_real_historical_data',
    'directional_models', 
    'directional_scalers', 
    'historical_entry', 
    'historical_exit',
    'hourly_avg_entry', 
    'hourly_avg_exit'
]