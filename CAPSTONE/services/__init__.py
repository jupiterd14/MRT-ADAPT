"""
Services module - Business logic with no Flask dependencies
"""

# Import from predictor (needed for lstm_performance)
from .predictor import (
    get_directional_prediction, 
    get_station_prediction, 
    get_fallback_directional_prediction, 
    clamp_prediction_by_time, 
    get_wait_time, 
    get_best_time_to_travel
)

# Import from feature_engineering
from .feature_engineering import (
    get_feature_sequence_for_station,
    add_cyclical_time_features,
    add_smart_operating_flags,
    infer_direction,
    is_christmas_season,
    is_payday,
    is_friday
)

# Import from model_loader
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

# Import from lstm_performance
from .lstm_performance import LSTMPerformanceService

__all__ = [
    'get_directional_prediction', 
    'get_station_prediction', 
    'get_fallback_directional_prediction', 
    'clamp_prediction_by_time', 
    'get_wait_time', 
    'get_best_time_to_travel',
    'get_feature_sequence_for_station',
    'add_cyclical_time_features',
    'add_smart_operating_flags',
    'infer_direction',
    'is_christmas_season',
    'is_payday',
    'is_friday',
    'load_directional_models', 
    'load_real_historical_data',
    'directional_models', 
    'directional_scalers', 
    'historical_entry', 
    'historical_exit',
    'hourly_avg_entry', 
    'hourly_avg_exit',
    'LSTMPerformanceService'
]

print("=" * 50)
print("✅ services module loaded")
print("=" * 50)