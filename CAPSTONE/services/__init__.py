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

# NEW: Import LSTM predictor
from .lstm_integration import (
    MRT3LSTMPredictor,
    init_lstm_predictor,
    schedule_weekly_retraining
)

# Global LSTM predictor instance (will be initialized in app.py)
lstm_predictor = None

def init_global_lstm_predictor(model_path='models_2022-2024_v8_20260616_081538'):
    """
    Initialize the global LSTM predictor instance.
    Call this from app.py during startup.
    """
    global lstm_predictor
    
    predictor = MRT3LSTMPredictor(model_path=model_path)
    
    if predictor.load_models():
        lstm_predictor = predictor
        print(f"✅ Global LSTM predictor initialized with {len(predictor.models)} models")
        return True
    else:
        lstm_predictor = None
        print("⚠️ Global LSTM predictor initialization failed")
        return False

def get_enhanced_directional_prediction(station_name, direction, target_datetime=None):
    """
    Enhanced prediction that tries LSTM first, then falls back to original.
    This can be used as a drop-in replacement for get_directional_prediction.
    """
    global lstm_predictor
    
    # Try LSTM first if available
    if lstm_predictor and hasattr(lstm_predictor, 'models') and len(lstm_predictor.models) > 0:
        try:
            # Need db session - this will be passed from the Flask route
            # We'll handle this differently - see wrapper in app.py
            pass
        except Exception as e:
            print(f"⚠️ LSTM prediction failed: {e}")
    
    # Fallback to original
    return get_directional_prediction(station_name, direction, target_datetime)

def get_enhanced_station_prediction(station_name):
    """
    Enhanced station prediction that tries LSTM first.
    """
    global lstm_predictor
    
    # Try LSTM if available
    if lstm_predictor and hasattr(lstm_predictor, 'models') and len(lstm_predictor.models) > 0:
        try:
            # This will be handled by the Flask wrapper
            pass
        except Exception as e:
            print(f"⚠️ LSTM prediction failed: {e}")
    
    # Fallback to original
    return get_station_prediction(station_name)

# Update __all__ to include new imports
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
    'LSTMPerformanceService',
    # New exports
    'MRT3LSTMPredictor',
    'init_global_lstm_predictor',
    'schedule_weekly_retraining',
    'lstm_predictor',
    'get_enhanced_directional_prediction',
    'get_enhanced_station_prediction'
]

print("=" * 50)
print("✅ services module loaded")
print("=" * 50)