"""
Prediction business logic - Re-exports from fixed api_predict.py
"""

from config import Config
from flask import current_app

# ============================================================
# LAZY IMPORTS - Only import when functions are called
# ============================================================

def get_directional_prediction(station_name, direction, target_datetime=None):
    """Get directional prediction - lazy import to avoid cache reload"""
    from routes.api_predict import get_directional_prediction as _get
    return _get(station_name, direction, target_datetime)

def get_station_prediction(station_name, target_datetime=None):
    """Get average congestion for a station"""
    from routes.api_predict import get_station_prediction as _get
    return _get(station_name, target_datetime)

def get_fallback_directional_prediction(station_name, direction, target_datetime=None):
    """Fallback prediction"""
    from routes.api_predict import get_fallback_directional_prediction as _get
    return _get(station_name, direction, target_datetime)

def clamp_prediction_by_time(congestion, target_datetime):
    """Clamp prediction by time"""
    from routes.api_predict import clamp_prediction_by_time as _clamp
    return _clamp(congestion, target_datetime)

def get_wait_time(congestion):
    """Get wait time based on congestion"""
    from routes.api_predict import get_wait_time as _get
    return _get(congestion)

def get_best_time_to_travel(station_name=None):
    """Get best time to travel"""
    from routes.api_predict import get_best_time_to_travel as _get
    return _get(station_name)

directional_models = {}
directional_scalers = {}

__all__ = [
    'get_directional_prediction',
    'get_station_prediction',
    'get_fallback_directional_prediction',
    'clamp_prediction_by_time',
    'get_wait_time',
    'get_best_time_to_travel',
    'directional_models',
    'directional_scalers'
]

print("=" * 50)
print("✅ predictor.py loaded - Lazy imports (no cache reload)")
print("=" * 50)