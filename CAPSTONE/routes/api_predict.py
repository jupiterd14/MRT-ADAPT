"""
PREDICTION API - ML Model Endpoints Only
Purpose: Raw congestion predictions from ML models
Use for: Forecasts, batch predictions, route planning
NOT for: Live map display, alerts, broadcasts
"""

from flask import Blueprint, request, jsonify, session, current_app
from datetime import datetime, timedelta
import numpy as np

api_predict_bp = Blueprint('api_predict', __name__)

# These will be imported from main app or services
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

# ========== HELPER FUNCTIONS ==========
def get_directional_prediction(station_name, direction, target_datetime=None):
    """Get directional congestion prediction directly from model"""
    # Import from main app's cached models
    from flask import current_app
    
    print(f"[API] Called for {station_name} {direction}")
    
    # Try to get the prediction function from app config (set in main app.py)
    prediction_func = current_app.config.get('GET_DIRECTIONAL_PREDICTION')
    
    if prediction_func:
        try:
            result = prediction_func(station_name, direction, target_datetime)
            print(f"[API] Real prediction: {result:.1f}%")
            return result
        except Exception as e:
            print(f"[API] Error with prediction func: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[API] No prediction function found in config")
        
        # Fallback: Try to import directly from services (should have cached models)
        try:
            from services import get_directional_prediction as real_prediction
            from services.model_loader import directional_models, directional_scalers
            from services import get_feature_sequence_for_station
            
            result = real_prediction(
                station_name, direction, target_datetime,
                directional_models, directional_scalers,
                get_feature_sequence_for_station
            )
            print(f"[API] Direct import prediction: {result:.1f}%")
            return result
        except Exception as e2:
            print(f"[API] Direct import failed: {e2}")
    
    # Fallback based on time of day
    print(f"[API] Using TIME-BASED FALLBACK")
    if target_datetime is None:
        target_datetime = datetime.now()
    hour = target_datetime.hour
    
    # More realistic fallback based on time
    if 7 <= hour <= 9:  # Morning rush
        return 75 + (hour - 7) * 5
    elif 17 <= hour <= 19:  # Evening rush
        return 70 + (hour - 17) * 5
    elif 10 <= hour <= 16:  # Midday
        return 45
    else:  # Late night
        return 25

def get_station_prediction(station_name):
    """Get average congestion for a station"""
    north = get_directional_prediction(station_name, 'Northbound')
    south = get_directional_prediction(station_name, 'Southbound')
    return (north + south) / 2

# ========== MAIN PREDICTION ENDPOINTS ==========

@api_predict_bp.route('/predict/<station_name>')
def predict_congestion(station_name):
    """Get current congestion for a station (direct from model)"""
    name = station_name.replace('%20', ' ')
    
    date_param = request.args.get('date')
    time_param = request.args.get('time')
    
    if date_param and time_param:
        try:
            year, month, day = map(int, date_param.split('-'))
            hour, minute = map(int, time_param.split(':'))
            target_datetime = datetime(year, month, day, hour, minute)
            north_congestion = get_directional_prediction(name, 'Northbound', target_datetime)
            south_congestion = get_directional_prediction(name, 'Southbound', target_datetime)
            congestion = (north_congestion + south_congestion) / 2
        except:
            north_congestion = get_directional_prediction(name, 'Northbound')
            south_congestion = get_directional_prediction(name, 'Southbound')
            congestion = (north_congestion + south_congestion) / 2
    else:
        north_congestion = get_directional_prediction(name, 'Northbound')
        south_congestion = get_directional_prediction(name, 'Southbound')
        congestion = (north_congestion + south_congestion) / 2
    
    # Determine status from congestion directly
    if congestion > 80: status = "CRITICAL"
    elif congestion > 50: status = "BUSY"
    elif congestion > 20: status = "MODERATE"
    else: status = "LIGHT"
    
    return jsonify({
        "station": name,
        "congestion": round(congestion, 1),
        "northbound": round(north_congestion, 1),
        "southbound": round(south_congestion, 1),
        "status": status
    })

@api_predict_bp.route('/directional-forecast/<station_name>')
def directional_forecast(station_name):
    """Get 6-hour forecast (current + next 5 hours) for both directions"""
    name = station_name.replace('%20', ' ')
    
    date_str = request.args.get('date')
    time_str = request.args.get('time')
    
    if date_str and time_str:
        try:
            year, month, day = map(int, date_str.split('-'))
            hour, minute = map(int, time_str.split(':'))
            target_datetime = datetime(year, month, day, hour, minute)
        except:
            target_datetime = datetime.now()
    else:
        target_datetime = datetime.now()
    
    forecasts = []
    
    # Current hour (hour 0)
    current_north = get_directional_prediction(name, 'Northbound', target_datetime)
    current_south = get_directional_prediction(name, 'Southbound', target_datetime)
    
    forecasts.append({
        'hour': target_datetime.hour,
        'time': target_datetime.strftime('%I:%M %p'),
        'northbound': round(current_north, 1),
        'southbound': round(current_south, 1)
    })
    
    # Next 5 hours - IMPORTANT: Create new datetime for each hour
    for i in range(1, 6):
        # Create a NEW datetime object for each forecast hour
        forecast_time = target_datetime + timedelta(hours=i)
        
        north = get_directional_prediction(name, 'Northbound', forecast_time)
        south = get_directional_prediction(name, 'Southbound', forecast_time)
        
        forecasts.append({
            'hour': forecast_time.hour,
            'time': forecast_time.strftime('%I:%M %p'),
            'northbound': round(north, 1),
            'southbound': round(south, 1)
        })
    
    return jsonify({
        'station': name,
        'forecasts': forecasts,
        'current': {
            'northbound': round(current_north, 1),
            'southbound': round(current_south, 1)
        }
    })

# ========== BATCH AND ROUTE ENDPOINTS ==========

@api_predict_bp.route('/batch-predict')
def batch_predict():
    """Get congestion for all stations at once"""
    results = []
    for station in STATIONS:
        north = get_directional_prediction(station, 'Northbound')
        south = get_directional_prediction(station, 'Southbound')
        congestion = (north + south) / 2
        
        if congestion >= 75: status = "🔴 CRITICAL"
        elif congestion >= 55: status = "🟠 BUSY"
        elif congestion >= 30: status = "🟡 MODERATE"
        else: status = "🟢 LIGHT"
        
        results.append({
            "station": station,
            "congestion": round(congestion, 1),
            "northbound": round(north, 1),
            "southbound": round(south, 1),
            "status": status
        })
    
    return jsonify(results)

@api_predict_bp.route('/predict-direction/<station_name>')
def predict_direction(station_name):
    """Get congestion with direction-specific advice"""
    name = station_name.replace('%20', ' ')
    
    # Get directional congestion directly from model
    north_congestion = get_directional_prediction(name, 'Northbound')
    south_congestion = get_directional_prediction(name, 'Southbound')
    congestion = (north_congestion + south_congestion) / 2
    
    station_idx = STATIONS.index(name) if name in STATIONS else 0
    
    # Determine dominant direction based on station position
    if station_idx < 6:
        direction = "southbound"
        next_station = STATIONS[station_idx + 1] if station_idx + 1 < len(STATIONS) else STATIONS[0]
    elif station_idx > 6:
        direction = "northbound"
        next_station = STATIONS[station_idx - 1] if station_idx - 1 >= 0 else STATIONS[-1]
    else:
        direction = "both"
        next_station = STATIONS[station_idx + 1] if station_idx + 1 < len(STATIONS) else STATIONS[0]
    
    if congestion > 80: 
        status = "SEVERELY CONGESTED"
        color = "critical"
        wait_time = "15-20 min"
    elif congestion > 60: 
        status = "CONGESTED"
        color = "congested"
        wait_time = "10-15 min"
    elif congestion > 30: 
        status = "MODERATE"
        color = "moderate"
        wait_time = "5-10 min"
    else: 
        status = "LIGHT"
        color = "light"
        wait_time = "2-5 min"
    
    return jsonify({
        "station": name,
        "congestion": round(congestion, 1),
        "northbound": round(north_congestion, 1),
        "southbound": round(south_congestion, 1),
        "status": status,
        "color": color,
        "direction": direction,
        "next_station": next_station,
        "wait_time": wait_time
    })

@api_predict_bp.route('/predict-route')
def predict_route():
    """Get route prediction between two stations"""
    from_station = request.args.get('from')
    to_station = request.args.get('to')
    date = request.args.get('date')
    time = request.args.get('time')
    
    if not from_station or not to_station:
        return jsonify({"error": "Missing station parameters"}), 400
    
    if date and time:
        try:
            year, month, day = map(int, date.split('-'))
            hour, minute = map(int, time.split(':'))
            target_datetime = datetime(year, month, day, hour, minute)
            north_from = get_directional_prediction(from_station, 'Northbound', target_datetime)
            south_from = get_directional_prediction(from_station, 'Southbound', target_datetime)
            north_to = get_directional_prediction(to_station, 'Northbound', target_datetime)
            south_to = get_directional_prediction(to_station, 'Southbound', target_datetime)
            congestion_from = (north_from + south_from) / 2
            congestion_to = (north_to + south_to) / 2
        except:
            congestion_from = get_station_prediction(from_station)
            congestion_to = get_station_prediction(to_station)
    else:
        congestion_from = get_station_prediction(from_station)
        congestion_to = get_station_prediction(to_station)
    
    avg_congestion = (congestion_from + congestion_to) / 2
    
    from_idx = STATIONS.index(from_station) if from_station in STATIONS else 0
    to_idx = STATIONS.index(to_station) if to_station in STATIONS else len(STATIONS) - 1
    station_diff = abs(from_idx - to_idx)
    travel_time = station_diff * 3 + 5
    
    if avg_congestion > 80: 
        status = "CRITICAL"
        recommendation = "Consider postponing your trip"
    elif avg_congestion > 60: 
        status = "HEAVY"
        recommendation = "Allow extra time for your journey"
    elif avg_congestion > 30: 
        status = "MODERATE"
        recommendation = "Normal travel conditions"
    else: 
        status = "LIGHT"
        recommendation = "Good time to travel!"
    
    return jsonify({
        "from_station": from_station,
        "to_station": to_station,
        "from_congestion": round(congestion_from, 1),
        "to_congestion": round(congestion_to, 1),
        "avg_congestion": round(avg_congestion, 1),
        "status": status,
        "travel_time": travel_time,
        "stations_between": station_diff,
        "recommendation": recommendation
    })
