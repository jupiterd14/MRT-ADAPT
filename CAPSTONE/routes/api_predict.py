from flask import Blueprint, request, jsonify, session
from datetime import datetime, timedelta
import numpy as np

api_predict_bp = Blueprint('api_predict', __name__)

# These will be imported from main app or services
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

STATION_BASE_CAPACITY = {
    "North Ave": 12000, "Quezon Ave": 9000, "Kamuning": 7500, "Cubao": 15000,
    "Santolan": 8000, "Ortigas": 9500, "Shaw Blvd": 11000, "Boni Ave": 8500,
    "Guadalupe": 10000, "Buendia": 9000, "Ayala Ave": 14000, "Magallanes": 9000, "Taft": 16000
}

def get_directional_prediction(station_name, direction, target_datetime=None):
    """Will be set in app.py - placeholder"""
    from flask import current_app
    if hasattr(current_app, 'config') and 'GET_DIRECTIONAL_PREDICTION' in current_app.config:
        return current_app.config['GET_DIRECTIONAL_PREDICTION'](station_name, direction, target_datetime)
    return 50

def get_station_prediction(station_name):
    """Get station-level prediction (average of both directions)"""
    from flask import current_app
    if hasattr(current_app, 'config') and 'GET_STATION_PREDICTION' in current_app.config:
        return current_app.config['GET_STATION_PREDICTION'](station_name)
    
    now = datetime.now()
    north = get_directional_prediction(station_name, 'Northbound', now)
    south = get_directional_prediction(station_name, 'Southbound', now)
    return (north + south) / 2

@api_predict_bp.route('/predict/<station_name>')
def predict_congestion(station_name):
    name = station_name.replace('%20', ' ')
    
    date_param = request.args.get('date')
    time_param = request.args.get('time')
    
    if date_param and time_param:
        try:
            year, month, day = map(int, date_param.split('-'))
            hour, minute = map(int, time_param.split(':'))
            target_datetime = datetime(year, month, day, hour, minute)
            ridership = get_station_prediction_for_datetime(name, target_datetime)
        except:
            ridership = get_station_prediction(name)
    else:
        ridership = get_station_prediction(name)
    
    capacity = STATION_BASE_CAPACITY.get(name, 10000)
    congestion = min(100, int((ridership / capacity) * 100))
    
    if congestion > 80: status = "CRITICAL"
    elif congestion > 50: status = "BUSY"
    elif congestion > 20: status = "MODERATE"
    else: status = "LIGHT"
    
    return jsonify({
        "station": name, "ridership": ridership,
        "congestion": congestion, "status": status
    })
# routes/api_predict.py - Update the directional_forecast function
@api_predict_bp.route('/directional-forecast/<station_name>')
def directional_forecast(station_name):
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
    
    # Generate forecasts for current + next 5 hours (total 6)
    forecasts = []
    
    # Add current hour as first forecast
    current_north = get_directional_prediction_direct(name, 'Northbound', target_datetime)
    current_south = get_directional_prediction_direct(name, 'Southbound', target_datetime)
    
    forecasts.append({
        'hour': target_datetime.hour,
        'time': target_datetime.strftime('%I:%M %p'),
        'northbound': round(current_north, 1),
        'southbound': round(current_south, 1)
    })
    
    # Add next 5 hours
    for i in range(1, 6):
        forecast_time = target_datetime + timedelta(hours=i)
        
        north = get_directional_prediction_direct(name, 'Northbound', forecast_time)
        south = get_directional_prediction_direct(name, 'Southbound', forecast_time)
        
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

def get_directional_prediction_direct(station_name, direction, target_datetime=None, for_passenger_display=True):
    """Get prediction with optional passenger display mode"""
    from services.model_loader import directional_models, directional_scalers
    from services import get_feature_sequence_for_station
    
    if target_datetime is None:
        target_datetime = datetime.now()
    
    model_key = f"{station_name}_{direction}"
    
    # Get model prediction (always get the real value first)
    prediction = 35  # default fallback
    
    if model_key in directional_models:
        try:
            sequence = get_feature_sequence_for_station(station_name, direction, target_datetime)
            
            if sequence is not None and len(sequence) == 24:
                feature_scaler = directional_scalers.get(f'{model_key}_feature')
                target_scaler = directional_scalers.get(f'{model_key}_target')
                
                if feature_scaler and target_scaler:
                    scaled_sequence = feature_scaler.transform(sequence)
                    input_sequence = scaled_sequence.reshape(1, 24, -1)
                    pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
                    prediction = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                    prediction = max(0, min(100, prediction))
        except Exception as e:
            print(f"⚠️ Prediction error for {model_key}: {e}")
    else:
        # Fallback based on time of day
        hour = target_datetime.hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            prediction = 65
        elif 10 <= hour <= 16:
            prediction = 45
        else:
            prediction = 25
    
    # Apply passenger display override if requested
    if for_passenger_display:
        hour = target_datetime.hour
        minute = target_datetime.minute
        current_time = hour + minute / 60
        
        OPERATING_START = 4.5   # 4:30 AM - First train
        OPERATING_END = 21.5    # 9:30 PM - Last train (adjust as needed)
        LAST_TRAIN = 21.5       # Last train departure
        
        # For passenger display, show 0 outside service hours
        if current_time < OPERATING_START or current_time > LAST_TRAIN:
            return 0
        
        # Optionally reduce late-night values (9:30 PM - 10:30 PM)
        if current_time > LAST_TRAIN and current_time < OPERATING_END:
            return int(prediction * 0.3)  # Show reduced activity
    
    return prediction

@api_predict_bp.route('/station-forecast/<station_name>')
def station_forecast_api(station_name):
    name = station_name.replace('%20', ' ')
    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute
    
    OPERATING_START = 4.5
    OPERATING_END = 22.5
    current_time = current_hour + current_minute / 60
    
    if current_time < OPERATING_START or current_time >= OPERATING_END:
        return jsonify({
            "station": name,
            "forecast": [5, 5, 5, 10, 15, 20],
            "intervals": ["+1h", "+2h", "+3h", "+4h", "+5h", "+6h"],
            "current": 5,
            "data_source": "Station Closed - Forecast for opening hours",
            "operating_hours": "4:30 AM - 10:30 PM"
        })
    
    current_ridership = get_station_prediction(name)
    capacity = STATION_BASE_CAPACITY.get(name, 10000)
    current_congestion = min(100, int((current_ridership / capacity) * 100))
    
    forecast = []
    prev_ridership = current_ridership
    
    for i in range(6):
        forecast_time = now + timedelta(hours=i+1)
        forecast_hour = forecast_time.hour
        
        if 7 <= forecast_hour <= 9:
            multiplier = 1.2
        elif 17 <= forecast_hour <= 20:
            multiplier = 1.15
        elif forecast_hour <= 6 or forecast_hour >= 22:
            multiplier = 0.7
        else:
            multiplier = 0.95
        
        forecast_ridership = int(prev_ridership * multiplier)
        forecast_ridership = max(50, min(forecast_ridership, capacity))
        forecast_congestion = min(100, int((forecast_ridership / capacity) * 100))
        prev_ridership = forecast_ridership
        forecast.append(forecast_congestion)
    
    return jsonify({
        "station": name, "forecast": forecast,
        "intervals": ["+1h", "+2h", "+3h", "+4h", "+5h", "+6h"],
        "current": current_congestion,
        "data_source": "LSTM Model + Historical Patterns",
        "operating_hours": "4:30 AM - 10:30 PM"
    })

@api_predict_bp.route('/station-forecast-badge/<station_name>')
def station_forecast_api_badge(station_name):
    name = station_name.replace('%20', ' ')
    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute
    
    OPERATING_START = 4.5
    OPERATING_END = 22.5
    current_time = current_hour + current_minute / 60
    
    current_ridership = get_station_prediction(name)
    capacity = STATION_BASE_CAPACITY.get(name, 10000)
    current_congestion = min(100, int((current_ridership / capacity) * 100))
    
    station_idx = STATIONS.index(name) if name in STATIONS else 0
    is_morning_rush = 7 <= current_hour <= 9
    is_evening_rush = 17 <= current_hour <= 20
    
    north_congestion = current_congestion
    south_congestion = current_congestion
    
    if current_congestion >= 70:
        if is_morning_rush:
            if station_idx <= 6:
                south_congestion = current_congestion
                north_congestion = max(40, int(current_congestion * 0.6))
            else:
                north_congestion = current_congestion
                south_congestion = max(40, int(current_congestion * 0.6))
        elif is_evening_rush:
            if station_idx <= 6:
                north_congestion = current_congestion
                south_congestion = max(40, int(current_congestion * 0.6))
            else:
                south_congestion = current_congestion
                north_congestion = max(40, int(current_congestion * 0.6))
    
    if current_time < OPERATING_START or current_time >= OPERATING_END:
        return jsonify({
            "northbound": {"station": name, "forecast": [5, 5, 5, 10, 15, 20], "current": 5,
                          "origin": "northbound", "operating_hours": "4:30 AM - 10:30 PM"},
            "southbound": {"station": name, "forecast": [5, 5, 5, 10, 15, 20], "current": 5,
                          "origin": "southbound", "operating_hours": "4:30 AM - 10:30 PM"}
        })
    
    forecast = []
    prev_ridership = current_ridership
    
    for i in range(6):
        forecast_time = now + timedelta(hours=i+1)
        forecast_hour = forecast_time.hour
        
        if 7 <= forecast_hour <= 9:
            multiplier = 1.2
        elif 17 <= forecast_hour <= 20:
            multiplier = 1.15
        elif forecast_hour <= 6 or forecast_hour >= 22:
            multiplier = 0.7
        else:
            multiplier = 0.95
        
        forecast_ridership = int(prev_ridership * multiplier)
        forecast_ridership = max(50, min(forecast_ridership, capacity))
        forecast_congestion = min(100, int((forecast_ridership / capacity) * 100))
        prev_ridership = forecast_ridership
        forecast.append(forecast_congestion)
    
    return jsonify({
        "northbound": {"station": name, "forecast": forecast, "current": north_congestion,
                      "origin": "northbound", "operating_hours": "4:30 AM - 10:30 PM"},
        "southbound": {"station": name, "forecast": forecast, "current": south_congestion,
                      "origin": "southbound", "operating_hours": "4:30 AM - 10:30 PM"}
    })

@api_predict_bp.route('/batch-predict')
def batch_predict():
    results = []
    for station in STATIONS:
        ridership = get_station_prediction(station)
        capacity = STATION_BASE_CAPACITY.get(station, 10000)
        congestion = int((ridership / capacity) * 100)
        congestion = max(0, min(100, congestion))
        
        if congestion >= 75: status = "🔴 CRITICAL"
        elif congestion >= 55: status = "🟠 BUSY"
        elif congestion >= 30: status = "🟡 MODERATE"
        else: status = "🟢 LIGHT"
        
        results.append({"station": station, "ridership": ridership,
                       "congestion": congestion, "status": status})
    
    return jsonify(results)

@api_predict_bp.route('/predict-direction/<station_name>')
def predict_direction(station_name):
    name = station_name.replace('%20', ' ')
    
    ridership = get_station_prediction(name)
    capacity = STATION_BASE_CAPACITY.get(name, 10000)
    congestion = min(100, int((ridership / capacity) * 100))
    
    station_idx = STATIONS.index(name) if name in STATIONS else 0
    
    if station_idx < 6:
        direction = "southbound"
        next_station = STATIONS[station_idx + 1] if station_idx + 1 < len(STATIONS) else STATIONS[0]
    elif station_idx > 6:
        direction = "northbound"
        next_station = STATIONS[station_idx - 1] if station_idx - 1 >= 0 else STATIONS[-1]
    else:
        direction = "both"
        next_station = STATIONS[station_idx + 1] if station_idx + 1 < len(STATIONS) else STATIONS[0]
    
    if congestion > 80: status = "SEVERELY CONGESTED"; color = "critical"; wait_time = "15-20 min"
    elif congestion > 60: status = "CONGESTED"; color = "congested"; wait_time = "10-15 min"
    elif congestion > 30: status = "MODERATE"; color = "moderate"; wait_time = "5-10 min"
    else: status = "LIGHT"; color = "light"; wait_time = "2-5 min"
    
    return jsonify({
        "station": name, "congestion": congestion, "status": status, "color": color,
        "direction": direction, "next_station": next_station,
        "ridership": ridership, "capacity": capacity, "wait_time": wait_time
    })

@api_predict_bp.route('/predict-route')
def predict_route():
    from_station = request.args.get('from')
    to_station = request.args.get('to')
    date = request.args.get('date')
    time = request.args.get('time')
    
    if not from_station or not to_station:
        return jsonify({"error": "Missing station parameters"}), 400
    
    def get_prediction_for_station(station, dt=None):
        if dt:
            return get_station_prediction_for_datetime(station, dt) if 'get_station_prediction_for_datetime' in globals() else get_station_prediction(station)
        return get_station_prediction(station)
    
    if date and time:
        year, month, day = map(int, date.split('-'))
        hour, minute = map(int, time.split(':'))
        target_datetime = datetime(year, month, day, hour, minute)
        ridership_from = get_prediction_for_station(from_station, target_datetime)
        ridership_to = get_prediction_for_station(to_station, target_datetime)
    else:
        ridership_from = get_station_prediction(from_station)
        ridership_to = get_station_prediction(to_station)
    
    capacity_from = STATION_BASE_CAPACITY.get(from_station, 10000)
    capacity_to = STATION_BASE_CAPACITY.get(to_station, 10000)
    
    congestion_from = min(100, int((ridership_from / capacity_from) * 100))
    congestion_to = min(100, int((ridership_to / capacity_to) * 100))
    avg_congestion = (congestion_from + congestion_to) / 2
    
    from_idx = STATIONS.index(from_station) if from_station in STATIONS else 0
    to_idx = STATIONS.index(to_station) if to_station in STATIONS else len(STATIONS) - 1
    station_diff = abs(from_idx - to_idx)
    travel_time = station_diff * 3 + 5
    
    if avg_congestion > 80: status = "CRITICAL"; recommendation = "Consider postponing your trip"
    elif avg_congestion > 60: status = "HEAVY"; recommendation = "Allow extra time for your journey"
    elif avg_congestion > 30: status = "MODERATE"; recommendation = "Normal travel conditions"
    else: status = "LIGHT"; recommendation = "Good time to travel!"
    
    return jsonify({
        "from_station": from_station, "to_station": to_station,
        "from_congestion": congestion_from, "to_congestion": congestion_to,
        "avg_congestion": round(avg_congestion, 1), "status": status,
        "travel_time": travel_time, "stations_between": station_diff, "recommendation": recommendation
    })

def get_station_prediction_for_datetime(station_name, target_datetime):
    """Get prediction for specific datetime"""
    return get_station_prediction(station_name)