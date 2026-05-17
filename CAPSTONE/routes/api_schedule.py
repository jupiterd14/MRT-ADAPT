from flask import Blueprint, request, jsonify
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional

api_schedule_bp = Blueprint('api_schedule', __name__)

# ========== MRT-3 SCHEDULE FUNCTIONS ==========

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

# Weekday Headway Schedule
WEEKDAY_HEADWAY = {
    "early_morning": (time(4, 30), time(7, 0), 7, 4),
    "am_peak": (time(7, 1), time(9, 0), 3.5, 3.5),
    "off_peak": (time(9, 1), time(17, 0), 5, 5.5),
    "pm_peak": (time(17, 1), time(19, 0), 3.5, 3.5),
    "night": (time(19, 1), time(21, 30), 5, 8),
    "extended": (time(21, 31), time(23, 40), 15, 15)
}

# Weekend Headway Schedule
WEEKEND_HEADWAY = {
    "saturday_morning": (time(4, 30), time(17, 0), 5.5, 6),
    "saturday_afternoon": (time(17, 1), time(19, 0), 5, 5.5),
    "saturday_night": (time(19, 1), time(22, 40), 6.5, 7),
    "sunday": (time(4, 30), time(22, 40), 6.5, 7)
}

STATION_BASE_CAPACITY = {
    "North Ave": 12000, "Quezon Ave": 9000, "Kamuning": 7500, "Cubao": 15000,
    "Santolan": 8000, "Ortigas": 9500, "Shaw Blvd": 11000, "Boni Ave": 8500,
    "Guadalupe": 10000, "Buendia": 9000, "Ayala Ave": 14000, "Magallanes": 9000, "Taft": 16000
}

def get_station_prediction(station_name):
    from flask import current_app
    if hasattr(current_app, 'config') and 'GET_STATION_PREDICTION' in current_app.config:
        return current_app.config['GET_STATION_PREDICTION'](station_name)
    return STATION_BASE_CAPACITY.get(station_name, 10000) * 0.5

def get_current_period() -> Tuple[str, bool]:
    now = datetime.now()
    current_time = now.time()
    is_weekend = now.weekday() >= 5
    is_sunday = now.weekday() == 6
    
    if not is_weekend:
        for period, (start, end, min_h, max_h) in WEEKDAY_HEADWAY.items():
            if start <= current_time < end:
                return period, False
    else:
        if is_sunday:
            return "sunday", True
        else:
            for period, (start, end, min_h, max_h) in WEEKEND_HEADWAY.items():
                if start <= current_time < end:
                    return period, True
        return "saturday_night", True
    return "night", False

def get_headway_info(period_key: str = None) -> Dict:
    period, is_weekend = get_current_period()
    
    if not is_weekend:
        if period == "early_morning":
            return {"headway": 420, "status": "normal", "message": "Early morning service: trains every 7 minutes"}
        elif period == "am_peak":
            return {"headway": 210, "status": "peak", "message": "AM Peak hour: trains every 3.5 minutes"}
        elif period == "off_peak":
            return {"headway": 300, "status": "normal", "message": "Off-peak: trains every 5-5.5 minutes"}
        elif period == "pm_peak":
            return {"headway": 210, "status": "peak", "message": "PM Peak hour: trains every 3.5 minutes"}
        elif period == "night":
            return {"headway": 360, "status": "normal", "message": "Night service: trains every 5-8 minutes"}
        else:
            return {"headway": 900, "status": "limited", "message": "Extended hours: trains every 15 minutes"}
    else:
        if period == "saturday_morning":
            return {"headway": 330, "status": "normal", "message": "Saturday morning: trains every 5.5-6 minutes"}
        elif period == "saturday_afternoon":
            return {"headway": 300, "status": "normal", "message": "Saturday afternoon: trains every 5-5.5 minutes"}
        else:
            return {"headway": 390, "status": "normal", "message": "Weekend service: trains every 6.5-7 minutes"}

def calculate_next_trains(station_name: str, target_time: datetime = None) -> Dict:
    if target_time is None:
        target_time = datetime.now()
    
    current_time = target_time.time()
    
    if current_time < time(4, 30) or current_time >= time(23, 40):
        return {"is_operating": False, "next_open": "4:30 AM"}
    
    headway_info = get_headway_info()
    headway_minutes = max(2, headway_info["headway"] // 60)
    
    try:
        station_idx = STATIONS.index(station_name)
    except ValueError:
        station_idx = 0
    
    southbound_travel_time = station_idx * 3
    southbound_next = (headway_minutes - (target_time.minute % headway_minutes)) % headway_minutes
    southbound_minutes = max(1, southbound_next + southbound_travel_time)
    
    northbound_travel_time = (len(STATIONS) - 1 - station_idx) * 3
    northbound_next = (headway_minutes - (target_time.minute % headway_minutes)) % headway_minutes
    northbound_minutes = max(1, northbound_next + northbound_travel_time)
    
    return {
        "is_operating": True,
        "headway": headway_minutes,
        "northbound": {"minutes": northbound_minutes, "from_station": "Taft" if station_idx < len(STATIONS)-1 else "North Ave"},
        "southbound": {"minutes": southbound_minutes, "from_station": "North Ave" if station_idx > 0 else "Taft"}
    }

def get_trip_schedule(from_station: str, to_station: str, departure_time: datetime = None) -> Dict:
    if departure_time is None:
        departure_time = datetime.now()
    
    try:
        from_idx = STATIONS.index(from_station)
        to_idx = STATIONS.index(to_station)
    except ValueError:
        return {"error": "Invalid station name"}
    
    if from_idx == to_idx:
        return {"error": "Start and destination stations are the same"}
    
    num_stops = abs(to_idx - from_idx)
    base_travel_time = num_stops * 3 + 2
    
    headway_info = get_headway_info()
    headway = headway_info["headway"] // 60
    
    minute = departure_time.minute
    wait_time = (headway - (minute % headway)) % headway
    wait_time = max(2, wait_time)
    
    total_duration = wait_time + base_travel_time
    arrival_time = departure_time + timedelta(minutes=total_duration)
    
    return {
        "from_station": from_station,
        "to_station": to_station,
        "stops": num_stops,
        "wait_time": wait_time,
        "travel_time": base_travel_time,
        "total_duration": total_duration,
        "departure": departure_time.strftime("%I:%M %p"),
        "arrival": arrival_time.strftime("%I:%M %p"),
        "headway": headway,
        "status": headway_info["status"]
    }

def get_all_trains_for_station(station_name: str) -> Dict:
    next_trains = calculate_next_trains(station_name)
    
    if not next_trains.get("is_operating", True):
        return {"is_operating": False, "error": "Station closed"}
    
    north_trains = []
    south_trains = []
    headway = next_trains["headway"]
    now = datetime.now()
    
    for i in range(3):
        north_minutes = next_trains["northbound"]["minutes"] + (i * headway)
        north_trains.append({
            "minutes": north_minutes,
            "time": (now + timedelta(minutes=north_minutes)).strftime("%I:%M %p"),
            "from_station": next_trains["northbound"]["from_station"]
        })
        
        south_minutes = next_trains["southbound"]["minutes"] + (i * headway)
        south_trains.append({
            "minutes": south_minutes,
            "time": (now + timedelta(minutes=south_minutes)).strftime("%I:%M %p"),
            "from_station": next_trains["southbound"]["from_station"]
        })
    
    return {
        "station": station_name,
        "is_operating": True,
        "headway": headway,
        "trains": {"northbound": north_trains, "southbound": south_trains},
        "status": get_headway_info()["status"]
    }

# ========== ROUTES ==========

@api_schedule_bp.route('/schedule/headway')
def get_headway_route():
    try:
        headway_info = get_headway_info()
        return jsonify({
            "headway": headway_info["headway"] // 60,
            "status": headway_info["status"],
            "message": headway_info["message"]
        })
    except Exception as e:
        return jsonify({"headway": 5, "status": "normal", "message": "Normal service"})

@api_schedule_bp.route('/schedule/next-trains/<station_name>')
def get_next_trains_route(station_name):
    name = station_name.replace('%20', ' ')
    
    try:
        trains = calculate_next_trains(name)
        
        if not trains.get("is_operating"):
            return jsonify({
                "is_operating": False,
                "northbound": {"minutes": None, "origin": None},
                "southbound": {"minutes": None, "origin": None},
                "headway": 0
            })
        
        return jsonify({
            "is_operating": True,
            "northbound": {"minutes": trains["northbound"]["minutes"], "origin": trains["northbound"]["from_station"]},
            "southbound": {"minutes": trains["southbound"]["minutes"], "origin": trains["southbound"]["from_station"]},
            "headway": trains["headway"]
        })
    except Exception as e:
        return jsonify({
            "is_operating": True,
            "northbound": {"minutes": 5, "origin": "Taft"},
            "southbound": {"minutes": 3, "origin": "North Ave"},
            "headway": 5
        })