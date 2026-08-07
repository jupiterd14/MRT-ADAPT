from flask import Blueprint, request, jsonify
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional

api_schedule_bp = Blueprint('api_schedule', __name__)

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

# ========== WEEKDAY ENTRANCE OPENING (SB / NB) ==========
WEEKDAY_OPENING_SB = {
    "North Ave": time(4, 20), "Quezon Ave": time(4, 22), "Kamuning": time(4, 24),
    "Cubao": time(4, 27), "Santolan": time(4, 30), "Ortigas": time(4, 33),
    "Shaw Blvd": time(4, 36), "Boni": time(4, 38), "Guadalupe": time(4, 40),
    "Buendia": time(4, 43), "Ayala": time(4, 45), "Magallanes": time(4, 47),
    "Taft": time(4, 50)
}

WEEKDAY_OPENING_NB = {
    "North Ave": time(5, 24), "Quezon Ave": time(5, 23), "Kamuning": time(5, 21),
    "Cubao": time(5, 17), "Santolan": time(5, 14), "Ortigas": time(5, 11),
    "Shaw Blvd": time(5, 10), "Boni": time(5, 7), "Guadalupe": time(5, 6),
    "Buendia": time(5, 2), "Ayala": time(5, 0), "Magallanes": time(4, 58),
    "Taft": time(4, 55)
}

# ========== WEEKEND ENTRANCE OPENING ==========
WEEKEND_OPENING_SB = {
    "North Ave": time(4, 20), "Quezon Ave": time(4, 22), "Kamuning": time(4, 24),
    "Cubao": time(4, 28), "Santolan": time(4, 31), "Ortigas": time(4, 34),
    "Shaw Blvd": time(4, 36), "Boni": time(4, 38), "Guadalupe": time(4, 40),
    "Buendia": time(4, 44), "Ayala": time(4, 46), "Magallanes": time(4, 49),
    "Taft": time(4, 52)
}

WEEKEND_OPENING_NB = {
    "North Ave": time(5, 27), "Quezon Ave": time(5, 25), "Kamuning": time(5, 22),
    "Cubao": time(5, 19), "Santolan": time(5, 15), "Ortigas": time(5, 12),
    "Shaw Blvd": time(5, 10), "Boni": time(5, 8), "Guadalupe": time(5, 6),
    "Buendia": time(5, 3), "Ayala": time(5, 1), "Magallanes": time(4, 58),
    "Taft": time(4, 55)
}

# ========== LAST TRAIN DEPARTURES ==========
# Southbound last train from each station (weekday)
WEEKDAY_LAST_SB = {
    "North Ave": time(22, 30), "Quezon Ave": time(22, 33), "Kamuning": time(22, 35),
    "Cubao": time(22, 38), "Santolan": time(22, 40), "Ortigas": time(22, 43),
    "Shaw Blvd": time(22, 48), "Boni": time(22, 48), "Guadalupe": time(22, 50),
    "Buendia": time(22, 56), "Ayala": time(22, 55), "Magallanes": time(22, 58),
    "Taft": None  # Terminal
}

# Northbound last train from each station (weekday)
WEEKDAY_LAST_NB = {
    "North Ave": None, "Quezon Ave": time(23, 37), "Kamuning": time(23, 35),
    "Cubao": time(23, 33), "Santolan": time(23, 35), "Ortigas": time(23, 26),
    "Shaw Blvd": time(23, 27), "Boni": time(23, 22), "Guadalupe": time(23, 20),
    "Buendia": time(23, 19), "Ayala": time(23, 15), "Magallanes": time(23, 14),
    "Taft": time(23, 9)
}

# Weekend last trains
WEEKEND_LAST_SB = {
    "North Ave": time(21, 30), "Quezon Ave": time(21, 32), "Kamuning": time(21, 34),
    "Cubao": time(21, 38), "Santolan": time(21, 40), "Ortigas": time(21, 44),
    "Shaw Blvd": time(21, 45), "Boni": time(21, 48), "Guadalupe": time(21, 49),
    "Buendia": time(21, 53), "Ayala": time(21, 55), "Magallanes": time(21, 57),
    "Taft": None
}

WEEKEND_LAST_NB = {
    "North Ave": None, "Quezon Ave": time(22, 37), "Kamuning": time(22, 35),
    "Cubao": time(22, 32), "Santolan": time(22, 29), "Ortigas": time(22, 26),
    "Shaw Blvd": time(22, 26), "Boni": time(22, 22), "Guadalupe": time(22, 20),
    "Buendia": time(22, 16), "Ayala": time(22, 26), "Magallanes": time(22, 12),
    "Taft": time(22, 9)
}

# ========== HEADWAY SCHEDULES ==========
# Format: (start_time, end_time, min_headway_seconds, max_headway_seconds, description)
WEEKDAY_HEADWAY = [
    (time(4, 30), time(7, 0), 7*60, 4*60, "Morning", "7-4 minutes"),
    (time(7, 1), time(9, 0), 210, 210, "AM Peak", "3.5 minutes"),
    (time(9, 1), time(17, 0), 5*60, 5.5*60, "Off Peak", "5-5.5 minutes"),
    (time(17, 1), time(19, 0), 210, 210, "PM Peak", "3.5 minutes"),
    (time(19, 1), time(21, 30), 5*60, 8*60, "Night", "5-8 minutes"),
    (time(21, 31), time(23, 40), 15*60, 15*60, "Extended", "15 minutes"),
]

WEEKEND_HEADWAY = [
    (time(4, 30), time(17, 0), 5.5*60, 6*60, "Saturday Morning", "5.5-6 minutes"),
    (time(17, 1), time(19, 0), 5*60, 5.5*60, "Saturday Afternoon", "5-5.5 minutes"),
    (time(19, 1), time(22, 40), 6.5*60, 7*60, "Saturday Night", "6.5-7 minutes"),
]

SUNDAY_HEADWAY = [
    (time(4, 30), time(22, 40), 6.5*60, 7*60, "Sunday/Holiday", "6.5-7 minutes"),
]

def is_weekend() -> bool:
    return datetime.now().weekday() >= 5

def is_sunday() -> bool:
    return datetime.now().weekday() == 6

def get_station_opening_hours(station: str, direction: str) -> Tuple[time, time]:
    """Returns (opening_time, last_train_time) for a station and direction"""
    now = datetime.now()
    weekend = is_weekend()
    sunday = is_sunday()
    
    if direction == "southbound":
        opening = WEEKEND_OPENING_SB.get(station) if weekend else WEEKDAY_OPENING_SB.get(station)
        last = WEEKEND_LAST_SB.get(station) if weekend else WEEKDAY_LAST_SB.get(station)
    else:  # northbound
        opening = WEEKEND_OPENING_NB.get(station) if weekend else WEEKDAY_OPENING_NB.get(station)
        last = WEEKEND_LAST_NB.get(station) if weekend else WEEKDAY_LAST_NB.get(station)
    
    return opening, last

def get_current_headway() -> Dict:
    """Get current headway based on time of day"""
    now = datetime.now()
    current_time = now.time()
    
    # Check if MRT is operating (4:30 AM - 11:40 PM)
    if current_time < time(4, 30) or current_time >= time(23, 40):
        return {"is_operating": False, "headway_avg": 0, "description": "Station closed"}
    
    if is_sunday():
        schedule = SUNDAY_HEADWAY
    elif is_weekend():
        schedule = WEEKEND_HEADWAY
    else:
        schedule = WEEKDAY_HEADWAY
    
    for start, end, min_h, max_h, period, desc in schedule:
        if start <= current_time < end:
            return {
                "is_operating": True,  # ADD THIS - important!
                "headway_min": int(min_h // 60),
                "headway_max": int(max_h // 60),
                "headway_avg": int((min_h + max_h) // 120),
                "period": period,
                "description": desc,
                "is_peak": "Peak" in period
            }
    
    # Default fallback - operating with normal headway
    return {
        "is_operating": True,
        "headway_avg": 5,
        "description": "Normal operations"
    }
    
@api_schedule_bp.route('/schedule/test')
def test_schedule():
    """Test endpoint to verify schedule API is working"""
    now = datetime.now()
    test_station = "North Ave"
    result = calculate_next_trains(test_station, now)
    return jsonify({
        "status": "ok",
        "timestamp": now.isoformat(),
        "test_result": result
    })
def calculate_next_trains(station: str, target_time: datetime = None) -> Dict:
    """Calculate next train arrival times for a station"""
    if target_time is None:
        target_time = datetime.now()
    
    current_time = target_time.time()
    
    # Check if station is open for each direction
    sb_open, sb_last = get_station_opening_hours(station, "southbound")
    nb_open, nb_last = get_station_opening_hours(station, "northbound")
    
    headway_info = get_current_headway()
    
    result = {
        "station": station,
        "is_operating": headway_info.get("is_operating", True),
        "is_closed": not headway_info.get("is_operating", True),
        "headway": headway_info.get("headway_avg", 5),
        "headway_description": headway_info.get("description", ""),
        "period": headway_info.get("period", ""),
        "southbound": {"available": True, "minutes": None, "last_train": None, "origin": "Taft"},
        "northbound": {"available": True, "minutes": None, "last_train": None, "origin": "North Ave"}
    }
    
    if not result["is_operating"]:
        return result
    
    headway_min = headway_info["headway_avg"]
    if headway_min < 1:
        headway_min = 5
    
    # Get station index
    try:
        station_idx = STATIONS.index(station)
    except ValueError:
        return result
    
    # ===== SOUTHBOUND CALCULATION (going TO Taft) =====
    if sb_last and current_time > sb_last:
        result["southbound"]["available"] = False
        result["southbound"]["last_train"] = sb_last.strftime("%I:%M %p")
        result["southbound"]["origin"] = "Last train departed"
    elif sb_open and current_time < sb_open:
        open_delta = (datetime.combine(target_time.date(), sb_open) - target_time).total_seconds() / 60
        result["southbound"]["minutes"] = max(1, int(open_delta))
        result["southbound"]["origin"] = "Opens at " + sb_open.strftime("%I:%M %p")
    else:
        # For southbound: train comes from the station immediately north (1 station away)
        # This is the ARRIVAL time, not travel time from terminal
        if station_idx == 0:  # North Ave (terminal)
            travel_to_station = 0  # Train starts here
        else:
            travel_to_station = 3  # 1 station away = ~3 minutes
        
        minutes_since_midnight = target_time.hour * 60 + target_time.minute + target_time.second / 60.0
        
        first_train_minutes = 4 * 60 + 30  # 4:30 AM from North Ave
        
        if minutes_since_midnight < first_train_minutes:
            wait_until_first = first_train_minutes - minutes_since_midnight
            total_minutes = wait_until_first + travel_to_station
            result["southbound"]["minutes"] = max(1, int(total_minutes))
            result["southbound"]["origin"] = "First train from North Ave"
        else:
            time_since_first = minutes_since_midnight - first_train_minutes
            cycle_pos = time_since_first % headway_min
            wait_time = 0 if cycle_pos == 0 else headway_min - cycle_pos
            total_minutes = wait_time + travel_to_station
            result["southbound"]["minutes"] = max(1, int(total_minutes))
            
            # SOUTHBOUND: Train comes from NORTH (station before current)
            if station_idx == 0:
                result["southbound"]["origin"] = "North Ave (Terminal)"
            else:
                from_idx = station_idx - 1
                result["southbound"]["origin"] = STATIONS[from_idx]
    
    # ===== NORTHBOUND CALCULATION (going TO North Ave) =====
    if nb_last and current_time > nb_last:
        result["northbound"]["available"] = False
        result["northbound"]["last_train"] = nb_last.strftime("%I:%M %p")
        result["northbound"]["origin"] = "Last train departed"
    elif nb_open and current_time < nb_open:
        open_delta = (datetime.combine(target_time.date(), nb_open) - target_time).total_seconds() / 60
        result["northbound"]["minutes"] = max(1, int(open_delta))
        result["northbound"]["origin"] = "Opens at " + nb_open.strftime("%I:%M %p")
    else:
        # For northbound: train comes from the station immediately south (1 station away)
        # This is the ARRIVAL time, not travel time from terminal
        if station_idx == len(STATIONS) - 1:  # Taft (terminal)
            travel_to_station = 0  # Train starts here
        else:
            travel_to_station = 3  # 1 station away = ~3 minutes
        
        minutes_since_midnight = target_time.hour * 60 + target_time.minute + target_time.second / 60.0
        
        first_train_minutes = 5 * 60 + 5  # 5:05 AM from Taft
        
        if minutes_since_midnight < first_train_minutes:
            wait_until_first = first_train_minutes - minutes_since_midnight
            total_minutes = wait_until_first + travel_to_station
            result["northbound"]["minutes"] = max(1, int(total_minutes))
            result["northbound"]["origin"] = "First train from Taft"
        else:
            time_since_first = minutes_since_midnight - first_train_minutes
            cycle_pos = time_since_first % headway_min
            wait_time = 0 if cycle_pos == 0 else headway_min - cycle_pos
            total_minutes = wait_time + travel_to_station
            result["northbound"]["minutes"] = max(1, int(total_minutes))
            
            # NORTHBOUND: Train comes from SOUTH (station after current)
            if station_idx == len(STATIONS) - 1:
                result["northbound"]["origin"] = "Taft (Terminal)"
            else:
                from_idx = station_idx + 1
                result["northbound"]["origin"] = STATIONS[from_idx]
    
    return result
@api_schedule_bp.route('/schedule/compare')
def compare_stations():
    """Compare wait times at different stations"""
    now = datetime.now()
    results = {}
    
    stations_to_test = ["North Ave", "Cubao", "Taft"]
    
    for station in stations_to_test:
        result = calculate_next_trains(station, now)
        results[station] = {
            "southbound": result["southbound"]["minutes"],
            "northbound": result["northbound"]["minutes"]
        }
    
    return jsonify({
        "current_time": now.strftime("%I:%M:%S %p"),
        "stations": results
    })
    
@api_schedule_bp.route('/schedule/debug/<station_name>')
def debug_schedule(station_name):
    """See how time affects the calculation"""
    name = station_name.replace('%20', ' ')
    
    results = {}
    
    # Test at different times on the same day
    base_date = datetime.now().date()
    
    test_times = [
        ("Now", datetime.now()),
        ("+1 min", datetime.now() + timedelta(minutes=1)),
        ("+5 min", datetime.now() + timedelta(minutes=5)),
        ("+10 min", datetime.now() + timedelta(minutes=10)),
        ("+30 min", datetime.now() + timedelta(minutes=30)),
    ]
    
    for label, test_time in test_times:
        result = calculate_next_trains(name, test_time)
        results[label] = {
            "time": test_time.strftime("%I:%M:%S %p"),
            "southbound_minutes": result["southbound"]["minutes"],
            "northbound_minutes": result["northbound"]["minutes"],
            "is_closed": result["is_closed"]
        }
    
    return jsonify({
        "station": name,
        "current_time": datetime.now().strftime("%I:%M:%S %p"),
        "scenarios": results
    })
    
# ========== ROUTES ==========

@api_schedule_bp.route('/schedule/headway')
def get_headway_route():
    headway = get_current_headway()
    return jsonify(headway)
@api_schedule_bp.route('/schedule/next-trains/<station_name>')
def get_next_trains_route(station_name):
    name = station_name.replace('%20', ' ')
    result = calculate_next_trains(name)
    
    # Ensure consistent field names
    return jsonify({
        "station": result.get("station"),
        "is_operating": result.get("is_operating", True),
        "is_closed": result.get("is_closed", False),
        "headway": result.get("headway", 5),
        "headway_description": result.get("headway_description", ""),
        "period": result.get("period", ""),
        "northbound": {
            "available": result.get("northbound", {}).get("available", True),
            "minutes": result.get("northbound", {}).get("minutes"),
            "last_train": result.get("northbound", {}).get("last_train"),
            "origin": result.get("northbound", {}).get("origin", "Taft")
        },
        "southbound": {
            "available": result.get("southbound", {}).get("available", True),
            "minutes": result.get("southbound", {}).get("minutes"),
            "last_train": result.get("southbound", {}).get("last_train"),
            "origin": result.get("southbound", {}).get("origin", "North Ave")
        }
    })
@api_schedule_bp.route('/schedule/station-info/<station_name>')
def station_info_route(station_name):
    name = station_name.replace('%20', ' ')
    now = datetime.now()
    weekend = is_weekend()
    
    sb_open, sb_last = get_station_opening_hours(name, "southbound")
    nb_open, nb_last = get_station_opening_hours(name, "northbound")
    
    return jsonify({
        "station": name,
        "date_type": "Weekend" if weekend else "Weekday",
        "southbound": {
            "entrance_opens": sb_open.strftime("%I:%M %p") if sb_open else None,
            "last_train": sb_last.strftime("%I:%M %p") if sb_last else "Terminal"
        },
        "northbound": {
            "entrance_opens": nb_open.strftime("%I:%M %p") if nb_open else None,
            "last_train": nb_last.strftime("%I:%M %p") if nb_last else "Terminal"
        }
    })