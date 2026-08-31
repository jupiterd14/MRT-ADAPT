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
# Format: (start_time, end_time, min_headway_seconds, max_headway_seconds, period, description)
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
    """Check if today is weekend (Saturday or Sunday)"""
    return datetime.now().weekday() >= 5


def is_sunday() -> bool:
    """Check if today is Sunday"""
    return datetime.now().weekday() == 6


def get_station_opening_hours(station: str, direction: str) -> Tuple[Optional[time], Optional[time]]:
    """Returns (opening_time, last_train_time) for a station and direction"""
    now = datetime.now()
    weekend = is_weekend()
    
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
    
    # Select schedule based on day type
    if is_sunday():
        schedule = SUNDAY_HEADWAY
    elif is_weekend():
        schedule = WEEKEND_HEADWAY
    else:
        schedule = WEEKDAY_HEADWAY
    
    for start, end, min_h, max_h, period, desc in schedule:
        if start <= current_time < end:
            return {
                "is_operating": True,
                "headway_min": int(min_h // 60),
                "headway_max": int(max_h // 60),
                "headway_avg": int((min_h + max_h) // 120),
                "period": period,
                "description": desc,
                "is_peak": "Peak" in period
            }
    
    # Fallback - should never reach here
    return {
        "is_operating": True,
        "headway_avg": 5,
        "description": "Normal operations"
    }


def calculate_next_trains(station: str, target_time: datetime = None) -> Dict:
    """Calculate next train arrival times for a station"""
    if target_time is None:
        target_time = datetime.now()
    
    current_time = target_time.time()
    
    # Get opening hours
    sb_open, sb_last = get_station_opening_hours(station, "southbound")
    nb_open, nb_last = get_station_opening_hours(station, "northbound")
    
    headway_info = get_current_headway()
    
    # Check if MRT is operating
    is_operating = headway_info.get("is_operating", True)
    is_closed = not is_operating
    
    result = {
        "station": station,
        "is_operating": is_operating,
        "is_closed": is_closed,
        "headway": headway_info.get("headway_avg", 5),
        "headway_description": headway_info.get("description", ""),
        "period": headway_info.get("period", ""),
        "southbound": {"available": True, "minutes": None, "last_train": None, "origin": "Taft"},
        "northbound": {"available": True, "minutes": None, "last_train": None, "origin": "North Ave"}
    }
    
    # If station is closed, return early
    if is_closed:
        return result
    
    headway_min = headway_info["headway_avg"]
    if headway_min < 1:
        headway_min = 5
    
    try:
        station_idx = STATIONS.index(station)
    except ValueError:
        return result
    
    minutes_since_midnight = target_time.hour * 60 + target_time.minute + target_time.second / 60.0
    
    # ===== SOUTHBOUND =====
    # Check if last train has passed
    if sb_last and current_time > sb_last:
        result["southbound"]["available"] = False
        result["southbound"]["last_train"] = sb_last.strftime("%I:%M %p")
        result["southbound"]["origin"] = "Last train departed"
    # Check if station hasn't opened yet
    elif sb_open and current_time < sb_open:
        open_delta = (datetime.combine(target_time.date(), sb_open) - target_time).total_seconds() / 60
        result["southbound"]["minutes"] = max(1, int(open_delta))
        # Set origin based on station
        if station_idx == 0:  # North Ave
            result["southbound"]["origin"] = "North Ave (Terminal)"
        else:
            from_idx = station_idx - 1
            result["southbound"]["origin"] = STATIONS[from_idx]
    else:
        # Calculate next train arrival
        if station_idx == 0:  # North Ave (Terminal)
            travel_to_station = 0
            result["southbound"]["origin"] = "North Ave (Terminal)"
        else:
            travel_to_station = 3
            from_idx = station_idx - 1
            result["southbound"]["origin"] = STATIONS[from_idx]
        
        first_train_minutes = 4 * 60 + 30  # 4:30 AM from North Ave
        
        if minutes_since_midnight < first_train_minutes:
            wait_until_first = first_train_minutes - minutes_since_midnight
            total_minutes = wait_until_first + travel_to_station
            result["southbound"]["minutes"] = max(1, int(total_minutes))
        else:
            time_since_first = minutes_since_midnight - first_train_minutes
            cycle_pos = time_since_first % headway_min
            wait_time = 0 if cycle_pos == 0 else headway_min - cycle_pos
            total_minutes = wait_time + travel_to_station
            result["southbound"]["minutes"] = max(1, int(total_minutes))
        
        # Special case: Taft should show "Magallanes"
        if station_idx == len(STATIONS) - 1:  # Taft
            result["southbound"]["origin"] = "Magallanes"
    
    # ===== NORTHBOUND =====
    # Check if last train has passed
    if nb_last and current_time > nb_last:
        result["northbound"]["available"] = False
        result["northbound"]["last_train"] = nb_last.strftime("%I:%M %p")
        result["northbound"]["origin"] = "Last train departed"
    # Check if station hasn't opened yet
    elif nb_open and current_time < nb_open:
        open_delta = (datetime.combine(target_time.date(), nb_open) - target_time).total_seconds() / 60
        result["northbound"]["minutes"] = max(1, int(open_delta))
        # Set origin based on station
        if station_idx == len(STATIONS) - 1:  # Taft
            result["northbound"]["origin"] = "Taft (Terminal)"
        else:
            from_idx = station_idx + 1
            result["northbound"]["origin"] = STATIONS[from_idx]
    else:
        # Calculate next train arrival
        if station_idx == len(STATIONS) - 1:  # Taft (Terminal)
            travel_to_station = 0
            result["northbound"]["origin"] = "Taft (Terminal)"
        else:
            travel_to_station = 3
            from_idx = station_idx + 1
            result["northbound"]["origin"] = STATIONS[from_idx]
        
        first_train_minutes = 5 * 60 + 5  # 5:05 AM from Taft
        
        if minutes_since_midnight < first_train_minutes:
            wait_until_first = first_train_minutes - minutes_since_midnight
            total_minutes = wait_until_first + travel_to_station
            result["northbound"]["minutes"] = max(1, int(total_minutes))
        else:
            time_since_first = minutes_since_midnight - first_train_minutes
            cycle_pos = time_since_first % headway_min
            wait_time = 0 if cycle_pos == 0 else headway_min - cycle_pos
            total_minutes = wait_time + travel_to_station
            result["northbound"]["minutes"] = max(1, int(total_minutes))
        
        # Special case: North Ave should show "Quezon Ave"
        if station_idx == 0:  # North Ave
            result["northbound"]["origin"] = "Quezon Ave"
    
    return result


# ============================================================
# ROUTES
# ============================================================

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


@api_schedule_bp.route('/schedule/headway')
def get_headway_route():
    """Get current headway information"""
    headway = get_current_headway()
    return jsonify(headway)


@api_schedule_bp.route('/schedule/next-trains/<station_name>')
def get_next_trains_route(station_name):
    """Get next train arrival times for a station"""
    name = station_name.replace('%20', ' ')
    result = calculate_next_trains(name)
    
    return jsonify({
        "station": result.get("station"),
        "is_operating": result.get("is_operating", True),
        "is_closed": result.get("is_closed", False),
        "headway": result.get("headway", 5),
        "headway_description": result.get("headway_description", ""),
        "period": result.get("period", ""),
        "southbound": {
            "available": result.get("southbound", {}).get("available", True),
            "minutes": result.get("southbound", {}).get("minutes"),
            "last_train": result.get("southbound", {}).get("last_train"),
            "origin": result.get("southbound", {}).get("origin", "North Ave")
        },
        "northbound": {
            "available": result.get("northbound", {}).get("available", True),
            "minutes": result.get("northbound", {}).get("minutes"),
            "last_train": result.get("northbound", {}).get("last_train"),
            "origin": result.get("northbound", {}).get("origin", "Taft")
        }
    })


@api_schedule_bp.route('/schedule/station-info/<station_name>')
def station_info_route(station_name):
    """Get station opening hours and last train times"""
    name = station_name.replace('%20', ' ')
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


@api_schedule_bp.route('/schedule/test-time/<station_name>/<int:hour>/<int:minute>')
def test_time_schedule(station_name, hour, minute):
    """Test schedule at a specific time (debugging)"""
    name = station_name.replace('%20', ' ')
    
    # Create a mock time
    now = datetime.now()
    mock_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    # Calculate trains at mock time
    result = calculate_next_trains(name, mock_time)
    
    return jsonify({
        "station": name,
        "mock_time": mock_time.strftime("%I:%M %p"),
        "is_closed": result["is_closed"],
        "southbound": {
            "minutes": result["southbound"]["minutes"],
            "origin": result["southbound"]["origin"],
            "available": result["southbound"]["available"]
        },
        "northbound": {
            "minutes": result["northbound"]["minutes"],
            "origin": result["northbound"]["origin"],
            "available": result["northbound"]["available"]
        },
        "note": "This is a test with a simulated time"
    })


@api_schedule_bp.route('/schedule/compare')
def compare_stations():
    """Compare wait times at different stations (debugging)"""
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
    """Debug how schedule changes over time (debugging)"""
    name = station_name.replace('%20', ' ')
    
    results = {}
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


# ============================================================
# COMBINED SCHEDULE + CONGESTION ENDPOINT
# ============================================================

@api_schedule_bp.route('/schedule/with-congestion/<station_name>')
def schedule_with_congestion(station_name):
    """
    Get both schedule and congestion prediction for a station.
    Combines train arrival times with AI congestion predictions.
    """
    from routes.api_predict import get_directional_prediction
    from config import Config
    
    name = station_name.replace('%20', ' ')
    now = Config.get_current_time()
    
    # Get schedule
    schedule = calculate_next_trains(name, now)
    
    # Get congestion predictions
    north_cong = get_directional_prediction(name, 'Northbound', now)
    south_cong = get_directional_prediction(name, 'Southbound', now)
    
    def get_congestion_status(cong):
        if cong is None:
            return "Unknown"
        elif cong > 80:
            return "Severe"
        elif cong > 50:
            return "Congested"
        elif cong > 25:
            return "Moderate"
        else:
            return "Light"
    
    return jsonify({
        "station": name,
        "timestamp": now.isoformat(),
        "schedule": {
            "is_operating": schedule.get("is_operating", True),
            "headway": schedule.get("headway", 5),
            "headway_description": schedule.get("headway_description", ""),
            "period": schedule.get("period", ""),
            "southbound": {
                "next_train_minutes": schedule["southbound"]["minutes"],
                "origin": schedule["southbound"]["origin"],
                "available": schedule["southbound"]["available"],
                "congestion": round(south_cong, 1) if south_cong is not None else None,
                "status": get_congestion_status(south_cong)
            },
            "northbound": {
                "next_train_minutes": schedule["northbound"]["minutes"],
                "origin": schedule["northbound"]["origin"],
                "available": schedule["northbound"]["available"],
                "congestion": round(north_cong, 1) if north_cong is not None else None,
                "status": get_congestion_status(north_cong)
            }
        }
    })


@api_schedule_bp.route('/schedule/debug-headway')
def debug_headway():
    """Debug current headway calculation"""
    now = datetime.now()
    headway = get_current_headway()
    
    return jsonify({
        "current_time": now.strftime("%I:%M:%S %p"),
        "weekday": now.strftime("%A"),
        "weekday_number": now.weekday(),
        "is_weekend": is_weekend(),
        "is_sunday": is_sunday(),
        "headway": headway
    })


print("✅ api_schedule.py loaded successfully!")