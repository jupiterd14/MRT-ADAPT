from flask import Blueprint, request, jsonify
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional

api_schedule_bp = Blueprint('api_schedule', __name__)

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan",
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia",
            "Ayala Ave", "Magallanes", "Taft"]

# ============================================================
# TUNING CONSTANTS
# ============================================================
PER_STOP_MINUTES = 3
FIRST_SB_DEPARTURE_H = 4
FIRST_SB_DEPARTURE_M = 30
FIRST_NB_DEPARTURE_H = 5
FIRST_NB_DEPARTURE_M = 5

# ============================================================
# WEEKDAY ENTRANCE OPENING (SB / NB)
# ============================================================
WEEKDAY_OPENING_SB = {
    "North Ave": time(4, 20), "Quezon Ave": time(4, 22), "Kamuning": time(4, 24),
    "Cubao": time(4, 27), "Santolan": time(4, 30), "Ortigas": time(4, 33),
    "Shaw Blvd": time(4, 36), "Boni Ave": time(4, 38), "Guadalupe": time(4, 40),
    "Buendia": time(4, 43), "Ayala Ave": time(4, 45), "Magallanes": time(4, 47),
    "Taft": time(4, 50)
}

WEEKDAY_OPENING_NB = {
    "North Ave": time(5, 24), "Quezon Ave": time(5, 23), "Kamuning": time(5, 21),
    "Cubao": time(5, 17), "Santolan": time(5, 14), "Ortigas": time(5, 11),
    "Shaw Blvd": time(5, 10), "Boni Ave": time(5, 7), "Guadalupe": time(5, 6),
    "Buendia": time(5, 2), "Ayala Ave": time(5, 0), "Magallanes": time(4, 58),
    "Taft": time(4, 55)
}

# ============================================================
# WEEKEND ENTRANCE OPENING
# ============================================================
WEEKEND_OPENING_SB = {
    "North Ave": time(4, 20), "Quezon Ave": time(4, 22), "Kamuning": time(4, 24),
    "Cubao": time(4, 28), "Santolan": time(4, 31), "Ortigas": time(4, 34),
    "Shaw Blvd": time(4, 36), "Boni Ave": time(4, 38), "Guadalupe": time(4, 40),
    "Buendia": time(4, 44), "Ayala Ave": time(4, 46), "Magallanes": time(4, 49),
    "Taft": time(4, 52)
}

WEEKEND_OPENING_NB = {
    "North Ave": time(5, 27), "Quezon Ave": time(5, 25), "Kamuning": time(5, 22),
    "Cubao": time(5, 19), "Santolan": time(5, 15), "Ortigas": time(5, 12),
    "Shaw Blvd": time(5, 10), "Boni Ave": time(5, 8), "Guadalupe": time(5, 6),
    "Buendia": time(5, 3), "Ayala Ave": time(5, 1), "Magallanes": time(4, 58),
    "Taft": time(4, 55)
}

# ============================================================
# LAST TRAIN DEPARTURES
# ============================================================
WEEKDAY_LAST_SB = {
    "North Ave": time(22, 30), "Quezon Ave": time(22, 33), "Kamuning": time(22, 35),
    "Cubao": time(22, 38), "Santolan": time(22, 40), "Ortigas": time(22, 43),
    "Shaw Blvd": time(22, 48), "Boni Ave": time(22, 48), "Guadalupe": time(22, 50),
    "Buendia": time(22, 56), "Ayala Ave": time(22, 55), "Magallanes": time(22, 58),
    "Taft": None
}

WEEKDAY_LAST_NB = {
    "North Ave": None, "Quezon Ave": time(23, 37), "Kamuning": time(23, 35),
    "Cubao": time(23, 33), "Santolan": time(23, 35), "Ortigas": time(23, 26),
    "Shaw Blvd": time(23, 27), "Boni Ave": time(23, 22), "Guadalupe": time(23, 20),
    "Buendia": time(23, 19), "Ayala Ave": time(23, 15), "Magallanes": time(23, 14),
    "Taft": time(23, 9)
}

WEEKEND_LAST_SB = {
    "North Ave": time(21, 30), "Quezon Ave": time(21, 32), "Kamuning": time(21, 34),
    "Cubao": time(21, 38), "Santolan": time(21, 40), "Ortigas": time(21, 44),
    "Shaw Blvd": time(21, 45), "Boni Ave": time(21, 48), "Guadalupe": time(21, 49),
    "Buendia": time(21, 53), "Ayala Ave": time(21, 55), "Magallanes": time(21, 57),
    "Taft": None
}

WEEKEND_LAST_NB = {
    "North Ave": None, "Quezon Ave": time(22, 37), "Kamuning": time(22, 35),
    "Cubao": time(22, 32), "Santolan": time(22, 29), "Ortigas": time(22, 26),
    "Shaw Blvd": time(22, 26), "Boni Ave": time(22, 22), "Guadalupe": time(22, 20),
    "Buendia": time(22, 16), "Ayala Ave": time(22, 26), "Magallanes": time(22, 12),
    "Taft": time(22, 9)
}

# ============================================================
# HEADWAY SCHEDULES — continuous windows, no gaps
# ============================================================
WEEKDAY_HEADWAY = [
    (time(4, 30),  time(7, 0),   7*60,   4*60,   "Morning",  "7-4 minutes"),
    (time(7, 0),   time(9, 0),   210,    210,    "AM Peak",  "3.5 minutes"),
    (time(9, 0),   time(17, 0),  5*60,   5.5*60, "Off Peak", "5-5.5 minutes"),
    (time(17, 0),  time(19, 0),  210,    210,    "PM Peak",  "3.5 minutes"),
    (time(19, 0),  time(21, 30), 5*60,   8*60,   "Night",    "5-8 minutes"),
    (time(21, 30), time(23, 40), 15*60,  15*60,  "Extended", "15 minutes"),
]

WEEKEND_HEADWAY = [
    (time(4, 30),  time(17, 0),  5.5*60, 6*60,   "Saturday Morning",   "5.5-6 minutes"),
    (time(17, 0),  time(19, 0),  5*60,   5.5*60, "Saturday Afternoon", "5-5.5 minutes"),
    (time(19, 0),  time(22, 40), 6.5*60, 7*60,   "Saturday Night",     "6.5-7 minutes"),
]

SUNDAY_HEADWAY = [
    (time(4, 30), time(22, 40), 6.5*60, 7*60, "Sunday/Holiday", "6.5-7 minutes"),
]


def is_weekend(target_time: datetime = None) -> bool:
    if target_time is None:
        target_time = datetime.now()
    return target_time.weekday() >= 5


def is_sunday(target_time: datetime = None) -> bool:
    if target_time is None:
        target_time = datetime.now()
    return target_time.weekday() == 6


def get_station_opening_hours(station: str, direction: str,
                              target_time: datetime = None) -> Tuple[Optional[time], Optional[time]]:
    """Returns (opening_time, last_train_time) for a station and direction."""
    if target_time is None:
        target_time = datetime.now()
    weekend = target_time.weekday() >= 5

    if direction == "southbound":
        opening = WEEKEND_OPENING_SB.get(station) if weekend else WEEKDAY_OPENING_SB.get(station)
        last = WEEKEND_LAST_SB.get(station) if weekend else WEEKDAY_LAST_SB.get(station)
    else:
        opening = WEEKEND_OPENING_NB.get(station) if weekend else WEEKDAY_OPENING_NB.get(station)
        last = WEEKEND_LAST_NB.get(station) if weekend else WEEKDAY_LAST_NB.get(station)

    return opening, last


def get_current_headway(target_time: datetime = None) -> Dict:
    """Get headway based on target_time (or now if None)."""
    if target_time is None:
        target_time = datetime.now()
    current_time = target_time.time()

    if current_time < time(4, 30) or current_time >= time(23, 40):
        return {"is_operating": False, "headway_avg": 0, "headway_avg_seconds": 0,
                "description": "Station closed"}

    weekday = target_time.weekday()
    if weekday == 6:
        schedule = SUNDAY_HEADWAY
    elif weekday >= 5:
        schedule = WEEKEND_HEADWAY
    else:
        schedule = WEEKDAY_HEADWAY

    for start, end, min_h, max_h, period, desc in schedule:
        if start <= current_time < end:
            avg_sec = (min_h + max_h) / 2.0
            return {
                "is_operating": True,
                "headway_min": min_h // 60,
                "headway_max": max_h // 60,
                "headway_avg": round(avg_sec / 60.0, 1),
                "headway_avg_seconds": avg_sec,
                "period": period,
                "description": desc,
                "is_peak": "Peak" in period,
            }

    return {"is_operating": True, "headway_avg": 5, "headway_avg_seconds": 300,
            "description": "Normal operations"}


def _first_departure_minutes(direction: str) -> float:
    if direction == "southbound":
        return FIRST_SB_DEPARTURE_H * 60 + FIRST_SB_DEPARTURE_M
    return FIRST_NB_DEPARTURE_H * 60 + FIRST_NB_DEPARTURE_M


def _travel_minutes_to_station(station_idx: int, direction: str) -> float:
    if direction == "southbound":
        return station_idx * PER_STOP_MINUTES
    return (len(STATIONS) - 1 - station_idx) * PER_STOP_MINUTES


def _calculate_wait_minutes(station_idx: int, direction: str,
                            minutes_since_midnight: float,
                            headway_minutes: float) -> float:
    first_departure = _first_departure_minutes(direction)
    travel_time = _travel_minutes_to_station(station_idx, direction)
    first_arrival = first_departure + travel_time

    if minutes_since_midnight < first_arrival:
        return max(1.0, first_arrival - minutes_since_midnight)

    time_since_first_arrival = minutes_since_midnight - first_arrival
    cycle_pos = time_since_first_arrival % headway_minutes
    wait = 0.0 if cycle_pos == 0 else headway_minutes - cycle_pos
    return max(1.0, wait)


def calculate_next_trains(station: str, target_time: datetime = None) -> Dict:
    """Calculate next train arrival times for a station (respects target_time)."""
    if target_time is None:
        target_time = datetime.now()

    current_time = target_time.time()

    sb_open, sb_last = get_station_opening_hours(station, "southbound", target_time)
    nb_open, nb_last = get_station_opening_hours(station, "northbound", target_time)

    headway_info = get_current_headway(target_time)
    is_operating = headway_info.get("is_operating", True)
    is_closed = not is_operating

    result = {
        "station": station,
        "is_operating": is_operating,
        "is_closed": is_closed,
        "headway": headway_info.get("headway_avg", 5),
        "headway_avg_seconds": headway_info.get("headway_avg_seconds", 300),
        "headway_description": headway_info.get("description", ""),
        "period": headway_info.get("period", ""),
        "southbound": {"available": True, "minutes": None, "last_train": None, "origin": "Taft"},
        "northbound": {"available": True, "minutes": None, "last_train": None, "origin": "North Ave"},
    }

    # ============================================================
    # CASE: station closed — populate last_train from schedule tables
    # ============================================================
    if is_closed:
        # Southbound
        if sb_last:
            result["southbound"]["last_train"] = sb_last.strftime("%I:%M %p")
            result["southbound"]["origin"] = "Last SB departure"
        else:
            result["southbound"]["origin"] = "Taft (Terminal)"
        result["southbound"]["available"] = False

        # Northbound
        if nb_last:
            result["northbound"]["last_train"] = nb_last.strftime("%I:%M %p")
            result["northbound"]["origin"] = "Last NB departure"
        else:
            result["northbound"]["origin"] = "North Ave (Terminal)"
        result["northbound"]["available"] = False

        return result

    # ============================================================
    # Station is open — compute per-direction wait times
    # ============================================================
    headway_min_seconds = headway_info.get("headway_avg_seconds", 300)
    headway_minutes = headway_min_seconds / 60.0
    if headway_minutes < 1:
        headway_minutes = 5

    try:
        station_idx = STATIONS.index(station)
    except ValueError:
        return result

    minutes_since_midnight = target_time.hour * 60 + target_time.minute + target_time.second / 60.0

    # ===== SOUTHBOUND =====
    if station_idx == len(STATIONS) - 1:  # Taft — SB terminal, no SB departures
        result["southbound"]["available"] = False
        result["southbound"]["origin"] = "Taft (Terminal)"
    elif sb_last and current_time > sb_last:
        result["southbound"]["available"] = False
        result["southbound"]["last_train"] = sb_last.strftime("%I:%M %p")
        result["southbound"]["origin"] = "Last train departed"
    elif sb_open and current_time < sb_open:
        open_delta = (datetime.combine(target_time.date(), sb_open) - target_time).total_seconds() / 60
        result["southbound"]["minutes"] = max(1, int(open_delta))
        result["southbound"]["origin"] = "Station closed — opens soon"
    else:
        wait = _calculate_wait_minutes(station_idx, "southbound",
                                        minutes_since_midnight, headway_minutes)
        result["southbound"]["minutes"] = max(1, int(round(wait)))
        if station_idx == 0:
            result["southbound"]["origin"] = "North Ave (Terminal)"
        else:
            result["southbound"]["origin"] = STATIONS[station_idx - 1]

    # ===== NORTHBOUND =====
    if station_idx == 0:  # North Ave — NB terminal, no NB departures
        result["northbound"]["available"] = False
        result["northbound"]["origin"] = "North Ave (Terminal)"
    elif nb_last and current_time > nb_last:
        result["northbound"]["available"] = False
        result["northbound"]["last_train"] = nb_last.strftime("%I:%M %p")
        result["northbound"]["origin"] = "Last train departed"
    elif nb_open and current_time < nb_open:
        open_delta = (datetime.combine(target_time.date(), nb_open) - target_time).total_seconds() / 60
        result["northbound"]["minutes"] = max(1, int(open_delta))
        result["northbound"]["origin"] = "Station closed — opens soon"
    else:
        wait = _calculate_wait_minutes(station_idx, "northbound",
                                        minutes_since_midnight, headway_minutes)
        result["northbound"]["minutes"] = max(1, int(round(wait)))
        if station_idx == len(STATIONS) - 1:
            result["northbound"]["origin"] = "Taft (Terminal)"
        else:
            result["northbound"]["origin"] = STATIONS[station_idx + 1]

    return result


# ============================================================
# ROUTES
# ============================================================

@api_schedule_bp.route('/schedule/test')
def test_schedule():
    now = datetime.now()
    result = calculate_next_trains("North Ave", now)
    return jsonify({"status": "ok", "timestamp": now.isoformat(), "test_result": result})


@api_schedule_bp.route('/schedule/headway')
def get_headway_route():
    return jsonify(get_current_headway())


@api_schedule_bp.route('/schedule/next-trains/<station_name>')
def get_next_trains_route(station_name):
    name = station_name.replace('%20', ' ')
    result = calculate_next_trains(name)
    return jsonify({
        "station": result.get("station"),
        "is_operating": result.get("is_operating", True),
        "is_closed": result.get("is_closed", False),
        "headway": result.get("headway", 5),
        "headway_avg_seconds": result.get("headway_avg_seconds", 300),
        "headway_description": result.get("headway_description", ""),
        "period": result.get("period", ""),
        "southbound": {
            "available": result.get("southbound", {}).get("available", True),
            "minutes": result.get("southbound", {}).get("minutes"),
            "last_train": result.get("southbound", {}).get("last_train"),
            "origin": result.get("southbound", {}).get("origin", "North Ave"),
        },
        "northbound": {
            "available": result.get("northbound", {}).get("available", True),
            "minutes": result.get("northbound", {}).get("minutes"),
            "last_train": result.get("northbound", {}).get("last_train"),
            "origin": result.get("northbound", {}).get("origin", "Taft"),
        },
    })


@api_schedule_bp.route('/schedule/station-info/<station_name>')
def station_info_route(station_name):
    name = station_name.replace('%20', ' ')
    now = datetime.now()
    sb_open, sb_last = get_station_opening_hours(name, "southbound", now)
    nb_open, nb_last = get_station_opening_hours(name, "northbound", now)
    return jsonify({
        "station": name,
        "date_type": "Weekend" if is_weekend(now) else "Weekday",
        "southbound": {
            "entrance_opens": sb_open.strftime("%I:%M %p") if sb_open else None,
            "last_train": sb_last.strftime("%I:%M %p") if sb_last else "Terminal",
        },
        "northbound": {
            "entrance_opens": nb_open.strftime("%I:%M %p") if nb_open else None,
            "last_train": nb_last.strftime("%I:%M %p") if nb_last else "Terminal",
        },
    })


@api_schedule_bp.route('/schedule/test-time/<station_name>/<int:hour>/<int:minute>')
def test_time_schedule(station_name, hour, minute):
    name = station_name.replace('%20', ' ')
    now = datetime.now()
    mock_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    result = calculate_next_trains(name, mock_time)
    return jsonify({
        "station": name,
        "mock_time": mock_time.strftime("%I:%M %p"),
        "is_closed": result["is_closed"],
        "headway_avg": result["headway"],
        "headway_avg_seconds": result["headway_avg_seconds"],
        "period": result["period"],
        "southbound": {
            "minutes": result["southbound"]["minutes"],
            "origin": result["southbound"]["origin"],
            "available": result["southbound"]["available"],
            "last_train": result["southbound"].get("last_train"),
        },
        "northbound": {
            "minutes": result["northbound"]["minutes"],
            "origin": result["northbound"]["origin"],
            "available": result["northbound"]["available"],
            "last_train": result["northbound"].get("last_train"),
        },
        "note": "This is a test with a simulated time",
    })


@api_schedule_bp.route('/schedule/compare')
def compare_stations():
    now = datetime.now()
    results = {}
    for station in ["North Ave", "Cubao", "Taft"]:
        result = calculate_next_trains(station, now)
        results[station] = {
            "southbound": result["southbound"]["minutes"],
            "northbound": result["northbound"]["minutes"],
        }
    return jsonify({"current_time": now.strftime("%I:%M:%S %p"), "stations": results})


@api_schedule_bp.route('/schedule/debug/<station_name>')
def debug_schedule(station_name):
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
            "is_closed": result["is_closed"],
        }
    return jsonify({
        "station": name,
        "current_time": datetime.now().strftime("%I:%M:%S %p"),
        "scenarios": results,
    })


@api_schedule_bp.route('/schedule/with-congestion/<station_name>')
def schedule_with_congestion(station_name):
    from routes.api_predict import get_directional_prediction
    from config import Config

    name = station_name.replace('%20', ' ')
    now = Config.get_current_time()
    schedule = calculate_next_trains(name, now)
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
            "headway_avg_seconds": schedule.get("headway_avg_seconds", 300),
            "headway_description": schedule.get("headway_description", ""),
            "period": schedule.get("period", ""),
            "southbound": {
                "next_train_minutes": schedule["southbound"]["minutes"],
                "origin": schedule["southbound"]["origin"],
                "available": schedule["southbound"]["available"],
                "congestion": round(south_cong, 1) if south_cong is not None else None,
                "status": get_congestion_status(south_cong),
            },
            "northbound": {
                "next_train_minutes": schedule["northbound"]["minutes"],
                "origin": schedule["northbound"]["origin"],
                "available": schedule["northbound"]["available"],
                "congestion": round(north_cong, 1) if north_cong is not None else None,
                "status": get_congestion_status(north_cong),
            },
        },
    })


@api_schedule_bp.route('/schedule/debug-headway')
def debug_headway():
    now = datetime.now()
    headway = get_current_headway()
    return jsonify({
        "current_time": now.strftime("%I:%M:%S %p"),
        "weekday": now.strftime("%A"),
        "weekday_number": now.weekday(),
        "is_weekend": is_weekend(),
        "is_sunday": is_sunday(),
        "headway": headway,
    })