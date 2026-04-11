# mrt_schedule.py - Complete MRT-3 Schedule with detailed headway by period
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Station order for MRT-3 (Northbound = Taft → North Ave, Southbound = North Ave → Taft)
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

# Travel time between stations (in seconds) - based on official distances
TRAVEL_TIMES = {
    ("North Ave", "Quezon Ave"): 110,   # 1:50
    ("Quezon Ave", "Kamuning"): 87,     # 1:27
    ("Kamuning", "Cubao"): 190,         # 3:10
    ("Cubao", "Santolan"): 168,         # 2:48
    ("Santolan", "Ortigas"): 180,       # 3:00
    ("Ortigas", "Shaw Blvd"): 115,      # 1:55
    ("Shaw Blvd", "Boni Ave"): 131,     # 2:11
    ("Boni Ave", "Guadalupe"): 112,     # 1:52
    ("Guadalupe", "Buendia"): 190,      # 3:10
    ("Buendia", "Ayala Ave"): 122,      # 2:02
    ("Ayala Ave", "Magallanes"): 101,   # 1:41
    ("Magallanes", "Taft"): 355,        # 5:55
}

# Station dwell time (seconds)
DWELL_TIME = 40

# ========== WEEKDAY SCHEDULE ==========
# Headway configuration for WEEKDAYS (Monday-Friday)
WEEKDAY_HEADWAY = [
    {"period": "Morning", "start": "04:30", "end": "07:00", "headway": 330, "trains": 14},      # 5.5 min avg
    {"period": "AM_Peak", "start": "07:01", "end": "09:00", "headway": 210, "trains": 19},      # 3.5 min
    {"period": "Off_Peak", "start": "09:01", "end": "17:00", "headway": 315, "trains": 14},     # 5.25 min avg
    {"period": "PM_Peak", "start": "17:01", "end": "19:00", "headway": 210, "trains": 19},      # 3.5 min
    {"period": "Night", "start": "19:01", "end": "21:30", "headway": 390, "trains": 14},        # 6.5 min avg
    {"period": "Extended", "start": "21:31", "end": "23:40", "headway": 900, "trains": 4}       # 15 min
]

# ========== SATURDAY SCHEDULE ==========
SATURDAY_HEADWAY = [
    {"period": "Morning", "start": "04:30", "end": "17:00", "headway": 345, "trains": 14},      # 5.75 min avg
    {"period": "Afternoon", "start": "17:01", "end": "19:00", "headway": 315, "trains": 16},    # 5.25 min avg
    {"period": "Evening", "start": "19:01", "end": "22:40", "headway": 405, "trains": 12}       # 6.75 min avg
]

# ========== SUNDAY/HOLIDAY SCHEDULE ==========
SUNDAY_HEADWAY = [
    {"period": "Full_Day", "start": "04:30", "end": "22:40", "headway": 405, "trains": 12}      # 6.75 min avg
]

# ========== STATION ENTRANCE OPENING TIMES ==========
ENTRANCE_OPENING = {
    "weekday": {
        "North Ave": {"southbound": "04:20", "northbound": "05:24"},
        "Quezon Ave": {"southbound": "04:22", "northbound": "05:23"},
        "Kamuning": {"southbound": "04:24", "northbound": "05:21"},
        "Cubao": {"southbound": "04:27", "northbound": "05:17"},
        "Santolan": {"southbound": "04:30", "northbound": "05:14"},
        "Ortigas": {"southbound": "04:33", "northbound": "05:11"},
        "Shaw Blvd": {"southbound": "04:36", "northbound": "05:10"},
        "Boni Ave": {"southbound": "04:38", "northbound": "05:07"},
        "Guadalupe": {"southbound": "04:40", "northbound": "05:06"},
        "Buendia": {"southbound": "04:43", "northbound": "05:02"},
        "Ayala Ave": {"southbound": "04:45", "northbound": "05:00"},
        "Magallanes": {"southbound": "04:47", "northbound": "04:58"},
        "Taft": {"southbound": "04:50", "northbound": "04:55"}
    },
    "weekend": {
        "North Ave": {"southbound": "04:20", "northbound": "05:27"},
        "Quezon Ave": {"southbound": "04:22", "northbound": "05:25"},
        "Kamuning": {"southbound": "04:24", "northbound": "05:22"},
        "Cubao": {"southbound": "04:28", "northbound": "05:19"},
        "Santolan": {"southbound": "04:31", "northbound": "05:15"},
        "Ortigas": {"southbound": "04:34", "northbound": "05:12"},
        "Shaw Blvd": {"southbound": "04:36", "northbound": "05:10"},
        "Boni Ave": {"southbound": "04:38", "northbound": "05:08"},
        "Guadalupe": {"southbound": "04:40", "northbound": "05:06"},
        "Buendia": {"southbound": "04:44", "northbound": "05:03"},
        "Ayala Ave": {"southbound": "04:46", "northbound": "05:01"},
        "Magallanes": {"southbound": "04:49", "northbound": "04:58"},
        "Taft": {"southbound": "04:52", "northbound": "04:55"}
    }
}

# ========== LAST TRAIN DEPARTURES ==========
LAST_TRAINS = {
    "weekday": {
        "North Ave": {"southbound_departure": "22:30", "northbound_entrance_close": None},
        "Quezon Ave": {"southbound_departure": "22:33", "northbound_departure": "23:37"},
        "Kamuning": {"southbound_departure": "22:35", "northbound_departure": "23:35"},
        "Cubao": {"southbound_departure": "22:38", "northbound_departure": "23:33"},
        "Santolan": {"southbound_departure": "22:40", "northbound_departure": "23:35"},
        "Ortigas": {"southbound_departure": "22:43", "northbound_departure": "23:26"},
        "Shaw Blvd": {"southbound_departure": "22:48", "northbound_departure": "23:27"},
        "Boni Ave": {"southbound_departure": "22:48", "northbound_departure": "23:22"},
        "Guadalupe": {"southbound_departure": "22:50", "northbound_departure": "23:20"},
        "Buendia": {"southbound_departure": "22:56", "northbound_departure": "23:19"},
        "Ayala Ave": {"southbound_departure": "22:55", "northbound_departure": "23:15"},
        "Magallanes": {"southbound_departure": "22:58", "northbound_departure": "23:14"},
        "Taft": {"southbound_departure": None, "northbound_departure": "23:09"}
    },
    "weekend": {
        "North Ave": {"southbound_departure": "21:30", "northbound_entrance_close": None},
        "Quezon Ave": {"southbound_departure": "21:32", "northbound_departure": "22:37"},
        "Kamuning": {"southbound_departure": "21:34", "northbound_departure": "22:35"},
        "Cubao": {"southbound_departure": "21:38", "northbound_departure": "22:32"},
        "Santolan": {"southbound_departure": "21:40", "northbound_departure": "22:29"},
        "Ortigas": {"southbound_departure": "21:44", "northbound_departure": "22:26"},
        "Shaw Blvd": {"southbound_departure": "21:45", "northbound_departure": "22:26"},
        "Boni Ave": {"southbound_departure": "21:48", "northbound_departure": "22:22"},
        "Guadalupe": {"southbound_departure": "21:49", "northbound_departure": "22:20"},
        "Buendia": {"southbound_departure": "21:53", "northbound_departure": "22:16"},
        "Ayala Ave": {"southbound_departure": "21:55", "northbound_departure": "22:26"},
        "Magallanes": {"southbound_departure": "21:57", "northbound_departure": "22:12"},
        "Taft": {"southbound_departure": None, "northbound_departure": "22:09"}
    }
}

# First train departure times
FIRST_TRAINS = {
    "weekday": {
        "northbound": "05:18",  # From Taft
        "southbound": "04:36"   # From North Ave
    },
    "saturday": {
        "northbound": "05:18",  # From Taft
        "southbound": "04:37"   # From North Ave
    },
    "sunday": {
        "northbound": "05:19",  # From Taft
        "southbound": "04:38"   # From North Ave
    }
}

def get_day_type() -> str:
    """Determine if today is weekday, Saturday, or Sunday/holiday"""
    now = datetime.now()
    weekday = now.weekday()
    
    if weekday == 5:  # Saturday
        return "saturday"
    elif weekday == 6:  # Sunday
        return "sunday"
    else:  # Monday to Friday
        return "weekday"

def get_day_schedule_type() -> str:
    """Get schedule type for headway lookup"""
    day_type = get_day_type()
    if day_type == "weekday":
        return "weekday"
    elif day_type == "saturday":
        return "saturday"
    else:
        return "sunday"

def time_to_seconds(time_str: str) -> int:
    """Convert HH:MM or HH:MM:SS to seconds since midnight"""
    parts = time_str.split(':')
    
    if len(parts) == 2:
        h, m = map(int, parts)
        s = 0
    elif len(parts) == 3:
        h, m, s = map(int, parts)
    else:
        raise ValueError(f"Invalid time format: {time_str}")
    
    return h * 3600 + m * 60 + s

def seconds_to_time(seconds: int) -> str:
    """Convert seconds since midnight to HH:MM string"""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h:02d}:{m:02d}"

def get_current_headway() -> int:
    """Get current headway in seconds based on day type and time of day"""
    now = datetime.now()
    current_seconds = now.hour * 3600 + now.minute * 60
    day_schedule = get_day_schedule_type()
    
    # Select headway schedule based on day type
    if day_schedule == "weekday":
        headway_schedule = WEEKDAY_HEADWAY
    elif day_schedule == "saturday":
        headway_schedule = SATURDAY_HEADWAY
    else:  # sunday
        headway_schedule = SUNDAY_HEADWAY
    
    # Find current period
    for period in headway_schedule:
        start_seconds = time_to_seconds(period["start"])
        end_seconds = time_to_seconds(period["end"])
        
        if start_seconds <= current_seconds < end_seconds:
            return period["headway"]
    
    # Default fallback
    return 300  # 5 minutes

def get_headway_info() -> dict:
    """Get current headway info with status and period details"""
    now = datetime.now()
    current_seconds = now.hour * 3600 + now.minute * 60
    day_schedule = get_day_schedule_type()
    
    # Get operating hours based on day type
    if day_schedule == "weekday":
        opening_time = time_to_seconds("04:30")
        closing_time = time_to_seconds("22:30")  # Last train from North Ave
    else:
        opening_time = time_to_seconds("04:30")
        closing_time = time_to_seconds("21:30")  # Earlier closing on weekends
    
    # Check if operating
    if current_seconds < opening_time or current_seconds >= closing_time:
        return {
            "headway": None,
            "status": "CLOSED",
            "message": f"MRT-3 is closed. Operating hours: 4:30 AM - {seconds_to_time(closing_time)}"
        }
    
    # Get current period
    if day_schedule == "weekday":
        headway_schedule = WEEKDAY_HEADWAY
    elif day_schedule == "saturday":
        headway_schedule = SATURDAY_HEADWAY
    else:
        headway_schedule = SUNDAY_HEADWAY
    
    for period in headway_schedule:
        start_seconds = time_to_seconds(period["start"])
        end_seconds = time_to_seconds(period["end"])
        
        if start_seconds <= current_seconds < end_seconds:
            headway_min = period["headway"] / 60
            return {
                "headway": period["headway"],
                "status": "PEAK HOUR" if "Peak" in period["period"] else "NORMAL",
                "message": f"{period['period']} service - trains every {headway_min:.1f} minutes",
                "period": period["period"]
            }
    
    return {
        "headway": 300,
        "status": "NORMAL",
        "message": "Normal service - trains every 5 minutes",
        "period": "Normal"
    }

def get_station_entrance_status(station_name: str, direction: str) -> dict:
    """Check if station entrance is open for specific direction"""
    now = datetime.now()
    current_seconds = now.hour * 3600 + now.minute * 60
    day_type = get_day_type()
    schedule_type = "weekday" if day_type == "weekday" else "weekend"
    
    # Get opening time for this station/direction
    opening_info = ENTRANCE_OPENING[schedule_type].get(station_name, {})
    opening_time_str = opening_info.get(direction, "04:30")
    opening_seconds = time_to_seconds(opening_time_str)
    
    # Get closing time
    last_train_info = LAST_TRAINS[schedule_type].get(station_name, {})
    if direction == "southbound":
        closing_time_str = last_train_info.get("southbound_departure")
    else:
        closing_time_str = last_train_info.get("northbound_departure")
    
    if not closing_time_str:
        # Terminal stations have different rules
        if station_name == "North Ave" and direction == "northbound":
            return {"is_open": False, "message": "North Ave northbound entrance not available"}
        if station_name == "Taft" and direction == "southbound":
            return {"is_open": False, "message": "Taft southbound entrance not available"}
        closing_seconds = opening_seconds + 15 * 3600  # Default 15 hours later
    else:
        closing_seconds = time_to_seconds(closing_time_str)
    
    is_open = opening_seconds <= current_seconds < closing_seconds
    
    return {
        "is_open": is_open,
        "opens_at": opening_time_str,
        "closes_at": closing_time_str if closing_time_str else "N/A",
        "message": "Open" if is_open else f"Opens at {opening_time_str}"
    }

def calculate_next_trains(station_name: str, direction: str = None) -> dict:
    """
    Calculate next train arrivals for a station using complete schedule.
    """
    now = datetime.now()
    current_seconds = now.hour * 3600 + now.minute * 60 + now.second
    day_type = get_day_type()
    
    # Get station index
    try:
        station_idx = STATIONS.index(station_name)
    except ValueError:
        station_idx = 6
    
    # Check if station entrance is open
    north_entrance = get_station_entrance_status(station_name, "northbound")
    south_entrance = get_station_entrance_status(station_name, "southbound")
    
    # ========== NORTHBOUND (Taft → North Ave) ==========
    # Get first train time based on day type
    if day_type == "weekday":
        first_train_taft = time_to_seconds(FIRST_TRAINS["weekday"]["northbound"])
    elif day_type == "saturday":
        first_train_taft = time_to_seconds(FIRST_TRAINS["saturday"]["northbound"])
    else:
        first_train_taft = time_to_seconds(FIRST_TRAINS["sunday"]["northbound"])
    
    # Calculate travel time from Taft to this station
    travel_from_taft = 0
    for i in range(12, station_idx, -1):
        station_pair = (STATIONS[i], STATIONS[i-1])
        travel_from_taft += TRAVEL_TIMES.get(station_pair, 180) + DWELL_TIME
    
    first_northbound_arrival = first_train_taft + travel_from_taft
    
    # Get last northbound train
    schedule_type = "weekday" if day_type == "weekday" else "weekend"
    last_northbound_info = LAST_TRAINS[schedule_type].get(station_name, {})
    last_northbound_str = last_northbound_info.get("northbound_departure")
    
    if last_northbound_str:
        last_northbound_seconds = time_to_seconds(last_northbound_str)
    else:
        last_northbound_seconds = first_northbound_arrival + 12 * 3600
    
    # Calculate next northbound
    if current_seconds < first_northbound_arrival:
        next_north_seconds = first_northbound_arrival
    elif current_seconds >= last_northbound_seconds:
        next_north_seconds = None
    else:
        headway = get_current_headway()
        elapsed = current_seconds - first_northbound_arrival
        trains_passed = elapsed // headway
        next_north_seconds = first_northbound_arrival + (trains_passed + 1) * headway
    
    # ========== SOUTHBOUND (North Ave → Taft) ==========
    if day_type == "weekday":
        first_train_north = time_to_seconds(FIRST_TRAINS["weekday"]["southbound"])
    elif day_type == "saturday":
        first_train_north = time_to_seconds(FIRST_TRAINS["saturday"]["southbound"])
    else:
        first_train_north = time_to_seconds(FIRST_TRAINS["sunday"]["southbound"])
    
    # Calculate travel time from North Ave to this station
    travel_from_north = 0
    for i in range(0, station_idx):
        station_pair = (STATIONS[i], STATIONS[i+1])
        travel_from_north += TRAVEL_TIMES.get(station_pair, 180) + DWELL_TIME
    
    first_southbound_arrival = first_train_north + travel_from_north
    
    # Get last southbound train
    last_southbound_str = last_northbound_info.get("southbound_departure")
    if last_southbound_str:
        last_southbound_seconds = time_to_seconds(last_southbound_str)
    else:
        last_southbound_seconds = first_southbound_arrival + 12 * 3600
    
    # Calculate next southbound
    if current_seconds < first_southbound_arrival:
        next_south_seconds = first_southbound_arrival
    elif current_seconds >= last_southbound_seconds:
        next_south_seconds = None
    else:
        headway = get_current_headway()
        elapsed = current_seconds - first_southbound_arrival
        trains_passed = elapsed // headway
        next_south_seconds = first_southbound_arrival + (trains_passed + 1) * headway
    
    # Calculate minutes
    north_minutes = None
    south_minutes = None
    
    if next_north_seconds and north_entrance["is_open"]:
        north_minutes = max(1, (next_north_seconds - current_seconds) // 60)
        north_minutes = min(north_minutes, 15)  # Cap at 15 minutes
        north_source_idx = station_idx - 1 if station_idx > 0 else 0
        north_source = STATIONS[north_source_idx]
    else:
        north_source = "Taft"
    
    if next_south_seconds and south_entrance["is_open"]:
        south_minutes = max(1, (next_south_seconds - current_seconds) // 60)
        south_minutes = min(south_minutes, 15)  # Cap at 15 minutes
        south_source_idx = station_idx + 1 if station_idx < 12 else 12
        south_source = STATIONS[south_source_idx]
    else:
        south_source = "North Ave"
    
    # Terminal station adjustments
    if station_name == "North Ave":
        if north_minutes:
            north_minutes = max(6, north_minutes)
        if south_minutes:
            south_minutes = min(3, south_minutes)
    elif station_name == "Taft":
        if north_minutes:
            north_minutes = min(3, north_minutes)
        if south_minutes:
            south_minutes = max(6, south_minutes)
    
    # Ensure reasonable values
    if north_minutes:
        north_minutes = max(2, min(12, north_minutes))
    if south_minutes:
        south_minutes = max(2, min(12, south_minutes))
    
    # Default fallbacks if no train found
    if not north_minutes:
        north_minutes = 5
        north_source = "Taft"
    if not south_minutes:
        south_minutes = 5
        south_source = "North Ave"
    
    headway_seconds = get_current_headway()
    
    result = {
        "northbound": {
            "minutes": north_minutes,
            "time": (now + timedelta(minutes=north_minutes)).strftime("%I:%M %p"),
            "from_station": north_source,
            "status": "on_time",
            "entrance_open": north_entrance["is_open"]
        },
        "southbound": {
            "minutes": south_minutes,
            "time": (now + timedelta(minutes=south_minutes)).strftime("%I:%M %p"),
            "from_station": south_source,
            "status": "on_time",
            "entrance_open": south_entrance["is_open"]
        },
        "headway": headway_seconds // 60,
        "is_operating": True
    }
    
    if direction == "north":
        return result["northbound"]
    elif direction == "south":
        return result["southbound"]
    else:
        return result

def get_all_trains_for_station(station_name: str, limit: int = 5) -> dict:
    """Get next N trains for both directions"""
    result = calculate_next_trains(station_name)
    
    if not result.get("is_operating", True):
        return {"error": "Station closed", "status": "closed"}
    
    north_trains = []
    south_trains = []
    now = datetime.now()
    headway = result["headway"] * 60
    
    for i in range(limit):
        # Northbound
        north_minutes = result["northbound"]["minutes"] + i * (headway // 60)
        north_trains.append({
            "time": (now + timedelta(minutes=north_minutes)).strftime("%I:%M %p"),
            "minutes": north_minutes,
            "from_station": result["northbound"]["from_station"]
        })
        
        # Southbound
        south_minutes = result["southbound"]["minutes"] + i * (headway // 60)
        south_trains.append({
            "time": (now + timedelta(minutes=south_minutes)).strftime("%I:%M %p"),
            "minutes": south_minutes,
            "from_station": result["southbound"]["from_station"]
        })
    
    return {
        "station": station_name,
        "trains": {
            "northbound": north_trains,
            "southbound": south_trains
        },
        "headway": result["headway"],
        "status": "normal",
        "is_operating": True
    }

# For debugging
if __name__ == "__main__":
    print("=" * 60)
    print("MRT-3 Schedule Test")
    print("=" * 60)
    
    test_stations = ["North Ave", "Cubao", "Ayala Ave", "Taft"]
    
    for station in test_stations:
        print(f"\n📍 {station}")
        result = calculate_next_trains(station)
        print(f"   Northbound: {result['northbound']['minutes']} min (from {result['northbound']['from_station']})")
        print(f"   Southbound: {result['southbound']['minutes']} min (from {result['southbound']['from_station']})")
        print(f"   Headway: {result['headway']} minutes")
        
# Add this to the END of your mrt_schedule.py file

def get_trip_schedule(from_station: str, to_station: str, target_time: datetime = None) -> dict:
    """
    Get schedule for a trip between two stations
    """
    if target_time is None:
        target_time = datetime.now()
    
    try:
        from_idx = STATIONS.index(from_station)
        to_idx = STATIONS.index(to_station)
    except ValueError:
        return {"error": "Invalid station name"}
    
    # Determine direction
    if from_idx < to_idx:
        direction = "southbound"  # Going towards Taft
        travel_direction = "south"
    else:
        direction = "northbound"  # Going towards North Ave
        travel_direction = "north"
    
    # Get next train from origin station
    train_info = calculate_next_trains(from_station, direction=travel_direction)
    
    if not train_info.get("minutes"):
        return {
            "error": "No trains available or station closed",
            "from_station": from_station,
            "to_station": to_station
        }
    
    # Calculate travel time between stations
    travel_seconds = 0
    if from_idx < to_idx:  # Southbound
        for i in range(from_idx, to_idx):
            station_pair = (STATIONS[i], STATIONS[i + 1])
            travel_seconds += TRAVEL_TIMES.get(station_pair, 180) + DWELL_TIME
    else:  # Northbound
        for i in range(from_idx, to_idx, -1):
            station_pair = (STATIONS[i], STATIONS[i - 1])
            travel_seconds += TRAVEL_TIMES.get(station_pair, 180) + DWELL_TIME
    
    # Calculate departure and arrival times
    departure_time = target_time + timedelta(minutes=train_info["minutes"])
    arrival_time = departure_time + timedelta(seconds=travel_seconds)
    
    return {
        "from_station": from_station,
        "to_station": to_station,
        "direction": direction,
        "departure_time": departure_time.strftime("%I:%M %p"),
        "arrival_time": arrival_time.strftime("%I:%M %p"),
        "travel_time_minutes": travel_seconds // 60,
        "stops_between": abs(to_idx - from_idx),
        "next_train_minutes": train_info["minutes"],
        "status": "scheduled"
    }