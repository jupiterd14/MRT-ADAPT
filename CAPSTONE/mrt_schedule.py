# mrt_schedule.py - Real MRT-3 Schedule Data
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Station order for MRT-3 (Northbound = Taft → North Ave, Southbound = North Ave → Taft)
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

# Travel time between stations (in seconds)
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

# First train departures (based on your schedule data)
FIRST_TRAINS = {
    "northbound": {  # From Taft to North Ave
        "station": "Taft",
        "time": "05:05:00"  # 5:05 AM
    },
    "southbound": {  # From North Ave to Taft
        "station": "North Ave",
        "time": "04:30:00"  # 4:30 AM
    }
}

# Headway by time period (based on your schedule)
HEADWAY_CONFIG = {
    "peak_morning": {
        "start": "07:01",
        "end": "09:00",
        "headway": 210,  # 3.5 minutes in seconds
        "trains": 19
    },
    "peak_evening": {
        "start": "17:01",
        "end": "19:00",
        "headway": 210,  # 3.5 minutes
        "trains": 19
    },
    "off_peak_morning": {
        "start": "04:30",
        "end": "07:00",
        "headway": 300,  # 5 minutes
        "trains": 30
    },
    "off_peak_midday": {
        "start": "09:01",
        "end": "17:00",
        "headway": 300,  # 5 minutes
        "trains": 30
    },
    "off_peak_evening": {
        "start": "19:01",
        "end": "22:30",
        "headway": 300,  # 5 minutes
        "trains": 30
    }
}

# Station dwell times (seconds at each station)
DWELL_TIME = 40  # 40 seconds average dwell time

def time_to_seconds(time_str: str) -> int:
    """Convert HH:MM or HH:MM:SS to seconds since midnight"""
    parts = time_str.split(':')
    
    if len(parts) == 2:
        # Format: HH:MM
        h, m = map(int, parts)
        s = 0
    elif len(parts) == 3:
        # Format: HH:MM:SS
        h, m, s = map(int, parts)
    else:
        raise ValueError(f"Invalid time format: {time_str}")
    
    return h * 3600 + m * 60 + s


def seconds_to_time(seconds: int) -> str:
    """Convert seconds since midnight to HH:MM:SS string"""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def seconds_to_time(seconds: int) -> str:
    """Convert seconds since midnight to HH:MM:SS string"""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
def get_current_headway() -> int:
    """Get current headway in seconds based on time of day"""
    now = datetime.now()
    current_seconds = now.hour * 3600 + now.minute * 60
    
    # Define time periods in seconds
    morning_peak_start = time_to_seconds("07:01")
    morning_peak_end = time_to_seconds("09:00")
    evening_peak_start = time_to_seconds("17:01")
    evening_peak_end = time_to_seconds("19:00")
    early_morning_start = time_to_seconds("04:30")
    early_morning_end = time_to_seconds("07:00")
    midday_start = time_to_seconds("09:01")
    midday_end = time_to_seconds("17:00")
    evening_start = time_to_seconds("19:01")
    evening_end = time_to_seconds("22:30")
    
    # Check each period
    if morning_peak_start <= current_seconds <= morning_peak_end:
        return 210  # 3.5 minutes during morning rush
    elif evening_peak_start <= current_seconds <= evening_peak_end:
        return 210  # 3.5 minutes during evening rush
    elif early_morning_start <= current_seconds <= early_morning_end:
        return 270  # 4.5 minutes early morning
    elif midday_start <= current_seconds <= midday_end:
        return 300  # 5 minutes mid-day
    elif evening_start <= current_seconds <= evening_end:
        return 330  # 5.5 minutes evening
    else:
        return 300  # Default 5 minutes

def get_headway_info() -> dict:
    """Get current headway info with status"""
    now = datetime.now()
    current_seconds = now.hour * 3600 + now.minute * 60
    
    opening_time = time_to_seconds("04:30")
    closing_time = time_to_seconds("22:30")
    
    # Check if operating hours
    if current_seconds < opening_time or current_seconds >= closing_time:
        return {
            "headway": None,
            "status": "CLOSED",
            "message": "MRT-3 is closed. Operating hours: 4:30 AM - 10:30 PM"
        }
    
    # Check peak hours
    morning_peak_start = time_to_seconds("07:01")
    morning_peak_end = time_to_seconds("09:00")
    evening_peak_start = time_to_seconds("17:01")
    evening_peak_end = time_to_seconds("19:00")
    
    if morning_peak_start <= current_seconds <= morning_peak_end:
        return {
            "headway": 210,
            "status": "PEAK HOUR",
            "message": "Peak hour service - trains every 3.5 minutes",
            "period": "peak_morning"
        }
    elif evening_peak_start <= current_seconds <= evening_peak_end:
        return {
            "headway": 210,
            "status": "PEAK HOUR",
            "message": "Peak hour service - trains every 3.5 minutes",
            "period": "peak_evening"
        }
    else:
        return {
            "headway": 300,
            "status": "NORMAL",
            "message": "Normal service - trains every 5 minutes",
            "period": "off_peak"
        }
        
def calculate_next_trains(station_name: str, direction: str = None) -> dict:
    """
    Calculate next train arrivals for a station.
    Returns northbound and southbound next trains with DYNAMIC times.
    """
    now = datetime.now()
    current_seconds = now.hour * 3600 + now.minute * 60 + now.second
    
    # Define operating hours in seconds
    opening_time = time_to_seconds("04:30")
    closing_time = time_to_seconds("22:30")
    
    # Check if station is open
    if current_seconds < opening_time or current_seconds >= closing_time:
        return {
            "northbound": {"minutes": None, "time": "CLOSED", "from_station": None, "status": "closed"},
            "southbound": {"minutes": None, "time": "CLOSED", "from_station": None, "status": "closed"},
            "is_operating": False
        }
    
    # Get current headway in seconds
    headway = get_current_headway()
    
    # Find station index
    try:
        station_idx = STATIONS.index(station_name)
    except ValueError:
        station_idx = 6  # Default to Cubao area
    
    result = {}
    
    # ========== NORTHBOUND (Taft → North Ave) ==========
    # First northbound train from Taft at 5:05 AM
    first_train_taft_seconds = time_to_seconds("05:05:00")
    
    # Calculate cumulative travel times from Taft to each station
    travel_times_from_taft = {12: 0}  # Index 12 is Taft
    for i in range(11, -1, -1):  # From index 11 down to 0
        station_pair = (STATIONS[i+1], STATIONS[i])
        travel = TRAVEL_TIMES.get(station_pair, 180)
        travel_times_from_taft[i] = travel_times_from_taft[i+1] + travel + DWELL_TIME
    
    travel_from_taft = travel_times_from_taft.get(station_idx, 0)
    first_northbound_arrival = first_train_taft_seconds + travel_from_taft
    
    # Calculate next northbound train
    if current_seconds < first_northbound_arrival:
        next_north_seconds = first_northbound_arrival
    else:
        elapsed = current_seconds - first_northbound_arrival
        trains_passed = elapsed // headway
        next_north_seconds = first_northbound_arrival + (trains_passed + 1) * headway
    
    north_minutes = max(1, (next_north_seconds - current_seconds) // 60)
    
    # Find which station the northbound train is coming from
    north_source_idx = station_idx - 1
    if north_source_idx < 0:
        north_source_idx = 0
    north_source = STATIONS[north_source_idx]
    
    # ========== SOUTHBOUND (North Ave → Taft) ==========
    # First southbound train from North Ave at 4:30 AM
    first_train_north_seconds = time_to_seconds("04:30:00")
    
    # Calculate cumulative travel times from North Ave to each station
    travel_times_from_north = {0: 0}  # Index 0 is North Ave
    for i in range(0, 12):  # From index 0 to 11
        station_pair = (STATIONS[i], STATIONS[i+1])
        travel = TRAVEL_TIMES.get(station_pair, 180)
        travel_times_from_north[i+1] = travel_times_from_north[i] + travel + DWELL_TIME
    
    travel_from_north = travel_times_from_north.get(station_idx, 0)
    first_southbound_arrival = first_train_north_seconds + travel_from_north
    
    # Calculate next southbound train
    if current_seconds < first_southbound_arrival:
        next_south_seconds = first_southbound_arrival
    else:
        elapsed = current_seconds - first_southbound_arrival
        trains_passed = elapsed // headway
        next_south_seconds = first_southbound_arrival + (trains_passed + 1) * headway
    
    south_minutes = max(1, (next_south_seconds - current_seconds) // 60)
    
    # Find which station the southbound train is coming from
    south_source_idx = station_idx + 1
    if south_source_idx >= len(STATIONS):
        south_source_idx = len(STATIONS) - 1
    south_source = STATIONS[south_source_idx]
    
    # Cap maximum wait time to realistic values
    max_wait = (headway // 60) * 2
    north_minutes = min(north_minutes, max_wait)
    south_minutes = min(south_minutes, max_wait)
    
    # Adjust based on station position
    if station_name == "North Ave":
        north_minutes = max(6, north_minutes)  # Northbound from North Ave takes longer
        south_minutes = min(3, south_minutes)  # Southbound is frequent
    elif station_name == "Taft":
        north_minutes = min(3, north_minutes)  # Northbound is frequent
        south_minutes = max(6, south_minutes)  # Southbound from Taft takes longer
    
    # Ensure reasonable values
    north_minutes = max(2, min(12, north_minutes))
    south_minutes = max(2, min(12, south_minutes))
    
    result["north"] = {
        "minutes": north_minutes,
        "time": (now + timedelta(minutes=north_minutes)).strftime("%I:%M %p"),
        "from_station": north_source,
        "status": "on_time"
    }
    
    result["south"] = {
        "minutes": south_minutes,
        "time": (now + timedelta(minutes=south_minutes)).strftime("%I:%M %p"),
        "from_station": south_source,
        "status": "on_time"
    }
    
    # Debug print
    print(f"🚆 {station_name}: North={north_minutes}min (from {north_source}), South={south_minutes}min (from {south_source})")
    
    # Return appropriate result
    if direction == "north":
        return result.get("north", {})
    elif direction == "south":
        return result.get("south", {})
    else:
        return {
            "northbound": result.get("north", {}),
            "southbound": result.get("south", {}),
            "headway": headway // 60,
            "is_operating": True
        }

def calculate_travel_time(from_idx: int, to_idx: int, direction: str) -> int:
    """Calculate travel time between stations in seconds"""
    if from_idx == to_idx:
        return 0
    
    total_time = 0
    
    if direction == "north":
        # Traveling from lower index to higher (Taft → North Ave)
        for i in range(from_idx, to_idx):
            station_pair = (STATIONS[i], STATIONS[i + 1])
            if station_pair in TRAVEL_TIMES:
                total_time += TRAVEL_TIMES[station_pair]
            elif (station_pair[1], station_pair[0]) in TRAVEL_TIMES:
                total_time += TRAVEL_TIMES[(station_pair[1], station_pair[0])]
            else:
                total_time += 180  # Default 3 minutes
            total_time += DWELL_TIME  # Add dwell time at station
    else:
        # Traveling from higher index to lower (North Ave → Taft)
        for i in range(from_idx, to_idx, -1):
            station_pair = (STATIONS[i], STATIONS[i - 1])
            if station_pair in TRAVEL_TIMES:
                total_time += TRAVEL_TIMES[station_pair]
            elif (station_pair[1], station_pair[0]) in TRAVEL_TIMES:
                total_time += TRAVEL_TIMES[(station_pair[1], station_pair[0])]
            else:
                total_time += 180
            total_time += DWELL_TIME
    
    return total_time


def get_trip_schedule(from_station: str, to_station: str, time: datetime = None) -> dict:
    """Get schedule for a specific trip"""
    if time is None:
        time = datetime.now()
    
    try:
        from_idx = STATIONS.index(from_station)
        to_idx = STATIONS.index(to_station)
    except ValueError:
        return {"error": "Invalid station name"}
    
    # Determine direction
    if from_idx < to_idx:
        direction = "north"  # Actually southbound? Let's fix this
        # Actually: If from is North Ave (0) to Taft (12), that's southbound
        # Let's use the actual train direction naming
        if from_station == "North Ave" or from_idx < to_idx:
            direction_type = "southbound"
        else:
            direction_type = "northbound"
    else:
        direction_type = "northbound" if from_idx > to_idx else "southbound"
    
    # Get next train from origin
    train_info = calculate_next_trains(from_station, direction="north" if direction_type == "northbound" else "south")
    
    if not train_info.get("minutes"):
        return {
            "error": "Station closed or no trains available",
            "from_station": from_station,
            "to_station": to_station
        }
    
    departure_time = time + timedelta(minutes=train_info["minutes"])
    travel_seconds = calculate_travel_time(from_idx, to_idx, "north" if from_idx < to_idx else "south")
    arrival_time = departure_time + timedelta(seconds=travel_seconds)
    
    return {
        "from_station": from_station,
        "to_station": to_station,
        "direction": direction_type,
        "departure_time": departure_time.strftime("%I:%M %p"),
        "arrival_time": arrival_time.strftime("%I:%M %p"),
        "travel_time_minutes": travel_seconds // 60,
        "stops_between": abs(from_idx - to_idx),
        "next_train_minutes": train_info["minutes"],
        "status": "scheduled"
    }


def get_all_trains_for_station(station_name: str, limit: int = 5) -> dict:
    """Get next N trains for both directions"""
    result = calculate_next_trains(station_name)
    
    if not result.get("is_operating", True):
        return {"error": "Station closed"}
    
    # Get more detailed train list
    north_trains = []
    south_trains = []
    
    now = datetime.now()
    headway = result["headway"] * 60
    
    for i in range(limit):
        # Northbound
        north_arrival = now + timedelta(minutes=result["northbound"]["minutes"] + i * (headway // 60))
        north_trains.append({
            "time": north_arrival.strftime("%I:%M %p"),
            "minutes": result["northbound"]["minutes"] + i * (headway // 60),
            "from_station": result["northbound"].get("from_station", "Terminal")
        })
        
        # Southbound
        south_arrival = now + timedelta(minutes=result["southbound"]["minutes"] + i * (headway // 60))
        south_trains.append({
            "time": south_arrival.strftime("%I:%M %p"),
            "minutes": result["southbound"]["minutes"] + i * (headway // 60),
            "from_station": result["southbound"].get("from_station", "Terminal")
        })
    
    return {
        "station": station_name,
        "northbound": north_trains,
        "southbound": south_trains,
        "headway_minutes": result["headway"],
        "is_operating": True
    }