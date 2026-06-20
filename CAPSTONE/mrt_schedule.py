# mrt_schedule.py
# MRT Train Schedule Data

# Regular Weekdays Train Schedule
WEEKDAY_TRAIN_SCHEDULE = {
    'Morning': {'start': '4:30 AM', 'end': '7:00 AM', 'trains': 14, 'headway': '7 - 4 Minutes'},
    'AM Peak': {'start': '7:01 AM', 'end': '9:00 AM', 'trains': 19, 'headway': '3.5 Minutes'},
    'Off Peak': {'start': '9:01 AM', 'end': '5:00 PM', 'trains': 14, 'headway': '5 - 5.5 Minutes'},
    'PM Peak': {'start': '5:01 PM', 'end': '7:00 PM', 'trains': 19, 'headway': '3.5 Minutes'},
    'Night': {'start': '7:01 PM', 'end': '9:30 PM', 'trains': 14, 'headway': '5 - 8 Minutes'},
    'Extended': {'start': '9:31 PM', 'end': '11:40 PM', 'trains': 4, 'headway': '15 Minutes'}
}

# Regular Weekends Train Schedule
WEEKEND_TRAIN_SCHEDULE = {
    'Saturday Morning': {'start': '4:30 AM', 'end': '5:00 PM', 'trains': 14, 'headway': '5.5 - 6 Minutes'},
    'Saturday Afternoon': {'start': '5:01 PM', 'end': '7:00 PM', 'trains': 16, 'headway': '5 - 5.5 Minutes'},
    'Saturday Evening': {'start': '7:01 PM', 'end': '10:40 PM', 'trains': 12, 'headway': '6.5 - 7 Minutes'},
    'Sunday/Holidays': {'start': '4:30 AM', 'end': '10:40 PM', 'trains': 12, 'headway': '6.5 - 7 Minutes'}
}

# Station Entrance Opening Schedule - Weekday
WEEKDAY_ENTRANCE_SCHEDULE = {
    'North Avenue': {'southbound': '04:20 AM', 'northbound': '05:24 AM'},
    'Quezon Avenue': {'southbound': '04:22 AM', 'northbound': '05:23 AM'},
    'GMA Kamuning': {'southbound': '04:24 AM', 'northbound': '05:21 AM'},
    'Araneta Cubao': {'southbound': '04:27 AM', 'northbound': '05:17 AM'},
    'Santolan': {'southbound': '04:30 AM', 'northbound': '05:14 AM'},
    'Ortigas': {'southbound': '04:33 AM', 'northbound': '05:11 AM'},
    'Shaw Boulevard': {'southbound': '04:36 AM', 'northbound': '05:10 AM'},
    'Boni': {'southbound': '04:38 AM', 'northbound': '05:07 AM'},
    'Guadalupe': {'southbound': '04:40 AM', 'northbound': '05:06 AM'},
    'Buendia': {'southbound': '04:43 AM', 'northbound': '05:02 AM'},
    'Ayala': {'southbound': '04:45 AM', 'northbound': '05:00 AM'},
    'Magallanes': {'southbound': '04:47 AM', 'northbound': '04:58 AM'},
    'Taft Avenue': {'southbound': '04:50 AM', 'northbound': '04:55 AM'}
}

# Station Entrance Opening Schedule - Weekend
WEEKEND_ENTRANCE_SCHEDULE = {
    'North Avenue': {'southbound': '04:20 AM', 'northbound': '05:27 AM'},
    'Quezon Avenue': {'southbound': '04:22 AM', 'northbound': '05:25 AM'},
    'GMA Kamuning': {'southbound': '04:24 AM', 'northbound': '05:22 AM'},
    'Araneta Cubao': {'southbound': '04:28 AM', 'northbound': '05:19 AM'},
    'Santolan': {'southbound': '04:31 AM', 'northbound': '05:15 AM'},
    'Ortigas': {'southbound': '04:34 AM', 'northbound': '05:12 AM'},
    'Shaw Boulevard': {'southbound': '04:36 AM', 'northbound': '05:10 AM'},
    'Boni': {'southbound': '04:38 AM', 'northbound': '05:08 AM'},
    'Guadalupe': {'southbound': '04:40 AM', 'northbound': '05:06 AM'},
    'Buendia': {'southbound': '04:44 AM', 'northbound': '05:03 AM'},
    'Ayala': {'southbound': '04:46 AM', 'northbound': '05:01 AM'},
    'Magallanes': {'southbound': '04:49 AM', 'northbound': '04:58 AM'},
    'Taft Avenue': {'southbound': '04:52 AM', 'northbound': '04:55 AM'}
}

# List of all stations (southbound order from North Avenue to Taft)
STATIONS = [
    'North Avenue', 'Quezon Avenue', 'GMA Kamuning', 'Araneta Cubao',
    'Santolan', 'Ortigas', 'Shaw Boulevard', 'Boni', 'Guadalupe',
    'Buendia', 'Ayala', 'Magallanes', 'Taft Avenue'
]

# Helper function to convert time string to minutes since midnight
def time_to_minutes(time_str):
    """Convert time string (e.g., '04:30 AM') to minutes since midnight"""
    time_str = time_str.strip()
    parts = time_str.split()
    time_part = parts[0]
    period = parts[1] if len(parts) > 1 else ''
    
    hour, minute = map(int, time_part.split(':'))
    
    if period == 'PM' and hour != 12:
        hour += 12
    elif period == 'AM' and hour == 12:
        hour = 0
    
    return hour * 60 + minute

# Function to get current headway based on time and day
def get_current_headway(current_time, day_type='weekday'):
    """
    Get the headway for the current time
    current_time: datetime.time object or string in 'HH:MM AM/PM' format
    day_type: 'weekday' or 'weekend'
    Returns: headway string (e.g., '3.5 Minutes')
    """
    if isinstance(current_time, str):
        current_minutes = time_to_minutes(current_time)
    else:
        current_minutes = current_time.hour * 60 + current_time.minute
    
    schedule = WEEKDAY_TRAIN_SCHEDULE if day_type == 'weekday' else WEEKEND_TRAIN_SCHEDULE
    
    for period, details in schedule.items():
        start_min = time_to_minutes(details['start'])
        end_min = time_to_minutes(details['end'])
        
        if start_min <= current_minutes <= end_min:
            return details['headway']
    
    # Default if outside operating hours
    return "Service not available"

# Function to get next train time
def get_next_train_time(current_time, station, direction, day_type='weekday'):
    """
    Get the next train time from a specific station
    Returns: string with next train time
    """
    # This is a simplified version - you can expand this based on your needs
    entrance_schedule = WEEKDAY_ENTRANCE_SCHEDULE if day_type == 'weekday' else WEEKEND_ENTRANCE_SCHEDULE
    
    if station in entrance_schedule:
        if direction.lower() == 'southbound':
            return entrance_schedule[station]['southbound']
        else:
            return entrance_schedule[station]['northbound']
    return "Schedule not available"

# Function to check if station is open
def is_station_open(current_time, station, day_type='weekday'):
    """
    Check if a station entrance is open at the given time
    Returns: boolean
    """
    if isinstance(current_time, str):
        current_minutes = time_to_minutes(current_time)
    else:
        current_minutes = current_time.hour * 60 + current_time.minute
    
    entrance_schedule = get_entrance_schedule(day_type)
    
    if station in entrance_schedule:
        south_min = time_to_minutes(entrance_schedule[station]['southbound'])
        # Assume stations close at 11:40 PM (last train)
        close_min = time_to_minutes('11:40 PM')
        
        return south_min <= current_minutes <= close_min
    
    return False

# Helper function to get train schedule based on day type and time
def get_train_schedule(day_type='weekday'):
    """Returns the appropriate train schedule based on day type"""
    if day_type.lower() == 'weekday':
        return WEEKDAY_TRAIN_SCHEDULE
    else:
        return WEEKEND_TRAIN_SCHEDULE

# Helper function to get entrance schedule based on day type
def get_entrance_schedule(day_type='weekday'):
    """Returns the appropriate entrance schedule based on day type"""
    if day_type.lower() == 'weekday':
        return WEEKDAY_ENTRANCE_SCHEDULE
    else:
        return WEEKEND_ENTRANCE_SCHEDULE
    
    # Add this function to your mrt_schedule.py file

def get_headway_info(period_key, day_type='weekday'):
    """
    Get headway information for a specific period
    period_key: e.g., 'Morning', 'AM Peak', 'Off Peak', 'PM Peak', 'Night', 'Extended'
    day_type: 'weekday' or 'weekend'
    Returns: dict with headway info or None if not found
    """
    schedule = WEEKDAY_TRAIN_SCHEDULE if day_type == 'weekday' else WEEKEND_TRAIN_SCHEDULE
    
    # Handle different period naming conventions
    period_mapping = {
        'Morning': ['Morning', 'Saturday Morning'],
        'AM Peak': ['AM Peak'],
        'Off Peak': ['Off Peak', 'Saturday Afternoon'],
        'PM Peak': ['PM Peak'],
        'Night': ['Night', 'Saturday Evening'],
        'Extended': ['Extended'],
        'Sunday/Holidays': ['Sunday/Holidays']
    }
    
    # Find matching period
    possible_keys = period_mapping.get(period_key, [period_key])
    
    for key in possible_keys:
        if key in schedule:
            info = schedule[key]
            return {
                'period': key,
                'start': info['start'],
                'end': info['end'],
                'trains': info['trains'],
                'headway': info['headway']
            }
    
    return None

# Also keep your existing get_current_headway function
def get_current_headway(current_time, day_type='weekday'):
    """
    Get the headway for the current time
    current_time: datetime.time object or string in 'HH:MM AM/PM' format
    day_type: 'weekday' or 'weekend'
    Returns: headway string (e.g., '3.5 Minutes')
    """
    if isinstance(current_time, str):
        current_minutes = time_to_minutes(current_time)
    else:
        current_minutes = current_time.hour * 60 + current_time.minute
    
    schedule = WEEKDAY_TRAIN_SCHEDULE if day_type == 'weekday' else WEEKEND_TRAIN_SCHEDULE
    
    for period, details in schedule.items():
        start_min = time_to_minutes(details['start'])
        end_min = time_to_minutes(details['end'])
        
        if start_min <= current_minutes <= end_min:
            return details['headway']
    
    # Default if outside operating hours
    return "Service not available"


def calculate_next_trains(current_time, station, direction='southbound', day_type='weekday'):
    """
    Calculate the next train times from a specific station
    current_time: datetime.time object or string in 'HH:MM AM/PM' format
    station: station name (e.g., 'North Avenue')
    direction: 'southbound' or 'northbound'
    day_type: 'weekday' or 'weekend'
    Returns: list of next train times (up to 3)
    """
    if isinstance(current_time, str):
        current_minutes = time_to_minutes(current_time)
    else:
        current_minutes = current_time.hour * 60 + current_time.minute
    
    # Get the entrance schedule
    entrance_schedule = get_entrance_schedule(day_type)
    
    if station not in entrance_schedule:
        return ["Station not found"]
    
    # Get the first train time for this station and direction
    if direction.lower() == 'southbound':
        first_train_str = entrance_schedule[station]['southbound']
    else:
        first_train_str = entrance_schedule[station]['northbound']
    
    first_train_minutes = time_to_minutes(first_train_str)
    
    # If current time is before first train, first train is the next
    if current_minutes < first_train_minutes:
        next_times = [first_train_str]
    else:
        # Get current headway to calculate subsequent trains
        headway_str = get_current_headway(current_time, day_type)
        
        # Parse headway (handle ranges like '7 - 4 Minutes' or simple like '3.5 Minutes')
        if ' - ' in headway_str:
            # For ranges, use the average or the lower value
            parts = headway_str.split(' - ')
            headway_min = float(parts[0].split()[0])
        elif ' ' in headway_str:
            headway_min = float(headway_str.split()[0])
        else:
            headway_min = 5  # default fallback
        
        # Calculate minutes after first train
        minutes_elapsed = current_minutes - first_train_minutes
        trains_passed = int(minutes_elapsed / headway_min) if headway_min > 0 else 0
        
        # Calculate next train times
        next_times = []
        for i in range(1, 4):  # Get next 3 trains
            next_train_minutes = first_train_minutes + (trains_passed + i) * headway_min
            
            # Check if within operating hours (until 11:40 PM)
            if next_train_minutes > time_to_minutes('11:40 PM'):
                next_times.append("Last train passed")
            else:
                # Convert back to time string
                hours = next_train_minutes // 60
                minutes = next_train_minutes % 60
                period = "AM" if hours < 12 else "PM"
                if hours > 12:
                    hours -= 12
                elif hours == 0:
                    hours = 12
                
                time_str = f"{hours:02d}:{minutes:02d} {period}"
                next_times.append(time_str)
    
    return next_times


def get_headway_info(period_key, day_type='weekday'):
    """
    Get headway information for a specific period
    period_key: e.g., 'Morning', 'AM Peak', 'Off Peak', 'PM Peak', 'Night', 'Extended'
    day_type: 'weekday' or 'weekend'
    Returns: dict with headway info or None if not found
    """
    schedule = WEEKDAY_TRAIN_SCHEDULE if day_type == 'weekday' else WEEKEND_TRAIN_SCHEDULE
    
    # Handle different period naming conventions
    period_mapping = {
        'Morning': ['Morning', 'Saturday Morning'],
        'AM Peak': ['AM Peak'],
        'Off Peak': ['Off Peak', 'Saturday Afternoon'],
        'PM Peak': ['PM Peak'],
        'Night': ['Night', 'Saturday Evening'],
        'Extended': ['Extended'],
        'Sunday/Holidays': ['Sunday/Holidays']
    }
    
    # Find matching period
    possible_keys = period_mapping.get(period_key, [period_key])
    
    for key in possible_keys:
        if key in schedule:
            info = schedule[key]
            return {
                'period': key,
                'start': info['start'],
                'end': info['end'],
                'trains': info['trains'],
                'headway': info['headway']
            }
    
    return None

# Add this function to your mrt_schedule.py file

def get_trip_schedule(origin_station, destination_station, current_time, day_type='weekday'):
    """
    Get the trip schedule between two stations
    origin_station: starting station
    destination_station: ending station  
    current_time: current time (datetime.time object or string)
    day_type: 'weekday' or 'weekend'
    Returns: dict with trip information including estimated travel time and next trains
    """
    if isinstance(current_time, str):
        current_minutes = time_to_minutes(current_time)
    else:
        current_minutes = current_time.hour * 60 + current_time.minute
    
    # Check if stations exist
    if origin_station not in STATIONS or destination_station not in STATIONS:
        return {"error": "Station not found"}
    
    # Get indices for travel time calculation
    origin_index = STATIONS.index(origin_station)
    dest_index = STATIONS.index(destination_station)
    
    # Calculate number of stations between (assuming southbound direction)
    stations_between = abs(dest_index - origin_index)
    
    # Estimate travel time (approx 2-3 minutes per station including stops)
    # Average travel time between stations: ~2.5 minutes
    estimated_travel_minutes = stations_between * 2.5
    
    # Get next train times from origin station
    next_trains = calculate_next_trains(current_time, origin_station, 'southbound', day_type)
    
    # Get current headway
    current_headway = get_current_headway(current_time, day_type)
    
    # Get entrance schedule for origin station
    entrance_schedule = get_entrance_schedule(day_type)
    first_train = entrance_schedule.get(origin_station, {}).get('southbound', 'Unknown')
    
    return {
        'origin': origin_station,
        'destination': destination_station,
        'stations_between': stations_between,
        'estimated_travel_time': f"{estimated_travel_minutes:.1f} minutes",
        'estimated_travel_time_minutes': estimated_travel_minutes,
        'next_trains': next_trains,
        'current_headway': current_headway,
        'first_train': first_train,
        'day_type': day_type
    }


def calculate_next_trains(current_time, station, direction='southbound', day_type='weekday'):
    """
    Calculate the next train times from a specific station
    current_time: datetime.time object or string in 'HH:MM AM/PM' format
    station: station name (e.g., 'North Avenue')
    direction: 'southbound' or 'northbound'
    day_type: 'weekday' or 'weekend'
    Returns: list of next train times (up to 3)
    """
    if isinstance(current_time, str):
        current_minutes = time_to_minutes(current_time)
    else:
        current_minutes = current_time.hour * 60 + current_time.minute
    
    # Get the entrance schedule
    entrance_schedule = get_entrance_schedule(day_type)
    
    if station not in entrance_schedule:
        return ["Station not found"]
    
    # Get the first train time for this station and direction
    if direction.lower() == 'southbound':
        first_train_str = entrance_schedule[station]['southbound']
    else:
        first_train_str = entrance_schedule[station]['northbound']
    
    first_train_minutes = time_to_minutes(first_train_str)
    
    # If current time is before first train, first train is the next
    if current_minutes < first_train_minutes:
        next_times = [first_train_str]
        # Add subsequent trains based on headway
        headway_str = get_current_headway(first_train_str, day_type)
    else:
        # Get current headway to calculate subsequent trains
        headway_str = get_current_headway(current_time, day_type)
        
        # Parse headway (handle ranges like '7 - 4 Minutes' or simple like '3.5 Minutes')
        if ' - ' in headway_str:
            # For ranges, use the average or the lower value
            parts = headway_str.split(' - ')
            headway_min = float(parts[0].split()[0])
        elif ' ' in headway_str:
            headway_min = float(headway_str.split()[0])
        else:
            headway_min = 5  # default fallback
        
        # Calculate minutes after first train
        minutes_elapsed = current_minutes - first_train_minutes
        trains_passed = int(minutes_elapsed / headway_min) if headway_min > 0 else 0
        
        # Calculate next train times
        next_times = []
        for i in range(1, 4):  # Get next 3 trains
            next_train_minutes = first_train_minutes + (trains_passed + i) * headway_min
            
            # Check if within operating hours (until 11:40 PM)
            if next_train_minutes > time_to_minutes('11:40 PM'):
                next_times.append("Last train passed")
            else:
                # Convert back to time string
                hours = next_train_minutes // 60
                minutes = next_train_minutes % 60
                period = "AM" if hours < 12 else "PM"
                if hours > 12:
                    hours -= 12
                elif hours == 0:
                    hours = 12
                
                time_str = f"{hours:02d}:{minutes:02d} {period}"
                next_times.append(time_str)
        
        return next_times
    
    # If we got here (current time before first train), calculate based on first train
    headway_min = 5  # default
    if ' - ' in headway_str:
        headway_min = float(headway_str.split(' - ')[0].split()[0])
    elif ' ' in headway_str:
        headway_min = float(headway_str.split()[0])
    
    next_times = [first_train_str]
    for i in range(1, 3):
        next_minutes = first_train_minutes + (i * headway_min)
        if next_minutes <= time_to_minutes('11:40 PM'):
            hours = next_minutes // 60
            minutes = next_minutes % 60
            period = "AM" if hours < 12 else "PM"
            if hours > 12:
                hours -= 12
            elif hours == 0:
                hours = 12
            time_str = f"{hours:02d}:{minutes:02d} {period}"
            next_times.append(time_str)
    
    return next_times


def get_headway_info(period_key, day_type='weekday'):
    """
    Get headway information for a specific period
    period_key: e.g., 'Morning', 'AM Peak', 'Off Peak', 'PM Peak', 'Night', 'Extended'
    day_type: 'weekday' or 'weekend'
    Returns: dict with headway info or None if not found
    """
    schedule = WEEKDAY_TRAIN_SCHEDULE if day_type == 'weekday' else WEEKEND_TRAIN_SCHEDULE
    
    # Handle different period naming conventions
    period_mapping = {
        'Morning': ['Morning', 'Saturday Morning'],
        'AM Peak': ['AM Peak'],
        'Off Peak': ['Off Peak', 'Saturday Afternoon'],
        'PM Peak': ['PM Peak'],
        'Night': ['Night', 'Saturday Evening'],
        'Extended': ['Extended'],
        'Sunday/Holidays': ['Sunday/Holidays']
    }
    
    # Find matching period
    possible_keys = period_mapping.get(period_key, [period_key])
    
    for key in possible_keys:
        if key in schedule:
            info = schedule[key]
            return {
                'period': key,
                'start': info['start'],
                'end': info['end'],
                'trains': info['trains'],
                'headway': info['headway']
            }
    
    return None


# Also keep your existing get_current_headway function
def get_current_headway(current_time, day_type='weekday'):
    """
    Get the headway for the current time
    current_time: datetime.time object or string in 'HH:MM AM/PM' format
    day_type: 'weekday' or 'weekend'
    Returns: headway string (e.g., '3.5 Minutes')
    """
    if isinstance(current_time, str):
        current_minutes = time_to_minutes(current_time)
    else:
        current_minutes = current_time.hour * 60 + current_time.minute
    
    schedule = WEEKDAY_TRAIN_SCHEDULE if day_type == 'weekday' else WEEKEND_TRAIN_SCHEDULE
    
    for period, details in schedule.items():
        start_min = time_to_minutes(details['start'])
        end_min = time_to_minutes(details['end'])
        
        if start_min <= current_minutes <= end_min:
            return details['headway']
    
    # Default if outside operating hours
    return "Service not available"

# Add this function to your mrt_schedule.py file

def get_all_trains_for_station(station, day_type='weekday'):
    """
    Get all train schedules for a specific station for the entire day
    station: station name
    day_type: 'weekday' or 'weekend'
    Returns: dict with all train times throughout the day
    """
    entrance_schedule = get_entrance_schedule(day_type)
    
    if station not in entrance_schedule:
        return {"error": "Station not found"}
    
    # Get first and last train times for this station
    first_train_sb = entrance_schedule[station]['southbound']
    first_train_nb = entrance_schedule[station]['northbound']
    
    # Last train from Taft (southbound end) is 9:30 PM for weekdays? 
    # Based on schedule, last train from North Ave is 9:30 PM (Night period ends)
    last_train_sb = "9:30 PM"  # Last southbound train from North Ave
    last_train_nb = "11:40 PM"  # Last northbound train from Taft (Extended period)
    
    # Get all train periods with their headways
    schedule = get_train_schedule(day_type)
    
    # Calculate approximate train times throughout the day
    all_trains = {
        'station': station,
        'day_type': day_type,
        'first_train': {
            'southbound': first_train_sb,
            'northbound': first_train_nb
        },
        'last_train': {
            'southbound': last_train_sb,
            'northbound': last_train_nb
        },
        'periods': [],
        'estimated_train_frequency': {}
    }
    
    # Add period information
    for period, details in schedule.items():
        period_info = {
            'period': period,
            'start': details['start'],
            'end': details['end'],
            'headway': details['headway'],
            'trains_per_hour': calculate_trains_per_hour(details['headway'])
        }
        all_trains['periods'].append(period_info)
        all_trains['estimated_train_frequency'][period] = period_info['trains_per_hour']
    
    return all_trains


def calculate_trains_per_hour(headway_str):
    """
    Calculate approximate number of trains per hour based on headway
    headway_str: e.g., '3.5 Minutes', '5 - 5.5 Minutes'
    Returns: float (trains per hour)
    """
    # Parse headway string to get minutes
    if ' - ' in headway_str:
        # For ranges, use the average
        parts = headway_str.split(' - ')
        min_minutes = float(parts[0].split()[0])
        max_minutes = float(parts[1].split()[0])
        avg_minutes = (min_minutes + max_minutes) / 2
    else:
        avg_minutes = float(headway_str.split()[0])
    
    # Calculate trains per hour (60 minutes / headway in minutes)
    if avg_minutes > 0:
        trains_per_hour = 60 / avg_minutes
        return round(trains_per_hour, 1)
    return 0


def get_trip_schedule(origin_station, destination_station, current_time, day_type='weekday'):
    """
    Get the trip schedule between two stations
    origin_station: starting station
    destination_station: ending station  
    current_time: current time (datetime.time object or string)
    day_type: 'weekday' or 'weekend'
    Returns: dict with trip information including estimated travel time and next trains
    """
    if isinstance(current_time, str):
        current_minutes = time_to_minutes(current_time)
    else:
        current_minutes = current_time.hour * 60 + current_time.minute
    
    # Check if stations exist
    if origin_station not in STATIONS or destination_station not in STATIONS:
        return {"error": "Station not found"}
    
    # Get indices for travel time calculation
    origin_index = STATIONS.index(origin_station)
    dest_index = STATIONS.index(destination_station)
    
    # Calculate number of stations between (assuming southbound direction)
    stations_between = abs(dest_index - origin_index)
    
    # Estimate travel time (approx 2-3 minutes per station including stops)
    # Average travel time between stations: ~2.5 minutes
    estimated_travel_minutes = stations_between * 2.5
    
    # Get next train times from origin station
    next_trains = calculate_next_trains(current_time, origin_station, 'southbound', day_type)
    
    # Get current headway
    current_headway = get_current_headway(current_time, day_type)
    
    # Get entrance schedule for origin station
    entrance_schedule = get_entrance_schedule(day_type)
    first_train = entrance_schedule.get(origin_station, {}).get('southbound', 'Unknown')
    
    return {
        'origin': origin_station,
        'destination': destination_station,
        'stations_between': stations_between,
        'estimated_travel_time': f"{estimated_travel_minutes:.1f} minutes",
        'estimated_travel_time_minutes': estimated_travel_minutes,
        'next_trains': next_trains,
        'current_headway': current_headway,
        'first_train': first_train,
        'day_type': day_type
    }


def calculate_next_trains(current_time, station, direction='southbound', day_type='weekday'):
    """
    Calculate the next train times from a specific station
    current_time: datetime.time object or string in 'HH:MM AM/PM' format
    station: station name (e.g., 'North Avenue')
    direction: 'southbound' or 'northbound'
    day_type: 'weekday' or 'weekend'
    Returns: list of next train times (up to 3)
    """
    if isinstance(current_time, str):
        current_minutes = time_to_minutes(current_time)
    else:
        current_minutes = current_time.hour * 60 + current_time.minute
    
    # Get the entrance schedule
    entrance_schedule = get_entrance_schedule(day_type)
    
    if station not in entrance_schedule:
        return ["Station not found"]
    
    # Get the first train time for this station and direction
    if direction.lower() == 'southbound':
        first_train_str = entrance_schedule[station]['southbound']
    else:
        first_train_str = entrance_schedule[station]['northbound']
    
    first_train_minutes = time_to_minutes(first_train_str)
    
    # If current time is before first train, first train is the next
    if current_minutes < first_train_minutes:
        next_times = [first_train_str]
        # Add subsequent trains based on headway from the first period
        # Find headway for the first period of the day
        schedule = get_train_schedule(day_type)
        first_period = list(schedule.keys())[0]
        headway_str = schedule[first_period]['headway']
    else:
        # Get current headway to calculate subsequent trains
        headway_str = get_current_headway(current_time, day_type)
    
    # Parse headway (handle ranges like '7 - 4 Minutes' or simple like '3.5 Minutes')
    if headway_str == "Service not available":
        return ["No more trains today"]
    
    if ' - ' in headway_str:
        # For ranges, use the average or the lower value
        parts = headway_str.split(' - ')
        headway_min = float(parts[0].split()[0])
    elif ' ' in headway_str:
        headway_min = float(headway_str.split()[0])
    else:
        headway_min = 5  # default fallback
    
    if current_minutes < first_train_minutes:
        # Calculate subsequent trains from first train
        next_times = [first_train_str]
        for i in range(1, 3):
            next_minutes = first_train_minutes + (i * headway_min)
            if next_minutes <= time_to_minutes('11:40 PM'):
                hours = next_minutes // 60
                minutes = next_minutes % 60
                period = "AM" if hours < 12 else "PM"
                if hours > 12:
                    hours -= 12
                elif hours == 0:
                    hours = 12
                time_str = f"{hours:02d}:{minutes:02d} {period}"
                next_times.append(time_str)
        return next_times
    else:
        # Calculate minutes after first train
        minutes_elapsed = current_minutes - first_train_minutes
        trains_passed = int(minutes_elapsed / headway_min) if headway_min > 0 else 0
        
        # Calculate next train times
        next_times = []
        for i in range(1, 4):  # Get next 3 trains
            next_train_minutes = first_train_minutes + (trains_passed + i) * headway_min
            
            # Check if within operating hours (until 11:40 PM)
            if next_train_minutes > time_to_minutes('11:40 PM'):
                next_times.append("Last train passed")
            else:
                # Convert back to time string
                hours = next_train_minutes // 60
                minutes = next_train_minutes % 60
                period = "AM" if hours < 12 else "PM"
                if hours > 12:
                    hours -= 12
                elif hours == 0:
                    hours = 12
                
                time_str = f"{hours:02d}:{minutes:02d} {period}"
                next_times.append(time_str)
        
        return next_times


def get_headway_info(period_key, day_type='weekday'):
    """
    Get headway information for a specific period
    period_key: e.g., 'Morning', 'AM Peak', 'Off Peak', 'PM Peak', 'Night', 'Extended'
    day_type: 'weekday' or 'weekend'
    Returns: dict with headway info or None if not found
    """
    schedule = WEEKDAY_TRAIN_SCHEDULE if day_type == 'weekday' else WEEKEND_TRAIN_SCHEDULE
    
    # Handle different period naming conventions
    period_mapping = {
        'Morning': ['Morning', 'Saturday Morning'],
        'AM Peak': ['AM Peak'],
        'Off Peak': ['Off Peak', 'Saturday Afternoon'],
        'PM Peak': ['PM Peak'],
        'Night': ['Night', 'Saturday Evening'],
        'Extended': ['Extended'],
        'Sunday/Holidays': ['Sunday/Holidays']
    }
    
    # Find matching period
    possible_keys = period_mapping.get(period_key, [period_key])
    
    for key in possible_keys:
        if key in schedule:
            info = schedule[key]
            return {
                'period': key,
                'start': info['start'],
                'end': info['end'],
                'trains': info['trains'],
                'headway': info['headway']
            }
    
    return None