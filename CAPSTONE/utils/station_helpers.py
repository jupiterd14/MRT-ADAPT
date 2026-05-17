"""
Station helper utilities - Pure data helpers with no business logic
"""
import json

# Station list (consistent across the application)
STATIONS = [
    "North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
    "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
    "Ayala Ave", "Magallanes", "Taft"
]

# Station base capacities (ridership capacity per station)
STATION_BASE_CAPACITY = {
    "North Ave": 12000,
    "Quezon Ave": 9000,
    "Kamuning": 7500,
    "Cubao": 15000,
    "Santolan": 8000,
    "Ortigas": 9500,
    "Shaw Blvd": 11000,
    "Boni Ave": 8500,
    "Guadalupe": 10000,
    "Buendia": 9000,
    "Ayala Ave": 14000,
    "Magallanes": 9000,
    "Taft": 16000
}

# Station coordinates for maps
STATION_COORDINATES = {
    "North Ave": {"lat": 14.6556, "lng": 121.0302},
    "Quezon Ave": {"lat": 14.6390, "lng": 121.0380},
    "Kamuning": {"lat": 14.6249, "lng": 121.0431},
    "Cubao": {"lat": 14.6213, "lng": 121.0529},
    "Santolan": {"lat": 14.6135, "lng": 121.0630},
    "Ortigas": {"lat": 14.5864, "lng": 121.0565},
    "Shaw Blvd": {"lat": 14.5789, "lng": 121.0532},
    "Boni Ave": {"lat": 14.5716, "lng": 121.0492},
    "Guadalupe": {"lat": 14.5655, "lng": 121.0446},
    "Buendia": {"lat": 14.5547, "lng": 121.0329},
    "Ayala Ave": {"lat": 14.5497, "lng": 121.0305},
    "Magallanes": {"lat": 14.5450, "lng": 121.0254},
    "Taft": {"lat": 14.5378, "lng": 121.0112}
}

# Zone definitions for operators
ZONES = {
    'north': ['North Ave', 'Quezon Ave', 'Kamuning', 'Cubao', 'Santolan'],
    'central': ['Ortigas', 'Shaw Blvd', 'Boni Ave', 'Guadalupe'],
    'south': ['Buendia', 'Ayala Ave', 'Magallanes', 'Taft']
}


def get_station_list():
    """
    Get list of all stations
    
    Returns:
        list: List of station names
    """
    return STATIONS.copy()


def get_capacity(station_name):
    """
    Get capacity for a station
    
    Args:
        station_name: Name of the station
    
    Returns:
        int: Station capacity or default 10000
    """
    return STATION_BASE_CAPACITY.get(station_name, 10000)


def get_station_index(station_name):
    """
    Get index of a station in the list
    
    Args:
        station_name: Name of the station
    
    Returns:
        int: Index position or -1 if not found
    """
    try:
        return STATIONS.index(station_name)
    except ValueError:
        return -1


def get_adjacent_stations(station_name):
    """
    Get previous and next stations for a given station
    
    Args:
        station_name: Name of the station
    
    Returns:
        tuple: (previous_station, next_station)
    """
    idx = get_station_index(station_name)
    if idx == -1:
        return None, None
    
    prev_station = STATIONS[idx - 1] if idx > 0 else None
    next_station = STATIONS[idx + 1] if idx + 1 < len(STATIONS) else None
    
    return prev_station, next_station


def get_station_coordinates(station_name):
    """
    Get coordinates for a station
    
    Args:
        station_name: Name of the station
    
    Returns:
        dict: {'lat': float, 'lng': float} or None if not found
    """
    return STATION_COORDINATES.get(station_name)


def get_stations_in_zone(zone_name):
    """
    Get list of stations in a zone
    
    Args:
        zone_name: 'north', 'central', or 'south'
    
    Returns:
        list: List of station names in the zone
    """
    return ZONES.get(zone_name, []).copy()


def get_operator_stations(user=None, user_model=None):
    """
    Get list of stations assigned to an operator
    
    Args:
        user: User object (injected)
        user_model: User model class (injected for type checking)
    
    Returns:
        list: List of station names assigned to the operator
    """
    if user is None:
        return []
    
    # Get user type if model provided
    if user_model and not isinstance(user, user_model):
        return []
    
    # LINE-WIDE ACCESS - return ALL stations
    if hasattr(user, 'access_level') and user.access_level == 'line_wide':
        return STATIONS.copy()
    
    # ZONE ACCESS
    if hasattr(user, 'access_level') and user.access_level == 'zone':
        zone = getattr(user, 'assigned_zone', None)
        if zone and zone in ZONES:
            return ZONES[zone].copy()
        return []
    
    # STATION-LEVEL ACCESS
    stations = []
    
    # First check assigned_stations (JSON field)
    if hasattr(user, 'assigned_stations') and user.assigned_stations:
        try:
            stations = json.loads(user.assigned_stations)
            if stations and isinstance(stations, list):
                return stations
        except:
            pass
    
    # Fallback to favorite_station
    if hasattr(user, 'favorite_station') and user.favorite_station:
        if user.favorite_station in STATIONS:
            return [user.favorite_station]
    
    # Ultimate fallback
    return ['North Ave']


def validate_station(station_name):
    """
    Validate if a station name exists
    
    Args:
        station_name: Name to validate
    
    Returns:
        bool: True if station exists
    """
    return station_name in STATIONS


def get_all_station_data():
    """
    Get all station data in one dictionary
    
    Returns:
        dict: Dictionary with all station information
    """
    stations_data = {}
    for station in STATIONS:
        stations_data[station] = {
            'name': station,
            'capacity': get_capacity(station),
            'index': get_station_index(station),
            'coordinates': get_station_coordinates(station),
            'adjacent': {
                'previous': get_adjacent_stations(station)[0],
                'next': get_adjacent_stations(station)[1]
            }
        }
    return stations_data


def get_stations_by_capacity(min_capacity=None, max_capacity=None):
    """
    Get stations filtered by capacity range
    
    Args:
        min_capacity: Minimum capacity (optional)
        max_capacity: Maximum capacity (optional)
    
    Returns:
        list: List of station names matching the criteria
    """
    result = []
    for station, capacity in STATION_BASE_CAPACITY.items():
        if min_capacity is not None and capacity < min_capacity:
            continue
        if max_capacity is not None and capacity > max_capacity:
            continue
        result.append(station)
    return result


def get_station_type(station_name):
    """
    Get station type based on capacity
    
    Args:
        station_name: Name of the station
    
    Returns:
        str: 'major', 'medium', or 'minor'
    """
    capacity = get_capacity(station_name)
    
    if capacity >= 14000:
        return 'major'
    elif capacity >= 9000:
        return 'medium'
    else:
        return 'minor'