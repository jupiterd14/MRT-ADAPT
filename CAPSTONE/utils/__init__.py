"""
Utility functions - Pure helpers with no business logic
"""
from .spam_protection import (
    track_report_submission,
    is_rate_limited,
    is_suspicious_remarks,
    check_duplicate_report,
    report_tracker
)
from .audit import log_activity, get_user_ip, get_user_agent
from .station_helpers import (
    get_operator_stations,
    get_station_list,
    get_capacity,
    get_station_index,
    get_station_coordinates,
    get_adjacent_stations,
    STATIONS,
    STATION_BASE_CAPACITY,
    STATION_COORDINATES
)

__all__ = [
    'track_report_submission',
    'is_rate_limited',
    'is_suspicious_remarks',
    'check_duplicate_report',
    'report_tracker',
    'log_activity',
    'get_user_ip',
    'get_user_agent',
    'get_operator_stations',
    'get_station_list',
    'get_capacity',
    'get_station_index',
    'get_station_coordinates',
    'get_adjacent_stations',
    'STATIONS',
    'STATION_BASE_CAPACITY',
    'STATION_COORDINATES'
]