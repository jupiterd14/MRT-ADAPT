"""
Spam protection utilities - Pure helpers with no Flask dependencies
"""
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta

# Global dictionary to track report submissions (prevents refresh/exit spam)
report_tracker = defaultdict(list)


def track_report_submission(user_id, ip_address, station):
    """
    Track report submissions to prevent refresh/exit spam
    
    Args:
        user_id: User ID or None for anonymous
        ip_address: IP address of the submitter
        station: Station name being reported
    """
    key = f"{user_id if user_id else ip_address}_{station}"
    timestamp = time.time()
    report_tracker[key].append(timestamp)
    
    # Clean old entries (older than 1 hour)
    current_time = time.time()
    report_tracker[key] = [t for t in report_tracker[key] if current_time - t < 3600]


def is_rate_limited(user_id, ip_address, limit=3, window=3600):
    """
    Check if user has exceeded rate limit
    
    Args:
        user_id: User ID or None for anonymous
        ip_address: IP address of the requester
        limit: Maximum number of reports allowed (default: 3)
        window: Time window in seconds (default: 3600 = 1 hour)
    
    Returns:
        bool: True if rate limited, False otherwise
    """
    key = user_id if user_id else ip_address
    current_time = time.time()
    
    # Get all reports from this user/IP in the last window
    recent_reports = [t for t in report_tracker.get(key, []) if current_time - t < window]
    
    return len(recent_reports) >= limit


def is_suspicious_remarks(remarks):
    """
    Check if remarks look like spam
    
    Args:
        remarks: String of user remarks
    
    Returns:
        bool: True if remarks look suspicious, False otherwise
    """
    if not remarks:
        return False
    
    # Check for repeated characters (spam like "AAAAA")
    if re.search(r'(.)\1{10,}', remarks):
        return True
    
    # Check if remarks are all the same character repeated
    if len(set(remarks.lower())) == 1 and len(remarks) > 5:
        return True
    
    # Check for common spam patterns
    spam_patterns = [
        r'^[a-zA-Z]$',      # Single character
        r'^[0-9]+$',        # Only numbers
        r'^(.)\1+$',        # Same character repeated
        r'^(test|spam|asdf|qwerty|12345)+$',  # Common spam words
        r'^[!@#$%^&*()]+$',  # Only symbols
    ]
    
    for pattern in spam_patterns:
        if re.match(pattern, remarks, re.IGNORECASE):
            return True
    
    # Check for URL spam
    if re.search(r'https?://|www\.', remarks):
        return True
    
    # Check for excessive emojis
    emoji_count = len(re.findall(r'[\U00010000-\U0010FFFF]', remarks))
    if emoji_count > 5:
        return True
    
    return False


def check_duplicate_report(station, congestion_value, user_id, minutes=10, report_model=None):
    """
    Check if user already reported same station recently
    
    Args:
        station: Station name
        congestion_value: Reported congestion value
        user_id: User ID (database ID)
        minutes: Time window in minutes (default: 10)
        report_model: SQLAlchemy Report model (injected to avoid import)
    
    Returns:
        bool: True if duplicate found, False otherwise
    """
    if not user_id or report_model is None:
        return False
    
    time_threshold = datetime.now() - timedelta(minutes=minutes)
    
    # Check for reports with similar congestion (±15%)
    min_congestion = congestion_value - 15
    max_congestion = congestion_value + 15
    
    duplicate = report_model.query.filter(
        report_model.station == station,
        report_model.user_id == user_id,
        report_model.timestamp > time_threshold,
        report_model.reported_congestion.between(min_congestion, max_congestion)
    ).first()
    
    return duplicate is not None


def clear_old_tracker_entries(max_age_hours=1):
    """
    Clean up old entries from tracker
    
    Args:
        max_age_hours: Maximum age in hours (default: 1)
    """
    current_time = time.time()
    cutoff = current_time - (max_age_hours * 3600)
    
    for key in list(report_tracker.keys()):
        report_tracker[key] = [t for t in report_tracker[key] if t > cutoff]
        if not report_tracker[key]:
            del report_tracker[key]


def get_tracker_stats():
    """
    Get statistics about the report tracker
    
    Returns:
        dict: Statistics about tracked reports
    """
    total_entries = sum(len(v) for v in report_tracker.values())
    unique_keys = len(report_tracker)
    
    return {
        'total_tracked_reports': total_entries,
        'unique_keys': unique_keys,
        'keys': list(report_tracker.keys())[:10]  # First 10 keys
    }