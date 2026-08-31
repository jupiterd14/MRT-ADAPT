"""
Audit logging utilities - Pure helpers with no business logic
"""
from datetime import datetime
import socket


def get_user_ip(request=None):
    """
    Get user IP address from request
    
    Args:
        request: Flask request object (injected)
    
    Returns:
        str: IP address or '127.0.0.1' if not available
    """
    if request is None:
        return '127.0.0.1'
    
    # Check for proxy headers
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    
    if request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    
    if request.remote_addr:
        return request.remote_addr
    
    return '127.0.0.1'


def get_user_agent(request=None):
    """
    Get user agent string from request
    
    Args:
        request: Flask request object (injected)
    
    Returns:
        str: User agent string or 'Unknown'
    """
    if request is None:
        return 'Unknown'
    
    return request.headers.get('User-Agent', 'Unknown')[:200]


def log_activity(user_id, user_type, user_email, action, details=None, 
                 activity_log_model=None, db_session=None, request=None):
    """
    Log user activity - Pure function that requires dependencies injected
    
    Args:
        user_id: User ID
        user_type: Type of user (admin, operator, commuter)
        user_email: User email
        action: Action performed
        details: Additional details
        activity_log_model: SQLAlchemy ActivityLog model (injected)
        db_session: Database session (injected)
        request: Flask request object (injected)
    
    Returns:
        bool: True if logged successfully, False otherwise
    """
    try:
        if activity_log_model is None or db_session is None:
            # Skip logging if dependencies not provided
            return False
        
        ip_address = get_user_ip(request)
        
        log = activity_log_model(
            user_id=user_id,
            user_type=user_type,
            user_email=user_email,
            action=action,
            details=details,
            ip_address=ip_address
        )
        
        db_session.add(log)
        db_session.commit()
        return True
        
    except Exception as e:
        print(f"Error logging activity: {e}")
        if db_session:
            db_session.rollback()
        return False


def format_activity_details(action, station=None, old_value=None, new_value=None, 
                            reason=None, extra=None):
    """
    Format activity details string
    
    Args:
        action: Action name
        station: Station name (optional)
        old_value: Old value (optional)
        new_value: New value (optional)
        reason: Reason (optional)
        extra: Extra details (optional)
    
    Returns:
        str: Formatted details string
    """
    parts = [action]
    
    if station:
        parts.append(f"Station: {station}")
    
    if old_value is not None:
        parts.append(f"From: {old_value}")
    
    if new_value is not None:
        parts.append(f"To: {new_value}")
    
    if reason:
        parts.append(f"Reason: {reason}")
    
    if extra:
        parts.append(str(extra))
    
    return " | ".join(parts)


def get_action_category(action):
    """
    Categorize an action for reporting
    
    Args:
        action: Action name
    
    Returns:
        str: Category name
    """
    categories = {
        'auth': ['login', 'logout', 'signup', 'login_attempt', 'login_failed'],
        'reports': ['submit_report', 'flag_report', 'view_reports', 'review_flagged'],
        'admin': ['create_operator', 'deactivate_operator', 'reactivate_operator', 
                  'view_operators', 'view_audit_log'],
        'operator': ['override_congestion', 'clear_override', 'send_broadcast', 
                     'deactivate_broadcast'],
        'user': ['update_favorite', 'save_route', 'delete_route', 'view_page']
    }
    
    for category, actions in categories.items():
        if action in actions:
            return category
    
    return 'other'