# utils/rate_limit.py
import time
import threading
from collections import defaultdict

# Thread-safe in-memory trackers. NOTE: with multiple workers (gunicorn -w 4),
# each worker has its own copy. For true distributed limiting, back this with Redis.
_lock = threading.Lock()

# IP → [(timestamp, date_str), ...]
_report_tracker = defaultdict(list)

# IP → [timestamps] for flags
_flag_ip_tracker = defaultdict(list)

# (user_id, report_id) → [timestamps] for flags (prevents double-flagging the same report)
_flag_user_report_tracker = defaultdict(list)

# user_id → [timestamps] for reports (supplements IP check)
_report_user_tracker = defaultdict(list)


def _cleanup(lst, window):
    now = time.time()
    return [t for t in lst if now - t < window]


def is_report_rate_limited(user_id, ip_address, limit=3, window=86400):
    """
    Returns True if the reporter exceeded their daily limit.
    Checks BOTH user_id and ip_address — so rotating IPs won't help a logged-in user,
    and logging out won't help an IP-hopper.
    """
    with _lock:
        if user_id:
            _report_user_tracker[user_id] = _cleanup(_report_user_tracker[user_id], window)
            if len(_report_user_tracker[user_id]) >= limit:
                return True
        # IP check always runs
        _report_tracker[ip_address] = _cleanup(_report_tracker[ip_address], window)
        if len(_report_tracker[ip_address]) >= limit:
            return True
        return False


def track_report(user_id, ip_address):
    with _lock:
        now = time.time()
        _report_tracker[ip_address].append(now)
        if user_id:
            _report_user_tracker[user_id].append(now)


def is_flag_rate_limited(ip_address, user_id=None, limit=10, window=3600):
    """
    Two-tier flag limit:
    - per IP:     10/hour
    - per user:   20/hour (if logged in)
    """
    with _lock:
        _flag_ip_tracker[ip_address] = _cleanup(_flag_ip_tracker[ip_address], window)
        if len(_flag_ip_tracker[ip_address]) >= limit:
            return True
        if user_id:
            key = f"u:{user_id}"
            _flag_ip_tracker[key] = _cleanup(_flag_ip_tracker[key], window)
            if len(_flag_ip_tracker[key]) >= limit * 2:
                return True
        return False


def track_flag(ip_address, user_id=None):
    with _lock:
        now = time.time()
        _flag_ip_tracker[ip_address].append(now)
        if user_id:
            _flag_ip_tracker[f"u:{user_id}"].append(now)


def user_already_flagged(user_id, report_id, window=86400):
    """True if this user flagged this specific report in the last `window` seconds."""
    if not user_id:
        return False
    key = (user_id, report_id)
    with _lock:
        _flag_user_report_tracker[key] = _cleanup(_flag_user_report_tracker[key], window)
        return len(_flag_user_report_tracker[key]) > 0


def mark_user_flag(user_id, report_id):
    if not user_id:
        return
    with _lock:
        _flag_user_report_tracker[(user_id, report_id)].append(time.time())


def get_remaining_reports(user_id, ip_address, limit=3, window=86400):
    with _lock:
        if user_id:
            _report_user_tracker[user_id] = _cleanup(_report_user_tracker[user_id], window)
            return max(0, limit - len(_report_user_tracker[user_id]))
        _report_tracker[ip_address] = _cleanup(_report_tracker[ip_address], window)
        return max(0, limit - len(_report_tracker[ip_address]))