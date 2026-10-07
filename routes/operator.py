from flask import Blueprint, session, request, jsonify, flash, redirect, url_for, render_template, current_app
from models import User, Report, Broadcast, ActivityLog, db
from datetime import datetime, timedelta
import json, time, math, os, threading
from .auth import login_required, log_activity
from flask_caching import Cache
from extensions import cache
from config import Config

operator_bp = Blueprint('operator', __name__)

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan",
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia",
            "Ayala Ave", "Magallanes", "Taft"]

VALID_DIRECTIONS = ('northbound', 'southbound', 'both')

# DOTr Official Platform Capacities (for congestion calculation)
MRT3_PLATFORM_CAPACITY = {
    "North Ave": 1142, "Quezon Ave": 1195, "Kamuning": 1364, "Cubao": 1747,
    "Santolan": 1306, "Ortigas": 1331, "Shaw Blvd": 1619, "Boni Ave": 1417,
    "Guadalupe": 1301, "Buendia": 1645, "Ayala Ave": 1222, "Magallanes": 1202,
    "Taft": 720
}

STATION_BASE_CAPACITY = {
    "North Ave": 12000, "Quezon Ave": 9000, "Kamuning": 7500, "Cubao": 15000,
    "Santolan": 8000, "Ortigas": 9500, "Shaw Blvd": 11000, "Boni Ave": 8500,
    "Guadalupe": 10000, "Buendia": 9000, "Ayala Ave": 14000, "Magallanes": 9000, "Taft": 16000
}


# ======================================================================
# BLUEPRINT-LEVEL AUTH GUARD  (FIX #1)
# Every /operator/ and /api/operator/ route now requires a valid session.
# check_session_validity in auth.py is a second line of defense.
# ======================================================================
PUBLIC_OPERATOR_ENDPOINTS = {
    'operator.get_public_broadcasts',
}


@operator_bp.before_request
def _require_operator_auth():
    """Block all operator routes unless logged in OR endpoint is allowlisted."""
    if request.endpoint in PUBLIC_OPERATOR_ENDPOINTS:
        return None

    # Env-admin can browse everything
    if session.get('is_admin') and session.get('role') == 'admin':
        return None

    is_api = request.path.startswith('/api/')

    if not session.get('user_id'):
        if is_api:
            return jsonify({'error': 'unauthorized'}), 401
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('auth.login'))

    user = User.query.get(session['user_id'])
    if not user or not user.is_active:
        session.clear()
        if is_api:
            return jsonify({'error': 'session expired'}), 401
        flash('Your session has expired.', 'warning')
        return redirect(url_for('auth.login'))

    if user.role not in ('operator', 'admin'):
        if is_api:
            return jsonify({'error': 'forbidden'}), 403
        flash('Access denied.', 'error')
        return redirect(url_for('auth.login'))

    return None


# ======================================================================
# HELPERS
# ======================================================================

def _require_station_access(user, station):
    """Raise PermissionError if user cannot act on `station`."""
    if user is None:
        raise PermissionError('User not found')
    if user.role == 'admin' or user.access_level == 'line_wide':
        return True
    managed = get_operator_stations(user.id)
    if station not in managed:
        raise PermissionError(f'You do not manage {station}')
    return True


def _validate_override_payload(data):
    """FIX #7 — validate station, direction, congestion_value."""
    station = data.get('station')
    direction = (data.get('direction') or 'southbound').lower()
    level = data.get('level')
    duration = data.get('duration')

    if station not in STATIONS:
        return None, 'Invalid station'
    if direction not in VALID_DIRECTIONS:
        return None, 'Invalid direction'

    try:
        congestion_value = float(data.get('congestion_value'))
    except (TypeError, ValueError):
        return None, 'congestion_value must be a number'

    if not (0 <= congestion_value <= 100):
        return None, 'congestion_value must be between 0 and 100'

    # Validate duration
    duration_minutes = 0
    if duration != 'manual':
        try:
            duration_minutes = int(duration)
            if duration_minutes <= 0 or duration_minutes > 24 * 60:
                return None, 'Invalid duration'
        except (TypeError, ValueError):
            return None, 'Invalid duration'

    return {
        'station': station,
        'direction': direction,
        'level': level,
        'congestion_value': congestion_value,
        'duration': duration,
        'duration_minutes': duration_minutes,
        'reason': (data.get('reason') or '')[:500],
    }, None


# ======================================================================
# PERSISTENT OVERRIDE STORAGE  (FIX #8 — file lock)
# ======================================================================
OVERRIDES_FILE = 'overrides.json'
_OVERRIDES_LOCK = threading.Lock()


def load_overrides():
    """Load overrides from file (thread-safe)."""
    with _OVERRIDES_LOCK:
        if os.path.exists(OVERRIDES_FILE):
            try:
                with open(OVERRIDES_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading overrides: {e}")
                return {}
        return {}


def save_overrides(overrides):
    """Save overrides to file (thread-safe)."""
    with _OVERRIDES_LOCK:
        try:
            with open(OVERRIDES_FILE, 'w') as f:
                json.dump(overrides, f, indent=2)
            print(f"✅ Saved {len(overrides)} overrides to file")
        except Exception as e:
            print(f"Error saving overrides: {e}")


def get_active_overrides():
    """Get active overrides from file with expiry check."""
    overrides = load_overrides()

    now_timestamp = Config.get_current_time().timestamp()

    active_overrides = {}
    for key, override in overrides.items():
        expiry = override.get('expiry')
        if expiry is None or expiry > now_timestamp:
            active_overrides[key] = override
        else:
            print(f"⏰ Override expired: {key}")

    return active_overrides


def _get_congestion_from_prediction(pred_scaled, target_scaler, station_name, direction='southbound'):
    """Convert model prediction to congestion percentage using P90."""
    raw_value = float(pred_scaled[0][0]) if hasattr(pred_scaled, '__getitem__') else float(pred_scaled)

    if target_scaler is not None:
        try:
            passenger_count = float(target_scaler.inverse_transform([[raw_value]])[0][0])
        except Exception as e:
            print(f"   ⚠️ Inverse transform failed: {e}")
            capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
            passenger_count = raw_value * capacity * 1.5
    else:
        capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
        passenger_count = raw_value * capacity * 1.5

    passenger_count = max(0, passenger_count)

    try:
        from routes.api_predict import get_p90_percentile
        p90 = get_p90_percentile(station_name, direction)
    except Exception as e:
        print(f"⚠️ Could not get P90 for {station_name} {direction}: {e}")
        capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
        p90 = capacity * 0.8

    congestion = (passenger_count / p90) * 100
    congestion = max(0, min(congestion, 100))

    return congestion, passenger_count


# ======================================================================
# REPORTS
# ======================================================================

@operator_bp.route('/api/reports', methods=['GET'])
def get_reports():
    """Get reports for operator dashboard - Shows ALL active reports."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401

        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        reports = Report.query.filter(
            Report.archived == False
        ).order_by(Report.timestamp.desc()).all()

        result = []
        for report in reports:
            photo_paths = []
            if report.photo_path:
                try:
                    if report.photo_path.startswith('['):
                        photo_paths = json.loads(report.photo_path)
                    else:
                        photo_paths = [report.photo_path]
                except Exception:
                    photo_paths = []

            username = None
            if report.user:
                username = report.user.username

            congestion = report.reported_congestion
            if congestion >= 80:
                status_text, status_class = "Severe", "status-severe"
            elif congestion >= 60:
                status_text, status_class = "Congested", "status-congested"
            elif congestion >= 30:
                status_text, status_class = "Moderate", "status-moderate"
            else:
                status_text, status_class = "Light", "status-light"

            result.append({
                'id': report.id,
                'station': report.station,
                'direction': getattr(report, 'direction', 'both'),
                'reported_congestion': report.reported_congestion,
                'remarks': report.remarks,
                'timestamp': report.timestamp.isoformat(),
                'username': username or 'Anonymous',
                'anonymous': report.anonymous,
                'flagged': getattr(report, 'flagged', False),
                'flag_count': getattr(report, 'flag_count', 0),
                'archived': report.archived,
                'photo_paths': photo_paths,
                'photo_path': report.photo_path,
                'reviewed': getattr(report, 'reviewed', False),
                'status_text': status_text,
                'status_class': status_class,
            })

        return jsonify(result)
    except Exception as e:
        current_app.logger.exception("get_reports failed")
        return jsonify([]), 500


# ======================================================================
# PUBLIC BROADCASTS  (allowlisted — no auth)
# ======================================================================

@operator_bp.route('/api/broadcasts/public', methods=['GET'])
def get_public_broadcasts():
    """Get active broadcasts for public alerts page."""
    try:
        now = datetime.now()

        expired_by_date = Broadcast.query.filter(
            Broadcast.is_active == True,
            Broadcast.expires_at != None,
            Broadcast.expires_at <= now
        ).all()

        for broadcast in expired_by_date:
            broadcast.is_active = False

        if expired_by_date:
            db.session.commit()

        active_broadcasts = Broadcast.query.filter(
            Broadcast.is_active == True
        ).order_by(Broadcast.created_at.desc()).all()

        result = []
        for broadcast in active_broadcasts:
            stations = json.loads(broadcast.stations) if broadcast.stations else []

            created_at = broadcast.created_at
            diff_seconds = (now - created_at).total_seconds()

            if diff_seconds < 60:
                time_display = 'Just now'
            elif diff_seconds < 3600:
                time_display = f'{int(diff_seconds // 60)} min ago'
            elif diff_seconds < 86400:
                time_display = f'{int(diff_seconds // 3600)}h ago'
            else:
                time_display = f'{int(diff_seconds // 86400)}d ago'

            result.append({
                'id': broadcast.id,
                'title': broadcast.title,
                'message': broadcast.message,
                'type': broadcast.disruption_type,
                'severity': broadcast.severity,
                'direction': getattr(broadcast, 'direction', 'both'),
                'stations': stations,
                'created_at': broadcast.created_at.isoformat(),
                'expires_at': broadcast.expires_at.isoformat() if broadcast.expires_at else None,
                'is_active': broadcast.is_active,
                'time': time_display,
            })

        return jsonify({'success': True, 'broadcasts': result})
    except Exception as e:
        current_app.logger.exception("get_public_broadcasts failed")
        return jsonify({'success': False, 'error': str(e)}), 500


# ======================================================================
# STATION STATUS / FORECAST
# ======================================================================

def get_live_map_data_direct():
    """Get live map data directly without HTTP calls."""
    try:
        try:
            from .live_map import get_directional_data
            return get_directional_data()
        except ImportError:
            pass

        cache_ext = current_app.extensions.get('cache')
        if cache_ext:
            cached_data = cache_ext.get('live_map_data')
            if cached_data:
                return cached_data

        data = {'northbound': {}, 'southbound': {}}
        for station in STATIONS:
            base = 20 + (hash(station) % 50)
            data['northbound'][station] = {
                'congestion': base,
                'status': _get_status_from_congestion(base),
                'wait_time': _get_wait_time(base),
                'ridership': base * 10,
            }
            base2 = 25 + (hash(station + 'south') % 50)
            data['southbound'][station] = {
                'congestion': base2,
                'status': _get_status_from_congestion(base2),
                'wait_time': _get_wait_time(base2),
                'ridership': base2 * 10,
            }
        return data

    except Exception as e:
        current_app.logger.exception("get_live_map_data_direct failed")
        return None


@operator_bp.route('/api/operator/station-status')
def operator_station_status():
    """Get station status for operator dashboard — USING V2 PREDICTION API."""
    try:
        # FIX: call the function directly instead of going through test_client.
        # That was spawning a real HTTP request against the local server,
        # which caused deadlocks under gunicorn with a single worker.
        try:
            from routes.api_other import live_map_directions_v2
            # Call the underlying function — it's registered as a route but
            # we can also call it directly if it returns a Flask Response.
            # Simplest safe approach: replicate the shape here.
            from routes.api_predict import get_directional_prediction
            northbound_data = {}
            southbound_data = {}
            now = datetime.now()
            for st in STATIONS:
                north_cong = get_directional_prediction(st, 'Northbound', now) or 0
                south_cong = get_directional_prediction(st, 'Southbound', now) or 0
                northbound_data[st] = {
                    'congestion': north_cong,
                    'status': _get_status_from_congestion(north_cong),
                    'wait_time': _get_wait_time(north_cong),
                    'ridership': int(north_cong * 10),
                }
                southbound_data[st] = {
                    'congestion': south_cong,
                    'status': _get_status_from_congestion(south_cong),
                    'wait_time': _get_wait_time(south_cong),
                    'ridership': int(south_cong * 10),
                }
        except Exception as inner:
            current_app.logger.warning(f"Prediction unavailable: {inner}")
            return jsonify({'stations': _generate_fallback_stations(), 'fallback': True})

        active_overrides = get_active_overrides()
        overrides_lower = {k.lower() for k in active_overrides.keys()}

        result = []
        for station in STATIONS:
            north = northbound_data.get(station, {})
            south = southbound_data.get(station, {})

            north_key = f"{station}_northbound".lower()
            south_key = f"{station}_southbound".lower()

            result.append({
                'name': station,
                'northbound': {
                    'congestion': north.get('congestion', 0),
                    'status': north.get('status', _get_status_from_congestion(north.get('congestion', 0))),
                    'wait_time': north.get('wait_time', _get_wait_time(north.get('congestion', 0))),
                    'ridership': north.get('ridership', 0),
                    'overridden': north_key in overrides_lower,
                },
                'southbound': {
                    'congestion': south.get('congestion', 0),
                    'status': south.get('status', _get_status_from_congestion(south.get('congestion', 0))),
                    'wait_time': south.get('wait_time', _get_wait_time(south.get('congestion', 0))),
                    'ridership': south.get('ridership', 0),
                    'overridden': south_key in overrides_lower,
                },
            })

        return jsonify({'stations': result})

    except Exception as e:
        current_app.logger.exception("operator_station_status failed")
        return jsonify({'stations': _generate_fallback_stations(), 'error': str(e)}), 500


@operator_bp.route('/api/operator/forecast/<station_name>')
def operator_forecast(station_name):
    """Get 6-hour forecast for a station."""
    try:
        station = station_name.replace('%20', ' ')
        if station not in STATIONS:
            return jsonify({'error': 'Invalid station'}), 400

        now = datetime.now()
        base_time = now.replace(minute=0, second=0, microsecond=0)

        forecasts = []
        from routes.api_predict import get_directional_prediction

        for i in range(7):
            target_time = base_time + timedelta(hours=i)

            try:
                north_cong = get_directional_prediction(station, 'Northbound', target_time)
                south_cong = get_directional_prediction(station, 'Southbound', target_time)
            except Exception as e:
                current_app.logger.warning(f"forecast prediction failed: {e}")
                north_cong = None
                south_cong = None

            # FIX #18-style: propagate None instead of faking 0
            north_cong = north_cong if north_cong is not None else None
            south_cong = south_cong if south_cong is not None else None

            if north_cong is None or south_cong is None:
                avg_cong = None
                status = 'UNKNOWN'
                color = 'unknown'
            else:
                avg_cong = (north_cong + south_cong) / 2
                if avg_cong >= 80:
                    status, color = "SEVERE", "critical"
                elif avg_cong >= 50:
                    status, color = "CONGESTED", "congested"
                elif avg_cong >= 25:
                    status, color = "MODERATE", "moderate"
                else:
                    status, color = "LIGHT", "light"

            if i == 0:
                time_display = "NOW"
            elif i == 1:
                time_display = "1h"
            else:
                time_display = f"{i}h"

            forecasts.append({
                'hour': target_time.hour,
                'time': time_display,
                'time_full': target_time.strftime('%I:%M %p'),
                'northbound': round(north_cong, 1) if north_cong is not None else None,
                'southbound': round(south_cong, 1) if south_cong is not None else None,
                'average': round(avg_cong, 1) if avg_cong is not None else None,
                'status': status,
                'color': color,
            })

        return jsonify({
            'station': station,
            'timestamp': now.isoformat(),
            'forecasts': forecasts,
        })

    except Exception as e:
        current_app.logger.exception("operator_forecast failed")
        return jsonify({'error': str(e)}), 500


def _generate_fallback_stations():
    fallback = []
    for station in STATIONS:
        fallback.append({
            'name': station,
            'northbound': {'congestion': 25, 'status': 'MODERATE', 'wait_time': '5-10 min', 'ridership': 500, 'overridden': False},
            'southbound': {'congestion': 25, 'status': 'MODERATE', 'wait_time': '5-10 min', 'ridership': 550, 'overridden': False},
        })
    return fallback


def _get_status_from_congestion(congestion):
    # FIX #13: use >= to match admin.py
    if congestion >= 80:
        return 'SEVERE'
    elif congestion >= 50:
        return 'CONGESTED'
    elif congestion >= 25:
        return 'MODERATE'
    else:
        return 'LIGHT'


def _get_wait_time(congestion):
    if congestion >= 80:
        return '15-20 min'
    elif congestion >= 50:
        return '10-15 min'
    elif congestion >= 25:
        return '5-10 min'
    else:
        return '2-5 min'


# ======================================================================
# DEBUG  (FIX #5 — dev only)
# ======================================================================

@operator_bp.route('/api/operator/debug-override')
def debug_override():
    if not current_app.debug:
        return jsonify({'error': 'not available'}), 404

    active_overrides = get_active_overrides()
    return jsonify({
        'active_overrides': active_overrides,
        'current_timestamp': time.time(),
        'overrides_file_exists': os.path.exists(OVERRIDES_FILE),
    })


# ======================================================================
# STATION HELPERS
# ======================================================================

def get_operator_stations(user_id):
    user = User.query.get(user_id)
    if not user:
        return []

    if user.role == 'admin' or user.access_level == 'line_wide':
        return STATIONS
    elif user.access_level == 'zone':
        zones = {
            'north': ['North Ave', 'Quezon Ave', 'Kamuning', 'Cubao', 'Santolan'],
            'central': ['Ortigas', 'Shaw Blvd', 'Boni Ave', 'Guadalupe'],
            'south': ['Buendia', 'Ayala Ave', 'Magallanes', 'Taft'],
        }
        return zones.get(user.assigned_zone, [])
    else:
        if user.assigned_stations:
            try:
                return json.loads(user.assigned_stations)
            except Exception:
                return []
        return [user.favorite_station] if user.favorite_station else ['North Ave']


# ======================================================================
# DASHBOARD
# ======================================================================

@operator_bp.route('/operator-dashboard')
@operator_bp.route('/operator_dashboard')
def operator_dashboard():
    # FIX #9: guard against deleted user row
    user = User.query.get(session['user_id'])
    if not user or not user.is_active:
        session.clear()
        flash('Session expired.', 'warning')
        return redirect(url_for('auth.login'))

    if user.role == 'admin':
        return redirect(url_for('admin.admin_dashboard'))
    elif user.role == 'commuter':
        return redirect(url_for('user.user_dashboard'))
    elif user.role == 'operator':
        managed_stations = get_operator_stations(user.id)
        return render_template('operator_dashboard.html',
                               username=user.username,
                               role=user.role,
                               managed_stations=managed_stations,
                               all_stations=STATIONS,
                               access_level=user.access_level,
                               assigned_zone=user.assigned_zone,
                               now=datetime.now())
    else:
        return redirect(url_for('user.user_dashboard'))


# ======================================================================
# BROADCASTS
# ======================================================================

@operator_bp.route('/api/operator/broadcasts', methods=['GET'])
def get_operator_broadcasts():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401

        now = datetime.now()

        expired_by_date = Broadcast.query.filter(
            Broadcast.is_active == True,
            Broadcast.expires_at != None,
            Broadcast.expires_at <= now
        ).all()

        if expired_by_date:
            for broadcast in expired_by_date:
                broadcast.is_active = False
            db.session.commit()

        managed_stations = get_operator_stations(user_id)

        all_broadcasts = Broadcast.query.order_by(
            Broadcast.created_at.desc()
        ).all()

        typeIcons = {
            "Train Breakdown": "fa-train", "Overcrowding": "fa-users",
            "Maintenance": "fa-wrench", "Signal Issue": "fa-satellite-dish",
            "Gate Closure": "fa-door-closed", "General Notice": "fa-bullhorn",
        }

        result = []
        for broadcast in all_broadcasts:
            stations = json.loads(broadcast.stations) if broadcast.stations else []
            if any(s in managed_stations for s in stations):
                result.append({
                    'id': broadcast.id,
                    'title': broadcast.title,
                    'message': broadcast.message,
                    'disruption_type': broadcast.disruption_type,
                    'stations': stations,
                    'severity': broadcast.severity,
                    'direction': getattr(broadcast, 'direction', 'both'),
                    'created_at': broadcast.created_at.isoformat(),
                    'expires_at': broadcast.expires_at.isoformat() if broadcast.expires_at else None,
                    'duration_minutes': getattr(broadcast, 'duration_minutes', 60),
                    'is_active': broadcast.is_active,
                    'icon': typeIcons.get(broadcast.disruption_type, 'fa-bullhorn'),
                })

        return jsonify({'success': True, 'broadcasts': result})
    except Exception as e:
        current_app.logger.exception("get_operator_broadcasts failed")
        return jsonify({'success': False, 'error': str(e)}), 500


@operator_bp.route('/api/operator/broadcast/<int:broadcast_id>', methods=['GET'])
def get_broadcast(broadcast_id):
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401

        broadcast = Broadcast.query.get(broadcast_id)
        if not broadcast:
            return jsonify({'success': False, 'error': 'Broadcast not found'}), 404

        user = User.query.get(user_id)
        managed_stations = get_operator_stations(user_id)
        stations = json.loads(broadcast.stations) if broadcast.stations else []

        if not any(s in managed_stations for s in stations):
            if user.access_level != 'line_wide' and user.role != 'admin':
                return jsonify({'success': False, 'error': 'Permission denied'}), 403

        return jsonify({
            'success': True,
            'broadcast': {
                'id': broadcast.id,
                'title': broadcast.title,
                'message': broadcast.message,
                'disruption_type': broadcast.disruption_type,
                'stations': stations,
                'severity': broadcast.severity,
                'direction': getattr(broadcast, 'direction', 'both'),
                'duration_minutes': getattr(broadcast, 'duration_minutes', 60),
                'is_active': broadcast.is_active,
                'created_at': broadcast.created_at.isoformat(),
                'expires_at': broadcast.expires_at.isoformat() if broadcast.expires_at else None,
            },
        })
    except Exception as e:
        current_app.logger.exception("get_broadcast failed")
        return jsonify({'success': False, 'error': str(e)}), 500


@operator_bp.route('/api/operator/send-broadcast', methods=['POST'])
def send_broadcast():
    try:
        data = request.json or {}
        title = data.get('title')
        message = data.get('message')
        disruption_type = data.get('disruption_type')
        stations = data.get('stations') or []
        severity = data.get('severity')
        direction = (data.get('direction') or 'both').lower()
        duration_minutes = data.get('duration_minutes', 60)

        operator_id = session.get('user_id')
        if not operator_id:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401

        # Basic validation
        if not isinstance(stations, list) or not stations:
            return jsonify({'success': False, 'error': 'No stations selected'}), 400
        if direction not in VALID_DIRECTIONS:
            return jsonify({'success': False, 'error': 'Invalid direction'}), 400
        for s in stations:
            if s not in STATIONS:
                return jsonify({'success': False, 'error': f'Invalid station: {s}'}), 400

        # ✅ FIX #2: verify operator can broadcast to every station listed
        user = User.query.get(operator_id)
        if user.role != 'admin' and user.access_level != 'line_wide':
            managed = get_operator_stations(operator_id)
            unauthorized = [s for s in stations if s not in managed]
            if unauthorized:
                return jsonify({
                    'success': False,
                    'error': f'You do not manage: {", ".join(unauthorized)}'
                }), 403

        expires_at = None
        if duration_minutes and duration_minutes > 0:
            expires_at = datetime.now() + timedelta(minutes=int(duration_minutes))

        broadcast = Broadcast(
            title=title,
            message=message,
            disruption_type=disruption_type,
            stations=json.dumps(stations),
            severity=severity,
            operator_id=operator_id,
            created_at=datetime.now(),
            is_active=True,
            direction=direction,
            duration_minutes=duration_minutes,
            expires_at=expires_at,
        )

        db.session.add(broadcast)
        db.session.commit()

        log_activity(operator_id, 'operator', session.get('username'), 'send_broadcast',
                     f'Broadcast: "{title}" to {len(stations)} stations (Direction: {direction}, Duration: {duration_minutes} min)')

        return jsonify({'success': True, 'message': 'Broadcast sent', 'broadcast_id': broadcast.id})
    except Exception as e:
        db.session.rollback()
        current_app.logger.exception("send_broadcast failed")
        return jsonify({'success': False, 'error': str(e)}), 500


@operator_bp.route('/api/operator/broadcast/<int:broadcast_id>', methods=['PUT'])
def update_broadcast(broadcast_id):
    try:
        data = request.json or {}
        broadcast = Broadcast.query.get(broadcast_id)
        if not broadcast:
            return jsonify({'success': False, 'error': 'Broadcast not found'}), 404

        # ✅ FIX #4: verify ownership before mutating
        user_id = session.get('user_id')
        user = User.query.get(user_id)
        broadcast_stations = json.loads(broadcast.stations) if broadcast.stations else []
        if user.role != 'admin' and user.access_level != 'line_wide':
            managed = get_operator_stations(user_id)
            if not any(s in managed for s in broadcast_stations):
                return jsonify({
                    'success': False,
                    'error': 'You cannot modify this broadcast'
                }), 403

        if 'is_active' in data:
            broadcast.is_active = data['is_active']
            if data['is_active'] is False:
                broadcast.archived_at = datetime.now()
                broadcast.archived_by = session.get('username', 'operator')
            else:
                broadcast.archived_at = None
                broadcast.archived_by = None
        else:
            if 'title' in data:
                broadcast.title = data['title']
            if 'message' in data:
                broadcast.message = data['message']
            if 'severity' in data:
                broadcast.severity = data['severity']

        db.session.commit()

        log_activity(user_id, 'operator', session.get('username'),
                     'edit_broadcast', f'Updated broadcast #{broadcast_id}: {broadcast.title}')

        return jsonify({'success': True, 'message': 'Broadcast updated'})
    except Exception as e:
        db.session.rollback()
        current_app.logger.exception("update_broadcast failed")
        return jsonify({'success': False, 'error': str(e)}), 500


@operator_bp.route('/api/operator/broadcast/<int:broadcast_id>', methods=['DELETE'])
def delete_broadcast(broadcast_id):
    try:
        broadcast = Broadcast.query.get(broadcast_id)
        if not broadcast:
            return jsonify({'success': False, 'error': 'Broadcast not found'}), 404

        # ✅ FIX #4: verify ownership before archiving
        user_id = session.get('user_id')
        user = User.query.get(user_id)
        broadcast_stations = json.loads(broadcast.stations) if broadcast.stations else []
        if user.role != 'admin' and user.access_level != 'line_wide':
            managed = get_operator_stations(user_id)
            if not any(s in managed for s in broadcast_stations):
                return jsonify({
                    'success': False,
                    'error': 'You cannot archive this broadcast'
                }), 403

        broadcast.is_active = False
        broadcast.archived_at = datetime.now()
        broadcast.archived_by = session.get('username', 'operator')

        db.session.commit()

        log_activity(user_id, 'operator', session.get('username'),
                     'archive_broadcast', f'Archived broadcast #{broadcast_id}: {broadcast.title}')

        return jsonify({'success': True, 'message': 'Broadcast archived'})
    except Exception as e:
        db.session.rollback()
        current_app.logger.exception("delete_broadcast failed")
        return jsonify({'success': False, 'error': str(e)}), 500


# ======================================================================
# OVERRIDES
# ======================================================================

@operator_bp.route('/api/operator/override-congestion', methods=['POST'])
def override_congestion():
    try:
        data = request.json or {}
        # ✅ FIX #7: validate payload
        parsed, err = _validate_override_payload(data)
        if err:
            return jsonify({'success': False, 'error': err}), 400

        station = parsed['station']
        direction = parsed['direction']
        level = parsed['level']
        congestion_value = parsed['congestion_value']
        duration = parsed['duration']
        duration_minutes = parsed['duration_minutes']
        reason = parsed['reason']

        operator_id = session.get('user_id')
        operator_email = session.get('username')

        # ✅ FIX #2: verify operator manages this station
        user = User.query.get(operator_id)
        try:
            _require_station_access(user, station)
        except PermissionError as pe:
            return jsonify({'success': False, 'error': str(pe)}), 403

        override_key = f"{station}_{direction}"

        overrides = load_overrides()

        current_time = datetime.now()
        current_timestamp = current_time.timestamp()

        expiry = None
        if duration != 'manual':
            expiry = current_timestamp + (duration_minutes * 60)
        else:
            current_time = current_time.replace(minute=0, second=0, microsecond=0)
            current_timestamp = current_time.timestamp()

        overrides[override_key] = {
            'station': station,
            'direction': direction,
            'level': level,
            'congestion': congestion_value,
            'operator': operator_email,
            'reason': reason,
            'expiry': expiry,
            'timestamp': current_time.isoformat(),
            'duration_minutes': duration_minutes,
            'created_at': current_time.isoformat(),
        }

        save_overrides(overrides)

        if 'overrides' not in current_app.config:
            current_app.config['overrides'] = {}
        current_app.config['overrides'][override_key] = overrides[override_key]

        _clear_override_cache(station, current_time)

        log_activity(operator_id, 'operator', operator_email, 'override_congestion',
                     f'Overrode {station} ({direction}) to {level} ({congestion_value}%)')

        clear_v3_cache()

        return jsonify({'success': True, 'message': f'{station} ({direction}) set to {level}'})
    except Exception as e:
        current_app.logger.exception("override_congestion failed")
        return jsonify({'success': False, 'error': str(e)}), 500


@operator_bp.route('/api/operator/clear-override', methods=['POST'])
def clear_override():
    try:
        data = request.json or {}
        station = data.get('station')
        direction = (data.get('direction') or '').lower()

        if station not in STATIONS or direction not in VALID_DIRECTIONS:
            return jsonify({'success': False, 'error': 'Invalid station or direction'}), 400

        # ✅ FIX #3: verify operator manages this station
        user_id = session.get('user_id')
        user = User.query.get(user_id)
        try:
            _require_station_access(user, station)
        except PermissionError as pe:
            return jsonify({'success': False, 'error': str(pe)}), 403

        target_key = f"{station}_{direction}"

        overrides = load_overrides()
        if target_key in overrides:
            del overrides[target_key]
            save_overrides(overrides)

        if 'overrides' in current_app.config:
            current_app.config['overrides'].pop(target_key, None)
            current_app.config['overrides'].pop(target_key.lower(), None)

        _clear_override_cache(station, datetime.now())

        active = get_active_overrides()

        log_activity(user_id, 'operator', session.get('username'),
                     'clear_override', f'Cleared override for {station} ({direction})')

        clear_v3_cache()

        return jsonify({
            'success': True,
            'message': f'Override cleared successfully for {station} ({direction})',
            'active_overrides': active,
        })
    except Exception as e:
        current_app.logger.exception("clear_override failed")
        return jsonify({'success': False, 'error': str(e)}), 500


def _clear_override_cache(station, current_time):
    """Clear cache keys touched by an override change."""
    try:
        cache_instance = current_app.extensions.get('cache')
        if not cache_instance:
            return
        keys = [
            'live_map_v2', 'view//live_map_v2', 'view/live_map_v2',
            'cache//live_map_v2', 'operator_station_status',
            'view//operator_station_status', 'view/operator_station_status',
        ]
        for hour in range(24):
            keys.extend([
                f"forecast_{station}_{hour}",
                f"forecast_{station.lower()}_{hour}",
                f"all_stations_{hour}",
            ])
        for key in keys:
            try:
                cache_instance.delete(key)
            except Exception:
                pass
        try:
            cache_instance.clear()
        except Exception:
            pass
    except Exception as e:
        current_app.logger.warning(f"cache clear failed: {e}")


def clear_v3_cache():
    """Clear all live_map_v3 cache keys (for all hours)."""
    cache_instance = current_app.extensions.get('cache')
    if not cache_instance:
        return
    for h in range(24):
        key = f"live_map_v3_{datetime.now().replace(hour=h, minute=0).strftime('%Y%m%d%H')}"
        cache_instance.delete(key)
    cache_instance.delete(f"live_map_v3_{datetime.now().strftime('%Y%m%d%H')}")


# ======================================================================
# REPORT REVIEW
# ======================================================================

@operator_bp.route('/api/operator/review-report/<int:report_id>', methods=['POST'])
def review_report(report_id):
    try:
        data = request.json or {}
        verdict = data.get('verdict')

        user_id = session.get('user_id')
        user = User.query.get(user_id)
        report = Report.query.get(report_id)

        if not report:
            return jsonify({'success': False, 'error': 'Report not found'}), 404

        managed_stations = get_operator_stations(user_id)
        if report.station not in managed_stations and user.role != 'admin':
            return jsonify({'success': False, 'error': 'You cannot review reports from this station'}), 403

        if verdict == 'true_positive':
            report.flagged = False
            report.flag_count = 0
            report.reviewed = True
            report.reviewed_at = datetime.now()
            report.reviewed_by = user.username
            message = 'Report kept as True Positive, flags cleared'
        elif verdict == 'false_positive':
            report.archived = True
            report.archived_at = datetime.now()
            report.archived_by = user.username
            report.flagged = False
            report.flag_count = 0
            report.reviewed = True
            message = 'Report archived as False Positive'
        else:
            return jsonify({'success': False, 'error': 'Invalid verdict'}), 400

        db.session.commit()

        log_activity(user_id, user.role, user.username, 'review_report',
                     f'{verdict} for report #{report_id} from {report.station}')

        return jsonify({'success': True, 'message': message})
    except Exception as e:
        db.session.rollback()
        current_app.logger.exception("review_report failed")
        return jsonify({'success': False, 'error': str(e)}), 500


@operator_bp.route('/api/operator/keep-report/<int:report_id>', methods=['POST'])
def keep_report(report_id):
    try:
        user_id = session.get('user_id')
        user = User.query.get(user_id)
        report = Report.query.get(report_id)

        if not report:
            return jsonify({'success': False, 'error': 'Report not found'}), 404

        managed_stations = get_operator_stations(user_id)
        if report.station not in managed_stations and user.role != 'admin':
            return jsonify({'success': False, 'error': 'You cannot review reports from this station'}), 403

        report.flagged = False
        report.flag_count = 0
        report.reviewed = True
        report.reviewed_at = datetime.now()
        report.reviewed_by = user.username

        db.session.commit()

        log_activity(user_id, user.role, user.username, 'keep_report',
                     f'Kept report #{report_id} from {report.station}, flags removed')

        return jsonify({'success': True, 'message': 'Report kept, flags removed'})
    except Exception as e:
        db.session.rollback()
        current_app.logger.exception("keep_report failed")
        return jsonify({'success': False, 'error': str(e)}), 500


@operator_bp.route('/api/operator/archive-report/<int:report_id>', methods=['POST'])
def archive_report(report_id):
    try:
        user_id = session.get('user_id')
        user = User.query.get(user_id)
        report = Report.query.get(report_id)

        if not report:
            return jsonify({'success': False, 'error': 'Report not found'}), 404
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        managed_stations = get_operator_stations(user_id)
        if report.station not in managed_stations and user.role != 'admin':
            return jsonify({'success': False, 'error': 'You cannot archive reports from this station'}), 403

        report.archived = True
        report.archived_at = datetime.now()
        report.archived_by = user.username
        report.flagged = False
        report.flag_count = 0
        report.reviewed = True

        db.session.commit()

        log_activity(user_id, user.role, user.username, 'archive_report',
                     f'Archived report #{report_id} from {report.station}')

        return jsonify({'success': True, 'message': 'Report archived'})
    except Exception as e:
        db.session.rollback()
        current_app.logger.exception("archive_report failed")
        return jsonify({'success': False, 'error': str(e)}), 500


@operator_bp.route('/api/operator/review-flagged/<int:report_id>', methods=['POST'])
def review_flagged_report(report_id):
    try:
        data = request.json or {}
        action = data.get('action')
        reason = (data.get('reason') or '')[:500]

        user_id = session.get('user_id')
        user = User.query.get(user_id)
        report = Report.query.get(report_id)

        if not report:
            return jsonify({'error': 'Report not found'}), 404

        managed_stations = get_operator_stations(user_id)
        if report.station not in managed_stations and user.role != 'admin':
            return jsonify({'error': 'You cannot review reports from this station'}), 403

        if action == 'keep':
            report.flagged = False
            report.reviewed = True
            message = 'Report kept and flag removed'
        elif action == 'delete':
            db.session.delete(report)
            message = 'Report deleted'
        else:
            return jsonify({'error': 'Invalid action'}), 400

        db.session.commit()

        log_activity(user_id, user.role, user.username, 'review_flagged',
                     f'{action} flagged report #{report_id} from {report.station}. Reason: {reason}')

        return jsonify({'success': True, 'message': message})
    except Exception as e:
        db.session.rollback()
        current_app.logger.exception("review_flagged_report failed")
        return jsonify({'error': str(e)}), 500


@operator_bp.route('/api/operator/reports/stats')
def get_operator_report_stats():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401

    managed_stations = get_operator_stations(user_id)

    total_reports = Report.query.filter(Report.station.in_(managed_stations)).count()
    flagged_reports = Report.query.filter(
        Report.flagged == True, Report.station.in_(managed_stations), Report.reviewed == False
    ).count()

    # ✅ FIX #6: compare against a datetime, not a date
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_reports = Report.query.filter(
        Report.station.in_(managed_stations),
        Report.timestamp >= today_start,
    ).count()

    return jsonify({
        'total_reports': total_reports,
        'flagged_reports': flagged_reports,
        'today_reports': today_reports,
    })


# ======================================================================
# PROFILE / ACTIVITY
# ======================================================================

@operator_bp.route('/profile')
@operator_bp.route('/profile.html')
def operator_profile():
    user = User.query.get(session['user_id'])
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('auth.login'))

    managed_stations = get_operator_stations(user.id)

    return render_template('profile.html',
                           user=user,
                           managed_stations=managed_stations,
                           all_stations=STATIONS)


@operator_bp.route('/api/operator/my-activity')
def operator_my_activity():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401

        logs = ActivityLog.query.filter_by(user_id=user_id).order_by(
            ActivityLog.timestamp.desc()
        ).limit(50).all()

        log_data = [{
            'action': log.action,
            'details': log.details,
            'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        } for log in logs]

        return jsonify(log_data)
    except Exception as e:
        current_app.logger.exception("operator_my_activity failed")
        return jsonify([]), 500