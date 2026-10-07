# routes/admin.py
from flask import (
    Blueprint, render_template, session, request, jsonify, flash,
    redirect, url_for, current_app
)
from models import User, Report, Broadcast, ActivityLog, db
from datetime import datetime, timedelta
import secrets, string, json, os
from functools import wraps

from .auth import log_activity, no_cache
from routes.api_predict import get_directional_prediction
from extensions import limiter

admin_bp = Blueprint('admin', __name__)

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan",
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia",
            "Ayala Ave", "Magallanes", "Taft"]

MRT3_PLATFORM_CAPACITY = {
    "North Ave": 1142, "Quezon Ave": 1195, "Kamuning": 1364, "Cubao": 1747,
    "Santolan": 1306, "Ortigas": 1331, "Shaw Blvd": 1619, "Boni Ave": 1417,
    "Guadalupe": 1301, "Buendia": 1645, "Ayala Ave": 1222, "Magallanes": 1202,
    "Taft": 720
}

INVITE_TTL_HOURS = 24

# Only allow invites for these domains (empty set = allow all)
ALLOWED_INVITE_DOMAINS = set()


# Rate-limit key for admin actions — keyed by admin username so a single
# admin session can't spray requests even if multiple IPs are involved.
def _admin_key():
    return session.get('username') or 'anon'


# ======================================================================
# AUTH GUARD
# ======================================================================
ADMIN_PATH_PREFIXES = ('/admin/', '/api/admin/')


def _is_env_admin():
    """Env-admin = session-flagged admin with NO backing DB user row."""
    return bool(
        session.get('is_admin')
        and session.get('role') == 'admin'
        and session.get('user_id') is None
    )


def admin_required(f):
    """Strict admin-only guard."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if _is_env_admin() and request.path.startswith(ADMIN_PATH_PREFIXES):
            return f(*args, **kwargs)

        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'unauthorized'}), 401

        user = User.query.get(user_id)
        if not user or not user.is_active or user.role != 'admin':
            try:
                log_activity(
                    user_id,
                    getattr(user, 'role', 'unknown') if user else 'unknown',
                    session.get('username', 'unknown'),
                    'admin_access_denied',
                    f'Attempted {request.method} {request.path} without admin role'
                )
            except Exception as e:
                print(f"[admin_required] log failed: {e}")
            return jsonify({'error': 'forbidden'}), 403

        return f(*args, **kwargs)
    return decorated_function


def _current_admin_id():
    return session.get('user_id')


def _current_admin_email():
    return session.get('username', 'admin')


def _get_station_prediction(station_name):
    if 'GET_STATION_PREDICTION' in current_app.config:
        return current_app.config['GET_STATION_PREDICTION'](station_name)
    return 50


def _get_status_class(congestion):
    if congestion is None:
        return 'unknown'
    if congestion >= 80:
        return 'severe'
    if congestion >= 60:
        return 'heavy'
    if congestion >= 40:
        return 'moderate'
    return 'light'


def _get_status_text(congestion):
    if congestion is None:
        return 'UNKNOWN'
    return _get_status_class(congestion).upper()


# ======================================================================
# DASHBOARD (HTML)
# ======================================================================
@admin_bp.route('/admin/dashboard')
@no_cache
@admin_required
def admin_dashboard():
    total_users = User.query.count()
    operator_count = User.query.filter_by(role='operator').count()
    commuter_count = User.query.filter_by(role='commuter').count()
    admin_count = User.query.filter_by(role='admin').count()

    users_data = []
    now = datetime.now()
    for user in User.query.order_by(User.created_at.desc()).limit(200).all():
        joined_date = user.created_at.strftime('%b %d, %Y') if user.created_at else 'Unknown'
        last_active = 'Never'
        if user.last_login:
            days_ago = (now - user.last_login).days
            last_active = (
                'Today' if days_ago == 0
                else 'Yesterday' if days_ago == 1
                else f'{days_ago} days ago'
            )

        users_data.append({
            'id': user.id,
            'email': user.username,
            'role': user.role or 'commuter',
            'joined': joined_date,
            'last': last_active,
            'active': user.is_active,
        })

    return render_template(
        'admin_dashboard.html',
        admin_email=session.get('username', 'Admin'),
        total_users=total_users,
        operator_count=operator_count,
        commuter_count=commuter_count,
        admin_count=admin_count,
        users_data=users_data,
    )


# ======================================================================
# RECENT ACTIVITIES
# ======================================================================
@admin_bp.route('/api/admin/recent-activities-list')
@admin_required
@limiter.limit("120 per minute", key_func=_admin_key)
def admin_recent_activities():
    try:
        recent_logs = ActivityLog.query.order_by(
            ActivityLog.timestamp.desc()
        ).limit(4).all()

        activities = []
        for log in recent_logs:
            icon = 'user'
            icon_color = '#3B82F6'
            title = log.action.replace('_', ' ').title()
            description = log.details or f'{log.user_email} performed {log.action}'

            if 'login' in log.action and 'failed' not in log.action:
                icon, icon_color = 'sign-in-alt', '#22C55E'
                title = 'Login'
                description = f'{log.user_email} logged in'
            elif 'logout' in log.action:
                icon, icon_color = 'sign-out-alt', '#EF4444'
                title = 'Logout'
                description = f'{log.user_email} logged out'
            elif 'broadcast' in log.action:
                icon, icon_color = 'bullhorn', '#8B5CF6'
                title = 'Broadcast Sent'
                if log.details and 'Station:' in log.details:
                    description = f"{log.user_email} sent alert to {log.details.split('Station:')[-1].strip()}"
            elif 'override' in log.action:
                icon, icon_color = 'edit', '#F59E0B'
                title = 'Override'
            elif 'create_operator' in log.action or 'reactivate_via_invite' in log.action:
                icon, icon_color = 'user-plus', '#10B981'
                title = 'Operator Created'
            elif 'deactivate_operator' in log.action:
                icon, icon_color = 'user-slash', '#EF4444'
                title = 'Operator Deactivated'
            elif 'reactivate_operator' in log.action:
                icon, icon_color = 'user-check', '#10B981'
                title = 'Operator Reactivated'
            elif 'login_failed' in log.action or 'admin_access_denied' in log.action:
                icon, icon_color = 'exclamation-triangle', '#EF4444'
                title = 'Security Event'

            activities.append({
                'icon': icon,
                'icon_color': icon_color,
                'title': title,
                'description': (description or '')[:100],
                'station': None,
                'time': log.timestamp.strftime('%I:%M %p') if log.timestamp else '—',
            })

        if not activities:
            activities = [{
                'icon': 'user-plus', 'icon_color': '#22C55E',
                'title': 'System Ready',
                'description': 'Admin dashboard initialized',
                'station': None, 'time': 'Just now',
            }]

        return jsonify(activities)
    except Exception as e:
        print(f"[admin_recent_activities] {e}")
        return jsonify([])


# ======================================================================
# PROFILE
# ======================================================================
@admin_bp.route('/api/admin/profile')
@admin_required
def admin_profile():
    return jsonify({
        'username': session.get('username', 'Admin'),
        'role': 'admin',
    })


# ======================================================================
# FLAGGED ACTIONS
# ======================================================================
@admin_bp.route('/api/admin/flagged-actions')
@admin_required
@limiter.limit("60 per minute", key_func=_admin_key)
def flagged_actions():
    try:
        flagged = ActivityLog.query.filter(
            ActivityLog.is_flagged == True
        ).order_by(ActivityLog.timestamp.desc()).all()

        out = []
        for log in flagged:
            out.append({
                'id': log.id,
                'userType': log.user_type or 'system',
                'userName': log.user_email or 'System',
                'action': log.action,
                'target': log.details or '-',
                'details': log.details or '-',
                'flag_reason': getattr(log, 'flag_reason', None) or 'No reason given',
                'ip_address': log.ip_address or '-',
                'timestamp': log.timestamp.isoformat() if log.timestamp else None,
                'flagged_at': (
                    log.flagged_at.isoformat()
                    if getattr(log, 'flagged_at', None) else None
                ),
            })
        return jsonify(out)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[flagged_actions] {e}")
        return jsonify([])


@admin_bp.route('/api/admin/flag-audit-entry/<int:entry_id>', methods=['POST'])
@admin_required
@limiter.limit("60 per minute", key_func=_admin_key)
def flag_audit_entry(entry_id):
    try:
        data = request.get_json(silent=True) or {}
        reason = (data.get('reason') or 'No reason provided')[:500]

        entry = ActivityLog.query.get(entry_id)
        if not entry:
            return jsonify({'success': False, 'error': 'Entry not found'}), 404

        entry.is_flagged = True
        if hasattr(entry, 'flag_reason'):
            entry.flag_reason = reason
        if hasattr(entry, 'flagged_at'):
            entry.flagged_at = datetime.now()

        log_activity(
            _current_admin_id(), 'admin', _current_admin_email(),
            'flag_audit_entry',
            f'Flagged entry {entry_id}. Reason: {reason}'
        )
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


def _clear_flag(entry, admin_notes, action_name):
    entry.is_flagged = False
    if hasattr(entry, 'flag_reason'):
        entry.flag_reason = None
    if hasattr(entry, 'flagged_at'):
        entry.flagged_at = None
    if hasattr(entry, 'admin_review_notes'):
        entry.admin_review_notes = admin_notes
    if hasattr(entry, 'reviewed_by'):
        entry.reviewed_by = _current_admin_email()
    if hasattr(entry, 'reviewed_at'):
        entry.reviewed_at = datetime.now()

    log_activity(
        _current_admin_id(), 'admin', _current_admin_email(),
        action_name, f'Entry {entry.id}. Notes: {admin_notes}'
    )


@admin_bp.route('/api/admin/approve-flagged/<int:entry_id>', methods=['POST'])
@admin_required
@limiter.limit("60 per minute", key_func=_admin_key)
def approve_flagged(entry_id):
    try:
        data = request.get_json(silent=True) or {}
        notes = (data.get('admin_notes') or '')[:500]

        entry = ActivityLog.query.get(entry_id)
        if not entry:
            return jsonify({'success': False, 'error': 'Entry not found'}), 404

        if not getattr(entry, 'is_flagged', False):
            return jsonify({'success': False, 'error': 'Entry is not flagged'}), 400

        _clear_flag(entry, notes, 'approve_flagged')
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@admin_bp.route('/api/admin/dismiss-flagged/<int:entry_id>', methods=['POST'])
@admin_required
@limiter.limit("60 per minute", key_func=_admin_key)
def dismiss_flagged(entry_id):
    try:
        data = request.get_json(silent=True) or {}
        notes = (data.get('admin_notes') or '')[:500]

        entry = ActivityLog.query.get(entry_id)
        if not entry:
            return jsonify({'success': False, 'error': 'Entry not found'}), 404

        if not getattr(entry, 'is_flagged', False):
            return jsonify({'success': False, 'error': 'Entry is not flagged'}), 400

        _clear_flag(entry, notes, 'dismiss_flagged')
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@admin_bp.route('/api/admin/delete-flagged/<int:entry_id>', methods=['DELETE'])
@admin_required
@limiter.limit("30 per minute", key_func=_admin_key)
def delete_flagged(entry_id):
    try:
        entry = ActivityLog.query.get(entry_id)
        if not entry:
            return jsonify({'success': False, 'error': 'Entry not found'}), 404

        if not getattr(entry, 'is_flagged', False):
            return jsonify({'success': False, 'error': 'Entry is not flagged'}), 400

        details = (
            f'Deleted flagged entry {entry_id}: '
            f'action={entry.action}, user={entry.user_email}, '
            f'reason={getattr(entry, "flag_reason", None)}'
        )[:500]

        db.session.delete(entry)
        db.session.flush()

        log_activity(
            _current_admin_id(), 'admin', _current_admin_email(),
            'delete_flagged', details
        )
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


# ======================================================================
# AUDIT STATS + DASHBOARD STATS
# ======================================================================
@admin_bp.route('/api/admin/audit-stats')
@admin_required
@limiter.limit("60 per minute", key_func=_admin_key)
def admin_audit_stats():
    try:
        total_actions = ActivityLog.query.count()
        active_admins = User.query.filter_by(role='admin', is_active=True).count()
        active_operators = User.query.filter_by(role='operator', is_active=True).count()
        flagged_count = ActivityLog.query.filter(ActivityLog.is_flagged == True).count()

        return jsonify({
            'total_actions': total_actions,
            'active_admins': active_admins,
            'active_operators': active_operators,
            'flagged': flagged_count,
        })
    except Exception as e:
        print(f"[audit-stats] {e}")
        return jsonify({
            'total_actions': 0, 'active_admins': 0,
            'active_operators': 0, 'flagged': 0,
        })


@admin_bp.route('/api/admin/dashboard-stats')
@admin_required
@limiter.limit("60 per minute", key_func=_admin_key)
def dashboard_stats():
    try:
        total_reports = Report.query.count()
        active_operators = User.query.filter_by(role='operator', is_active=True).count()

        one_week_ago = datetime.now() - timedelta(days=7)
        broadcasts_this_week = Broadcast.query.filter(
            Broadcast.created_at >= one_week_ago
        ).count()

        severe_count = 0
        now = datetime.now()
        for station in STATIONS:
            try:
                north = get_directional_prediction(station, 'Northbound', now) or 0
                south = get_directional_prediction(station, 'Southbound', now) or 0
                if max(north, south) > 80:
                    severe_count += 1
            except Exception as e:
                print(f"[dashboard-stats] prediction failed for {station}: {e}")

        return jsonify({
            'total_reports': total_reports,
            'severe_count': severe_count,
            'active_operators': active_operators,
            'broadcasts_this_week': broadcasts_this_week,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'total_reports': 0, 'severe_count': 0,
            'active_operators': 0, 'broadcasts_this_week': 0,
            'error': str(e),
        }), 500


# ======================================================================
# OPERATOR MANAGEMENT
# ======================================================================
@admin_bp.route('/api/admin/operator-list')
@admin_required
@limiter.limit("120 per minute", key_func=_admin_key)
def operator_list():
    try:
        now = datetime.now()
        operators = User.query.filter_by(role='operator').filter(
            db.or_(
                User.is_active == True,
                User.invite_expires_at != None,
            )
        ).all()

        result = []
        for op in operators:
            name = op.username.split('@')[0] if '@' in op.username else op.username

            if op.access_level == 'line_wide':
                station_display = "All Stations"
            elif op.access_level == 'zone':
                station_display = f"{op.assigned_zone.upper()} Zone" if op.assigned_zone else "Zone Access"
            else:
                if op.assigned_stations:
                    try:
                        assigned = json.loads(op.assigned_stations)
                        station_display = assigned[0] if assigned else (op.favorite_station or "Not Assigned")
                    except Exception:
                        station_display = op.favorite_station or "Not Assigned"
                else:
                    station_display = op.favorite_station or "Not Assigned"

            invite_pending = (
                not op.is_active
                and op.invite_expires_at is not None
                and op.invite_expires_at > now
            )
            invite_expired = (
                not op.is_active
                and op.invite_expires_at is not None
                and op.invite_expires_at <= now
            )

            if invite_pending:
                status = 'pending'
            elif invite_expired:
                status = 'expired'
            elif op.is_active:
                status = 'active'
            else:
                status = 'deactivated'

            result.append({
                'id': op.id,
                'name': name,
                'email': op.username,
                'station': station_display,
                'joined': op.created_at.strftime('%b %d, %Y') if op.created_at else 'Unknown',
                'last_login': op.last_login.strftime('%b %d, %Y') if op.last_login else 'Never',
                'active': op.is_active,
                'access_level': op.access_level,
                'status': status,
                'invite_expires_at': op.invite_expires_at.isoformat() if op.invite_expires_at else None,
            })

        return jsonify(result)
    except Exception as e:
        print(f"[operator_list] {e}")
        return jsonify([])


def _generate_temp_password(length=10):
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def _generate_invite_token():
    return secrets.token_urlsafe(32)


@admin_bp.route('/api/admin/generate-invite', methods=['POST'])
@admin_required
@limiter.limit("20 per hour", key_func=_admin_key)
def generate_invite():
    try:
        data = request.get_json(silent=True) or {}
        email = (data.get('email') or '').strip().lower()
        station = data.get('station')
        access_level_type = data.get('access_level', 'standard')
        auth_method = data.get('auth_method', 'password')
        confirm_reactivate = bool(data.get('confirm_reactivate', False))

        if not email or '@' not in email:
            return jsonify({'success': False, 'error': 'Valid email required'}), 400

        if ALLOWED_INVITE_DOMAINS:
            domain = email.rsplit('@', 1)[-1]
            if domain not in ALLOWED_INVITE_DOMAINS:
                return jsonify({
                    'success': False,
                    'error': f'Email domain @{domain} is not allowed for operator invites'
                }), 400

        expiry = datetime.now() + timedelta(hours=INVITE_TTL_HOURS)

        if access_level_type == 'full' or station == 'All Stations (Line-Wide)':
            db_access_level = 'line_wide'
            assigned_stations = STATIONS
            favorite_station = None
        else:
            db_access_level = 'station'
            assigned_stations = [station] if station in STATIONS else ['North Ave']
            favorite_station = assigned_stations[0]

        existing = User.query.filter_by(username=email).first()

        # --- Existing active user ---
        if existing and existing.is_active:
            return jsonify({'success': False, 'error': 'Email already registered and active'}), 400

        # --- Existing deactivated user ---
        if existing and not existing.is_active:
            if not confirm_reactivate:
                return jsonify({
                    'success': False,
                    'error': (
                        'This account was deactivated. Pass confirm_reactivate=true '
                        'to revive it, or use a different email.'
                    ),
                }), 409

            if getattr(existing, 'banned', False):
                return jsonify({
                    'success': False,
                    'error': 'This account is banned. Unban it first.'
                }), 403

            invite_token = _generate_invite_token()
            existing.invite_token = invite_token
            existing.is_active = False
            existing.invite_expires_at = expiry
            existing.access_level = db_access_level
            existing.assigned_stations = json.dumps(assigned_stations)
            existing.favorite_station = favorite_station

            if auth_method == 'google':
                existing.password_hash = None
                db.session.commit()
                log_activity(
                    _current_admin_id(), 'admin', _current_admin_email(),
                    'reactivate_via_invite',
                    f'Re-invited deactivated Google operator {email}'
                )
                invite_link = (
                    f"{request.host_url}login/google/authorize"
                    f"?invite=true&email={email}"
                )
                return jsonify({
                    'success': True, 'link': invite_link,
                    'auth_method': 'google', 'expires_at': expiry.isoformat(),
                })

            temp_password = _generate_temp_password()
            existing.password = temp_password
            db.session.commit()

            log_activity(
                _current_admin_id(), 'admin', _current_admin_email(),
                'reactivate_via_invite',
                f'Re-invited deactivated operator {email}'
            )
            invite_link = f"{request.host_url}login?invite={invite_token}"
            return jsonify({
                'success': True, 'link': invite_link,
                'auth_method': 'password',
                'expires_at': expiry.isoformat(),
            })

        # --- Brand new user ---
        if auth_method == 'google':
            new_op = User(
                username=email, role='operator',
                access_level=db_access_level,
                assigned_stations=json.dumps(assigned_stations),
                favorite_station=favorite_station,
                created_at=datetime.now(),
                is_active=True,
                invite_expires_at=expiry,
                invite_token=_generate_invite_token(),
            )
            db.session.add(new_op)
            db.session.commit()

            log_activity(
                _current_admin_id(), 'admin', _current_admin_email(),
                'create_operator', f'Created Google-auth operator {email}'
            )
            invite_link = (
                f"{request.host_url}login/google/authorize"
                f"?invite=true&email={email}"
            )
            return jsonify({
                'success': True, 'link': invite_link,
                'auth_method': 'google', 'expires_at': expiry.isoformat(),
            })

        # Password-auth new user: inactive until they complete signup
        invite_token = _generate_invite_token()
        temp_password = _generate_temp_password()
        new_op = User(
            username=email, role='operator',
            access_level=db_access_level,
            assigned_stations=json.dumps(assigned_stations),
            favorite_station=favorite_station,
            created_at=datetime.now(),
            is_active=False,
            invite_expires_at=expiry,
            invite_token=invite_token,
        )
        new_op.password = temp_password
        db.session.add(new_op)
        db.session.commit()

        log_activity(
            _current_admin_id(), 'admin', _current_admin_email(),
            'create_operator', f'Created operator invite for {email}'
        )

        invite_link = f"{request.host_url}login?invite={invite_token}"
        return jsonify({
            'success': True, 'link': invite_link,
            'auth_method': 'password',
            'expires_at': expiry.isoformat(),
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@admin_bp.route('/api/admin/deactivate-operator/<int:operator_id>', methods=['POST'])
@admin_required
@limiter.limit("30 per minute", key_func=_admin_key)
def deactivate_operator(operator_id):
    try:
        operator = User.query.get(operator_id)
        if not operator or operator.role != 'operator':
            return jsonify({'success': False, 'error': 'Operator not found'}), 404

        operator.is_active = False
        operator.invite_token = None
        operator.invite_expires_at = None
        db.session.commit()

        log_activity(
            _current_admin_id(), 'admin', _current_admin_email(),
            'deactivate_operator', f'Deactivated operator: {operator.username}'
        )
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@admin_bp.route('/api/admin/reactivate-operator/<int:operator_id>', methods=['POST'])
@admin_required
@limiter.limit("30 per minute", key_func=_admin_key)
def reactivate_operator(operator_id):
    try:
        operator = User.query.get(operator_id)
        if not operator or operator.role != 'operator':
            return jsonify({'success': False, 'error': 'Operator not found'}), 404

        operator.is_active = True
        db.session.commit()

        log_activity(
            _current_admin_id(), 'admin', _current_admin_email(),
            'reactivate_operator', f'Reactivated operator: {operator.username}'
        )
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


# ======================================================================
# AUDIT LOG
# ======================================================================
@admin_bp.route('/api/admin/audit-log')
@admin_required
@limiter.limit("60 per minute", key_func=_admin_key)
def audit_log():
    try:
        limit = min(request.args.get('limit', 200, type=int), 1000)
        logs = ActivityLog.query.order_by(
            ActivityLog.timestamp.desc()
        ).limit(limit).all()

        user_ids = {log.user_id for log in logs if log.user_id}
        users = {
            u.id: u.username
            for u in User.query.filter(User.id.in_(user_ids)).all()
        } if user_ids else {}

        out = []
        for log in logs:
            user_name = (
                users.get(log.user_id)
                or log.user_email
                or 'System'
            )
            out.append({
                'id': log.id,
                'userType': log.user_type or 'system',
                'userName': user_name,
                'userEmail': log.user_email,
                'action': log.action,
                'details': log.details or '-',
                'target': log.details or '-',
                'ip_address': log.ip_address or '-',
                'timestamp': log.timestamp.isoformat() if log.timestamp else None,
                'is_flagged': bool(getattr(log, 'is_flagged', False)),
            })
        return jsonify(out)
    except Exception as e:
        print(f"[audit_log] {e}")
        return jsonify([]), 500


# ======================================================================
# DEBUG (admin-gated + dev-only)
# ======================================================================
@admin_bp.route('/api/admin/debug-check-flags')
@admin_required
def debug_check_flags():
    if not current_app.debug:
        return jsonify({'error': 'not available'}), 404

    try:
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('activity_log')]

        has_flag_columns = {
            'is_flagged': 'is_flagged' in columns,
            'flag_reason': 'flag_reason' in columns,
            'flagged_at': 'flagged_at' in columns,
        }

        flagged_count = (
            ActivityLog.query.filter(ActivityLog.is_flagged == True).count()
            if 'is_flagged' in columns else 0
        )
        total_count = ActivityLog.query.count()

        sample = []
        for log in ActivityLog.query.limit(5).all():
            sample.append({
                'id': log.id,
                'action': log.action,
                'is_flagged': getattr(log, 'is_flagged', 'column_missing'),
                'flag_reason': getattr(log, 'flag_reason', 'column_missing'),
            })

        return jsonify({
            'columns_exist': has_flag_columns,
            'flagged_count': flagged_count,
            'total_entries': total_count,
            'sample_entries': sample,
            'all_columns': columns,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ======================================================================
# STATION STATUS
# ======================================================================
@admin_bp.route('/api/admin/station-status')
@admin_required
@limiter.limit("120 per minute", key_func=_admin_key)
def station_status():
    try:
        now = datetime.now()
        result = []
        for station in STATIONS:
            north_cong = None
            south_cong = None
            try:
                north_cong = get_directional_prediction(station, 'Northbound', now)
                south_cong = get_directional_prediction(station, 'Southbound', now)
            except Exception as e:
                print(f"[station-status] {station}: {e}")

            result.append({
                'name': station,
                'northbound': {
                    'congestion': north_cong,
                    'status_text': _get_status_text(north_cong),
                    'status_class': _get_status_class(north_cong),
                    'ridership': 0,
                    'wait_time': '2-5 min',
                },
                'southbound': {
                    'congestion': south_cong,
                    'status_text': _get_status_text(south_cong),
                    'status_class': _get_status_class(south_cong),
                    'ridership': 0,
                    'wait_time': '2-5 min',
                },
            })

        return jsonify({'stations': result})
    except Exception as e:
        print(f"[station_status] {e}")
        return jsonify({'stations': []}), 500