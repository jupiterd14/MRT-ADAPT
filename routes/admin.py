from flask import Blueprint, render_template, session, request, jsonify, flash, redirect, url_for
from models import User, Report, Broadcast, ActivityLog, db
from datetime import datetime, timedelta
import secrets, string, json, os
from .auth import log_activity
from routes.api_predict import get_directional_prediction

admin_bp = Blueprint('admin', __name__)

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

# DOTr Official Platform Capacities (for congestion calculation)
MRT3_PLATFORM_CAPACITY = {
    "North Ave": 1142, "Quezon Ave": 1195, "Kamuning": 1364, "Cubao": 1747,
    "Santolan": 1306, "Ortigas": 1331, "Shaw Blvd": 1619, "Boni Ave": 1417,
    "Guadalupe": 1301, "Buendia": 1645, "Ayala Ave": 1222, "Magallanes": 1202,
    "Taft": 720
}

def get_station_prediction(station_name):
    """Get station prediction from app config"""
    from flask import current_app
    
    # Use the correct config key (same as your app.py)
    if 'GET_STATION_PREDICTION' in current_app.config:
        return current_app.config['GET_STATION_PREDICTION'](station_name)
    
    # Fallback
    return 50


@admin_bp.route('/admin/dashboard')
def admin_dashboard():
    is_admin = (session.get('admin_logged_in') or session.get('is_admin') or session.get('role') == 'admin')
    
    if not is_admin:
        flash('Please login as admin to access this page.', 'warning')
        return redirect(url_for('auth.login'))
    
    total_users = User.query.count()
    operator_count = User.query.filter_by(role='operator').count()
    commuter_count = User.query.filter_by(role='commuter').count()
    admin_count = User.query.filter_by(role='admin').count()
    
    users_data = []
    for user in User.query.all():
        joined_date = user.created_at.strftime('%b %d, %Y') if user.created_at else 'Unknown'
        last_active = 'Never'
        if user.last_login:
            days_ago = (datetime.now() - user.last_login).days
            last_active = 'Today' if days_ago == 0 else 'Yesterday' if days_ago == 1 else f'{days_ago} days ago'
        
        users_data.append({
            'id': user.id, 'email': user.username, 'role': user.role or 'commuter',
            'joined': joined_date, 'last': last_active, 'active': user.is_active
        })
    
    return render_template('admin_dashboard.html',
                         admin_email=session.get('username', 'Admin'),
                         total_users=total_users, operator_count=operator_count,
                         commuter_count=commuter_count, admin_count=admin_count,
                         users_data=users_data)

    
@admin_bp.route('/api/admin/recent-activities-list')
def admin_recent_activities():
    """Get recent IMPORTANT activities for admin dashboard (limit to 4 most recent)"""
    try:
        # Get recent activity logs - limit to last 4 actions
        recent_logs = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(4).all()
        
        activities = []
        for log in recent_logs:
            # Determine icon based on action type
            icon = 'user'
            icon_color = '#3B82F6'
            title = ''
            description = ''
            
            if 'login' in log.action and 'failed' not in log.action:
                icon = 'sign-in-alt'
                icon_color = '#22C55E'
                title = 'Login'
                description = f'{log.user_email} logged in'
            elif 'logout' in log.action:
                icon = 'sign-out-alt'
                icon_color = '#EF4444'
                title = 'Logout'
                description = f'{log.user_email} logged out'
            elif 'broadcast' in log.action:
                icon = 'bullhorn'
                icon_color = '#8B5CF6'
                title = 'Broadcast Sent'
                # Extract station info from details
                if 'Station:' in log.details:
                    station_part = log.details.split('Station:')[-1].strip()
                    description = f'{log.user_email} sent alert to {station_part}'
                else:
                    description = log.details or f'{log.user_email} sent broadcast'
            elif 'override' in log.action:
                icon = 'edit'
                icon_color = '#F59E0B'
                title = 'Override'
                description = log.details or f'{log.user_email} overrode congestion'
            elif 'create_operator' in log.action:
                icon = 'user-plus'
                icon_color = '#10B981'
                title = 'Operator Created'
                description = log.details or f'{log.user_email} created new operator'
            elif 'deactivate_operator' in log.action:
                icon = 'user-slash'
                icon_color = '#EF4444'
                title = 'Operator Deactivated'
                description = log.details or f'{log.user_email} deactivated operator'
            elif 'reactivate_operator' in log.action:
                icon = 'user-check'
                icon_color = '#10B981'
                title = 'Operator Reactivated'
                description = log.details or f'{log.user_email} reactivated operator'
            elif 'login_failed' in log.action:
                icon = 'exclamation-triangle'
                icon_color = '#EF4444'
                title = 'Login Failed'
                description = f'Failed login attempt for {log.user_email}'
            else:
                title = log.action.replace('_', ' ').title()
                description = log.details or f'{log.user_email} performed {log.action}'
            
            activities.append({
                'icon': icon,
                'icon_color': icon_color,
                'title': title,
                'description': description[:100],  # Limit description length
                'station': None,
                'time': log.timestamp.strftime('%I:%M %p')
            })
        
        # If no activities, show sample (only 1-2)
        if not activities:
            activities = [
                {'icon': 'user-plus', 'icon_color': '#22C55E', 'title': 'System Ready', 'description': 'Admin dashboard initialized', 'station': None, 'time': 'Just now'}
            ]
        
        return jsonify(activities)
    except Exception as e:
        print(f"Error getting recent activities: {e}")
        return jsonify([])
    

@admin_bp.route('/api/admin/profile')
def admin_profile():
    """Get admin profile info"""
    try:
        if session.get('admin_logged_in') or session.get('is_admin'):
            return jsonify({
                'username': session.get('username', 'Admin'),
                'role': 'admin'
            })
        return jsonify({'error': 'Not authorized'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========== FLAGGED ACTIONS ENDPOINTS (ONLY ONE OF EACH) ==========
@admin_bp.route('/api/admin/flagged-actions')
def flagged_actions():
    """Get flagged reports for admin review"""
    try:
        # Get reports that have been flagged (using the 'flagged' column)
        flagged_reports = Report.query.filter(
            Report.flagged == True
        ).order_by(Report.timestamp.desc()).all()
        
        print(f"Found {len(flagged_reports)} flagged reports")  # Debug
        
        flagged_data = []
        for report in flagged_reports:
            reporter_name = 'Anonymous'
            if not report.anonymous and report.user:
                reporter_name = report.user.username.split('@')[0] if '@' in report.user.username else report.user.username
            
            flagged_data.append({
                'id': report.id,
                'userType': 'report',
                'userName': reporter_name,
                'action': 'Flagged Report',
                'target': f'{report.station} - {report.direction or "both"}',
                'details': report.remarks or 'No remarks',
                'flag_reason': f'Flagged {report.flag_count} times by users',
                'ip_address': '-',
                'timestamp': report.timestamp.isoformat(),
                'station': report.station,
                'congestion': report.reported_congestion,
                'remarks': report.remarks,
                'photo_path': report.photo_path,
                'flag_count': report.flag_count
            })
        
        return jsonify(flagged_data)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([])


@admin_bp.route('/api/admin/flag-audit-entry/<int:entry_id>', methods=['POST'])
def flag_audit_entry(entry_id):
    """Flag an audit entry for review"""
    try:
        data = request.get_json()
        reason = data.get('reason', 'No reason provided')
        
        log_entry = ActivityLog.query.get(entry_id)
        if not log_entry:
            return jsonify({'success': False, 'error': 'Entry not found'}), 404
        
        log_entry.is_flagged = True
        log_entry.flag_reason = reason
        log_entry.flagged_at = datetime.now()
        db.session.commit()
        
        log_activity(session.get('user_id'), 'admin', session.get('username'), 
                    'flag_audit_entry', f'Flagged entry {entry_id} for review. Reason: {reason}')
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@admin_bp.route('/api/admin/approve-flagged/<int:entry_id>', methods=['POST'])
def approve_flagged(entry_id):
    """Approve a flagged entry - clear the flag"""
    try:
        data = request.get_json() or {}
        admin_notes = data.get('admin_notes', '')
        
        log_entry = ActivityLog.query.get(entry_id)
        if not log_entry:
            return jsonify({'success': False, 'error': 'Entry not found'}), 404
        
        log_entry.is_flagged = False
        if hasattr(log_entry, 'flag_reason'):
            log_entry.flag_reason = None
        if hasattr(log_entry, 'flagged_at'):
            log_entry.flagged_at = None
        if hasattr(log_entry, 'admin_review_notes'):
            log_entry.admin_review_notes = admin_notes
        if hasattr(log_entry, 'reviewed_by'):
            log_entry.reviewed_by = session.get('username', 'admin')
        if hasattr(log_entry, 'reviewed_at'):
            log_entry.reviewed_at = datetime.now()
        
        db.session.commit()
        
        log_activity(session.get('user_id'), 'admin', session.get('username'), 
                    'approve_flagged', f'Approved flagged entry {entry_id}. Notes: {admin_notes}')
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@admin_bp.route('/api/admin/dismiss-flagged/<int:entry_id>', methods=['POST'])
def dismiss_flagged(entry_id):
    """Dismiss a flag but keep the entry"""
    try:
        data = request.get_json() or {}
        admin_notes = data.get('admin_notes', '')
        
        log_entry = ActivityLog.query.get(entry_id)
        if not log_entry:
            return jsonify({'success': False, 'error': 'Entry not found'}), 404
        
        log_entry.is_flagged = False
        if hasattr(log_entry, 'flag_reason'):
            log_entry.flag_reason = None
        if hasattr(log_entry, 'flagged_at'):
            log_entry.flagged_at = None
        if hasattr(log_entry, 'admin_review_notes'):
            log_entry.admin_review_notes = admin_notes
        if hasattr(log_entry, 'reviewed_by'):
            log_entry.reviewed_by = session.get('username', 'admin')
        if hasattr(log_entry, 'reviewed_at'):
            log_entry.reviewed_at = datetime.now()
        
        db.session.commit()
        
        log_activity(session.get('user_id'), 'admin', session.get('username'), 
                    'dismiss_flagged', f'Dismissed flag on entry {entry_id}. Notes: {admin_notes}')
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@admin_bp.route('/api/admin/delete-flagged/<int:entry_id>', methods=['DELETE'])
def delete_flagged(entry_id):
    """Delete a flagged entry permanently"""
    try:
        data = request.get_json() or {}
        admin_notes = data.get('admin_notes', '')
        
        log_entry = ActivityLog.query.get(entry_id)
        if not log_entry:
            return jsonify({'success': False, 'error': 'Entry not found'}), 404
        
        log_activity(session.get('user_id'), 'admin', session.get('username'), 
                    'delete_flagged', f'Deleted flagged entry {entry_id}. Notes: {admin_notes}')
        
        db.session.delete(log_entry)
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


# ========== AUDIT & STATS ENDPOINTS ==========
@admin_bp.route('/api/admin/audit-stats')
def admin_audit_stats():
    try:
        total_actions = ActivityLog.query.count()
        active_admins = User.query.filter_by(role='admin', is_active=True).count()
        active_operators = User.query.filter_by(role='operator', is_active=True).count()
        
        # Count reports that are flagged (using the 'flagged' column)
        flagged_reports = Report.query.filter(Report.flagged == True).count()
        
        return jsonify({
            'total_actions': total_actions,
            'active_admins': active_admins,
            'active_operators': active_operators,
            'flagged': flagged_reports  # Now shows flagged reports count
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'total_actions': 0, 'active_admins': 0, 'active_operators': 0, 'flagged': 0})
    

@admin_bp.route('/api/admin/dashboard-stats')
def dashboard_stats():
    """Get dashboard stats - uses capacity-based congestion for severe count"""
    try:
        from flask import current_app
        from datetime import datetime, timedelta
        from routes.api_predict import get_directional_prediction  # ✅ Add this import
        
        print("\n" + "="*50)
        print("📊 FETCHING DASHBOARD STATS")
        print("="*50)
        
        # Count reports
        total_reports = Report.query.count()
        print(f"   Total Reports: {total_reports}")
        
        # Count operators
        active_operators = User.query.filter_by(role='operator', is_active=True).count()
        print(f"   Active Operators: {active_operators}")
        
        # Count broadcasts this week
        one_week_ago = datetime.now() - timedelta(days=7)
        broadcasts_this_week = Broadcast.query.filter(
            Broadcast.created_at >= one_week_ago
        ).count()
        print(f"   Broadcasts This Week: {broadcasts_this_week}")
        
        # Calculate severe count from live map API
        severe_count = 0
        try:
            # Use the live map API to get current congestion
            import requests
            live_response = requests.get('http://localhost:5000/api/live-map/directions/v2', timeout=5)
            if live_response.status_code == 200:
                live_data = live_response.json()
                for station in STATIONS:
                    north = live_data.get('northbound', {}).get(station, {}).get('congestion', 0)
                    south = live_data.get('southbound', {}).get(station, {}).get('congestion', 0)
                    if north > 80 or south > 80:
                        severe_count += 1
                print(f"   Severe Count (from live map): {severe_count}")
            else:
                print(f"   ⚠️ Live map API returned status {live_response.status_code}")
        except Exception as e:
            print(f"   ⚠️ Could not fetch live map data: {e}")
            # ✅ FIX: Use the same import as station_status()
            now = datetime.now()
            for station in STATIONS:
                try:
                    north = get_directional_prediction(station, 'Northbound', now)
                    south = get_directional_prediction(station, 'Southbound', now)
                    if max(north or 0, south or 0) > 80:
                        severe_count += 1
                except Exception as e2:
                    print(f"   ⚠️ Error getting prediction for {station}: {e2}")
        
        print(f"   Final Severe Count: {severe_count}")
        print("="*50)
        
        return jsonify({
            'total_reports': total_reports,
            'severe_count': severe_count,
            'active_operators': active_operators,
            'broadcasts_this_week': broadcasts_this_week
        })
        
    except Exception as e:
        import traceback
        print(f"❌ Error in dashboard_stats: {e}")
        traceback.print_exc()
        
        return jsonify({
            'total_reports': 0,
            'severe_count': 0,
            'active_operators': 0,
            'broadcasts_this_week': 0,
            'error': str(e)
        }), 200

@admin_bp.route('/api/admin/operator-list')
def operator_list():
    try:
        log_activity(session.get('user_id'), 'admin', session.get('username'), 
                    'view_operators', 'Viewed operator list')
        
        operators = User.query.filter_by(role='operator').all()
        operator_data = []
        
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
                        station_display = assigned[0] if assigned else op.favorite_station or "Not Assigned"
                    except:
                        station_display = op.favorite_station or "Not Assigned"
                else:
                    station_display = op.favorite_station or "Not Assigned"
            
            operator_data.append({
                'id': op.id, 'name': name, 'email': op.username, 'station': station_display,
                'joined': op.created_at.strftime('%b %d, %Y') if op.created_at else 'Unknown',
                'last_login': op.last_login.strftime('%b %d, %Y') if op.last_login else 'Never',
                'active': op.is_active, 'access_level': op.access_level
            })
        
        return jsonify(operator_data)
    except Exception as e:
        return jsonify([])


@admin_bp.route('/api/admin/generate-invite', methods=['POST'])
def generate_invite():
    try:
        data = request.json
        email = data.get('email')
        station = data.get('station')
        access_level_type = data.get('access_level', 'standard')
        auth_method = data.get('auth_method', 'password')
        
        existing_user = User.query.filter_by(username=email).first()
        if existing_user and existing_user.is_active:
            return jsonify({'success': False, 'error': 'Email already registered and active'}), 400
        
        if existing_user and not existing_user.is_active:
            existing_user.is_active = True
            if auth_method == 'google':
                existing_user.password_hash = None
            else:
                temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
                existing_user.password = temp_password
            db.session.commit()
            
            if auth_method == 'google':
                invite_link = f"{request.host_url}login/google/authorize?invite=true&email={email}"
                return jsonify({'success': True, 'link': invite_link, 'auth_method': 'google'})
            else:
                invite_link = f"{request.host_url}login?email={email}&temp={temp_password}&station={station}"
                return jsonify({'success': True, 'link': invite_link, 'auth_method': 'password'})
        
        if access_level_type == 'full' or station == 'All Stations (Line-Wide)':
            db_access_level = 'line_wide'
            assigned_stations = STATIONS
            favorite_station = None
        else:
            db_access_level = 'station'
            assigned_stations = [station] if station in STATIONS else ['North Ave']
            favorite_station = assigned_stations[0]
        
        if auth_method == 'google':
            new_operator = User(
                username=email, role='operator', access_level=db_access_level,
                assigned_stations=json.dumps(assigned_stations), favorite_station=favorite_station,
                created_at=datetime.now(), is_active=True
            )
            db.session.add(new_operator)
            db.session.commit()
            invite_link = f"{request.host_url}login/google/authorize?invite=true&email={email}"
            return jsonify({'success': True, 'link': invite_link, 'auth_method': 'google'})
        else:
            new_operator = User(
                username=email, role='operator', access_level=db_access_level,
                assigned_stations=json.dumps(assigned_stations), favorite_station=favorite_station,
                created_at=datetime.now(), is_active=False
            )
            temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
            new_operator.password = temp_password
            db.session.add(new_operator)
            db.session.commit()
            invite_link = f"{request.host_url}login?email={email}&temp={temp_password}&station={station}"
            return jsonify({'success': True, 'link': invite_link, 'auth_method': 'password'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@admin_bp.route('/api/admin/deactivate-operator/<int:operator_id>', methods=['POST'])
def deactivate_operator(operator_id):
    try:
        operator = User.query.get(operator_id)
        if operator and operator.role == 'operator':
            operator.is_active = False
            db.session.commit()
            log_activity(session.get('user_id'), 'admin', session.get('username'), 
                        'deactivate_operator', f'Deactivated operator: {operator.username}')
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Operator not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@admin_bp.route('/api/admin/reactivate-operator/<int:operator_id>', methods=['POST'])
def reactivate_operator(operator_id):
    try:
        operator = User.query.get(operator_id)
        if operator and operator.role == 'operator':
            operator.is_active = True
            db.session.commit()
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Operator not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@admin_bp.route('/api/admin/audit-log')
def audit_log():
    try:
        limit = request.args.get('limit', 200, type=int)
        logs = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(limit).all()
        
        log_data = []
        for log in logs:
            user_name = log.user_email or 'System'
            if log.user_id:
                user = User.query.get(log.user_id)
                if user:
                    user_name = user.username
            
            log_data.append({
                'id': log.id, 'userType': log.user_type or 'system', 'userName': user_name,
                'userEmail': log.user_email, 'action': log.action, 'details': log.details or '-',
                'target': log.details or '-', 'ip_address': log.ip_address or '-',
                'timestamp': log.timestamp.isoformat()
            })
        return jsonify(log_data)
    except Exception as e:
        return jsonify([]), 500


@admin_bp.route('/api/admin/debug-check-flags')
def debug_check_flags():
    """Debug endpoint to check flag columns and data"""
    try:
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('activity_log')]
        
        # Check what columns exist
        has_flag_columns = {
            'is_flagged': 'is_flagged' in columns,
            'flag_reason': 'flag_reason' in columns,
            'flagged_at': 'flagged_at' in columns
        }
        
        # Check if any entries have is_flagged = True
        flagged_count = ActivityLog.query.filter(ActivityLog.is_flagged == True).count() if 'is_flagged' in columns else 0
        
        # Check total entries
        total_count = ActivityLog.query.count()
        
        # Show some sample entries
        sample_entries = []
        for log in ActivityLog.query.limit(5).all():
            sample_entries.append({
                'id': log.id,
                'action': log.action,
                'is_flagged': getattr(log, 'is_flagged', 'column_missing'),
                'flag_reason': getattr(log, 'flag_reason', 'column_missing')
            })
        
        return jsonify({
            'columns_exist': has_flag_columns,
            'flagged_count': flagged_count,
            'total_entries': total_count,
            'sample_entries': sample_entries,
            'all_columns': columns
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@admin_bp.route('/api/admin/station-status')
def station_status():
    """Get station status for admin dashboard - fetches both directions at once"""
    try:
        from flask import current_app
        from datetime import datetime
        from routes.api_predict import get_directional_prediction  # IMPORT FROM API_PREDICT
        
        stations = current_app.config.get('STATIONS', STATIONS)
        now = datetime.now()
        
        def get_congestion(station_name, direction):
            """Get congestion using the SAME function as live map (api_predict)"""
            try:
                # Use the SAME function as api_other.py live-map endpoint
                prediction = get_directional_prediction(station_name, direction, now)
                return max(0, min(100, prediction))
            except Exception as e:
                print(f"⚠️ Error predicting {station_name} {direction}: {e}")
                # Fallback based on time
                hour = now.hour
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    return 65
                return 35
        
        def get_status_info(congestion):
            if congestion > 80:
                return "SEVERE", "status-severe"
            elif congestion > 60:
                return "CONGESTED", "status-congested"
            elif congestion > 30:
                return "MODERATE", "status-moderate"
            else:
                return "LIGHT", "status-light"
        
        result = []
        for station in stations:
            north_congestion = get_congestion(station, 'Northbound')
            south_congestion = get_congestion(station, 'Southbound')
            
            north_text, north_class = get_status_info(north_congestion)
            south_text, south_class = get_status_info(south_congestion)
            
            # Add ridership based on DOTr capacity
            capacity = MRT3_PLATFORM_CAPACITY.get(station, 1000)
            
            result.append({
                'name': station,
                'northbound': {
                    'congestion': round(north_congestion, 1),
                    'status_text': north_text,
                    'status_class': north_class,
                    'ridership': int((north_congestion / 100) * capacity)
                },
                'southbound': {
                    'congestion': round(south_congestion, 1),
                    'status_text': south_text,
                    'status_class': south_class,
                    'ridership': int((south_congestion / 100) * capacity)
                }
            })
        
        # Debug print first few
        if result:
            print(f"📊 Admin Dashboard - First station: {result[0]['name']} North: {result[0]['northbound']['congestion']}%")
        
        return jsonify({
            'stations': result,
            'timestamp': now.isoformat()
        })
    except Exception as e:
        print(f"Error in station_status: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'stations': []})