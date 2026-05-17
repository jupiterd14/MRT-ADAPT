from flask import Blueprint, render_template, session, request, jsonify, flash, redirect, url_for
from models import User, Report, Broadcast, ActivityLog, db
from datetime import datetime, timedelta
import secrets, string, json, os
from .auth import log_activity

admin_bp = Blueprint('admin', __name__)

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

def get_station_prediction(station_name):
    """Will be imported from main app - placeholder"""
    from flask import current_app
    return current_app.config.get('PREDICTION_FUNC', lambda x: 50)(station_name)

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


@admin_bp.route('/api/admin/audit-stats')
def admin_audit_stats():
    """Get real audit statistics from database"""
    try:
        total_actions = ActivityLog.query.count()
        
        # Count unique active admins (users who logged in recently)
        active_admins = User.query.filter_by(role='admin', is_active=True).count()
        
        # Count active operators
        active_operators = User.query.filter_by(role='operator', is_active=True).count()
        
        # Count flagged actions (failed logins, deactivations, etc.)
        flagged = ActivityLog.query.filter(
            ActivityLog.action.in_(['login_failed', 'deactivate_operator'])
        ).count()
        
        return jsonify({
            'total_actions': total_actions,
            'active_admins': active_admins,
            'active_operators': active_operators,
            'flagged': flagged
        })
    except Exception as e:
        print(f"Error getting audit stats: {e}")
        return jsonify({'total_actions': 0, 'active_admins': 0, 'active_operators': 0, 'flagged': 0})


@admin_bp.route('/api/admin/dashboard-stats')
def dashboard_stats():
    try:
        total_reports = Report.query.count()
        severe_count = 0
        for station in STATIONS:
            congestion = get_station_prediction(station)
            if congestion > 80:
                severe_count += 1
        
        active_operators = User.query.filter_by(role='operator', is_active=True).count()
        one_week_ago = datetime.now() - timedelta(days=7)
        broadcasts_this_week = Broadcast.query.filter(Broadcast.created_at >= one_week_ago).count()
        
        return jsonify({
            'total_reports': total_reports, 'severe_count': severe_count,
            'active_operators': active_operators, 'broadcasts_this_week': broadcasts_this_week
        })
    except Exception as e:
        return jsonify({'total_reports': 0, 'severe_count': 0, 'active_operators': 0, 'broadcasts_this_week': 0})

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

@admin_bp.route('/api/admin/station-status')
def station_status():
    """Get station status for admin dashboard - fetches both directions at once"""
    try:
        from flask import current_app
        from services.model_loader import directional_models, directional_scalers
        from services import get_feature_sequence_for_station
        from datetime import datetime
        
        stations = current_app.config.get('STATIONS', STATIONS)
        now = datetime.now()
        
        def get_congestion(station_name, direction):
            """Get congestion for a specific station and direction"""
            model_key = f"{station_name}_{direction}"
            
            if model_key in directional_models:
                try:
                    sequence = get_feature_sequence_for_station(station_name, direction, now)
                    if sequence is not None and len(sequence) == 24:
                        feature_scaler = directional_scalers.get(f'{model_key}_feature')
                        target_scaler = directional_scalers.get(f'{model_key}_target')
                        
                        if feature_scaler and target_scaler:
                            scaled_sequence = feature_scaler.transform(sequence)
                            input_sequence = scaled_sequence.reshape(1, 24, -1)
                            pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
                            prediction = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                            return max(0, min(100, prediction))
                except Exception as e:
                    print(f"⚠️ Error predicting {station_name} {direction}: {e}")
            
            # Fallback based on time of day
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
        
        # Calculate both directions once
        result = []
        for station in stations:
            north_congestion = get_congestion(station, 'Northbound')
            south_congestion = get_congestion(station, 'Southbound')
            
            north_text, north_class = get_status_info(north_congestion)
            south_text, south_class = get_status_info(south_congestion)
            
            result.append({
                'name': station,
                'northbound': {
                    'congestion': round(north_congestion, 1),
                    'status_text': north_text,
                    'status_class': north_class
                },
                'southbound': {
                    'congestion': round(south_congestion, 1),
                    'status_text': south_text,
                    'status_class': south_class
                }
            })
        
        return jsonify({
            'stations': result,
            'timestamp': now.isoformat()
        })
    except Exception as e:
        print(f"Error in station_status: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'stations': []})