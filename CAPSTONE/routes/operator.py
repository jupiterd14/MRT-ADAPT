from flask import Blueprint, session, request, jsonify, flash, redirect, url_for, render_template, current_app
from models import User, Report, Broadcast, db
from datetime import datetime, timedelta  # ADD timedelta here
import json, time
from .auth import login_required, log_activity

operator_bp = Blueprint('operator', __name__)

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

def get_operator_stations(user_id):
    """Get list of stations assigned to an operator"""
    user = User.query.get(user_id)
    if not user:
        return []
    
    if user.access_level == 'line_wide':
        return STATIONS
    elif user.access_level == 'zone':
        zones = {
            'north': ['North Ave', 'Quezon Ave', 'Kamuning', 'Cubao', 'Santolan'],
            'central': ['Ortigas', 'Shaw Blvd', 'Boni Ave', 'Guadalupe'],
            'south': ['Buendia', 'Ayala Ave', 'Magallanes', 'Taft']
        }
        return zones.get(user.assigned_zone, [])
    else:
        if user.assigned_stations:
            try:
                return json.loads(user.assigned_stations)
            except:
                return []
        return [user.favorite_station] if user.favorite_station else ['North Ave']

@operator_bp.route('/operator-dashboard')
@operator_bp.route('/operator_dashboard')
def operator_dashboard():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('auth.login'))
    
    user = User.query.get(session['user_id'])
    
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

# ========== ADD THIS MISSING ENDPOINT ==========
@operator_bp.route('/api/operator/broadcasts', methods=['GET'])
def get_operator_broadcasts():
    """Get broadcasts for operator's stations"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        # Auto-expire old broadcasts (but DON'T delete them from history)
        now = datetime.now()
        
        # Expire broadcasts older than 7 days
        auto_expire_cutoff = now - timedelta(days=7)
        old_broadcasts = Broadcast.query.filter(
            Broadcast.is_active == True,
            Broadcast.created_at < auto_expire_cutoff
        ).all()
        
        for broadcast in old_broadcasts:
            broadcast.is_active = False
        
        # Also expire by expires_at if set
        expired_by_date = Broadcast.query.filter(
            Broadcast.is_active == True,
            Broadcast.expires_at != None,
            Broadcast.expires_at <= now
        ).all()
        
        for broadcast in expired_by_date:
            broadcast.is_active = False
        
        if old_broadcasts or expired_by_date:
            db.session.commit()
            print(f"Auto-expired {len(old_broadcasts) + len(expired_by_date)} broadcasts")
        
        managed_stations = get_operator_stations(user_id)
        
        # Get ALL broadcasts (active AND inactive) for history
        # But limit to last 30 days to avoid overwhelming the UI
        thirty_days_ago = now - timedelta(days=30)
        all_broadcasts = Broadcast.query.filter(
            Broadcast.created_at >= thirty_days_ago
        ).order_by(Broadcast.created_at.desc()).limit(100).all()
        
        typeIcons = {
            "Train Breakdown": "fa-train", "Overcrowding": "fa-users", 
            "Maintenance": "fa-wrench", "Signal Issue": "fa-satellite-dish",
            "Gate Closure": "fa-door-closed", "General Notice": "fa-bullhorn"
        }
        
        result = []
        for broadcast in all_broadcasts:
            stations = json.loads(broadcast.stations) if broadcast.stations else []
            # Only show if broadcast affects any of operator's stations
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
                    'is_active': broadcast.is_active,  # IMPORTANT: Include this field
                    'icon': typeIcons.get(broadcast.disruption_type, 'fa-bullhorn')
                })
        
        return jsonify({'success': True, 'broadcasts': result})
    except Exception as e:
        print(f"Error getting broadcasts: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ========== BROADCAST CRUD ENDPOINTS ==========
@operator_bp.route('/api/operator/broadcast/<int:broadcast_id>', methods=['GET'])
def get_broadcast(broadcast_id):
    """Get a single broadcast by ID"""
    try:
        broadcast = Broadcast.query.get(broadcast_id)
        if not broadcast:
            return jsonify({'success': False, 'error': 'Broadcast not found'}), 404
        
        return jsonify({
            'success': True,
            'broadcast': {
                'id': broadcast.id,
                'title': broadcast.title,
                'message': broadcast.message,
                'severity': broadcast.severity,
                'disruption_type': broadcast.disruption_type,
                'direction': getattr(broadcast, 'direction', 'both'),
                'stations': json.loads(broadcast.stations) if broadcast.stations else []
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@operator_bp.route('/api/operator/broadcast/<int:broadcast_id>', methods=['PUT'])
def update_broadcast(broadcast_id):
    """Update a broadcast"""
    try:
        data = request.json
        broadcast = Broadcast.query.get(broadcast_id)
        
        if not broadcast:
            return jsonify({'success': False, 'error': 'Broadcast not found'}), 404
        
        # Update fields
        if 'title' in data:
            broadcast.title = data['title']
        if 'message' in data:
            broadcast.message = data['message']
        if 'severity' in data:
            broadcast.severity = data['severity']
        
        db.session.commit()
        
        log_activity(session.get('user_id'), 'operator', session.get('username'), 
                    'edit_broadcast', f'Edited broadcast #{broadcast_id}: {broadcast.title}')
        
        return jsonify({'success': True, 'message': 'Broadcast updated'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@operator_bp.route('/api/operator/broadcast/<int:broadcast_id>', methods=['DELETE'])
def delete_broadcast(broadcast_id):
    """Delete a broadcast (hard delete)"""
    try:
        broadcast = Broadcast.query.get(broadcast_id)
        
        if not broadcast:
            return jsonify({'success': False, 'error': 'Broadcast not found'}), 404
        
        db.session.delete(broadcast)
        db.session.commit()
        
        log_activity(session.get('user_id'), 'operator', session.get('username'), 
                    'delete_broadcast', f'Deleted broadcast #{broadcast_id}')
        
        return jsonify({'success': True, 'message': 'Broadcast deleted'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@operator_bp.route('/api/operator/send-broadcast', methods=['POST'])
def send_broadcast():
    try:
        data = request.json
        title = data.get('title')
        message = data.get('message')
        disruption_type = data.get('disruption_type')
        stations = data.get('stations')
        severity = data.get('severity')
        direction = data.get('direction', 'both')
        duration_minutes = data.get('duration_minutes', 60)
        
        operator_id = session.get('user_id')
        if not operator_id:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        # Calculate expiry time
        expires_at = None
        if duration_minutes and duration_minutes > 0:
            expires_at = datetime.now() + timedelta(minutes=duration_minutes)
        
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
            expires_at=expires_at
        )
        
        db.session.add(broadcast)
        db.session.commit()
        
        log_activity(operator_id, 'operator', session.get('username'), 'send_broadcast',
                    f'Broadcast: "{title}" to {len(stations)} stations (Direction: {direction}, Duration: {duration_minutes} min)')
        
        return jsonify({'success': True, 'message': 'Broadcast sent', 'broadcast_id': broadcast.id})
    except Exception as e:
        db.session.rollback()
        print(f"Error sending broadcast: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@operator_bp.route('/api/operator/deactivate-broadcast/<int:broadcast_id>', methods=['POST'])
def deactivate_broadcast(broadcast_id):
    try:
        broadcast = Broadcast.query.get(broadcast_id)
        if broadcast:
            broadcast.is_active = False
            db.session.commit()
        
        log_activity(session.get('user_id'), 'operator', session.get('username'),
                    'deactivate_broadcast', f'Deactivated broadcast ID: {broadcast_id}')
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@operator_bp.route('/api/operator/override-congestion', methods=['POST'])
def override_congestion():
    try:
        data = request.json
        station = data.get('station')
        level = data.get('level')
        congestion_value = data.get('congestion_value')
        duration = data.get('duration')
        reason = data.get('reason', '')
        direction = data.get('direction', 'southbound')
        
        operator_id = session.get('user_id')
        operator_email = session.get('username')
        
        override_key = f"{station}_{direction}"
        
        if 'overrides' not in current_app.config:
            current_app.config['overrides'] = {}
        
        expiry = None
        if duration != 'manual':
            expiry = time.time() + (int(duration) * 60)
        
        current_app.config['overrides'][override_key] = {
            'station': station, 'direction': direction, 'level': level,
            'congestion': congestion_value, 'operator': operator_email,
            'reason': reason, 'expiry': expiry, 'timestamp': datetime.now().isoformat()
        }
        
        log_activity(operator_id, 'operator', operator_email, 'override_congestion',
                    f'Overrode {station} ({direction}) to {level} ({congestion_value}%)')
        
        return jsonify({'success': True, 'message': f'{station} ({direction}) set to {level}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@operator_bp.route('/api/operator/clear-override', methods=['POST'])
def clear_override():
    try:
        data = request.json
        station = data.get('station')
        
        if 'overrides' in current_app.config and station in current_app.config['overrides']:
            override_info = current_app.config['overrides'][station]
            del current_app.config['overrides'][station]
            
            log_activity(session.get('user_id'), 'operator', session.get('username'),
                        'clear_override', f'Cleared override for {station}')
            
            return jsonify({'success': True, 'message': f'Override cleared for {station}'})
        
        return jsonify({'success': False, 'error': 'No active override found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@operator_bp.route('/api/operator/get-overrides', methods=['GET'])
def get_overrides():
    if 'overrides' not in current_app.config:
        current_app.config['overrides'] = {}
    
    current_time = time.time()
    active_overrides = {}
    
    for station, override in current_app.config['overrides'].items():
        if override['expiry'] is None or override['expiry'] > current_time:
            active_overrides[station] = override
    
    return jsonify({'overrides': active_overrides})

@operator_bp.route('/api/operator/flagged-reports')
def get_flagged_reports():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    managed_stations = get_operator_stations(user_id)
    
    flagged_reports = Report.query.filter(
        Report.flagged == True,
        Report.station.in_(managed_stations),
        Report.reviewed == False
    ).order_by(Report.timestamp.desc()).all()
    
    result = []
    for report in flagged_reports:
        reporter_name = 'Anonymous'
        if not report.anonymous and report.user:
            reporter_name = report.user.username.split('@')[0] if '@' in report.user.username else report.user.username
        
        result.append({
            'id': report.id, 'station': report.station,
            'reported_congestion': report.reported_congestion,
            'remarks': report.remarks, 'timestamp': report.timestamp.isoformat(),
            'reporter': reporter_name, 'anonymous': report.anonymous,
            'flag_count': getattr(report, 'flag_count', 1), 'photo_path': report.photo_path
        })
    
    return jsonify(result)

@operator_bp.route('/api/operator/review-flagged/<int:report_id>', methods=['POST'])
def review_flagged_report(report_id):
    data = request.json
    action = data.get('action')
    reason = data.get('reason', '')
    
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
    today_reports = Report.query.filter(
        Report.station.in_(managed_stations), Report.timestamp >= datetime.now().date()
    ).count()
    
    return jsonify({
        'total_reports': total_reports, 'flagged_reports': flagged_reports, 'today_reports': today_reports
    })

@operator_bp.route('/api/operator/my-activity')
def operator_my_activity():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        
        logs = ActivityLog.query.filter_by(user_id=user_id).order_by(ActivityLog.timestamp.desc()).limit(50).all()
        
        log_data = [{
            'action': log.action, 'details': log.details,
            'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        } for log in logs]
        
        return jsonify(log_data)
    except Exception as e:
        return jsonify([]), 500