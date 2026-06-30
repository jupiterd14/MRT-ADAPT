from flask import Blueprint, session, request, jsonify, flash, redirect, url_for, render_template, current_app
from models import User, Report, Broadcast, db
from datetime import datetime, timedelta
import json, time, math, os
from .auth import login_required, log_activity

operator_bp = Blueprint('operator', __name__)

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

STATION_BASE_CAPACITY = {
    "North Ave": 12000, "Quezon Ave": 9000, "Kamuning": 7500, "Cubao": 15000,
    "Santolan": 8000, "Ortigas": 9500, "Shaw Blvd": 11000, "Boni Ave": 8500,
    "Guadalupe": 10000, "Buendia": 9000, "Ayala Ave": 14000, "Magallanes": 9000, "Taft": 16000
}

# ========== PERSISTENT OVERRIDE STORAGE ==========
OVERRIDES_FILE = 'overrides.json'

def load_overrides():
    """Load overrides from file"""
    if os.path.exists(OVERRIDES_FILE):
        try:
            with open(OVERRIDES_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading overrides: {e}")
            return {}
    return {}

def save_overrides(overrides):
    """Save overrides to file"""
    try:
        with open(OVERRIDES_FILE, 'w') as f:
            json.dump(overrides, f, indent=2)
        print(f"✅ Saved {len(overrides)} overrides to file")
    except Exception as e:
        print(f"Error saving overrides: {e}")

def get_active_overrides():
    """Get active overrides from file with expiry check"""
    overrides = load_overrides()
    current_time = time.time()
    active = {}
    expired_keys = []
    
    for key, override in overrides.items():
        expiry = override.get('expiry')
        if expiry is None or expiry > current_time:
            active[key] = override
        else:
            expired_keys.append(key)
    
    # Remove expired keys
    if expired_keys:
        for key in expired_keys:
            del overrides[key]
        save_overrides(overrides)
    
    return active

def _get_congestion_from_prediction(pred_scaled, target_scaler, station_name):
    """
    Convert model prediction to congestion percentage.
    MATCHES api_other.py implementation for consistency.
    """
    raw_value = float(pred_scaled[0][0]) if hasattr(pred_scaled, '__getitem__') else float(pred_scaled)
    
    # Inverse transform using target scaler
    if target_scaler is not None:
        try:
            passenger_count = float(target_scaler.inverse_transform([[raw_value]])[0][0])
        except Exception as e:
            print(f"   ⚠️ Inverse transform failed: {e}")
            capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
            passenger_count = raw_value * capacity * 1.5
    else:
        # Fallback
        capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
        passenger_count = raw_value * capacity * 1.5
    
    # ========== FIX: Cap passenger_count at 0 ==========
    passenger_count = max(0, passenger_count)
    
    # Get station capacity
    capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
    
    # Cap at station capacity for congestion percentage
    capped_passengers = min(passenger_count, capacity)
    
    # Calculate congestion percentage (0-100%)
    congestion = (capped_passengers / capacity * 100)
    congestion = max(0, min(congestion, 100))
    
    return congestion, passenger_count

@operator_bp.route('/api/reports', methods=['GET'])
def get_reports():
    """Get reports for operator dashboard - ONLY non-archived reports"""
    try:
        # Debug: Check total reports count
        total_reports = Report.query.count()
        archived_count = Report.query.filter(Report.archived == True).count()
        active_count = Report.query.filter(Report.archived == False).count()
        
        print(f"📊 REPORT STATS: Total={total_reports}, Archived={archived_count}, Active={active_count}")
        
        # Get all reports that are NOT archived
        reports = Report.query.filter(
            Report.archived == False
        ).order_by(Report.timestamp.desc()).all()
        
        print(f"📊 Found {len(reports)} active reports")
        
        result = []
        for report in reports:
            # Handle photo paths safely
            photo_paths = []
            if report.photo_path:
                try:
                    if report.photo_path.startswith('['):
                        import json
                        photo_paths = json.loads(report.photo_path)
                    else:
                        photo_paths = [report.photo_path]
                except:
                    photo_paths = []
            
            # Get username safely
            username = None
            if report.user:
                username = report.user.username
            
            result.append({
                'id': report.id,
                'station': report.station,
                'direction': getattr(report, 'direction', 'both'),
                'reported_congestion': report.reported_congestion,
                'remarks': report.remarks,
                'timestamp': report.timestamp.isoformat(),
                'username': username,
                'anonymous': report.anonymous,
                'flagged': report.flagged,
                'flag_count': getattr(report, 'flag_count', 0),
                'archived': report.archived,
                'photo_paths': photo_paths,
                'photo_path': report.photo_path
            })
        
        return jsonify(result)
    except Exception as e:
        print(f"Error fetching reports: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([]), 500
    
@operator_bp.route('/api/broadcasts/active', methods=['GET'])
def get_active_broadcasts():
    """Get only active broadcasts for user dashboard"""
    try:
        now = datetime.now()
        
        # Auto-expire old broadcasts
        expired_by_date = Broadcast.query.filter(
            Broadcast.is_active == True,
            Broadcast.expires_at != None,
            Broadcast.expires_at <= now
        ).all()
        
        for broadcast in expired_by_date:
            broadcast.is_active = False
        
        if expired_by_date:
            db.session.commit()
            print(f"Auto-expired {len(expired_by_date)} broadcasts")
        
        # Only return active broadcasts
        active_broadcasts = Broadcast.query.filter(
            Broadcast.is_active == True
        ).order_by(Broadcast.created_at.desc()).all()
        
        result = []
        for broadcast in active_broadcasts:
            stations = json.loads(broadcast.stations) if broadcast.stations else []
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
                'is_active': broadcast.is_active,
                'time': broadcast.created_at.isoformat()
            })
        
        return jsonify({'success': True, 'broadcasts': result})
    except Exception as e:
        print(f"Error getting active broadcasts: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@operator_bp.route('/api/operator/station-status')
def operator_station_status():
    """Get station status for operator dashboard - SHOWS ALL STATIONS with overrides"""
    try:
        from flask import current_app
        from datetime import datetime
        from services.model_loader import directional_models, directional_scalers
        from services import get_feature_sequence_for_station
        import time
        
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # ========== GET ACTIVE OVERRIDES FROM FILE FIRST ==========
        active_overrides = get_active_overrides()
        
        print(f"🔍 Active overrides: {len(active_overrides)}")
        for key, val in active_overrides.items():
            print(f"   {key}: {val.get('congestion', '?')}%")
        
        # Also update app config for consistency
        if 'overrides' not in current_app.config:
            current_app.config['overrides'] = {}
        current_app.config['overrides'] = active_overrides
        
        stations_to_show = STATIONS
        
        # Get models from config
        directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
        directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
        
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        current_time = hour + minute / 60
        
        OPERATING_START = 4.5
        OPERATING_END = 22.5
        
        def get_congestion(station_name, direction):
            """Get congestion with OVERRIDE support - returns (congestion, is_overridden)"""
            override_key = f"{station_name}_{direction.lower()}"
            
            # CHECK OVERRIDE FIRST - THIS IS THE KEY FIX
            if override_key in active_overrides:
                override_value = active_overrides[override_key].get('congestion', 50)
                print(f"🔧 OVERRIDE: {station_name} {direction} = {override_value}%")
                return override_value, True
            
            # If MRT is closed and no override, return 0
            if current_time < OPERATING_START or current_time >= OPERATING_END:
                return 0, False
            
            # Otherwise use model prediction
            model_key = f"{station_name}_{direction}"
            
            if model_key not in directional_models:
                # Time-based fallback
                if 7 <= hour <= 9:
                    return 65 + (hour - 7) * 5, False
                elif 17 <= hour <= 19:
                    return 60 + (hour - 17) * 5, False
                elif 10 <= hour <= 16:
                    return 40, False
                return 25, False
            
            try:
                sequence = get_feature_sequence_for_station(station_name, direction, now)
                if sequence is not None and len(sequence) == 24:
                    feature_scaler = directional_scalers.get(f'{model_key}_feature')
                    target_scaler = directional_scalers.get(f'{model_key}_target')
                    
                    if feature_scaler and target_scaler:
                        input_sequence = sequence.reshape(1, 24, -1)
                        pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
                        congestion, _ = _get_congestion_from_prediction(
                            pred_scaled, target_scaler, station_name
                        )
                        return congestion, False
            except Exception as e:
                print(f"⚠️ Error for {model_key}: {e}")
            
            # Fallback
            if 7 <= hour <= 9:
                return 65 + (hour - 7) * 5, False
            elif 17 <= hour <= 19:
                return 60 + (hour - 17) * 5, False
            elif 10 <= hour <= 16:
                return 40, False
            return 25, False
        
        def get_status_info(congestion):
            if congestion > 80:
                return "SEVERE", "status-severe", "15-20 min"
            elif congestion > 60:
                return "CONGESTED", "status-congested", "10-15 min"
            elif congestion > 30:
                return "MODERATE", "status-moderate", "5-10 min"
            else:
                return "LIGHT", "status-light", "2-5 min"
        
        result = []
        for station in stations_to_show:
            north_congestion, north_overridden = get_congestion(station, 'Northbound')
            south_congestion, south_overridden = get_congestion(station, 'Southbound')
            
            print("Operator status north: ", north_congestion)
            print("Operator status south: ", south_congestion)
            
            
            # If overridden, show the override value even if MRT is closed
            # But if not overridden and MRT is closed, show CLOSED
            if not north_overridden and (current_time < OPERATING_START or current_time >= OPERATING_END):
                north_text = "CLOSED"
                north_class = "status-light"
                north_wait = "Closed"
            else:
                north_text, north_class, north_wait = get_status_info(north_congestion)
            
            if not south_overridden and (current_time < OPERATING_START or current_time >= OPERATING_END):
                south_text = "CLOSED"
                south_class = "status-light"
                south_wait = "Closed"
            else:
                south_text, south_class, south_wait = get_status_info(south_congestion)
            
            platform_capacity = MRT3_PLATFORM_CAPACITY.get(station, 1000)
            
            result.append({
                'name': station,
                'northbound': {
                    'congestion': round(north_congestion, 1),
                    'status_text': north_text,
                    'status_class': north_class,
                    'wait_time': north_wait,
                    'ridership': int((north_congestion / 100) * platform_capacity) if north_congestion > 0 else 0,
                    'overridden': north_overridden
                },
                'southbound': {
                    'congestion': round(south_congestion, 1),
                    'status_text': south_text,
                    'status_class': south_class,
                    'wait_time': south_wait,
                    'ridership': int((south_congestion / 100) * platform_capacity) if south_congestion > 0 else 0,
                    'overridden': south_overridden
                }
            })
        
        return jsonify({'stations': result})
        
    except Exception as e:
        print(f"Error in operator_station_status: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'stations': []})
    
@operator_bp.route('/api/operator/debug-override')
def debug_override():
    """Debug override status"""
    active_overrides = get_active_overrides()
    
    return jsonify({
        'active_overrides': active_overrides,
        'taft_override': active_overrides.get('Taft_southbound'),
        'current_timestamp': time.time(),
        'overrides_file_exists': os.path.exists(OVERRIDES_FILE)
    })
    
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

@operator_bp.route('/api/operator/broadcasts', methods=['GET'])
def get_operator_broadcasts():
    """Get ALL broadcasts for operator's stations - no auto-deletion"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        now = datetime.now()
        
        # ✅ ONLY expire broadcasts by their set expiration time
        # DO NOT auto-expire based on age
        expired_by_date = Broadcast.query.filter(
            Broadcast.is_active == True,
            Broadcast.expires_at != None,
            Broadcast.expires_at <= now
        ).all()
        
        if expired_by_date:
            for broadcast in expired_by_date:
                broadcast.is_active = False
            db.session.commit()
            print(f"Auto-expired {len(expired_by_date)} broadcasts by expiry date")
        
        managed_stations = get_operator_stations(user_id)
        
        # ✅ Get ALL broadcasts - NO time limit, NO auto-deletion
        all_broadcasts = Broadcast.query.order_by(
            Broadcast.created_at.desc()
        ).all()
        
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
                    'is_active': broadcast.is_active,  # CRITICAL: Include this field
                    'icon': typeIcons.get(broadcast.disruption_type, 'fa-bullhorn')
                })
        
        return jsonify({'success': True, 'broadcasts': result})
    except Exception as e:
        print(f"Error getting broadcasts: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@operator_bp.route('/profile')
@operator_bp.route('/profile.html')
def operator_profile():
    """Operator profile page"""
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('auth.login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('auth.login'))
    
    managed_stations = get_operator_stations(user.id)
    
    return render_template('profile.html',
                         user=user,
                         managed_stations=managed_stations,
                         all_stations=STATIONS)
    
@operator_bp.route('/api/operator/broadcast/<int:broadcast_id>', methods=['PUT'])
def update_broadcast(broadcast_id):
    """Update a broadcast - handles both editing and archiving"""
    try:
        data = request.json
        broadcast = Broadcast.query.get(broadcast_id)
        
        if not broadcast:
            return jsonify({'success': False, 'error': 'Broadcast not found'}), 404
        
        # Check if this is an archive request
        if 'is_active' in data:
            # Archive/restore broadcast
            broadcast.is_active = data['is_active']
            if data['is_active'] == False:
                broadcast.archived_at = datetime.now()
                broadcast.archived_by = session.get('username', 'operator')
            else:
                # Restoring - clear archive info
                broadcast.archived_at = None
                broadcast.archived_by = None
        else:
            # Update fields for editing
            if 'title' in data:
                broadcast.title = data['title']
            if 'message' in data:
                broadcast.message = data['message']
            if 'severity' in data:
                broadcast.severity = data['severity']
        
        db.session.commit()
        
        log_activity(session.get('user_id'), 'operator', session.get('username'), 
                    'edit_broadcast', f'Updated broadcast #{broadcast_id}: {broadcast.title}')
        
        return jsonify({'success': True, 'message': 'Broadcast updated'})
    except Exception as e:
        db.session.rollback()
        print(f"Error updating broadcast: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    
@operator_bp.route('/api/operator/broadcast/<int:broadcast_id>', methods=['DELETE'])
def delete_broadcast(broadcast_id):
    """Archive a broadcast (soft delete)"""
    try:
        broadcast = Broadcast.query.get(broadcast_id)
        
        if not broadcast:
            return jsonify({'success': False, 'error': 'Broadcast not found'}), 404
        
        # Archive instead of hard delete
        broadcast.is_active = False
        broadcast.archived_at = datetime.now()
        broadcast.archived_by = session.get('username', 'operator')
        
        db.session.commit()
        
        log_activity(session.get('user_id'), 'operator', session.get('username'), 
                    'archive_broadcast', f'Archived broadcast #{broadcast_id}: {broadcast.title}')
        
        return jsonify({'success': True, 'message': 'Broadcast archived'})
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
        
        print(f"🔧 SETTING OVERRIDE:")
        print(f"   Station: {station}")
        print(f"   Direction: {direction}")
        print(f"   Level: {level}")
        print(f"   Congestion: {congestion_value}")
        print(f"   Duration: {duration}")
        print(f"   Override Key: {override_key}")
        
        # Load existing overrides from file
        overrides = load_overrides()
        
        expiry = None
        if duration != 'manual':
            expiry = time.time() + (int(duration) * 60)
            print(f"   Expiry: {expiry} ({duration} minutes)")
        else:
            print(f"   Expiry: Manual (no expiration)")
        
        overrides[override_key] = {
            'station': station, 
            'direction': direction, 
            'level': level,
            'congestion': congestion_value, 
            'operator': operator_email,
            'reason': reason, 
            'expiry': expiry, 
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        save_overrides(overrides)
        
        # Also update app config for immediate use
        if 'overrides' not in current_app.config:
            current_app.config['overrides'] = {}
        current_app.config['overrides'][override_key] = overrides[override_key]
        
        print(f"✅ Override saved to file: {override_key} = {congestion_value}%")
        
        log_activity(operator_id, 'operator', operator_email, 'override_congestion',
                    f'Overrode {station} ({direction}) to {level} ({congestion_value}%)')
        
        return jsonify({'success': True, 'message': f'{station} ({direction}) set to {level}'})
    except Exception as e:
        print(f"❌ Error setting override: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@operator_bp.route('/api/operator/review-report/<int:report_id>', methods=['POST'])
def review_report(report_id):
    """
    Review a flagged report - handles True Positive / False Positive actions
    Called by markAsTruePositive() and markAsFalsePositive() in frontend
    """
    try:
        data = request.json
        verdict = data.get('verdict')  # 'true_positive' or 'false_positive'
        
        user_id = session.get('user_id')
        user = User.query.get(user_id)
        report = Report.query.get(report_id)
        
        if not report:
            return jsonify({'success': False, 'error': 'Report not found'}), 404
        
        # Check permission
        managed_stations = get_operator_stations(user_id)
        if report.station not in managed_stations and user.role != 'admin':
            return jsonify({'success': False, 'error': 'You cannot review reports from this station'}), 403
        
        if verdict == 'true_positive':
            # True Positive: Report was correctly flagged, keep it and clear flags
            report.flagged = False
            report.flag_count = 0
            report.reviewed = True
            report.reviewed_at = datetime.now()
            report.reviewed_by = user.username
            message = 'Report kept as True Positive, flags cleared'
            
        elif verdict == 'false_positive':
            # False Positive: Report was incorrectly flagged, archive it
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
        print(f"Error reviewing report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@operator_bp.route('/api/operator/keep-report/<int:report_id>', methods=['POST'])
def keep_report(report_id):
    """
    Keep a report and remove flags (alternative to true_positive)
    Called by keepAndRemoveFlag() in frontend
    """
    try:
        user_id = session.get('user_id')
        user = User.query.get(user_id)
        report = Report.query.get(report_id)
        
        if not report:
            return jsonify({'success': False, 'error': 'Report not found'}), 404
        
        # Check permission
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
        print(f"Error keeping report: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@operator_bp.route('/api/operator/archive-report/<int:report_id>', methods=['POST'])
def archive_report(report_id):
    """Archive a report (soft delete)"""
    try:
        data = request.json
        user_id = session.get('user_id')
        user = User.query.get(user_id)
        report = Report.query.get(report_id)
        
        if not report:
            return jsonify({'success': False, 'error': 'Report not found'}), 404
        
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        # Check permission
        managed_stations = get_operator_stations(user_id)
        if report.station not in managed_stations and user.role != 'admin':
            return jsonify({'success': False, 'error': 'You cannot archive reports from this station'}), 403
        
        # Archive the report - NOW THESE COLUMNS EXIST!
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
        print(f"Error archiving report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    
    
@operator_bp.route('/api/operator/broadcast/<int:broadcast_id>', methods=['GET'])
def get_broadcast(broadcast_id):
    """Get a single broadcast for editing"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        broadcast = Broadcast.query.get(broadcast_id)
        
        if not broadcast:
            return jsonify({'success': False, 'error': 'Broadcast not found'}), 404
        
        # Check if user has permission (broadcast affects their stations)
        managed_stations = get_operator_stations(user_id)
        stations = json.loads(broadcast.stations) if broadcast.stations else []
        
        if not any(s in managed_stations for s in stations):
            # Check if user is admin or has line_wide access
            user = User.query.get(user_id)
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
                'expires_at': broadcast.expires_at.isoformat() if broadcast.expires_at else None
            }
        })
    except Exception as e:
        print(f"Error getting broadcast: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    
    
@operator_bp.route('/api/operator/clear-override', methods=['POST'])
def clear_override():
    try:
        data = request.json or {}
        station = data.get('station')
        direction = data.get('direction')
        
        if not station or not direction:
            return jsonify({'success': False, 'error': 'Station and direction are required'}), 400
        
        target_key = f"{station}_{direction.lower()}"
        
        print(f"🔧 CLEARING OVERRIDE: {target_key}")
        
        # Load from file
        overrides = load_overrides()
        
        if target_key not in overrides:
            return jsonify({'success': False, 'error': f'No active override found for {station} ({direction})'}), 404
        
        # Remove from file
        del overrides[target_key]
        save_overrides(overrides)
        
        # Remove from app config
        if 'overrides' in current_app.config and target_key in current_app.config['overrides']:
            del current_app.config['overrides'][target_key]
        
        print(f"✅ Override cleared: {target_key}")
        
        log_activity(
            session.get('user_id'), 'operator', session.get('username'),
            'clear_override', f'Cleared override for {station} ({direction})'
        )
        
        return jsonify({
            'success': True, 
            'message': f'Override cleared successfully for {station} ({direction})'
        })
        
    except Exception as e:
        print(f"Error clearing override: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    

@operator_bp.route('/api/operator/get-overrides', methods=['GET'])
def get_overrides():
    """Get active overrides from file"""
    active_overrides = get_active_overrides()
    
    print(f"📋 Active overrides: {list(active_overrides.keys())}")
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