from flask import Blueprint, session, request, jsonify, flash, redirect, url_for, render_template, current_app
from models import User, Report, Broadcast, db
from datetime import datetime, timedelta
import json, time, math, os
from .auth import login_required, log_activity
from flask_caching import Cache
from extensions import cache
from config import Config

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
    print(f"📝 SAVING OVERRIDES: {overrides}")
    try:
        with open(OVERRIDES_FILE, 'w') as f:
            json.dump(overrides, f, indent=2)
        print(f"✅ Saved {len(overrides)} overrides to file")
    except Exception as e:
        print(f"Error saving overrides: {e}")

# ========== SINGLE get_active_overrides FUNCTION ==========
def get_active_overrides():
    """Get active overrides from file with expiry check"""
    overrides = load_overrides()
    
    # Use datetime.now() for consistency
    now = datetime.now()
    now_timestamp = now.timestamp()
    
    # Filter out expired overrides
    active_overrides = {}
    for key, override in overrides.items():
        expiry = override.get('expiry')
        if expiry is None or expiry > now_timestamp:
            active_overrides[key] = override
        else:
            print(f"⏰ Override expired: {key}")
    
    print(f"📄 Active overrides: {active_overrides}")
    return active_overrides

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
    """Get reports for operator dashboard - Shows ALL active reports (no station filtering)"""
    try:
        # Get user info
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Debug: Check total reports count
        total_reports = Report.query.count()
        archived_count = Report.query.filter(Report.archived == True).count()
        active_count = Report.query.filter(Report.archived == False).count()
        
        print(f"📊 REPORT STATS: Total={total_reports}, Archived={archived_count}, Active={active_count}")
        
        # ✅ GET ALL ACTIVE REPORTS - NO STATION FILTERING
        # This shows ALL reports regardless of which station they're from
        reports = Report.query.filter(
            Report.archived == False
        ).order_by(Report.timestamp.desc()).all()
        
        print(f"📊 Found {len(reports)} active reports (showing all to operator)")
        
        result = []
        for report in reports:
            # Handle photo paths safely
            photo_paths = []
            if report.photo_path:
                try:
                    if report.photo_path.startswith('['):
                        photo_paths = json.loads(report.photo_path)
                    else:
                        photo_paths = [report.photo_path]
                except:
                    photo_paths = []
            
            # Get username safely
            username = None
            if report.user:
                username = report.user.username
            
            # Get status text from congestion
            congestion = report.reported_congestion
            if congestion >= 80:
                status_text = "Severe"
                status_class = "status-severe"
            elif congestion >= 60:
                status_text = "Congested"
                status_class = "status-congested"
            elif congestion >= 30:
                status_text = "Moderate"
                status_class = "status-moderate"
            else:
                status_text = "Light"
                status_class = "status-light"
            
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
                'status_class': status_class
            })
        
        # Log the count for debugging
        print(f"✅ Returning {len(result)} reports to operator dashboard")
        
        return jsonify(result)
    except Exception as e:
        print(f"❌ Error fetching reports: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([]), 500

@operator_bp.route('/api/broadcasts/public', methods=['GET'])
def get_public_broadcasts():
    """Get active broadcasts for public alerts page"""
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
        
        # Only return active broadcasts, ordered by newest first
        active_broadcasts = Broadcast.query.filter(
            Broadcast.is_active == True
        ).order_by(Broadcast.created_at.desc()).all()
        
        result = []
        for broadcast in active_broadcasts:
            stations = json.loads(broadcast.stations) if broadcast.stations else []
            
            # ========== FORMAT TIME ==========
            created_at = broadcast.created_at
            now = datetime.now()
            diff = now - created_at
            diff_seconds = diff.total_seconds()
            
            if diff_seconds < 60:
                time_display = 'Just now'
            elif diff_seconds < 3600:
                minutes = int(diff_seconds // 60)
                time_display = f'{minutes} min ago'
            elif diff_seconds < 86400:
                hours = int(diff_seconds // 3600)
                time_display = f'{hours}h ago'
            else:
                days = int(diff_seconds // 86400)
                time_display = f'{days}d ago'
            
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
                'time': time_display  # ← Human-readable time
            })
        
        return jsonify({'success': True, 'broadcasts': result})
    except Exception as e:
        print(f"Error getting public broadcasts: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
# Add this helper function at the top of your operator.py file
def get_live_map_data_direct():
    """Get live map data directly without HTTP calls"""
    try:
        from flask import current_app
        
        # Try to import the live_map module
        try:
            from .live_map import get_directional_data
            return get_directional_data()
        except ImportError:
            pass
        
        # Try using the cache
        cache = current_app.extensions.get('cache')
        if cache:
            cached_data = cache.get('live_map_data')
            if cached_data:
                return cached_data
        
        # Fallback: generate data
        data = {'northbound': {}, 'southbound': {}}
        for station in STATIONS:
            base = 20 + (hash(station) % 50)
            data['northbound'][station] = {
                'congestion': base,
                'status': _get_status_from_congestion(base),
                'wait_time': _get_wait_time(base),
                'ridership': base * 10
            }
            base2 = 25 + (hash(station + 'south') % 50)
            data['southbound'][station] = {
                'congestion': base2,
                'status': _get_status_from_congestion(base2),
                'wait_time': _get_wait_time(base2),
                'ridership': base2 * 10
            }
        return data
        
    except Exception as e:
        print(f"Error getting live map data: {e}")
        return None
@operator_bp.route('/api/operator/station-status')
def operator_station_status():
    """Get station status for operator dashboard - USING V2 PREDICTION API"""
    try:
        print("🔍 Starting operator_station_status (using V2 API)...")
        
        # ========== USE THE SAME V2 ENDPOINT AS LIVE MAP ==========
        from flask import current_app
        
        # Make an internal request to the V2 endpoint
        with current_app.test_client() as client:
            response = client.get('/api/live-map/directions/v2')
            data = response.get_json()
        
        if not data or 'northbound' not in data or 'southbound' not in data:
            print("⚠️ V2 API returned no data, using fallback")
            return jsonify({'stations': _generate_fallback_stations(), 'fallback': True})
        
        northbound_data = data.get('northbound', {})
        southbound_data = data.get('southbound', {})
        
        # ========== GET ACTIVE OVERRIDES ==========
        active_overrides = get_active_overrides()
        
        result = []
        for station in STATIONS:
            north = northbound_data.get(station, {})
            south = southbound_data.get(station, {})
            
            # Check overrides
            north_key = f"{station}_northbound"
            south_key = f"{station}_southbound"
            
            is_north_overridden = north_key.lower() in {k.lower() for k in active_overrides.keys()}
            is_south_overridden = south_key.lower() in {k.lower() for k in active_overrides.keys()}
            
            result.append({
                'name': station,
                'northbound': {
                    'congestion': north.get('congestion', 0),
                    'status': north.get('status', _get_status_from_congestion(north.get('congestion', 0))),
                    'wait_time': north.get('wait_time', _get_wait_time(north.get('congestion', 0))),
                    'ridership': north.get('ridership', 0),
                    'overridden': is_north_overridden
                },
                'southbound': {
                    'congestion': south.get('congestion', 0),
                    'status': south.get('status', _get_status_from_congestion(south.get('congestion', 0))),
                    'wait_time': south.get('wait_time', _get_wait_time(south.get('congestion', 0))),
                    'ridership': south.get('ridership', 0),
                    'overridden': is_south_overridden
                }
            })
        
        print(f"✅ Returning {len(result)} stations from V2 API")
        return jsonify({'stations': result})
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'stations': _generate_fallback_stations(), 'error': str(e)})
    
@operator_bp.route('/api/operator/forecast/<station_name>')
def operator_forecast(station_name):
    """Get 6-hour forecast for a station - USES SAME PREDICTIONS AS LIVE MAP"""
    try:
        from flask import current_app
        
        station = station_name.replace('%20', ' ')
        now = datetime.now()
        
        # Get current time rounded to the hour
        base_time = now.replace(minute=0, second=0, microsecond=0)
        
        forecasts = []
        
        for i in range(7):  # 0-6 hours ahead
            target_time = base_time + timedelta(hours=i)
            
            # Use the same prediction function as the live map
            try:
                # Import from the prediction module
                from routes.api_predict import get_directional_prediction
                
                north_cong = get_directional_prediction(station, 'Northbound', target_time)
                south_cong = get_directional_prediction(station, 'Southbound', target_time)
            except ImportError:
                # Fallback: use the V2 endpoint
                with current_app.test_client() as client:
                    response = client.get(f'/api/live-map/directions/v2?date={target_time.strftime("%Y-%m-%d")}&time={target_time.strftime("%H:%M")}')
                    data = response.get_json()
                    
                    if data and 'northbound' in data and 'southbound' in data:
                        north_data = data['northbound'].get(station, {})
                        south_data = data['southbound'].get(station, {})
                        north_cong = north_data.get('congestion', 0)
                        south_cong = south_data.get('congestion', 0)
                    else:
                        north_cong = 0
                        south_cong = 0
            
            # Handle None values
            north_cong = north_cong if north_cong is not None else 0
            south_cong = south_cong if south_cong is not None else 0
            
            avg_cong = (north_cong + south_cong) / 2
            
            # Get status
            if avg_cong > 80:
                status = "SEVERE"
                color = "critical"
            elif avg_cong > 50:
                status = "CONGESTED"
                color = "congested"
            elif avg_cong > 25:
                status = "MODERATE"
                color = "moderate"
            else:
                status = "LIGHT"
                color = "light"
            
            # Format time display
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
                'northbound': round(north_cong, 1),
                'southbound': round(south_cong, 1),
                'average': round(avg_cong, 1),
                'status': status,
                'color': color
            })
        
        return jsonify({
            'station': station,
            'timestamp': now.isoformat(),
            'forecasts': forecasts
        })
        
    except Exception as e:
        print(f"❌ Error in operator_forecast: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
    
def _generate_fallback_stations():
    """Generate fallback station data for when the main data source fails"""
    fallback = []
    for station in STATIONS:
        fallback.append({
            'name': station,
            'northbound': {'congestion': 25, 'status': 'MODERATE', 'wait_time': '5-10 min', 'ridership': 500, 'overridden': False},
            'southbound': {'congestion': 25, 'status': 'MODERATE', 'wait_time': '5-10 min', 'ridership': 550, 'overridden': False}
        })
    return fallback
def _get_status_from_congestion(congestion):
    """Helper function to get status text from congestion"""
    if congestion > 80:
        return 'SEVERE'
    elif congestion > 50:
        return 'CONGESTED'
    elif congestion > 25:
        return 'MODERATE'
    else:
        return 'LIGHT'

def _get_wait_time(congestion):
    """Helper function to get wait time from congestion"""
    if congestion > 80:
        return '15-20 min'
    elif congestion > 50:
        return '10-15 min'
    elif congestion > 25:
        return '5-10 min'
    else:
        return '2-5 min'

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
        
        # ✅ FIX: Use ACTUAL system time, not Config time
        current_time = datetime.now()
        current_timestamp = current_time.timestamp()
        
        expiry = None
        duration_minutes = 0
        if duration != 'manual':
            duration_minutes = int(duration)
            expiry = current_timestamp + (duration_minutes * 60)
            print(f"   Current time (system): {current_time}")
            print(f"   Expiry: {expiry} ({duration_minutes} minutes from now)")
        else:
            # For manual overrides, round to the start of the hour
            current_time = current_time.replace(minute=0, second=0, microsecond=0)
            current_timestamp = current_time.timestamp()
            print(f"   Manual override - rounded to hour: {current_time}")
        
        # Get operator name
        operator_name = session.get('username', 'operator')
        
        overrides[override_key] = {
            'station': station, 
            'direction': direction, 
            'level': level,
            'congestion': congestion_value, 
            'operator': operator_name,
            'reason': reason, 
            'expiry': expiry,
            'timestamp': current_time.isoformat(),
            'duration_minutes': duration_minutes,
            'created_at': current_time.isoformat()
        }
        
        # Save to file
        save_overrides(overrides)
        
        # ========== UPDATE APP CONFIG ==========
        if 'overrides' not in current_app.config:
            current_app.config['overrides'] = {}
        current_app.config['overrides'][override_key] = overrides[override_key]
        print(f"✅ Updated app config with override: {override_key}")
        
     
        try:
            # Get the actual cache instance from the app extensions
            cache_instance = current_app.extensions.get('cache')
            
            if cache_instance:
                print(f"🗑️ Clearing cache for {station}...")
                
                # Delete ALL possible cache keys for this station
                for hour in range(24):
                    # Try with standard key
                    cache_instance.delete(f"forecast_{station}_{hour}")
                    
                    # Try with view prefix that Flask-Caching might add
                    cache_instance.delete(f"view//forecast_{station}_{hour}")
                    
                    # Try with prefix that Flask-Caching might use
                    cache_instance.delete(f"view/forecast_{station}_{hour}")
                
                # Also clear the "all stations" cache
                for hour in range(24):
                    cache_instance.delete(f"all_stations_{hour}")
                    cache_instance.delete(f"view//all_stations_{hour}")
                    cache_instance.delete(f"view/all_stations_{hour}")
                
                # Clear the specific forecast for current hour
                cache_instance.delete(f"forecast_{station}_{current_time.hour}")
                cache_instance.delete(f"view//forecast_{station}_{current_time.hour}")
                cache_instance.delete(f"view/forecast_{station}_{current_time.hour}")
                
                print(f"✅ Cleared all forecast caches for {station}")
            else:
                print("⚠️ Cache instance not found in app extensions")
                
        except Exception as cache_error:
            print(f"⚠️ Could not clear cache: {cache_error}")
            import traceback
            traceback.print_exc()
        
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
        
        # ========== 1. LOAD AND REMOVE FROM FILE ==========
        overrides = load_overrides()
        
        if target_key not in overrides:
            print(f"⚠️ No override found for {target_key} in file")
        else:
            del overrides[target_key]
            save_overrides(overrides)
            print(f"✅ Removed from file: {target_key}")
        
        # ========== 2. REMOVE FROM APP CONFIG ==========
        if 'overrides' in current_app.config:
            if target_key in current_app.config['overrides']:
                del current_app.config['overrides'][target_key]
                print(f"✅ Removed from app config: {target_key}")
            # ALSO check for lowercase version
            lower_key = target_key.lower()
            if lower_key in current_app.config['overrides']:
                del current_app.config['overrides'][lower_key]
                print(f"✅ Removed from app config: {lower_key}")
        
        # ========== 3. CLEAR CACHE ==========
        try:
            cache_instance = current_app.extensions.get('cache')
            
            if cache_instance:
                print(f"🗑️ Clearing cache for {station}...")
                
                # Clear all cache for this station
                keys_deleted = 0
                for hour in range(24):
                    key_variations = [
                        f"forecast_{station}_{hour}",
                        f"forecast_{station.lower()}_{hour}",
                        f"view//forecast_{station}_{hour}",
                        f"view/forecast_{station}_{hour}",
                        f"forecast_{station}_{hour}_northbound",
                        f"forecast_{station}_{hour}_southbound",
                        f"forecast_{station}_{hour}_both",
                        f"station_{station}_{hour}",
                        f"live_map_{station}_{hour}",
                        f"api/live-map/directions/v2_{station}_{hour}",
                    ]
                    
                    for key in key_variations:
                        try:
                            cache_instance.delete(key)
                            keys_deleted += 1
                        except:
                            pass
                
                # Clear "all stations" caches
                for hour in range(24):
                    all_keys = [
                        f"all_stations_{hour}",
                        f"view//all_stations_{hour}",
                        f"view/all_stations_{hour}",
                        f"live_map_all_{hour}",
                    ]
                    for key in all_keys:
                        try:
                            cache_instance.delete(key)
                            keys_deleted += 1
                        except:
                            pass
                
                print(f"✅ Cleared {keys_deleted} cache keys for {station}")
            else:
                print("⚠️ Cache instance not found")
                
        except Exception as cache_error:
            print(f"⚠️ Could not clear cache: {cache_error}")
        
        # ========== 4. FORCE RELOAD ==========
        active = get_active_overrides()
        print(f"📄 Active overrides after clear: {active}")
        
        # ========== 5. LOG ==========
        log_activity(
            session.get('user_id'), 'operator', session.get('username'),
            'clear_override', f'Cleared override for {station} ({direction})'
        )
        
        return jsonify({
            'success': True, 
            'message': f'Override cleared successfully for {station} ({direction})',
            'active_overrides': active
        })
        
    except Exception as e:
        print(f"❌ Error clearing override: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

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