from flask import Blueprint, request, jsonify, session
from models import Report, User, db
from datetime import datetime, timedelta
import json, os, re, time
from collections import defaultdict

api_reports_bp = Blueprint('api_reports', __name__)

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

STATION_BASE_CAPACITY = {
    "North Ave": 12000, "Quezon Ave": 9000, "Kamuning": 7500, "Cubao": 15000,
    "Santolan": 8000, "Ortigas": 9500, "Shaw Blvd": 11000, "Boni Ave": 8500,
    "Guadalupe": 10000, "Buendia": 9000, "Ayala Ave": 14000, "Magallanes": 9000, "Taft": 16000
}

report_tracker = defaultdict(list)

def track_report_submission(user_id, ip_address, station):
    key = f"{user_id if user_id else ip_address}_{station}"
    timestamp = time.time()
    report_tracker[key].append(timestamp)
    current_time = time.time()
    report_tracker[key] = [t for t in report_tracker[key] if current_time - t < 3600]

def is_rate_limited(user_id, ip_address, limit=3, window=3600):
    key = user_id if user_id else ip_address
    current_time = time.time()
    recent_reports = [t for t in report_tracker.get(key, []) if current_time - t < window]
    return len(recent_reports) >= limit

def is_suspicious_remarks(remarks):
    if not remarks:
        return False
    if re.search(r'(.)\1{10,}', remarks):
        return True
    if len(set(remarks.lower())) == 1 and len(remarks) > 5:
        return True
    spam_patterns = [r'^[a-zA-Z]$', r'^[0-9]+$', r'^(.)\1+$']
    for pattern in spam_patterns:
        if re.match(pattern, remarks):
            return True
    return False

def check_duplicate_report(station, congestion_value, user_id, minutes=10):
    if not user_id:
        return False
    time_threshold = datetime.now() - timedelta(minutes=minutes)
    min_congestion = congestion_value - 15
    max_congestion = congestion_value + 15
    duplicate = Report.query.filter(
        Report.station == station, Report.user_id == user_id,
        Report.timestamp > time_threshold,
        Report.reported_congestion.between(min_congestion, max_congestion)
    ).first()
    return duplicate is not None

def get_station_prediction(station_name):
    """Get prediction - will be set in app.py"""
    from flask import current_app
    if hasattr(current_app, 'config') and 'GET_STATION_PREDICTION' in current_app.config:
        return current_app.config['GET_STATION_PREDICTION'](station_name)
    capacity = STATION_BASE_CAPACITY.get(station_name, 10000)
    return capacity * 0.5

def log_activity(user_id, user_type, user_email, action, details=None):
    """Log activity - will be set in app.py"""
    from flask import current_app
    if hasattr(current_app, 'config') and 'LOG_ACTIVITY' in current_app.config:
        current_app.config['LOG_ACTIVITY'](user_id, user_type, user_email, action, details)

@api_reports_bp.route('/reports')
def get_reports():
    try:
        reports = Report.query.order_by(Report.timestamp.desc()).limit(50).all()
        
        result = []
        for report in reports:
            username = "Commuter"
            photo_paths = None
            if report.photo_path:
                try:
                    photo_paths = json.loads(report.photo_path)
                except:
                    photo_paths = [report.photo_path]
            
            result.append({
                'id': report.id, 'station': report.station, 'direction': report.direction,
                'reported_congestion': report.reported_congestion,
                'predicted_congestion': report.predicted_congestion,
                'remarks': report.remarks, 'anonymous': report.anonymous,
                'username': username, 'timestamp': report.timestamp.isoformat() if report.timestamp else None,
                'photo_path': report.photo_path, 'photo_paths': photo_paths
            })
        
        return jsonify(result)
    except Exception as e:
        return jsonify([])

def is_operating_hours(check_time=None):
    """
    Check if current time is within MRT operating hours
    Operating hours: 4:30 AM to 10:30 PM
    Returns: bool (True if open, False if closed)
    """
    if check_time is None:
        check_time = datetime.now()
    
    hour = check_time.hour
    minute = check_time.minute
    current_time = hour + minute / 60
    
    OPERATING_START = 4.5   # 4:30 AM
    OPERATING_END = 22.5    # 10:30 PM
    
    return OPERATING_START <= current_time < OPERATING_END

def get_next_opening_time():
    """Get the next opening time as a string"""
    now = datetime.now()
    today_open = now.replace(hour=4, minute=30, second=0, microsecond=0)
    
    if now < today_open:
        return today_open.strftime('%I:%M %p')
    else:
        # Next day opening
        next_day_open = now.replace(day=now.day + 1, hour=4, minute=30, second=0, microsecond=0)
        return next_day_open.strftime('%I:%M %p, %B %d')
    
@api_reports_bp.route('/report-congestion', methods=['POST'])
def report_congestion():
    try:
        user_id = session.get('user_id')
        ip_address = request.remote_addr
        
        if not is_operating_hours():
            next_open = get_next_opening_time()
            return jsonify({
                "success": False, 
                "error": f"MRT-3 is currently closed. Operating hours are 4:30 AM - 10:30 PM. Reports can only be submitted during operating hours. Next opening: {next_open}"
            }), 403
        
        if is_rate_limited(user_id, ip_address, limit=3, window=3600):
            return jsonify({"success": False, "error": "You've reached the limit of 3 reports per hour."}), 429
        
        
        station = None
        direction = None
        reported = None
        remarks = ""
        anonymous = False
        photo_paths = []
        
        if request.content_type and 'multipart/form-data' in request.content_type:
            station = request.form.get('station')
            direction = request.form.get('direction')
            reported = request.form.get('congestion')
            remarks = request.form.get('remarks', '')
            anonymous = request.form.get('anonymous', 'false').lower() == 'true'
            
            if 'images' in request.files:
                files = request.files.getlist('images')
                if len(files) > 5:
                    return jsonify({"success": False, "error": "Maximum 5 photos allowed"}), 400
                
                for file in files:
                    if file and file.filename:
                        file.seek(0, 2)
                        size = file.tell()
                        file.seek(0)
                        if size > 10 * 1024 * 1024:
                            return jsonify({"success": False, "error": "Image too large (max 10MB)"}), 400
                        
                        safe_filename = file.filename.replace(' ', '_').replace('%', '')
                        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_filename}"
                        upload_folder = os.path.join('static', 'uploads', 'reports')
                        os.makedirs(upload_folder, exist_ok=True)
                        filepath = os.path.join(upload_folder, filename)
                        file.save(filepath)
                        photo_paths.append(f"/uploads/reports/{filename}")
        else:
            data = request.json
            station = data.get('station')
            direction = data.get('direction')
            reported = data.get('congestion')
            remarks = data.get('remarks', '')
            anonymous = data.get('anonymous', False)
        
        if not station:
            return jsonify({"success": False, "error": "Station is required"}), 400
        if reported is None:
            return jsonify({"success": False, "error": "Congestion level is required"}), 400
        
        if is_suspicious_remarks(remarks):
            return jsonify({"success": False, "error": "Invalid remarks detected."}), 400
        
        try:
            reported = int(reported)
        except:
            return jsonify({"success": False, "error": "Congestion must be a number"}), 400
        
        if station not in STATIONS:
            return jsonify({"success": False, "error": "Invalid station"}), 400
        
        if not (0 <= reported <= 100):
            return jsonify({"success": False, "error": "Congestion must be between 0 and 100"}), 400
        
        if check_duplicate_report(station, reported, user_id, minutes=10):
            return jsonify({"success": False, "error": "You already submitted a similar report recently."}), 429
        
        ridership = get_station_prediction(station)
        capacity = STATION_BASE_CAPACITY.get(station, 10000)
        predicted = int((ridership / capacity) * 100)
        
        photo_path_json = json.dumps(photo_paths) if photo_paths else None
        
        report = Report(
            user_id=user_id, station=station, direction=direction,
            reported_congestion=reported, predicted_congestion=predicted,
            remarks=remarks[:500] if remarks else None,
            photo_path=photo_path_json, anonymous=anonymous
        )
        
        db.session.add(report)
        db.session.commit()
        track_report_submission(user_id, ip_address, station)
        
        return jsonify({"success": True, "message": "Report submitted successfully!",
                       "photos": len(photo_paths), "direction": direction, "report_id": report.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500

@api_reports_bp.route('/reports/<int:report_id>/flag', methods=['POST'])
def flag_report(report_id):
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'Please log in to flag reports'}), 401
        
        report = Report.query.get(report_id)
        if not report:
            return jsonify({'success': False, 'error': 'Report not found'}), 404
        
        report.flag_count = (report.flag_count or 0) + 1
        
        if report.flag_count >= 3:
            report.flagged = True
        
        db.session.commit()
        
        log_activity(user_id, session.get('role', 'user'), session.get('username', 'unknown'),
                    'flag_report', f'Flagged report #{report_id} from {report.station}')
        
        return jsonify({'success': True, 'message': 'Report flagged for review'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500