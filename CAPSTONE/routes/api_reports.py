from flask import Blueprint, request, jsonify, session
from models import Report, User, db
from datetime import datetime, timedelta
import json, os, re, time
from collections import defaultdict
from config import Config

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
    # Use Config.get_current_time() for consistent year
    time_threshold = Config.get_current_time() - timedelta(minutes=minutes)
    min_congestion = congestion_value - 15
    max_congestion = congestion_value + 15
    duplicate = Report.query.filter(
        Report.station == station, Report.user_id == user_id,
        Report.timestamp > time_threshold,
        Report.reported_congestion.between(min_congestion, max_congestion)
    ).first()
    return duplicate is not None

def get_station_prediction(station_name, direction=None, models_cached=None, scalers_cached=None, feature_sequence_func=None):
    """Get prediction - safely routes between initialization states"""
    from flask import current_app
    
    # 1. If the wrapper already gave us the models, skip the config forwarder!
    if models_cached is not None and feature_sequence_func is not None:
        # (Put your actual LSTM processing logic here if it's written in this file)
        capacity = STATION_BASE_CAPACITY.get(station_name, 10000)
        return capacity * 0.5

    # 2. If we don't have the models yet, call the wrapper in app.py with ONLY 1 argument
    if hasattr(current_app, 'config') and 'GET_STATION_PREDICTION' in current_app.config:
        return current_app.config['GET_STATION_PREDICTION'](station_name)
        
    # 3. Safe fallback
    capacity = STATION_BASE_CAPACITY.get(station_name, 10000)
    return capacity * 0.5

def log_activity(user_id, user_type, user_email, action, details=None):
    """Log activity - will be set in app.py"""
    from flask import current_app
    if hasattr(current_app, 'config') and 'LOG_ACTIVITY' in current_app.config:
        current_app.config['LOG_ACTIVITY'](user_id, user_type, user_email, action, details)

@api_reports_bp.route('/reports')
def get_reports():
    """Get reports - filtered for regular users (hide flagged reports)"""
    try:
        user_id = session.get('user_id')
        user_role = session.get('role', session.get('user_type', 'commuter'))
        
        print(f"🔍 DEBUG: User role = {user_role}")
        
        # Get ALL reports
        all_reports = Report.query.order_by(Report.id.desc()).all()
        print(f"🔍 DEBUG: Total reports in database: {len(all_reports)}")
        
        for r in all_reports[:5]:
            print(f"🔍 DEBUG: Report {r.id} - timestamp={r.timestamp}, flagged={getattr(r, 'is_flagged', 'MISSING')}")
        
        reports = all_reports
        
        result = []
        for report in reports:
            # Parse photo paths
            photo_paths = []
            if report.photo_path:
                try:
                    photo_paths = json.loads(report.photo_path)
                except:
                    photo_paths = [report.photo_path] if report.photo_path else []
            
            # Ensure timestamp is properly formatted
            timestamp_str = None
            if report.timestamp:
                timestamp_str = report.timestamp.isoformat()
            
            result.append({
                'id': report.id,
                'station': report.station,
                'direction': report.direction,
                'reported_congestion': report.reported_congestion,
                'predicted_congestion': report.predicted_congestion,
                'remarks': report.remarks,
                'anonymous': report.anonymous,
                'username': report.user.username if report.user and not report.anonymous else None,
                'timestamp': timestamp_str,
                'photo_path': report.photo_path,
                'photo_paths': photo_paths,
                'flag_count': getattr(report, 'flag_count', 0),
                'flagged': getattr(report, 'is_flagged', False),
                'is_hidden': getattr(report, 'is_flagged', False)
            })
        
        print(f"🔍 DEBUG: Returning {len(result)} reports to frontend")
        return jsonify(result)
    except Exception as e:
        print(f"Error in get_reports: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([])
    
@api_reports_bp.route('/debug/all-reports')
def debug_all_reports():
    """Debug endpoint to see all reports regardless of flags"""
    try:
        reports = Report.query.order_by(Report.timestamp.desc()).all()
        
        result = []
        for report in reports:
            result.append({
                'id': report.id,
                'station': report.station,
                'is_flagged': getattr(report, 'is_flagged', 'NO_COLUMN'),
                'flag_count': getattr(report, 'flag_count', 0),
                'status': getattr(report, 'status', 'NO_COLUMN'),
                'timestamp': report.timestamp.isoformat() if report.timestamp else None
            })
        
        return jsonify({
            'total_reports': len(result),
            'reports': result,
            'columns_check': {
                'has_is_flagged': hasattr(Report, 'is_flagged'),
                'has_flag_count': hasattr(Report, 'flag_count'),
                'has_status': hasattr(Report, 'status')
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def is_operating_hours(check_time=None):
    """
    Check if current time is within MRT operating hours
    Operating hours: 4:30 AM to 10:30 PM
    Returns: bool (True if open, False if closed)
    """
    if check_time is None:
        check_time = Config.get_current_time()  # Use Config for consistent year
    
    hour = check_time.hour
    minute = check_time.minute
    current_time = hour + minute / 60
    
    OPERATING_START = 4.5   # 4:30 AM
    OPERATING_END = 22.5    # 10:30 PM
    
    return OPERATING_START <= current_time < OPERATING_END

def get_next_opening_time():
    """Get the next opening time as a string"""
    now = Config.get_current_time()  # Use Config for consistent year
    today_open = now.replace(hour=4, minute=30, second=0, microsecond=0)
    
    if now < today_open:
        return today_open.strftime('%I:%M %p')
    else:
        # Next day opening
        next_day_open = now.replace(day=now.day + 1, hour=4, minute=30, second=0, microsecond=0)
        return next_day_open.strftime('%I:%M %p, %B %d')
    
@api_reports_bp.route('/debug/latest-report', methods=['GET'])
def debug_latest_report():
    """Get the most recent report regardless of flags"""
    try:
        # Get the latest report
        latest = Report.query.order_by(Report.id.desc()).first()
        
        if not latest:
            return jsonify({'error': 'No reports found'}), 404
        
        return jsonify({
            'id': latest.id,
            'station': latest.station,
            'direction': latest.direction,
            'reported_congestion': latest.reported_congestion,
            'predicted_congestion': latest.predicted_congestion,
            'remarks': latest.remarks,
            'anonymous': latest.anonymous,
            'timestamp': latest.timestamp.isoformat() if latest.timestamp else None,
            'user_id': latest.user_id,
            'is_flagged': getattr(latest, 'is_flagged', False),
            'flag_count': getattr(latest, 'flag_count', 0),
            'has_user': latest.user is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@api_reports_bp.route('/report-congestion', methods=['POST'])
def report_congestion():
    try:
        print("=" * 50)
        print("🚀 REPORT SUBMISSION ATTEMPT")
        user_id = session.get('user_id')
        ip_address = request.remote_addr
        print(f"👤 User: {user_id}, IP: {ip_address}")
        print(f"📋 Content-Type: {request.content_type}")
        print(f"📋 Request data: {request.get_data(as_text=True)}")
        
        # TEMPORARILY DISABLE OPERATING HOURS CHECK FOR TESTING
        # if not is_operating_hours():
        #     next_open = get_next_opening_time()
        #     return jsonify({
        #         "success": False, 
        #         "error": f"MRT-3 is currently closed. Operating hours are 4:30 AM - 10:30 PM. Reports can only be submitted during operating hours. Next opening: {next_open}"
        #     }), 403
        
        # TEMPORARILY DISABLE RATE LIMITING FOR TESTING
        # if is_rate_limited(user_id, ip_address, limit=3, window=3600):
        #     return jsonify({"success": False, "error": "You've reached the limit of 3 reports per hour."}), 429
        
        station = None
        direction = None
        reported = None
        remarks = ""
        anonymous = False
        photo_paths = []
        
        # Try to get data from different sources
        if request.content_type and 'multipart/form-data' in request.content_type:
            print("📁 Processing multipart/form-data")
            station = request.form.get('station')
            direction = request.form.get('direction')
            reported = request.form.get('congestion')
            remarks = request.form.get('remarks', '')
            anonymous = request.form.get('anonymous', 'false').lower() == 'true'
        else:
            # Try JSON first
            try:
                data = request.get_json()
                if data:
                    print("📋 Processing JSON data")
                    station = data.get('station')
                    direction = data.get('direction')
                    reported = data.get('congestion')
                    remarks = data.get('remarks', '')
                    anonymous = data.get('anonymous', False)
                else:
                    print("❌ No JSON data found")
                    # Try form data as fallback
                    data = request.form
                    if data:
                        print("📋 Processing form data")
                        station = data.get('station')
                        direction = data.get('direction')
                        reported = data.get('congestion')
                        remarks = data.get('remarks', '')
                        anonymous = data.get('anonymous', 'false').lower() == 'true'
                    else:
                        return jsonify({"success": False, "error": "No data provided"}), 400
            except Exception as json_error:
                print(f"❌ JSON parse error: {json_error}")
                # Try form data as fallback
                data = request.form
                if data:
                    print("📋 Processing form data (fallback)")
                    station = data.get('station')
                    direction = data.get('direction')
                    reported = data.get('congestion')
                    remarks = data.get('remarks', '')
                    anonymous = data.get('anonymous', 'false').lower() == 'true'
                else:
                    return jsonify({"success": False, "error": "No data provided"}), 400
        
        print(f"📋 Parsed: station={station}, congestion={reported}, remarks={remarks}")
        
        # Validation
        if not station:
            return jsonify({"success": False, "error": "Station is required"}), 400
        if reported is None:
            return jsonify({"success": False, "error": "Congestion level is required"}), 400
        
        # Convert to int
        try:
            reported = int(reported)
        except:
            return jsonify({"success": False, "error": "Congestion must be a number"}), 400
        
        # Validate station
        if station not in STATIONS:
            return jsonify({"success": False, "error": "Invalid station"}), 400
        
        # Validate range
        if not (0 <= reported <= 100):
            return jsonify({"success": False, "error": "Congestion must be between 0 and 100"}), 400
        
        # Get prediction
        try:
            print(f"📊 Getting prediction for {station}")
            ridership = get_station_prediction(station)
            capacity = STATION_BASE_CAPACITY.get(station, 10000)
            predicted = int((ridership / capacity) * 100)
            print(f"📊 Prediction: {predicted}%")
        except Exception as pred_error:
            print(f"❌ Prediction error: {pred_error}")
            import traceback
            traceback.print_exc()
            predicted = 50
        
        # Create report
        try:
            photo_path_json = json.dumps(photo_paths) if photo_paths else None
            current_time = Config.get_current_time()
            print(f"📅 Current time: {current_time}")
            
            report = Report(
                user_id=user_id,
                station=station,
                direction=direction,
                reported_congestion=reported,
                predicted_congestion=predicted,
                remarks=remarks[:500] if remarks else None,
                photo_path=photo_path_json,
                anonymous=anonymous,
                timestamp=current_time
            )
            
            print("💾 Saving to database...")
            db.session.add(report)
            db.session.commit()
            print(f"✅ Report saved! ID: {report.id}")
            
            return jsonify({
                "success": True,
                "message": "Report submitted successfully!",
                "photos": len(photo_paths),
                "direction": direction,
                "report_id": report.id,
                "timestamp": current_time.isoformat()
            })
            
        except Exception as db_error:
            print(f"❌ Database error: {db_error}")
            db.session.rollback()
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "error": f"Database error: {str(db_error)}"}), 500
            
    except Exception as e:
        db.session.rollback()
        print(f"❌ UNHANDLED EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500
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
            report.is_flagged = True
            report.flagged_at = Config.get_current_time()
            report.status = 'pending'
        
        db.session.commit()
        
        log_activity(user_id, session.get('role', 'user'), session.get('username', 'unknown'),
                    'flag_report', f'Flagged report #{report_id} from {report.station}')
        
        return jsonify({'success': True, 'message': 'Report flagged for review'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    
# routes/api_reports_bp.py

@api_reports_bp.route('/user-reports/<int:report_id>/flag', methods=['POST'])
def flag_report_user(report_id):
    """User flags a report as inappropriate"""
    try:
        report = Report.query.get(report_id)
        if not report:
            return jsonify({'success': False, 'error': 'Report not found'}), 404
        
        # Increment flag count
        report.flag_count = (report.flag_count or 0) + 1
        
        # Auto-hide after 3 flags
        if report.flag_count >= 3:
            report.is_flagged = True
            report.flagged_at = Config.get_current_time()  # Use Config for consistent year
            report.status = 'pending'  # Needs admin review
            print(f"🔴 Report {report_id} automatically hidden after {report.flag_count} flags")
        
        db.session.commit()
        
        message = f"Report flagged ({report.flag_count}/3). " + \
                  ("Report will be reviewed by admin." if report.flag_count >= 3 else "Report will be hidden after 3 flags.")
        
        return jsonify({'success': True, 'message': message, 'flag_count': report.flag_count})
    except Exception as e:
        db.session.rollback()
        print(f"Error flagging report: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500