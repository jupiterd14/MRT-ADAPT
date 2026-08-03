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
def get_station_prediction(station_name, direction='Northbound', **kwargs):
    """
    Get congestion prediction using LSTM model
    
    Args:
        station_name: Name of the station
        direction: 'Northbound' or 'Southbound'
    
    Returns:
        int: Predicted congestion percentage (0-100)
    """
    from flask import current_app
    
    # 1. Try LSTM predictor
    predictor = current_app.config.get('LSTM_PREDICTOR')
    
    if predictor is not None:
        try:
            with current_app.app_context():
                from models import db
                result = predictor.predict_congestion(station_name, direction, db.session)
                if result is not None:
                    return float(result)
        except Exception as e:
            print(f"⚠️ LSTM prediction error: {e}")
    
    # 2. Try config forwarder
    if hasattr(current_app, 'config') and 'GET_STATION_PREDICTION' in current_app.config:
        return current_app.config['GET_STATION_PREDICTION'](station_name, direction)
    
    # 3. Last resort: time-based estimate
    now = datetime.now()
    hour = now.hour
    
    # Station-specific base
    capacity = STATION_BASE_CAPACITY.get(station_name, 10000)
    base = 30 if capacity > 10000 else 40
    
    # Time-based adjustments
    if 7 <= hour <= 9:
        if direction == 'Southbound':
            return min(95, base + 40)  # Morning rush southbound
        else:
            return min(70, base + 20)
    elif 17 <= hour <= 19:
        if direction == 'Northbound':
            return min(95, base + 40)  # Evening rush northbound
        else:
            return min(70, base + 20)
    elif 12 <= hour <= 14:
        return base + 10
    else:
        return max(10, base - 10)
    
@api_reports_bp.route('/debug/taft-recent', methods=['GET'])
def debug_taft_recent():
    """Check for recent Taft reports (last 30 days)"""
    try:
        from datetime import timedelta
        cutoff = Config.get_current_time() - timedelta(days=30)
        
        taft_reports = Report.query.filter(
            Report.station == 'Taft',
            Report.timestamp > cutoff
        ).order_by(Report.timestamp.desc()).all()
        
        return jsonify({
            'recent_taft_count': len(taft_reports),
            'reports': [{
                'id': r.id,
                'timestamp': r.timestamp.isoformat() if r.timestamp else None,
                'is_flagged': getattr(r, 'flagged', False),
                'flag_count': getattr(r, 'flag_count', 0),
                'archived': getattr(r, 'archived', False),
                'congestion': r.reported_congestion,
                'direction': r.direction
            } for r in taft_reports]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
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
                'is_flagged': getattr(report, 'flagged', 'NO_COLUMN'),
                'flag_count': getattr(report, 'flag_count', 0),
                'status': getattr(report, 'status', 'NO_COLUMN'),
                'timestamp': report.timestamp.isoformat() if report.timestamp else None
            })
        
        return jsonify({
            'total_reports': len(result),
            'reports': result,
            'columns_check': {
                'has_is_flagged': hasattr(Report, 'flagged'),
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
    
    
# Add to routes/api_reports_bp.py (at the end of the file)

# ============================================
# LSTM MODEL MANAGEMENT ENDPOINTS
# ============================================

@api_reports_bp.route('/lstm-status', methods=['GET'])
def lstm_status():
    """Check LSTM model status"""
    from flask import current_app
    
    predictor = current_app.config.get('LSTM_PREDICTOR')
    
    if not predictor:
        return jsonify({
            'status': 'not_initialized',
            'message': 'LSTM predictor not initialized'
        })
    
    return jsonify({
        'status': 'ready',
        'models_loaded': len(predictor.models),
        'station_directions': predictor.station_directions,
        'model_path': predictor.model_path,
        'feature_cols_count': len(predictor.feature_cols) if predictor.feature_cols else 0
    })

@api_reports_bp.route('/predict-station', methods=['POST'])
def predict_station():
    """Predict congestion for a station using LSTM"""
    try:
        data = request.get_json()
        station = data.get('station')
        direction = data.get('direction', 'Northbound')
        
        if not station:
            return jsonify({'error': 'Station required'}), 400
        
        # Use the enhanced prediction function
        prediction = get_station_prediction(station, direction)
        
        return jsonify({
            'station': station,
            'direction': direction,
            'predicted_congestion': prediction,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_reports_bp.route('/retrain-models', methods=['POST'])
def retrain_models():
    """Admin endpoint to manually trigger retraining"""
    try:
        user_id = session.get('user_id')
        user_role = session.get('role', session.get('user_type', 'commuter'))
        
        if user_role not in ['admin', 'staff', 'admin_staff']:
            return jsonify({'error': 'Admin access required'}), 403
        
        from training.scheduled_trainer import retrain_models_with_reports
        success = retrain_models_with_reports(db.session)
        
        log_activity(user_id, user_role, session.get('username', 'unknown'),
                    'retrain_models', f'Manual retraining {"successful" if success else "failed"}')
        
        return jsonify({
            'success': success,
            'message': 'Retraining completed' if success else 'Retraining failed'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@api_reports_bp.route('/retrain-now', methods=['POST'])
def retrain_now():
    """Admin endpoint to manually trigger retraining"""
    try:
        user_id = session.get('user_id')
        user_role = session.get('role', session.get('user_type', 'commuter'))
        
        # Check if user is admin
        if user_role not in ['admin', 'staff', 'admin_staff']:
            return jsonify({'error': 'Admin access required'}), 403
        
        # Import and run retraining
        from training.scheduled_trainer import retrain_models_with_reports
        success = retrain_models_with_reports(db.session)
        
        # Log the activity
        log_activity(user_id, user_role, session.get('username', 'unknown'),
                    'manual_retrain', f'Manual retraining {"successful" if success else "failed"}')
        
        return jsonify({
            'success': success,
            'message': 'Retraining completed successfully' if success else 'Retraining failed or insufficient data'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
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
            'is_flagged': getattr(latest, 'flagged', False),
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
    
# Add this debug route to check
@api_reports_bp.route('/debug/check-image/<filename>')
def debug_check_image(filename):
    import os
    from flask import current_app
    
    # Check multiple possible locations
    possible_paths = [
        os.path.join('static', 'uploads', 'reports', filename),
        os.path.join('uploads', 'reports', filename),
        os.path.join('static', 'uploads', filename),
        os.path.join('uploads', filename),
    ]
    
    results = {}
    for path in possible_paths:
        exists = os.path.exists(path)
        results[path] = exists
        if exists:
            results['found_at'] = path
            results['size'] = os.path.getsize(path)
            break
    
    return jsonify({
        'filename': filename,
        'search_results': results,
        'current_directory': os.getcwd(),
        'static_exists': os.path.exists('static'),
        'uploads_exists': os.path.exists(os.path.join('static', 'uploads', 'reports'))
    })
    
@api_reports_bp.route('/uploads/reports/<filename>')
def serve_upload(filename):
    from flask import send_from_directory, abort, send_file
    import os
    from io import BytesIO
    
    upload_folder = os.path.join(os.path.dirname(__file__), 'static', 'uploads', 'reports')
    
    # Check if file exists
    file_path = os.path.join(upload_folder, filename)
    if os.path.exists(file_path):
        return send_from_directory(upload_folder, filename)
    
    # If file doesn't exist, return a placeholder image
    # This prevents 404 errors in the frontend
    try:
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (300, 200), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((100, 80), "No Image", fill=(150, 150, 150))
        draw.text((80, 110), "Report Photo", fill=(150, 150, 150))
        img_io = BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except:
        # If PIL not installed, return a simple SVG
        svg = '''<svg width="300" height="200" xmlns="http://www.w3.org/2000/svg">
            <rect width="300" height="200" fill="#f0f0f0"/>
            <text x="150" y="100" font-family="Arial" font-size="20" fill="#999" text-anchor="middle">No Image</text>
            <text x="150" y="130" font-family="Arial" font-size="14" fill="#999" text-anchor="middle">Report Photo</text>
        </svg>'''
        return send_file(BytesIO(svg.encode()), mimetype='image/svg+xml')

@api_reports_bp.route('/debug/route-check', methods=['GET'])
def debug_route_check():
    """Check what routes are registered for reports"""
    from flask import current_app
    
    routes = []
    for rule in current_app.url_map.iter_rules():
        if 'reports' in str(rule):
            routes.append({
                'path': str(rule),
                'endpoint': rule.endpoint,
                'methods': list(rule.methods),
                'blueprint': rule.endpoint.split('.')[0] if '.' in rule.endpoint else 'none'
            })
    
    return jsonify({
        'report_routes': routes,
        'total_routes': len(routes)
    })
    
    
@api_reports_bp.route('/public-reports', methods=['GET'])
def get_reports():
    """Get all reports - Public endpoint"""
    try:
        print("=" * 50)
        print("📊 GET /api/reports called")
        
        reports = Report.query.order_by(Report.timestamp.desc()).all()
        
        result = []
        for report in reports:
            # Parse photo paths if they exist
            photo_paths = []
            if report.photo_path:
                try:
                    if isinstance(report.photo_path, str) and report.photo_path.startswith('['):
                        photo_paths = json.loads(report.photo_path)
                    elif isinstance(report.photo_path, str) and report.photo_path:
                        photo_paths = [report.photo_path]
                    elif isinstance(report.photo_path, list):
                        photo_paths = report.photo_path
                except:
                    photo_paths = []
            
            result.append({
                'id': report.id,
                'station': report.station,
                'direction': report.direction,
                'reported_congestion': report.reported_congestion,
                'predicted_congestion': report.predicted_congestion,
                'remarks': report.remarks,
                'photo_paths': photo_paths,
                'photo_path': report.photo_path,  # Keep for backward compatibility
                'anonymous': report.anonymous,
                'timestamp': report.timestamp.isoformat() if report.timestamp else None,
                'status': getattr(report, 'status', 'active'),
                'flagged': getattr(report, 'flagged', False),
                'flag_count': getattr(report, 'flag_count', 0),
                'user_id': report.user_id
            })
        
        response = jsonify({
            'total_reports': len(result),
            'reports': result,
        })
        
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        print(f"❌ Error in /reports: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    


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
            report.flagged = True
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