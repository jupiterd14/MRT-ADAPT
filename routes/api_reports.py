from flask import Blueprint, request, jsonify, session, current_app
from models import Report, User, db
from datetime import datetime, timedelta
import json, os, re, time
from collections import defaultdict
from werkzeug.utils import secure_filename
from config import Config
from utils import log_activity
from utils.rate_limit import (
    is_report_rate_limited,
    track_report,
    is_flag_rate_limited,
    track_flag,
    user_already_flagged,
    mark_user_flag,
    get_remaining_reports as _get_remaining,
)

api_reports_bp = Blueprint('api_reports', __name__)

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan",
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia",
            "Ayala Ave", "Magallanes", "Taft"]

STATION_BASE_CAPACITY = {
    "North Ave": 12000, "Quezon Ave": 9000, "Kamuning": 7500, "Cubao": 15000,
    "Santolan": 8000, "Ortigas": 9500, "Shaw Blvd": 11000, "Boni Ave": 8500,
    "Guadalupe": 10000, "Buendia": 9000, "Ayala Ave": 14000, "Magallanes": 9000, "Taft": 16000
}

# ── Image magic bytes ──
_MAGIC_BYTES = (
    b'\x89PNG\r\n\x1a\n',  # PNG
    b'\xff\xd8\xff',        # JPEG
    b'GIF87a',              # GIF
    b'GIF89a',
    b'RIFF',                # WebP container (loose check)
)
_MAX_IMAGE_BYTES = 5 * 1024 * 1024

def _is_valid_image(file):
    """Check size + magic bytes. Leaves file pointer at 0."""
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size == 0 or size > _MAX_IMAGE_BYTES:
        return False
    header = file.read(12)
    file.seek(0)
    return any(header.startswith(m) for m in _MAGIC_BYTES)


# ── Helpers that stay in this module ──
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
    predictor = current_app.config.get('LSTM_PREDICTOR')
    if predictor is not None:
        try:
            with current_app.app_context():
                result = predictor.predict_congestion(station_name, direction, db.session)
                if result is not None:
                    return float(result)
        except Exception as e:
            print(f"⚠️ LSTM prediction error: {e}")

    if 'GET_STATION_PREDICTION' in current_app.config:
        return current_app.config['GET_STATION_PREDICTION'](station_name, direction)

    now = datetime.now()
    hour = now.hour
    capacity = STATION_BASE_CAPACITY.get(station_name, 10000)
    base = 30 if capacity > 10000 else 40
    if 7 <= hour <= 9:
        return min(95, base + 40) if direction == 'Southbound' else min(70, base + 20)
    elif 17 <= hour <= 19:
        return min(95, base + 40) if direction == 'Northbound' else min(70, base + 20)
    elif 12 <= hour <= 14:
        return base + 10
    else:
        return max(10, base - 10)


def is_operating_hours(check_time=None):
    if check_time is None:
        check_time = Config.get_current_time()
    current = check_time.hour + check_time.minute / 60
    return 4.5 <= current < 22.5


def get_next_opening_time():
    now = Config.get_current_time()
    today_open = now.replace(hour=4, minute=30, second=0, microsecond=0)
    if now < today_open:
        return today_open.strftime('%I:%M %p')
    next_day_open = (now + timedelta(days=1)).replace(
        hour=4, minute=30, second=0, microsecond=0
    )
    return next_day_open.strftime('%I:%M %p, %B %d')


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@api_reports_bp.route('/remaining-reports', methods=['GET'])
def remaining_reports():
    """How many more reports can this identity submit today?"""
    try:
        user_id = session.get('user_id')
        ip_address = request.remote_addr
        remaining = _get_remaining(user_id, ip_address, limit=3, window=86400)
        return jsonify({
            'remaining': remaining,
            'max_per_day': 3,
            'reset_at': Config.get_current_time().replace(
                hour=0, minute=0, second=0, microsecond=0
            ).isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)[:200]}), 500


# Keep old name as alias so existing imports still work
get_remaining_reports = remaining_reports


@api_reports_bp.route('/lstm-status', methods=['GET'])
def lstm_status():
    predictor = current_app.config.get('LSTM_PREDICTOR')
    if not predictor:
        return jsonify({'status': 'not_initialized', 'message': 'LSTM predictor not initialized'})
    return jsonify({
        'status': 'ready',
        'models_loaded': len(predictor.models),
        'station_directions': predictor.station_directions,
        'model_path': predictor.model_path,
        'feature_cols_count': len(predictor.feature_cols) if predictor.feature_cols else 0
    })


@api_reports_bp.route('/predict-station', methods=['POST'])
def predict_station():
    try:
        data = request.get_json() or {}
        station = data.get('station')
        direction = data.get('direction', 'Northbound')
        if not station:
            return jsonify({'error': 'Station required'}), 400
        prediction = get_station_prediction(station, direction)
        return jsonify({
            'station': station,
            'direction': direction,
            'predicted_congestion': prediction,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)[:200]}), 500


@api_reports_bp.route('/retrain-models', methods=['POST'])
def retrain_models():
    try:
        user_id = session.get('user_id')
        user_role = session.get('role', session.get('user_type', 'commuter'))
        if user_role not in ['admin', 'staff', 'admin_staff']:
            return jsonify({'error': 'Admin access required'}), 403
        from training.scheduled_trainer import retrain_models_with_reports
        success = retrain_models_with_reports(db.session)
        log_activity(user_id, user_role, session.get('username', 'unknown'),
                     'retrain_models', f'Manual retraining {"successful" if success else "failed"}')
        return jsonify({'success': success,
                        'message': 'Retraining completed' if success else 'Retraining failed'})
    except Exception as e:
        return jsonify({'error': str(e)[:200]}), 500


@api_reports_bp.route('/retrain-now', methods=['POST'])
def retrain_now():
    try:
        user_id = session.get('user_id')
        user_role = session.get('role', session.get('user_type', 'commuter'))
        if user_role not in ['admin', 'staff', 'admin_staff']:
            return jsonify({'error': 'Admin access required'}), 403
        from training.scheduled_trainer import retrain_models_with_reports
        success = retrain_models_with_reports(db.session)
        log_activity(user_id, user_role, session.get('username', 'unknown'),
                     'manual_retrain', f'Manual retraining {"successful" if success else "failed"}')
        return jsonify({'success': success,
                        'message': 'Retraining completed successfully' if success else 'Retraining failed or insufficient data'})
    except Exception as e:
        return jsonify({'error': str(e)[:200]}), 500


@api_reports_bp.route('/report-congestion', methods=['POST'])
def report_congestion():
    try:
        print("=" * 50)
        print("🚀 REPORT SUBMISSION ATTEMPT")
        user_id = session.get('user_id')
        ip_address = request.remote_addr
        print(f"👤 User: {user_id}, IP: {ip_address}")

        # Rate limiting (both user + IP)
        if is_report_rate_limited(user_id, ip_address, limit=3, window=86400):
            remaining = _get_remaining(user_id, ip_address, limit=3, window=86400)
            return jsonify({
                "success": False,
                "error": f"You've reached the limit of 3 reports per day. You have {remaining} report(s) remaining today."
            }), 429

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

            files = request.files.getlist('images')
            print(f"📁 Received {len(files)} file(s)")

            upload_folder = os.path.join('static', 'uploads', 'reports')
            os.makedirs(upload_folder, exist_ok=True)

            for file in files:
                if not file or not file.filename:
                    continue
                if not _is_valid_image(file):
                    print(f"⚠️ Rejected invalid/oversized image: {file.filename}")
                    continue
                safe_name = secure_filename(file.filename)
                if not safe_name:
                    continue
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                safe_filename = f"report_{timestamp}_{safe_name}"
                file_path = os.path.join(upload_folder, safe_filename)
                file.save(file_path)
                photo_paths.append(f"static/uploads/reports/{safe_filename}")
        else:
            try:
                data = request.get_json(silent=True)
            except Exception:
                data = None
            if data:
                station = data.get('station')
                direction = data.get('direction')
                reported = data.get('congestion')
                remarks = data.get('remarks', '')
                anonymous = data.get('anonymous', False)
            else:
                data = request.form
                if not data:
                    return jsonify({"success": False, "error": "No data provided"}), 400
                station = data.get('station')
                direction = data.get('direction')
                reported = data.get('congestion')
                remarks = data.get('remarks', '')
                anonymous = data.get('anonymous', 'false').lower() == 'true'

        if not station:
            return jsonify({"success": False, "error": "Station is required"}), 400
        if reported is None:
            return jsonify({"success": False, "error": "Congestion level is required"}), 400
        try:
            reported = int(reported)
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "Congestion must be a number"}), 400
        if station not in STATIONS:
            return jsonify({"success": False, "error": "Invalid station"}), 400
        if not (0 <= reported <= 100):
            return jsonify({"success": False, "error": "Congestion must be between 0 and 100"}), 400

        # Suspicious remarks guard (existing helper, now actually used)
        if is_suspicious_remarks(remarks):
            return jsonify({"success": False, "error": "Remarks look like spam. Please be more descriptive."}), 400

        try:
            ridership = get_station_prediction(station)
            capacity = STATION_BASE_CAPACITY.get(station, 10000)
            predicted = int((ridership / capacity) * 100)
        except Exception:
            predicted = 50

        try:
            photo_path_json = json.dumps(photo_paths) if photo_paths else None
            current_time = Config.get_current_time()
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
            db.session.add(report)
            db.session.commit()
            track_report(user_id, ip_address)
            return jsonify({
                "success": True,
                "message": "Report submitted successfully!",
                "photos": len(photo_paths),
                "photo_paths": photo_paths,
                "direction": direction,
                "report_id": report.id,
                "timestamp": current_time.isoformat()
            })
        except Exception as db_error:
            try:
                db.session.rollback()
            except Exception:
                pass
            print(f"❌ Database error: {type(db_error).__name__}")
            return jsonify({
                "success": False,
                "error": "Database temporarily unavailable. Please try again in a moment."
            }), 503

    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        print(f"❌ UNHANDLED EXCEPTION: {type(e).__name__}")
        return jsonify({"success": False, "error": "Server error. Please try again."}), 500


@api_reports_bp.route('/public-reports', methods=['GET'])
def get_reports():
    """Get all reports - Public endpoint (no user_id leak)."""
    try:
        try:
            limit = min(int(request.args.get('limit', 200)), 500)
        except (TypeError, ValueError):
            limit = 200

        reports = (Report.query
                   .order_by(Report.timestamp.desc())
                   .limit(limit)
                   .all())

        current_user = session.get('user_id')
        result = []
        for report in reports:
            photo_paths = []
            if report.photo_path:
                try:
                    if isinstance(report.photo_path, str) and report.photo_path.startswith('['):
                        photo_paths = json.loads(report.photo_path)
                    elif isinstance(report.photo_path, str) and report.photo_path:
                        photo_paths = [report.photo_path]
                    elif isinstance(report.photo_path, list):
                        photo_paths = report.photo_path
                except Exception:
                    photo_paths = []

            # Hide flagged reports from public view
            if getattr(report, 'flagged', False) and getattr(report, 'status', '') == 'pending':
                continue

            result.append({
                'id': report.id,
                'station': report.station,
                'direction': report.direction,
                'reported_congestion': report.reported_congestion,
                'predicted_congestion': report.predicted_congestion,
                'remarks': report.remarks,
                'photo_paths': photo_paths,
                'photo_path': report.photo_path,
                'anonymous': report.anonymous,
                'timestamp': report.timestamp.isoformat() if report.timestamp else None,
                'status': getattr(report, 'status', 'active'),
                'flagged': getattr(report, 'flagged', False),
                'flag_count': getattr(report, 'flag_count', 0),
                # ✅ user_id removed. Instead, tell the caller if it's theirs:
                'is_mine': current_user is not None and report.user_id == current_user,
            })

        response = jsonify({
            'total_reports': len(result),
            'reports': result,
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        print(f"❌ Error in /public-reports: {type(e).__name__}: {str(e)[:200]}")
        return jsonify({'error': 'Database temporarily unavailable'}), 503


# Alias — some blueprints may still reference `get_reports`
get_reports_public = get_reports


@api_reports_bp.route('/reports/<int:report_id>/flag', methods=['POST'])
def flag_report(report_id):
    """Logged-in user flags a report (moderation queue)."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'Please log in to flag reports'}), 401

        ip_address = request.remote_addr
        if is_flag_rate_limited(ip_address, user_id=user_id, limit=10, window=3600):
            return jsonify({'success': False, 'error': 'Too many flags. Try again later.'}), 429

        if user_already_flagged(user_id, report_id, window=86400):
            return jsonify({'success': False, 'error': 'You already flagged this report.'}), 409

        if report_id <= 0:
            return jsonify({'success': False, 'error': 'Invalid report id'}), 400

        report = Report.query.get(report_id)
        if not report:
            return jsonify({'success': False, 'error': 'Report not found'}), 404

        report.flag_count = (report.flag_count or 0) + 1
        if report.flag_count >= 3:
            report.flagged = True
            report.flagged_at = Config.get_current_time()
            report.status = 'pending'

        db.session.commit()
        track_flag(ip_address, user_id=user_id)
        mark_user_flag(user_id, report_id)

        log_activity(user_id, session.get('role', 'user'), session.get('username', 'unknown'),
                     'flag_report', f'Flagged report #{report_id} from {report.station}')

        return jsonify({'success': True, 'message': 'Report flagged for review'})
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        return jsonify({'success': False, 'error': str(e)[:200]}), 503


@api_reports_bp.route('/user-reports/<int:report_id>/flag', methods=['POST'])
def flag_report_user(report_id):
    """Anonymous-friendly flag endpoint."""
    try:
        ip_address = request.remote_addr
        user_id = session.get('user_id')

        if is_flag_rate_limited(ip_address, user_id=user_id, limit=10, window=3600):
            return jsonify({'success': False, 'error': 'Too many flags. Try again later.'}), 429

        if user_id and user_already_flagged(user_id, report_id, window=86400):
            return jsonify({'success': False, 'error': 'You already flagged this report.'}), 409

        if report_id <= 0:
            return jsonify({'success': False, 'error': 'Invalid report id'}), 400

        report = Report.query.get(report_id)
        if not report:
            return jsonify({'success': False, 'error': 'Report not found'}), 404

        report.flag_count = (report.flag_count or 0) + 1
        if report.flag_count >= 3:
            report.flagged = True
            report.flagged_at = Config.get_current_time()
            report.status = 'pending'
            print(f"🔴 Report {report_id} automatically hidden after {report.flag_count} flags")

        db.session.commit()
        track_flag(ip_address, user_id=user_id)
        if user_id:
            mark_user_flag(user_id, report_id)

        message = f"Report flagged ({report.flag_count}/3). " + \
                  ("Report will be reviewed by admin." if report.flag_count >= 3 else "Report will be hidden after 3 flags.")
        return jsonify({'success': True, 'message': message, 'flag_count': report.flag_count})
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        print(f"Error flagging report: {type(e).__name__}")
        return jsonify({'success': False, 'error': str(e)[:200]}), 503