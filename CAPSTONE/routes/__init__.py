from .auth import auth_bp
from .user import user_bp
from .admin import admin_bp
from .operator import operator_bp
from .public import public_bp
from .api_predict import api_predict_bp
from .api_schedule import api_schedule_bp
from .api_reports import api_reports_bp
from .api_other import api_other_bp

__all__ = [
    'auth_bp', 'user_bp', 'admin_bp', 'operator_bp', 'public_bp',
    'api_predict_bp', 'api_schedule_bp', 'api_reports_bp', 'api_other_bp'
]