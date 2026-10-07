# config.py
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# Make sure the kaggle package sees the token (it reads os.environ directly)
if os.getenv("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = os.getenv("KAGGLE_API_TOKEN")


class Config:
    # ------------------------------------------------------------------
    # SECRETS
    # ------------------------------------------------------------------
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY must be set in environment")

    # ------------------------------------------------------------------
    # SESSION / COOKIE HARDENING
    # ------------------------------------------------------------------
    # Secure flag: on Render (RENDER=true) or when explicitly in production.
    # Never rely on FLASK_ENV alone — Render doesn't set it by default.
    SESSION_COOKIE_SECURE = (
        os.environ.get('RENDER') is not None
        or os.environ.get('FLASK_ENV') == 'production'
    )
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # Don't leak that the backend is Flask
    SESSION_COOKIE_NAME = 'mrt3_session'

    # Sessions expire after 8 hours of inactivity, not Flask's default 31 days.
    # `session.permanent = True` in auth.py activates this.
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)
    SESSION_REFRESH_EACH_REQUEST = True

    # ------------------------------------------------------------------
    # DATABASE
    # ------------------------------------------------------------------
    # Use Postgres URL from env if set, otherwise fall back to local SQLite
    _DB_URL = os.environ.get(
        'DATABASE_URL',
        'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'mrt.db')
    )
    SQLALCHEMY_DATABASE_URI = _DB_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Connection pool — important for Aiven (closes idle connections).
    # NOTE: `sslmode` and `connect_timeout` are Postgres-only. Passing them
    # to SQLite will crash the driver, so build options conditionally.
    if _DB_URL.startswith('postgres'):
        SQLALCHEMY_ENGINE_OPTIONS = {
            'pool_pre_ping': True,   # reconnect if the connection went stale
            'pool_recycle': 300,     # recycle connections after 5 minutes
            'connect_args': {
                'connect_timeout': 5,   # fail fast if Aiven is unreachable
                'sslmode': 'require',   # Aiven requires SSL
            },
        }
    else:
        SQLALCHEMY_ENGINE_OPTIONS = {}

    # ------------------------------------------------------------------
    # OAUTH / EXTERNAL
    # ------------------------------------------------------------------
    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
    KAGGLE_API_TOKEN = os.environ.get('KAGGLE_API_TOKEN')

    # ------------------------------------------------------------------
    # ADMIN ENV BYPASS (hashed, not plaintext)
    # ------------------------------------------------------------------
    # Set ADMIN_PASSWORD_HASH in env, NOT ADMIN_PASSWORD.
    # Generate one with:
    #   python -c "from werkzeug.security import generate_password_hash; \
    #              print(generate_password_hash('your-password'))"
    ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL')
    ADMIN_PASSWORD_HASH = os.environ.get('ADMIN_PASSWORD_HASH')

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def get_current_time():
        """
        Returns 'now' in Manila time.

        NOTE: year is pinned to 2025 on purpose so predictions align with
        the model's training window (2022-2024). Do NOT change without
        retraining / re-verifying the model's date features.
        """
        return datetime.now(ZoneInfo('Asia/Manila')).replace(year=2025)