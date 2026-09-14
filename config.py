# config.py
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# Make sure the kaggle package sees the token (it reads os.environ directly)
if os.getenv("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = os.getenv("KAGGLE_API_TOKEN")

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24))
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # --- Database ---
    # Use Postgres URL from env if set, otherwise fall back to local SQLite
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'mrt.db')
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Connection pool — important for Aiven (closes idle connections)
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,   # reconnect if the connection went stale
        'pool_recycle': 300,     # recycle connections after 5 minutes
    }

    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
    KAGGLE_API_TOKEN = os.environ.get('KAGGLE_API_TOKEN')

    @staticmethod
    def get_current_time():
        return datetime.now(ZoneInfo('Asia/Manila')).replace(year=2025)