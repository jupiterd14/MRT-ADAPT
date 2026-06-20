# config.py
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24))
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'mrt.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')

    # ========== SIMULATION CLOCK CENTER ==========
    @staticmethod
    def get_current_time():
        """Returns the real current time with year set to 2025 for dataset compatibility"""
        now = datetime.now()
        # Keep real month, day, hour, minute, second
        # Only change the year to 2025 to match your dataset
        return now.replace(year=2025)