from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Import all models so they are registered with SQLAlchemy
from .user import User
from .report import Report
from .broadcast import Broadcast
from .activity_log import ActivityLog
from .saved_route import SavedRoute
from .station_data import StationData