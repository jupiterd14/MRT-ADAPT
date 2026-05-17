from datetime import datetime
from . import db

class StationData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    station = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.now)
    congestion = db.Column(db.Float)
    ridership = db.Column(db.Integer)
    hour = db.Column(db.Integer)
    weekday = db.Column(db.Integer)
    
    def __repr__(self):
        return f'<StationData {self.station} at {self.timestamp}>'