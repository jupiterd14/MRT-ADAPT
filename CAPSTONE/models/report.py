from datetime import datetime
from . import db

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    station = db.Column(db.String(50))
    direction = db.Column(db.String(20), nullable=True)
    reported_congestion = db.Column(db.Integer)
    predicted_congestion = db.Column(db.Integer)
    remarks = db.Column(db.String(500), nullable=True)
    photo_path = db.Column(db.Text, nullable=True)
    anonymous = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    flagged = db.Column(db.Boolean, default=False)
    reviewed = db.Column(db.Boolean, default=False)
    flag_count = db.Column(db.Integer, default=0)
    
    user = db.relationship('User', backref=db.backref('reports', lazy=True))