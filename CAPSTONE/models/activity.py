from datetime import datetime
from . import db

class Activity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    activity_type = db.Column(db.String(50))
    title = db.Column(db.String(200))
    description = db.Column(db.String(500))
    icon = db.Column(db.String(50))
    icon_color = db.Column(db.String(50))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    station = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    user = db.relationship('User', backref=db.backref('activities', lazy=True))