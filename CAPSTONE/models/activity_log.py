from datetime import datetime
from . import db

class ActivityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    user_type = db.Column(db.String(20))
    user_email = db.Column(db.String(100))
    action = db.Column(db.String(100))
    details = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.now)
    
    # Flag columns for admin review system
    is_flagged = db.Column(db.Boolean, default=False)
    flag_reason = db.Column(db.String(500), nullable=True)
    flagged_at = db.Column(db.DateTime, nullable=True)
    admin_review_notes = db.Column(db.Text, nullable=True)
    reviewed_by = db.Column(db.String(100), nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    
    user = db.relationship('User', backref=db.backref('activity_logs', lazy=True))