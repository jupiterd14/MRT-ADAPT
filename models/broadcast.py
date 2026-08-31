from datetime import datetime, timedelta
from . import db

class Broadcast(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    message = db.Column(db.Text)
    disruption_type = db.Column(db.String(50))
    stations = db.Column(db.Text)  # JSON
    severity = db.Column(db.String(20))
    operator_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.now)
    is_active = db.Column(db.Boolean, default=True)
    direction = db.Column(db.String(20), default='both')
    
    # NEW FIELDS
    duration_minutes = db.Column(db.Integer, default=60)  # Duration in minutes
    expires_at = db.Column(db.DateTime, nullable=True)   # When this broadcast expires
    
    def is_expired(self):
        """Check if broadcast has expired"""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def get_remaining_time(self):
        """Get remaining time as string"""
        if not self.expires_at:
            return "No expiry"
        remaining = self.expires_at - datetime.now()
        if remaining.total_seconds() <= 0:
            return "Expired"
        hours = int(remaining.total_seconds() // 3600)
        minutes = int((remaining.total_seconds() % 3600) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"