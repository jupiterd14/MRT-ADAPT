from datetime import datetime
from . import db


class PasswordReset(db.Model):
    __tablename__ = 'password_resets'

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    code_hash  = db.Column(db.String(255), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False, index=True)
    attempts   = db.Column(db.Integer, default=0, nullable=False)
    used       = db.Column(db.Boolean, default=False, nullable=False, index=True)
    used_at    = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)