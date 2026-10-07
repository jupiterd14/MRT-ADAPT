from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
import json


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=True)
    role = db.Column(db.String(20), default='user')
    google_id = db.Column(db.String(100), unique=True, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    favorite_station = db.Column(db.String(50), nullable=True)
    last_login = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    access_level = db.Column(db.String(20), default='station')
    assigned_zone = db.Column(db.String(20), nullable=True)
    assigned_stations = db.Column(db.Text, nullable=True)

    # Invitation expiry tracking
    invite_expires_at = db.Column(db.DateTime, nullable=True)

    # ✅ NEW: opaque invite token — replaces the "temp password in URL" flow.
    # Unique + indexed so lookups by token are O(1). Nullable because most
    # users never have an active invite.
    invite_token = db.Column(db.String(128), unique=True, nullable=True, index=True)

    # ✅ NEW: soft-ban flag. Lets admin mark an account as banned without
    # deleting it, so a re-invite can't silently revive it.
    banned = db.Column(db.Boolean, default=False, nullable=False)
    session_epoch = db.Column(db.Integer, default=0, nullable=False)

    @property
    def password(self):
        raise AttributeError('password is not readable')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        if self.password_hash is None:
            return False
        # ✅ FIX: normalize to bool so callers can't accidentally treat a
        # truthy non-bool as success.
        return bool(check_password_hash(self.password_hash, password))

    def has_password(self):
        return self.password_hash is not None

    def get_assigned_stations_list(self):
        # ✅ FIX: moved import to module top; kept method signature.
        if self.assigned_stations:
            try:
                return json.loads(self.assigned_stations)
            except Exception:
                return []
        return []

    def set_assigned_stations(self, stations_list):
        self.assigned_stations = json.dumps(stations_list)

    # ---------------- Invite helpers ----------------
    def is_invite_expired(self):
        """Returns True if the user has an invite expiry set and it's in the past."""
        if self.invite_expires_at is None:
            return False
        return self.invite_expires_at < datetime.now()

    def clear_invite(self):
        """Call after successful signup to consume the invite."""
        self.invite_expires_at = None
        # ✅ NEW: also clear the token so the link can't be replayed
        self.invite_token = None

    # ✅ NEW: helper for the invite flow
    def has_active_invite(self):
        return (
            self.invite_token is not None
            and self.invite_expires_at is not None
            and self.invite_expires_at > datetime.now()
        )