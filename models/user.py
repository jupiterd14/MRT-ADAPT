from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from . import db

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
    
    @property
    def password(self):
        raise AttributeError('password is not readable')
    
    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def verify_password(self, password):
        if self.password_hash is None:
            return False
        return check_password_hash(self.password_hash, password)
    
    def has_password(self):
        return self.password_hash is not None
    
    def get_assigned_stations_list(self):
        import json
        if self.assigned_stations:
            try:
                return json.loads(self.assigned_stations)
            except:
                return []
        return []
    
    def set_assigned_stations(self, stations_list):
        import json
        self.assigned_stations = json.dumps(stations_list)