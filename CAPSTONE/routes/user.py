from flask import Blueprint, render_template, session, request, jsonify, redirect, url_for, flash
from models import User, SavedRoute, db
from datetime import datetime
from .auth import login_required
from werkzeug.security import generate_password_hash, check_password_hash

user_bp = Blueprint('user', __name__)

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

# ========== PASSWORD METHODS FOR USER MODEL ==========
def check_password(self, password):
    """Check if password matches hash"""
    return check_password_hash(self.password_hash, password)

def set_password(self, password):
    """Set password hash"""
    self.password_hash = generate_password_hash(password)

# Add methods to User model (if not already present)
if not hasattr(User, 'check_password'):
    User.check_password = check_password

if not hasattr(User, 'set_password'):
    User.set_password = set_password

@user_bp.route('/user-dashboard')
def user_dashboard():
    if session.get('admin_logged_in') or session.get('is_admin'):
        return redirect(url_for('admin.admin_dashboard'))
    
    username = session.get('username', 'Public User')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    
    if not favorite_station or favorite_station not in STATIONS:
        favorite_station = "North Ave"
        if is_logged_in:
            session['favorite_station'] = "North Ave"
            user = User.query.get(session['user_id'])
            if user and user.favorite_station not in STATIONS:
                user.favorite_station = "North Ave"
                db.session.commit()
    
    return render_template('user-dashboard.html', 
                         stations=STATIONS, 
                         username=username, 
                         is_logged_in=is_logged_in,
                         google_user=google_user,
                         favorite_station=favorite_station)

@user_bp.route('/api/saved-routes', methods=['GET'])
def get_saved_routes():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        user_id = session.get('user_id')
        routes = SavedRoute.query.filter_by(user_id=user_id).order_by(SavedRoute.created_at.desc()).all()
        
        result = [{
            'id': route.id,
            'from_station': route.from_station,
            'to_station': route.to_station,
            'route_name': route.route_name,
            'created_at': route.created_at.isoformat()
        } for route in routes]
        
        return jsonify(result)
    except Exception as e:
        print(f"Error fetching saved routes: {e}")
        return jsonify({"error": str(e)}), 500

@user_bp.route('/api/saved-routes', methods=['POST'])
def save_route():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.json
        user_id = session.get('user_id')
        from_station = data.get('from_station')
        to_station = data.get('to_station')
        route_name = data.get('route_name', f"{from_station} to {to_station}")
        
        existing = SavedRoute.query.filter_by(
            user_id=user_id, from_station=from_station, to_station=to_station
        ).first()
        
        if existing:
            return jsonify({"success": True, "message": "Route already saved"})
        
        route = SavedRoute(
            user_id=user_id, from_station=from_station, 
            to_station=to_station, route_name=route_name
        )
        db.session.add(route)
        db.session.commit()
        
        return jsonify({"success": True, "message": "Route saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@user_bp.route('/api/saved-routes/<int:route_id>', methods=['DELETE'])
def delete_saved_route(route_id):
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        user_id = session.get('user_id')
        route = SavedRoute.query.filter_by(id=route_id, user_id=user_id).first()
        
        if not route:
            return jsonify({"error": "Route not found"}), 404
        
        db.session.delete(route)
        db.session.commit()
        
        return jsonify({"success": True, "message": "Route deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@user_bp.route('/api/user/favorite-station', methods=['POST'])
@login_required
def update_favorite_station():
    try:
        data = request.json
        favorite_station = data.get('favorite_station')
        
        if favorite_station not in STATIONS:
            return jsonify({'success': False, 'error': 'Invalid station'}), 400
        
        user = User.query.get(session['user_id'])
        user.favorite_station = favorite_station
        db.session.commit()
        
        session['favorite_station'] = favorite_station
        
        from .auth import log_activity
        log_activity(user.id, user.role, user.username, 'update_favorite', 
                    f'Updated favorite station to {favorite_station}')
        
        return jsonify({'success': True, 'message': 'Favorite station updated'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ========== USER PASSWORD PROFILE ENDPOINT (ONLY UPDATE PASSWORD) ==========
@user_bp.route('/api/profile/update-password', methods=['POST'])
@login_required
def update_password():
    """Update user password"""
    try:
        data = request.json
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not current_password or not new_password:
            return jsonify({'success': False, 'error': 'Current password and new password are required'}), 400
        
        user_id = session.get('user_id')
        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        # Verify current password
        if not check_password_hash(user.password_hash, current_password):
            return jsonify({'success': False, 'error': 'Current password is incorrect'}), 400
        
        # Validate new password
        if len(new_password) < 8:
            return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400
        
        # Update password
        user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        
        from .auth import log_activity
        log_activity(user.id, user.role, user.username, 'password_change', 
                    'Password updated successfully')
        
        return jsonify({'success': True, 'message': 'Password updated successfully'})
        
    except Exception as e:
        db.session.rollback()
        print(f"Error updating password: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500