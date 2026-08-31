from flask import Blueprint, render_template, session, redirect, url_for

public_bp = Blueprint('public', __name__)

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

@public_bp.route('/')
def home():
    session.clear()
    session['guest_mode'] = True
    return redirect(url_for('user.user_dashboard'))

@public_bp.route('/live-map')
def live_map():
    username = session.get('username', 'Public User')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    
    station_coords = {
        "North Ave": {"lat": 14.6556, "lng": 121.0302},
        "Quezon Ave": {"lat": 14.6390, "lng": 121.0380},
        "Kamuning": {"lat": 14.6249, "lng": 121.0431},
        "Cubao": {"lat": 14.6213, "lng": 121.0529},
        "Santolan": {"lat": 14.6135, "lng": 121.0630},
        "Ortigas": {"lat": 14.5864, "lng": 121.0565},
        "Shaw Blvd": {"lat": 14.5789, "lng": 121.0532},
        "Boni Ave": {"lat": 14.5716, "lng": 121.0492},
        "Guadalupe": {"lat": 14.5655, "lng": 121.0446},
        "Buendia": {"lat": 14.5547, "lng": 121.0329},
        "Ayala Ave": {"lat": 14.5497, "lng": 121.0305},
        "Magallanes": {"lat": 14.5450, "lng": 121.0254},
        "Taft": {"lat": 14.5378, "lng": 121.0112}
    }
    
    return render_template('live-map.html', 
                         stations=STATIONS, 
                         station_coords=station_coords,
                         username=username, 
                         is_logged_in=is_logged_in,
                         google_user=google_user,
                         favorite_station=favorite_station)

@public_bp.route('/travel-plan')
def travel_plan():
    username = session.get('username', 'Public User')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    
    return render_template('travel-plan.html', 
                         stations=STATIONS, 
                         username=username, 
                         is_logged_in=is_logged_in,
                         google_user=google_user,
                         favorite_station=favorite_station)

@public_bp.route('/alerts')
def alerts():
    username = session.get('username', 'Public User')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    
    return render_template('alerts.html', 
                         stations=STATIONS, 
                         username=username, 
                         is_logged_in=is_logged_in,
                         google_user=google_user,
                         favorite_station=favorite_station)

@public_bp.route('/report')
def report():
    username = session.get('username', 'Public User')
    is_logged_in = 'user_id' in session
    google_user = session.get('google_user', False)
    favorite_station = session.get('favorite_station')
    
    return render_template('report.html', 
                         stations=STATIONS, 
                         username=username, 
                         is_logged_in=is_logged_in,
                         google_user=google_user,
                         favorite_station=favorite_station)