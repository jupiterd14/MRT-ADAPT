# routes/api_other.py
from flask import Blueprint, request, jsonify, session, current_app
from models import Broadcast, User, ActivityLog, Report
from datetime import datetime, timedelta
import json
import time

api_other_bp = Blueprint('api_other', __name__)

# Station data (will be overridden by app.config)
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

STATION_BASE_CAPACITY = {
    "North Ave": 12000, "Quezon Ave": 9000, "Kamuning": 7500, "Cubao": 15000,
    "Santolan": 8000, "Ortigas": 9500, "Shaw Blvd": 11000, "Boni Ave": 8500,
    "Guadalupe": 10000, "Buendia": 9000, "Ayala Ave": 14000, "Magallanes": 9000, "Taft": 16000
}

typeIcons = {
    "Train Breakdown": "fa-train", "Overcrowding": "fa-users", 
    "Maintenance": "fa-wrench", "Signal Issue": "fa-satellite-dish",
    "Gate Closure": "fa-door-closed", "General Notice": "fa-bullhorn"
}


def get_station_predictions_from_config(station_name):
    """Get prediction from app config"""
    from flask import current_app
    if hasattr(current_app, 'config') and 'GET_STATION_PREDICTION' in current_app.config:
        return current_app.config['GET_STATION_PREDICTION'](station_name)
    return 50


def get_directional_from_config(station_name, direction, target_datetime=None):
    """Get directional prediction from app config"""
    from flask import current_app
    if hasattr(current_app, 'config') and 'GET_DIRECTIONAL_PREDICTION' in current_app.config:
        return current_app.config['GET_DIRECTIONAL_PREDICTION'](station_name, direction, target_datetime)
    return 50


def get_stations_from_config():
    """Get stations from app config"""
    from flask import current_app
    if hasattr(current_app, 'config') and 'STATIONS' in current_app.config:
        return current_app.config['STATIONS']
    return STATIONS

# routes/api_other.py - COMPLETE REWRITE OF THE V2 ENDPOINT

@api_other_bp.route('/live-map/directions/v2')
def live_map_directions_v2():
    """New version using directional models - Called by frontend"""
    print("\n" + "="*60)
    print("📍 LIVE MAP V2 API CALLED")
    print("="*60)
    
    try:
        from flask import current_app
        from services.model_loader import directional_models, directional_scalers
        from services import get_feature_sequence_for_station
        from datetime import datetime
        import numpy as np
        
        stations_list = current_app.config.get('STATIONS', STATIONS)
        northbound = {}
        southbound = {}
        now = datetime.now()
        
        print(f"🕐 Time: {now.strftime('%H:%M:%S')}")
        print(f"📍 Processing {len(stations_list)} stations...")
        
        for i, station in enumerate(stations_list):
            # Get predictions using DIRECT model access (same as debug)
            north_pred = None
            south_pred = None
            
            # Northbound prediction
            model_key_north = f"{station}_Northbound"
            if model_key_north in directional_models:
                try:
                    sequence = get_feature_sequence_for_station(station, 'Northbound', now)
                    if sequence is not None and len(sequence) == 24:
                        feature_scaler = directional_scalers.get(f'{model_key_north}_feature')
                        target_scaler = directional_scalers.get(f'{model_key_north}_target')
                        
                        if feature_scaler and target_scaler:
                            scaled_sequence = feature_scaler.transform(sequence)
                            input_sequence = scaled_sequence.reshape(1, 24, -1)
                            pred_scaled = directional_models[model_key_north].predict(input_sequence, verbose=0)
                            north_pred = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                except Exception as e:
                    print(f"  ⚠️ Error predicting {station} Northbound: {e}")
            
            # Southbound prediction
            model_key_south = f"{station}_Southbound"
            if model_key_south in directional_models:
                try:
                    sequence = get_feature_sequence_for_station(station, 'Southbound', now)
                    if sequence is not None and len(sequence) == 24:
                        feature_scaler = directional_scalers.get(f'{model_key_south}_feature')
                        target_scaler = directional_scalers.get(f'{model_key_south}_target')
                        
                        if feature_scaler and target_scaler:
                            scaled_sequence = feature_scaler.transform(sequence)
                            input_sequence = scaled_sequence.reshape(1, 24, -1)
                            pred_scaled = directional_models[model_key_south].predict(input_sequence, verbose=0)
                            south_pred = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                except Exception as e:
                    print(f"  ⚠️ Error predicting {station} Southbound: {e}")
            
            # Use fallback if predictions failed
            if north_pred is None:
                north_pred = 50
            if south_pred is None:
                south_pred = 50
            
            # Print first few for debugging
            if i < 3:
                print(f"  📍 {station}: North={north_pred:.1f}%, South={south_pred:.1f}%")
            
            capacity = STATION_BASE_CAPACITY.get(station, 10000)
            
            def get_status(cong):
                if cong > 80:
                    return "SEVERE", "15-20 min"
                elif cong > 60:
                    return "CONGESTED", "10-15 min"
                elif cong > 30:
                    return "MODERATE", "5-10 min"
                else:
                    return "LIGHT", "2-5 min"
            
            north_status, north_wait = get_status(north_pred)
            south_status, south_wait = get_status(south_pred)
            
            northbound[station] = {
                "congestion": round(north_pred, 1),
                "wait_time": north_wait,
                "status": north_status,
                "ridership": int((north_pred/100) * capacity)
            }
            
            southbound[station] = {
                "congestion": round(south_pred, 1),
                "wait_time": south_wait,
                "status": south_status,
                "ridership": int((south_pred/100) * capacity)
            }
        
        print(f"✅ Returning data for {len(northbound)} stations")
        
        return jsonify({
            "northbound": northbound,
            "southbound": southbound,
            "timestamp": now.isoformat(),
            "model_version": "directional_2023-2024"
        })
        
    except Exception as e:
        print(f"❌ Error in live_map_directions_v2: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
# In routes/api_other.py

@api_other_bp.route('/live-map/directions')
def live_map_directions():
    """Get congestion data for both directions - uses DIRECT model access"""
    try:
        from flask import current_app
        from services.model_loader import directional_models, directional_scalers
        from services import get_feature_sequence_for_station
        from datetime import datetime
        import time
        
        stations_list = current_app.config.get('STATIONS', STATIONS)
        northbound = {}
        southbound = {}
        
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        current_time = hour + minute / 60
        
        OPERATING_START = 4.5
        OPERATING_END = 22.5
        
        # Get active overrides
        if 'overrides' not in current_app.config:
            current_app.config['overrides'] = {}
        
        current_timestamp = time.time()
        active_overrides = {}
        for key, override in current_app.config['overrides'].items():
            if override.get('expiry') is None or override.get('expiry', 0) > current_timestamp:
                active_overrides[key] = override
        
        def get_direct_prediction(station_name, direction):
            """Get prediction using DIRECT model access (same as forecast endpoint)"""
            model_key = f"{station_name}_{direction}"
            
            if model_key not in directional_models:
                # Fallback based on time of day
                hour_now = datetime.now().hour
                if 7 <= hour_now <= 9 or 17 <= hour_now <= 19:
                    return 65
                return 35
            
            try:
                sequence = get_feature_sequence_for_station(station_name, direction, now)
                if sequence is not None and len(sequence) == 24:
                    feature_scaler = directional_scalers.get(f'{model_key}_feature')
                    target_scaler = directional_scalers.get(f'{model_key}_target')
                    
                    if feature_scaler and target_scaler:
                        scaled_sequence = feature_scaler.transform(sequence)
                        input_sequence = scaled_sequence.reshape(1, 24, -1)
                        pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
                        prediction = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                        return max(0, min(100, prediction))
            except Exception as e:
                print(f"⚠️ Prediction error for {model_key}: {e}")
            
            # Fallback based on time of day
            hour_now = datetime.now().hour
            if 7 <= hour_now <= 9 or 17 <= hour_now <= 19:
                return 65
            return 35
        
        def get_wait_time(congestion):
            if congestion > 80:
                return "15-20 min"
            elif congestion > 60:
                return "10-15 min"
            elif congestion > 30:
                return "5-10 min"
            return "2-5 min"
        
        def get_status_text(congestion):
            if congestion > 80:
                return "SEVERELY CONGESTED"
            elif congestion > 60:
                return "CONGESTED"
            elif congestion > 30:
                return "MODERATE"
            return "LIGHT"
        
        # Check if operating
        if current_time < OPERATING_START or current_time >= OPERATING_END:
            for station in stations_list:
                north_override_key = f"{station}_northbound"
                south_override_key = f"{station}_southbound"
                
                northbound[station] = {
                    "congestion": 0, "wait_time": "CLOSED", "status": "CLOSED",
                    "ridership": 0, "overridden": north_override_key in active_overrides
                }
                southbound[station] = {
                    "congestion": 0, "wait_time": "CLOSED", "status": "CLOSED",
                    "ridership": 0, "overridden": south_override_key in active_overrides
                }
        else:
            for station in stations_list:
                north_override_key = f"{station}_northbound"
                south_override_key = f"{station}_southbound"
                
                # Check for overrides first
                if north_override_key in active_overrides:
                    north_congestion = active_overrides[north_override_key].get('congestion', 40)
                    is_north_overridden = True
                else:
                    north_congestion = get_direct_prediction(station, 'Northbound')
                    is_north_overridden = False
                
                if south_override_key in active_overrides:
                    south_congestion = active_overrides[south_override_key].get('congestion', 40)
                    is_south_overridden = True
                else:
                    south_congestion = get_direct_prediction(station, 'Southbound')
                    is_south_overridden = False
                
                capacity = STATION_BASE_CAPACITY.get(station, 10000)
                north_ridership = int((north_congestion / 100) * capacity) if north_congestion > 0 else 0
                south_ridership = int((south_congestion / 100) * capacity) if south_congestion > 0 else 0
                
                north_status = get_status_text(north_congestion)
                south_status = get_status_text(south_congestion)
                north_wait = get_wait_time(north_congestion)
                south_wait = get_wait_time(south_congestion)
                
                northbound[station] = {
                    "congestion": round(north_congestion, 1),
                    "wait_time": north_wait,
                    "status": north_status,
                    "ridership": north_ridership,
                    "overridden": is_north_overridden
                }
                
                southbound[station] = {
                    "congestion": round(south_congestion, 1),
                    "wait_time": south_wait,
                    "status": south_status,
                    "ridership": south_ridership,
                    "overridden": is_south_overridden
                }
        
        return jsonify({
            "northbound": northbound,
            "southbound": southbound,
            "timestamp": now.isoformat(),
            "is_operating": OPERATING_START <= current_time < OPERATING_END,
            "active_overrides": len(active_overrides)
        })
        
    except Exception as e:
        print(f"❌ Error in live_map_directions: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api_other_bp.route('/stations')
def get_stations():
    """Get list of all stations"""
    stations_list = get_stations_from_config()
    return jsonify({
        "stations": stations_list,
        "count": len(stations_list)
    })


@api_other_bp.route('/test')
def test_api():
    """Test if API is working"""
    return jsonify({
        "status": "ok",
        "message": "API is working",
        "time": datetime.now().isoformat(),
        "stations": get_stations_from_config()
    })

@api_other_bp.route('/alerts/count')
def alerts_count():
    try:
        stations_list = get_stations_from_config()
        critical_count = 0
        
        # Use the V2 directional API to get real congestion data
        try:
            from flask import current_app
            from services.model_loader import directional_models, directional_scalers
            from services import get_feature_sequence_for_station
            from datetime import datetime
            import numpy as np
            
            now = datetime.now()
            
            print(f"🔍 Checking congestion for {len(stations_list)} stations at {now.strftime('%H:%M')}")
            
            for station in stations_list:
                # Check northbound congestion
                north_cong = 0
                south_cong = 0
                
                try:
                    model_key_north = f"{station}_Northbound"
                    if model_key_north in directional_models:
                        sequence = get_feature_sequence_for_station(station, 'Northbound', now)
                        if sequence is not None and len(sequence) == 24:
                            feature_scaler = directional_scalers.get(f'{model_key_north}_feature')
                            target_scaler = directional_scalers.get(f'{model_key_north}_target')
                            if feature_scaler and target_scaler:
                                scaled_sequence = feature_scaler.transform(sequence)
                                input_sequence = scaled_sequence.reshape(1, 24, -1)
                                pred_scaled = directional_models[model_key_north].predict(input_sequence, verbose=0)
                                north_cong = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                except Exception as e:
                    print(f"⚠️ Error getting northbound for {station}: {e}")
                
                try:
                    model_key_south = f"{station}_Southbound"
                    if model_key_south in directional_models:
                        sequence = get_feature_sequence_for_station(station, 'Southbound', now)
                        if sequence is not None and len(sequence) == 24:
                            feature_scaler = directional_scalers.get(f'{model_key_south}_feature')
                            target_scaler = directional_scalers.get(f'{model_key_south}_target')
                            if feature_scaler and target_scaler:
                                scaled_sequence = feature_scaler.transform(sequence)
                                input_sequence = scaled_sequence.reshape(1, 24, -1)
                                pred_scaled = directional_models[model_key_south].predict(input_sequence, verbose=0)
                                south_cong = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
                except Exception as e:
                    print(f"⚠️ Error getting southbound for {station}: {e}")
                
                # Count if EITHER direction has critical congestion (>70%)
                max_cong = max(north_cong, south_cong)
                if max_cong > 70:
                    critical_count += 1
                    print(f"  🔴 CRITICAL: {station} - {max_cong:.1f}% (N:{north_cong:.1f}%, S:{south_cong:.1f}%)")
                elif max_cong > 30:
                    print(f"  🟡 MODERATE: {station} - {max_cong:.1f}%")
                else:
                    print(f"  🟢 LIGHT: {station} - {max_cong:.1f}%")
            
            print(f"✅ Total critical stations: {critical_count}")
            
        except Exception as e:
            print(f"❌ Error using models: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to using the V2 endpoint
            try:
                from flask import current_app
                with current_app.test_client() as client:
                    response = client.get('/api/live-map/directions/v2')
                    data = response.get_json()
                    
                    if data and 'northbound' in data and 'southbound' in data:
                        for station in stations_list:
                            north_cong = data['northbound'].get(station, {}).get('congestion', 0)
                            south_cong = data['southbound'].get(station, {}).get('congestion', 0)
                            max_cong = max(north_cong, south_cong)
                            if max_cong > 70:
                                critical_count += 1
                    print(f"✅ Fallback count: {critical_count}")
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
        
        total = critical_count
        display = str(total) if total < 10 else "9+"
        
        return jsonify({"count": total, "display": display})
        
    except Exception as e:
        print(f"❌ Error in alerts_count: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"count": 0, "display": "0"})

@api_other_bp.route('/alerts/list')
def alerts_list():
    """Get list of active alerts"""
    try:
        alerts = []
        now = datetime.now()
        hour = now.hour
        
        if 7 <= hour <= 9:
            alerts.append({
                "id": "rush-morning", "type": "rush_hour", "severity": "warning",
                "title": "Morning Rush Hour",
                "message": "Expect heavy traffic at North Ave, Quezon Ave, and Cubao stations",
                "time": now.strftime("%I:%M %p")
            })
        elif 17 <= hour <= 20:
            alerts.append({
                "id": "rush-evening", "type": "rush_hour", "severity": "warning",
                "title": "Evening Rush Hour",
                "message": "Expect heavy traffic at Ayala, Magallanes, and Taft stations",
                "time": now.strftime("%I:%M %p")
            })
        
        stations_list = get_stations_from_config()
        for station in stations_list:
            ridership = get_station_predictions_from_config(station)
            capacity = STATION_BASE_CAPACITY.get(station, 10000)
            congestion = min(100, int((ridership / capacity) * 100))
            
            if congestion > 80:
                alerts.append({
                    "id": f"critical-{station}", "type": "critical",
                    "title": f"Critical Congestion at {station}",
                    "message": f"Congestion at {congestion}%. Expect delays of 15-20 minutes.",
                    "time": now.strftime("%I:%M %p"), "severity": "critical"
                })
                break
        
        return jsonify(alerts)
    except Exception as e:
        return jsonify([])

@api_other_bp.route('/broadcasts/public')
def get_public_broadcasts():
    """Get all active broadcasts for public view (no login required)"""
    try:
        broadcasts = Broadcast.query.filter(
            Broadcast.is_active == True
        ).order_by(Broadcast.created_at.desc()).limit(20).all()
        
        result = []
        for broadcast in broadcasts:
            stations = json.loads(broadcast.stations) if broadcast.stations else []
            icon = typeIcons.get(broadcast.disruption_type, 'fa-bullhorn')
            
            if broadcast.severity == 'critical':
                icon_color = 'red'
            elif broadcast.severity == 'warning':
                icon_color = 'orange'
            else:
                icon_color = 'purple'
            
            # FIX: Only add prefix if disruption_type is not already in the title
            title_prefix = ''
            if broadcast.disruption_type == 'Train Breakdown':
                title_prefix = 'TRAIN BREAKDOWN'
            elif broadcast.disruption_type == 'Overcrowding':
                title_prefix = 'OVERCROWDING'
            elif broadcast.disruption_type == 'Maintenance':
                title_prefix = 'MAINTENANCE'
            elif broadcast.disruption_type == 'Signal Issue':
                title_prefix = 'SIGNAL ISSUE'
            elif broadcast.disruption_type == 'Gate Closure':
                title_prefix = 'GATE CLOSURE'
            else:
                title_prefix = 'SERVICE NOTICE'
            
            # Check if the prefix is already in the title to avoid duplication
            final_title = broadcast.title
            if not broadcast.title.startswith(title_prefix):
                final_title = f'{title_prefix}: {broadcast.title}'
            
            stations_text = ', '.join(stations[:3])
            if len(stations) > 3:
                stations_text += f' +{len(stations) - 3} more'
            
            result.append({
                'id': broadcast.id,
                'type': 'broadcast',
                'priority': 1,
                'icon': icon,
                'icon_color': icon_color,
                'title': final_title,
                'message': f'{broadcast.message} [Affected: {stations_text}]',
                'time': broadcast.created_at.isoformat(),
                'unread': True,
                'direction': getattr(broadcast, 'direction', 'both')
            })
        
        return jsonify({'success': True, 'broadcasts': result})
    except Exception as e:
        print(f"Error getting public broadcasts: {e}")
        return jsonify({'success': False, 'broadcasts': [], 'error': str(e)}), 500


@api_other_bp.route('/recommendation/<station_name>')
def get_recommendation(station_name):
    """Get travel recommendation for a station"""
    name = station_name.replace('%20', ' ')
    
    try:
        ridership = get_station_predictions_from_config(name)
        capacity = STATION_BASE_CAPACITY.get(name, 10000)
        congestion = min(100, int((ridership / capacity) * 100))
        
        now = datetime.now()
        hour = now.hour
        
        if congestion > 80:
            if 7 <= hour <= 9 or 17 <= hour <= 20:
                recommendation = "Consider postponing your trip until after rush hour"
            else:
                recommendation = "Severe congestion. Consider alternative routes or wait 30 minutes"
        elif congestion > 60:
            recommendation = "Heavy traffic. Allow extra 10-15 minutes for your journey"
        elif congestion > 30:
            recommendation = "Moderate traffic. Normal wait times expected"
        else:
            recommendation = "Light traffic. Good time to travel!"
        
        def get_best_time():
            hour = datetime.now().hour
            if 7 <= hour <= 9:
                return "10:00 AM - 3:00 PM"
            elif 17 <= hour <= 20:
                return "Before 5:00 PM or after 8:00 PM"
            else:
                return "Now is a good time to travel"
        
        return jsonify({
            "station": name, "congestion": congestion,
            "recommendation": recommendation, "best_time": get_best_time()
        })
    except Exception as e:
        return jsonify({
            "recommendation": "Normal operations. Trains running on schedule.",
            "best_time": "10:00 AM - 3:00 PM"
        })


@api_other_bp.route('/station-info/<station_name>')
def station_info(station_name):
    """Get detailed information about a station"""
    name = station_name.replace('%20', ' ')
    stations_list = get_stations_from_config()
    
    try:
        ridership = get_station_predictions_from_config(name)
        capacity = STATION_BASE_CAPACITY.get(name, 10000)
        congestion = min(100, int((ridership / capacity) * 100))
        
        station_idx = stations_list.index(name) if name in stations_list else 0
        
        prev_station = stations_list[station_idx - 1] if station_idx > 0 else None
        next_station = stations_list[station_idx + 1] if station_idx + 1 < len(stations_list) else None
        
        if congestion > 80:
            status = "SEVERELY CONGESTED"
            color = "critical"
            description = "Extremely crowded. Expect significant delays."
        elif congestion > 60:
            status = "CONGESTED"
            color = "congested"
            description = "Very busy. Allow extra time."
        elif congestion > 30:
            status = "MODERATE"
            color = "moderate"
            description = "Moderate crowds. Normal wait times."
        else:
            status = "LIGHT"
            color = "light"
            description = "Light traffic. Good time to travel."
        
        return jsonify({
            "station": name, "congestion": congestion, "status": status,
            "color": color, "description": description, "ridership": ridership,
            "capacity": capacity, "previous_station": prev_station,
            "next_station": next_station, "index": station_idx
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_other_bp.route('/historical-patterns')
def historical_patterns():
    """Get historical congestion patterns"""
    stations_list = get_stations_from_config()
    try:
        patterns = {}
        for station in stations_list:
            station_patterns = {}
            for hour in range(24):
                if 7 <= hour <= 9:
                    base = 75
                elif 17 <= hour <= 20:
                    base = 80
                elif 10 <= hour <= 16:
                    base = 55
                elif 21 <= hour <= 22:
                    base = 30
                elif 5 <= hour <= 6:
                    base = 20
                else:
                    base = 5
                
                if station in ["Cubao", "Ayala Ave", "North Ave"]:
                    base += 10
                elif station in ["Santolan", "Magallanes"]:
                    base -= 5
                
                station_patterns[hour] = min(100, max(0, base))
            patterns[station] = station_patterns
        return jsonify(patterns)
    except Exception as e:
        return jsonify({}), 500