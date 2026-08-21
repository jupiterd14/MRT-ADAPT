"""
LIVE API - Real-time Station Data
Purpose: Enriched station data for UI display
Use for: Live map, alerts, station info pages
NOT for: Pure ML predictions or forecasting
"""

# routes/api_other.py
import os
from flask import Blueprint, request, jsonify, session, current_app
from models import Broadcast, User, ActivityLog, Report
from datetime import datetime, timedelta
import json
import time
import traceback
import math
from config import Config
TESTING_MODE = False
from routes.api_predict import get_directional_prediction

api_other_bp = Blueprint('api_other', __name__)

# Station data (will be overridden by app.config)
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

# DOTr Official Platform Capacities (for congestion calculation)
MRT3_PLATFORM_CAPACITY = {
    "North Ave": 1142, "Quezon Ave": 1195, "Kamuning": 1364, "Cubao": 1747,
    "Santolan": 1306, "Ortigas": 1331, "Shaw Blvd": 1619, "Boni Ave": 1417,
    "Guadalupe": 1301, "Buendia": 1645, "Ayala Ave": 1222, "Magallanes": 1202,
    "Taft": 720
}


P95_CACHE = {}
for station in STATIONS:
    for direction in ['Northbound', 'Southbound']:
        P95_CACHE[f"{station}_{direction}"] = MRT3_PLATFORM_CAPACITY.get(station, 1000)
        
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


# ========== PERSISTENT OVERRIDE STORAGE (SAME AS OPERATOR) ==========
OVERRIDES_FILE = 'overrides.json'

def load_overrides():
    """Load overrides from file"""
    if os.path.exists(OVERRIDES_FILE):
        try:
            with open(OVERRIDES_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading overrides: {e}")
            return {}
    return {}
def get_active_overrides():
    """Get active overrides - filters out expired ones"""
    from flask import current_app
    import time
    
    # 1. Check in-memory config first
    overrides = current_app.config.get('overrides', {})
    
    # 2. If config is empty, load from file
    if not overrides:
        overrides_file = 'overrides.json'
        if os.path.exists(overrides_file):
            try:
                with open(overrides_file, 'r') as f:
                    overrides = json.load(f)
                if overrides:
                    current_app.config['overrides'] = overrides
            except Exception as e:
                print(f"Error loading overrides: {e}")
                return {}
    
    # ========== FILTER OUT EXPIRED OVERRIDES ==========
    now = time.time()
    active_overrides = {}
    expired_keys = []
    
    for key, override in overrides.items():
        expiry = override.get('expiry')
        if expiry is None or expiry > now:
            active_overrides[key] = override
        else:
            expired_keys.append(key)
            print(f"⏰ api_other.py: Override expired: {key} (expired at {datetime.fromtimestamp(expiry)})")
    
    # If any expired, update the cache
    if expired_keys:
        # Update app config with only active overrides
        current_app.config['overrides'] = active_overrides
        
        # Also save back to file (optional)
        if active_overrides:
            try:
                with open(OVERRIDES_FILE, 'w') as f:
                    json.dump(active_overrides, f, indent=2)
                print(f"🗑️ Removed {len(expired_keys)} expired overrides from file")
            except Exception as e:
                print(f"Error saving cleaned overrides: {e}")
    
    print(f"📄 api_other.py ACTIVE overrides: {list(active_overrides.keys())}")
    return active_overrides

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

def _get_congestion_from_prediction(pred_scaled, target_scaler, station_name):
    raw_value = float(pred_scaled[0][0]) if hasattr(pred_scaled, '__getitem__') else float(pred_scaled)
    
    if target_scaler is not None:
        try:
            passenger_count = float(target_scaler.inverse_transform([[raw_value]])[0][0])
        except Exception as e:
            capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
            passenger_count = raw_value * capacity * 1.5
    else:
        capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
        passenger_count = raw_value * capacity * 1.5
    
    # ========== FIX: Cap passenger_count at 0 ==========
    passenger_count = max(0, passenger_count)
    
    capacity = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
    capped_passengers = min(passenger_count, capacity)
    congestion = (capped_passengers / capacity * 100)
    congestion = max(0, min(congestion, 100))
    
    return congestion, passenger_count

@api_other_bp.route('/travel-prediction')
def travel_prediction():
    """Get predictions for travel planning with date/time"""
    date_param = request.args.get('date')
    time_param = request.args.get('time')
    station = request.args.get('station', 'North Ave')
    
    if not date_param or not time_param:
        return jsonify({
            "error": "Please provide date and time parameters",
            "example": "/api/travel-prediction?date=2025-06-17&time=08:00&station=North%20Ave"
        }), 400
    
    # Parse the date and time
    try:
        year, month, day = map(int, date_param.split('-'))
        hour, minute = map(int, time_param.split(':'))
        target_datetime = datetime(year, month, day, hour, minute)
    except:
        return jsonify({"error": "Invalid date or time format"}), 400
    
    # Get predictions using the same logic as V2
    try:
        from flask import current_app
        from services import get_feature_sequence_for_station
        
        directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
        directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
        
        north_pred = 0
        south_pred = 0
        north_passengers = 0
        south_passengers = 0
        
        # Northbound prediction
        model_key_north = f"{station}_Northbound"
        if model_key_north in directional_models:
            try:
                # Get features (ALREADY SCALED by feature_engineering)
                sequence = get_feature_sequence_for_station(station, 'Northbound', target_datetime)
                if sequence is not None and len(sequence) == 24:
                    target_scaler = directional_scalers.get(f'{model_key_north}_target')
                    if target_scaler:
                        # ========== FIX: DO NOT apply feature scaler again! ==========
                        # sequence is already scaled (feature scaler applied in get_feature_sequence_for_station)
                        input_sequence = sequence.reshape(1, 24, -1)
                        pred_scaled = directional_models[model_key_north].predict(input_sequence, verbose=0)
                        north_pred, north_passengers = _get_congestion_from_prediction(
                            pred_scaled, target_scaler, station
                        )
            except Exception as e:
                print(f"Error predicting northbound: {e}")
                north_pred = 50
        
        # Southbound prediction
        model_key_south = f"{station}_Southbound"
        if model_key_south in directional_models:
            try:
                # Get features (ALREADY SCALED by feature_engineering)
                sequence = get_feature_sequence_for_station(station, 'Southbound', target_datetime)
                if sequence is not None and len(sequence) == 24:
                    target_scaler = directional_scalers.get(f'{model_key_south}_target')
                    if target_scaler:
                        # ========== FIX: DO NOT apply feature scaler again! ==========
                        input_sequence = sequence.reshape(1, 24, -1)
                        pred_scaled = directional_models[model_key_south].predict(input_sequence, verbose=0)
                        south_pred, south_passengers = _get_congestion_from_prediction(
                            pred_scaled, target_scaler, station
                        )
            except Exception as e:
                print(f"Error predicting southbound: {e}")
                south_pred = 50
        
        avg_cong = (north_pred + south_pred) / 2
        
        def get_level(cong):
            if cong > 80: return "SEVERE"
            if cong > 60: return "CONGESTED"
            if cong > 40: return "MODERATE"
            if cong > 20: return "LIGHT"
            return "LIGHT"
        
        return jsonify({
            "station": station,
            "date": date_param,
            "time": time_param,
            "northbound": round(north_pred, 1),
            "southbound": round(south_pred, 1),
            "northbound_passengers": round(north_passengers, 0),
            "southbound_passengers": round(south_passengers, 0),
            "average_congestion": round(avg_cong, 1),
            "status": get_level(avg_cong),
            "timestamp": target_datetime.isoformat()
        })
        
    except Exception as e:
        print(f"Error in travel_prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
from extensions import cache

@api_other_bp.route('/live-map/directions/v2')
@cache.cached(timeout=60, key_prefix='live_map_v2')  # Cache for 60 seconds
def live_map_directions_v2():
    """Consistent with prediction API – uses percentile‑based congestion."""
    try:
        # ========== GET DATE/TIME PARAMETERS ==========
        date_param = request.args.get('date')
        time_param = request.args.get('time')
        
        if date_param and time_param:
            try:
                year, month, day = map(int, date_param.split('-'))
                hour, minute = map(int, time_param.split(':'))
                now = datetime(year, month, day, hour, minute)
                print(f"📅 Using custom date/time: {now.strftime('%Y-%m-%d %H:%M')}")
            except:
                now = Config.get_current_time()
                print(f"⚠️ Invalid date/time format, using current time")
        else:
            now = Config.get_current_time()
            print(f"🕐 Using current time: {now.strftime('%Y-%m-%d %H:%M')}")
        
        # ========== GET ACTIVE OVERRIDES ==========
        active_overrides = get_active_overrides()
        print(f"🔍 Active overrides: {list(active_overrides.keys())}")

        stations_list = current_app.config.get('STATIONS', STATIONS)
        northbound = {}
        southbound = {}

        for station in stations_list:
            # ========== CHECK OVERRIDES ==========
            north_override_key = f"{station}_northbound"
            south_override_key = f"{station}_southbound"
            
            is_north_overridden = north_override_key in active_overrides
            is_south_overridden = south_override_key in active_overrides

            # ----- Northbound -----
            if is_north_overridden:
                override = active_overrides[north_override_key]
                north_pred = override.get('congestion', 50)
                p95_north = P95_CACHE.get(f"{station}_Northbound", MRT3_PLATFORM_CAPACITY.get(station, 1000))
                north_passengers = int((north_pred / 100) * p95_north)
            else:
                north_pred = get_directional_prediction(station, 'Northbound', now)
                p95_north = P95_CACHE.get(f"{station}_Northbound", MRT3_PLATFORM_CAPACITY.get(station, 1000))
                north_passengers = int((north_pred / 100) * p95_north)

            # ----- Southbound -----
            if is_south_overridden:
                override = active_overrides[south_override_key]
                south_pred = override.get('congestion', 50)
                p95_south = P95_CACHE.get(f"{station}_Southbound", MRT3_PLATFORM_CAPACITY.get(station, 1000))
                south_passengers = int((south_pred / 100) * p95_south)
            else:
                south_pred = get_directional_prediction(station, 'Southbound', now)
                p95_south = P95_CACHE.get(f"{station}_Southbound", MRT3_PLATFORM_CAPACITY.get(station, 1000))
                south_passengers = int((south_pred / 100) * p95_south)

            # ----- Status & wait time -----
            def get_status(cong):
                if cong > 80:
                    return "SEVERE", "15-20 min"
                elif cong > 50:
                    return "CONGESTED", "10-15 min"
                elif cong > 25:
                    return "MODERATE", "5-10 min"
                else:
                    return "LIGHT", "2-5 min"

            north_status, north_wait = get_status(north_pred)
            south_status, south_wait = get_status(south_pred)

            northbound[station] = {
                "congestion": round(north_pred, 1),
                "wait_time": north_wait,
                "status": north_status,
                "ridership": north_passengers,
                "overridden": is_north_overridden
            }
            southbound[station] = {
                "congestion": round(south_pred, 1),
                "wait_time": south_wait,
                "status": south_status,
                "ridership": south_passengers,
                "overridden": is_south_overridden
            }

        # Check if MRT is closed for display
        current_time = now.hour + now.minute / 60
        is_closed = current_time < 4.5 or current_time >= 22.5
        
        return jsonify({
            "northbound": northbound,
            "southbound": southbound,
            "timestamp": now.isoformat(),
            "model_version": "percentile_based (p95)",
            "active_overrides": len(active_overrides),
            "requested_time": f"{date_param} {time_param}" if date_param and time_param else None,
            "is_operating": not is_closed,
            "cached": True  # Indicate this is cached
        })

    except Exception as e:
        print(f"❌ Error in live_map_directions_v2: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@api_other_bp.route('/debug/data-inspection')
def debug_data_inspection():
    """Inspect the actual data being loaded"""
    from services.feature_engineering import load_data_fast
    import pandas as pd
    import traceback
    
    try:
        df = load_data_fast()
        
        if df is None:
            return jsonify({"error": "No data loaded"})
        
        # Get basic info
        info = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "date_range": {
                "min": df['datetime'].min().isoformat(),
                "max": df['datetime'].max().isoformat()
            },
            "data_source": "Loaded from cache/preprocessed",
            "sample_data": df.head(5).to_dict()
        }
        
        # Check passenger counts
        if 'TotalPassenger' in df.columns:
            info["passenger_stats"] = {
                "mean": float(df['TotalPassenger'].mean()),
                "median": float(df['TotalPassenger'].median()),
                "max": float(df['TotalPassenger'].max()),
                "min": float(df['TotalPassenger'].min()),
                "by_hour": df.groupby('hour')['TotalPassenger'].mean().to_dict()
            }
        
        # Check if data looks realistic (should have thousands of passengers)
        if 'TotalPassenger' in df.columns:
            avg_passenger = df['TotalPassenger'].mean()
            if avg_passenger < 100:
                info["warning"] = "⚠️ Average passenger count is only {:.1f} - This seems too low! Expected thousands.".format(avg_passenger)
            elif avg_passenger < 1000:
                info["warning"] = "⚠️ Average passenger count is {:.1f} - This seems low. Expected 2000-5000 during peak.".format(avg_passenger)
            else:
                info["status"] = "✅ Passenger counts look realistic (avg: {:.0f})".format(avg_passenger)
        
        # Check for station data
        if 'StationEntry' in df.columns:
            info["stations_in_data"] = {
                "unique_entries": int(df['StationEntry'].nunique()),
                "station_values": sorted(df['StationEntry'].unique().tolist())
            }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@api_other_bp.route('/debug/data-analysis')
def debug_data_analysis():
    """Analyze data aggregation and suggest scaling factors"""
    from services.feature_engineering import load_data_fast
    import pandas as pd
    
    try:
        df = load_data_fast()
        
        if df is None:
            return jsonify({"error": "No data loaded"})
        
        analysis = {}
        
        # Check typical rush hour passenger counts
        for station_num, station_name in [(1, "North Ave"), (4, "Cubao"), (11, "Ayala Ave")]:
            # Get entry data for southbound (people entering station)
            station_df = df[df['StationEntry'] == station_num]
            
            if len(station_df) > 0:
                # Get rush hour data (7-9 AM and 5-7 PM)
                morning_rush = station_df[station_df['hour'].between(7, 9)]
                evening_rush = station_df[station_df['hour'].between(17, 19)]
                
                # Calculate average passengers per record
                avg_morning = morning_rush['TotalPassenger'].mean() if len(morning_rush) > 0 else 0
                avg_evening = evening_rush['TotalPassenger'].mean() if len(evening_rush) > 0 else 0
                
                # Estimate how many records per hour (frequency)
                records_per_hour = len(station_df[station_df['hour'] == 17]) / len(station_df['datetime'].dt.date.unique())
                
                # Suggested scaling factor to reach realistic hourly totals (3000-5000 passengers per hour)
                target_hourly = 3500  # Target hourly passengers during rush
                current_hourly = avg_evening * records_per_hour if records_per_hour > 0 else avg_evening
                suggested_scale = target_hourly / current_hourly if current_hourly > 0 else 1
                
                analysis[station_name] = {
                    "avg_morning_per_record": round(avg_morning, 1),
                    "avg_evening_per_record": round(avg_evening, 1),
                    "estimated_records_per_hour": round(records_per_hour, 1),
                    "estimated_hourly_passengers": round(current_hourly, 1),
                    "suggested_scaling_factor": round(suggested_scale, 1),
                    "total_records": len(station_df)
                }
        
        return jsonify({
            "analysis": analysis,
            "recommendation": "If your data is per-train or per-15min, apply scaling factor to TotalPassenger"
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@api_other_bp.route('/debug/list-files')
def debug_list_files():
    """List available data files"""
    import os
    
    data_paths = [
        'data (2022-2024)',
        '../data (2022-2024)',
        'data',
        '.'
    ]
    
    results = {}
    
    for path in data_paths:
        if os.path.exists(path):
            try:
                files = os.listdir(path)
                csv_files = [f for f in files if f.endswith('.csv')]
                parquet_files = [f for f in files if f.endswith('.parquet')]
                results[path] = {
                    "exists": True,
                    "csv_files": csv_files,
                    "parquet_files": parquet_files,
                    "total_files": len(csv_files) + len(parquet_files)
                }
            except Exception as e:
                results[path] = {"exists": True, "error": str(e)}
        else:
            results[path] = {"exists": False}
    
    return jsonify(results)


@api_other_bp.route('/debug/clear-data-cache')
def debug_clear_data_cache():
    """Clear the data cache and reload from raw CSV"""
    from services.feature_engineering import _DATA_CACHE, _STATION_DATA_CACHE
    import os
    
    # Clear caches
    _DATA_CACHE = None
    _STATION_DATA_CACHE.clear()
    
    # Delete preprocessed cache file
    preprocessed_path = 'data/preprocessed.parquet'
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)
        print(f"🗑️ Deleted {preprocessed_path}")
    
    # Reload data
    from services.feature_engineering import load_data_fast
    df = load_data_fast()
    
    if df is not None:
        return jsonify({
            "success": True,
            "message": "Cache cleared and data reloaded",
            "rows_loaded": len(df),
            "date_range": {
                "min": df['datetime'].min().isoformat(),
                "max": df['datetime'].max().isoformat()
            }
        })
    else:
        return jsonify({"success": False, "message": "Failed to reload data"})

@api_other_bp.route('/live-map/directions')
def live_map_directions():
    """Get congestion data for both directions - uses DIRECT model access with capacity-based scaling"""
    try:
        from flask import current_app
        from services.model_loader import directional_models, directional_scalers
        from services import get_feature_sequence_for_station
        import time
        
        stations_list = current_app.config.get('STATIONS', STATIONS)
        northbound = {}
        southbound = {}
        
        # Use Config.get_current_time() instead of datetime.now()
        now = Config.get_current_time()
        current_time = now.hour + now.minute / 60
        
        OPERATING_START = 4.5
        OPERATING_END = 22.5
        
        # Get active overrides
        if 'overrides' not in current_app.config:
            current_app.config['overrides'] = {}
        
        current_timestamp = time.time()
        active_overrides = get_active_overrides()
        for key, override in current_app.config['overrides'].items():
            if override.get('expiry') is None or override.get('expiry', 0) > current_timestamp:
                active_overrides[key] = override
        
        def get_direct_prediction(station_name, direction):
            """Get prediction using DIRECT model access with capacity-based scaling"""
            model_key = f"{station_name}_{direction}"
            
            if model_key not in directional_models:
                # Fallback based on time of day
                hour_now = now.hour
                if 7 <= hour_now <= 9 or 17 <= hour_now <= 19:
                    return 65, 0
                return 35, 0
            
            try:
                # Get features (ALREADY SCALED by feature_engineering)
                sequence = get_feature_sequence_for_station(station_name, direction, now)
                if sequence is not None and len(sequence) == 24:
                    target_scaler = directional_scalers.get(f'{model_key}_target')
                    
                    if target_scaler:
                        # ========== FIX: DO NOT apply feature scaler again! ==========
                        input_sequence = sequence.reshape(1, 24, -1)
                        pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
                        congestion, passengers = _get_congestion_from_prediction(
                            pred_scaled, target_scaler, station_name
                        )
                        return congestion, passengers
            except Exception as e:
                print(f"⚠️ Prediction error for {model_key}: {e}")
            
            # Fallback based on time of day
            hour_now = now.hour
            if 7 <= hour_now <= 9 or 17 <= hour_now <= 19:
                return 65, 0
            return 35, 0
        
        def get_wait_time(congestion):
            if congestion > 80:
                return "15-20 min"
            elif congestion > 50:
                return "10-15 min"
            elif congestion > 25:
                return "5-10 min"
            return "2-5 min"
        
        def get_status_text(congestion):
            if congestion > 80:
                return "SEVERELY CONGESTED"
            elif congestion > 50:
                return "CONGESTED"
            elif congestion > 25:
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
                    capacity = MRT3_PLATFORM_CAPACITY.get(station, 1000)
                    north_passengers = int((north_congestion / 100) * capacity)
                else:
                    north_congestion, north_passengers = get_direct_prediction(station, 'Northbound')
                    is_north_overridden = False
                
                if south_override_key in active_overrides:
                    south_congestion = active_overrides[south_override_key].get('congestion', 40)
                    is_south_overridden = True
                    capacity = MRT3_PLATFORM_CAPACITY.get(station, 1000)
                    south_passengers = int((south_congestion / 100) * capacity)
                else:
                    south_congestion, south_passengers = get_direct_prediction(station, 'Southbound')
                    is_south_overridden = False
                
                north_status = get_status_text(north_congestion)
                south_status = get_status_text(south_congestion)
                north_wait = get_wait_time(north_congestion)
                south_wait = get_wait_time(south_congestion)
                
                northbound[station] = {
                    "congestion": round(north_congestion, 1),
                    "wait_time": north_wait,
                    "status": north_status,
                    "ridership": int(north_passengers),
                    "overridden": is_north_overridden
                }
                
                southbound[station] = {
                    "congestion": round(south_congestion, 1),
                    "wait_time": south_wait,
                    "status": south_status,
                    "ridership": int(south_passengers),
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

@api_other_bp.route('/debug/raw-prediction')
def debug_raw_prediction():
    """Get raw model output for debugging with capacity-based scaling"""
    from services import get_feature_sequence_for_station
    
    date_param = request.args.get('date', '2025-01-15')
    time_param = request.args.get('time', '08:00')
    station = request.args.get('station', 'North Ave')
    direction = request.args.get('direction', 'Northbound')
    
    try:
        year, month, day = map(int, date_param.split('-'))
        hour, minute = map(int, time_param.split(':'))
        target_datetime = datetime(year, month, day, hour, minute)
    except:
        return jsonify({"error": "Invalid date/time format"}), 400
    
    # Get models
    directional_models = current_app.config.get('DIRECTIONAL_MODELS', {})
    directional_scalers = current_app.config.get('DIRECTIONAL_SCALERS', {})
    
    model_key = f"{station}_{direction}"
    if model_key not in directional_models:
        return jsonify({"error": f"Model {model_key} not found"}), 404
    
    try:
        # Get features
        features = get_feature_sequence_for_station(station, direction, target_datetime)
        if features is None:
            return jsonify({"error": "No features returned"}), 400
        
        # Get scalers
        feature_scaler = directional_scalers.get(f'{model_key}_feature')
        target_scaler = directional_scalers.get(f'{model_key}_target')
        
        if feature_scaler is None or target_scaler is None:
            return jsonify({"error": f"Scalers not found - feature: {feature_scaler is not None}, target: {target_scaler is not None}"}), 404
        
        # Scale features
        scaled_features = feature_scaler.transform(features)
        
        # Reshape for LSTM
        input_sequence = scaled_features.reshape(1, 24, -1)
        
        # Raw model output
        pred_scaled = directional_models[model_key].predict(input_sequence, verbose=0)
        raw_output = float(pred_scaled[0][0])
        
        # Get congestion and passenger count using capacity-based scaling
        congestion, pred_passengers = _get_congestion_from_prediction(
            pred_scaled, target_scaler, station
        )
        
        capacity = MRT3_PLATFORM_CAPACITY.get(station, 1000)
        
        # Calculate normalized output
        center = 0.5
        temperature = 0.8
        normalized = 1 / (1 + math.exp(-(raw_output - center) / temperature))
        
        return jsonify({
            "station": station,
            "direction": direction,
            "datetime": target_datetime.isoformat(),
            "raw_model_output": raw_output,
            "normalized_output": round(normalized, 4),
            "station_capacity": capacity,
            "predicted_passengers": round(pred_passengers, 0),
            "predicted_congestion": round(congestion, 1),
            "target_scaler_info": {
                "data_min": float(target_scaler.data_min_[0]) if hasattr(target_scaler, 'data_min_') else None,
                "data_max": float(target_scaler.data_max_[0]) if hasattr(target_scaler, 'data_max_') else None,
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# Keep all other endpoints unchanged (they don't use model predictions directly)
@api_other_bp.route('/debug/check-2025-data')
def debug_check_2025_data():
    """Check what data is loaded for 2025"""
    from services.feature_engineering import load_data_fast
    
    df = load_data_fast()
    if df is None:
        return jsonify({"error": "No data loaded"})
    
    # Check date range
    date_range = {
        "min": df['datetime'].min().isoformat(),
        "max": df['datetime'].max().isoformat()
    }
    
    # Check if 2025 data exists
    df_2025 = df[df['datetime'].dt.year == 2025]
    
    return jsonify({
        "date_range": date_range,
        "has_2025_data": len(df_2025) > 0,
        "rows_2025": len(df_2025),
        "sample_2025": df_2025.head(3).to_dict() if len(df_2025) > 0 else None,
        "total_rows": len(df)
    })
    
    
@api_other_bp.route('/live-map/debug')
def debug_live_map():
    """Debug endpoint to check model loading"""
    try:
        from services.model_loader import directional_models
        from flask import current_app
        from services import get_feature_sequence_for_station
        
        stations_list = current_app.config.get('STATIONS', STATIONS)
        
        # Use Config.get_current_time() instead of datetime.now()
        now = Config.get_current_time()
        
        result = {}
        for station in stations_list[:3]:
            model_key = f"{station}_Northbound"
            result[station] = {
                'model_exists': model_key in directional_models,
                'total_models': len(directional_models),
                'sample_models': list(directional_models.keys())[:3]
            }
            
            if model_key in directional_models:
                try:
                    sequence = get_feature_sequence_for_station(station, 'Northbound', now)
                    result[station]['sequence_shape'] = sequence.shape if sequence is not None else None
                except Exception as e:
                    result[station]['sequence_error'] = str(e)
        
        return jsonify({
            'models_loaded': len(directional_models),
            'station_check': result,
            'time': now.isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    # Use Config.get_current_time() instead of datetime.now()
    now = Config.get_current_time()
    return jsonify({
        "status": "ok",
        "message": "API is working",
        "time": now.isoformat(),
        "stations": get_stations_from_config()
    })

# Fix the /alerts/count endpoint in api_other.py

@api_other_bp.route('/alerts/count')
def alerts_count():
    """Get alert count - counts ALL congestion levels (Light, Moderate, Congested, Severe)"""
    try:
        stations_list = get_stations_from_config()
        severe_count = 0
        congested_count = 0
        moderate_count = 0
        light_count = 0
        station_statuses = {}
        
        # Use the V2 directional API to get real congestion data
        try:
            from flask import current_app
            from services.model_loader import directional_models, directional_scalers
            from services import get_feature_sequence_for_station
            import numpy as np
            
            now = Config.get_current_time()
            
            for station in stations_list:
                north_cong = 0
                south_cong = 0
                
                # Get northbound prediction
                try:
                    model_key_north = f"{station}_Northbound"
                    if model_key_north in directional_models:
                        sequence = get_feature_sequence_for_station(station, 'Northbound', now)
                        if sequence is not None and len(sequence) == 24:
                            target_scaler = directional_scalers.get(f'{model_key_north}_target')
                            if target_scaler:
                                input_sequence = sequence.reshape(1, 24, -1)
                                pred_scaled = directional_models[model_key_north].predict(input_sequence, verbose=0)
                                north_cong, _ = _get_congestion_from_prediction(
                                    pred_scaled, target_scaler, station
                                )
                except Exception as e:
                    print(f"⚠️ Error getting northbound for {station}: {e}")
                
                # Get southbound prediction
                try:
                    model_key_south = f"{station}_Southbound"
                    if model_key_south in directional_models:
                        sequence = get_feature_sequence_for_station(station, 'Southbound', now)
                        if sequence is not None and len(sequence) == 24:
                            target_scaler = directional_scalers.get(f'{model_key_south}_target')
                            if target_scaler:
                                input_sequence = sequence.reshape(1, 24, -1)
                                pred_scaled = directional_models[model_key_south].predict(input_sequence, verbose=0)
                                south_cong, _ = _get_congestion_from_prediction(
                                    pred_scaled, target_scaler, station
                                )
                except Exception as e:
                    print(f"⚠️ Error getting southbound for {station}: {e}")
                
                # Use AVERAGE congestion (matches dashboard)
                avg_cong = (north_cong + south_cong) / 2
                
                # ========== COUNT ALL CONGESTION LEVELS ==========
                # SEVERE: > 80%
                # CONGESTED: 61-80%
                # MODERATE: 31-60%
                # LIGHT: 0-30% (any congestion > 0)
                
                if avg_cong > 80:  # SEVERE
                    severe_count += 1
                    severity = 'severe'
                elif avg_cong > 50:  # CONGESTED
                    congested_count += 1
                    severity = 'congested'
                elif avg_cong > 25:  # MODERATE
                    moderate_count += 1
                    severity = 'moderate'
                elif avg_cong > 0:  # LIGHT (any congestion > 0)
                    light_count += 1
                    severity = 'light'
                else:  # NO CONGESTION (0%)
                    severity = 'none'
                
                station_statuses[station] = {
                    'congestion': round(avg_cong, 1),
                    'northbound': round(north_cong, 1),
                    'southbound': round(south_cong, 1),
                    'severity': severity
                }
            
            # ========== COUNT ALL STATIONS WITH ANY CONGESTION > 0 ==========
            alert_count = severe_count + congested_count + moderate_count + light_count
            print(f"🔔 Alert count (ALL congestion): {alert_count}")
            
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
                            avg_cong = (north_cong + south_cong) / 2
                            
                            if avg_cong > 80:
                                severe_count += 1
                                severity = 'severe'
                            elif avg_cong > 50:
                                congested_count += 1
                                severity = 'congested'
                            elif avg_cong > 25:
                                moderate_count += 1
                                severity = 'moderate'
                            elif avg_cong > 0:
                                light_count += 1
                                severity = 'light'
                            else:
                                severity = 'none'
                            
                            station_statuses[station] = {
                                'congestion': round(avg_cong, 1),
                                'northbound': round(north_cong, 1),
                                'southbound': round(south_cong, 1),
                                'severity': severity
                            }
                    alert_count = severe_count + congested_count + moderate_count + light_count
                    print(f"✅ Fallback breakdown - Severe: {severe_count}, Congested: {congested_count}, Moderate: {moderate_count}, Light: {light_count}")
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                alert_count = 0
        
        total = severe_count + congested_count + moderate_count + light_count
        display = str(total) if total < 10 else "9+"
        
        return jsonify({
            "count": total,  # ALL congestion levels
            "display": display,
            "breakdown": {
                "severe": severe_count,
                "congested": congested_count,
                "moderate": moderate_count,
                "light": light_count,
                "total_stations": len(stations_list)
            },
            "station_statuses": station_statuses
        })
        
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
        
        # Use Config.get_current_time() instead of datetime.now()
        now = Config.get_current_time()
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
            
            # FIX: Only show affected stations if there are any
            if stations:
                stations_text = ', '.join(stations[:3])
                if len(stations) > 3:
                    stations_text += f' +{len(stations) - 3} more'
                message = f'{broadcast.message} Affected Station(s): {stations_text}'
            else:
                # No affected stations - just show the message without the Affected part
                message = broadcast.message
            
            result.append({
                'id': broadcast.id,
                'type': 'broadcast',
                'priority': 1,
                'icon': icon,
                'icon_color': icon_color,
                'title': final_title,
                'message': message,
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
        
        # Use Config.get_current_time() instead of datetime.now()
        now = Config.get_current_time()
        hour = now.hour
        
        if congestion > 80:
            if 7 <= hour <= 9 or 17 <= hour <= 20:
                recommendation = "Consider postponing your trip until after rush hour"
            else:
                recommendation = "Severe congestion. Consider alternative routes or wait 30 minutes"
        elif congestion > 50:
            recommendation = "Heavy traffic. Allow extra 10-15 minutes for your journey"
        elif congestion > 25:
            recommendation = "Moderate traffic. Normal wait times expected"
        else:
            recommendation = "Light traffic. Good time to travel!"
        
        def get_best_time():
            hour = now.hour
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
        elif congestion > 50:
            status = "CONGESTED"
            color = "congested"
            description = "Very busy. Allow extra time."
        elif congestion > 25:
            status = "MODERATE"
            color = "moderate"
            description = "Moderate crowds. Normal wait times."
        else:
            status = "n"
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

@api_other_bp.route('/operator/review-report/<int:report_id>', methods=['POST'])
def review_report(report_id):
    """Operator reviews a flagged report - True Positive or False Positive"""
    try:
        data = request.get_json()
        verdict = data.get('verdict')
        reason = data.get('reason', '')
        reviewed_by = data.get('reviewed_by', 'Operator')
        
        if verdict not in ['true_positive', 'false_positive']:
            return jsonify({'success': False, 'error': 'Invalid verdict'}), 400
        
        # Get the report
        report = Report.query.get(report_id)
        if not report:
            return jsonify({'success': False, 'error': 'Report not found'}), 404
        
        # Update based on verdict
        if verdict == 'true_positive':
            # Keep the report, remove flags
            report.flagged = False
            report.flag_count = 0
            report.reviewed = True
            report.reviewed_by = reviewed_by
            report.reviewed_at = datetime.utcnow()
            report.review_notes = reason
            report.status = 'confirmed'
            
        else:  # false_positive
            # Archive the report
            report.flagged = False
            report.flag_count = 0
            report.reviewed = True
            report.reviewed_by = reviewed_by
            report.reviewed_at = datetime.utcnow()
            report.review_notes = reason
            report.archived = True
            report.archived_at = datetime.utcnow()
            report.archived_by = reviewed_by
            report.status = 'archived'
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Report {verdict}',
            'verdict': verdict,
            'report_id': report_id
        })
        
    except Exception as e:
        print(f"Error reviewing report: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    
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
        return jsonify({})
    
    
@api_other_bp.route('/debug/feature-debug/<station_name>')
def debug_features(station_name):
    """Debug what features are being passed to the model"""
    from services import get_feature_sequence_for_station
    from services.feature_engineering import get_station_dataframe
    
    now = Config.get_current_time()
    station = station_name.replace('%20', ' ')
    
    # Get the raw hourly data first
    hourly = get_station_dataframe(station, 'Northbound')
    
    if hourly is not None:
        # Get last 24 hours of actual data
        last_24 = hourly.tail(24)
        
        result = {
            "station": station,
            "current_time": now.isoformat(),
            "last_24_hours_passengers": last_24['TotalPassenger'].tolist(),
            "last_24_hours_congestion": last_24['congestion'].tolist(),
            "avg_passenger_last_24": last_24['TotalPassenger'].mean(),
            "feature_sequence_shape": None
        }
        
        # Get the features that will be passed to the model
        features = get_feature_sequence_for_station(station, 'Northbound', now)
        if features is not None:
            result["feature_sequence_shape"] = features.shape
            result["last_column_of_features"] = features[:, -1].tolist() if len(features) > 0 else []
        
        return jsonify(result)
    
    return jsonify({"error": "No data"})


@api_other_bp.route('/debug/check-lookback-data')
def debug_check_lookback_data():
    """Check what lookback data is being used for a specific date/time"""
    from services.feature_engineering import get_feature_sequence_for_station, get_station_dataframe
    
    date_param = request.args.get('date', '2025-01-15')
    time_param = request.args.get('time', '07:00')
    station = request.args.get('station', 'North Ave')
    direction = request.args.get('direction', 'Northbound')
    
    try:
        year, month, day = map(int, date_param.split('-'))
        hour, minute = map(int, time_param.split(':'))
        target_datetime = datetime(year, month, day, hour, minute)
    except:
        return jsonify({"error": "Invalid date/time format"}), 400
    
    # Get the raw hourly data for this station/direction
    hourly = get_station_dataframe(station, direction)
    
    if hourly is None:
        return jsonify({"error": "No hourly data"})
    
    # Determine lookback window
    if target_datetime.year >= 2025:
        lookback_end = target_datetime.replace(year=2024)
        start_lookback = lookback_end - timedelta(hours=24)
    else:
        start_lookback = target_datetime - timedelta(hours=24)
        lookback_end = target_datetime
    
    # Get lookback data
    lookback_data = hourly[(hourly.index >= start_lookback) & (hourly.index < lookback_end)]
    
    return jsonify({
        "station": station,
        "direction": direction,
        "target_datetime": target_datetime.isoformat(),
        "lookback_start": start_lookback.isoformat(),
        "lookback_end": lookback_end.isoformat(),
        "lookback_rows": len(lookback_data),
        "lookback_passenger_stats": {
            "min": float(lookback_data['TotalPassenger'].min()) if len(lookback_data) > 0 else 0,
            "max": float(lookback_data['TotalPassenger'].max()) if len(lookback_data) > 0 else 0,
            "mean": float(lookback_data['TotalPassenger'].mean()) if len(lookback_data) > 0 else 0
        },
        "sample_lookback": lookback_data.head(3).to_dict() if len(lookback_data) > 0 else None
    })