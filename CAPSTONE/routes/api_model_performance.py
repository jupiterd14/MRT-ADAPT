# routes/api_model_performance.py
"""
Model Performance Routes - LSTM Testing & Visualization
Handles: Manual predictions, batch uploads, metrics, chart data
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import pandas as pd
import numpy as np
import os
import pickle
import random
from werkzeug.utils import secure_filename
import services
from services.feature_engineering import get_feature_sequence_for_station
from services import get_directional_prediction
from services.model_loader import directional_models, directional_scalers

model_perf_bp = Blueprint('model_performance', __name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('test_results', exist_ok=True)

# Load per-direction max passengers
MAX_PATH = 'models_2022-2024_v5/per_direction_max_passengers.pkl'
PER_DIRECTION_MAX = {}
if os.path.exists(MAX_PATH):
    with open(MAX_PATH, 'rb') as f:
        PER_DIRECTION_MAX = pickle.load(f)
        print(f"✅ Loaded {len(PER_DIRECTION_MAX)} per-direction max values")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_data_file():
    """Find the data file in various possible locations"""
    possible_paths = [
        'data (2022-2024)/2025.csv',
        '../data (2022-2024)/2025.csv',
        '2025.csv',
        'data/2025.csv'
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# ============= CRITICAL FIX: COMPLETE FEATURE ENGINEERING (MATCHES TRAINING) =============
def add_cyclical_time_features(df):
    """Add cyclical time features (MUST MATCH TRAINING)"""
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    df['time_decimal'] = df['hour'] + df['minute'] / 60
    df['is_operating_hour'] = ((df['time_decimal'] >= 4.5) & (df['time_decimal'] < 23.0)).astype(np.int8)
    df['minute_normalized'] = df['minute'] / 60.0
    return df

def add_smart_operating_flags(df):
    """Add operating hour flags (CRITICAL for correct predictions)"""
    time_decimal = df['time_decimal']
    df['is_morning_rush'] = ((time_decimal >= 7.0) & (time_decimal <= 9.0)).astype(np.int8)
    df['is_evening_rush'] = ((time_decimal >= 17.0) & (time_decimal <= 19.0)).astype(np.int8)
    df['is_noon'] = ((time_decimal >= 12.0) & (time_decimal <= 13.0)).astype(np.int8)
    df['is_pre_opening'] = ((time_decimal >= 4.5) & (time_decimal < 5.0)).astype(np.int8)
    df['is_post_closing'] = ((time_decimal >= 22.5) & (time_decimal < 23.0)).astype(np.int8)
    minutes_until = (23.0 - time_decimal) * 60
    df['minutes_until_closing'] = minutes_until.clip(lower=0).astype(np.float32)
    minutes_since = (time_decimal - 4.5) * 60
    df['minutes_since_opening'] = minutes_since.clip(lower=0).astype(np.float32)
    df['time_normalized'] = ((time_decimal - 4.5) / (23.0 - 4.5)).clip(0, 1)
    return df

def smart_data_cleaner(df):
    time_decimal = df['time_decimal']
    passenger_count = df['TotalPassenger']
    df['is_maintenance_record'] = ((time_decimal < 5.0) & (passenger_count < 10)).astype(np.int8)
    df['is_extended_hours'] = ((time_decimal >= 22.0) & (time_decimal < 23.0) & (passenger_count >= 10)).astype(np.int8)
    df.loc[df['is_maintenance_record'] == 1, 'congestion'] = 0
    return df

def infer_direction_correct(row):
    """Infer direction based on entry and exit station numbers"""
    entry = row['StationEntry']
    exit_station = row['StationExit']
    if entry < exit_station:
        return 'Southbound'
    elif entry > exit_station:
        return 'Northbound'
    else:
        return 'Unknown'

def is_christmas_season(date):
    month_day = date.strftime('%m-%d')
    return (month_day >= '12-15') or (month_day <= '01-05')

def is_payday(date):
    return date.day in [15, 30, 31]

def is_friday(date):
    return date.weekday() == 4

# ============= CRITICAL FIX: CORRECT STATION FILTERING FOR TERMINALS =============
def get_station_data_for_direction(df, station_num, direction):
    """Get correct data for station and direction (handles terminals correctly)"""
    station_numbers_reverse = {
        1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
        5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
        9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
    }
    station_name = station_numbers_reverse.get(station_num, "Unknown")
    
    if station_name == "North Ave":
        if direction == 'Northbound':
            return df[df['StationExit'] == station_num].copy()
        else:
            return df[df['StationEntry'] == station_num].copy()
    elif station_name == "Taft":
        if direction == 'Northbound':
            return df[df['StationEntry'] == station_num].copy()  # FIXED: Entries at Taft for Northbound
        else:
            return df[df['StationExit'] == station_num].copy()   # FIXED: Exits at Taft for Southbound
    else:  # Middle stations
        if direction == 'Northbound':
            return df[df['StationExit'] == station_num].copy()
        else:
            return df[df['StationEntry'] == station_num].copy()

# ============= HEALTH CHECK =============
@model_perf_bp.route('/health', methods=['GET'])
def health_check():
    """Check if everything is configured correctly"""
    data_exists = find_data_file() is not None
    test_results_exist = os.path.exists('test_results')
    
    return jsonify({
        "success": True,
        "status": {
            "models_loaded": len(directional_models),
            "expected_models": 26,
            "scalers_loaded": len(directional_scalers),
            "data_file_exists": data_exists,
            "test_results_directory": test_results_exist,
            "ready_for_testing": len(directional_models) > 0 and data_exists
        }
    })

# ============= DEBUG ENDPOINTS =============
@model_perf_bp.route('/debug/check-models', methods=['GET'])
def debug_check_models():
    """Debug endpoint to check what models are accessible"""
    return jsonify({
        "models_loaded_in_services": len(directional_models),
        "model_keys": list(directional_models.keys()),
        "scalers_loaded": len(directional_scalers)
    })

@model_perf_bp.route('/debug/models', methods=['GET'])
def debug_models():
    return jsonify({
        "models_loaded": list(directional_models.keys()),
        "count": len(directional_models),
        "expected": 26
    })
    

@model_perf_bp.route('/debug/test-single-prediction/<station>/<direction>', methods=['GET'])
def debug_single_prediction(station, direction):
    """Test if get_directional_prediction works"""
    target_time = datetime.now()
    
    try:
        result = get_directional_prediction(
            station, direction, target_time,
            directional_models, directional_scalers,
            get_feature_sequence_for_station
        )
        
        return jsonify({
            "success": True,
            "station": station,
            "direction": direction,
            "target_time": target_time.isoformat(),
            "prediction": result
        })
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@model_perf_bp.route('/debug/why-no-predictions', methods=['GET'])
def debug_why_no_predictions():
    """Debug why predictions aren't being generated for all stations"""
    data_file = find_data_file()
    
    if not data_file:
        return jsonify({
            "total_models": len(directional_models),
            "error": "Data file not found"
        }), 404
    
    df = pd.read_csv(data_file)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df['minute'] = df['datetime'].dt.minute
    
    df = add_cyclical_time_features(df)
    df = add_smart_operating_flags(df)
    df['direction'] = df.apply(infer_direction_correct, axis=1)
    
    station_numbers_reverse = {
        1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
        5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
        9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
    }
    
    max_passengers = df['TotalPassenger'].quantile(0.99)
    results = {}
    
    for station_name in station_numbers_reverse.values():
        for direction in ['Northbound', 'Southbound']:
            model_key = f"{station_name}_{direction}"
            
            if model_key not in directional_models:
                results[f"{station_name}_{direction}"] = "SKIP: No model"
                continue
            
            station_num = [k for k, v in station_numbers_reverse.items() if v == station_name][0]
            
            # Use correct filtering for terminals
            station_df = get_station_data_for_direction(df, station_num, direction)
            station_df = station_df[station_df['direction'] == direction]
            
            if len(station_df) < 100:
                results[f"{station_name}_{direction}"] = f"SKIP: Only {len(station_df)} rows (need 100+)"
                continue
            
            station_df['hour_timestamp'] = station_df['datetime'].dt.floor('h')
            hourly = station_df.groupby('hour_timestamp').agg({
                'TotalPassenger': 'sum',
                'datetime': 'first'
            }).reset_index()
            hourly['actual_congestion'] = (hourly['TotalPassenger'] / max_passengers * 100).clip(0, 100)
            hourly = hourly.sort_values('hour_timestamp')
            
            if len(hourly) < 25:
                results[f"{station_name}_{direction}"] = f"SKIP: Only {len(hourly)} hourly groups (need 25+)"
                continue
            
            available_indices = list(range(24, len(hourly)))
            results[f"{station_name}_{direction}"] = f"OK: {len(hourly)} hourly groups, {len(available_indices)} testable indices"
    
    return jsonify({
        "total_models": len(directional_models),
        "station_analysis": results
    })

@model_perf_bp.route('/debug/model-path-check', methods=['GET'])
def debug_model_path_check():
    """Check if model path exists and what files are there"""
    model_path = 'models_2022-2024_v5'
    absolute_path = os.path.abspath(model_path)
    
    result = {
        "path_checked": model_path,
        "absolute_path": absolute_path,
        "path_exists": os.path.exists(model_path),
        "files_in_path": []
    }
    
    if os.path.exists(model_path):
        try:
            files = os.listdir(model_path)
            keras_files = [f for f in files if f.endswith('.keras')]
            pkl_files = [f for f in files if f.endswith('.pkl')]
            
            result["files_in_path"] = {
                "keras_models": keras_files[:10],
                "keras_count": len(keras_files),
                "pkl_files": pkl_files[:10],
                "pkl_count": len(pkl_files)
            }
        except Exception as e:
            result["error"] = str(e)
    else:
        alt_paths = [
            '../models_2022-2024_v5',
            '../../models_2022-2024_v5',
            'capstone/models_2022-2024_v5'
        ]
        result["alternative_paths"] = {}
        for alt in alt_paths:
            result["alternative_paths"][alt] = os.path.exists(alt)
    
    return jsonify(result)

@model_perf_bp.route('/debug/force-load-models', methods=['GET'])
def force_load_models():
    """Force load models directly"""
    import tensorflow as tf
    from tensorflow.keras.saving import register_keras_serializable
    
    @register_keras_serializable()
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    
    stations = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
                "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
                "Ayala Ave", "Magallanes", "Taft"]
    
    model_path_base = 'models_2022-2024_v5'
    
    loaded_models = {}
    loaded_scalers = {}
    
    results = []
    
    for station in stations:
        for direction in ['Northbound', 'Southbound']:
            model_key = f"{station}_{direction}"
            
            model_file = os.path.join(model_path_base, f'{model_key}_lstm_enhanced.keras')
            feature_scaler_file = os.path.join(model_path_base, f'{model_key}_feature_scaler.pkl')
            target_scaler_file = os.path.join(model_path_base, f'{model_key}_target_scaler.pkl')
            
            result = {
                "model_key": model_key,
                "model_exists": os.path.exists(model_file),
                "feature_scaler_exists": os.path.exists(feature_scaler_file),
                "target_scaler_exists": os.path.exists(target_scaler_file),
                "loaded": False,
                "error": None
            }
            
            if os.path.exists(model_file) and os.path.exists(feature_scaler_file):
                try:
                    print(f"Loading {model_key}...")
                    loaded_models[model_key] = tf.keras.models.load_model(
                        model_file,
                        custom_objects={'rmse': rmse}
                    )
                    
                    with open(feature_scaler_file, 'rb') as f:
                        loaded_scalers[f'{model_key}_feature'] = pickle.load(f)
                    
                    if os.path.exists(target_scaler_file):
                        with open(target_scaler_file, 'rb') as f:
                            loaded_scalers[f'{model_key}_target'] = pickle.load(f)
                    
                    result["loaded"] = True
                    print(f"✅ Loaded {model_key}")
                    
                except Exception as e:
                    result["error"] = str(e)
                    print(f"❌ Error loading {model_key}: {e}")
            else:
                result["error"] = "Missing files"
            
            results.append(result)
    
    from services import model_loader
    model_loader.directional_models = loaded_models
    model_loader.directional_scalers = loaded_scalers
    
    return jsonify({
        "success": True,
        "models_loaded": len(loaded_models),
        "scalers_loaded": len(loaded_scalers),
        "details": results[:5],
        "model_keys": list(loaded_models.keys())
    })

@model_perf_bp.route('/debug/loaded-models', methods=['GET'])
def debug_loaded_models():
    """Check which models are actually loaded"""
    return jsonify({
        "models_loaded": list(directional_models.keys()),
        "count": len(directional_models)
    })

@model_perf_bp.route('/debug/test-prediction', methods=['GET'])
def debug_test_prediction():
    """Test a single prediction with detailed logging"""
    from services.feature_engineering import get_feature_sequence_for_station
    
    station = "North Ave"
    direction = "Northbound"
    target_time = pd.to_datetime("2025-01-06 09:00:00")
    
    print(f"\n{'='*50}")
    print(f"DEBUG PREDICTION: {station} {direction} at {target_time}")
    print(f"{'='*50}")
    
    model_key = f"{station}_{direction}"
    print(f"1. Model exists: {model_key in directional_models}")
    
    if model_key not in directional_models:
        return jsonify({"error": f"Model {model_key} not found"})
    
    print(f"2. Getting feature sequence...")
    sequence = get_feature_sequence_for_station(station, direction, target_time)
    
    if sequence is None:
        print(f"   ❌ Sequence is None")
        return jsonify({"error": "Sequence is None"})
    
    print(f"   ✅ Sequence shape: {sequence.shape}")
    print(f"   Sequence range: [{sequence.min():.3f}, {sequence.max():.3f}]")
    
    print(f"3. Getting prediction...")
    result = get_directional_prediction(
        station, direction, target_time,
        directional_models, directional_scalers,
        get_feature_sequence_for_station
    )
    
    print(f"4. Prediction result: {result}")
    
    return jsonify({
        "station": station,
        "direction": direction,
        "target_time": str(target_time),
        "sequence_shape": sequence.shape if sequence is not None else None,
        "prediction": result,
        "model_exists": model_key in directional_models
    })

@model_perf_bp.route('/debug/prediction-failures', methods=['GET'])
def debug_prediction_failures():
    """Debug why predictions are failing"""
    from services.feature_engineering import get_feature_sequence_for_station
    import traceback
    
    results = []
    test_stations = ["North Ave", "Cubao", "Taft"]
    test_dates = ["2025-02-07 14:00:00", "2025-02-14 12:00:00"]
    
    for station in test_stations:
        for direction in ["Northbound", "Southbound"]:
            model_key = f"{station}_{direction}"
            if model_key not in directional_models:
                results.append({
                    "station": station,
                    "direction": direction,
                    "error": f"No model found for {model_key}"
                })
                continue
            
            for target_time_str in test_dates:
                target_time = pd.to_datetime(target_time_str)
                try:
                    result = get_directional_prediction(
                        station, direction, target_time,
                        directional_models, directional_scalers,
                        get_feature_sequence_for_station
                    )
                    
                    results.append({
                        "station": station,
                        "direction": direction,
                        "target_time": target_time_str,
                        "success": result is not None,
                        "prediction": result,
                        "error": None if result else "Prediction returned None"
                    })
                except Exception as e:
                    results.append({
                        "station": station,
                        "direction": direction,
                        "target_time": target_time_str,
                        "success": False,
                        "prediction": None,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
    
    return jsonify({
        "total_tests": len(results),
        "results": results,
        "summary": {
            "successful": len([r for r in results if r.get('success')]),
            "failed": len([r for r in results if not r.get('success')])
        }
    })

@model_perf_bp.route('/debug/taft-check', methods=['GET'])
def check_taft_models():
    """Check if Taft models are loaded"""
    taft_models = {
        "Taft_Northbound": "Taft_Northbound" in directional_models,
        "Taft_Southbound": "Taft_Southbound" in directional_models
    }
    
    return jsonify({
        "taft_models_loaded": taft_models,
        "all_models": list(directional_models.keys()),
        "total_models": len(directional_models)
    })

# ============= PERFORMANCE METRICS =============
@model_perf_bp.route('/model/performance/metrics', methods=['GET'])
def get_performance_metrics():
    """Calculate metrics from existing test results"""
    try:
        if not os.path.exists('test_results'):
            return jsonify({
                "success": True,
                "data": {"message": "No test results yet", "total_predictions": 0}
            })
        
        all_results = []
        for filename in os.listdir('test_results'):
            if filename.endswith('_results.csv') and not filename.startswith('full_'):
                try:
                    df = pd.read_csv(os.path.join('test_results', filename))
                    
                    # 🔧 FIX: Convert string columns to numeric where needed
                    numeric_cols = ['predicted', 'actual', 'absolute_error', 'percentage_error']
                    for col in numeric_cols:
                        if col in df.columns:
                            # Convert to numeric, coerce errors to NaN
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Drop rows with NaN in critical columns
                    if 'absolute_error' in df.columns:
                        df = df.dropna(subset=['absolute_error'])
                    
                    if len(df) > 0:
                        all_results.append(df)
                        
                except Exception as e:
                    print(f"⚠️ Error reading {filename}: {e}")
                    continue
        
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            
            # Calculate verdict counts if verdict column exists
            verdict_counts = {}
            if 'verdict' in combined.columns:
                verdict_counts = combined['verdict'].value_counts().to_dict()
            else:
                verdict_counts = {
                    'EXCELLENT': 0, 'GOOD': 0, 'OKAY': 0, 'NEEDS_IMPROVEMENT': 0
                }
            
            # Calculate per-station metrics
            per_station = {}
            if 'station' in combined.columns and 'direction' in combined.columns and 'absolute_error' in combined.columns:
                for station in combined['station'].unique():
                    station_df = combined[combined['station'] == station]
                    
                    north_df = station_df[station_df['direction'] == 'Northbound']
                    north_mae = north_df['absolute_error'].mean() if len(north_df) > 0 else None
                    
                    south_df = station_df[station_df['direction'] == 'Southbound']
                    south_mae = south_df['absolute_error'].mean() if len(south_df) > 0 else None
                    
                    per_station[station] = {
                        "northbound": {"mae": round(north_mae, 1) if north_mae is not None else None},
                        "southbound": {"mae": round(south_mae, 1) if south_mae is not None else None}
                    }
            
            metrics = {
                "overall_mae": round(combined['absolute_error'].mean(), 2) if 'absolute_error' in combined.columns else 0,
                "overall_mape": round(combined['percentage_error'].mean(), 2) if 'percentage_error' in combined.columns else 0,
                "total_predictions": len(combined),
                "total_tests": len(combined),
                "avg_absolute_error": round(combined['absolute_error'].mean(), 2) if 'absolute_error' in combined.columns else 0,
                "avg_percentage_error": round(combined['percentage_error'].mean(), 2) if 'percentage_error' in combined.columns else 0,
                "min_error": round(combined['absolute_error'].min(), 2) if 'absolute_error' in combined.columns else 0,
                "max_error": round(combined['absolute_error'].max(), 2) if 'absolute_error' in combined.columns else 0,
                "excellent_predictions": verdict_counts.get('EXCELLENT', 0),
                "good_predictions": verdict_counts.get('GOOD', 0),
                "okay_predictions": verdict_counts.get('OKAY', 0),
                "needs_improvement": verdict_counts.get('NEEDS_IMPROVEMENT', 0),
                "per_station": per_station,
                "last_evaluated": combined['timestamp'].max() if 'timestamp' in combined.columns else None
            }
        else:
            metrics = {"total_predictions": 0, "message": "No valid test results found"}
        
        return jsonify({
            "success": True,
            "data": metrics
        })
    except Exception as e:
        import traceback
        print(f"❌ Error in get_performance_metrics: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@model_perf_bp.route('/model/chart/data', methods=['GET'])
def get_chart_data():
    """Get data for prediction vs actual chart"""
    try:
        station = request.args.get('station', 'all')
        direction = request.args.get('direction', 'both')
        limit = request.args.get('limit', 20, type=int)
        
        if not os.path.exists('test_results'):
            return jsonify({"success": True, "data": {"labels": [], "predicted": [], "actual": [], "errors": []}})
        
        all_results = []
        for filename in os.listdir('test_results'):
            if filename.endswith('_results.csv') and not filename.startswith('full_'):
                try:
                    df = pd.read_csv(os.path.join('test_results', filename))
                    
                    # 🔧 FIX: Convert string columns to numeric
                    if 'predicted' in df.columns:
                        df['predicted'] = pd.to_numeric(df['predicted'], errors='coerce')
                    if 'actual' in df.columns:
                        df['actual'] = pd.to_numeric(df['actual'], errors='coerce')
                    if 'absolute_error' in df.columns:
                        df['absolute_error'] = pd.to_numeric(df['absolute_error'], errors='coerce')
                    
                    # Drop rows with NaN predictions
                    df = df.dropna(subset=['predicted', 'actual'])
                    
                    if len(df) > 0:
                        all_results.append(df)
                except Exception as e:
                    print(f"⚠️ Error reading {filename}: {e}")
                    continue
        
        if not all_results:
            return jsonify({"success": True, "data": {"labels": [], "predicted": [], "actual": [], "errors": []}})
        
        combined = pd.concat(all_results, ignore_index=True)
        
        # Filter by station and direction
        if station != "all" and 'station' in combined.columns:
            combined = combined[combined['station'] == station]
        if direction != "both" and 'direction' in combined.columns:
            combined = combined[combined['direction'].str.lower() == direction.lower()]
        
        # Sort by target_time if available
        if 'target_time' in combined.columns:
            combined['target_time_dt'] = pd.to_datetime(combined['target_time'])
            combined = combined.sort_values('target_time_dt')
            labels = combined['target_time'].tolist()
        else:
            labels = [f"Test {i+1}" for i in range(len(combined))]
        
        # Get predictions and actuals
        predicted = combined['predicted'].tolist() if 'predicted' in combined.columns else []
        actual = combined['actual'].tolist() if 'actual' in combined.columns else []
        
        # Calculate errors
        errors = [abs(p - a) for p, a in zip(predicted, actual)] if predicted and actual else []
        
        # Apply limit
        if limit and limit > 0 and len(labels) > limit:
            labels = labels[-limit:]
            predicted = predicted[-limit:]
            actual = actual[-limit:]
            errors = errors[-limit:]
        
        return jsonify({
            "success": True,
            "data": {
                "labels": labels,
                "predicted": predicted,
                "actual": actual,
                "errors": errors
            }
        })
    except Exception as e:
        import traceback
        print(f"❌ Error in get_chart_data: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False, 
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

# ============= STATION ENDPOINTS =============
@model_perf_bp.route('/model/stations', methods=['GET'])
def get_available_stations():
    """Get list of stations with trained models"""
    try:
        stations = set()
        for model_key in directional_models.keys():
            station = model_key.split('_')[0]
            stations.add(station)
        
        return jsonify({
            "success": True,
            "data": sorted(list(stations))  # Sort for consistent display
        })
    except Exception as e:
        import traceback
        print(f"❌ Error in get_available_stations: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
        


@model_perf_bp.route('/debug/station-details-debug', methods=['GET'])
def debug_station_details():
    """Debug station details endpoint"""
    try:
        station = request.args.get('station', 'North Ave')
        direction = request.args.get('direction', 'Northbound')
        
        filename = f"test_results/{station}_{direction}_results.csv"
        
        result = {
            "station": station,
            "direction": direction,
            "filename": filename,
            "file_exists": os.path.exists(filename),
            "error": None,
            "data_types": None,
            "sample": None
        }
        
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                result["rows"] = int(len(df))  # Convert to native int
                result["columns"] = list(df.columns)
                
                # Check data types before conversion (convert to strings for JSON)
                result["data_types_before"] = {
                    "predicted": str(df['predicted'].dtype) if 'predicted' in df.columns else None,
                    "actual": str(df['actual'].dtype) if 'actual' in df.columns else None,
                    "absolute_error": str(df['absolute_error'].dtype) if 'absolute_error' in df.columns else None,
                }
                
                # Check for string values
                if 'predicted' in df.columns:
                    sample_pred = df['predicted'].head(3).tolist()
                    # Convert any numpy types to native Python types
                    sample_pred = [float(x) if hasattr(x, 'item') else x for x in sample_pred]
                    result["sample_predicted"] = sample_pred
                    result["predicted_are_strings"] = any(isinstance(x, str) for x in sample_pred)
                
                # Try the conversion
                if 'predicted' in df.columns:
                    df['predicted'] = pd.to_numeric(df['predicted'], errors='coerce')
                    result["after_conversion"] = {
                        "has_nan": bool(df['predicted'].isna().any()),  # Convert to bool
                        "nan_count": int(df['predicted'].isna().sum()),  # Convert to int
                        "dtype": str(df['predicted'].dtype)
                    }
                
                result["success"] = True
                
            except Exception as e:
                result["error"] = str(e)
                import traceback
                result["traceback"] = traceback.format_exc()
        
        # Convert any numpy types in the entire result
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        result = convert_numpy_types(result)
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
        
@model_perf_bp.route('/model/station-details', methods=['GET'])
def get_station_details():
    """Get detailed test results for a specific station and direction"""
    try:
        # Get parameters
        station = request.args.get('station', '').strip()
        direction = request.args.get('direction', '').strip().capitalize()
        
        print(f"📊 Station Details Request: station='{station}', direction='{direction}'")
        
        if not station or not direction:
            return jsonify({"success": False, "error": "Missing station or direction"}), 400
        
        # Validate direction
        if direction not in ['Northbound', 'Southbound']:
            return jsonify({"success": False, "error": f"Invalid direction: {direction}"}), 400
        
        # Build filename
        filename = f"test_results/{station}_{direction}_results.csv"
        
        if not os.path.exists(filename):
            print(f"   File not found: {filename}")
            return jsonify({"success": True, "data": []}), 200
        
        # Read CSV - try with header first
        try:
            df = pd.read_csv(filename)
            print(f"   Columns found: {list(df.columns)}")
        except Exception as e:
            print(f"   Error reading CSV: {e}")
            return jsonify({"success": False, "error": f"Error reading file: {e}"}), 500
        
        if df.empty:
            return jsonify({"success": True, "data": []}), 200
        
        # Check if columns are correct - if not, try to fix
        if 'target_time' in df.columns:
            # Check if target_time contains valid dates
            sample = df['target_time'].iloc[0] if len(df) > 0 else None
            if sample and isinstance(sample, str) and sample in ['Northbound', 'Southbound']:
                # Columns are shifted - rebuild the dataframe
                print(f"   Detected shifted columns! Rebuilding...")
                # Try to read with proper column names
                proper_columns = ['station', 'direction', 'target_time', 'predicted', 'actual', 
                                  'total_passengers', 'station_max', 'absolute_error', 
                                  'percentage_error', 'verdict', 'timestamp']
                
                # Read raw data without header
                raw_df = pd.read_csv(filename, header=None)
                if len(raw_df.columns) >= len(proper_columns):
                    raw_df.columns = proper_columns[:len(raw_df.columns)]
                    df = raw_df
                    print(f"   Rebuilt with {len(df.columns)} columns")
        
        # Now process the data
        # Convert numeric columns
        numeric_cols = ['predicted', 'actual', 'absolute_error', 'percentage_error', 
                       'total_passengers', 'station_max']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by target_time if it contains valid dates
        if 'target_time' in df.columns:
            try:
                # Try to convert to datetime
                df['target_time_dt'] = pd.to_datetime(df['target_time'], errors='coerce')
                # Drop rows with invalid dates
                df = df.dropna(subset=['target_time_dt'])
                df = df.sort_values('target_time_dt', ascending=False)
                print(f"   Sorted by target_time, {len(df)} valid records")
            except Exception as e:
                print(f"   Could not parse target_time as dates: {e}")
                # Fall back to timestamp if available
                if 'timestamp' in df.columns:
                    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    df = df.dropna(subset=['timestamp_dt'])
                    df = df.sort_values('timestamp_dt', ascending=False)
                    print(f"   Sorted by timestamp instead")
        
        # Select columns to return
        return_cols = []
        for col in ['target_time', 'predicted', 'actual', 'absolute_error', 'verdict']:
            if col in df.columns:
                return_cols.append(col)
        
        # Get top 50 records
        results = df[return_cols].head(50).to_dict('records')
        
        # Clean NaN values for JSON
        for record in results:
            for key, value in list(record.items()):
                if pd.isna(value):
                    record[key] = None
                elif hasattr(value, 'item'):
                    record[key] = value.item()
        
        print(f"   Returning {len(results)} records")
        
        return jsonify({
            "success": True,
            "data": results
        })
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
        

# ============= PREDICTION ENDPOINTS =============
@model_perf_bp.route('/model/predict/single', methods=['POST'])
def predict_single():
    """Run a single manual prediction and return actual vs predicted"""
    try:
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Content-Type must be application/json"
            }), 415
        
        data = request.get_json()
        
        station = data.get('station')
        direction_raw = data.get('direction')
        target_datetime = data.get('datetime')
        
        if not all([station, direction_raw, target_datetime]):
            return jsonify({
                "success": False,
                "error": "Missing required fields: station, direction, datetime"
            }), 400
        
        direction = direction_raw.capitalize()
        target_time = pd.to_datetime(target_datetime)
        
        print(f"🔮 Manual prediction request: {station} {direction} at {target_time}")
        
        use_test_max = request.args.get('use_test_max', 'false').lower() == 'true'
        
        # 1. Get prediction from model
        pred_congestion = get_directional_prediction(
            station, direction, target_time,
            directional_models, directional_scalers,
            get_feature_sequence_for_station
        )
        
        if pred_congestion is None:
            return jsonify({
                "success": False,
                "error": f"Prediction failed. Model may not exist for {station} {direction}"
            }), 400
        
        # 2. Try to get actual congestion from historical data
        actual_congestion = None
        data_file = find_data_file()
        if data_file:
            try:
                df = pd.read_csv(data_file)
                df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                df['direction'] = df.apply(infer_direction_correct, axis=1)
                
                station_numbers_reverse = {
                    1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
                    5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
                    9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
                }
                station_num = [k for k, v in station_numbers_reverse.items() if v == station][0]
                
                # Use correct filtering for terminals
                station_df = get_station_data_for_direction(df, station_num, direction)
                station_df = station_df[station_df['direction'] == direction]
                station_df['hour_timestamp'] = station_df['datetime'].dt.floor('h')
                
                matching = station_df[station_df['hour_timestamp'] == target_time.floor('h')]
                if not matching.empty:
                    total_pass = matching['TotalPassenger'].sum()
                    model_key = f"{station}_{direction}"
                    
                
                    station_max = PER_DIRECTION_MAX.get(model_key, 100)
                    
                    actual_congestion = min(100, total_pass / station_max * 100)
                    actual_congestion = round(actual_congestion, 1)
                    
            except Exception as e:
                print(f"Warning: Could not fetch actual congestion: {e}")
        
        response = {
            "success": True,
            "prediction": {
                "predicted": round(pred_congestion, 1),
                "station": station,
                "direction": direction,
                "target_time": target_datetime
            }
        }
        if actual_congestion is not None:
            response["prediction"]["actual"] = actual_congestion
            response["prediction"]["error"] = round(abs(pred_congestion - actual_congestion), 1)
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@model_perf_bp.route('/debug/inspect-csv', methods=['GET'])
def inspect_csv_files():
    """Inspect the structure of CSV files in test_results"""
    import os
    import pandas as pd
    
    if not os.path.exists('test_results'):
        return jsonify({"error": "test_results folder not found"})
    
    results = {}
    for filename in os.listdir('test_results'):
        if filename.endswith('.csv'):
            filepath = os.path.join('test_results', filename)
            try:
                df = pd.read_csv(filepath)
                results[filename] = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "sample": df.head(2).to_dict('records') if len(df) > 0 else []
                }
            except Exception as e:
                results[filename] = {"error": str(e)}
    
    return jsonify(results)


# ============= RUN AUTO TESTS (FIXED WITH TERMINAL STATIONS) =============
@model_perf_bp.route('/model/run-auto-tests', methods=['POST'])
def run_auto_tests():
    """FIXED: Properly handle directional predictions for 2025 data including terminals"""
    try:
        use_test_max = request.args.get('use_test_max', 'false').lower() == 'true'
        
        if not directional_models:
            return jsonify({"success": False, "error": "No models loaded"}), 500
        
        data_file = find_data_file()
        if not data_file:
            return jsonify({"success": False, "error": "Data file not found"}), 404
        
        df = pd.read_csv(data_file)
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday
        df['month'] = df['datetime'].dt.month
        df['minute'] = df['datetime'].dt.minute
        
        # Add ALL features (CRITICAL)
        df = add_cyclical_time_features(df)
        df = add_smart_operating_flags(df)
        df = smart_data_cleaner(df)
        
        # Add date-based features
        df['is_weekend'] = (df['datetime'].dt.weekday >= 5).astype(np.int8)
        df['is_christmas_season'] = df['datetime'].apply(is_christmas_season).astype(np.int8)
        df['is_payday'] = df['datetime'].apply(is_payday).astype(np.int8)
        df['is_friday'] = df['datetime'].apply(is_friday).astype(np.int8)
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(np.int8)
        
        # Holidays 2025
        holidays_2025 = [
            '2025-01-01', '2025-04-09', '2025-04-17', '2025-04-18',
            '2025-05-01', '2025-06-12', '2025-08-25', '2025-11-30',
            '2025-12-25', '2025-12-30', '2025-12-31'
        ]
        df['is_holiday'] = df['datetime'].dt.date.astype(str).isin(holidays_2025).astype(np.int8)
        df['is_special_event'] = 0
        
        # Direction inference
        df['direction'] = df.apply(infer_direction_correct, axis=1)
        
        station_numbers_reverse = {
            1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
            5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
            9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
        }
        
        FEATURE_COLS = [
            'hour', 'weekday', 'month',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
            'is_operating_hour', 'is_morning_rush', 'is_evening_rush', 'is_noon',
            'is_pre_opening', 'is_post_closing',
            'minutes_until_closing', 'minutes_since_opening', 'time_normalized', 'minute_normalized',
            'is_weekend', 'is_holiday', 'is_special_event', 'is_christmas_season', 'is_payday', 'is_friday',
            'is_rush_hour', 'is_maintenance_record', 'is_extended_hours', 'congestion'
        ]
        
        results = []
        
        for station_name in station_numbers_reverse.values():
            for direction in ['Northbound', 'Southbound']:
                model_key = f"{station_name}_{direction}"
                if model_key not in directional_models:
                    print(f"⚠️ No model: {model_key}")
                    continue
                
                print(f"\n📊 Processing: {station_name} {direction}")
                station_num = [k for k, v in station_numbers_reverse.items() if v == station_name][0]
                
                # USE CORRECT FILTERING FOR TERMINAL STATIONS
                station_df = get_station_data_for_direction(df, station_num, direction)
                station_df = station_df[station_df['direction'] == direction]
                
                if len(station_df) < 100:
                    print(f"  ⚠️ Only {len(station_df)} records (need 100+)")
                    continue
                
                station_df['hour_timestamp'] = station_df['datetime'].dt.floor('h')
                
                # Aggregate by hour
                hourly = station_df.groupby('hour_timestamp').agg({
                    'TotalPassenger': 'sum',
                    'hour': 'first', 'weekday': 'first', 'month': 'first',
                    'hour_sin': 'first', 'hour_cos': 'first',
                    'dow_sin': 'first', 'dow_cos': 'first',
                    'month_sin': 'first', 'month_cos': 'first',
                    'time_decimal': 'first',
                    'is_operating_hour': 'first',
                    'minute_normalized': 'first',
                    'is_morning_rush': 'first', 'is_evening_rush': 'first', 'is_noon': 'first',
                    'is_pre_opening': 'first', 'is_post_closing': 'first',
                    'minutes_until_closing': 'first', 'minutes_since_opening': 'first',
                    'time_normalized': 'first',
                    'is_weekend': 'first', 'is_holiday': 'first', 'is_special_event': 'first',
                    'is_christmas_season': 'first', 'is_payday': 'first', 'is_friday': 'first',
                    'is_rush_hour': 'first',
                    'is_maintenance_record': 'first', 'is_extended_hours': 'first'
                }).reset_index()
                
                if len(hourly) < 25:
                    print(f"  ⚠️ Only {len(hourly)} hours (need 25+)")
                    continue
                
              
                station_max = PER_DIRECTION_MAX.get(model_key, 100)
                print(f"  📊 Using training max: {station_max:.0f} passengers/hour")
                
                hourly['congestion'] = (hourly['TotalPassenger'] / station_max * 100).clip(0, 100)
                hourly = hourly.sort_values('hour_timestamp')
                
                # Test up to 10 random dates
                available_indices = list(range(24, len(hourly)))
                if not available_indices:
                    continue
                
                num_tests = min(10, len(available_indices))
                test_indices = random.sample(available_indices, num_tests)
                print(f"  📅 Testing {num_tests} dates")
                
                for idx in test_indices:
                    target = hourly.iloc[idx]
                    target_time = target['hour_timestamp']
                    
                    try:
                        pred_result = get_directional_prediction(
                            station_name, direction, target_time,
                            directional_models, directional_scalers,
                            get_feature_sequence_for_station
                        )
                        
                        if pred_result is not None and pred_result >= 0:
                            pred_congestion = pred_result
                            actual_congestion = target['congestion']
                            
                            if actual_congestion > 100 or actual_congestion < 0:
                                continue
                            
                            abs_error = abs(pred_congestion - actual_congestion)
                            
                            if abs_error <= 5:
                                verdict = 'EXCELLENT'
                            elif abs_error <= 10:
                                verdict = 'GOOD'
                            elif abs_error <= 15:
                                verdict = 'OKAY'
                            else:
                                verdict = 'NEEDS_IMPROVEMENT'
                            
                            results.append({
                                'station': station_name,
                                'direction': direction,
                                'target_time': target_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'predicted': round(pred_congestion, 1),
                                'actual': round(actual_congestion, 1),
                                'total_passengers': int(target['TotalPassenger']),
                                'station_max': int(station_max),
                                'absolute_error': round(abs_error, 1),
                                'percentage_error': round((abs_error / max(actual_congestion, 0.1) * 100), 1),
                                'verdict': verdict,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            print(f"  ✅ {target_time.strftime('%Y-%m-%d %H:%M')}: Pred={round(pred_congestion,1)}%, Actual={round(actual_congestion,1)}%, Error={round(abs_error,1)}%")
                        else:
                            print(f"  ⚠️ Prediction returned None for {target_time}")
                            
                    except Exception as e:
                        print(f"  ❌ Error: {e}")
                        continue
        
        if results:
            df_results = pd.DataFrame(results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            df_results.to_csv(f'test_results/full_2025_test_{timestamp}.csv', index=False)
            
            for (station, direction), group_df in df_results.groupby(['station', 'direction']):
                group_df.to_csv(f"test_results/{station}_{direction}_results.csv", index=False)
            
            summary = {
                "total_tests_run": len(results),
                "stations_tested": len(df_results['station'].unique()),
                "total_predictions": len(results),
                "avg_absolute_error": round(df_results['absolute_error'].mean(), 2),
                "avg_percentage_error": round(df_results['percentage_error'].mean(), 2),
                "excellent_count": len(df_results[df_results['verdict'] == 'EXCELLENT']),
                "good_count": len(df_results[df_results['verdict'] == 'GOOD']),
                "okay_count": len(df_results[df_results['verdict'] == 'OKAY']),
                "needs_improvement": len(df_results[df_results['verdict'] == 'NEEDS_IMPROVEMENT']),
                "date_range": {
                    "min": df_results['target_time'].min(),
                    "max": df_results['target_time'].max()
                }
            }
            
            return jsonify({
                "success": True,
                "message": f"✅ Generated {len(results)} predictions for 2025 data",
                "summary": summary
            })
        else:
            return jsonify({"success": False, "error": "No predictions could be generated"}), 400
        
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500

# ============= VALIDATION ENDPOINTS =============
@model_perf_bp.route('/model/validate-2025-data', methods=['GET'])
def validate_2025_data():
    """Check if 2025 data is ready for testing"""
    data_file = find_data_file()
    if not data_file:
        return jsonify({"error": "No data file found"})
    
    df = pd.read_csv(data_file)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    years = df['datetime'].dt.year.unique()
    
    return jsonify({
        "data_file": data_file,
        "years_in_data": years.tolist(),
        "is_2025_data": 2025 in years,
        "total_records": len(df),
        "date_range": {
            "min": df['datetime'].min().isoformat(),
            "max": df['datetime'].max().isoformat()
        }
    })

# ============= GENERATE PREDICTIONS =============
@model_perf_bp.route('/model/generate-predictions', methods=['POST'])
def generate_predictions():
    """Generate fresh predictions using your LSTM models on random dates"""
    try:
        data_file = find_data_file()
        
        if not data_file:
            return jsonify({"success": False, "error": "Data file not found"}), 404
        
        print(f"📊 Loading data from: {data_file}")
        df = pd.read_csv(data_file)
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday
        df['month'] = df['datetime'].dt.month
        df['minute'] = df['datetime'].dt.minute
        
        # Add ALL features
        df = add_cyclical_time_features(df)
        df = add_smart_operating_flags(df)
        df['direction'] = df.apply(infer_direction_correct, axis=1)
        df['is_weekend'] = (df['datetime'].dt.weekday >= 5).astype(np.int8)
        df['is_holiday'] = 0
        df['is_special_event'] = 0
        df['is_christmas_season'] = df['datetime'].apply(is_christmas_season).astype(np.int8)
        df['is_payday'] = df['datetime'].apply(is_payday).astype(np.int8)
        df['is_friday'] = df['datetime'].apply(is_friday).astype(np.int8)
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(np.int8)
        df['is_maintenance_record'] = 0
        df['is_extended_hours'] = 0
        
        station_numbers_reverse = {
            1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
            5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
            9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
        }
        
        all_stations = list(station_numbers_reverse.values())
        results = []
        
        for station_name in all_stations:
            for direction in ['Northbound', 'Southbound']:
                model_key = f"{station_name}_{direction}"
                
                if model_key not in directional_models:
                    print(f"No model for {model_key}, skipping...")
                    continue
                
                station_num = [k for k, v in station_numbers_reverse.items() if v == station_name][0]
                
                # Use correct filtering for terminals
                station_df = get_station_data_for_direction(df, station_num, direction)
                station_df = station_df[station_df['direction'] == direction]
                
                if len(station_df) < 100:
                    print(f"Not enough data for {station_name} {direction}: {len(station_df)} rows")
                    continue
                
                station_df['hour_timestamp'] = station_df['datetime'].dt.floor('h')
                hourly = station_df.groupby('hour_timestamp').agg({
                    'TotalPassenger': 'sum',
                    'datetime': 'first'
                }).reset_index()
                
                # ALWAYS use training max (matching diagnosis script)
                station_max = PER_DIRECTION_MAX.get(model_key, 1)
                hourly['actual_congestion'] = (hourly['TotalPassenger'] / station_max * 100).clip(0, 100)
                
                if len(hourly) < 25:
                    continue
                
                available_indices = list(range(24, len(hourly)))
                random_indices = random.sample(available_indices, min(3, len(available_indices)))
                
                for idx in random_indices:
                    target = hourly.iloc[idx]
                    
                    try:
                        prediction_result = get_directional_prediction(
                            station_name, direction, target['datetime'],
                            directional_models, directional_scalers,
                            get_feature_sequence_for_station
                        )
                        
                        if prediction_result is not None:
                            pred_congestion = prediction_result
                            actual_congestion = target['actual_congestion']
                            absolute_error = abs(pred_congestion - actual_congestion)
                            
                            if absolute_error <= 5:
                                verdict = 'EXCELLENT'
                            elif absolute_error <= 10:
                                verdict = 'GOOD'
                            elif absolute_error <= 15:
                                verdict = 'OKAY'
                            else:
                                verdict = 'NEEDS_IMPROVEMENT'
                            
                            results.append({
                                'station': station_name,
                                'direction': direction,
                                'target_time': target['hour_timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                                'predicted': round(pred_congestion, 1),
                                'actual': round(actual_congestion, 1),
                                'absolute_error': round(absolute_error, 1),
                                'percentage_error': round((absolute_error / max(actual_congestion, 0.1) * 100), 1),
                                'verdict': verdict,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            print(f"✅ {station_name} {direction}: Pred={round(pred_congestion,1)}%, Actual={round(actual_congestion,1)}%")
                    except Exception as e:
                        print(f"Error for {station_name} {direction}: {e}")
                        continue
        
        if results:
            df_results = pd.DataFrame(results)
            os.makedirs('test_results', exist_ok=True)
            
            for (station, direction), group_df in df_results.groupby(['station', 'direction']):
                result_filename = f"test_results/{station}_{direction}_results.csv"
                group_df.to_csv(result_filename, index=False)
            
            summary = {
                "total_tests_run": len(results),
                "stations_tested": len(df_results['station'].unique()),
                "avg_absolute_error": round(df_results['absolute_error'].mean(), 2)
            }
            
            return jsonify({
                "success": True,
                "message": f"Generated {len(results)} fresh predictions using random dates from your data",
                "summary": summary
            })
        else:
            return jsonify({"success": False, "error": "No predictions could be generated"}), 400
        
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500

# ============= CHART DATA =============


# ============= BATCH UPLOAD =============
@model_perf_bp.route('/model/upload/batch', methods=['POST'])
def upload_batch_test():
    """Upload a CSV file with batch test results"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Only CSV files are allowed"}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        df = pd.read_csv(filepath)
        
        raw_mrt_columns = ['TotalPassenger', 'Time', 'Date', 'StationEntry', 'StationExit']
        test_results_columns = ['station', 'direction', 'target_time', 'predicted', 'actual']
        
        is_raw_mrt = all(col in df.columns for col in raw_mrt_columns)
        is_test_results = all(col in df.columns for col in test_results_columns)
        
        if is_raw_mrt:
            return jsonify({
                "success": True,
                "message": "Raw MRT data uploaded. Click 'Run Auto-Tests' to generate predictions.",
                "format": "raw_mrt",
                "rows": len(df)
            })
        
        elif is_test_results:
            if 'absolute_error' not in df.columns:
                df['absolute_error'] = abs(df['predicted'] - df['actual'])
            if 'percentage_error' not in df.columns:
                df['percentage_error'] = (df['absolute_error'] / df['actual'] * 100).fillna(0)
            if 'verdict' not in df.columns:
                df['verdict'] = df['absolute_error'].apply(
                    lambda x: 'EXCELLENT' if x <= 5 else ('GOOD' if x <= 10 else ('OKAY' if x <= 15 else 'NEEDS_IMPROVEMENT'))
                )
            
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            
            os.makedirs('test_results', exist_ok=True)
            
            for (station, direction), group_df in df.groupby(['station', 'direction']):
                result_filename = f"test_results/{station}_{direction}_results.csv"
                if os.path.exists(result_filename):
                    existing = pd.read_csv(result_filename)
                    combined = pd.concat([existing, group_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=['target_time'], keep='last')
                    combined.to_csv(result_filename, index=False)
                else:
                    group_df.to_csv(result_filename, index=False)
            
            summary = {
                "total_records": len(df),
                "stations_processed": len(df['station'].unique()),
                "avg_absolute_error": round(df['absolute_error'].mean(), 2),
            }
            
            return jsonify({
                "success": True,
                "message": f"Successfully uploaded {len(df)} test records",
                "format": "test_results",
                "summary": summary
            })
        
        else:
            return jsonify({
                "success": False,
                "error": "Unknown file format. Expected columns: 'station, direction, target_time, predicted, actual' OR 'TotalPassenger, Time, Date, StationEntry, StationExit'",
                "found_columns": list(df.columns)
            }), 400
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ============= LEGACY/EXTRA ENDPOINTS =============
@model_perf_bp.route('/model/generate-all-test-data', methods=['POST'])
def generate_all_test_data():
    """Generate test data for ALL 26 stations"""
    return generate_predictions()

@model_perf_bp.route('/model/force-generate-tests', methods=['POST'])
def force_generate_tests():
    """Force generate tests using loaded models"""
    return generate_predictions()

@model_perf_bp.route('/debug/test-feature-sequence', methods=['GET'])
def test_feature_sequence():
    """Test what get_feature_sequence_for_station receives"""
    station = request.args.get('station', 'North Ave')
    direction = request.args.get('direction', 'Northbound')
    target_time_str = request.args.get('datetime', '2025-01-06 09:00:00')
    
    print(f"=== DEBUG ===")
    print(f"Station: {station}, Type: {type(station)}")
    print(f"Direction: {direction}, Type: {type(direction)}")
    print(f"Target time string: {target_time_str}, Type: {type(target_time_str)}")
    
    # Convert to datetime
    target_datetime = pd.to_datetime(target_time_str)
    print(f"Converted datetime: {target_datetime}, Type: {type(target_datetime)}")
    
    try:
        sequence = get_feature_sequence_for_station(station, direction, target_datetime)
        if sequence is not None:
            return jsonify({
                "success": True,
                "sequence_shape": sequence.shape,
                "sequence_dtype": str(sequence.dtype),
                "sequence_min": float(sequence.min()),
                "sequence_max": float(sequence.max())
            })
        else:
            return jsonify({"success": False, "error": "Sequence is None"})
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500