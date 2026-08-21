"""
Model Performance Routes - LSTM Testing & Visualization
Handles: Manual predictions, batch uploads, metrics, chart data, evaluation metrics
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import pandas as pd
import numpy as np
import os
import pickle
import random
import json
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from werkzeug.utils import secure_filename
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import services
from services.feature_engineering import get_feature_sequence_for_station
from services import get_directional_prediction
from services.model_loader import directional_models, directional_scalers
from routes.api_other import MRT3_PLATFORM_CAPACITY

model_perf_bp = Blueprint('model_performance', __name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('test_results', exist_ok=True)
os.makedirs('evaluation_results', exist_ok=True)

# ============================================================
# STATIONS LIST (MUST MATCH api_predict.py)
# ============================================================
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

# ============================================================
# LOAD HISTORICAL PEAKS FOR REFERENCE ONLY (NOT used for congestion)
# ============================================================
HISTORICAL_PEAKS = {}


@model_perf_bp.route('/model/drift-detection', methods=['POST'])
def detect_drift():
    """
    Compare uploaded 2026 data against baseline 2025 performance
    to detect if the model is drifting
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400
        
        # Read uploaded file
        df = pd.read_csv(file)
        
        required_cols = ['station', 'direction', 'datetime', 'actual_congestion']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({
                'success': False, 
                'error': f'Missing required columns: {missing_cols}'
            }), 400
        
        from routes.api_predict import get_directional_prediction as api_prediction
        
        # Load baseline (2025) performance
        baseline_file = 'test_results/full_2025_test_*.csv'
        baseline_files = [f for f in os.listdir('test_results') if f.startswith('full_2025_test_')]
        
        baseline_errors = []
        if baseline_files:
            baseline_df = pd.read_csv(os.path.join('test_results', baseline_files[-1]))
            if 'absolute_error' in baseline_df.columns:
                baseline_errors = baseline_df['absolute_error'].tolist()
        
        # Process 2026 data
        current_errors = []
        drift_results = []
        
        for idx, row in df.iterrows():
            try:
                station = str(row['station']).strip()
                direction = str(row['direction']).strip().capitalize()
                target_time = pd.to_datetime(row['datetime'])
                actual = float(row['actual_congestion'])
                
                pred = api_prediction(station, direction, target_time)
                if pred is not None:
                    error = abs(pred - actual)
                    current_errors.append(error)
                    drift_results.append({
                        'station': station,
                        'direction': direction,
                        'datetime': target_time.isoformat(),
                        'predicted': round(pred, 1),
                        'actual': round(actual, 1),
                        'error': round(error, 1)
                    })
            except Exception as e:
                print(f"Row {idx} error: {e}")
                continue
        
        # Calculate drift metrics
        if baseline_errors and current_errors:
            baseline_mean = np.mean(baseline_errors)
            current_mean = np.mean(current_errors)
            drift_percentage = ((current_mean - baseline_mean) / baseline_mean) * 100
            
            # Determine drift severity
            if drift_percentage > 30:
                severity = "SEVERE DRIFT - Model needs retraining"
            elif drift_percentage > 15:
                severity = "MODERATE DRIFT - Monitor closely"
            elif drift_percentage > 5:
                severity = "MINOR DRIFT - Acceptable"
            else:
                severity = "NO DRIFT DETECTED"
            
            return jsonify({
                'success': True,
                'message': f'Drift analysis complete for {len(current_errors)} samples from 2026 data',
                'drift_analysis': {
                    'baseline_mae': round(baseline_mean, 2),
                    'current_mae': round(current_mean, 2),
                    'drift_percentage': round(drift_percentage, 1),
                    'severity': severity,
                    'recommendation': 'Retrain model with new data' if drift_percentage > 15 else 'Model still valid'
                },
                'results_preview': drift_results[:20]
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Not enough data to detect drift. Need baseline data.'
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
    
def load_historical_peaks():
    """Load historical peaks for REFERENCE ONLY - NOT used for congestion calculation"""
    global HISTORICAL_PEAKS
    
    peaks_file = 'historical_peaks.json'
    if os.path.exists(peaks_file):
        try:
            with open(peaks_file, 'r') as f:
                HISTORICAL_PEAKS = json.load(f)
            print(f"✅ Loaded historical peaks from {peaks_file} (for reference only)")
            return HISTORICAL_PEAKS
        except Exception as e:
            print(f"⚠️ Could not load peaks file: {e}")
    
    print("📊 Calculating historical peaks for reference only...")
    from services.feature_engineering import load_data_fast, get_station_dataframe
    import numpy as np
    
    df = load_data_fast()
    if df is None:
        print("❌ Could not load data for peak calculation")
        return {}
    
    for station in STATIONS:
        for direction in ['Northbound', 'Southbound']:
            hourly = get_station_dataframe(station, direction)
            if hourly is not None and len(hourly) > 0:
                passengers = hourly['TotalPassenger'].values
                peak_abs = float(passengers.max())
                key = f"{station}_{direction}"
                HISTORICAL_PEAKS[key] = {
                    "peak": peak_abs,
                    "absolute_max": peak_abs,
                    "percentile": 100
                }
                print(f"   {key}: abs max = {peak_abs:.0f}")
    
    try:
        with open(peaks_file, 'w') as f:
            json.dump(HISTORICAL_PEAKS, f, indent=2)
        print(f"✅ Saved historical peaks to {peaks_file}")
    except Exception as e:
        print(f"⚠️ Could not save peaks file: {e}")
    
    return HISTORICAL_PEAKS

# Load peaks on startup
HISTORICAL_PEAKS = load_historical_peaks()

# ============================================================
# LOAD CORRECTION FACTORS (MATCHES api_predict.py)
# ============================================================
CORRECTION_FACTORS = {}


def load_correction_factors():
    """Load correction factors - MATCHES api_predict.py"""
    global CORRECTION_FACTORS
    correction_file = 'correction_factors.pkl'
    if os.path.exists(correction_file):
        try:
            with open(correction_file, 'rb') as f:
                CORRECTION_FACTORS = pickle.load(f)
            print(f"✅ Loaded correction factors for {len(CORRECTION_FACTORS)} station-directions")
        except Exception as e:
            print(f"⚠️ Could not load correction factors: {e}")
            CORRECTION_FACTORS = {}
    else:
        print("⚠️ No correction factors found – using 1.0")
        CORRECTION_FACTORS = {}

load_correction_factors()

# ============================================================
# CONGESTION CATEGORIES - Based on Platform Capacity
# ============================================================
# These categories are based on platform congestion:
# - Light (< 30%): Platform underutilized
# - Moderate (30-60%): Normal platform usage
# - Heavy (60-80%): Platform busy
# - Severe (> 80%): Platform overcrowded
# ============================================================
CATEGORY_ORDER = ['Light', 'Moderate', 'Heavy', 'Severe']

def get_congestion_category(congestion_value):
    """Convert congestion percentage to category - MATCHES api_predict.py"""
    if congestion_value > 80:
        return 'Severe'
    elif congestion_value > 60:
        return 'Heavy'
    elif congestion_value > 30:
        return 'Moderate'
    else:
        return 'Light'

def get_capacity(station_name):
    """Get platform capacity for a station"""
    return MRT3_PLATFORM_CAPACITY.get(station_name, 1000)

def calculate_congestion(passenger_count, station_name):
    """Calculate congestion using platform capacity (MATCHES api_predict.py)"""
    capacity = get_capacity(station_name)
    congestion = (passenger_count / capacity) * 100
    return min(congestion, 100)

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

# ============= FEATURE ENGINEERING (MATCHES TRAINING) =============
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
    """Add operating hour flags"""
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

# ============= STATION FILTERING FOR TERMINALS =============
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
            return df[df['StationEntry'] == station_num].copy()
        else:
            return df[df['StationExit'] == station_num].copy()
    else:
        if direction == 'Northbound':
            return df[df['StationExit'] == station_num].copy()
        else:
            return df[df['StationEntry'] == station_num].copy()

def get_historical_peak(station_name, direction):
    """Get the historical peak for REFERENCE ONLY - NOT used for congestion"""
    key = f"{station_name}_{direction}"
    peak_data = HISTORICAL_PEAKS.get(key)
    
    if peak_data:
        if isinstance(peak_data, dict):
            return peak_data.get("peak", peak_data.get("absolute_max", 5000))
        else:
            return peak_data
    return 5000

import numpy as np

def calculate_commuter_congestion(passenger_count, station, direction):
    """
    Calculate 0-100% congestion score based on historical percentile.
    This tells commuters: "This hour is busier than X% of typical hours"
    
    Args:
        passenger_count: Hourly passenger count
        station: Station name (e.g., "Taft")
        direction: "Northbound" or "Southbound"
    
    Returns:
        float: 0-100% congestion score
    """
    # Get historical data for this station/direction
    from services.feature_engineering import get_station_dataframe
    hourly = get_station_dataframe(station, direction)
    
    if hourly is None or len(hourly) == 0:
        # Fallback: use a simple scaling
        return min((passenger_count / 1000) * 100, 100)
    
    # Get historical passenger counts
    historical_counts = hourly['TotalPassenger'].values
    
    # Use 95th percentile as "100%" (very busy)
    # This means only 5% of hours are considered "full"
    p95 = np.percentile(historical_counts, 95)
    
    # Scale to 0-100%
    congestion = min((passenger_count / p95) * 100, 100)
    
    # Ensure minimum value is 0
    congestion = max(congestion, 0)
    
    return congestion

# ============================================================
# OPTIMIZED DATA LOADING WITH CHUNKING
# ============================================================
def load_data_optimized(filepath, max_rows=50000):
    """
    Load data optimized for performance.
    Only loads first max_rows to prevent memory issues.
    """
    try:
        if not os.path.exists(filepath):
            print(f"⚠️ File not found: {filepath}")
            return None
        
        print(f"📊 Loading data from {filepath} (max {max_rows} rows)...")
        
        # Read only necessary columns with row limit
        df = pd.read_csv(
            filepath, 
            nrows=max_rows,
            usecols=['TotalPassenger', 'Date', 'Time', 'StationEntry', 'StationExit']
        )
        
        print(f"✅ Loaded {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# ============= EVALUATION METRICS ENDPOINT =============
@model_perf_bp.route('/model/evaluate', methods=['POST'])
def evaluate_model():
    """Run model evaluation and return confusion matrix and classification metrics"""
    try:
        data = request.json or {}
        station = data.get('station', 'all')
        direction = data.get('direction', 'both')
        days_back = data.get('days_back', 30)
        
        # Load test results
        all_results = []
        if not os.path.exists('test_results'):
            return jsonify({
                "success": False,
                "error": "No test results found. Run auto-tests first.",
                "total_samples": 0
            }), 400
        
        for filename in os.listdir('test_results'):
            if filename.endswith('_results.csv') and not filename.startswith('full_'):
                try:
                    df = pd.read_csv(os.path.join('test_results', filename))
                    
                    if station != 'all' and 'station' in df.columns:
                        df = df[df['station'] == station]
                    if direction != 'both' and 'direction' in df.columns:
                        df = df[df['direction'].str.lower() == direction.lower()]
                    
                    if not df.empty:
                        all_results.append(df)
                except Exception as e:
                    print(f"⚠️ Error reading {filename}: {e}")
                    continue
        
        if not all_results:
            return jsonify({
                "success": False,
                "error": "No test data available for the selected filters.",
                "total_samples": 0
            }), 400
        
        combined = pd.concat(all_results, ignore_index=True)
        
        for col in ['predicted', 'actual', 'absolute_error']:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors='coerce')
        
        combined = combined.dropna(subset=['predicted', 'actual'])
        
        if len(combined) < 10:
            return jsonify({
                "success": False,
                "error": f"Only {len(combined)} valid samples. Need at least 10 for evaluation.",
                "total_samples": len(combined)
            }), 400
        
        predictions = combined['predicted'].values
        actuals = combined['actual'].values
        
        pred_categories = [get_congestion_category(p) for p in predictions]
        actual_categories = [get_congestion_category(a) for a in actuals]
        
        cm = confusion_matrix(actual_categories, pred_categories, labels=CATEGORY_ORDER)
        
        accuracy = accuracy_score(actual_categories, pred_categories)
        precision = precision_score(actual_categories, pred_categories, labels=CATEGORY_ORDER, average='weighted', zero_division=0)
        recall = recall_score(actual_categories, pred_categories, labels=CATEGORY_ORDER, average='weighted', zero_division=0)
        f1 = f1_score(actual_categories, pred_categories, labels=CATEGORY_ORDER, average='weighted', zero_division=0)
        
        class_report = classification_report(actual_categories, pred_categories, labels=CATEGORY_ORDER, output_dict=True, zero_division=0)
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        
        epsilon = 1e-8
        mape = np.mean(np.abs((actuals - predictions) / (actuals + epsilon))) * 100
        
        per_class_metrics = {}
        for category in CATEGORY_ORDER:
            if category in class_report:
                per_class_metrics[category] = {
                    'precision': round(class_report[category]['precision'] * 100, 1),
                    'recall': round(class_report[category]['recall'] * 100, 1),
                    'f1_score': round(class_report[category]['f1-score'] * 100, 1),
                    'support': class_report[category]['support']
                }
        
        category_distribution = {}
        for category in CATEGORY_ORDER:
            category_distribution[category] = {
                'actual': actual_categories.count(category),
                'predicted': pred_categories.count(category)
            }
        
        evaluation_data = {
            'station': station,
            'direction': direction,
            'total_samples': len(predictions),
            'timestamp': datetime.now().isoformat(),
            'confusion_matrix': {
                'labels': CATEGORY_ORDER,
                'matrix': cm.tolist()
            },
            'accuracy': round(accuracy * 100, 2),
            'precision_weighted': round(precision * 100, 2),
            'recall_weighted': round(recall * 100, 2),
            'f1_weighted': round(f1 * 100, 2),
            'regression_metrics': {
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'r2': round(r2, 4),
                'mape': round(mape, 2)
            },
            'per_class_metrics': per_class_metrics,
            'category_distribution': category_distribution
        }
        
        os.makedirs('evaluation_results', exist_ok=True)
        filename = f"evaluation_results/eval_{station}_{direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(evaluation_data, f, indent=2)
        
        return jsonify({
            "success": True,
            "data": evaluation_data
        })
        
    except Exception as e:
        import traceback
        print(f"❌ Evaluation error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@model_perf_bp.route('/model/evaluation/history', methods=['GET'])
def get_evaluation_history():
    """Get historical evaluation results"""
    try:
        station = request.args.get('station', 'all')
        direction = request.args.get('direction', 'both')
        
        if not os.path.exists('evaluation_results'):
            return jsonify({"success": True, "data": []})
        
        evaluations = []
        for f in os.listdir('evaluation_results'):
            if f.endswith('.json'):
                try:
                    with open(os.path.join('evaluation_results', f), 'r') as file:
                        data = json.load(file)
                        if station != 'all' and data.get('station') != station:
                            continue
                        if direction != 'both' and data.get('direction') != direction:
                            continue
                        evaluations.append(data)
                except:
                    continue
        
        evaluations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return jsonify({"success": True, "data": evaluations})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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
            "historical_peaks_loaded": len(HISTORICAL_PEAKS),
            "correction_factors_loaded": len(CORRECTION_FACTORS),
            "ready_for_testing": len(directional_models) > 0 and data_exists
        }
    })

# ============= DEBUG ENDPOINTS =============

@model_perf_bp.route('/debug/analyze-scalers', methods=['GET'])
def debug_analyze_scalers():
    """Analyze feature and target scalers to understand what the model learned"""
    try:
        station = request.args.get('station', 'Taft')
        direction = request.args.get('direction', 'Southbound')
        model_key = f"{station}_{direction}"
        
        result = {
            "model_key": model_key,
            "station": station,
            "direction": direction,
            "analysis": {}
        }
        
        feature_scaler = directional_scalers.get(f'{model_key}_feature')
        if feature_scaler is not None:
            congestion_idx = -1
            result["analysis"]["feature_scaler"] = {
                "congestion_feature_index": congestion_idx,
                "congestion_data_min": float(feature_scaler.data_min_[congestion_idx]),
                "congestion_data_max": float(feature_scaler.data_max_[congestion_idx]),
                "interpretation": f"During training, congestion was scaled from {feature_scaler.data_min_[congestion_idx]*100:.0f}% to {feature_scaler.data_max_[congestion_idx]*100:.0f}%"
            }
        else:
            result["analysis"]["feature_scaler"] = {"error": "Feature scaler not found"}
        
        target_scaler = directional_scalers.get(f'{model_key}_target')
        if target_scaler is not None:
            result["analysis"]["target_scaler"] = {
                "target_feature": "TotalPassenger (raw passenger counts)",
                "data_min": float(target_scaler.data_min_[0]),
                "data_max": float(target_scaler.data_max_[0]),
                "interpretation": f"The model predicts passenger counts from {target_scaler.data_min_[0]:.0f} to {target_scaler.data_max_[0]:.0f}"
            }
        else:
            result["analysis"]["target_scaler"] = {"error": "Target scaler not found"}
        
        # Get platform capacity
        capacity = get_capacity(station)
        result["analysis"]["platform_capacity"] = {
            "value": capacity,
            "source": "DOTr official data",
            "note": f"Congestion = (passenger_count / {capacity:.0f}) * 100"
        }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@model_perf_bp.route('/debug/trace-prediction/<station>/<direction>', methods=['GET'])
def debug_trace_prediction(station, direction):
    """Trace the complete prediction flow from model to congestion"""
    try:
        from routes.api_predict import get_directional_prediction as api_prediction
        
        station = station.replace('%20', ' ')
        direction = direction.capitalize()
        target_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        model_key = f"{station}_{direction}"
        
        target_scaler = directional_scalers.get(f'{model_key}_target')
        feature_scaler = directional_scalers.get(f'{model_key}_feature')
        
        pred_congestion = api_prediction(station, direction, target_time)
        
        # Get capacity
        capacity = get_capacity(station)
        
        response = {
            "station": station,
            "direction": direction,
            "target_time": target_time.isoformat(),
            "model_key": model_key,
            "platform_capacity": capacity,
            "prediction_flow": {}
        }
        
        if target_scaler is not None:
            response["target_scaler_info"] = {
                "data_min": float(target_scaler.data_min_[0]),
                "data_max": float(target_scaler.data_max_[0]),
            }
        
        if feature_scaler is not None:
            congestion_idx = -1
            response["feature_scaler_info"] = {
                "congestion_data_min": float(feature_scaler.data_min_[congestion_idx]),
                "congestion_data_max": float(feature_scaler.data_max_[congestion_idx]),
            }
        
        if pred_congestion is not None:
            # Calculate passenger count from congestion
            pred_passengers = (pred_congestion / 100) * capacity
            
            response["prediction_flow"] = {
                "step_1_model_output": f"{pred_congestion:.1f}%",
                "step_2_passenger_count": f"{pred_passengers:.0f} passengers",
                "step_3_congestion_formula": f"({pred_passengers:.0f} / {capacity:.0f}) * 100 = {pred_congestion:.1f}%",
                "conclusion": "Model → inverse_transform → passenger count → congestion % (using platform capacity)"
            }
        else:
            response["prediction_flow"] = {"error": "Prediction returned None"}
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@model_perf_bp.route('/debug/check-models', methods=['GET'])
def debug_check_models():
    """Debug endpoint to check what models are accessible"""
    return jsonify({
        "models_loaded_in_services": len(directional_models),
        "model_keys": list(directional_models.keys()),
        "scalers_loaded": len(directional_scalers),
        "historical_peaks_loaded": len(HISTORICAL_PEAKS),
        "correction_factors_loaded": len(CORRECTION_FACTORS)
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
        from routes.api_predict import get_directional_prediction as api_prediction
        
        result = api_prediction(station, direction, target_time)
        capacity = get_capacity(station)
        
        return jsonify({
            "success": True,
            "station": station,
            "direction": direction,
            "target_time": target_time.isoformat(),
            "prediction": result,
            "platform_capacity": capacity
        })
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

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
                    numeric_cols = ['predicted', 'actual', 'absolute_error', 'percentage_error']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    if 'absolute_error' in df.columns:
                        df = df.dropna(subset=['absolute_error'])
                    
                    if len(df) > 0:
                        all_results.append(df)
                except Exception as e:
                    print(f"⚠️ Error reading {filename}: {e}")
                    continue
        
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            
            latest_eval = None
            if os.path.exists('evaluation_results'):
                eval_files = [f for f in os.listdir('evaluation_results') if f.endswith('.json')]
                if eval_files:
                    try:
                        with open(os.path.join('evaluation_results', eval_files[-1]), 'r') as f:
                            latest_eval = json.load(f)
                    except:
                        pass
            
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
                "per_station": per_station,
                "last_evaluated": combined['timestamp'].max() if 'timestamp' in combined.columns else None,
                "accuracy": latest_eval.get('accuracy', 0) if latest_eval else 0,
                "f1_score": latest_eval.get('f1_weighted', 0) if latest_eval else 0
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

# ============= CHART DATA =============
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
                    if 'predicted' in df.columns:
                        df['predicted'] = pd.to_numeric(df['predicted'], errors='coerce')
                    if 'actual' in df.columns:
                        df['actual'] = pd.to_numeric(df['actual'], errors='coerce')
                    if 'absolute_error' in df.columns:
                        df['absolute_error'] = pd.to_numeric(df['absolute_error'], errors='coerce')
                    
                    df = df.dropna(subset=['predicted', 'actual'])
                    
                    if len(df) > 0:
                        all_results.append(df)
                except Exception as e:
                    print(f"⚠️ Error reading {filename}: {e}")
                    continue
        
        if not all_results:
            return jsonify({"success": True, "data": {"labels": [], "predicted": [], "actual": [], "errors": []}})
        
        combined = pd.concat(all_results, ignore_index=True)
        
        if station != "all" and 'station' in combined.columns:
            combined = combined[combined['station'] == station]
        if direction != "both" and 'direction' in combined.columns:
            combined = combined[combined['direction'].str.lower() == direction.lower()]
        
        if 'target_time' in combined.columns:
            combined['target_time_dt'] = pd.to_datetime(combined['target_time'])
            combined = combined.sort_values('target_time_dt')
            labels = combined['target_time'].tolist()
        else:
            labels = [f"Test {i+1}" for i in range(len(combined))]
        
        predicted = combined['predicted'].tolist() if 'predicted' in combined.columns else []
        actual = combined['actual'].tolist() if 'actual' in combined.columns else []
        errors = [abs(p - a) for p, a in zip(predicted, actual)] if predicted and actual else []
        
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

# ============= STATION DETAILS =============
@model_perf_bp.route('/model/station-details', methods=['GET'])
def get_station_details():
    """Get detailed test results for a specific station and direction"""
    try:
        station = request.args.get('station', '').strip()
        direction = request.args.get('direction', '').strip().capitalize()
        
        if not station or not direction:
            return jsonify({"success": False, "error": "Missing station or direction"}), 400
        
        if direction not in ['Northbound', 'Southbound']:
            return jsonify({"success": False, "error": f"Invalid direction: {direction}"}), 400
        
        filename = f"test_results/{station}_{direction}_results.csv"
        
        if not os.path.exists(filename):
            return jsonify({"success": True, "data": []}), 200
        
        df = pd.read_csv(filename)
        
        if df.empty:
            return jsonify({"success": True, "data": []}), 200
        
        numeric_cols = ['predicted', 'actual', 'absolute_error', 'percentage_error']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'target_time' in df.columns:
            try:
                df['target_time_dt'] = pd.to_datetime(df['target_time'], errors='coerce')
                df = df.dropna(subset=['target_time_dt'])
                df = df.sort_values('target_time_dt', ascending=False)
            except:
                pass
        
        return_cols = []
        for col in ['target_time', 'predicted', 'actual', 'absolute_error', 'verdict']:
            if col in df.columns:
                return_cols.append(col)
        
        results = df[return_cols].head(50).to_dict('records')
        
        for record in results:
            for key, value in list(record.items()):
                if pd.isna(value):
                    record[key] = None
                elif hasattr(value, 'item'):
                    record[key] = value.item()
        
        return jsonify({
            "success": True,
            "data": results
        })
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@model_perf_bp.route('/model/debug-csv', methods=['GET'])
def debug_csv():
    """
    Debug endpoint to inspect uploaded CSV structure
    Returns column names and first few rows
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400
        
        # Read the CSV
        df = pd.read_csv(file)
        
        return jsonify({
            'success': True,
            'columns': list(df.columns),
            'column_count': len(df.columns),
            'row_count': len(df),
            'sample_data': df.head(5).to_dict('records'),
            'data_types': {col: str(df[col].dtype) for col in df.columns},
            'missing_values': {col: int(df[col].isnull().sum()) for col in df.columns}
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
        
        
@model_perf_bp.route('/debug/check-2025-csv', methods=['GET'])
def debug_check_2025_csv():
    """
    Debug endpoint to inspect the actual 2025 CSV file structure
    """
    try:
        # Find the data file
        data_file = find_data_file()
        
        if not data_file:
            return jsonify({
                'success': False,
                'error': '2025 CSV file not found',
                'searched_locations': [
                    'data (2022-2024)/2025.csv',
                    '../data (2022-2024)/2025.csv',
                    '2025.csv',
                    'data/2025.csv'
                ]
            }), 404
        
        print(f"📁 Found data file: {data_file}")
        
        # Read first 10 rows to inspect
        df_sample = pd.read_csv(data_file, nrows=10)
        
        # Also check file size
        file_size = os.path.getsize(data_file) / (1024 * 1024)  # MB
        
        # Get total row count (fast)
        total_rows = sum(1 for _ in open(data_file)) - 1  # Subtract header
        
        return jsonify({
            'success': True,
            'file_info': {
                'path': data_file,
                'size_mb': round(file_size, 2),
                'total_rows': total_rows,
                'columns': df_sample.columns.tolist(),
                'column_count': len(df_sample.columns)
            },
            'sample_data': df_sample.to_dict('records'),
            'data_types': {col: str(df_sample[col].dtype) for col in df_sample.columns},
            'sample_statistics': {
                col: {
                    'unique_values': df_sample[col].nunique(),
                    'null_count': df_sample[col].isnull().sum(),
                    'sample_values': df_sample[col].head(5).tolist()
                }
                for col in df_sample.columns
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@model_perf_bp.route('/debug/check-raw-mrt-data', methods=['GET'])
def debug_check_raw_mrt_data():
    """
    Check raw MRT data structure and sample entries
    """
    try:
        data_file = find_data_file()
        
        if not data_file:
            return jsonify({'success': False, 'error': 'Data file not found'}), 404
        
        # Read a larger sample for analysis
        df = pd.read_csv(data_file, nrows=1000)
        
        # Check if it has raw MRT columns
        raw_mrt_columns = ['TotalPassenger', 'StationEntry', 'StationExit', 'Date', 'Time']
        is_raw_mrt = all(col in df.columns for col in raw_mrt_columns)
        
        result = {
            'is_raw_mrt_format': is_raw_mrt,
            'columns': df.columns.tolist(),
            'sample_data': df.head(10).to_dict('records'),
            'statistics': {}
        }
        
        if is_raw_mrt:
            # Analyze raw MRT data
            result['statistics'] = {
                'station_entry_values': df['StationEntry'].value_counts().head(10).to_dict(),
                'station_exit_values': df['StationExit'].value_counts().head(10).to_dict(),
                'total_passenger_stats': {
                    'min': float(df['TotalPassenger'].min()),
                    'max': float(df['TotalPassenger'].max()),
                    'mean': float(df['TotalPassenger'].mean()),
                    'std': float(df['TotalPassenger'].std())
                },
                'date_range': {
                    'min': df['Date'].min(),
                    'max': df['Date'].max()
                }
            }
            
            # Test direction inference on sample
            df['inferred_direction'] = df.apply(infer_direction_correct, axis=1)
            result['statistics']['direction_distribution'] = df['inferred_direction'].value_counts().to_dict()
            
            # Show sample with direction inference
            result['direction_samples'] = df[['StationEntry', 'StationExit', 'inferred_direction']].head(20).to_dict('records')
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@model_perf_bp.route('/debug/test-preprocessing-on-sample', methods=['GET'])
def debug_test_preprocessing_on_sample():
    """
    Test preprocessing on a small sample of the CSV
    """
    try:
        data_file = find_data_file()
        
        if not data_file:
            return jsonify({'success': False, 'error': 'Data file not found'}), 404
        
        # Read first 1000 rows and test preprocessing
        df_sample = pd.read_csv(data_file, nrows=1000)
        
        print("📊 Testing preprocessing on sample...")
        print(f"Sample shape: {df_sample.shape}")
        print(f"Sample columns: {df_sample.columns.tolist()}")
        
        # Check if raw MRT format
        raw_mrt_columns = ['TotalPassenger', 'StationEntry', 'StationExit', 'Date', 'Time']
        is_raw_mrt = all(col in df_sample.columns for col in raw_mrt_columns)
        
        if not is_raw_mrt:
            return jsonify({
                'success': False,
                'error': 'Sample is not in raw MRT format',
                'available_columns': df_sample.columns.tolist(),
                'expected_columns': raw_mrt_columns
            })
        
        # Process sample
        df = df_sample.copy()
        
        # Parse dates
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        
        # Infer direction
        df['direction'] = df.apply(infer_direction_correct, axis=1)
        df = df[df['direction'] != 'Unknown']
        
        # Map station
        station_names = {
            1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
            5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
            9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
        }
        df['StationName'] = df['StationExit'].map(station_names)
        df = df.dropna(subset=['StationName'])
        
        # Group by hour
        df['Hour'] = df['datetime'].dt.floor('h')
        
        # Aggregate
        agg_dict = {
            'TotalPassenger': 'sum',
            'datetime': 'first'
        }
        
        grouped = df.groupby(['StationName', 'direction', 'Hour']).agg(agg_dict).reset_index()
        
        # Calculate congestion
        def calc_congestion(row):
            return calculate_commuter_congestion(
                row['TotalPassenger'], 
                row['StationName'], 
                row['direction']
            )
        
        grouped['actual_congestion'] = grouped.apply(calc_congestion, axis=1)
        
        # Final output
        result = grouped[['StationName', 'direction', 'Hour', 'actual_congestion']].copy()
        result.columns = ['station', 'direction', 'datetime', 'actual_congestion']
        result['datetime'] = result['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        result['actual_congestion'] = result['actual_congestion'].round(1)
        
        return jsonify({
            'success': True,
            'preprocessing_summary': {
                'original_rows': len(df_sample),
                'after_dropna': len(df),
                'after_unknown_direction': len(df[df['direction'] != 'Unknown']),
                'after_station_mapping': len(df.dropna(subset=['StationName'])),
                'grouped_rows': len(grouped),
                'final_rows': len(result)
            },
            'sample_output': result.head(10).to_dict('records'),
            'unique_stations': result['station'].unique().tolist(),
            'unique_directions': result['direction'].unique().tolist(),
            'sample_original_data': df_sample.head(5).to_dict('records'),
            'sample_processed_data': df.head(5).to_dict('records'),
            'sample_grouped_data': grouped.head(5).to_dict('records')
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
        

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
        
        from routes.api_predict import get_directional_prediction as api_prediction
        
        pred_congestion = api_prediction(station, direction, target_time)
        
        if pred_congestion is None:
            return jsonify({
                "success": False,
                "error": f"Prediction failed. Model may not exist for {station} {direction}"
            }), 400
        
        # Try to get actual congestion from historical data using PLATFORM CAPACITY
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
                
                station_df = get_station_data_for_direction(df, station_num, direction)
                station_df = station_df[station_df['direction'] == direction]
                station_df['hour_timestamp'] = station_df['datetime'].dt.floor('h')
                
                matching = station_df[station_df['hour_timestamp'] == target_time.floor('h')]
                if not matching.empty:
                    total_pass = matching['TotalPassenger'].sum()
                    capacity = get_capacity(station)
                    actual_congestion = (total_pass / capacity * 100)
                    actual_congestion = min(actual_congestion, 100)
                    actual_congestion = round(actual_congestion, 1)
                    
            except Exception as e:
                print(f"Warning: Could not fetch actual congestion: {e}")
        
        response = {
            "success": True,
            "prediction": {
                "predicted": round(pred_congestion, 1),
                "station": station,
                "direction": direction,
                "target_time": target_datetime,
                "category": get_congestion_category(pred_congestion)
            }
        }
        if actual_congestion is not None:
            response["prediction"]["actual"] = actual_congestion
            response["prediction"]["error"] = round(abs(pred_congestion - actual_congestion), 1)
            response["prediction"]["actual_category"] = get_congestion_category(actual_congestion)
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ============= RUN AUTO TESTS WITH OPTIMIZED LOADING =============
@model_perf_bp.route('/model/run-auto-tests', methods=['POST'])
def run_auto_tests():
    """Run auto-tests with configurable sampling"""
    try:
        if not directional_models:
            return jsonify({"success": False, "error": "No models loaded"}), 500
        
        use_test_max = request.args.get('use_test_max', 'false').lower() == 'true'
        if request.is_json:
            data = request.json or {}
            if 'use_test_max' in data:
                use_test_max = data.get('use_test_max', False)
        
        # ============================================================
        # CONFIGURATION: How many samples per hour?
        # ============================================================
        samples_per_hour = 5  # Reduced for faster testing
        
        print(f"🔧 Configuration: {samples_per_hour} samples per hour")
        
        data_file = find_data_file()
        if not data_file:
            return jsonify({"success": False, "error": "Data file not found"}), 404
        
        print("📊 Loading data...")
        
        # Use optimized loading with row limit
        df = load_data_optimized(data_file, max_rows=50000)
        
        if df is None or len(df) == 0:
            return jsonify({"success": False, "error": "Could not load data"}), 400
        
        print(f"✅ Loaded {len(df)} rows")
        
        # Parse dates
        print("📊 Parsing dates...")
        try:
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                           format='%m/%d/%Y %H:%M:%S', 
                                           errors='coerce')
        except:
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                           errors='coerce')
        
        df = df.dropna(subset=['datetime'])
        
        if len(df) == 0:
            return jsonify({"success": False, "error": "Could not parse any dates"}), 400
        
        # Extract hour for stratified sampling
        df['hour'] = df['datetime'].dt.hour
        
        # Map station numbers to names
        station_names = {
            1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
            5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
            9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
        }
        df['StationName'] = df['StationExit'].map(station_names)
        
        # Infer direction
        df['direction'] = df.apply(infer_direction_correct, axis=1)
        df = df[df['direction'] != 'Unknown']
        
        # ============================================================
        # STRATIFIED SAMPLING: Sample evenly across ALL hours
        # ============================================================
        print(f"📊 Stratified sampling by hour ({samples_per_hour} per hour)...")
        
        station_direction_pairs = []
        for station_name in station_names.values():
            for direction in ['Northbound', 'Southbound']:
                model_key = f"{station_name}_{direction}"
                if model_key in directional_models:
                    station_direction_pairs.append((station_name, direction))
        
        sampled_dfs = []
        
        for station_name, direction in station_direction_pairs:
            # Filter for this station-direction
            subset = df[(df['StationName'] == station_name) & (df['direction'] == direction)]
            
            if len(subset) > 0:
                # ============================================================
                # STRATIFIED BY HOUR: Take samples from each hour
                # ============================================================
                hourly_samples = []
                for hour in range(24):
                    hour_subset = subset[subset['hour'] == hour]
                    if len(hour_subset) > 0:
                        # Take up to samples_per_hour from this hour
                        n_samples = min(samples_per_hour, len(hour_subset))
                        sampled = hour_subset.sample(n=n_samples, random_state=42)
                        hourly_samples.append(sampled)
                
                if hourly_samples:
                    combined = pd.concat(hourly_samples, ignore_index=True)
                    sampled_dfs.append(combined)
                    print(f"   {station_name} {direction}: {len(combined)} rows")
        
        if not sampled_dfs:
            return jsonify({"success": False, "error": "No data sampled"}), 400
        
        # Combine sampled data
        df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"✅ Total sampled: {len(df):,} rows")
        
        # ============================================================
        # Feature engineering on the sampled dataset
        # ============================================================
        print("📊 Feature engineering...")
        
        df['weekday'] = df['datetime'].dt.weekday
        df['month'] = df['datetime'].dt.month
        df['minute'] = df['datetime'].dt.minute
        
        df = add_cyclical_time_features(df)
        df = add_smart_operating_flags(df)
        df = smart_data_cleaner(df)
        
        df['is_weekend'] = (df['datetime'].dt.weekday >= 5).astype(np.int8)
        df['is_christmas_season'] = df['datetime'].apply(is_christmas_season).astype(np.int8)
        df['is_payday'] = df['datetime'].apply(is_payday).astype(np.int8)
        df['is_friday'] = df['datetime'].apply(is_friday).astype(np.int8)
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(np.int8)
        
        holidays_2025 = [
            '2025-01-01', '2025-04-09', '2025-04-17', '2025-04-18',
            '2025-05-01', '2025-06-12', '2025-08-25', '2025-11-30',
            '2025-12-25', '2025-12-30', '2025-12-31'
        ]
        df['is_holiday'] = df['datetime'].dt.date.astype(str).isin(holidays_2025).astype(np.int8)
        df['is_special_event'] = 0
        
        station_numbers_reverse = {
            1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
            5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
            9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
        }
        
        from routes.api_predict import get_directional_prediction as api_prediction
        
        print("📊 Running predictions...")
        results = []
        
        for station_name, direction in station_direction_pairs:
            capacity = get_capacity(station_name)
            
            from services.feature_engineering import get_station_dataframe
            historical_hourly = get_station_dataframe(station_name, direction)
            if historical_hourly is None or len(historical_hourly) == 0:
                continue
            
            historical_counts = historical_hourly['TotalPassenger'].values
            p95 = np.percentile(historical_counts, 95)
            
            station_num = [k for k, v in station_numbers_reverse.items() if v == station_name][0]
            station_df = get_station_data_for_direction(df, station_num, direction)
            station_df = station_df[station_df['direction'] == direction]
            
            if len(station_df) < 100:
                continue
            
            station_df['hour_timestamp'] = station_df['datetime'].dt.floor('h')
            
            agg_dict = {
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
            }
            
            existing_cols = [col for col in agg_dict.keys() if col in station_df.columns]
            agg_dict_filtered = {col: agg_dict[col] for col in existing_cols}
            
            hourly = station_df.groupby('hour_timestamp').agg(agg_dict_filtered).reset_index()
            
            if len(hourly) < 25:
                continue
            
            hourly['actual_congestion'] = hourly['TotalPassenger'].apply(
                lambda x: calculate_commuter_congestion(x, station_name, direction)
            )
            hourly = hourly.sort_values('hour_timestamp')
            
            available_indices = list(range(24, len(hourly)))
            if not available_indices:
                continue
            
            # Limit to 50 tests per station-direction for performance
            num_tests = min(len(available_indices), 50)
            test_indices = sorted(random.sample(available_indices, num_tests))
            
            for idx in test_indices:
                target = hourly.iloc[idx]
                target_time = target['hour_timestamp']
                try:
                    pred_congestion_pct = api_prediction(station_name, direction, target_time)
                    
                    if pred_congestion_pct is not None and pred_congestion_pct >= 0:
                        pred_congestion = pred_congestion_pct
                        actual_congestion = target['actual_congestion']
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
                            'platform_capacity': round(capacity, 0),
                            'absolute_error': round(abs_error, 1),
                            'percentage_error': round((abs_error / max(actual_congestion, 0.1) * 100), 1),
                            'verdict': verdict,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'predicted_category': get_congestion_category(pred_congestion),
                            'actual_category': get_congestion_category(actual_congestion)
                        })
                except Exception:
                    continue
        
        if results:
            df_results = pd.DataFrame(results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs('test_results', exist_ok=True)
            
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
                },
                "stations_tested_list": df_results['station'].unique().tolist()
            }
            
            print(f"✅ Complete! {len(results)} predictions")
            
            return jsonify({
                "success": True,
                "message": f"Generated {len(results)} predictions",
                "summary": summary
            })
        else:
            return jsonify({"success": False, "error": "No predictions could be generated"}), 400
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500

def preprocess_large_csv(filepath, max_rows=None):
    """
    Preprocess large CSV by grouping data to reduce prediction count.
    Returns preprocessed dataframe ready for predictions.
    """
    print(f"📊 Preprocessing large CSV: {filepath}")
    
    # Read in chunks to handle large files
    chunk_size = 100000
    chunks = []
    total_rows = 0
    
    # First, check if it's raw MRT format or formatted
    sample_df = pd.read_csv(filepath, nrows=5)
    raw_mrt_columns = ['TotalPassenger', 'StationEntry', 'StationExit', 'Date', 'Time']
    is_raw_mrt = all(col in sample_df.columns for col in raw_mrt_columns)
    
    if is_raw_mrt:
        print("📊 Detected RAW MRT format - grouping by station/direction/hour...")
        
        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            total_rows += len(chunk)
            
            # CRITICAL FIX: Handle time format with no leading zeros
            # Format: "8/1/2025" and "0:00:00"
            try:
                # Try multiple formats
                chunk['datetime'] = pd.to_datetime(
                    chunk['Date'] + ' ' + chunk['Time'], 
                    errors='coerce'
                )
            except:
                chunk['datetime'] = pd.to_datetime(
                    chunk['Date'] + ' ' + chunk['Time'], 
                    format='%m/%d/%Y %H:%M:%S', 
                    errors='coerce'
                )
            
            # Drop invalid dates
            chunk = chunk.dropna(subset=['datetime'])
            
            if len(chunk) == 0:
                print(f"⚠️ Chunk {total_rows} had no valid dates, skipping...")
                continue
            
            # Infer direction
            chunk['direction'] = chunk.apply(infer_direction_correct, axis=1)
            chunk = chunk[chunk['direction'] != 'Unknown']
            
            if len(chunk) == 0:
                continue
            
            # Map station
            station_names = {
                1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
                5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
                9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
            }
            chunk['StationName'] = chunk['StationExit'].map(station_names)
            chunk = chunk.dropna(subset=['StationName'])
            
            if len(chunk) == 0:
                continue
            
            # Group by hour
            chunk['Hour'] = chunk['datetime'].dt.floor('h')
            
            # Aggregate - CRITICAL: Keep station and direction in groupby
            agg_dict = {
                'TotalPassenger': 'sum',
                'datetime': 'first'
            }
            
            grouped = chunk.groupby(['StationName', 'direction', 'Hour']).agg(agg_dict).reset_index()
            chunks.append(grouped)
            
            print(f"   Processed {total_rows} rows so far...")
            
            if max_rows and total_rows >= max_rows:
                break
        
        # Combine all chunks
        if chunks:
            combined = pd.concat(chunks, ignore_index=True)
            
            # Calculate congestion for each group
            def calc_congestion(row):
                return calculate_commuter_congestion(
                    row['TotalPassenger'], 
                    row['StationName'], 
                    row['direction']
                )
            
            combined['actual_congestion'] = combined.apply(calc_congestion, axis=1)
            
            # Format output - KEEP station and direction
            result = combined[['StationName', 'direction', 'Hour', 'actual_congestion']].copy()
            result.columns = ['station', 'direction', 'datetime', 'actual_congestion']
            result['datetime'] = result['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            result['actual_congestion'] = result['actual_congestion'].round(1)
            
            print(f"✅ Preprocessed {total_rows} rows → {len(result)} hourly records")
            print(f"📊 Sample output: {result.head(2).to_dict('records')}")
            return result
        else:
            print("❌ No valid data chunks processed")
            return None
            
    else:
        # Already formatted - just validate and return
        print("📊 Detected FORMATTED data - validating columns...")
        required = ['station', 'direction', 'datetime', 'actual_congestion']
        
        if all(col in sample_df.columns for col in required):
            # Read full file
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.dropna(subset=['datetime', 'actual_congestion', 'station', 'direction'])
            
            print(f"✅ Formatted data: {len(df)} rows")
            print(f"📊 Sample: {df.head(2).to_dict('records')}")
            
            # If still too large, sample intelligently
            if len(df) > 50000:
                print(f"⚠️ Formatted data has {len(df)} rows - sampling to 50,000...")
                sampled = df.groupby(['station', 'direction'], group_keys=False).apply(
                    lambda x: x.sample(min(len(x), 5000), random_state=42)
                )
                df = sampled.reset_index(drop=True)
                print(f"✅ Sampled to {len(df)} rows")
            
            return df
        else:
            print(f"❌ Missing required columns. Found: {sample_df.columns.tolist()}")
            print(f"   Required: {required}")
            return None
@model_perf_bp.route('/model/upload/batch', methods=['POST'])
def upload_batch_test():
    """
    Upload a CSV file and run batch predictions.
    NOW WITH FIXED PREDICTION LOGIC - Uses prediction directly without recalculation.
    """
    try:
        # Check if file exists
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Only CSV files are supported'}), 400
        
        # Secure the filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # ============================================================
        # PREPROCESS THE DATA FIRST
        # ============================================================
        print("🔄 Starting preprocessing...")
        df = preprocess_large_csv(filepath, max_rows=1000000)
        
        if df is None:
            return jsonify({
                'success': False,
                'error': 'Could not preprocess data. Please check column format.'
            }), 400
        
        print(f"✅ After preprocessing - columns: {df.columns.tolist()}, shape: {df.shape}")
        print(f"📊 Sample from preprocessed: {df.head(2).to_dict('records')}")
        
        # ============================================================
        # MANUAL STRATIFIED SAMPLING (preserves all columns)
        # ============================================================
        if len(df) > 50000:
            print(f"⚠️ Data has {len(df)} rows - performing manual stratified sampling...")
            
            # Get unique station-direction pairs
            groups = df.groupby(['station', 'direction'])
            sampled_dfs = []
            
            for (station, direction), group in groups:
                # Determine sample size (max 5000 per group, or all if smaller)
                sample_size = min(200, len(group))
                # Sample without replacement
                sampled_group = group.sample(n=sample_size, random_state=42)
                sampled_dfs.append(sampled_group)
                print(f"   Sampled {sample_size} from {station} {direction} (total {len(group)})")
            
            # Combine all sampled groups
            df = pd.concat(sampled_dfs, ignore_index=True)
            print(f"✅ Final dataset after sampling: {len(df)} rows, columns: {df.columns.tolist()}")
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        # ============================================================
        # VALIDATE COLUMNS BEFORE PREDICTIONS
        # ============================================================
        required_cols = ['station', 'direction', 'datetime', 'actual_congestion']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return jsonify({
                'success': False,
                'error': f'Missing required columns: {missing}',
                'available_columns': df.columns.tolist(),
                'sample_data': df.head(3).to_dict('records')
            }), 400
        
        print(f"✅ Validation passed. Columns: {df.columns.tolist()}")
        print(f"📊 First row: {df.iloc[0].to_dict()}")
        
        # ============================================================
        # RUN PREDICTIONS - FIXED: Use prediction directly
        # ============================================================
        from routes.api_predict import get_directional_prediction as api_prediction
        
        results = []
        errors = 0
        total = len(df)
        
        print(f"📊 Running {total} predictions in batches of 500...")
        
        batch_size = 500
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch = df.iloc[start_idx:end_idx]
            
            batch_results = []
            
            for idx, row in batch.iterrows():
                try:
                    station = str(row['station']).strip()
                    direction = str(row['direction']).strip().capitalize()
                    target_time = pd.to_datetime(row['datetime'])
                    actual_congestion = float(row['actual_congestion'])
                    
                    # FIXED: Get prediction (already returns congestion % 0-100)
                    pred_congestion = api_prediction(station, direction, target_time)
                    
                    if pred_congestion is None or pred_congestion < 0:
                        errors += 1
                        continue
                    
                    # Calculate errors
                    abs_error = abs(pred_congestion - actual_congestion)
                    pct_error = (abs_error / max(actual_congestion, 0.1)) * 100
                    
                    # Determine verdict
                    if abs_error <= 5:
                        verdict = 'EXCELLENT'
                    elif abs_error <= 10:
                        verdict = 'GOOD'
                    elif abs_error <= 15:
                        verdict = 'OKAY'
                    else:
                        verdict = 'NEEDS_IMPROVEMENT'
                    
                    batch_results.append({
                        'station': station,
                        'direction': direction,
                        'target_time': target_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'predicted': round(pred_congestion, 1),
                        'actual': round(actual_congestion, 1),
                        'absolute_error': round(abs_error, 1),
                        'percentage_error': round(pct_error, 1),
                        'verdict': verdict,
                        'predicted_category': get_congestion_category(pred_congestion),
                        'actual_category': get_congestion_category(actual_congestion)
                    })
                        
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"❌ Error at row {idx}: {e}")
                    continue
            
            results.extend(batch_results)
            
            if (start_idx // batch_size) % 10 == 0:
                print(f"   Progress: {end_idx}/{total}, success: {len(results)}, errors: {errors}")
        
        if not results:
            return jsonify({
                'success': False,
                'error': f'No valid predictions. {errors} errors out of {total}.',
                'debug': {
                    'sample_data': df.head(3).to_dict('records'),
                    'columns': df.columns.tolist()
                }
            }), 400
        
        # Save results
        df_results = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('test_results', exist_ok=True)
        
        df_results.to_csv(f'test_results/upload_batch_{timestamp}.csv', index=False)
        
        for (station, direction), group_df in df_results.groupby(['station', 'direction']):
            group_df.to_csv(f"test_results/{station}_{direction}_results.csv", index=False)
        
        summary = {
            'total_processed': len(results),
            'errors': errors,
            'total_rows': total,
            'avg_absolute_error': round(df_results['absolute_error'].mean(), 2),
            'avg_percentage_error': round(df_results['percentage_error'].mean(), 2),
            'excellent_count': len(df_results[df_results['verdict'] == 'EXCELLENT']),
            'good_count': len(df_results[df_results['verdict'] == 'GOOD']),
            'okay_count': len(df_results[df_results['verdict'] == 'OKAY']),
            'needs_improvement': len(df_results[df_results['verdict'] == 'NEEDS_IMPROVEMENT']),
            'stations': df_results['station'].unique().tolist()
        }
        
        print(f"\n✅ Complete! {len(results)} predictions")
        print(f"   Avg Error: {summary['avg_absolute_error']}%")
        print(f"   Excellent: {summary['excellent_count']}, Good: {summary['good_count']}, Okay: {summary['okay_count']}, Needs Improvement: {summary['needs_improvement']}")
        
        return jsonify({
            'success': True,
            'message': f'Processed {len(results)} predictions from {total} rows',
            'summary': summary,
            'results_preview': results[:10]
        })
        
    except Exception as e:
        import traceback
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        
        # Return proper JSON error with traceback
        return jsonify({
            'success': False,
            'error': f'Upload failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500