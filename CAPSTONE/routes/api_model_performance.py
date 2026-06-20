# routes/api_model_performance.py
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

model_perf_bp = Blueprint('model_performance', __name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('test_results', exist_ok=True)
os.makedirs('evaluation_results', exist_ok=True)

# Load per-direction max passengers
MAX_PATH = 'models_2022-2024_v8/per_direction_max_passengers.pkl'
PER_DIRECTION_MAX = {}
if os.path.exists(MAX_PATH):
    with open(MAX_PATH, 'rb') as f:
        PER_DIRECTION_MAX = pickle.load(f)
        print(f"✅ Loaded {len(PER_DIRECTION_MAX)} per-direction max values")

# Congestion categories
CATEGORY_ORDER = ['Light', 'Moderate', 'Heavy', 'Severe']

def get_congestion_category(congestion_value):
    """Convert congestion percentage to category"""
    if congestion_value > 80:
        return 'Severe'
    elif congestion_value > 60:
        return 'Heavy'
    elif congestion_value > 30:
        return 'Moderate'
    else:
        return 'Light'

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
                    
                    # Filter by station
                    if station != 'all' and 'station' in df.columns:
                        df = df[df['station'] == station]
                    
                    # Filter by direction
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
        
        # Convert numeric columns
        for col in ['predicted', 'actual', 'absolute_error']:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors='coerce')
        
        # Drop NaN rows
        combined = combined.dropna(subset=['predicted', 'actual'])
        
        if len(combined) < 10:
            return jsonify({
                "success": False,
                "error": f"Only {len(combined)} valid samples. Need at least 10 for evaluation.",
                "total_samples": len(combined)
            }), 400
        
        # Get predictions and actuals
        predictions = combined['predicted'].values
        actuals = combined['actual'].values
        
        # Convert to categories
        pred_categories = [get_congestion_category(p) for p in predictions]
        actual_categories = [get_congestion_category(a) for a in actuals]
        
        # Calculate confusion matrix
        cm = confusion_matrix(actual_categories, pred_categories, labels=CATEGORY_ORDER)
        
        # Calculate classification metrics
        accuracy = accuracy_score(actual_categories, pred_categories)
        precision = precision_score(actual_categories, pred_categories, labels=CATEGORY_ORDER, average='weighted', zero_division=0)
        recall = recall_score(actual_categories, pred_categories, labels=CATEGORY_ORDER, average='weighted', zero_division=0)
        f1 = f1_score(actual_categories, pred_categories, labels=CATEGORY_ORDER, average='weighted', zero_division=0)
        
        # Per-class metrics
        class_report = classification_report(actual_categories, pred_categories, labels=CATEGORY_ORDER, output_dict=True, zero_division=0)
        
        # Regression metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        
        epsilon = 1e-8
        mape = np.mean(np.abs((actuals - predictions) / (actuals + epsilon))) * 100
        
        # Per-class metrics for frontend
        per_class_metrics = {}
        for category in CATEGORY_ORDER:
            if category in class_report:
                per_class_metrics[category] = {
                    'precision': round(class_report[category]['precision'] * 100, 1),
                    'recall': round(class_report[category]['recall'] * 100, 1),
                    'f1_score': round(class_report[category]['f1-score'] * 100, 1),
                    'support': class_report[category]['support']
                }
        
        # Category distribution
        category_distribution = {}
        for category in CATEGORY_ORDER:
            category_distribution[category] = {
                'actual': actual_categories.count(category),
                'predicted': pred_categories.count(category)
            }
        
        # Prepare response
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
        
        # Save evaluation results
        os.makedirs('evaluation_results', exist_ok=True)
        filename = f"evaluation_results/eval_{station}_{direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            import json
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
                        import json
                        data = json.load(file)
                        # Filter by station/direction if specified
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
            
            # Get latest evaluation for summary
            latest_eval = None
            if os.path.exists('evaluation_results'):
                eval_files = [f for f in os.listdir('evaluation_results') if f.endswith('.json')]
                if eval_files:
                    try:
                        with open(os.path.join('evaluation_results', eval_files[-1]), 'r') as f:
                            import json
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
        
        # Check if columns are shifted
        if 'target_time' in df.columns:
            sample = df['target_time'].iloc[0] if len(df) > 0 else None
            if sample and isinstance(sample, str) and sample in ['Northbound', 'Southbound']:
                proper_columns = ['station', 'direction', 'target_time', 'predicted', 'actual', 
                                  'total_passengers', 'station_max', 'absolute_error', 
                                  'percentage_error', 'verdict', 'timestamp']
                raw_df = pd.read_csv(filename, header=None)
                if len(raw_df.columns) >= len(proper_columns):
                    raw_df.columns = proper_columns[:len(raw_df.columns)]
                    df = raw_df
        
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
        
        # Try to get actual congestion from historical data
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
            response["prediction"]["category"] = get_congestion_category(pred_congestion)
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ============= RUN AUTO TESTS =============
@model_perf_bp.route('/model/run-auto-tests', methods=['POST'])
def run_auto_tests():
    """Run auto-tests on 2025 data using station capacity"""
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
        df['direction'] = df.apply(infer_direction_correct, axis=1)
        
        station_numbers_reverse = {
            1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
            5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
            9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
        }
        
        # ========== STATION CAPACITIES (USE THIS FOR ALL) ==========
        MRT3_CAPACITY = {
            "North Ave": 1142, "Quezon Ave": 1195, "Kamuning": 1364, 
            "Cubao": 1747, "Santolan": 1306, "Ortigas": 1331,
            "Shaw Blvd": 1619, "Boni Ave": 1417, "Guadalupe": 1301,
            "Buendia": 1645, "Ayala Ave": 1222, "Magallanes": 1202,
            "Taft": 720
        }
        
        results = []
        
        # Import the prediction function
        from routes.api_predict import get_directional_prediction as api_prediction
        
        for station_name in station_numbers_reverse.values():
            for direction in ['Northbound', 'Southbound']:
                model_key = f"{station_name}_{direction}"
                if model_key not in directional_models:
                    print(f"⚠️ No model: {model_key}")
                    continue
                
                print(f"\n📊 Processing: {station_name} {direction}")
                station_num = [k for k, v in station_numbers_reverse.items() if v == station_name][0]
                
                station_df = get_station_data_for_direction(df, station_num, direction)
                station_df = station_df[station_df['direction'] == direction]
                
                if len(station_df) < 100:
                    print(f"  ⚠️ Only {len(station_df)} records (need 100+)")
                    continue
                
                station_df['hour_timestamp'] = station_df['datetime'].dt.floor('h')
                
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
                
                # ========== ALWAYS USE STATION CAPACITY, NOT PER_DIRECTION_MAX ==========
                station_capacity = MRT3_CAPACITY.get(station_name, 1000)
                print(f"  📊 Using station capacity: {station_capacity} passengers/hour")
                
                # Calculate actual congestion based on station capacity
                hourly['congestion'] = (hourly['TotalPassenger'] / station_capacity * 100).clip(0, 100)
                hourly = hourly.sort_values('hour_timestamp')
                
                available_indices = list(range(24, len(hourly)))
                if not available_indices:
                    continue
                
                # Use up to 30 samples per station-direction
                max_samples = min(30, len(available_indices))
                num_tests = max(10, max_samples)
                
                print(f"  📅 Testing {num_tests} dates (out of {len(available_indices)} available)")
                
                test_indices = sorted(random.sample(available_indices, min(num_tests, len(available_indices))))
                
                for idx in test_indices:
                    target = hourly.iloc[idx]
                    target_time = target['hour_timestamp']
                    
                    try:
                        pred_result = api_prediction(
                            station_name, direction, target_time
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
                                'station_capacity': station_capacity,
                                'absolute_error': round(abs_error, 1),
                                'percentage_error': round((abs_error / max(actual_congestion, 0.1) * 100), 1),
                                'verdict': verdict,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'predicted_category': get_congestion_category(pred_congestion),
                                'actual_category': get_congestion_category(actual_congestion)
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