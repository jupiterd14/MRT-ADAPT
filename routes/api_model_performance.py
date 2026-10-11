"""
Model Performance Routes - LSTM Testing & Visualization
Handles: Manual predictions, batch uploads, metrics, chart data, evaluation metrics

FIXED:
- No matplotlib / seaborn imports (heavy, unused)
- Lazy TensorFlow import (cold start friendly)
- Module-level MAX_ROWS_TO_READ
- Unified station mapping (handles terminals correctly)
- get_p90_for_station delegates to api_predict (single source of truth)
- ALL congestion calculations now use P90 (NOT capacity)
- calculate_congestion() now P90-based
- predict_single() actual calculation now P90-based
- calculate_commuter_congestion() delegates to api_predict.get_p90_percentile
- upload_batch_test() reads FULL file for user uploads (no row cap)
- upload_batch_test() logs hour distribution for debugging
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import pandas as pd
import numpy as np
import os
import pickle
import random
import json
import logging

# ============================================================
# LAZY TENSORFLOW IMPORT (Render-friendly)
# ============================================================
class _LazyTF:
    _mod = None
    def __getattr__(self, name):
        if self._mod is None:
            import tensorflow as _t
            self._mod = _t
        return getattr(self._mod, name)

tf = _LazyTF()
logging.getLogger('tensorflow').setLevel(logging.ERROR)

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

# ============================================================
# MODULE-LEVEL CONSTANTS
# ============================================================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_ROWS_TO_READ = 20000   # used ONLY for auto-tests (server-side, memory-sensitive)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('test_results', exist_ok=True)
os.makedirs('evaluation_results', exist_ok=True)

# ============================================================
# STATIONS LIST (must match api_predict.py)
# ============================================================
STATIONS = [
    "North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan",
    "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia",
    "Ayala Ave", "Magallanes", "Taft"
]

# ============================================================
# STATION MAPPING (single source of truth)
# ============================================================
STATION_NUMBER_MAP = {
    1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
    5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
    9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
}
def get_station_name_fixed(row, direction=None):
    """
    Assign station name based on direction-aware convention.
    
    Northbound: passengers ENTER the station (except North Ave where they exit)
    Southbound: passengers EXIT the station (except North Ave where they enter)
    """
    entry = row['StationEntry']
    exit_station = row['StationExit']
    
    # If direction is inferred, use direction-aware convention
    if direction == 'Northbound':
        # Passengers board at StationEntry (going north)
        # Exception: North Ave is the terminal, passengers exit there
        if exit_station == 1:
            return "North Ave"
        return STATION_NUMBER_MAP.get(entry)
    elif direction == 'Southbound':
        # Passengers exit at StationExit (going south)
        # Exception: Taft is the terminal, passengers enter there
        if entry == 13:
            return "Taft"
        return STATION_NUMBER_MAP.get(exit_station)
    else:
        # Fallback
        if entry == 1 or exit_station == 1:
            return "North Ave"
        if entry == 13 or exit_station == 13:
            return "Taft"
        return STATION_NUMBER_MAP.get(exit_station)
# ============================================================
# HISTORICAL PEAKS (reference only, NOT for congestion)
# ============================================================
HISTORICAL_PEAKS = {}


def get_p90_for_station(station_name, direction):
    """
    Get P90 percentile for a station-direction.
    Delegates to api_predict.get_p90_percentile for a single source of truth.
    """
    try:
        from routes.api_predict import get_p90_percentile
        return get_p90_percentile(station_name, direction)
    except Exception as e:
        print(f"⚠️ Could not get P90 for {station_name} {direction}: {e}")
        # Last resort fallback
        return MRT3_PLATFORM_CAPACITY.get(station_name, 1000)


# ============================================================
# DRIFT DETECTION
# ============================================================
@model_perf_bp.route('/model/drift-detection', methods=['POST'])
def detect_drift():
    """
    Compare uploaded data (e.g. 2026) against baseline (e.g. 2025)
    to detect model drift.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

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
        baseline_errors = []
        if os.path.exists('test_results'):
            baseline_files = [f for f in os.listdir('test_results')
                              if f.startswith('full_2025_test_')]
            if baseline_files:
                baseline_df = pd.read_csv(os.path.join('test_results', baseline_files[-1]))
                if 'absolute_error' in baseline_df.columns:
                    baseline_errors = baseline_df['absolute_error'].tolist()

        # Process uploaded data
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

        if baseline_errors and current_errors:
            baseline_mean = float(np.mean(baseline_errors))
            current_mean = float(np.mean(current_errors))
            drift_percentage = ((current_mean - baseline_mean) / baseline_mean) * 100

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
                'message': f'Drift analysis complete for {len(current_errors)} samples',
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


# ============================================================
# 2025 DATA AGGREGATION (for evaluation)
# ============================================================
def load_and_aggregate_2025_data():
    """Load 2025 CSV and aggregate to hourly totals per station-direction."""
    data_file = find_data_file()
    if not data_file:
        print("❌ 2025 data file not found")
        return None

    print(f"📁 Loading 2025 data from: {data_file}")
    df = pd.read_csv(data_file)

    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    df = df.dropna(subset=['datetime'])

    df = df[(df['datetime'].dt.hour >= 5) & (df['datetime'].dt.hour < 22)]

    def infer_dir(row):
        if row['StationEntry'] < row['StationExit']:
            return 'Southbound'
        elif row['StationEntry'] > row['StationExit']:
            return 'Northbound'
        return 'Unknown'

    df['direction'] = df.apply(infer_dir, axis=1)
    df = df[df['direction'] != 'Unknown']

    df['station'] = df.apply(get_station_name_fixed, axis=1)
    df = df.dropna(subset=['station'])

    df['hour_timestamp'] = df['datetime'].dt.floor('h')
    df_hourly = df.groupby(['station', 'direction', 'hour_timestamp']).agg({
        'TotalPassenger': 'sum'
    }).reset_index()

    print(f"✅ Aggregated {len(df):,} rows → {len(df_hourly):,} hourly records")
    return df_hourly


# ============================================================
# HISTORICAL PEAKS LOADER
# ============================================================
def load_historical_peaks():
    """Load historical peaks for REFERENCE ONLY."""
    global HISTORICAL_PEAKS

    peaks_file = 'historical_peaks.json'
    if os.path.exists(peaks_file):
        try:
            with open(peaks_file, 'r') as f:
                HISTORICAL_PEAKS = json.load(f)
            print(f"✅ Loaded historical peaks from {peaks_file}")
            return HISTORICAL_PEAKS
        except Exception as e:
            print(f"⚠️ Could not load peaks file: {e}")

    # Compute from data if not cached
    try:
        from services.feature_engineering import load_data_fast, get_station_dataframe
        df = load_data_fast()
        if df is None:
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

        try:
            with open(peaks_file, 'w') as f:
                json.dump(HISTORICAL_PEAKS, f, indent=2)
            print(f"✅ Saved historical peaks to {peaks_file}")
        except Exception as e:
            print(f"⚠️ Could not save peaks file: {e}")
    except Exception as e:
        print(f"⚠️ Could not compute historical peaks: {e}")

    return HISTORICAL_PEAKS


# ============================================================
# CORRECTION FACTORS
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
        CORRECTION_FACTORS = {}


# ============================================================
# CONGESTION CATEGORIES
# ============================================================
CATEGORY_ORDER = ['Light', 'Moderate', 'Congested', 'Severe']


def get_congestion_category(congestion_value):
    if congestion_value > 80:
        return 'Severe'
    elif congestion_value > 45:
        return 'Congested'
    elif congestion_value > 25:
        return 'Moderate'
    else:
        return 'Light'


def get_capacity(station_name):
    """Platform capacity - for reference only, NOT for congestion"""
    return MRT3_PLATFORM_CAPACITY.get(station_name, 1000)


def calculate_congestion(passenger_count, station_name, direction='Northbound'):
    """
    Calculate congestion % using P90 (NOT capacity).
    Single source of truth: api_predict.get_p90_percentile
    """
    p90 = get_p90_for_station(station_name, direction)
    if p90 <= 0:
        p90 = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
    congestion = (passenger_count / p90) * 100
    return min(congestion, 100)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def find_data_file():
    """Find the data file in various possible locations."""
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


# ============================================================
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# ============================================================
def add_cyclical_time_features(df):
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
    time_decimal = df['time_decimal']
    df['is_morning_rush'] = ((time_decimal >= 7.0) & (time_decimal <= 9.0)).astype(np.int8)
    df['is_evening_rush'] = ((time_decimal >= 17.0) & (time_decimal <= 19.0)).astype(np.int8)
    return df


def smart_data_cleaner(df):
    time_decimal = df['time_decimal']
    passenger_count = df['TotalPassenger']
    df['is_maintenance_record'] = ((time_decimal < 5.0) & (passenger_count < 10)).astype(np.int8)
    df['is_extended_hours'] = ((time_decimal >= 22.0) & (time_decimal < 23.0) & (passenger_count >= 10)).astype(np.int8)
    df.loc[df['is_maintenance_record'] == 1, 'congestion'] = 0
    return df


def infer_direction_correct(row):
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
def get_station_data_for_direction(df, station_num, direction):
    """Get correct data for station and direction (handles terminals).
    
    Matches the convention used in services/feature_engineering.py:
    - Northbound: passengers ENTER the station (except at North Ave, where they exit)
    - Southbound: passengers EXIT the station (except at North Ave, where they enter)
    """
    station_name = STATION_NUMBER_MAP.get(station_num, "Unknown")

    if station_name == "North Ave":
        # Terminal: Northbound = passengers arriving (exit), Southbound = passengers departing
        if direction == 'Northbound':
            return df[df['StationExit'] == station_num].copy()
        else:
            return df[df['StationEntry'] == station_num].copy()
    elif station_name == "Taft":
        # Terminal: Northbound = passengers departing, Southbound = passengers arriving
        if direction == 'Northbound':
            return df[df['StationEntry'] == station_num].copy()
        else:
            return df[df['StationExit'] == station_num].copy()
    else:
        # Middle stations: Northbound = passengers entering to go north,
        # Southbound = passengers exiting (having come from the north)
        if direction == 'Northbound':
            return df[df['StationEntry'] == station_num].copy()   # ✅ FIXED
        else:
            return df[df['StationExit'] == station_num].copy()    # ✅ FIXED
        
def get_historical_peak(station_name, direction):
    key = f"{station_name}_{direction}"
    peak_data = HISTORICAL_PEAKS.get(key)
    if peak_data:
        if isinstance(peak_data, dict):
            return peak_data.get("peak", peak_data.get("absolute_max", 5000))
        return peak_data
    return 5000


def calculate_commuter_congestion(passenger_count, station, direction):
    """
    Calculate 0-100% congestion score based on P90 percentile.
    Delegates to api_predict.get_p90_percentile for single source of truth.
    """
    p90 = get_p90_for_station(station, direction)

    if p90 <= 0:
        p90 = MRT3_PLATFORM_CAPACITY.get(station, 1000)

    congestion = (passenger_count / p90) * 100
    congestion = max(0, min(congestion, 100))
    return round(congestion, 1)


# ============================================================
# DATA LOADING (optimized)
# ============================================================
def load_data_optimized(filepath, max_rows=None):
    """
    Load CSV data. If max_rows is None, reads the entire file.
    Used by run_auto_tests() with a row cap; upload_batch_test() bypasses this
    and reads the full file directly.
    """
    try:
        if not os.path.exists(filepath):
            print(f"⚠️ File not found: {filepath}")
            return None

        kwargs = {
            'usecols': ['TotalPassenger', 'Date', 'Time', 'StationEntry', 'StationExit']
        }
        if max_rows is not None:
            kwargs['nrows'] = max_rows
            print(f"📊 Loading {filepath} (max {max_rows} rows)...")
        else:
            print(f"📊 Loading full file: {filepath}")

        df = pd.read_csv(filepath, **kwargs)
        print(f"✅ Loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


# ============================================================
# EVALUATION METRICS
# ============================================================
@model_perf_bp.route('/model/evaluate', methods=['POST'])
def evaluate_model():
    """Run model evaluation and return confusion matrix + classification metrics."""
    try:
        data = request.json or {}
        station = data.get('station', 'all')
        direction = data.get('direction', 'both')

        if not os.path.exists('test_results'):
            return jsonify({
                "success": False,
                "error": "No test results found. Run auto-tests first.",
                "total_samples": 0
            }), 400

        all_results = []
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
        precision = precision_score(actual_categories, pred_categories,
                                    labels=CATEGORY_ORDER, average='weighted', zero_division=0)
        recall = recall_score(actual_categories, pred_categories,
                              labels=CATEGORY_ORDER, average='weighted', zero_division=0)
        f1 = f1_score(actual_categories, pred_categories,
                      labels=CATEGORY_ORDER, average='weighted', zero_division=0)

        class_report = classification_report(
            actual_categories, pred_categories,
            labels=CATEGORY_ORDER, output_dict=True, zero_division=0
        )

        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)

        valid_mask = actuals > 1.0
        if valid_mask.sum() > 0:
            mape = np.mean(np.abs((actuals[valid_mask] - predictions[valid_mask]) / actuals[valid_mask])) * 100
        else:
            mape = 0.0

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

        return jsonify({"success": True, "data": evaluation_data})

    except Exception as e:
        import traceback
        print(f"❌ Evaluation error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============================================================
# PER-CLASS METRICS (TP/FP/FN/TN + Precision/Recall/F1)
# ============================================================
@model_perf_bp.route('/model/evaluate/per-class', methods=['GET', 'POST'])
def evaluate_per_class():
    """
    Returns a 4-tier × 9-metric breakdown of the confusion matrix:

        Tier | TP | FP | FN | TN | Precision | Recall | F1 | Support
        ─────┼────┼────┼────┼────┼───────────┼────────┼────┼────────
        Light
        Moderate
        Congested
        Severely Congested
        ─────┼────┼────┼────┼────┼───────────┼────────┼────┼────────
        Macro Avg
        Weighted Avg

    Uses the same test results as /model/evaluate.
    Does NOT change any existing behavior.
    """
    try:
        # Accept filters same as evaluate_model
        data = request.json if request.is_json else {}
        data.update(request.args)
        station = data.get('station', 'all')
        direction = data.get('direction', 'both')

        # Load test results (same as evaluate_model)
        if not os.path.exists('test_results'):
            return jsonify({
                "success": False,
                "error": "No test results found. Run /model/upload/batch first."
            }), 400

        all_results = []
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
                except Exception:
                    continue

        if not all_results:
            return jsonify({
                "success": False,
                "error": "No test data for the selected filters."
            }), 400

        combined = pd.concat(all_results, ignore_index=True)

        for col in ['predicted', 'actual']:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors='coerce')
        combined = combined.dropna(subset=['predicted', 'actual'])

        if len(combined) < 10:
            return jsonify({
                "success": False,
                "error": f"Only {len(combined)} valid samples."
            }), 400

        # ── Categorize (uses existing CATEGORY_ORDER + get_congestion_category) ──
        predictions = combined['predicted'].values
        actuals = combined['actual'].values

        pred_cats = [get_congestion_category(p) for p in predictions]
        actual_cats = [get_congestion_category(a) for a in actuals]

        # Confusion matrix with consistent label order
        labels = CATEGORY_ORDER  # ['Light', 'Moderate', 'Congested', 'Severe']
        cm = confusion_matrix(actual_cats, pred_cats, labels=labels)

        n = cm.sum()
        per_class = {}

        for i, label in enumerate(labels):
            tp = int(cm[i, i])
            fn = int(cm[i, :].sum() - tp)      # actual = label, predicted != label
            fp = int(cm[:, i].sum() - tp)      # predicted = label, actual != label
            tn = int(n - tp - fp - fn)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            support   = int(cm[i, :].sum())

            per_class[label] = {
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': round(precision * 100, 2),
                'recall':    round(recall * 100, 2),
                'f1':        round(f1 * 100, 2),
                'support':   support,
            }

        # ── Macro average (unweighted) ──
        macro = {
            'tp': int(np.mean([v['tp'] for v in per_class.values()])),
            'fp': int(np.mean([v['fp'] for v in per_class.values()])),
            'fn': int(np.mean([v['fn'] for v in per_class.values()])),
            'tn': int(np.mean([v['tn'] for v in per_class.values()])),
            'precision': round(np.mean([v['precision'] for v in per_class.values()]), 2),
            'recall':    round(np.mean([v['recall'] for v in per_class.values()]), 2),
            'f1':        round(np.mean([v['f1'] for v in per_class.values()]), 2),
            'support':   int(sum(v['support'] for v in per_class.values())),
        }

        # ── Weighted average (by support) ──
        total_support = sum(v['support'] for v in per_class.values()) or 1
        weighted = {
            'tp': int(np.sum([v['tp'] * v['support'] / total_support for v in per_class.values()])),
            'fp': int(np.sum([v['fp'] * v['support'] / total_support for v in per_class.values()])),
            'fn': int(np.sum([v['fn'] * v['support'] / total_support for v in per_class.values()])),
            'tn': int(np.sum([v['tn'] * v['support'] / total_support for v in per_class.values()])),
            'precision': round(sum(v['precision'] * v['support'] for v in per_class.values()) / total_support, 2),
            'recall':    round(sum(v['recall']    * v['support'] for v in per_class.values()) / total_support, 2),
            'f1':        round(sum(v['f1']        * v['support'] for v in per_class.values()) / total_support, 2),
            'support':   int(total_support),
        }

        # ── Overall accuracy ──
        overall_accuracy = round(100 * sum(per_class[l]['tp'] for l in labels) / n, 2)

        # ── Build the 9-row × 6-column table ──
        # Rows: Tier, TP, FP, FN, TN, Precision, Recall, F1, Support
        # Columns: Light, Moderate, Congested, Severely Congested, Macro, Weighted
        column_labels = ['Light', 'Moderate', 'Congested', 'Severely Congested',
                         'Macro Average', 'Weighted Average']

        def safe(label):
            """Return per-class metrics with renamed 'Severe' → 'Severely Congested'."""
            src = 'Severe' if label == 'Severely Congested' else label
            return per_class.get(src, {})

        columns = [
            safe('Light'),
            safe('Moderate'),
            safe('Congested'),
            safe('Severely Congested'),
            macro,
            weighted,
        ]

        rows = []
        for metric in ['tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1', 'support']:
            row_label = {
                'tp': 'TP', 'fp': 'FP', 'fn': 'FN', 'tn': 'TN',
                'precision': 'Precision (%)',
                'recall':    'Recall (%)',
                'f1':        'F1-Score (%)',
                'support':   'Support',
            }[metric]
            rows.append({
                'metric': row_label,
                'values': [col.get(metric, 0) for col in columns],
            })

        return jsonify({
            'success': True,
            'data': {
                'column_labels': column_labels,
                'rows': rows,
                'overall': {
                    'accuracy': overall_accuracy,
                    'total_samples': int(n),
                    'station_filter': station,
                    'direction_filter': direction,
                },
                'confusion_matrix': {
                    'labels': labels,
                    'matrix': cm.tolist(),
                },
                'per_class_raw': per_class,   # for debugging
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
        
# ============================================================
# CONFUSION MATRIX — TIER-ROW TABLE (matches UI layout)
# ============================================================
@model_perf_bp.route('/model/evaluation/confusion-table', methods=['GET', 'POST'])
def get_confusion_table():
    """
    Returns a table shaped as:
        Rows    = [Light, Moderate, Congested, Severely Congested, Macro Avg, Weighted Avg]
        Columns = [Tier, TP, FP, FN, TN, Precision, Recall, F1, Support]

    Rules:
      - Tier rows (Light/Moderate/Congested/Severe): values in **percent** (0-100)
      - Macro / Weighted rows: values in **decimal float** (0.0-1.0)
      - Macro / Weighted rows: TP/FP/FN/TN are null (rendered as em-dash on UI)
    """
    try:
        data = request.json if request.is_json else {}
        data.update(request.args)

        with current_app.test_request_context(
            '/model/evaluate/per-class',
            method='POST',
            json=data
        ):
            resp = evaluate_per_class()
            if hasattr(resp, 'get_json'):
                payload = resp.get_json()
            else:
                payload = resp[0].get_json()

        if not payload.get('success'):
            return jsonify(payload), 400

        raw = payload['data']
        col_labels = raw['column_labels']              # 6 columns
        rows_by_metric = raw['rows']                   # 8 metric rows

        lookup = {r['metric']: r['values'] for r in rows_by_metric}

        def get_metric(key, idx):
            return lookup.get(key, [0] * 6)[idx]

        # ── 4 tier rows (values in PERCENT) ──
        tier_rows = []
        for idx, tier_label in enumerate(col_labels[:4]):
            tier_rows.append({
                'tier': tier_label,
                'TP': get_metric('TP', idx),
                'FP': get_metric('FP', idx),
                'FN': get_metric('FN', idx),
                'TN': get_metric('TN', idx),
                'Precision': get_metric('Precision (%)', idx),   # percent
                'Recall':    get_metric('Recall (%)', idx),      # percent
                'F1':        get_metric('F1-Score (%)', idx),    # percent
                'Support':   get_metric('Support', idx),
            })

        # ── Macro row (values in DECIMAL 0-1, counts are null) ──
        tier_rows.append({
            'tier': 'Macro Average',
            'TP': None, 'FP': None, 'FN': None, 'TN': None,
            'Precision': round(get_metric('Precision (%)', 4) / 100.0, 4),
            'Recall':    round(get_metric('Recall (%)',    4) / 100.0, 4),
            'F1':        round(get_metric('F1-Score (%)',  4) / 100.0, 4),
            'Support':   get_metric('Support', 4),
        })

        # ── Weighted row (values in DECIMAL 0-1, counts are null) ──
        tier_rows.append({
            'tier': 'Weighted Average',
            'TP': None, 'FP': None, 'FN': None, 'TN': None,
            'Precision': round(get_metric('Precision (%)', 5) / 100.0, 4),
            'Recall':    round(get_metric('Recall (%)',    5) / 100.0, 4),
            'F1':        round(get_metric('F1-Score (%)',  5) / 100.0, 4),
            'Support':   get_metric('Support', 5),
        })

        return jsonify({
            'success': True,
            'data': {
                'rows': tier_rows,
                'accuracy': raw['overall']['accuracy'],
                'total_samples': raw['overall']['total_samples'],
                'station_filter': raw['overall']['station_filter'],
                'direction_filter': raw['overall']['direction_filter'],
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
        
@model_perf_bp.route('/model/evaluation/history', methods=['GET'])
def get_evaluation_history():
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
                except Exception:
                    continue

        evaluations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return jsonify({"success": True, "data": evaluations})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# HEALTH CHECK
# ============================================================
@model_perf_bp.route('/health', methods=['GET'])
def health_check():
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


# ============================================================
# DEBUG ENDPOINTS
# ============================================================
@model_perf_bp.route('/debug/analyze-scalers', methods=['GET'])
def debug_analyze_scalers():
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
            from services.feature_engineering import FEATURE_COLS
            result["analysis"]["feature_scaler"] = {
                "feature_count": len(FEATURE_COLS),
                "features": FEATURE_COLS,
                "data_min": feature_scaler.data_min_.tolist(),
                "data_max": feature_scaler.data_max_.tolist(),
                "interpretation": "Features scaled via MinMaxScaler to [0,1]"
            }
        else:
            result["analysis"]["feature_scaler"] = {"error": "Feature scaler not found"}

        target_scaler = directional_scalers.get(f'{model_key}_target')
        if target_scaler is not None:
            result["analysis"]["target_scaler"] = {
                "target_feature": "TotalPassenger (raw passenger counts)",
                "data_min": float(target_scaler.data_min_[0]),
                "data_max": float(target_scaler.data_max_[0]),
                "interpretation": f"Predicts passenger counts from {target_scaler.data_min_[0]:.0f} to {target_scaler.data_max_[0]:.0f}"
            }
        else:
            result["analysis"]["target_scaler"] = {"error": "Target scaler not found"}

        p90 = get_p90_for_station(station, direction)
        result["analysis"]["p90_percentile"] = {
            "value": round(p90, 0),
            "source": "Historical data (90th percentile) via api_predict",
            "note": f"Congestion = (passenger_count / {round(p90, 0)}) * 100"
        }

        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@model_perf_bp.route('/debug/trace-prediction/<station>/<direction>', methods=['GET'])
def debug_trace_prediction(station, direction):
    try:
        from routes.api_predict import get_directional_prediction as api_prediction

        station = station.replace('%20', ' ')
        direction = direction.capitalize()
        target_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        model_key = f"{station}_{direction}"

        target_scaler = directional_scalers.get(f'{model_key}_target')
        pred_congestion = api_prediction(station, direction, target_time)
        p90 = get_p90_for_station(station, direction)

        response = {
            "station": station,
            "direction": direction,
            "target_time": target_time.isoformat(),
            "model_key": model_key,
            "p90_percentile": round(p90, 0),
            "prediction_flow": {}
        }

        if target_scaler is not None:
            response["target_scaler_info"] = {
                "data_min": float(target_scaler.data_min_[0]),
                "data_max": float(target_scaler.data_max_[0]),
            }

        if pred_congestion is not None:
            pred_passengers = (pred_congestion / 100) * p90
            response["prediction_flow"] = {
                "step_1_model_output": f"{pred_congestion:.1f}%",
                "step_2_passenger_count": f"{pred_passengers:.0f} passengers",
                "step_3_congestion_formula": f"({pred_passengers:.0f} / {p90:.0f}) * 100 = {pred_congestion:.1f}%",
                "conclusion": "Model → inverse_transform → passenger count → congestion % (using P90)"
            }
        else:
            response["prediction_flow"] = {"error": "Prediction returned None"}

        return jsonify(response)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@model_perf_bp.route('/model/evaluation/summary', methods=['GET'])
def get_evaluation_summary():
    try:
        station = request.args.get('station', 'all')
        direction = request.args.get('direction', 'both')

        if not os.path.exists('evaluation_results'):
            return jsonify({"success": True, "data": {"message": "No evaluation data found"}})

        all_evals = []
        for f in os.listdir('evaluation_results'):
            if f.endswith('.json'):
                try:
                    with open(os.path.join('evaluation_results', f), 'r') as file:
                        data = json.load(file)
                        if station != 'all' and data.get('station') != station:
                            continue
                        if direction != 'both' and data.get('direction') != direction:
                            continue
                        all_evals.append(data)
                except Exception:
                    continue

        if not all_evals:
            return jsonify({"success": True, "data": {"message": "No evaluation data found"}})

        total_samples = sum(e.get('total_samples', 0) for e in all_evals)
        avg_accuracy = float(np.mean([e.get('accuracy', 0) for e in all_evals]))
        avg_f1 = float(np.mean([e.get('f1_weighted', 0) for e in all_evals]))
        avg_mae = float(np.mean([e.get('regression_metrics', {}).get('mae', 0) for e in all_evals]))
        avg_r2 = float(np.mean([e.get('regression_metrics', {}).get('r2', 0) for e in all_evals]))

        per_station = {}
        for e in all_evals:
            key = f"{e.get('station', 'unknown')}_{e.get('direction', 'unknown')}"
            per_station[key] = {
                'accuracy': e.get('accuracy', 0),
                'f1': e.get('f1_weighted', 0),
                'mae': e.get('regression_metrics', {}).get('mae', 0),
                'samples': e.get('total_samples', 0)
            }

        return jsonify({
            "success": True,
            "data": {
                "total_evaluations": len(all_evals),
                "total_samples": total_samples,
                "average_accuracy": round(avg_accuracy, 2),
                "average_f1_score": round(avg_f1, 2),
                "average_mae": round(avg_mae, 2),
                "average_r2": round(avg_r2, 4),
                "per_station": per_station,
                "latest_evaluation": all_evals[-1] if all_evals else None
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@model_perf_bp.route('/debug/check-models', methods=['GET'])
def debug_check_models():
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
    target_time = datetime.now()
    try:
        from routes.api_predict import get_directional_prediction as api_prediction
        result = api_prediction(station, direction, target_time)
        p90 = get_p90_for_station(station, direction)

        return jsonify({
            "success": True,
            "station": station,
            "direction": direction,
            "target_time": target_time.isoformat(),
            "prediction": result,
            "p90_percentile": round(p90, 0)
        })
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============================================================
# PERFORMANCE METRICS
# ============================================================
@model_perf_bp.route('/model/performance/metrics', methods=['GET'])
def get_performance_metrics():
    try:
        if not os.path.exists('test_results'):
            return jsonify({
                "success": True,
                "data": {"message": "No test results yet", "total_predictions": 0, "overall_r2": 0.0}
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

            # ✅ Compute overall R² across all samples
            try:
                _valid = combined.dropna(subset=['predicted', 'actual'])
                if len(_valid) >= 2:
                    _r2_val = r2_score(_valid['actual'].values, _valid['predicted'].values)
                    overall_r2 = round(float(_r2_val), 3)
                else:
                    overall_r2 = 0.0
            except Exception:
                overall_r2 = 0.0

            latest_eval = None
            if os.path.exists('evaluation_results'):
                eval_files = [f for f in os.listdir('evaluation_results') if f.endswith('.json')]
                if eval_files:
                    try:
                        with open(os.path.join('evaluation_results', eval_files[-1]), 'r') as f:
                            latest_eval = json.load(f)
                    except Exception:
                        pass

            per_station = {}
            if all(c in combined.columns for c in ['station', 'direction', 'absolute_error']):
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
                "overall_r2": overall_r2,                                     # ← ADDED
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
            metrics = {
                "total_predictions": 0,
                "overall_r2": 0.0,
                "message": "No valid test results found"
            }

        return jsonify({"success": True, "data": metrics})
    except Exception as e:
        import traceback
        print(f"❌ Error in get_performance_metrics: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
# ============================================================
# CHART DATA
# ============================================================
@model_perf_bp.route('/model/chart/data', methods=['GET'])
def get_chart_data():
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
                    for col in ['predicted', 'actual', 'absolute_error']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

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
            combined['target_time_dt'] = pd.to_datetime(combined['target_time'], errors='coerce')
            combined = combined.dropna(subset=['target_time_dt']).sort_values('target_time_dt')
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
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# STATION DETAILS
# ============================================================
@model_perf_bp.route('/model/station-details', methods=['GET'])
def get_station_details():
    try:
        station = request.args.get('station', '').strip()
        direction = request.args.get('direction', '').strip().capitalize()

        if not station or not direction:
            return jsonify({"success": False, "error": "Missing station or direction"}), 400

        if direction not in ['Northbound', 'Southbound']:
            return jsonify({"success": False, "error": f"Invalid direction: {direction}"}), 400

        import glob

        # ── Handle "all stations" by aggregating per-station files ──
        if station.lower() == 'all':
            pattern = f"test_results/*_{direction}_results.csv"
            files = glob.glob(pattern)
            if not files:
                return jsonify({"success": True, "data": []}), 200
            dfs = []
            for f in files:
                try:
                    dfs.append(pd.read_csv(f))
                except Exception:
                    continue
            if not dfs:
                return jsonify({"success": True, "data": []}), 200
            df = pd.concat(dfs, ignore_index=True)
        else:
            filename = f"test_results/{station}_{direction}_results.csv"
            if not os.path.exists(filename):
                return jsonify({"success": True, "data": []}), 200
            df = pd.read_csv(filename)

        if df.empty:
            return jsonify({"success": True, "data": []}), 200

        # ---- Numeric conversion ----
        for col in ['predicted', 'actual', 'absolute_error', 'percentage_error']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # ---- Sort by target_time descending ----
        if 'target_time' in df.columns:
            try:
                df['target_time_dt'] = pd.to_datetime(df['target_time'], errors='coerce')
                df = df.dropna(subset=['target_time_dt']).sort_values('target_time_dt', ascending=False)
            except Exception:
                pass

        # ---- Pick columns to return ----
        return_cols = [
            c for c in ['target_time', 'predicted', 'actual', 'absolute_error',
                        'percentage_error', 'verdict']
            if c in df.columns
        ]

        results = df[return_cols].head(50).to_dict('records')

        # ---- Clean up NaN / numpy types / round numeric fields ----
        for record in results:
            for key, value in list(record.items()):
                if pd.isna(value):
                    record[key] = None
                elif hasattr(value, 'item'):
                    value = value.item()

                if key in ('predicted', 'actual', 'absolute_error', 'percentage_error'):
                    try:
                        record[key] = round(float(value), 1)
                    except (TypeError, ValueError):
                        record[key] = None
                else:
                    record[key] = value

        return jsonify({"success": True, "data": results})
    except Exception as e:
        import traceback
        print(f"❌ Error in get_station_details: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
# ============================================================
# DEBUG CSV ENDPOINTS
# ============================================================
@model_perf_bp.route('/model/debug-csv', methods=['GET'])
def debug_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

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
        return jsonify({'success': False, 'error': str(e)}), 500


@model_perf_bp.route('/debug/check-2025-csv', methods=['GET'])
def debug_check_2025_csv():
    try:
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

        df_sample = pd.read_csv(data_file, nrows=10)
        file_size = os.path.getsize(data_file) / (1024 * 1024)
        total_rows = sum(1 for _ in open(data_file)) - 1

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
            'data_types': {col: str(df_sample[col].dtype) for col in df_sample.columns}
        })
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@model_perf_bp.route('/debug/check-raw-mrt-data', methods=['GET'])
def debug_check_raw_mrt_data():
    try:
        data_file = find_data_file()
        if not data_file:
            return jsonify({'success': False, 'error': 'Data file not found'}), 404

        df = pd.read_csv(data_file, nrows=1000)

        raw_mrt_columns = ['TotalPassenger', 'StationEntry', 'StationExit', 'Date', 'Time']
        is_raw_mrt = all(col in df.columns for col in raw_mrt_columns)

        result = {
            'is_raw_mrt_format': is_raw_mrt,
            'columns': df.columns.tolist(),
            'sample_data': df.head(10).to_dict('records'),
            'statistics': {}
        }

        if is_raw_mrt:
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

            df['inferred_direction'] = df.apply(infer_direction_correct, axis=1)
            result['statistics']['direction_distribution'] = df['inferred_direction'].value_counts().to_dict()
            result['direction_samples'] = df[['StationEntry', 'StationExit', 'inferred_direction']].head(20).to_dict('records')

        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@model_perf_bp.route('/debug/test-preprocessing-on-sample', methods=['GET'])
def debug_test_preprocessing_on_sample():
    try:
        data_file = find_data_file()
        if not data_file:
            return jsonify({'success': False, 'error': 'Data file not found'}), 404

        df_sample = pd.read_csv(data_file, nrows=1000)

        raw_mrt_columns = ['TotalPassenger', 'StationEntry', 'StationExit', 'Date', 'Time']
        is_raw_mrt = all(col in df_sample.columns for col in raw_mrt_columns)

        if not is_raw_mrt:
            return jsonify({
                'success': False,
                'error': 'Sample is not in raw MRT format',
                'available_columns': df_sample.columns.tolist(),
                'expected_columns': raw_mrt_columns
            })

        df = df_sample.copy()
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        df['direction'] = df.apply(infer_direction_correct, axis=1)
        df = df[df['direction'] != 'Unknown']
        df['StationName'] = df.apply(get_station_name_fixed, axis=1)
        df = df.dropna(subset=['StationName'])
        df['Hour'] = df['datetime'].dt.floor('h')

        grouped = df.groupby(['StationName', 'direction', 'Hour']).agg({
            'TotalPassenger': 'sum',
            'datetime': 'first'
        }).reset_index()

        def calc_congestion_p90(row):
            """P90-based congestion (NOT capacity)"""
            return calculate_commuter_congestion(
                row['TotalPassenger'], row['StationName'], row['direction']
            )

        grouped['actual_congestion'] = grouped.apply(calc_congestion_p90, axis=1)

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
            'unique_directions': result['direction'].unique().tolist()
        })
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============================================================
# PREDICTION ENDPOINTS
# ============================================================
@model_perf_bp.route('/model/predict/single', methods=['POST'])
def predict_single():
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

        from routes.api_predict import get_directional_prediction as api_prediction
        pred_congestion = api_prediction(station, direction, target_time)

        if pred_congestion is None:
            return jsonify({
                "success": False,
                "error": f"Prediction failed. Model may not exist for {station} {direction}"
            }), 400

        # Try to fetch actual from historical data
        actual_congestion = None
        data_file = find_data_file()
        if data_file:
            try:
                df = pd.read_csv(data_file)
                df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
                df = df.dropna(subset=['datetime'])
                df['direction'] = df.apply(infer_direction_correct, axis=1)

                station_num = [k for k, v in STATION_NUMBER_MAP.items() if v == station][0]
                station_df = get_station_data_for_direction(df, station_num, direction)
                station_df = station_df[station_df['direction'] == direction]
                station_df['hour_timestamp'] = station_df['datetime'].dt.floor('h')

                matching = station_df[station_df['hour_timestamp'] == target_time.floor('h')]
                if not matching.empty:
                    total_pass = matching['TotalPassenger'].sum()
                    p90 = get_p90_for_station(station, direction)
                    actual_congestion = round(min(total_pass / p90 * 100, 100), 1)
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


# ============================================================
# AUTO TESTS
# ============================================================
@model_perf_bp.route('/model/run-auto-tests', methods=['POST'])
def run_auto_tests():
    """Run auto-tests with stratified sampling across all hours."""
    try:
        if not directional_models:
            return jsonify({"success": False, "error": "No models loaded"}), 500

        data_file = find_data_file()
        if not data_file:
            return jsonify({"success": False, "error": "Data file not found"}), 404

        # ✅ FIXED: read the FULL file, then random-sample if too big
        df = load_data_optimized(data_file, max_rows=None)
        if df is None or len(df) == 0:
            return jsonify({"success": False, "error": "Could not load data"}), 400

        # Cap memory by sampling (keeps hour distribution intact since we sample across rows)
        if len(df) > 200_000:
            print(f"   📊 Sampling 200k rows for auto-test (from {len(df):,})")
            df = df.sample(n=200_000, random_state=42)

        # Parse datetime
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        df = df.dropna(subset=['datetime'])

        if len(df) == 0:
            return jsonify({"success": False, "error": "Could not parse any dates"}), 400

        # ✅ Log hour distribution for visibility
        hour_dist = df['datetime'].dt.hour.value_counts().sort_index()
        print(f"   🕐 Hour distribution:\n{hour_dist}")

        df['hour'] = df['datetime'].dt.hour
        df['StationName'] = df.apply(get_station_name_fixed, axis=1)
        df['direction'] = df.apply(infer_direction_correct, axis=1)
        df = df[df['direction'] != 'Unknown']
        df = df.dropna(subset=['StationName'])

        # Station-direction pairs with models
        station_direction_pairs = []
        for station_name in STATION_NUMBER_MAP.values():
            for direction in ['Northbound', 'Southbound']:
                if f"{station_name}_{direction}" in directional_models:
                    station_direction_pairs.append((station_name, direction))

        # Stratified sampling: 1 per hour per station-direction
        sampled_dfs = []
        for station_name, direction in station_direction_pairs:
            subset = df[(df['StationName'] == station_name) & (df['direction'] == direction)]
            if len(subset) == 0:
                continue

            hourly_samples = []
            for hour in range(24):
                hour_subset = subset[subset['hour'] == hour]
                if len(hour_subset) > 0:
                    hourly_samples.append(hour_subset.sample(n=1, random_state=42))

            if hourly_samples:
                sampled_dfs.append(pd.concat(hourly_samples, ignore_index=True))

        if not sampled_dfs:
            return jsonify({"success": False, "error": "No data sampled"}), 400

        df = pd.concat(sampled_dfs, ignore_index=True)

        # Feature engineering
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

        from routes.api_predict import get_directional_prediction as api_prediction

        results = []

        for station_name, direction in station_direction_pairs:
            try:
                from services.feature_engineering import get_station_dataframe
                historical_hourly = get_station_dataframe(station_name, direction)
            except Exception:
                historical_hourly = None

            if historical_hourly is None or len(historical_hourly) == 0:
                continue

            station_num = [k for k, v in STATION_NUMBER_MAP.items() if v == station_name][0]
            station_df = get_station_data_for_direction(df, station_num, direction)

            if len(station_df) < 5:
                continue

            station_df['hour_timestamp'] = station_df['datetime'].dt.floor('h')

            existing_cols = [c for c in station_df.columns]
            agg_dict = {'TotalPassenger': 'sum'}
            for col in existing_cols:
                if col in ['hour', 'weekday', 'month', 'hour_sin', 'hour_cos',
                           'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
                           'time_decimal', 'is_operating_hour', 'minute_normalized',
                           'is_morning_rush', 'is_evening_rush',
                           'is_weekend', 'is_holiday', 'is_special_event',
                           'is_christmas_season', 'is_payday', 'is_friday',
                           'is_rush_hour', 'is_maintenance_record', 'is_extended_hours']:
                    agg_dict[col] = 'first'

            hourly = station_df.groupby('hour_timestamp').agg(agg_dict).reset_index()

            if len(hourly) < 3:
                continue

            hourly['actual_congestion'] = hourly.apply(
                lambda row: calculate_commuter_congestion(
                    row['TotalPassenger'], station_name, direction
                ), axis=1
            )

            hourly = hourly.sort_values('hour_timestamp')
            available_indices = list(range(24, len(hourly)))
            if not available_indices:
                continue

            num_tests = min(len(available_indices), 50)
            test_indices = sorted(random.sample(available_indices, num_tests))

            for idx in test_indices:
                target = hourly.iloc[idx]
                target_time = target['hour_timestamp']

                try:
                    pred_congestion_pct = api_prediction(station_name, direction, target_time)
                    if pred_congestion_pct is None or pred_congestion_pct < 0:
                        continue

                    actual_congestion = target['actual_congestion']
                    abs_error = abs(pred_congestion_pct - actual_congestion)

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
                        'predicted': round(pred_congestion_pct, 1),
                        'actual': round(actual_congestion, 1),
                        'total_passengers': int(target['TotalPassenger']),
                        'p90_percentile': round(get_p90_for_station(station_name, direction), 0),
                        'absolute_error': round(abs_error, 1),
                        'percentage_error': round((abs_error / max(actual_congestion, 0.1) * 100), 1),
                        'verdict': verdict,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'predicted_category': get_congestion_category(pred_congestion_pct),
                        'actual_category': get_congestion_category(actual_congestion)
                    })
                except Exception:
                    continue

        if not results:
            return jsonify({"success": False, "error": "No predictions could be generated"}), 400

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
            "excellent_count": int(len(df_results[df_results['verdict'] == 'EXCELLENT'])),
            "good_count": int(len(df_results[df_results['verdict'] == 'GOOD'])),
            "okay_count": int(len(df_results[df_results['verdict'] == 'OKAY'])),
            "needs_improvement": int(len(df_results[df_results['verdict'] == 'NEEDS_IMPROVEMENT'])),
            "date_range": {
                "min": df_results['target_time'].min(),
                "max": df_results['target_time'].max()
            },
            "stations_tested_list": df_results['station'].unique().tolist()
        }

        return jsonify({
            "success": True,
            "message": f"Generated {len(results)} predictions",
            "summary": summary
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ============================================================
# CSV UPLOAD / BATCH TEST
# ============================================================
# ============================================================
# CSV UPLOAD / BATCH TEST (BATCHED PREDICTION VERSION)
# ============================================================
@model_perf_bp.route('/model/upload/batch', methods=['POST'])
def upload_batch_test():
    """
    Upload CSV and run batch predictions with proper hourly aggregation.
    Uses REAL data to build features — matches manual test R² ≈ 0.98.

    BATCHED: runs model.predict() once per station-direction (not per sample).
    Expected runtime: ~1-2 minutes for 46,800 samples.
    """
    try:
        import glob, shutil
        backup_dir = 'test_results_backup_auto'
        os.makedirs(backup_dir, exist_ok=True)

        for f in glob.glob('test_results/full_2025_test_*.csv'):
            shutil.move(f, os.path.join(backup_dir, os.path.basename(f)))

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Only CSV files are supported'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # ------------------------------------------------------------
        # STEP 1: Load models once
        # ------------------------------------------------------------
        from services.model_loader import load_single_model

        loaded_models = {}
        loaded_scalers = {}

        print("📦 Loading models for evaluation...")
        for station in STATIONS:
            for direction in ["Northbound", "Southbound"]:
                key = f"{station}_{direction}"
                model, result = load_single_model(station, direction, 'models_2022-2024_v11')
                if model is not None:
                    loaded_models[key] = model
                    loaded_scalers[key] = (result.get('feature'), result.get('target'))
        print(f"✅ Loaded {len(loaded_models)} models for evaluation")

        # ------------------------------------------------------------
        # STEP 2: Read uploaded CSV
        # ------------------------------------------------------------
        sample_df = pd.read_csv(filepath, nrows=5)
        raw_mrt_columns = ['TotalPassenger', 'StationEntry', 'StationExit', 'Date', 'Time']
        is_raw_mrt = all(col in sample_df.columns for col in raw_mrt_columns)

        if not is_raw_mrt:
            return jsonify({
                'success': False,
                'error': 'Expected RAW MRT format (TotalPassenger, StationEntry, StationExit, Date, Time)'
            }), 400

        print("📊 Detected RAW MRT format - preprocessing...")
        df_raw = pd.read_csv(filepath)
        print(f"   📊 Loaded {len(df_raw):,} rows")
        df_raw['datetime'] = pd.to_datetime(df_raw['Date'] + ' ' + df_raw['Time'], errors='coerce')
        df_raw = df_raw.dropna(subset=['datetime'])

        # ------------------------------------------------------------
        # STEP 3: Build hourly series WITH REINDEX (fill gaps)
        # ------------------------------------------------------------
        station_hourly = {}

        for station_num, station_name in STATION_NUMBER_MAP.items():
            for direction in ['Northbound', 'Southbound']:
                sub = get_station_data_for_direction(df_raw, station_num, direction)
                if len(sub) == 0:
                    continue

                sub = sub.copy()
                sub['hour_timestamp'] = sub['datetime'].dt.floor('h')
                hourly = sub.groupby('hour_timestamp').agg({'TotalPassenger': 'sum'}).reset_index()
                hourly = hourly.rename(columns={'hour_timestamp': 'datetime'})

                # REINDEX to fill all hours
                if len(hourly) > 0:
                    full_range = pd.date_range(
                        start=hourly['datetime'].min(),
                        end=hourly['datetime'].max(),
                        freq='h'
                    )
                    hourly = hourly.set_index('datetime').reindex(full_range).reset_index()
                    hourly = hourly.rename(columns={'index': 'datetime'})
                    hourly['TotalPassenger'] = hourly['TotalPassenger'].fillna(0).clip(lower=0)

                hourly = hourly.sort_values('datetime').reset_index(drop=True)
                station_hourly[f"{station_name}_{direction}"] = hourly

        print(f"✅ Built hourly series for {len(station_hourly)} station-directions")

        # ------------------------------------------------------------
        # STEP 4: Enrich each series with calendar features
        # ------------------------------------------------------------
        hols = [
            '2025-01-01','2025-04-09','2025-04-17','2025-04-18','2025-05-01',
            '2025-06-12','2025-08-21','2025-08-25','2025-11-30','2025-12-08',
            '2025-12-25','2025-12-30','2025-12-31'
        ]

        def enrich(hourly):
            hourly = hourly.copy()
            hourly['hour'] = hourly['datetime'].dt.hour
            hourly['weekday'] = hourly['datetime'].dt.weekday
            hourly['month'] = hourly['datetime'].dt.month
            hourly['hour_sin'] = np.sin(2 * np.pi * hourly['hour'] / 24)
            hourly['hour_cos'] = np.cos(2 * np.pi * hourly['hour'] / 24)
            hourly['dow_sin'] = np.sin(2 * np.pi * hourly['weekday'] / 7)
            hourly['dow_cos'] = np.cos(2 * np.pi * hourly['weekday'] / 7)
            hourly['month_sin'] = np.sin(2 * np.pi * (hourly['month'] - 1) / 12)
            hourly['month_cos'] = np.cos(2 * np.pi * (hourly['month'] - 1) / 12)
            td = hourly['hour'] + hourly['datetime'].dt.minute / 60
            hourly['is_operating_hour'] = ((td >= 4.5) & (td < 23.0)).astype(np.int8)
            hourly['is_morning_rush'] = ((td >= 7.0) & (td <= 9.0)).astype(np.int8)
            hourly['is_evening_rush'] = ((td >= 17.0) & (td <= 19.0)).astype(np.int8)
            hourly['is_holiday'] = hourly['datetime'].dt.date.astype(str).isin(hols).astype(np.int8)
            hourly['is_christmas_season'] = hourly['datetime'].dt.strftime('%m-%d').map(
                lambda x: 1 if (x >= '12-15' or x <= '01-05') else 0
            ).astype(np.int8)
            hourly['is_payday'] = hourly['datetime'].dt.day.isin([15, 30, 31]).astype(np.int8)
            return hourly

        for key in station_hourly:
            station_hourly[key] = enrich(station_hourly[key])

        # ------------------------------------------------------------
        # STEP 5: Feature column list
        # ------------------------------------------------------------
        with open('models_2022-2024_v11/feature_cols.pkl', 'rb') as f:
            feats = pickle.load(f)
        print(f"   📋 Features: {len(feats)}")

        from routes.api_predict import get_p90_percentile as _get_p90

        # ------------------------------------------------------------
        # STEP 6: BATCH predict per station-direction
        # ------------------------------------------------------------
        all_hours = list(range(5, 23))
        SAMPLES_PER_HOUR = 100

        results = []
        errors = 0
        total_batches = 0

        for key, hourly in station_hourly.items():
            station_name, direction = key.rsplit('_', 1)

            if key not in loaded_models:
                print(f"   ⚠️ No model for {key}")
                continue

            model = loaded_models[key]
            sx, sy = loaded_scalers[key]
            if sx is None or sy is None:
                print(f"   ⚠️ Missing scaler for {key}")
                continue

            # Skip first 24 hours (need full window)
            if len(hourly) <= 24:
                print(f"   ⚠️ Too short: {key} ({len(hourly)} rows)")
                continue

            # ── Build ALL 24-hour windows at once ──
            vals = hourly[feats].values.astype(np.float32)     # (N, F)
            targets_passengers = hourly['TotalPassenger'].values.astype(np.float32)
            timestamps = hourly['datetime'].values             # numpy datetime64
            hours_arr = hourly['hour'].values
            N = len(hourly)

            # Rolling windows: for i in [24, N-1], window = vals[i-24:i]
            # Vectorize with stride tricks
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(vals, window_shape=24, axis=0)   # (N-24+1, F, 24)
            # Transpose to (N-24+1, 24, F)
            windows = windows.transpose(0, 2, 1)

            # Target for each window: the row after the window
            window_targets = targets_passengers[24:]           # (N-24,)
            window_times = timestamps[24:]
            window_hours = hours_arr[24:]
            n_windows = len(window_targets)

            # ── Sample target indices ──
            # For each operating hour, sample up to SAMPLES_PER_HOUR windows
            selected_indices = []
            for hour in all_hours:
                hour_mask = (window_hours == hour)
                hour_indices = np.where(hour_mask)[0]
                if len(hour_indices) == 0:
                    continue
                n = min(SAMPLES_PER_HOUR, len(hour_indices))
                # Deterministic sample
                rng = np.random.RandomState(42)
                chosen = rng.choice(hour_indices, size=n, replace=False)
                selected_indices.extend(chosen.tolist())

            if not selected_indices:
                continue

            selected_indices = np.array(selected_indices, dtype=np.int64)

            X_batch = windows[selected_indices]                # (M, 24, F)
            y_batch = window_targets[selected_indices]         # (M,)
            t_batch = window_times[selected_indices]           # (M,)
            h_batch = window_hours[selected_indices]           # (M,)

            # ── Scale the entire batch in ONE call ──
            M = X_batch.shape[0]
            X_flat = X_batch.reshape(-1, X_batch.shape[-1])    # (M*24, F)
            X_scaled_flat = sx.transform(X_flat)
            X_scaled = X_scaled_flat.reshape(M, 24, -1).astype(np.float32)

            # ── ONE model.predict() call per station ──
            preds_scaled = model.predict(X_scaled, verbose=0).flatten()
            preds_passengers = sy.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

            # ── Get P90 for this station ──
            p90 = _get_p90(station_name, direction)
            if p90 <= 0:
                p90 = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)

            # ── Build results ──
            for i in range(M):
                actual_pass = float(y_batch[i])
                pred_pass = float(preds_passengers[i])
                pred_cong = min((pred_pass / p90) * 100, 100)
                actual_cong = min((actual_pass / p90) * 100, 100)
                abs_err = abs(pred_cong - actual_cong)
                pct_err = (abs_err / max(actual_cong, 0.1)) * 100

                if abs_err <= 5:
                    verdict = 'EXCELLENT'
                elif abs_err <= 10:
                    verdict = 'GOOD'
                elif abs_err <= 15:
                    verdict = 'OKAY'
                else:
                    verdict = 'NEEDS_IMPROVEMENT'

                ts = pd.Timestamp(t_batch[i])

                results.append({
                    'station': station_name,
                    'direction': direction,
                    'target_time': ts.strftime('%Y-%m-%d %H:%M:%S'),
                    'hour': int(h_batch[i]),
                    'predicted': round(pred_cong, 1),
                    'actual': round(actual_cong, 1),
                    'absolute_error': round(abs_err, 1),
                    'percentage_error': round(pct_err, 1),
                    'verdict': verdict,
                    'predicted_category': get_congestion_category(pred_cong),
                    'actual_category': get_congestion_category(actual_cong)
                })

            total_batches += 1
            print(f"   ✅ {key}: {M} predictions")

        print(f"✅ Completed {total_batches} batches, {len(results)} total predictions, {errors} errors")

        if not results:
            return jsonify({'success': False, 'error': f'No valid predictions. {errors} errors.'}), 400

        # ------------------------------------------------------------
        # STEP 7: Save results
        # ------------------------------------------------------------
        df_results = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('test_results', exist_ok=True)

        df_results.to_csv(f'test_results/upload_batch_{timestamp}.csv', index=False)
        for (station, direction), group_df in df_results.groupby(['station', 'direction']):
            group_df.to_csv(f"test_results/{station}_{direction}_results.csv", index=False)
        df_results.to_csv(f'test_results/full_upload_batch_{timestamp}.csv', index=False)

        # Per-hour breakdown
        per_hour = {}
        for hour in sorted(df_results['hour'].unique()):
            hd = df_results[df_results['hour'] == hour]
            per_hour[int(hour)] = {
                'samples': int(len(hd)),
                'avg_predicted': float(round(hd['predicted'].mean(), 1)),
                'avg_actual': float(round(hd['actual'].mean(), 1)),
                'avg_error': float(round(hd['absolute_error'].mean(), 1)),
                'min_actual': float(round(hd['actual'].min(), 1)),
                'max_actual': float(round(hd['actual'].max(), 1)),
            }

        summary = {
            'total_processed': int(len(results)),
            'errors': int(errors),
            'avg_absolute_error': float(round(df_results['absolute_error'].mean(), 2)),
            'avg_percentage_error': float(round(df_results['percentage_error'].mean(), 2)),
            'excellent_count': int(len(df_results[df_results['verdict'] == 'EXCELLENT'])),
            'good_count': int(len(df_results[df_results['verdict'] == 'GOOD'])),
            'okay_count': int(len(df_results[df_results['verdict'] == 'OKAY'])),
            'needs_improvement': int(len(df_results[df_results['verdict'] == 'NEEDS_IMPROVEMENT'])),
            'per_hour': per_hour,
            'stations_tested': [str(s) for s in df_results['station'].unique().tolist()],
            'hours_covered': [int(h) for h in sorted(df_results['hour'].unique())],
        }

        response = {
            'success': True,
            'message': f'Processed {len(results)} predictions (batched)',
            'summary': summary,
            'results_preview': [
                {
                    'station': str(r['station']),
                    'direction': str(r['direction']),
                    'target_time': r['target_time'],
                    'hour': int(r['hour']),
                    'predicted': float(r['predicted']),
                    'actual': float(r['actual']),
                    'absolute_error': float(r['absolute_error']),
                    'percentage_error': float(r['percentage_error']),
                    'verdict': str(r['verdict']),
                    'predicted_category': str(r['predicted_category']),
                    'actual_category': str(r['actual_category'])
                }
                for r in results[:30]
            ]
        }

        return jsonify(response)
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
# ============================================================
# BOOTSTRAP (module-level side effects)
# ============================================================
try:
    HISTORICAL_PEAKS = load_historical_peaks()
except Exception as e:
    print(f"⚠️ Skipped historical peaks load: {e}")

try:
    load_correction_factors()
except Exception as e:
    print(f"⚠️ Skipped correction factors load: {e}")