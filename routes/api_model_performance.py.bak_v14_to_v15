"""
Model Performance Routes - LSTM Testing & Visualization
Handles: Manual predictions, batch uploads, metrics, chart data, evaluation metrics

UPDATED FOR 2026:
- v13 models (2023-2025 trained) as default
- Holiday list extended to 2026
- find_data_file() now searches for both 2025.csv and 2026.csv
- run_auto_tests() and upload_batch_test() use 2026 holidays
- All hardcoded paths updated
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
MAX_ROWS_TO_READ = 20000

# ✅ Current model folder — change this if you swap model versions
CURRENT_MODEL_FOLDER = 'models_2023-2025_v14'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('test_results', exist_ok=True)
os.makedirs('evaluation_results', exist_ok=True)

# ============================================================
# STATIONS LIST
# ============================================================
STATIONS = [
    "North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan",
    "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia",
    "Ayala Ave", "Magallanes", "Taft"
]

# ============================================================
# STATION MAPPING
# ============================================================
STATION_NUMBER_MAP = {
    1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
    5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
    9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
}


def get_station_name_fixed(row, direction=None):
    """Assign station name based on direction-aware convention."""
    entry = row['StationEntry']
    exit_station = row['StationExit']

    if direction == 'Northbound':
        if exit_station == 1:
            return "North Ave"
        return STATION_NUMBER_MAP.get(entry)
    elif direction == 'Southbound':
        if entry == 13:
            return "Taft"
        return STATION_NUMBER_MAP.get(exit_station)
    else:
        if entry == 1 or exit_station == 1:
            return "North Ave"
        if entry == 13 or exit_station == 13:
            return "Taft"
        return STATION_NUMBER_MAP.get(exit_station)


# ============================================================
# HISTORICAL PEAKS
# ============================================================
HISTORICAL_PEAKS = {}


def get_p90_for_station(station_name, direction):
    try:
        from routes.api_predict import get_p90_percentile
        return get_p90_percentile(station_name, direction)
    except Exception as e:
        print(f"⚠️ Could not get P90 for {station_name} {direction}: {e}")
        return MRT3_PLATFORM_CAPACITY.get(station_name, 1000)


# ============================================================
# 2025/2026 HOLIDAYS (SINGLE SOURCE OF TRUTH)
# ============================================================
HOLIDAYS_2025 = [
    '2025-01-01', '2025-04-09', '2025-04-17', '2025-04-18',
    '2025-05-01', '2025-06-12', '2025-08-21', '2025-08-25',
    '2025-11-30', '2025-12-08', '2025-12-25', '2025-12-30', '2025-12-31'
]

HOLIDAYS_2026 = [
    '2026-01-01', '2026-04-02', '2026-04-03', '2026-04-09',
    '2026-05-01', '2026-06-12', '2026-08-21', '2026-08-31',
    '2026-11-30', '2026-12-08', '2026-12-25', '2026-12-30', '2026-12-31'
]

# Combined list for backward compatibility
HOLIDAYS = HOLIDAYS_2025 + HOLIDAYS_2026


# ============================================================
# DATA FILE FINDER — supports 2025 AND 2026
# ============================================================
def find_data_file():
    """
    Find the most recent data file (2026 > 2025 > 2024 > etc.)
    Searches multiple locations.
    """
    possible_paths = [
        # 2026 first (newest)
        'data (2022-2024)/2026.csv',
        '../data (2022-2024)/2026.csv',
        '2026.csv',
        'data/2026.csv',
        # Fallback to 2025
        'data (2022-2024)/2025.csv',
        '../data (2022-2024)/2025.csv',
        '2025.csv',
        'data/2025.csv',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def find_data_file_for_year(year):
    """Find a specific year's data file."""
    possible_paths = [
        f'data (2022-2024)/{year}.csv',
        f'../data (2022-2024)/{year}.csv',
        f'{year}.csv',
        f'data/{year}.csv',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


# ============================================================
# DRIFT DETECTION
# ============================================================
@model_perf_bp.route('/model/drift-detection', methods=['POST'])
def detect_drift():
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

        # Load baseline errors from most recent test results
        baseline_errors = []
        if os.path.exists('test_results'):
            baseline_files = [f for f in os.listdir('test_results')
                              if f.startswith('full_')]
            if baseline_files:
                baseline_files.sort()
                baseline_df = pd.read_csv(os.path.join('test_results', baseline_files[-1]))
                if 'absolute_error' in baseline_df.columns:
                    baseline_errors = baseline_df['absolute_error'].tolist()

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
# DATA AGGREGATION FOR EVALUATION
# ============================================================
def load_and_aggregate_data(year=None):
    """
    Load CSV (2025 or 2026, or auto-detect) and aggregate to hourly totals.
    """
    if year:
        data_file = find_data_file_for_year(year)
    else:
        data_file = find_data_file()

    if not data_file:
        print(f"❌ Data file not found")
        return None

    print(f"📁 Loading data from: {data_file}")
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


# Keep backward-compatible name
def load_and_aggregate_2025_data():
    return load_and_aggregate_data(year=2025)


# ============================================================
# HISTORICAL PEAKS LOADER
# ============================================================
def load_historical_peaks():
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
    return MRT3_PLATFORM_CAPACITY.get(station_name, 1000)


def calculate_congestion(passenger_count, station_name, direction='Northbound'):
    p90 = get_p90_for_station(station_name, direction)
    if p90 <= 0:
        p90 = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)
    congestion = (passenger_count / p90) * 100
    return min(congestion, 100)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# FEATURE ENGINEERING HELPERS
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
    """Direction-aware filtering (handles terminals)."""
    station_name = STATION_NUMBER_MAP.get(station_num, "Unknown")

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
            return df[df['StationEntry'] == station_num].copy()
        else:
            return df[df['StationExit'] == station_num].copy()


def get_historical_peak(station_name, direction):
    key = f"{station_name}_{direction}"
    peak_data = HISTORICAL_PEAKS.get(key)
    if peak_data:
        if isinstance(peak_data, dict):
            return peak_data.get("peak", peak_data.get("absolute_max", 5000))
        return peak_data
    return 5000


def calculate_commuter_congestion(passenger_count, station, direction):
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
# HELPER: Build holiday mask from a date series
# ============================================================
def get_holiday_mask(dates_series):
    """Returns a boolean mask for whether each date is a holiday."""
    return dates_series.dt.date.astype(str).isin(HOLIDAYS).astype(np.int8)


# ============================================================
# EVALUATION METRICS (unchanged from your file, just uses new helpers)
# ============================================================
@model_perf_bp.route('/model/evaluate', methods=['POST'])
def evaluate_model():
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
# PER-CLASS METRICS (unchanged)
# ============================================================
@model_perf_bp.route('/model/evaluate/per-class', methods=['GET', 'POST'])
def evaluate_per_class():
    try:
        data = request.json if request.is_json else {}
        data.update(request.args)
        station = data.get('station', 'all')
        direction = data.get('direction', 'both')

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

        predictions = combined['predicted'].values
        actuals = combined['actual'].values

        pred_cats = [get_congestion_category(p) for p in predictions]
        actual_cats = [get_congestion_category(a) for a in actuals]

        labels = CATEGORY_ORDER
        cm = confusion_matrix(actual_cats, pred_cats, labels=labels)

        n = cm.sum()
        per_class = {}

        for i, label in enumerate(labels):
            tp = int(cm[i, i])
            fn = int(cm[i, :].sum() - tp)
            fp = int(cm[:, i].sum() - tp)
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

        overall_accuracy = round(100 * sum(per_class[l]['tp'] for l in labels) / n, 2)

        column_labels = ['Light', 'Moderate', 'Congested', 'Severely Congested',
                         'Macro Average', 'Weighted Average']

        def safe(label):
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
                'per_class_raw': per_class,
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
# CONFUSION MATRIX — TIER-ROW TABLE
# ============================================================
@model_perf_bp.route('/model/evaluation/confusion-table', methods=['GET', 'POST'])
def get_confusion_table():
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
        col_labels = raw['column_labels']
        rows_by_metric = raw['rows']

        lookup = {r['metric']: r['values'] for r in rows_by_metric}

        def get_metric(key, idx):
            return lookup.get(key, [0] * 6)[idx]

        tier_rows = []
        for idx, tier_label in enumerate(col_labels[:4]):
            tier_rows.append({
                'tier': tier_label,
                'TP': get_metric('TP', idx),
                'FP': get_metric('FP', idx),
                'FN': get_metric('FN', idx),
                'TN': get_metric('TN', idx),
                'Precision': get_metric('Precision (%)', idx),
                'Recall':    get_metric('Recall (%)', idx),
                'F1':        get_metric('F1-Score (%)', idx),
                'Support':   get_metric('Support', idx),
            })

        tier_rows.append({
            'tier': 'Macro Average',
            'TP': None, 'FP': None, 'FN': None, 'TN': None,
            'Precision': round(get_metric('Precision (%)', 4) / 100.0, 4),
            'Recall':    round(get_metric('Recall (%)',    4) / 100.0, 4),
            'F1':        round(get_metric('F1-Score (%)',  4) / 100.0, 4),
            'Support':   get_metric('Support', 4),
        })

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


# ============================================================
# REMAINING ENDPOINTS — Updated key parts
# ============================================================

# ============================================================
# AUTO TESTS — UPDATED for 2026
# ============================================================
@model_perf_bp.route('/model/run-auto-tests', methods=['POST'])
def run_auto_tests():
    """Run auto-tests with stratified sampling across all hours."""
    try:
        if not directional_models:
            return jsonify({"success": False, "error": "No models loaded"}), 500

        # ✅ UPDATED: uses find_data_file() which finds 2026 or 2025
        data_file = find_data_file()
        if not data_file:
            return jsonify({"success": False, "error": "Data file not found"}), 404

        df = load_data_optimized(data_file, max_rows=None)
        if df is None or len(df) == 0:
            return jsonify({"success": False, "error": "Could not load data"}), 400

        if len(df) > 200_000:
            print(f"   📊 Sampling 200k rows for auto-test (from {len(df):,})")
            df = df.sample(n=200_000, random_state=42)

        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        df = df.dropna(subset=['datetime'])

        if len(df) == 0:
            return jsonify({"success": False, "error": "Could not parse any dates"}), 400

        hour_dist = df['datetime'].dt.hour.value_counts().sort_index()
        print(f"   🕐 Hour distribution:\n{hour_dist}")

        df['hour'] = df['datetime'].dt.hour
        df['StationName'] = df.apply(get_station_name_fixed, axis=1)
        df['direction'] = df.apply(infer_direction_correct, axis=1)
        df = df[df['direction'] != 'Unknown']
        df = df.dropna(subset=['StationName'])

        station_direction_pairs = []
        for station_name in STATION_NUMBER_MAP.values():
            for direction in ['Northbound', 'Southbound']:
                if f"{station_name}_{direction}" in directional_models:
                    station_direction_pairs.append((station_name, direction))

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

        # ✅ UPDATED: uses combined HOLIDAYS (2025 + 2026)
        df['is_holiday'] = df['datetime'].dt.date.astype(str).isin(HOLIDAYS).astype(np.int8)
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

        df_results.to_csv(f'test_results/full_auto_test_{timestamp}.csv', index=False)

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
# CSV UPLOAD / BATCH TEST — UPDATED for v13 and 2026
# ============================================================
@model_perf_bp.route('/model/upload/batch', methods=['POST'])
def upload_batch_test():
    """
    Upload CSV and run batch predictions with proper hourly aggregation.

    ✅ Uses CURRENT_MODEL_FOLDER (models_2023-2025_v14)
    ✅ Holiday list includes both 2025 AND 2026
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
        # STEP 1: Load models (v13)
        # ------------------------------------------------------------
        from services.model_loader import load_single_model

        loaded_models = {}
        loaded_scalers = {}

        print(f"📦 Loading models from {CURRENT_MODEL_FOLDER}...")
        for station in STATIONS:
            for direction in ["Northbound", "Southbound"]:
                key = f"{station}_{direction}"
                model, result = load_single_model(station, direction, CURRENT_MODEL_FOLDER)
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
        # STEP 3: Build hourly series
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
        # STEP 4: Enrich — uses combined HOLIDAYS (2025 + 2026)
        # ------------------------------------------------------------
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
            # ✅ Uses combined HOLIDAYS (2025 + 2026)
            hourly['is_holiday'] = hourly['datetime'].dt.date.astype(str).isin(HOLIDAYS).astype(np.int8)
            hourly['is_christmas_season'] = hourly['datetime'].dt.strftime('%m-%d').map(
                lambda x: 1 if (x >= '12-15' or x <= '01-05') else 0
            ).astype(np.int8)
            hourly['is_payday'] = hourly['datetime'].dt.day.isin([15, 30, 31]).astype(np.int8)
            return hourly

        for key in station_hourly:
            station_hourly[key] = enrich(station_hourly[key])

        # ------------------------------------------------------------
        # STEP 5: Feature columns from v13
        # ------------------------------------------------------------
        feature_cols_path = os.path.join(CURRENT_MODEL_FOLDER, 'feature_cols.pkl')
        with open(feature_cols_path, 'rb') as f:
            feats = pickle.load(f)
        print(f"   📋 Features: {len(feats)}")

        from routes.api_predict import get_p90_percentile as _get_p90

        # ------------------------------------------------------------
        # STEP 6: Batch predict
        # ------------------------------------------------------------
        all_hours = list(range(5, 23))
        SAMPLES_PER_HOUR = 500

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

            if len(hourly) <= 24:
                print(f"   ⚠️ Too short: {key} ({len(hourly)} rows)")
                continue

            vals = hourly[feats].values.astype(np.float32)
            targets_passengers = hourly['TotalPassenger'].values.astype(np.float32)
            timestamps = hourly['datetime'].values
            hours_arr = hourly['hour'].values
            N = len(hourly)

            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(vals, window_shape=24, axis=0)
            windows = windows.transpose(0, 2, 1)

            window_targets = targets_passengers[24:]
            window_times = timestamps[24:]
            window_hours = hours_arr[24:]

            selected_indices = []
            for hour in all_hours:
                hour_mask = (window_hours == hour)
                hour_indices = np.where(hour_mask)[0]
                if len(hour_indices) == 0:
                    continue
                n = min(SAMPLES_PER_HOUR, len(hour_indices))
                rng = np.random.RandomState(42)
                chosen = rng.choice(hour_indices, size=n, replace=False)
                selected_indices.extend(chosen.tolist())

            if not selected_indices:
                continue

            selected_indices = np.array(selected_indices, dtype=np.int64)

            X_batch = windows[selected_indices]
            y_batch = window_targets[selected_indices]
            t_batch = window_times[selected_indices]
            h_batch = window_hours[selected_indices]

            M = X_batch.shape[0]
            X_flat = X_batch.reshape(-1, X_batch.shape[-1])
            X_scaled_flat = sx.transform(X_flat)
            X_scaled = X_scaled_flat.reshape(M, 24, -1).astype(np.float32)

            preds_scaled = model.predict(X_scaled, verbose=0).flatten()
            preds_passengers = sy.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

            p90 = _get_p90(station_name, direction)
            if p90 <= 0:
                p90 = MRT3_PLATFORM_CAPACITY.get(station_name, 1000)

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
# REMAINING ENDPOINTS (unchanged, just kept for compatibility)
# ============================================================

# [Rest of your file stays identical — get_evaluation_history, health_check,
#  debug endpoints, get_performance_metrics, get_chart_data, get_station_details,
#  debug_csv, debug_check_2025_csv, debug_check_raw_mrt_data,
#  debug_test_preprocessing_on_sample, predict_single, get_evaluation_summary]

# ⬇⬇⬇ PASTE YOUR EXISTING VERSIONS OF THESE HERE ⬇⬇⬇
# (They don't need changes for 2026 — they work off whatever's in test_results)


# ============================================================
# BOOTSTRAP
# ============================================================
# ============================================================
# PERFORMANCE METRICS (required by dashboard)
# ============================================================
@model_perf_bp.route('/model/performance/metrics', methods=['GET'])
def get_performance_metrics():
    try:
        if not os.path.exists('test_results'):
            return jsonify({
                "success": True,
                "data": {
                    "message": "No test results yet",
                    "total_predictions": 0,
                    "overall_r2": 0.0,
                    "overall_mae": None,
                    "overall_mape": None,
                    "last_evaluated": None,
                    "per_station": {}
                }
            })

        all_results = []
        for filename in os.listdir('test_results'):
            if filename.endswith('_results.csv') and not filename.startswith('full_'):
                try:
                    df = pd.read_csv(os.path.join('test_results', filename))
                    for col in ['predicted', 'actual', 'absolute_error', 'percentage_error']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    if 'absolute_error' in df.columns:
                        df = df.dropna(subset=['absolute_error'])
                    if len(df) > 0:
                        all_results.append(df)
                except Exception:
                    continue

        if not all_results:
            return jsonify({
                "success": True,
                "data": {
                    "total_predictions": 0,
                    "overall_r2": 0.0,
                    "overall_mae": None,
                    "overall_mape": None,
                    "last_evaluated": None,
                    "per_station": {},
                    "message": "No valid test results"
                }
            })

        combined = pd.concat(all_results, ignore_index=True)

        # ------------------------------------------------------------
        # Overall metrics
        # ------------------------------------------------------------
        try:
            _valid = combined.dropna(subset=['predicted', 'actual'])
            if len(_valid) >= 2:
                overall_r2 = round(float(r2_score(_valid['actual'].values, _valid['predicted'].values)), 3)
            else:
                overall_r2 = 0.0
        except Exception:
            overall_r2 = 0.0

        metrics = {
            "overall_mae": round(float(combined['absolute_error'].mean()), 2) if 'absolute_error' in combined.columns else None,
            "overall_mape": round(float(combined['percentage_error'].mean()), 2) if 'percentage_error' in combined.columns else None,
            "overall_r2": overall_r2,
            "total_predictions": int(len(combined)),
            "total_tests": int(len(combined)),
            "avg_absolute_error": round(float(combined['absolute_error'].mean()), 2) if 'absolute_error' in combined.columns else None,
            "avg_percentage_error": round(float(combined['percentage_error'].mean()), 2) if 'percentage_error' in combined.columns else None,
        }

        # ------------------------------------------------------------
        # Per-station metrics (northbound + southbound)
        # Needed by the "All Stations" table in Model Performance page.
        # ------------------------------------------------------------
        per_station = {}
        if 'station' in combined.columns and 'direction' in combined.columns:
            for (st, dr), grp in combined.groupby(['station', 'direction']):
                try:
                    _v = grp.dropna(subset=['predicted', 'actual'])
                    if len(_v) == 0:
                        continue

                    _err = (_v['predicted'] - _v['actual']).abs()
                    _mae = float(_err.mean())
                    _rmse = float(((_v['predicted'] - _v['actual']) ** 2).mean() ** 0.5)
                    _mape = float(((_err / _v['actual'].clip(lower=0.1)) * 100).mean())

                    # Per-station classification accuracy (4-tier)
                    try:
                        _pred_cats = [_v['predicted'].iloc[i] for i in range(len(_v))]
                        # Use the same category thresholds as api_predict
                        def _cat(v):
                            if v > 80: return 'Severe'
                            if v > 45: return 'Congested'
                            if v > 25: return 'Moderate'
                            return 'Light'
                        _pred_cat = [_cat(p) for p in _v['predicted'].values]
                        _actual_cat = [_cat(a) for a in _v['actual'].values]
                        _acc = sum(1 for p, a in zip(_pred_cat, _actual_cat) if p == a) / len(_actual_cat) * 100
                    except Exception:
                        _acc = None

                    if st not in per_station:
                        per_station[st] = {
                            'northbound': {'mae': None, 'rmse': None, 'mape': None, 'samples': 0, 'accuracy': None},
                            'southbound': {'mae': None, 'rmse': None, 'mape': None, 'samples': 0, 'accuracy': None},
                        }

                    key = 'northbound' if str(dr).lower() == 'northbound' else 'southbound'
                    per_station[st][key] = {
                        'mae': round(_mae, 2),
                        'rmse': round(_rmse, 2),
                        'mape': round(_mape, 2),
                        'samples': int(len(_v)),
                        'accuracy': round(_acc, 2) if _acc is not None else None,
                    }
                except Exception as e:
                    print(f"⚠️ per-station computation failed for {st}/{dr}: {e}")
                    continue

        metrics['per_station'] = per_station

        # ------------------------------------------------------------
        # Last evaluated timestamp
        # ------------------------------------------------------------
        try:
            if 'target_time' in combined.columns:
                metrics['last_evaluated'] = pd.to_datetime(combined['target_time']).max().isoformat()
            else:
                metrics['last_evaluated'] = None
        except Exception:
            metrics['last_evaluated'] = None

        return jsonify({"success": True, "data": metrics})

    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

# ============================================================
# CHART DATA (required by dashboard)
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
                except Exception:
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

        return jsonify({"success": True, "data": {"labels": labels, "predicted": predicted, "actual": actual, "errors": errors}})
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ============================================================
# STATION DETAILS (required by dashboard)
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

        # ------------------------------------------------------------
        # CASE 1: All stations -> return per-station MAE summary
        # ------------------------------------------------------------
        if station.lower() == 'all':
            files = glob.glob(f"test_results/*_{direction}_results.csv")
            if not files:
                return jsonify({"success": True, "data": []}), 200

            summary_rows = []
            for f in files:
                try:
                    part = pd.read_csv(f)
                    base = os.path.basename(f)
                    station_name = base.rsplit(f"_{direction}_results.csv", 1)[0]

                    for col in ['predicted', 'actual', 'absolute_error', 'percentage_error']:
                        if col in part.columns:
                            part[col] = pd.to_numeric(part[col], errors='coerce')

                    valid = part.dropna(subset=['predicted', 'actual'])
                    if len(valid) == 0:
                        continue

                    diff = valid['predicted'] - valid['actual']
                    mae = float(diff.abs().mean())
                    rmse = float(np.sqrt((diff ** 2).mean()))
                    mape_vals = (diff.abs() / valid['actual'].clip(lower=0.1)) * 100
                    mape = float(mape_vals.mean())

                    summary_rows.append({
                        'station': station_name,
                        'direction': direction,
                        'samples': int(len(valid)),
                        'mae': round(mae, 2),
                        'rmse': round(rmse, 2),
                        'mape': round(mape, 2),
                        'avg_predicted': round(float(valid['predicted'].mean()), 1),
                        'avg_actual': round(float(valid['actual'].mean()), 1),
                    })
                except Exception:
                    continue

            summary_rows.sort(key=lambda r: r['mae'])
            return jsonify({"success": True, "data": summary_rows})

        # ------------------------------------------------------------
        # CASE 2: Single station -> return raw predictions
        # ------------------------------------------------------------
        filename = f"test_results/{station}_{direction}_results.csv"
        if not os.path.exists(filename):
            return jsonify({"success": True, "data": []}), 200
        df = pd.read_csv(filename)

        if df.empty:
            return jsonify({"success": True, "data": []}), 200

        for col in ['predicted', 'actual', 'absolute_error', 'percentage_error']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'target_time' in df.columns:
            try:
                df['target_time_dt'] = pd.to_datetime(df['target_time'], errors='coerce')
                df = df.dropna(subset=['target_time_dt']).sort_values('target_time_dt', ascending=False)
            except Exception:
                pass

        # Ensure station/direction are present
        if 'station' not in df.columns:
            df['station'] = station
        if 'direction' not in df.columns:
            df['direction'] = direction

        return_cols = [c for c in [
            'station', 'direction',
            'target_time', 'predicted', 'actual',
            'absolute_error', 'percentage_error', 'verdict'
        ] if c in df.columns]

        results = df[return_cols].head(50).to_dict('records')

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
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500

try:
    HISTORICAL_PEAKS = load_historical_peaks()
except Exception as e:
    print(f"⚠️ Skipped historical peaks load: {e}")

try:
    load_correction_factors()
except Exception as e:
    print(f"⚠️ Skipped correction factors load: {e}")