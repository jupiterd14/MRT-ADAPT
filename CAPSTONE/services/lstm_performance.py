# services/lstm_performance.py - UPDATED with capacity-based evaluation
"""
LSTM Model Performance Testing Service - Wrapper for existing models
Includes confusion matrix, precision, recall, F1-score, and accuracy metrics
"""

from services.model_loader import directional_models, directional_scalers
from services.feature_engineering import get_feature_sequence_for_station
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
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

# Import the prediction function
from services import get_directional_prediction

RESULTS_FOLDER = 'test_results'
EVALUATION_FOLDER = 'evaluation_results'

# Congestion categories for classification metrics
CONGESTION_CATEGORIES = {
    'Light': (0, 30),
    'Moderate': (30, 60),
    'Heavy': (60, 80),
    'Severe': (80, 100)
}

CATEGORY_ORDER = ['Light', 'Moderate', 'Heavy', 'Severe']

# ========== STATION CAPACITIES ==========
MRT3_CAPACITY = {
    "North Ave": 1142, "Quezon Ave": 1195, "Kamuning": 1364,
    "Cubao": 1747, "Santolan": 1306, "Ortigas": 1331,
    "Shaw Blvd": 1619, "Boni Ave": 1417, "Guadalupe": 1301,
    "Buendia": 1645, "Ayala Ave": 1222, "Magallanes": 1202,
    "Taft": 720
}

class LSTMPerformanceService:
    def __init__(self):
        """Wrapper for existing model loader"""
        self.models = directional_models
        self.scalers = directional_scalers
        os.makedirs(EVALUATION_FOLDER, exist_ok=True)
        print(f"LSTMPerformanceService wrapping {len(self.models)} existing models")
    
    def predict_single(self, station, direction, target_datetime, historical_data=None):
        """Single prediction using existing get_directional_prediction"""
        try:
            target_time = pd.to_datetime(target_datetime)
            
            result = get_directional_prediction(
                station, direction, target_time,
                self.models, self.scalers,
                get_feature_sequence_for_station
            )
            
            if result is not None:
                return {
                    "success": True,
                    "predicted_congestion": round(float(result), 1),
                    "station": station,
                    "direction": direction,
                    "target_time": target_datetime
                }
            return {"success": False, "error": f"Prediction failed for {station} {direction}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_congestion_category(self, congestion_value):
        """Convert congestion percentage to category"""
        if congestion_value > 80:
            return 'Severe'
        elif congestion_value > 60:
            return 'Heavy'
        elif congestion_value > 30:
            return 'Moderate'
        else:
            return 'Light'
    
    def _calculate_actual_congestion(self, total_passengers, station_name):
        """Calculate actual congestion using station capacity"""
        capacity = MRT3_CAPACITY.get(station_name, 1000)
        if capacity > 0:
            congestion = (total_passengers / capacity * 100)
            return min(100, congestion)
        return 0
    
    def evaluate_model_performance(self, station=None, direction=None, days_back=30):
        """
        Evaluate model performance with confusion matrix and classification metrics
        Uses historical data from test_results or generates new predictions
        """
        # Load test results
        all_results = self._load_test_results(station, direction)
        
        if len(all_results) == 0:
            return {
                "success": False,
                "error": "No test data available. Run auto-tests first.",
                "total_samples": 0
            }
        
        # Get predictions and actuals
        predictions = all_results['predicted'].values
        
        # ========== FIX: Recalculate actual using capacity if needed ==========
        # Check if actual values are suspicious (all 100%)
        actuals = all_results['actual'].values
        if len(actuals) > 0 and np.mean(actuals) > 90:
            print("⚠️ Suspicious actual values detected (all > 90%). Recalculating from TotalPassenger...")
            # Recalculate actual from TotalPassenger using capacity
            recalculated_actuals = []
            for _, row in all_results.iterrows():
                station_name = row.get('station', station if station else 'North Ave')
                total_pass = row.get('total_passengers', 0)
                # Use capacity to calculate actual
                capacity = MRT3_CAPACITY.get(station_name, 1000)
                if capacity > 0:
                    actual = (total_pass / capacity * 100)
                    recalculated_actuals.append(min(100, actual))
                else:
                    recalculated_actuals.append(actuals.iloc[len(recalculated_actuals)])
            
            # Update actuals with recalculated values
            actuals = np.array(recalculated_actuals)
            all_results['actual'] = actuals
        
        # Convert to categories
        pred_categories = [self.get_congestion_category(p) for p in predictions]
        actual_categories = [self.get_congestion_category(a) for a in actuals]
        
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
        
        # Prepare per-class metrics for frontend
        per_class_metrics = {}
        for category in CATEGORY_ORDER:
            if category in class_report:
                per_class_metrics[category] = {
                    'precision': round(class_report[category]['precision'] * 100, 1),
                    'recall': round(class_report[category]['recall'] * 100, 1),
                    'f1_score': round(class_report[category]['f1-score'] * 100, 1),
                    'support': class_report[category]['support']
                }
        
        # Calculate category distribution
        category_distribution = {}
        for category in CATEGORY_ORDER:
            category_distribution[category] = {
                'actual': actual_categories.count(category),
                'predicted': pred_categories.count(category)
            }
        
        # Save evaluation results
        evaluation_data = {
            'station': station if station else 'all',
            'direction': direction if direction else 'both',
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
            'category_distribution': category_distribution,
            'capacity_used': True
        }
        
        # Save to file
        filename = f"{EVALUATION_FOLDER}/evaluation_{station if station else 'all'}_{direction if direction else 'both'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(evaluation_data, f, indent=2)
        
        return {
            "success": True,
            "data": evaluation_data
        }
    
    def get_evaluation_history(self, station=None, direction=None):
        """Get historical evaluation results"""
        if not os.path.exists(EVALUATION_FOLDER):
            return []
        
        evaluations = []
        for f in os.listdir(EVALUATION_FOLDER):
            if f.endswith('.json'):
                try:
                    with open(os.path.join(EVALUATION_FOLDER, f), 'r') as file:
                        data = json.load(file)
                        if station and data.get('station') != station:
                            continue
                        if direction and data.get('direction') != direction:
                            continue
                        evaluations.append(data)
                except:
                    continue
        
        evaluations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return evaluations
    
    def get_latest_evaluation(self, station=None, direction=None):
        """Get the latest evaluation results"""
        evaluations = self.get_evaluation_history(station, direction)
        if evaluations:
            return evaluations[0]
        return None
    
    def _load_test_results(self, station=None, direction=None):
        """Load test results from CSV files"""
        if not os.path.exists(RESULTS_FOLDER):
            return pd.DataFrame()
        
        all_results = []
        for f in os.listdir(RESULTS_FOLDER):
            if f.endswith('_results.csv') and not f.startswith('full_'):
                try:
                    df = pd.read_csv(os.path.join(RESULTS_FOLDER, f))
                    
                    # Ensure numeric columns
                    for col in ['predicted', 'actual', 'absolute_error']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Filter by station and direction
                    if station:
                        df = df[df['station'] == station]
                    if direction:
                        df = df[df['direction'].str.lower() == direction.lower()]
                    
                    # Drop rows with NaN
                    df = df.dropna(subset=['predicted', 'actual'])
                    
                    if not df.empty:
                        all_results.append(df)
                except Exception as e:
                    print(f"Error loading {f}: {e}")
                    continue
        
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()
    
    def get_performance_metrics(self):
        """Get metrics from test_results"""
        if not os.path.exists(RESULTS_FOLDER):
            return {"total_tests": 0}
        
        all_results = []
        for f in os.listdir(RESULTS_FOLDER):
            if f.endswith('_results.csv') and not f.startswith('full_'):
                try:
                    df = pd.read_csv(os.path.join(RESULTS_FOLDER, f))
                    if 'absolute_error' in df.columns:
                        df['absolute_error'] = pd.to_numeric(df['absolute_error'], errors='coerce')
                    all_results.append(df)
                except:
                    continue
        
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            latest_eval = self.get_latest_evaluation()
            
            return {
                "total_tests": len(combined),
                "overall_mae": round(combined['absolute_error'].mean(), 2) if 'absolute_error' in combined.columns else 0,
                "overall_mape": round(combined['percentage_error'].mean(), 2) if 'percentage_error' in combined.columns else 0,
                "accuracy": latest_eval.get('accuracy', 0) if latest_eval else 0,
                "f1_score": latest_eval.get('f1_weighted', 0) if latest_eval else 0,
                "last_evaluated": latest_eval.get('timestamp', 'Never') if latest_eval else 'Never'
            }
        return {"total_tests": 0}
    
    def get_available_stations(self):
        """Get stations from loaded models"""
        stations = set()
        for key in self.models.keys():
            station = key.split('_')[0]
            stations.add(station)
        return sorted(list(stations))
    
    def get_chart_data(self, station="all", direction="both"):
        """Get chart data from test_results"""
        if not os.path.exists(RESULTS_FOLDER):
            return {"labels": [], "predicted": [], "actual": [], "errors": []}
        
        all_results = []
        for f in os.listdir(RESULTS_FOLDER):
            if f.endswith('_results.csv') and not f.startswith('full_'):
                try:
                    df = pd.read_csv(os.path.join(RESULTS_FOLDER, f))
                    # Ensure numeric
                    for col in ['predicted', 'actual', 'absolute_error']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.dropna(subset=['predicted', 'actual'])
                    all_results.append(df)
                except:
                    continue
        
        if not all_results:
            return {"labels": [], "predicted": [], "actual": [], "errors": []}
        
        combined = pd.concat(all_results, ignore_index=True)
        
        if station != "all":
            combined = combined[combined['station'] == station]
        if direction != "both":
            combined = combined[combined['direction'].str.lower() == direction.lower()]
        
        if not combined.empty:
            combined = combined.sort_values('target_time')
            return {
                "labels": combined['target_time'].tolist(),
                "predicted": combined['predicted'].tolist(),
                "actual": combined['actual'].tolist(),
                "errors": combined['absolute_error'].tolist() if 'absolute_error' in combined.columns else [0] * len(combined)
            }
        
        return {"labels": [], "predicted": [], "actual": [], "errors": []}
    
    def get_station_details(self, station, direction):
        """Get detailed evaluation for a specific station-direction"""
        eval_data = self.get_latest_evaluation(station, direction)
        
        if not eval_data:
            eval_result = self.evaluate_model_performance(station, direction)
            if eval_result.get('success'):
                eval_data = eval_result.get('data')
            else:
                return {"success": False, "error": eval_result.get('error', 'No data available')}
        
        df = self._load_test_results(station, direction)
        
        if df.empty:
            return {"success": False, "error": "No test data available"}
        
        df = df.sort_values('target_time').tail(50)
        
        sample_data = []
        for _, row in df.iterrows():
            pred_cat = self.get_congestion_category(row['predicted'])
            actual_cat = self.get_congestion_category(row['actual'])
            verdict = 'CORRECT' if pred_cat == actual_cat else 'INCORRECT'
            
            sample_data.append({
                'target_time': row['target_time'],
                'predicted': round(row['predicted'], 1),
                'actual': round(row['actual'], 1),
                'absolute_error': round(row['absolute_error'], 1),
                'predicted_category': pred_cat,
                'actual_category': actual_cat,
                'verdict': verdict
            })
        
        return {
            "success": True,
            "data": sample_data,
            "metrics": eval_data
        }