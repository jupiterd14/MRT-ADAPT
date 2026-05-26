# services/lstm_performance.py - SIMPLE WRAPPER VERSION
"""
LSTM Model Performance Testing Service - Wrapper for existing models
"""

from services.model_loader import directional_models, directional_scalers
from services.feature_engineering import get_feature_sequence_for_station
import pandas as pd
import os
import pickle

# Import the prediction function from the correct location
from services import get_directional_prediction

MAX_PATH = 'models_2022-2024_v5/per_direction_max_passengers.pkl' 
with open(MAX_PATH, 'rb') as f:
    PER_DIRECTION_MAX = pickle.load(f)

RESULTS_FOLDER = 'test_results'

class LSTMPerformanceService:
    def __init__(self):
        """Wrapper for existing model loader"""
        self.models = directional_models
        self.scalers = directional_scalers
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
    
    def get_performance_metrics(self):
        """Get metrics from test_results"""
        if not os.path.exists(RESULTS_FOLDER):
            return {"total_tests": 0}
        
        all_results = []
        for f in os.listdir(RESULTS_FOLDER):
            if f.endswith('_results.csv'):
                df = pd.read_csv(os.path.join(RESULTS_FOLDER, f))
                all_results.append(df)
        
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            return {
                "total_tests": len(combined),
                "overall_mae": round(combined['absolute_error'].mean(), 2),
                "overall_mape": round(combined['percentage_error'].mean(), 2)
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
                df = pd.read_csv(os.path.join(RESULTS_FOLDER, f))
                all_results.append(df)
        
        if not all_results:
            return {"labels": [], "predicted": [], "actual": [], "errors": []}
        
        combined = pd.concat(all_results, ignore_index=True)
        
        if station != "all":
            combined = combined[combined['station'] == station]
        if direction != "both":
            combined = combined[combined['direction'].str.lower() == direction.lower()]
        
        combined = combined.sort_values('target_time')
        
        return {
            "labels": combined['target_time'].tolist(),
            "predicted": combined['predicted'].tolist(),
            "actual": combined['actual'].tolist(),
            "errors": combined['absolute_error'].tolist() if 'absolute_error' in combined.columns else [0] * len(combined)
        }