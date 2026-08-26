# services/lstm_integration.py

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import logging
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MRT3LSTMPredictor:
    """
    LSTM predictor that uses trained models from v10+ training pipeline.
    Uses the same feature engineering as the main prediction API.
    """
    
    def __init__(self, model_path='./models_2022-2024_v10_plus_latest/'):
        self.model_path = model_path
        self.models = {}
        self.feature_scalers = {}
        self.target_scalers = {}
        self.feature_cols = None
        self.station_directions = []
        
        # Station list
        self.stations = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
                        "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
                        "Ayala Ave", "Magallanes", "Taft"]
        self.directions = ['Northbound', 'Southbound']
        
    def load_models(self):
        """Load all trained LSTM models and scalers from v10+ training"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Model path not found: {self.model_path}")
                return False
                
            # Load feature columns
            feature_cols_path = f'{self.model_path}/feature_cols.pkl'
            if not os.path.exists(feature_cols_path):
                logger.error(f"❌ feature_cols.pkl not found at {feature_cols_path}")
                return False
                
            with open(feature_cols_path, 'rb') as f:
                self.feature_cols = pickle.load(f)
            
            logger.info(f"📋 Loaded {len(self.feature_cols)} feature columns")
            
            # Load models for each station-direction
            loaded_count = 0
            
            for station in self.stations:
                for direction in self.directions:
                    model_key = f"{station}_{direction}"
                    try:
                        # ========== FIX: Use v10+ file naming ==========
                        model_file = f'{self.model_path}/{model_key}_lstm_v10_plus.keras'
                        feature_scaler_file = f'{self.model_path}/{model_key}_feature_scaler.pkl'
                        target_scaler_file = f'{self.model_path}/{model_key}_target_scaler.pkl'
                        
                        if (os.path.exists(model_file) and 
                            os.path.exists(feature_scaler_file) and 
                            os.path.exists(target_scaler_file)):
                            
                            self.models[model_key] = load_model(model_file)
                            with open(feature_scaler_file, 'rb') as f:
                                self.feature_scalers[model_key] = pickle.load(f)
                            with open(target_scaler_file, 'rb') as f:
                                self.target_scalers[model_key] = pickle.load(f)
                            
                            self.station_directions.append(model_key)
                            loaded_count += 1
                            logger.debug(f"✅ Loaded model: {model_key}")
                    except Exception as e:
                        logger.debug(f"⚠️ Could not load {model_key}: {e}")
            
            logger.info(f"✅ Loaded {loaded_count} station-direction models")
            return loaded_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_feature_sequence(self, station, direction, target_datetime, seq_length=24):
        """
        Get feature sequence using the SAME function as the prediction API.
        This ensures consistency with the production system.
        """
        try:
            from services.feature_engineering import get_feature_sequence_for_station
            return get_feature_sequence_for_station(station, direction, target_datetime, seq_length)
        except ImportError as e:
            logger.error(f"❌ Could not import feature_engineering: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error getting feature sequence: {e}")
            return None
    
    def predict_congestion(self, station, direction, target_datetime=None):
        """
        Predict congestion using LSTM model.
        Uses the SAME approach as the main prediction API.
        
        Args:
            station: Station name
            direction: 'Northbound' or 'Southbound'
            target_datetime: datetime object (default: current time)
        
        Returns:
            predicted congestion percentage (0-100)
        """
        if target_datetime is None:
            target_datetime = datetime.now()
        
        model_key = f"{station}_{direction}"
        
        if model_key not in self.models:
            logger.warning(f"⚠️ No model for {model_key}")
            return None
        
        try:
            # ========== FIX: Use the SAME feature engineering as production ==========
            features_scaled = self.get_feature_sequence(station, direction, target_datetime)
            
            if features_scaled is None:
                logger.error(f"❌ No features for {model_key}")
                return None
            
            # ========== FIX: features_scaled is already scaled, just reshape ==========
            input_sequence = features_scaled.reshape(1, 24, -1)
            
            # Predict
            prediction_scaled = self.models[model_key].predict(input_sequence, verbose=0)
            raw_output = float(prediction_scaled[0][0])
            
            # Inverse transform to get passenger count
            target_scaler = self.target_scalers[model_key]
            passenger_count = float(target_scaler.inverse_transform([[raw_output]])[0][0])
            
            # ========== FIX: Use P95 instead of capacity ==========
            try:
                from routes.api_predict import get_p95_percentile
                p95 = get_p95_percentile(station, direction)
            except Exception as e:
                logger.warning(f"⚠️ Could not get P95 for {model_key}, using fallback: {e}")
                # Fallback: use station capacity * 0.8
                from constants import MRT3_PLATFORM_CAPACITY
                capacity = MRT3_PLATFORM_CAPACITY.get(station, 1000)
                p95 = capacity * 0.8
            
            # Calculate congestion using P95
            congestion = (passenger_count / p95) * 100
            congestion = max(0, min(100, congestion))
            
            logger.info(f"📊 {model_key}: {congestion:.1f}% ({passenger_count:.0f} passengers, P95={p95:.0f})")
            
            return congestion
            
        except Exception as e:
            logger.error(f"❌ Error predicting {model_key}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_all_stations(self, target_datetime=None):
        """
        Predict congestion for all station-direction combinations.
        """
        if target_datetime is None:
            target_datetime = datetime.now()
        
        results = {}
        
        for station in self.stations:
            results[station] = {}
            for direction in self.directions:
                congestion = self.predict_congestion(station, direction, target_datetime)
                results[station][direction] = congestion
        
        return results
    
    def get_model_info(self):
        """Get information about loaded models"""
        return {
            'total_models': len(self.models),
            'station_directions': self.station_directions,
            'feature_cols': self.feature_cols,
            'model_path': self.model_path
        }


# ============================================
# HELPER FUNCTIONS FOR FLASK INTEGRATION
# ============================================

def init_lstm_predictor(app):
    """
    Initialize LSTM predictor on app startup.
    Uses the latest v10+ model path.
    """
    # Try to find the latest v10+ model folder
    import glob
    
    model_pattern = './models_2022-2024_v10_plus_*/'
    matching_folders = glob.glob(model_pattern)
    
    if matching_folders:
        # Use the most recent folder (sorted by name, which includes timestamp)
        latest_folder = sorted(matching_folders)[-1]
        logger.info(f"📁 Using latest model folder: {latest_folder}")
    else:
        latest_folder = './models_2022-2024_v10_plus_latest/'
        logger.warning(f"⚠️ No v10+ model folder found, using: {latest_folder}")
    
    predictor = MRT3LSTMPredictor(model_path=latest_folder)
    
    if predictor.load_models():
        # Store in app config for access
        app.config['LSTM_PREDICTOR'] = predictor
        logger.info("✅ LSTM Predictor initialized successfully")
        return True
    else:
        logger.warning("⚠️ LSTM Predictor initialization failed")
        return False


def schedule_weekly_retraining(app):
    """
    Setup weekly retraining schedule.
    This should trigger the Kaggle notebook to retrain models.
    """
    import schedule
    import threading
    import time
    
    def weekly_retraining_job():
        """Job to retrain models weekly"""
        logger.info("📅 Running weekly model retraining...")
        
        with app.app_context():
            try:
                # Option 1: Call Kaggle API to trigger notebook
                import subprocess
                result = subprocess.run([
                    'kaggle', 'kernels', 'pull',
                    'your-kaggle-username/mrt3-lstm-training',
                    '-p', './models/'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ Kaggle retraining triggered successfully")
                    
                    # Reload models after retraining
                    predictor = app.config.get('LSTM_PREDICTOR')
                    if predictor:
                        predictor.load_models()
                        logger.info("✅ Models reloaded after retraining")
                else:
                    logger.error(f"❌ Kaggle retraining failed: {result.stderr}")
                    
            except ImportError:
                logger.warning("⚠️ Kaggle API not available, retraining skipped")
            except Exception as e:
                logger.error(f"❌ Retraining error: {e}")
    
    # Schedule weekly retraining (every Sunday at 3 AM)
    schedule.every().sunday.at("03:00").do(weekly_retraining_job)
    
    logger.info("📅 Weekly retraining scheduled for Sunday 3:00 AM")
    
    # Start scheduler in background
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    return scheduler_thread


# ============================================
# MODEL EVALUATION HELPER
# ============================================

def evaluate_model_performance(predictor, station, direction, test_days=30):
    """
    Evaluate model performance against historical data.
    Uses the same P95 approach as the main API.
    """
    try:
        from services.feature_engineering import get_station_dataframe
        from routes.api_predict import get_p95_percentile
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        df = get_station_dataframe(station, direction)
        if df is None or len(df) == 0:
            return None
        
        end_date = df.index.max()
        start_date = end_date - timedelta(days=test_days)
        test_data = df[(df.index >= start_date) & (df.index < end_date)]
        
        p95 = get_p95_percentile(station, direction)
        
        actuals = []
        predictions = []
        
        for timestamp in test_data.index:
            actual_passengers = test_data.loc[timestamp, 'TotalPassenger']
            actual_congestion = (actual_passengers / p95) * 100
            
            pred_congestion = predictor.predict_congestion(station, direction, timestamp)
            
            if pred_congestion is not None:
                predictions.append(pred_congestion)
                actuals.append(min(actual_congestion, 100))
        
        if len(predictions) == 0:
            return None
        
        return {
            'mae': mean_absolute_error(actuals, predictions),
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'r2': r2_score(actuals, predictions),
            'sample_count': len(predictions),
            'test_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error evaluating model: {e}")
        return None