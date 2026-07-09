# services/lstm_integration.py

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import func, and_
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MRT3LSTMPredictor:
    """LSTM predictor that uses trained models from Kaggle"""
    
    def __init__(self, model_path='./models_2022-2024_v8/'):
        self.model_path = model_path
        self.models = {}
        self.feature_scalers = {}
        self.target_scalers = {}
        self.feature_cols = None
        self.capacities = None
        self.station_directions = []
        
    def load_models(self):
        """Load all trained LSTM models and scalers"""
        try:
            # Check if model path exists
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
            
            # Load capacities
            capacities_path = f'{self.model_path}/station_platform_capacities.pkl'
            if not os.path.exists(capacities_path):
                logger.error(f"❌ station_platform_capacities.pkl not found at {capacities_path}")
                return False
                
            with open(capacities_path, 'rb') as f:
                self.capacities = pickle.load(f)
            
            # Load models for each station-direction
            stations = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
                       "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
                       "Ayala Ave", "Magallanes", "Taft"]
            
            directions = ['Northbound', 'Southbound']
            loaded_count = 0
            
            for station in stations:
                for direction in directions:
                    model_key = f"{station}_{direction}"
                    try:
                        model_file = f'{self.model_path}/{model_key}_lstm_enhanced.keras'
                        feature_scaler_file = f'{self.model_path}/{model_key}_feature_scaler.pkl'
                        target_scaler_file = f'{self.model_path}/{model_key}_target_scaler.pkl'
                        
                        if os.path.exists(model_file) and os.path.exists(feature_scaler_file) and os.path.exists(target_scaler_file):
                            self.models[model_key] = load_model(model_file)
                            self.feature_scalers[model_key] = pickle.load(open(feature_scaler_file, 'rb'))
                            self.target_scalers[model_key] = pickle.load(open(target_scaler_file, 'rb'))
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
    
    def prepare_features_for_prediction(self, station, direction, current_time, historical_data=None):
        """
        Prepare features for LSTM prediction
        
        Args:
            station: Station name
            direction: 'Northbound' or 'Southbound'
            current_time: datetime object
            historical_data: DataFrame with historical passenger data
        """
        if self.feature_cols is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Create features DataFrame
        features = {}
        
        # Time features
        features['hour'] = current_time.hour
        features['weekday'] = current_time.weekday()
        features['month'] = current_time.month
        
        # Cyclical features
        features['hour_sin'] = np.sin(2 * np.pi * current_time.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * current_time.hour / 24)
        features['dow_sin'] = np.sin(2 * np.pi * current_time.weekday() / 7)
        features['dow_cos'] = np.cos(2 * np.pi * current_time.weekday() / 7)
        features['month_sin'] = np.sin(2 * np.pi * (current_time.month - 1) / 12)
        features['month_cos'] = np.cos(2 * np.pi * (current_time.month - 1) / 12)
        
        # Operating hours
        time_decimal = current_time.hour + current_time.minute / 60
        features['time_decimal'] = time_decimal
        features['is_operating_hour'] = 1 if 4.5 <= time_decimal < 23.0 else 0
        
        # Rush hours
        features['is_morning_rush'] = 1 if 7.0 <= time_decimal <= 9.0 else 0
        features['is_evening_rush'] = 1 if 17.0 <= time_decimal <= 19.0 else 0
        features['is_noon'] = 1 if 12.0 <= time_decimal <= 13.0 else 0
        features['is_pre_opening'] = 1 if 4.5 <= time_decimal < 5.0 else 0
        features['is_post_closing'] = 1 if 22.5 <= time_decimal < 23.0 else 0
        
        # Time normalized
        features['minutes_until_closing'] = max(0, (23.0 - time_decimal) * 60)
        features['minutes_since_opening'] = max(0, (time_decimal - 4.5) * 60)
        features['time_normalized'] = max(0, min(1, (time_decimal - 4.5) / (23.0 - 4.5)))
        features['minute_normalized'] = current_time.minute / 60.0
        
        # Calendar features
        features['is_weekend'] = 1 if current_time.weekday() >= 5 else 0
        features['is_holiday'] = 0  # You can add holiday detection
        features['is_special_event'] = 0
        features['is_christmas_season'] = 1 if current_time.month == 12 or current_time.month == 1 else 0
        features['is_payday'] = 1 if current_time.day in [15, 30, 31] else 0
        features['is_friday'] = 1 if current_time.weekday() == 4 else 0
        features['is_rush_hour'] = 1 if features['is_morning_rush'] or features['is_evening_rush'] else 0
        
        # Maintenance flags
        features['is_maintenance_record'] = 0
        features['is_extended_hours'] = 0
        
        # Congestion from historical data for lookback
        if historical_data is not None and len(historical_data) > 0:
            if 'congestion' in historical_data.columns:
                features['congestion'] = float(historical_data.iloc[-1]['congestion'])
            else:
                features['congestion'] = 50.0
        else:
            features['congestion'] = 50.0
        
        # Create DataFrame with correct column order
        feature_df = pd.DataFrame([features])
        
        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        # Reorder columns to match training
        feature_df = feature_df[self.feature_cols]
        
        return feature_df
    
    def get_historical_data(self, station, direction, db_session, lookback_hours=24):
        """Get historical passenger data from reports or database"""
        try:
            from models import Report
            
            # Try to get from reports first
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            
            reports = db_session.query(
                Report.timestamp,
                Report.reported_congestion
            ).filter(
                and_(
                    Report.station == station,
                    Report.direction == direction,
                    Report.timestamp >= cutoff_time,
                    Report.is_flagged == False
                )
            ).order_by(Report.timestamp.asc()).all()
            
            if len(reports) > 0:
                # Convert to DataFrame
                historical = pd.DataFrame([
                    {'timestamp': r.timestamp, 'congestion': float(r.reported_congestion)}
                    for r in reports
                ])
                return historical
            
            # Fallback: use average congestion from last 7 days
            seven_days_ago = datetime.now() - timedelta(days=7)
            avg_congestion = db_session.query(
                func.avg(Report.reported_congestion)
            ).filter(
                and_(
                    Report.station == station,
                    Report.direction == direction,
                    Report.timestamp >= seven_days_ago,
                    Report.is_flagged == False
                )
            ).scalar()
            
            if avg_congestion is not None:
                # Create synthetic historical data
                timestamps = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
                historical = pd.DataFrame({
                    'timestamp': timestamps,
                    'congestion': [float(avg_congestion)] * 24
                })
                return historical
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None
    
    def predict_congestion(self, station, direction, db_session, current_time=None):
        """
        Predict congestion using LSTM model
        
        Returns: predicted congestion percentage (0-100)
        """
        if current_time is None:
            current_time = datetime.now()
        
        model_key = f"{station}_{direction}"
        
        if model_key not in self.models:
            logger.warning(f"⚠️ No model for {model_key}")
            return None
        
        try:
            # Get historical data for lookback
            historical_data = self.get_historical_data(station, direction, db_session)
            
            # Prepare features
            features_df = self.prepare_features_for_prediction(
                station, direction, current_time, historical_data
            )
            
            # Scale features
            features_scaled = self.feature_scalers[model_key].transform(features_df)
            
            # Reshape for LSTM (need 24 timesteps)
            sequence_length = 24
            features_reshaped = np.tile(features_scaled, (1, sequence_length, 1))
            
            # Predict
            prediction_scaled = self.models[model_key].predict(features_reshaped, verbose=0)
            
            # Inverse transform to get passenger count
            prediction_passengers = self.target_scalers[model_key].inverse_transform(
                prediction_scaled.reshape(-1, 1)
            )[0][0]
            
            # Convert to congestion percentage
            capacity = self.capacities.get(station, 1000)
            congestion = (prediction_passengers / capacity) * 100
            congestion = max(0, min(100, congestion))
            
            logger.info(f"📊 {model_key}: {congestion:.1f}% ({prediction_passengers:.0f} passengers)")
            
            return congestion
            
        except Exception as e:
            logger.error(f"❌ Error predicting {model_key}: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================
# HELPER FUNCTIONS FOR FLASK INTEGRATION
# ============================================

def init_lstm_predictor(app):
    """Initialize LSTM predictor on app startup"""
    predictor = MRT3LSTMPredictor()
    
    if predictor.load_models():
        # Store in app config for access
        app.config['LSTM_PREDICTOR'] = predictor
        logger.info("✅ LSTM Predictor initialized successfully")
        return True
    else:
        logger.warning("⚠️ LSTM Predictor initialization failed")
        return False


def schedule_weekly_retraining(app):
    """Setup weekly retraining schedule"""
    import schedule
    import threading
    import time
    
    def weekly_retraining_job():
        """Job to retrain models weekly with new report data"""
        logger.info("📅 Running weekly model retraining...")
        
        with app.app_context():
            from models import db
            # Call your retraining function here
            try:
                from training.scheduled_trainer import retrain_models_with_reports
                retrain_models_with_reports(db.session)
            except ImportError:
                logger.warning("⚠️ Training module not available, retraining skipped")
    
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