# prediction_api.py
# MRT3 Prediction API with LSTM Models

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class MRT3Predictor:
    """MRT-3 Station Congestion Predictor using LSTM models"""
    
    def __init__(self, models_dir='models/'):
        """
        Initialize the predictor with LSTM models
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.stations = [
            "North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan",
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia",
            "Ayala Ave", "Magallanes", "Taft"
        ]
        
        # Station base capacities
        self.station_capacities = {
            "North Ave": 12000, "Quezon Ave": 9000, "Kamuning": 7500, "Cubao": 15000,
            "Santolan": 8000, "Ortigas": 9500, "Shaw Blvd": 11000, "Boni Ave": 8500,
            "Guadalupe": 10000, "Buendia": 9000, "Ayala Ave": 14000, "Magallanes": 9000,
            "Taft": 16000
        }
        
        # Load all available models
        self._load_models()
        
    def _load_models(self):
        """Load all LSTM models and scalers"""
        loaded_count = 0
        
        for station in self.stations:
            # Try .keras format first (newer)
            model_path_keras = os.path.join(self.models_dir, f'{station}_lstm.keras')
            # Try .h5 format (older)
            model_path_h5 = os.path.join(self.models_dir, f'{station}_lstm.h5')
            scaler_path = os.path.join(self.models_dir, f'{station}_scaler.pkl')
            
            model_path = None
            if os.path.exists(model_path_keras):
                model_path = model_path_keras
            elif os.path.exists(model_path_h5):
                model_path = model_path_h5
            
            if model_path and os.path.exists(scaler_path):
                try:
                    # Load model without compilation to avoid optimizer issues
                    self.models[station] = tf.keras.models.load_model(
                        model_path, 
                        compile=False
                    )
                    with open(scaler_path, 'rb') as f:
                        self.scalers[station] = pickle.load(f)
                    loaded_count += 1
                    print(f"✅ Loaded LSTM model for {station}")
                except Exception as e:
                    print(f"⚠️ Error loading {station} model: {e}")
            else:
                print(f"⚠️ Model files not found for {station}")
        
        print(f"📊 Loaded {loaded_count}/{len(self.stations)} LSTM models")
        self.models_loaded = loaded_count
        
    def _create_sequence(self, data, seq_length=24):
        """Create sequences for LSTM prediction"""
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)
    
    def _prepare_features(self, station, target_datetime=None):
        """
        Prepare feature vector for prediction
        
        Args:
            station: Station name
            target_datetime: datetime object for prediction (default: now)
        
        Returns:
            numpy array of features
        """
        if target_datetime is None:
            target_datetime = datetime.now()
        
        # Extract time features
        hour = target_datetime.hour
        minute = target_datetime.minute
        weekday = target_datetime.weekday()
        is_weekend = 1 if weekday >= 5 else 0
        is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
        
        # Calculate time of day fraction
        time_fraction = (hour * 60 + minute) / (24 * 60)
        
        # Station index (one-hot encoding simplified)
        station_idx = self.stations.index(station) / len(self.stations)
        
        # Capacity ratio
        capacity = self.station_capacities.get(station, 10000)
        
        # Historical average for this hour (if available)
        historical_avg = self._get_historical_average(station, hour)
        
        # Create feature vector
        features = np.array([
            hour / 24.0,           # Normalized hour
            minute / 60.0,         # Normalized minute
            time_fraction,         # Time of day fraction
            weekday / 7.0,         # Normalized weekday
            is_weekend,            # Weekend flag
            is_rush_hour,          # Rush hour flag
            station_idx,           # Station index
            historical_avg / capacity,  # Normalized historical average
        ])
        
        return features.reshape(1, -1)
    
    def _get_historical_average(self, station, hour):
        """
        Get historical average ridership for a station at specific hour
        This should be loaded from your cache or generated
        
        Returns:
            float: Historical average ridership
        """
        # Define typical hourly patterns based on station
        base_patterns = {
            # Morning rush stations (northern)
            "North Ave": {7: 8500, 8: 9500, 9: 8000, 17: 7500, 18: 8000, 19: 7000},
            "Quezon Ave": {7: 7000, 8: 8000, 9: 7000, 17: 6500, 18: 7000, 19: 6000},
            "Kamuning": {7: 5500, 8: 6500, 9: 6000, 17: 5500, 18: 6000, 19: 5000},
            "Cubao": {7: 10000, 8: 12000, 9: 10000, 17: 9500, 18: 10000, 19: 8500},
            "Santolan": {7: 6000, 8: 7000, 9: 6000, 17: 5500, 18: 6000, 19: 5000},
            # Central stations
            "Ortigas": {7: 7000, 8: 8000, 9: 7000, 17: 7000, 18: 7500, 19: 6500},
            "Shaw Blvd": {7: 7500, 8: 8500, 9: 7500, 17: 7500, 18: 8000, 19: 7000},
            "Boni Ave": {7: 6000, 8: 7000, 9: 6000, 17: 6000, 18: 6500, 19: 5500},
            "Guadalupe": {7: 6500, 8: 7500, 9: 6500, 17: 6500, 18: 7000, 19: 6000},
            # Southern stations
            "Buendia": {7: 5500, 8: 6500, 9: 6000, 17: 6500, 18: 7000, 19: 6500},
            "Ayala Ave": {7: 8500, 8: 10000, 9: 9000, 17: 9500, 18: 10500, 19: 9000},
            "Magallanes": {7: 5000, 8: 6000, 9: 5500, 17: 6000, 18: 6500, 19: 6000},
            "Taft": {7: 7500, 8: 8500, 9: 8000, 17: 8500, 18: 9500, 19: 8500}
        }
        
        # Get pattern for this station
        pattern = base_patterns.get(station, {})
        
        # Return pattern for this hour or generate a reasonable default
        if hour in pattern:
            return pattern[hour]
        else:
            # Default based on time of day
            if 7 <= hour <= 9:
                return 6000
            elif 17 <= hour <= 19:
                return 6500
            elif 10 <= hour <= 16:
                return 4500
            else:
                return 2000
    
    def predict(self, station_name, target_datetime=None):
        """
        Predict congestion for a station
        
        Args:
            station_name: Name of the station
            target_datetime: datetime for prediction (default: now)
        
        Returns:
            dict: Prediction results
        """
        # Normalize station name
        station = station_name.replace('%20', ' ')
        
        if station not in self.stations:
            return {
                'success': False,
                'error': f'Station "{station}" not found',
                'station': station
            }
        
        if target_datetime is None:
            target_datetime = datetime.now()
        
        # Check if operating hours
        hour = target_datetime.hour
        minute = target_datetime.minute
        
        if hour < 4 or (hour == 4 and minute < 30) or hour >= 23:
            return {
                'success': True,
                'station': station,
                'predicted_ridership': 0,
                'congestion_percentage': 0,
                'status': 'STATION CLOSED',
                'is_operating': False,
                'message': 'Station is closed. Operating hours: 4:30 AM - 11:00 PM'
            }
        
        # Try LSTM prediction if model exists
        if station in self.models and station in self.scalers:
            try:
                # Prepare features
                features = self._prepare_features(station, target_datetime)
                
                # Scale features
                scaled_features = self.scalers[station].transform(features)
                
                # Reshape for LSTM [samples, timesteps, features]
                # Assuming model expects 3D input
                if len(scaled_features.shape) == 2:
                    scaled_features = scaled_features.reshape(1, 1, -1)
                
                # Predict
                prediction = self.models[station].predict(scaled_features, verbose=0)
                
                # Extract ridership value
                if isinstance(prediction, np.ndarray):
                    ridership = float(prediction[0][0] if len(prediction.shape) > 1 else prediction[0])
                else:
                    ridership = float(prediction)
                
                # Ensure ridership is reasonable
                capacity = self.station_capacities.get(station, 10000)
                ridership = max(50, min(ridership, capacity))
                
                # Calculate congestion percentage
                congestion_pct = min(100, int((ridership / capacity) * 100))
                
                # Determine status
                if congestion_pct > 80:
                    status = "CRITICAL"
                elif congestion_pct > 60:
                    status = "CONGESTED"
                elif congestion_pct > 30:
                    status = "MODERATE"
                else:
                    status = "LIGHT"
                
                return {
                    'success': True,
                    'station': station,
                    'predicted_ridership': int(ridership),
                    'congestion_percentage': congestion_pct,
                    'status': status,
                    'is_operating': True,
                    'model_used': 'LSTM',
                    'timestamp': target_datetime.isoformat(),
                    'capacity': capacity
                }
                
            except Exception as e:
                print(f"⚠️ LSTM prediction failed for {station}: {e}")
                # Fall through to rule-based
        
        # Fallback to rule-based prediction
        return self._predict_rule_based(station, target_datetime)
    
    def _predict_rule_based(self, station, target_datetime):
        """
        Rule-based fallback prediction when LSTM is unavailable
        
        Args:
            station: Station name
            target_datetime: datetime for prediction
        
        Returns:
            dict: Prediction results
        """
        hour = target_datetime.hour
        weekday = target_datetime.weekday()
        is_weekend = weekday >= 5
        
        capacity = self.station_capacities.get(station, 10000)
        
        # Base ridership by time of day
        if 7 <= hour <= 9:  # Morning rush
            base_ridership = capacity * 0.75
        elif 17 <= hour <= 19:  # Evening rush
            base_ridership = capacity * 0.70
        elif 10 <= hour <= 16:  # Midday
            base_ridership = capacity * 0.45
        elif 20 <= hour <= 22:  # Late evening
            base_ridership = capacity * 0.30
        elif 5 <= hour <= 6:  # Early morning
            base_ridership = capacity * 0.20
        else:  # Late night / early morning
            base_ridership = capacity * 0.10
        
        # Station-specific multiplier
        station_multipliers = {
            "North Ave": 1.2, "Quezon Ave": 1.0, "Kamuning": 0.9,
            "Cubao": 1.3, "Santolan": 0.8, "Ortigas": 1.1,
            "Shaw Blvd": 1.05, "Boni Ave": 0.85, "Guadalupe": 0.95,
            "Buendia": 0.8, "Ayala Ave": 1.25, "Magallanes": 0.75,
            "Taft": 1.15
        }
        
        multiplier = station_multipliers.get(station, 1.0)
        
        # Weekend adjustment
        if is_weekend:
            multiplier *= 0.7
        
        # Apply multiplier
        ridership = int(base_ridership * multiplier)
        ridership = max(50, min(ridership, capacity))
        
        congestion_pct = min(100, int((ridership / capacity) * 100))
        
        if congestion_pct > 80:
            status = "CRITICAL"
        elif congestion_pct > 60:
            status = "CONGESTED"
        elif congestion_pct > 30:
            status = "MODERATE"
        else:
            status = "LIGHT"
        
        return {
            'success': True,
            'station': station,
            'predicted_ridership': ridership,
            'congestion_percentage': congestion_pct,
            'status': status,
            'is_operating': True,
            'model_used': 'rule-based (fallback)',
            'timestamp': target_datetime.isoformat(),
            'capacity': capacity
        }
    
    def predict_all_stations(self, target_datetime=None):
        """
        Get predictions for all stations
        
        Args:
            target_datetime: datetime for prediction (default: now)
        
        Returns:
            dict: Predictions for all stations
        """
        if target_datetime is None:
            target_datetime = datetime.now()
        
        results = {}
        for station in self.stations:
            results[station] = self.predict(station, target_datetime)
        
        return {
            'success': True,
            'timestamp': target_datetime.isoformat(),
            'predictions': results,
            'summary': {
                'total_stations': len(self.stations),
                'critical_count': sum(1 for r in results.values() if r.get('congestion_percentage', 0) > 80),
                'congested_count': sum(1 for r in results.values() if 60 < r.get('congestion_percentage', 0) <= 80),
                'moderate_count': sum(1 for r in results.values() if 30 < r.get('congestion_percentage', 0) <= 60),
                'light_count': sum(1 for r in results.values() if r.get('congestion_percentage', 0) <= 30)
            }
        }
    
    def get_station_info(self, station_name):
        """
        Get station information including capacity and typical patterns
        
        Args:
            station_name: Name of the station
        
        Returns:
            dict: Station information
        """
        station = station_name.replace('%20', ' ')
        
        if station not in self.stations:
            return {'error': f'Station "{station}" not found'}
        
        return {
            'station': station,
            'capacity': self.station_capacities.get(station, 10000),
            'has_lstm_model': station in self.models,
            'index': self.stations.index(station),
            'typical_rush_hours': ['7:00-9:00', '17:00-19:00']
        }


# Optional: Test the predictor when run directly
if __name__ == '__main__':
    print("🧪 Testing MRT3 Predictor...")
    predictor = MRT3Predictor()
    
    # Test prediction for a station
    test_station = "North Ave"
    result = predictor.predict(test_station)
    
    print(f"\n📊 Prediction for {test_station}:")
    print(f"   Ridership: {result.get('predicted_ridership', 'N/A')}")
    print(f"   Congestion: {result.get('congestion_percentage', 'N/A')}%")
    print(f"   Status: {result.get('status', 'N/A')}")
    print(f"   Model: {result.get('model_used', 'N/A')}")
    
    # Test all stations
    print("\n📊 All stations summary:")
    all_results = predictor.predict_all_stations()
    summary = all_results.get('summary', {})
    print(f"   Critical: {summary.get('critical_count', 0)}")
    print(f"   Congested: {summary.get('congested_count', 0)}")
    print(f"   Moderate: {summary.get('moderate_count', 0)}")
    print(f"   Light: {summary.get('light_count', 0)}")