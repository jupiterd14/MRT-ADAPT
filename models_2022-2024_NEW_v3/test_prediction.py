#!/usr/bin/env python3
"""Complete test for directional models with proper feature engineering"""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
MODELS_PATH = 'models_2022-2024_NEW'
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

# Matching training.py feature columns (ORDER MATTERS!)
FEATURE_COLS = ['hour', 'weekday', 'month', 'is_weekend', 'is_holiday',
                'is_special_event', 'is_christmas_season', 'is_payday', 
                'is_friday', 'is_rush_hour', 'congestion']

def is_christmas_season(date):
    """Check if date is within Christmas season (Dec 15 - Jan 5)"""
    month_day = date.strftime('%m-%d')
    return (month_day >= '12-15') or (month_day <= '01-05')

def is_payday(date):
    """Check if date is a payday (15th, 30th, 31st)"""
    return date.day in [15, 30, 31]

def is_friday(date):
    """Check if date is Friday"""
    return date.weekday() == 4

def create_feature_sequence(station_name, direction, target_datetime, lookback_hours=24):
    """
    Create feature sequence exactly matching training.py format
    
    Features: hour, weekday, month, is_weekend, is_holiday, is_special_event,
              is_christmas_season, is_payday, is_friday, is_rush_hour, congestion
    """
    sequence = []
    
    for h in range(lookback_hours, 0, -1):  # 24 hours back
        past_time = target_datetime - timedelta(hours=h)
        
        # Use rule-based prediction for historical congestion
        # In production, you'd query actual historical data
        congestion = get_historical_congestion(station_name, direction, past_time)
        
        # Feature extraction
        features = [
            past_time.hour,                                    # hour
            past_time.weekday(),                               # weekday
            past_time.month,                                   # month
            1 if past_time.weekday() >= 5 else 0,             # is_weekend
            0,                                                 # is_holiday (TODO: add holiday calendar)
            0,                                                 # is_special_event
            1 if is_christmas_season(past_time) else 0,       # is_christmas_season
            1 if is_payday(past_time) else 0,                 # is_payday
            1 if is_friday(past_time) else 0,                 # is_friday
            1 if (7 <= past_time.hour <= 9) or (17 <= past_time.hour <= 19) else 0,  # is_rush_hour
            congestion                                         # previous congestion
        ]
        sequence.append(features)
    
    return np.array(sequence, dtype=np.float32)

def get_historical_congestion(station_name, direction, timestamp):
    """
    Generate realistic historical congestion for feature sequence
    This simulates what would come from your database
    """
    hour = timestamp.hour
    weekday = timestamp.weekday()
    
    # Base pattern by hour
    if 7 <= hour <= 9:  # Morning rush
        base = 60 + (hour - 7) * 10
    elif 17 <= hour <= 20:  # Evening rush
        base = 65 + (hour - 17) * 5
    elif 10 <= hour <= 16:  # Mid-day
        base = 40
    elif 21 <= hour <= 22:  # Late evening
        base = 20
    elif 5 <= hour <= 6:  # Early morning
        base = 15
    else:
        base = 10
    
    # Weekend reduction
    if weekday >= 5:
        base *= 0.6
    
    # Station-specific adjustments
    station_factors = {
        "North Ave": 1.2, "Cubao": 1.3, "Ayala Ave": 1.2, "Taft": 1.1,
        "Ortigas": 1.1, "Shaw Blvd": 1.0, "Guadalupe": 0.9, "Magallanes": 0.8
    }
    factor = station_factors.get(station_name, 1.0)
    base *= factor
    
    # Direction adjustment
    if direction == 'Southbound' and (7 <= hour <= 9):
        base *= 1.4  # Morning rush southbound heavier
    elif direction == 'Northbound' and (17 <= hour <= 20):
        base *= 1.4  # Evening rush northbound heavier
    
    return min(95, max(5, base))

def load_directional_models():
    """Load all directional models and scalers"""
    models = {}
    feature_scalers = {}
    target_scalers = {}
    
    print("="*70)
    print("LOADING DIRECTIONAL MODELS")
    print("="*70)
    
    for station in STATIONS:
        for direction in ['Northbound', 'Southbound']:
            model_key = f"{station}_{direction}"
            model_path = f'{MODELS_PATH}/{model_key}_lstm_enhanced.keras'
            feature_scaler_path = f'{MODELS_PATH}/{model_key}_feature_scaler.pkl'
            target_scaler_path = f'{MODELS_PATH}/{model_key}_target_scaler.pkl'
            
            if all(os.path.exists(p) for p in [model_path, feature_scaler_path, target_scaler_path]):
                try:
                    models[model_key] = tf.keras.models.load_model(model_path, compile=False)
                    with open(feature_scaler_path, 'rb') as f:
                        feature_scalers[model_key] = pickle.load(f)
                    with open(target_scaler_path, 'rb') as f:
                        target_scalers[model_key] = pickle.load(f)
                except Exception as e:
                    print(f"❌ Failed: {model_key} - {e}")
            else:
                print(f"⚠️ Missing files: {model_key}")
    
    print(f"\n✅ Loaded {len(models)} directional models")
    return models, feature_scalers, target_scalers

def predict_directional(station_name, direction, target_datetime, models, feature_scalers, target_scalers):
    """Make prediction using loaded directional model"""
    model_key = f"{station_name}_{direction}"
    
    if model_key not in models:
        print(f"⚠️ Model not found: {model_key}, using fallback")
        return None
    
    try:
        # Create feature sequence
        sequence = create_feature_sequence(station_name, direction, target_datetime)
        
        if sequence.shape[0] != 24:
            print(f"⚠️ Invalid sequence shape: {sequence.shape}")
            return None
        
        # Scale features
        feature_scaler = feature_scalers[model_key]
        scaled_sequence = feature_scaler.transform(sequence)
        input_sequence = scaled_sequence.reshape(1, 24, len(FEATURE_COLS))
        
        # Predict
        model = models[model_key]
        pred_scaled = model.predict(input_sequence, verbose=0)
        
        # Inverse transform
        target_scaler = target_scalers[model_key]
        pred = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
        pred = min(100, max(0, pred))
        
        return pred
    except Exception as e:
        print(f"❌ Prediction error for {model_key}: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_rush_hour_patterns(models, feature_scalers, target_scalers):
    """Test predictions during different times of day"""
    print("\n" + "="*70)
    print("TESTING RUSH HOUR PATTERNS")
    print("="*70)
    
    test_times = [
        (6, "Early Morning"),
        (8, "Morning Rush"),
        (12, "Noon"),
        (18, "Evening Rush"),
        (22, "Late Evening")
    ]
    
    results = {}
    
    for station in ["North Ave", "Cubao", "Ayala Ave", "Taft"]:
        print(f"\n📍 {station}:")
        print("-" * 50)
        
        for hour, label in test_times:
            test_time = datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
            
            north_pred = predict_directional(station, 'Northbound', test_time, models, feature_scalers, target_scalers)
            south_pred = predict_directional(station, 'Southbound', test_time, models, feature_scalers, target_scalers)
            
            if north_pred and south_pred:
                print(f"  {label:15s} (H{hour:02d}): NB={north_pred:5.1f}% | SB={south_pred:5.1f}% | Diff={abs(north_pred - south_pred):.1f}%")
                
                # Store results
                key = f"{station}_{label}"
                results[key] = {'north': north_pred, 'south': south_pred}
    
    return results

def test_directional_consistency(models, feature_scalers, target_scalers):
    """Test consistency: Southbound at North station vs Northbound at South station"""
    print("\n" + "="*70)
    print("TESTING DIRECTIONAL CONSISTENCY")
    print("="*70)
    print("Morning rush: Southbound should be high at North stations")
    print("Evening rush: Northbound should be high at South stations\n")
    
    morning_rush = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    evening_rush = datetime.now().replace(hour=18, minute=0, second=0, microsecond=0)
    
    # Morning rush check
    print("📍 MORNING RUSH (8 AM):")
    north_high = []
    for station in ["North Ave", "Quezon Ave", "Kamunting", "Cubao"]:
        if station in STATIONS:
            pred = predict_directional(station, 'Southbound', morning_rush, models, feature_scalers, target_scalers)
            if pred:
                north_high.append((station, pred))
                print(f"   {station} Southbound: {pred:.1f}%")
    
    if north_high:
        avg = sum(p for _, p in north_high) / len(north_high)
        print(f"   Average Southbound at North stations: {avg:.1f}%")
    
    # Evening rush check
    print("\n📍 EVENING RUSH (6 PM):")
    south_high = []
    for station in ["Ayala Ave", "Magallanes", "Taft"]:
        pred = predict_directional(station, 'Northbound', evening_rush, models, feature_scalers, target_scalers)
        if pred:
            south_high.append((station, pred))
            print(f"   {station} Northbound: {pred:.1f}%")
    
    if south_high:
        avg = sum(p for _, p in south_high) / len(south_high)
        print(f"   Average Northbound at South stations: {avg:.1f}%")
    
    return north_high, south_high

def test_specific_scenario(models, feature_scalers, target_scalers):
    """Test a specific scenario that might have failed before"""
    print("\n" + "="*70)
    print("TESTING SPECIFIC SCENARIO: Taft Southbound on Christmas Eve 2025")
    print("="*70)
    
    # The scenario that gave 76.1% error before
    test_date = datetime(2025, 12, 24, 19, 0, 0)  # Christmas Eve 7 PM
    
    pred = predict_directional('Taft', 'Southbound', test_date, models, feature_scalers, target_scalers)
    
    if pred:
        print(f"\n📊 Taft Southbound on {test_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Predicted congestion: {pred:.1f}%")
        
        # Expected pattern for Christmas Eve
        print(f"\n   Expected pattern for Christmas Eve (7 PM):")
        print(f"   - Usually moderate to high congestion (50-70%)")
        print(f"   - People going home for holidays")
        
        if 40 <= pred <= 80:
            print(f"   ✅ Prediction within expected range")
        else:
            print(f"   ⚠️ Prediction outside expected range")
        
        return pred
    else:
        print("❌ Failed to predict")
        return None

def compare_all_stations_at_time(models, feature_scalers, target_scalers, target_time=None):
    """Compare predictions for all stations at a specific time"""
    if target_time is None:
        target_time = datetime.now()
    
    print("\n" + "="*70)
    print(f"ALL STATIONS AT {target_time.strftime('%Y-%m-%d %H:%M')}")
    print("="*70)
    
    results = []
    for station in STATIONS:
        north = predict_directional(station, 'Northbound', target_time, models, feature_scalers, target_scalers)
        south = predict_directional(station, 'Southbound', target_time, models, feature_scalers, target_scalers)
        
        if north and south:
            results.append({
                'station': station,
                'northbound': north,
                'southbound': south,
                'max': max(north, south),
                'diff': abs(north - south)
            })
            
            # Visual bar
            north_bar = '█' * int(north/4)
            south_bar = '█' * int(south/4)
            print(f"{station:15s} | NB: {north:5.1f}% {north_bar:25s} | SB: {south:5.1f}% {south_bar:25s}")
    
    # Summary
    if results:
        avg_north = sum(r['northbound'] for r in results) / len(results)
        avg_south = sum(r['southbound'] for r in results) / len(results)
        avg_diff = sum(r['diff'] for r in results) / len(results)
        
        print(f"\n{'='*70}")
        print(f"📊 AVERAGES: Northbound: {avg_north:.1f}% | Southbound: {avg_south:.1f}%")
        print(f"📊 Average Directional Difference: {avg_diff:.1f}%")
        
        # Identify most congested
        max_station = max(results, key=lambda x: x['max'])
        print(f"\n⚠️ MOST CONGESTED: {max_station['station']} at {max_station['max']:.1f}%")
        
        # Identify largest directional imbalance
        max_diff = max(results, key=lambda x: x['diff'])
        print(f"🔄 LARGEST IMBALANCE: {max_diff['station']} (NB: {max_diff['northbound']:.0f}% vs SB: {max_diff['southbound']:.0f}%)")
    
    return results

if __name__ == '__main__':
    print("\n" + "="*70)
    print("DIRECTIONAL MODEL COMPREHENSIVE TEST")
    print("="*70)
    
    # Load models
    models, feature_scalers, target_scalers = load_directional_models()
    
    if len(models) == 0:
        print("\n❌ No models loaded! Run training.py first.")
        sys.exit(1)
    
    print(f"\n📊 Models loaded: {len(models)}/26")
    
    # Test 1: Current time all stations
    compare_all_stations_at_time(models, feature_scalers, target_scalers)
    
    # Test 2: Rush hour patterns
    test_rush_hour_patterns(models, feature_scalers, target_scalers)
    
    # Test 3: Directional consistency
    test_directional_consistency(models, feature_scalers, target_scalers)
    
    # Test 4: Specific scenario (Christmas Eve 2025 - your test case)
    christmas_pred = test_specific_scenario(models, feature_scalers, target_scalers)
    
    # Test 5: Test specific problematic station
    print("\n" + "="*70)
    print("TESTING SPECIFIC STATION: Taft Southbound (Hourly)")
    print("="*70)
    
    test_hours = [6, 8, 12, 15, 18, 20, 22]
    print(f"\n{'Hour':<10} {'Prediction':<15} {'Status':<15}")
    print("-" * 40)
    
    for hour in test_hours:
        test_time = datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
        pred = predict_directional('Taft', 'Southbound', test_time, models, feature_scalers, target_scalers)
        
        if pred:
            if pred > 70:
                status = "🔴 CRITICAL"
            elif pred > 50:
                status = "🟠 BUSY"
            elif pred > 30:
                status = "🟡 MODERATE"
            else:
                status = "🟢 LIGHT"
            
            print(f"{hour:02d}:00     {pred:>5.1f}%        {status}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
    if christmas_pred:
        print(f"\n🎯 For your original 2025 test case (Christmas Eve 7 PM):")
        print(f"   Old model error: 76.1%")
        print(f"   New model prediction: {christmas_pred:.1f}%")
        print(f"\n   Run your diagnosis script with actual 2025 data to compare!")