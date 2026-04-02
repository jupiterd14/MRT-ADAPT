"""
evaluate_model.py - FIXED VERSION
Matches the training data format!
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, f1_score
import pickle
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Station mapping
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

STATION_CAPACITY = {
    "North Ave": 12000, "Quezon Ave": 9000, "Kamuning": 7500, "Cubao": 15000,
    "Santolan": 8000, "Ortigas": 9500, "Shaw Blvd": 11000, "Boni Ave": 8500,
    "Guadalupe": 10000, "Buendia": 9000, "Ayala Ave": 14000, "Magallanes": 9000, 
    "Taft": 16000
}

def calculate_metrics(y_true, y_pred):
    """Calculate all evaluation metrics"""
    mask = ~(np.isnan(y_true) | np.isinf(y_true) | np.isnan(y_pred) | np.isinf(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return None
    
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    
    # R² Score
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # F1 Score for congestion levels
    def categorize(value):
        pct = (value / STATION_CAPACITY.get("North Ave", 10000)) * 100  # Approximate
        if pct > 80: return 4
        elif pct > 60: return 3
        elif pct > 30: return 2
        return 1
    
    y_true_cat = [categorize(v) for v in y_true_clean]
    y_pred_cat = [categorize(v) for v in y_pred_clean]
    
    f1 = f1_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'f1_score': f1,
        'samples': len(y_true_clean)
    }


def create_station_time_series(df, station_name):
    """MIMIC THE TRAINING SCRIPT - Create time series same as train_model.py"""
    
    # Filter for the station
    station_data = df[df['entry_station'] == station_name].copy()
    
    if len(station_data) == 0:
        return None
    
    # Create datetime column
    station_data['datetime'] = pd.to_datetime(
        station_data['Date'] + ' ' + station_data['Time'], 
        errors='coerce'
    )
    
    # Group by date and hour to get total passengers per hour
    chrono_data = station_data.groupby([
        station_data['datetime'].dt.date.rename('date_idx'), 
        station_data['datetime'].dt.hour.rename('hour_idx')
    ])['total_volume'].sum().reset_index()
    
    chrono_data.columns = ['date', 'hour', 'passengers']
    
    # Sort chronologically
    chrono_data = chrono_data.sort_values(['date', 'hour'])
    
    # Get the time series
    full_series = chrono_data['passengers'].values
    
    return full_series


def evaluate_lstm_model():
    """Evaluate LSTM model performance with sliding window"""
    print("\n" + "="*70)
    print("🔬 EVALUATING LSTM MODEL PERFORMANCE")
    print("="*70)
    
    # Load data
    data_folder = 'data_new_2025'
    all_data = []
    
    for year in ['2024', '2025']:
        file_path = os.path.join(data_folder, f'{year}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"✅ Loaded {year}.csv with {len(df)} records")
    
    if not all_data:
        print("❌ No data files found!")
        return
    
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Map station IDs
    STATION_ID_MAP = {
        1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
        5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
        9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
    }
    
    df_combined['entry_station'] = df_combined['StationEntry'].map(STATION_ID_MAP)
    df_combined['total_volume'] = df_combined['TotalPassenger']
    
    print("\n🤖 Loading and evaluating models...")
    station_results = []
    
    for station in STATIONS:
        m_path = f'models/{station}_lstm.keras'
        s_path = f'models/{station}_scaler.pkl'
        
        if not (os.path.exists(m_path) and os.path.exists(s_path)):
            print(f"⚠️ Missing files for {station}")
            continue
        
        try:
            # Load model and scaler
            model = tf.keras.models.load_model(m_path, compile=False)
            with open(s_path, 'rb') as f:
                scaler = pickle.load(f)
            
            print(f"\n📈 Evaluating {station}...")
            
            # Create time series SAME WAY as training
            time_series = create_station_time_series(df_combined, station)
            
            if time_series is None or len(time_series) < 100:
                print(f"   ⚠️ Insufficient data for {station}")
                continue
            
            print(f"   Time series length: {len(time_series)} hours")
            print(f"   Value range: {time_series.min():.0f} - {time_series.max():.0f}")
            
            # Use last 30% for testing
            test_size = int(len(time_series) * 0.3)
            test_series = time_series[-test_size:]
            
            # Create predictions using sliding window
            seq_length = 24
            y_true_list = []
            y_pred_list = []
            
            for i in range(seq_length, len(test_series)):
                # Input sequence (last 24 hours)
                input_seq = test_series[i-seq_length:i]
                
                # Scale
                input_scaled = scaler.transform(input_seq.reshape(-1, 1))
                X_input = input_scaled.reshape(1, seq_length, 1)
                
                # Predict
                pred_scaled = model.predict(X_input, verbose=0)
                pred_value = scaler.inverse_transform(pred_scaled)[0][0]
                
                y_true_list.append(test_series[i])
                y_pred_list.append(pred_value)
            
            # Calculate metrics
            y_true_arr = np.array(y_true_list)
            y_pred_arr = np.array(y_pred_list)
            
            # Show sample predictions
            print(f"   Sample predictions (last 5):")
            for j in range(-5, 0):
                if abs(j) <= len(y_true_arr):
                    print(f"      Actual: {y_true_arr[j]:.0f}, Pred: {y_pred_arr[j]:.0f}")
            
            metrics = calculate_metrics(y_true_arr, y_pred_arr)
            
            if metrics:
                station_results.append({'station': station, **metrics})
                print(f"   ✅ MAE: {metrics['mae']:.0f} | R²: {metrics['r2']:.4f} | F1: {metrics['f1_score']:.4f}")
            
            # Clear memory
            tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("📊 EVALUATION SUMMARY")
    print("="*70)
    
    if station_results:
        results_df = pd.DataFrame(station_results)
        
        print(f"\n📈 Overall Metrics:")
        print(f"   Average R²: {results_df['r2'].mean():.4f}")
        print(f"   Average MAE: {results_df['mae'].mean():.0f} passengers")
        print(f"   Average F1: {results_df['f1_score'].mean():.4f}")
        
        # Best and worst
        best = results_df.loc[results_df['r2'].idxmax()]
        worst = results_df.loc[results_df['r2'].idxmin()]
        print(f"\n🏆 Best: {best['station']} (R²={best['r2']:.4f}, MAE={best['mae']:.0f})")
        print(f"⚠️ Worst: {worst['station']} (R²={worst['r2']:.4f}, MAE={worst['mae']:.0f})")
        
        # Save results
        results_df.to_csv('evaluation_results.csv', index=False)
        print(f"\n✅ Results saved to evaluation_results.csv")
        
        # Readiness
        avg_r2 = results_df['r2'].mean()
        print("\n" + "="*70)
        print("🚦 DEPLOYMENT READINESS")
        print("="*70)
        
        if avg_r2 > 0.7:
            print("✅ READY FOR DEPLOYMENT! Models are excellent.")
        elif avg_r2 > 0.5:
            print("⚠️ ACCEPTABLE - Deploy with monitoring.")
        elif avg_r2 > 0.3:
            print("⚠️ NEEDS IMPROVEMENT - Consider retraining with more data.")
        else:
            print("❌ NOT READY - Models need significant improvement.")
        
        # Save readiness report
        readiness_report = {
            'evaluation_date': datetime.now().isoformat(),
            'avg_r2': float(avg_r2),
            'avg_mae': float(results_df['mae'].mean()),
            'avg_f1': float(results_df['f1_score'].mean()),
            'best_station': best['station'],
            'worst_station': worst['station']
        }
        
        with open('deployment_readiness.json', 'w') as f:
            json.dump(readiness_report, f, indent=2, default=str)
        
        print(f"\n💾 Readiness report saved to deployment_readiness.json")
        
    else:
        print("❌ No successful evaluations!")


if __name__ == '__main__':
    evaluate_lstm_model()