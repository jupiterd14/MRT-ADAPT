import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Station mapping
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

STATION_ID_MAP = {
    1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
    5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
    9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
}

def calculate_metrics(y_true, y_pred):
    """Calculate all evaluation metrics"""
    # Remove any NaN or infinite values
    mask = ~(np.isnan(y_true) | np.isinf(y_true) | np.isnan(y_pred) | np.isinf(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return None
    
    # MAE - Mean Absolute Error
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    
    # RMSE - Root Mean Square Error
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    
    # MAPE - Mean Absolute Percentage Error (avoid division by zero)
    non_zero_mask = y_true_clean != 0
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs((y_true_clean[non_zero_mask] - y_pred_clean[non_zero_mask]) / y_true_clean[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    # R² Score
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # For confusion matrix and F1, we need to categorize into congestion levels
    def categorize_congestion(value):
        if value > 80:
            return 4  # Critical
        elif value > 60:
            return 3  # Congested
        elif value > 30:
            return 2  # Moderate
        else:
            return 1  # Light
    
    y_true_cat = [categorize_congestion(v) for v in y_true_clean]
    y_pred_cat = [categorize_congestion(v) for v in y_pred_clean]
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=[1, 2, 3, 4])
    
    # F1 Score (weighted average)
    f1 = f1_score(y_true_cat, y_pred_cat, average='weighted')
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'confusion_matrix': cm,
        'f1_score': f1,
        'samples': len(y_true_clean)
    }

def prepare_time_features(df):
    """Prepare time-based features for prediction"""
    # Convert date and time to datetime
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    
    # Extract features
    df['hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour
    df['minute'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    # Weekend flag
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Rush hour flags
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    
    return df

def load_and_prepare_data(years=None):
    """Load data from specified years"""
    data_folder = 'data_new_2025'
    
    if years is None:
        years = ['2024', '2025']  # Default to recent years for evaluation
    
    all_data = []
    
    for year in years:
        file_path = os.path.join(data_folder, f'{year}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['year'] = int(year)
            all_data.append(df)
            print(f"✅ Loaded {year}.csv with {len(df)} records")
    
    if not all_data:
        print("❌ No data files found!")
        return None
    
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Convert station IDs to names
    df_combined['entry_station'] = df_combined['StationEntry'].map(STATION_ID_MAP)
    df_combined['exit_station'] = df_combined['StationExit'].map(STATION_ID_MAP)
    
    # Calculate total passengers per station (for congestion)
    # We'll use entry + exit as total passenger volume
    df_combined['total_volume'] = df_combined['TotalPassenger']
    
    # Add time features
    df_combined = prepare_time_features(df_combined)
    
    return df_combined

def evaluate_lstm_model():
    """Evaluate LSTM model performance"""
    print("\n" + "="*70)
    print("🔬 EVALUATING LSTM MODEL PERFORMANCE")
    print("="*70)
    
    # Load data
    print("\n📊 Loading data for evaluation...")
    df = load_and_prepare_data(years=['2024', '2025'])
    
    if df is None:
        return
    
    # Load LSTM models
    print("\n🤖 Loading LSTM models...")
    lstm_models = {}
    scalers = {}
    models_loaded = 0
    
    for station in STATIONS:
        m_path = f'models/{station}_lstm.h5'
        s_path = f'models/{station}_scaler.pkl'
        
        if os.path.exists(m_path) and os.path.exists(s_path):
            try:
                lstm_models[station] = tf.keras.models.load_model(m_path, compile=False)
                with open(s_path, 'rb') as f:
                    scalers[station] = pickle.load(f)
                models_loaded += 1
                print(f"  ✅ Loaded {station} model")
            except Exception as e:
                print(f"  ❌ Error loading {station}: {e}")
    
    print(f"\n📊 Loaded {models_loaded}/{len(STATIONS)} models")
    
    if models_loaded == 0:
        print("❌ No models loaded. Train models first!")
        return
    
    # Evaluate each station
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS BY STATION")
    print("="*70)
    
    all_results = {}
    station_results = []
    
    for station in STATIONS:
        if station not in lstm_models:
            print(f"\n❌ Skipping {station}: Model not available")
            continue
        
        print(f"\n📈 Evaluating {station}...")
        
        # Get data for this station
        station_entry_data = df[df['entry_station'] == station].copy()
        station_exit_data = df[df['exit_station'] == station].copy()
        
        # Combine entry and exit for total volume
        station_data = pd.concat([
            station_entry_data[['total_volume', 'hour', 'minute', 'day_of_week', 'month', 'is_weekend', 'is_morning_rush', 'is_evening_rush']],
            station_exit_data[['total_volume', 'hour', 'minute', 'day_of_week', 'month', 'is_weekend', 'is_morning_rush', 'is_evening_rush']]
        ])
        
        if len(station_data) == 0:
            print(f"  ⚠️ No data for {station}")
            continue
        
        # Group by hour to get average volume per hour
        hourly_avg = station_data.groupby('hour')['total_volume'].mean().reset_index()
        
        # For LSTM prediction, we need sequential data
        # Sort by hour to create a time series
        hourly_avg = hourly_avg.sort_values('hour')
        
        if len(hourly_avg) < 24:
            print(f"  ⚠️ Insufficient data for {station}")
            continue
        
        # Prepare data for LSTM
        scaler = scalers[station]
        model = lstm_models[station]
        
        # Use last 24 hours to predict next
        X_test = hourly_avg['total_volume'].values[-24:].reshape(1, -1, 1)
        X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(1, 24, 1)
        
        # Predict
        y_pred_scaled = model.predict(X_test_scaled, verbose=0)
        y_pred = scaler.inverse_transform(y_pred_scaled)[0][0]
        
        # True value (average of next hour)
        y_true = hourly_avg['total_volume'].values[-1] if len(hourly_avg) > 0 else y_pred
        
        # Calculate metrics
        y_true_list = hourly_avg['total_volume'].values
        y_pred_list = []
        
        # Generate predictions for each hour
        for i in range(24, len(hourly_avg)):
            input_seq = hourly_avg['total_volume'].values[i-24:i].reshape(1, -1, 1)
            input_scaled = scaler.transform(input_seq.reshape(-1, 1)).reshape(1, 24, 1)
            pred = model.predict(input_scaled, verbose=0)
            pred_inv = scaler.inverse_transform(pred)[0][0]
            y_pred_list.append(pred_inv)
        
        # Align for comparison
        y_true_align = hourly_avg['total_volume'].values[24:]
        
        if len(y_true_align) > 0 and len(y_pred_list) > 0:
            metrics = calculate_metrics(y_true_align, np.array(y_pred_list))
            
            if metrics:
                station_results.append({
                    'station': station,
                    **metrics
                })
                
                print(f"  ✅ {station} Results:")
                print(f"     MAE: {metrics['mae']:.2f}")
                print(f"     RMSE: {metrics['rmse']:.2f}")
                print(f"     MAPE: {metrics['mape']:.2f}%")
                print(f"     R²: {metrics['r2']:.4f}")
                print(f"     F1 Score: {metrics['f1_score']:.4f}")
                print(f"     Samples: {metrics['samples']}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 OVERALL PERFORMANCE SUMMARY")
    print("="*70)
    
    if station_results:
        avg_mae = np.mean([r['mae'] for r in station_results])
        avg_rmse = np.mean([r['rmse'] for r in station_results])
        avg_mape = np.mean([r['mape'] for r in station_results])
        avg_r2 = np.mean([r['r2'] for r in station_results])
        avg_f1 = np.mean([r['f1_score'] for r in station_results])
        
        print(f"\n📈 Average Metrics Across All Stations:")
        print(f"   MAE: {avg_mae:.2f}")
        print(f"   RMSE: {avg_rmse:.2f}")
        print(f"   MAPE: {avg_mape:.2f}%")
        print(f"   R² Score: {avg_r2:.4f}")
        print(f"   F1 Score: {avg_f1:.4f}")
        
        # Best performing station
        best_station = max(station_results, key=lambda x: x['r2'])
        print(f"\n🏆 Best Performing Station: {best_station['station']}")
        print(f"   R²: {best_station['r2']:.4f}")
        print(f"   F1: {best_station['f1_score']:.4f}")
        
        # Save results
        results_df = pd.DataFrame(station_results)
        results_df.to_csv('evaluation_results.csv', index=False)
        print(f"\n✅ Results saved to evaluation_results.csv")
        
        # Save confusion matrices
        print("\n📊 Confusion Matrices:")
        for r in station_results:
            print(f"\n  {r['station']}:")
            print(f"     Confusion Matrix:\n{r['confusion_matrix']}")
    
    return station_results

if __name__ == '__main__':
    evaluate_lstm_model()