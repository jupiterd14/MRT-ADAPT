
# train_models_with_real_data.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

STATION_ID_MAP = {
    1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
    5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
    9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
}
def load_real_data():
    """Load ONLY 2023, 2024, and 2025 data"""
    data_folder = 'data_new_2025'
    # We explicitly define the years we want here
    target_years = ['2023', '2024', '2025'] 
    all_data = []
    
    if not os.path.exists(data_folder):
        print(f"❌ Folder '{data_folder}' not found!")
        return None
    
    print(f"🎯 Target Years: {target_years}")
    
    for year in target_years:
        file_path = os.path.join(data_folder, f'{year}.csv')
        if os.path.exists(file_path):
            print(f"📖 Reading {year}.csv...")
            df = pd.read_csv(file_path)
            # Basic cleaning: remove rows with missing Date/Time immediately to save RAM
            df = df.dropna(subset=['Date', 'Time'])
            all_data.append(df)
        else:
            print(f"⚠️ Warning: {year}.csv not found in {data_folder}")
    
    if not all_data:
        print("❌ No target CSV files found!")
        return None
    
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"✅ Successfully loaded {len(df_combined)} records for training.")
    return df_combined

def create_station_time_series(df, station_name):
    """Create a REAL chronological time series (2023-2025)"""
    # 1. Filter for the specific station
    station_data = df[df['StationEntry'].map(STATION_ID_MAP) == station_name].copy()
    
    if len(station_data) == 0:
        return None

    # 2. Create a proper Datetime column
    station_data['datetime'] = pd.to_datetime(
        station_data['Date'] + ' ' + station_data['Time'], 
        errors='coerce'
    )
    
    # 3. Group by Hour and Date to get the actual sequence of history
    # This keeps the 'natural' order of 2023 -> 2024 -> 2025
    # 3. Group by Hour and Date using specific names to avoid the crash
    chrono_data = station_data.groupby([
        station_data['datetime'].dt.date.rename('date_idx'), 
        station_data['datetime'].dt.hour.rename('hour_idx')
    ])['TotalPassenger'].sum().reset_index()
    
    # Rename them to something clear for the next step
    chrono_data.columns = ['date', 'hour', 'passengers']
    
    # 4. Sort chronologically (CRITICAL for LSTM)
    chrono_data = chrono_data.sort_values(['date', 'hour'])
    
    # 5. Convert to a continuous array
    full_series = chrono_data['passengers'].values
    
    # Validation: If we have gaps (e.g., midnight), we should have at least some data
    if len(full_series) < 100:
        print(f"⚠️ {station_name} has very few chronological data points.")
        return None

    return full_series

def create_sequences(data, seq_length=24):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_station_model(station_name, time_series_data, seq_length=24):
    """Train LSTM model for a single station using real data"""
    print(f"\n{'='*50}")
    print(f"Training {station_name}...")
    
    if time_series_data is None or len(time_series_data) < seq_length + 10:
        print(f"⚠️ Insufficient data for {station_name}, skipping...")
        return None
    
    print(f"   Using {len(time_series_data)} hours of real data")
    
    # Normalize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(time_series_data.reshape(-1, 1))
    
    # Create sequences
    X, y = create_sequences(data_scaled.flatten(), seq_length)
    
    # Reshape for LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Save model and scaler
    model.save(f'models/{station_name}_lstm.keras')
    with open(f'models/{station_name}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"   ✅ Model saved! Final loss: {history.history['loss'][-1]:.4f}")
    return history.history['loss'][-1]

def main():
    print("="*70)
    print("🚇 TRAINING LSTM MODELS WITH REAL DATA")
    print("="*70)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load real data
    print("\n📊 Loading real data from data_new_2025...")
    df = load_real_data()
    
    if df is None:
        print("❌ No data found. Please run process_data.py first.")
        return
    
    # Train models for each station
    print("\n🤖 Training models using REAL historical data...")
    print("="*70)
    
    results = {}
    for station in STATIONS:
        time_series = create_station_time_series(df, station)
        loss = train_station_model(station, time_series)
        if loss:
            results[station] = loss
    
    # Summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    print(f"✅ Successfully trained {len(results)}/{len(STATIONS)} models")
    
    if results:
        print("\n📈 Final Loss per station:")
        for station, loss in results.items():
            print(f"   {station}: {loss:.4f}")
    
    print("\n🎉 Training complete! Your models are now trained on REAL data from data_new_2025/")

if __name__ == '__main__':
    main()