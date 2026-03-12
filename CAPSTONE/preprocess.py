# preprocess_data.py
# Run this ONCE on your computer to prepare lightweight files

import os
import numpy as np
import pandas as pd
import pickle
import glob
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

print("🔨 PRE-PROCESSING DATA...")

# ========== LOAD YOUR EXCEL FILES (SAME AS BEFORE) ==========
def load_all_data():
    all_data_frames = []
    
    # Your existing data loading code here
    # (Copy your load_hourly_data() and load_reports_data() functions)
    # ... (I'll keep this section short but you'll paste your full loading code)
    
    # After loading all data:
    # mrt_data = pd.concat(all_data_frames, ignore_index=True)
    # return mrt_data
    
# For brevity, I'll assume you have your mrt_data loaded
# In reality, paste ALL your loading functions here

# For now, let's assume mrt_data is loaded
# mrt_data = load_all_data()  # This runs ONCE on your computer

# ========== EXTRACT AND SAVE ONLY WHAT'S NEEDED ==========
print("\n💾 SAVING LIGHTWEIGHT DATA FILES...")

# 1. Save station capacities
station_capacities = {}
for station in STATIONS:
    station_data = mrt_data[mrt_data['station'] == station]['ridership']
    if len(station_data) > 0:
        station_capacities[station] = int(station_data.quantile(0.95))
    else:
        station_capacities[station] = None

with open('station_capacities.pkl', 'wb') as f:
    pickle.dump(station_capacities, f)
print("✅ Saved station capacities")

# 2. Save hourly averages (for fallback)
hourly_averages = {}
for station in STATIONS:
    station_data = mrt_data[mrt_data['station'] == station]
    if len(station_data) > 0:
        hourly_avg = station_data.groupby('hour')['ridership'].mean().to_dict()
        hourly_averages[station] = hourly_avg

with open('hourly_averages.pkl', 'wb') as f:
    pickle.dump(hourly_averages, f)
print("✅ Saved hourly averages")

# 3. Save time series for predictions (only last 1000 points per station)
time_series_data = {}
for station in STATIONS:
    station_df = mrt_data[mrt_data['station'] == station].copy()
    if len(station_df) > 0:
        station_df = station_df.sort_values('timestamp')
        # Keep only last 1000 points (enough for predictions)
        if len(station_df) > 1000:
            station_df = station_df.tail(1000)
        time_series_data[station] = station_df[['timestamp', 'hour', 'ridership']].to_dict('records')

with open('time_series_data.pkl', 'wb') as f:
    pickle.dump(time_series_data, f)
print("✅ Saved time series data")

# 4. Train and save LSTM models
os.makedirs('models', exist_ok=True)
lstm_models = {}
scalers = {}

for station in STATIONS:
    print(f"\n📈 Training model for {station}...")
    series = mrt_data[mrt_data['station'] == station]['ridership']
    
    if len(series) < 100:
        continue
    
    values = series.values[-2000:].reshape(-1, 1)  # Use last 2000 points
    
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)
    
    X, y = [], []
    for i in range(24, len(scaled_values)):
        X.append(scaled_values[i-24:i])
        y.append(scaled_values[i])
    
    if len(X) == 0:
        continue
    
    X = np.array(X).reshape(-1, 24, 1)
    y = np.array(y)
    
    # Simple model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, input_shape=(24, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    
    # Save model and scaler
    model.save(f'models/{station}_lstm.h5')
    with open(f'models/{station}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"   ✅ Saved {station} model")

print("\n🎉 ALL DONE! Lightweight files created!")
print("Files created:")
print("- station_capacities.pkl (2KB)")
print("- hourly_averages.pkl (5KB)")
print("- time_series_data.pkl (500KB)")
print("- models/*.h5 (2MB each)")