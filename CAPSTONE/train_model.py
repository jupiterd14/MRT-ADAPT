# train_model.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

print("="*70)
print("🚇 TRAINING MRT-3 MODELS WITH REAL PATTERNS")
print("="*70)

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

# Load data sa pkl
if os.path.exists('mrt_cleaned_master.pkl'):
    df = pd.read_pickle('mrt_cleaned_master.pkl')
    print(f"✅ Loaded {len(df):,} rows from mrt_cleaned_master.pkl")
    
elif os.path.exists('mrt_cleaned_master.csv'):
    df = pd.read_csv('mrt_cleaned_master.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ Loaded {len(df):,} rows from mrt_cleaned_master.csv")
else:
    print("No data found! Run prepare.py first.")
    exit()

# ========== CALCULATE STATION CAPACITIES ==========
print("\n📊 Calculating station capacities...")
STATION_CAPACITIES = {}
for station in STATIONS:
    station_data = df[df['station'] == station]['ridership']
    if len(station_data) > 0:
        # Use 95th percentile for capacity
        STATION_CAPACITIES[station] = int(station_data.quantile(0.95))
        print(f"   ✅ {station}: {STATION_CAPACITIES[station]:,}")
    else:
        STATION_CAPACITIES[station] = 8000
        print(f"   ⚠️ {station}: No data, using 8000")

# ========== CALCULATE HOURLY AVERAGES ==========
print("\n📈 Calculating hourly averages...")
hourly_averages = {}
for station in STATIONS:
    station_data = df[df['station'] == station]
    if len(station_data) > 0:
        hourly_avg = station_data.groupby('hour')['ridership'].mean().to_dict()
        # Fill missing hours
        for hour in range(24):
            if hour not in hourly_avg:
                # Find nearest hour with data
                nearby = [h for h in range(24) if h in hourly_avg]
                if nearby:
                    nearest = min(nearby, key=lambda x: abs(x - hour))
                    hourly_avg[hour] = hourly_avg[nearest]
                else:
                    hourly_avg[hour] = 2000
        hourly_averages[station] = hourly_avg
        print(f"   ✅ {station}")

# ========== CALCULATE REAL CAPACITY PATTERNS ==========
print("\n🎯 Calculating REAL capacity patterns...")

weekday_capacity_patterns = {}
weekend_capacity_patterns = {}

for station in STATIONS:
    station_df = df[df['station'] == station].copy()
    
    if len(station_df) > 0:
        # Add day of week
        station_df['day_of_week'] = pd.to_datetime(station_df['timestamp']).dt.dayofweek
        station_df['is_weekend'] = station_df['day_of_week'] >= 5
        
        # Weekday patterns (Mon-Fri)
        weekday_pattern = {}
        weekday_data = station_df[~station_df['is_weekend']]
        for hour in range(24):
            hour_data = weekday_data[weekday_data['hour'] == hour]['ridership']
            if len(hour_data) > 10:
                weekday_pattern[hour] = int(hour_data.quantile(0.95))
            else:
                # Not enough data, use overall percentile
                weekday_pattern[hour] = int(station_df['ridership'].quantile(0.95))
        
        # Weekend patterns (Sat-Sun)
        weekend_pattern = {}
        weekend_data = station_df[station_df['is_weekend']]
        for hour in range(24):
            hour_data = weekend_data[weekend_data['hour'] == hour]['ridership']
            if len(hour_data) > 5:
                weekend_pattern[hour] = int(hour_data.quantile(0.95))
            else:
                # Not enough weekend data, use 70% of weekday
                weekend_pattern[hour] = int(weekday_pattern.get(hour, 5000) * 0.7)
        
        weekday_capacity_patterns[station] = weekday_pattern
        weekend_capacity_patterns[station] = weekend_pattern
        
        print(f"   ✅ {station}: Weekday peak={max(weekday_pattern.values()):,} at {max(weekday_pattern, key=weekday_pattern.get)}:00")
    else:
        # Fallback
        weekday_capacity_patterns[station] = {hour: 8000 for hour in range(24)}
        weekend_capacity_patterns[station] = {hour: 5000 for hour in range(24)}
        print(f"   ⚠️ {station}: No data, using estimates")

# ========== CREATE TIME SERIES LAST 24 ==========
print("\n⏱️ Creating last 24h time series...")
station_time_series_last_24 = {}
for station in STATIONS:
    station_df = df[df['station'] == station].sort_values('timestamp')
    if len(station_df) > 0:
        # Keep last 1000 points
        if len(station_df) > 1000:
            station_df = station_df.tail(1000)
        station_time_series_last_24[station] = station_df[['timestamp', 'hour', 'ridership']]
        print(f"   ✅ {station}: {len(station_time_series_last_24[station])} records")
    else:
        station_time_series_last_24[station] = pd.DataFrame()
        print(f"   ⚠️ {station}: No data")

# ========== TRAIN LSTM MODELS ==========
print("\n🤖 Training LSTM models (this may take a while)...")
os.makedirs('models', exist_ok=True)

lstm_models = {}
scalers = {}
models_loaded = 0

for station in STATIONS:
    print(f"\n   Training {station}...")
    station_df = df[df['station'] == station].sort_values('timestamp')
    
    if len(station_df) < 100:
        print(f"   ⚠️ Not enough data for {station}, skipping")
        continue
    
    # Use last 2000 points for training
    values = station_df['ridership'].values[-2000:].reshape(-1, 1)
    
    if len(values) < 50:
        continue
    
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)
    
    # Create sequences
    X, y = [], []
    for i in range(24, len(scaled_values)):
        X.append(scaled_values[i-24:i])
        y.append(scaled_values[i])
    
    if len(X) < 10:
        continue
    
    X = np.array(X).reshape(-1, 24, 1)
    y = np.array(y)
    
    # Simple LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(24, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    model.fit(X, y, epochs=10, batch_size=32, verbose=0, validation_split=0.1)
    
    # Save
    model.save(f'models/{station}_lstm.h5')
    with open(f'models/{station}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    models_loaded += 1
    print(f"   ✅ Saved {station} model")

print(f"\n✅ Trained {models_loaded} models")

# ========== CREATE SYSTEM CACHE ==========
print("\n💾 Creating system_cache.pkl...")

system_cache = {
    'STATION_CAPACITIES': STATION_CAPACITIES,
    'hourly_averages': hourly_averages,
    'station_time_series_last_24': station_time_series_last_24,
    'weekday_capacity_patterns': weekday_capacity_patterns,
    'weekend_capacity_patterns': weekend_capacity_patterns
}

with open('system_cache.pkl', 'wb') as f:
    pickle.dump(system_cache, f)

print("✅ Created system_cache.pkl with REAL patterns!")

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)
print(f"📊 Stations: {len(STATIONS)}")
print(f"📈 Real patterns: ✅ NOW AVAILABLE")
print(f"🤖 LSTM models: {models_loaded}/{len(STATIONS)}")
print(f"💾 Cache file: system_cache.pkl")
print("="*70)