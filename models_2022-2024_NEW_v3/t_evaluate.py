# evaluate_all_13_complete.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import gc

# ========== DEFINE CUSTOM METRICS ==========
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def mape(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + 0.01))) * 100

# ========== CONFIGURATION ==========
BATCH_SIZE = 256
SEQ_LENGTH = 24
CHUNK_SIZE = 100000

ALL_STATIONS = [
    "North Ave", "Quezon Ave", "Kamuning", "Cubao",
    "Santolan", "Ortigas", "Shaw Blvd", "Boni Ave",
    "Guadalupe", "Buendia", "Ayala Ave", "Magallanes", "Taft"
]

STATION_NUMBERS = {
    "North Ave": 1, "Quezon Ave": 2, "Kamuning": 3, "Cubao": 4,
    "Santolan": 5, "Ortigas": 6, "Shaw Blvd": 7, "Boni Ave": 8,
    "Guadalupe": 9, "Buendia": 10, "Ayala Ave": 11, "Magallanes": 12,
    "Taft": 13
}

data_folder = 'data (2022-2024)'
files = ['2022.csv', '2023.csv', '2024.csv']
MODELS_PATH = 'models_2022-2024'

# ========== FUNCTIONS ==========
def create_sequences(features, target, seq_length=SEQ_LENGTH):
    n_sequences = len(features) - seq_length
    if n_sequences <= 0:
        return np.array([]), np.array([])
    X = np.zeros((n_sequences, seq_length, features.shape[1]), dtype=np.float32)
    y = np.zeros((n_sequences,), dtype=np.float32)
    for i in range(n_sequences):
        X[i] = features[i:i+seq_length]
        y[i] = target[i+seq_length]
    return X, y

print("="*70)
print("EVALUATING ALL 13 STATIONS WITH VALIDATION DATA")
print("="*70)

# ========== LOAD DATA ONCE ==========
print("\nLoading data...")
chunks = []
for file in files:
    file_path = os.path.join(data_folder, file)
    if os.path.exists(file_path):
        print(f"  Reading {file}...")
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
            chunk['datetime'] = pd.to_datetime(chunk['Time'])
            chunk['hour'] = chunk['datetime'].dt.hour
            chunk['weekday'] = chunk['datetime'].dt.weekday
            chunk['month'] = chunk['datetime'].dt.month
            chunk['is_weekend'] = (chunk['datetime'].dt.weekday >= 5).astype(np.int8)
            chunk['is_holiday'] = 0
            chunk['is_special_event'] = 0
            chunk['is_christmas_season'] = 0
            chunk['is_payday'] = 0
            chunk['is_friday'] = (chunk['datetime'].dt.weekday == 4).astype(np.int8)
            chunk['is_rush_hour'] = ((chunk['hour'].between(7, 9)) | (chunk['hour'].between(17, 19))).astype(np.int8)
            
            # Direction
            def infer_dir(row):
                entry = row['StationEntry']
                exit_s = row['StationExit']
                if entry < exit_s:
                    return 1
                elif entry > exit_s:
                    return 0
                else:
                    return 0.5
            chunk['direction_code'] = chunk.apply(infer_dir, axis=1)
            chunks.append(chunk)
            gc.collect()

df = pd.concat(chunks, ignore_index=True)
print(f"Loaded {len(df):,} records")

# Calculate congestion
max_passengers = df['TotalPassenger'].quantile(0.99)
df['congestion'] = (df['TotalPassenger'] / max_passengers * 100).clip(0, 100)

feature_cols = ['hour', 'weekday', 'month', 'is_weekend', 'is_holiday',
                'is_special_event', 'is_christmas_season', 'direction_code', 'congestion']

# ========== EVALUATE EACH STATION ==========
all_results = []

for station in ALL_STATIONS:
    print(f"\n{'='*50}")
    print(f"Evaluating: {station}")
    
    station_num = STATION_NUMBERS[station]
    
    # Filter data
    mask = (df['StationEntry'] == station_num) | (df['StationExit'] == station_num)
    station_df = df[mask].sort_values('datetime').copy()
    
    if len(station_df) < SEQ_LENGTH + 10:
        print(f"  ⚠️ Insufficient data: {len(station_df)} records")
        continue
    
    # Load scalers
    scaler_path = f'{MODELS_PATH}/{station}_feature_scaler.pkl'
    if not os.path.exists(scaler_path):
        print(f"  ❌ Scaler not found")
        continue
    
    with open(scaler_path, 'rb') as f:
        feature_scaler = pickle.load(f)
    with open(f'{MODELS_PATH}/{station}_target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    
    # Prepare features
    feature_data = station_df[feature_cols].values.astype(np.float32)
    features_scaled = feature_scaler.transform(feature_data).astype(np.float32)
    targets_raw = station_df['congestion'].values.reshape(-1, 1).astype(np.float32)
    targets_scaled = target_scaler.transform(targets_raw).flatten()
    
    # Create sequences
    X, y = create_sequences(features_scaled, targets_scaled)
    
    if len(X) == 0:
        print(f"  ⚠️ No sequences created")
        continue
    
    # Load model
    model = tf.keras.models.load_model(
        f'{MODELS_PATH}/{station}_lstm_enhanced.keras',
        custom_objects={'rmse': rmse, 'mape': mape}
    )
    
    # Predict
    y_pred_scaled = model.predict(X, verbose=0, batch_size=BATCH_SIZE)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_true = target_scaler.inverse_transform(y.reshape(-1, 1))
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = np.mean(np.abs((y_true - y_pred) / (y_true + 0.01))) * 100
    r2 = r2_score(y_true, y_pred)
    
    all_results.append({
        'station': station,
        'mae': mae,
        'rmse': rmse_val,
        'mape': mape_val,
        'r2': r2
    })
    
    print(f"  MAE: {mae:.2f}% | RMSE: {rmse_val:.2f}% | R²: {r2:.4f}")

# ========== RESULTS ==========
print("\n" + "="*70)
print("FINAL RESULTS - ALL 13 STATIONS")
print("="*70)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('mae')

print(f"\n{'Station':<15} {'MAE(%)':<12} {'RMSE(%)':<12} {'MAPE(%)':<12} {'R²':<10}")
print("-"*65)
for _, row in results_df.iterrows():
    print(f"{row['station']:<15} {row['mae']:<12.2f} {row['rmse']:<12.2f} {row['mape']:<12.2f} {row['r2']:<10.4f}")

print(f"\nAverage MAE: {results_df['mae'].mean():.2f}%")
print(f"Total stations: {len(results_df)}/13")

# Save
results_df.to_csv(f'{MODELS_PATH}/all_13_stations_complete.csv', index=False)
print(f"\n✅ Saved to: {MODELS_PATH}/all_13_stations_complete.csv")