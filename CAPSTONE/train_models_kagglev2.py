import os
import sys
from datetime import datetime

# ============================================
# CREATE A NEW UNIQUE FOLDER FOR THIS TRAINING
# ============================================
MODELS_PATH = f'models_2022-2024_v7_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
print(f"📁 Models will be saved to: {MODELS_PATH}")
print("="*60)

# ============================================
# YOUR COMPLETE TRAINING SCRIPT
# ============================================

# First, ensure data folder exists
os.makedirs("data (2022-2024)", exist_ok=True)

print("\n📁 Checking for CSV files...")

# Check if CSV files are available
import pandas as pd
data_folder = None
files = ['2022.csv', '2023.csv', '2024.csv']

# Find the data folder
if os.path.exists('/kaggle/input'):
    for root, dirs, filenames in os.walk('/kaggle/input'):
        for file in files:
            if file in filenames:
                data_folder = root
                print(f"✅ Found data in: {data_folder}")
                break
        if data_folder:
            break

if not data_folder:
    data_folder = 'data (2022-2024)'
    print(f"📁 Using local data folder: {data_folder}")

# Verify files exist
for file in files:
    file_path = os.path.join(data_folder, file)
    if os.path.exists(file_path):
        print(f"  ✅ {file} found")
    else:
        print(f"  ❌ {file} NOT found at {file_path}")
        raise FileNotFoundError(f"Missing {file}")

print("\n" + "="*60)
print("STARTING TRAINING...")
print("="*60)

# Now run the training
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import gc
import time
import warnings
warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
BATCH_SIZE = 128
EPOCHS = 120
PATIENCE_EARLY = 15 
PATIENCE_LR = 10
SEQ_LENGTH = 24
CHUNK_SIZE = 100000
USE_BIDIRECTIONAL = False
USE_ROBUST_SCALER = False

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao",
    "Santolan", "Ortigas", "Shaw Blvd", "Boni Ave",
    "Guadalupe", "Buendia", "Ayala Ave", "Magallanes",
    "Taft"]

STATION_NUMBERS = {
    "North Ave": 1, "Quezon Ave": 2, "Kamuning": 3, "Cubao": 4,
    "Santolan": 5, "Ortigas": 6, "Shaw Blvd": 7, "Boni Ave": 8,
    "Guadalupe": 9, "Buendia": 10, "Ayala Ave": 11, "Magallanes": 12,
    "Taft": 13
}

# ========== OFFICIAL DOTr PLATFORM CAPACITIES ==========
MRT3_PLATFORM_CAPACITY = {
    "North Ave": 1142,
    "Quezon Ave": 1195,
    "Kamuning": 1364,
    "Cubao": 1747,
    "Santolan": 1306,
    "Ortigas": 1331,
    "Shaw Blvd": 1619,
    "Boni Ave": 1417,
    "Guadalupe": 1301,
    "Buendia": 1645,
    "Ayala Ave": 1222,
    "Magallanes": 1202,
    "Taft": 720
}

# ========== HOLIDAYS & EVENTS ==========
holidays = [
    '2022-01-01', '2022-04-09', '2022-04-14', '2022-04-15', '2022-04-16',
    '2022-05-01', '2022-06-12', '2022-08-21', '2022-08-29', '2022-11-30',
    '2022-12-08', '2022-12-25', '2022-12-30', '2022-12-31',
    '2023-01-01', '2023-04-06', '2023-04-07', '2023-05-01', '2023-06-12',
    '2023-08-28', '2023-11-27', '2023-12-08', '2023-12-25', '2023-12-30',
    '2024-01-01', '2024-03-28', '2024-03-29', '2024-05-01', '2024-06-12',
    '2024-08-26', '2024-11-30', '2024-12-08', '2024-12-25', '2024-12-30', '2024-12-31'
]

special_events = {
    '2022-01-15': 'COVID-19 surge restrictions',
    '2022-03-01': 'Alert Level 1 implemented',
    '2022-05-09': 'Election Day',
    '2022-06-30': 'New President inauguration',
    '2022-09-01': 'School year opening',
    '2022-11-01': 'Undas/All Saints Day (non-working)',
    '2022-12-24': 'Christmas Eve (special)',
    '2023-01-09': 'Feast of Black Nazarene',
    '2023-04-21': 'Eid al-Fitr',
    '2023-06-28': 'Eid al-Adha',
    '2023-10-30': 'Barangay Elections',
    '2023-11-02': 'All Souls Day (special)',
    '2023-12-01': 'Fare adjustment announced',
    '2023-12-24': 'Christmas Eve (special)',
    '2024-02-10': 'Chinese New Year',
    '2024-03-11': 'Eid al-Fitr',
    '2024-04-09': 'Day of Valor',
    '2024-08-08': 'Technical issue Boni-Guadalupe',
    '2024-08-21': 'Ninoy Aquino Day',
    '2024-09-30': 'Regular maintenance',
    '2024-11-01': 'All Saints Day',
    '2024-12-24': 'Christmas Eve (special)'
}

def is_christmas_season(date):
    month_day = date.strftime('%m-%d')
    return (month_day >= '12-15') or (month_day <= '01-05')

def is_payday(date):
    return date.day in [15, 30, 31]

def is_friday(date):
    return date.weekday() == 4

def infer_direction(row):
    entry = row['StationEntry']
    exit_station = row['StationExit']
    if entry < exit_station:  
        return 'Southbound'
    elif entry > exit_station:   
        return 'Northbound'
    else:
        return 'Unknown'

def add_cyclical_time_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    df['time_decimal'] = df['hour'] + df['datetime'].dt.minute / 60
    df['is_operating_hour'] = ((df['time_decimal'] >= 4.5) & (df['time_decimal'] < 23.0)).astype(np.int8)
    df['minute_normalized'] = df['datetime'].dt.minute / 60.0
    return df

def add_smart_operating_flags(df):
    time_decimal = df['time_decimal']
    df['is_morning_rush'] = ((time_decimal >= 7.0) & (time_decimal <= 9.0)).astype(np.int8)
    df['is_evening_rush'] = ((time_decimal >= 17.0) & (time_decimal <= 19.0)).astype(np.int8)
    df['is_noon'] = ((time_decimal >= 12.0) & (time_decimal <= 13.0)).astype(np.int8)
    df['is_pre_opening'] = ((time_decimal >= 4.5) & (time_decimal < 5.0)).astype(np.int8)
    df['is_post_closing'] = ((time_decimal >= 22.5) & (time_decimal < 23.0)).astype(np.int8)
    minutes_until = (23.0 - time_decimal) * 60
    df['minutes_until_closing'] = minutes_until.clip(lower=0).astype(np.float32)
    minutes_since = (time_decimal - 4.5) * 60
    df['minutes_since_opening'] = minutes_since.clip(lower=0).astype(np.float32)
    df['time_normalized'] = (time_decimal - 4.5) / (23.0 - 4.5)
    df['time_normalized'] = df['time_normalized'].clip(0, 1)
    return df

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

def evaluate_model(model, X_val, y_val, target_scaler, model_name):
    y_pred_scaled = model.predict(X_val, verbose=0, batch_size=BATCH_SIZE)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_true = target_scaler.inverse_transform(y_val.reshape(-1, 1))
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    print(f"\n{model_name} Performance:")
    print(f"   MAE: {mae:.2f}% | RMSE: {rmse:.2f}% | MAPE: {mape:.2f}% | R2: {r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}

def build_lstm_model(input_shape):
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    if USE_BIDIRECTIONAL:
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
    else:
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape), 
            Dropout(0.2),
            LSTM(32, return_sequences=False),                          
            Dropout(0.2),
            Dense(16, activation='relu'),                               
            Dense(1)
        ])
    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss='mse', metrics=['mae', rmse])
    return model

# ========== LOAD DATA ==========
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

chunks = []
total_rows = 0
start_load = time.time()

for file in files:
    file_path = os.path.join(data_folder, file)
    print(f"Reading {file_path}...")
    if not os.path.exists(file_path):
        print(f"Skipping {file}: File not found.")
        continue
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)):
        chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'])
        chunk['hour'] = chunk['datetime'].dt.hour
        chunk['weekday'] = chunk['datetime'].dt.weekday
        chunk['month'] = chunk['datetime'].dt.month
        chunk['day'] = chunk['datetime'].dt.day
        chunk = add_cyclical_time_features(chunk)
        chunk = add_smart_operating_flags(chunk)
        chunk['is_weekend'] = (chunk['datetime'].dt.weekday >= 5).astype(np.int8)
        chunk['is_holiday'] = chunk['datetime'].dt.date.astype(str).isin(holidays).astype(np.int8)
        chunk['is_special_event'] = chunk['datetime'].dt.date.astype(str).isin(special_events.keys()).astype(np.int8)
        chunk['is_christmas_season'] = chunk['datetime'].apply(is_christmas_season).astype(np.int8)
        chunk['is_payday'] = chunk['datetime'].apply(is_payday).astype(np.int8)
        chunk['is_friday'] = chunk['datetime'].apply(is_friday).astype(np.int8)
        chunk['is_rush_hour'] = ((chunk['hour'].between(7, 9)) | (chunk['hour'].between(17, 19))).astype(np.int8)
        chunk['Direction'] = chunk.apply(infer_direction, axis=1)
        chunk['is_maintenance_record'] = ((chunk['time_decimal'] < 5.0) & (chunk['TotalPassenger'] < 10)).astype(np.int8)
        chunk['is_extended_hours'] = ((chunk['time_decimal'] >= 22.0) & (chunk['time_decimal'] < 23.0) & (chunk['TotalPassenger'] >= 10)).astype(np.int8)
        chunks.append(chunk)
        total_rows += len(chunk)
        if (i + 1) % 10 == 0:
            print(f"  Loaded {total_rows:,} records...")
            gc.collect()

print(f"\nConcatenating {len(chunks)} chunks...")
df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"Loaded {len(df):,} records in {time.time() - start_load:.1f} seconds")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# ========== CONGESTION CALCULATION ==========
print("\n📊 Computing congestion using DOTr platform capacities...")
station_num_to_name = {v: k for k, v in STATION_NUMBERS.items()}
df['station_name'] = df['StationEntry'].map(station_num_to_name)
df['capacity'] = df['station_name'].map(MRT3_PLATFORM_CAPACITY).fillna(1000)
df['congestion'] = (df['TotalPassenger'] / df['capacity'] * 100).clip(0, 100)

print(f"✅ Congestion computed - Range: {df['congestion'].min():.1f}% - {df['congestion'].max():.1f}%")

# ========== FEATURES ==========
feature_cols = [
    'hour', 'weekday', 'month',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'is_operating_hour', 'is_morning_rush', 'is_evening_rush', 'is_noon',
    'is_pre_opening', 'is_post_closing',
    'minutes_until_closing', 'minutes_since_opening', 'time_normalized', 'minute_normalized',
    'is_weekend', 'is_holiday', 'is_special_event', 'is_christmas_season', 'is_payday', 'is_friday',
    'is_rush_hour', 'is_maintenance_record', 'is_extended_hours', 'congestion'
]

# ========== PRE-PROCESSING ==========
print("\n" + "="*60)
print(f"PRE-PROCESSING {len(STATIONS)} STATIONS")
print("="*60)

station_data_dict = {}
scaler_type = RobustScaler if USE_ROBUST_SCALER else MinMaxScaler

for station_idx, station in enumerate(STATIONS):
    station_num = STATION_NUMBERS[station]
    print(f"\nProcessing {station} ({station_idx+1}/{len(STATIONS)})...")
    
    for direction in ['Northbound', 'Southbound']:
        if direction == 'Northbound':
            station_df = df[df['StationExit'] == station_num].copy()
        else:
            station_df = df[df['StationEntry'] == station_num].copy()
        
        if len(station_df) == 0:
            print(f"  [{direction}] No data")
            continue
        
        station_df['inferred_direction'] = station_df.apply(infer_direction, axis=1)
        station_df = station_df[station_df['inferred_direction'] == direction]
        
        if len(station_df) < SEQ_LENGTH + 10:
            print(f"  [{direction}] Insufficient data: {len(station_df)} records")
            continue
        
        print(f"  [{direction}] Records: {len(station_df):,}")
        
        station_df['hour_timestamp'] = station_df['datetime'].dt.floor('h')
        dir_df = station_df.groupby('hour_timestamp').agg({
            'TotalPassenger': 'sum',
            'congestion': 'mean',
            'hour': 'first', 'weekday': 'first', 'month': 'first',
            'is_weekend': 'first', 'is_holiday': 'first', 'is_special_event': 'first',
            'is_christmas_season': 'first', 'is_payday': 'first', 'is_friday': 'first',
            'is_rush_hour': 'max',
            'hour_sin': 'first', 'hour_cos': 'first',
            'dow_sin': 'first', 'dow_cos': 'first', 'month_sin': 'first', 'month_cos': 'first',
            'is_operating_hour': 'max', 'is_morning_rush': 'max', 'is_evening_rush': 'max',
            'is_noon': 'max', 'is_pre_opening': 'max', 'is_post_closing': 'max',
            'minutes_until_closing': 'mean', 'minutes_since_opening': 'mean',
            'time_normalized': 'mean', 'minute_normalized': 'mean',
            'is_maintenance_record': 'max', 'is_extended_hours': 'max'
        }).reset_index()
        
        dir_df = dir_df.sort_values('hour_timestamp')
        dir_df['congestion'] = dir_df['congestion'].clip(0, 100)
        
        n = len(dir_df)
        train_size = int(n * 0.8)
        train_df = dir_df.iloc[:train_size].copy()
        val_df = dir_df.iloc[train_size:].copy()
        
        feature_scaler = scaler_type()
        feature_scaler.fit(train_df[feature_cols])
        
        target_scaler = MinMaxScaler()
        target_scaler.fit(train_df[['congestion']])

        train_features = feature_scaler.transform(train_df[feature_cols])
        train_targets = target_scaler.transform(train_df[['congestion']]).flatten()
        
        val_features = feature_scaler.transform(val_df[feature_cols])
        val_targets = target_scaler.transform(val_df[['congestion']]).flatten()
        
        X_train, y_train = create_sequences(train_features, train_targets)
        X_val, y_val = create_sequences(val_features, val_targets)
        
        if len(X_train) == 0 or len(X_val) == 0:
            print(f"  [{direction}] Not enough sequences")
            continue
        
        model_key = f"{station}_{direction}"
        station_data_dict[model_key] = {
            'X_train': X_train, 'X_val': X_val,
            'y_train': y_train, 'y_val': y_val,
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        }
        print(f"    Train sequences: {len(X_train):,} | Val sequences: {len(X_val):,}")
        print(f"    Congestion range: {dir_df['congestion'].min():.1f}% - {dir_df['congestion'].max():.1f}%")

print(f"\n✅ Pre-processed {len(station_data_dict)} directional models")

os.makedirs(MODELS_PATH, exist_ok=True)

# Save capacities
with open(f'{MODELS_PATH}/station_platform_capacities.pkl', 'wb') as f:
    pickle.dump(MRT3_PLATFORM_CAPACITY, f)

del df
gc.collect()

all_results = []
start_time = time.time()

# ========== TRAINING ==========
for model_idx, (model_key, data) in enumerate(station_data_dict.items()):
    model_start = time.time()
    print(f"\n{'='*60}")
    print(f"Training: {model_key} ({model_idx+1}/{len(station_data_dict)})")
    print('='*60)
    
    input_shape = (SEQ_LENGTH, len(feature_cols))
    model = build_lstm_model(input_shape)
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE_EARLY, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE_LR, min_lr=0.00001, verbose=1)
    
    print(f"Train samples: {len(data['X_train']):,} | Val samples: {len(data['X_val']):,}")
    
    history = model.fit(data['X_train'], data['y_train'], epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(data['X_val'], data['y_val']), 
                        callbacks=[early_stop, reduce_lr], verbose=1)
    
    eval_metrics = evaluate_model(model, data['X_val'], data['y_val'], data['target_scaler'], model_key)
    
    model.save(f'{MODELS_PATH}/{model_key}_lstm_enhanced.keras')
    
    with open(f'{MODELS_PATH}/{model_key}_feature_scaler.pkl', 'wb') as f:
        pickle.dump(data['feature_scaler'], f)
    with open(f'{MODELS_PATH}/{model_key}_target_scaler.pkl', 'wb') as f:
        pickle.dump(data['target_scaler'], f)
    
    model_time = time.time() - model_start
    all_results.append({'station': model_key, 
                        'mae': eval_metrics['mae'], 
                        'rmse': eval_metrics['rmse'],
                        'r2': eval_metrics['r2'],
                        'time_min': model_time / 60})
    print(f"\n{model_key} complete in {model_time/60:.1f} min | MAE: {eval_metrics['mae']:.2f}%")
    
    del model, history
    gc.collect()

# ========== FINAL SUMMARY ==========
total_time = time.time() - start_time
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nTotal training time: {total_time/60:.1f} min")

if len(all_results) > 0:
    results_df = pd.DataFrame(all_results)
    print("\nRESULTS SUMMARY:")
    print(results_df[['station', 'mae', 'rmse', 'r2', 'time_min']].to_string(index=False))
    results_df.to_csv(f'{MODELS_PATH}/training_summary.csv', index=False)
    
    print(f"\n📁 All models saved to: {MODELS_PATH}/")
    print("\nSAVED FILES:")
    for model_key in station_data_dict.keys():
        print(f"   - {model_key}_lstm_enhanced.keras")
        print(f"   - {model_key}_feature_scaler.pkl")
        print(f"   - {model_key}_target_scaler.pkl")
else:
    print("\n❌ No models were trained!")

print("\n" + "="*60)
print("✅ TRAINING SCRIPT COMPLETE")
print("="*60)