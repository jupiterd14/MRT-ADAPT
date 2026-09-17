# ============================================
# KAGGLE NOTEBOOK - MRT-3 LSTM TRAINING (V10+)
# WITH ALL 4 METRICS: MAE, RMSE, R², MAPE
# ============================================

import tensorflow as tf
import os
import gc
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import pickle
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("KAGGLE GPU CHECK")
print("="*60)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU FOUND: {gpus[0]}")
    print(f"   Training will be FAST")
else:
    print("❌ NO GPU DETECTED! Please enable GPU in Settings.")

print("="*60)

# ========== CONFIGURATION ==========
BATCH_SIZE = 128
EPOCHS = 120
PATIENCE_EARLY = 15
PATIENCE_LR = 10
SEQ_LENGTH = 24
CHUNK_SIZE = 100000

USE_STANDARD_SCALER = True
USE_HUBER_LOSS = True

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao",
    "Santolan", "Ortigas", "Shaw Blvd", "Boni Ave",
    "Guadalupe", "Buendia", "Ayala Ave", "Magallanes", "Taft"]

STATION_NUMBERS = {
    "North Ave": 1, "Quezon Ave": 2, "Kamuning": 3, "Cubao": 4,
    "Santolan": 5, "Ortigas": 6, "Shaw Blvd": 7, "Boni Ave": 8,
    "Guadalupe": 9, "Buendia": 10, "Ayala Ave": 11, "Magallanes": 12,
    "Taft": 13
}

# ✅ CLEAN FEATURE LIST - Reduced redundancy
feature_cols = [
    'TotalPassenger',  # Historical passenger counts (AUTOREGRESSIVE)
    
    # Time features
    'hour', 'weekday', 'month',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    
    # Operating patterns
    'is_operating_hour', 'is_morning_rush', 'is_evening_rush',
    
    # Calendar features
    'is_holiday', 'is_christmas_season', 'is_payday'
]

print(f"\n✅ {len(feature_cols)} clean features (reduced redundancy)")

# ========== HOLIDAYS ==========
holidays = [
    '2022-01-01', '2022-04-09', '2022-04-14', '2022-04-15', '2022-04-16',
    '2022-05-01', '2022-06-12', '2022-08-21', '2022-08-29', '2022-11-30',
    '2022-12-08', '2022-12-25', '2022-12-30', '2022-12-31',
    '2023-01-01', '2023-04-06', '2023-04-07', '2023-05-01', '2023-06-12',
    '2023-08-28', '2023-11-27', '2023-12-08', '2023-12-25', '2023-12-30',
    '2024-01-01', '2024-03-28', '2024-03-29', '2024-05-01', '2024-06-12',
    '2024-08-26', '2024-11-30', '2024-12-08', '2024-12-25', '2024-12-30', '2024-12-31',
    '2025-01-01', '2025-04-09', '2025-04-17', '2025-04-18', '2025-05-01',
    '2025-06-12', '2025-08-21', '2025-08-25', '2025-11-30', '2025-12-08',
    '2025-12-25', '2025-12-30', '2025-12-31'
]

def is_christmas_season(date):
    month_day = date.strftime('%m-%d')
    return (month_day >= '12-15') or (month_day <= '01-05')

def is_payday(date):
    return date.day in [15, 30, 31]

def add_cyclical_time_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    return df

def add_smart_operating_flags(df):
    time_decimal = df['hour'] + df['datetime'].dt.minute / 60
    df['is_operating_hour'] = ((time_decimal >= 4.5) & (time_decimal < 23.0)).astype(np.int8)
    df['is_morning_rush'] = ((time_decimal >= 7.0) & (time_decimal <= 9.0)).astype(np.int8)
    df['is_evening_rush'] = ((time_decimal >= 17.0) & (time_decimal <= 19.0)).astype(np.int8)
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

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    if USE_HUBER_LOSS:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=['mae']
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
            loss='mse',
            metrics=['mae']
        )
    
    return model

# ========== LOAD DATA ==========
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

data_folder = '/kaggle/input/datasets/jupiterd14/mrt3-passenger-data-2022-2024'
files = ['2022.csv', '2023.csv', '2024.csv']

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
        chunk = add_cyclical_time_features(chunk)
        chunk = add_smart_operating_flags(chunk)
        chunk['is_holiday'] = chunk['datetime'].dt.date.astype(str).isin(holidays).astype(np.int8)
        chunk['is_christmas_season'] = chunk['datetime'].apply(is_christmas_season).astype(np.int8)
        chunk['is_payday'] = chunk['datetime'].apply(is_payday).astype(np.int8)
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

# ========== TRAINING ==========
MODELS_PATH = f'models_2022-2024_v10_plus_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(MODELS_PATH, exist_ok=True)

all_results = []
total_start = time.time()

for station_idx, station in enumerate(STATIONS):
    station_num = STATION_NUMBERS[station]
    print(f"\n{'='*60}")
    print(f"Processing {station} ({station_idx+1}/{len(STATIONS)})")
    print('='*60)
    
    for direction in ['Northbound', 'Southbound']:
        # Filtering logic
        if station == "Taft":
            if direction == 'Northbound':
                station_df = df[df['StationEntry'] == station_num].copy()
            else:
                station_df = df[df['StationExit'] == station_num].copy()
        elif station == "North Ave":
            if direction == 'Southbound':
                station_df = df[df['StationEntry'] == station_num].copy()
            else:
                station_df = df[df['StationExit'] == station_num].copy()
        else:
            if direction == 'Northbound':
                station_df = df[df['StationExit'] == station_num].copy()
            else:
                station_df = df[df['StationEntry'] == station_num].copy()
        
        if len(station_df) < SEQ_LENGTH + 10:
            print(f"  [{direction}] Insufficient data")
            continue
        
        print(f"  [{direction}] Records: {len(station_df):,}")
        
        # Aggregate to hourly
        station_df['hour_timestamp'] = station_df['datetime'].dt.floor('h')
        
        dir_df = station_df.groupby('hour_timestamp').agg({
            'TotalPassenger': 'sum',
            'hour': 'first', 'weekday': 'first', 'month': 'first',
            'is_holiday': 'first',
            'is_christmas_season': 'first', 'is_payday': 'first',
            'hour_sin': 'first', 'hour_cos': 'first',
            'dow_sin': 'first', 'dow_cos': 'first', 'month_sin': 'first', 'month_cos': 'first',
            'is_operating_hour': 'max', 'is_morning_rush': 'max', 'is_evening_rush': 'max'
        }).reset_index()
        
        dir_df = dir_df.sort_values('hour_timestamp')
        
        # ========== FIX: Reindex to fill missing hours ==========
        all_hours = pd.date_range(start=dir_df['hour_timestamp'].min(),
                                   end=dir_df['hour_timestamp'].max(),
                                   freq='h')
        dir_df = dir_df.set_index('hour_timestamp').reindex(all_hours).reset_index()
        dir_df.rename(columns={'index': 'hour_timestamp'}, inplace=True)
        
        # ========== FIX: Track missing hours ==========
        dir_df['was_missing'] = dir_df['TotalPassenger'].isna().astype(np.int8)
        
        # TotalPassenger: fill missing with 0
        dir_df['TotalPassenger'] = dir_df['TotalPassenger'].fillna(0).clip(0, None)
        
        # ========== FIX: RECALCULATE time features from hour_timestamp ==========
        dir_df['hour'] = dir_df['hour_timestamp'].dt.hour
        dir_df['weekday'] = dir_df['hour_timestamp'].dt.weekday
        dir_df['month'] = dir_df['hour_timestamp'].dt.month
        
        dir_df['hour_sin'] = np.sin(2 * np.pi * dir_df['hour'] / 24)
        dir_df['hour_cos'] = np.cos(2 * np.pi * dir_df['hour'] / 24)
        dir_df['dow_sin'] = np.sin(2 * np.pi * dir_df['weekday'] / 7)
        dir_df['dow_cos'] = np.cos(2 * np.pi * dir_df['weekday'] / 7)
        dir_df['month_sin'] = np.sin(2 * np.pi * (dir_df['month'] - 1) / 12)
        dir_df['month_cos'] = np.cos(2 * np.pi * (dir_df['month'] - 1) / 12)
        
        time_decimal = dir_df['hour'] + dir_df['hour_timestamp'].dt.minute / 60
        dir_df['is_operating_hour'] = ((time_decimal >= 4.5) & (time_decimal < 23.0)).astype(np.int8)
        dir_df['is_morning_rush'] = ((time_decimal >= 7.0) & (time_decimal <= 9.0)).astype(np.int8)
        dir_df['is_evening_rush'] = ((time_decimal >= 17.0) & (time_decimal <= 19.0)).astype(np.int8)
        
        dir_df['is_holiday'] = dir_df['hour_timestamp'].dt.date.astype(str).isin(holidays).astype(np.int8)
        dir_df['is_christmas_season'] = dir_df['hour_timestamp'].apply(is_christmas_season).astype(np.int8)
        dir_df['is_payday'] = dir_df['hour_timestamp'].apply(is_payday).astype(np.int8)
        
        # ========== CHRONOLOGICAL SPLIT ==========
        n = len(dir_df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        train_df = dir_df.iloc[:train_end].copy()
        val_df = dir_df.iloc[train_end:val_end].copy()
        test_df = dir_df.iloc[val_end:].copy()
        
        # Scale features
        feature_scaler = MinMaxScaler()
        feature_scaler.fit(train_df[feature_cols])
        
        if USE_STANDARD_SCALER:
            target_scaler = StandardScaler()
        else:
            target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        target_scaler.fit(train_df[['TotalPassenger']])
        
        # Training
        train_features = feature_scaler.transform(train_df[feature_cols])
        train_targets = target_scaler.transform(train_df[['TotalPassenger']]).flatten()
        X_train, y_train = create_sequences(train_features, train_targets)
        
        # Validation (with proper context)
        combined_df = pd.concat([train_df, val_df], axis=0)
        combined_features = feature_scaler.transform(combined_df[feature_cols])
        combined_targets = target_scaler.transform(combined_df[['TotalPassenger']]).flatten()
        
        X_combined, y_combined = create_sequences(combined_features, combined_targets)
        
        start_idx = len(train_df) - SEQ_LENGTH
        X_val = X_combined[start_idx:]
        y_val = y_combined[start_idx:]
        
        if len(X_train) == 0 or len(X_val) == 0:
            print(f"  [{direction}] Not enough sequences")
            continue
        
        model_key = f"{station}_{direction}"
        input_shape = (SEQ_LENGTH, len(feature_cols))
        model = build_lstm_model(input_shape)
        
        print(f"    Train: {len(X_train):,} | Val: {len(X_val):,}")
        print(f"    Target range: {train_df['TotalPassenger'].min():.0f} - {train_df['TotalPassenger'].max():.0f}")
        
        early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE_EARLY,
                                   restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=PATIENCE_LR, min_lr=0.00001, verbose=1)
        
        history = model.fit(X_train, y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stop, reduce_lr],
                            verbose=1)
        
        # ========== EVALUATION WITH ALL 4 METRICS ==========
        test_features = feature_scaler.transform(test_df[feature_cols])
        test_targets = target_scaler.transform(test_df[['TotalPassenger']]).flatten()
        
        X_test, y_test = create_sequences(test_features, test_targets)
        
        if len(X_test) > 0:
            y_pred_scaled = model.predict(X_test, verbose=0)
            y_pred_passengers = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
            y_true_passengers = target_scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # ✅ 4 METRICS
            mae = mean_absolute_error(y_true_passengers, y_pred_passengers)
            rmse = np.sqrt(mean_squared_error(y_true_passengers, y_pred_passengers))
            r2 = r2_score(y_true_passengers, y_pred_passengers)
            
            # ✅ MAPE (only on non-zero actuals to avoid division by zero)
            non_zero_mask = y_true_passengers > 0
            if non_zero_mask.sum() > 0:
                mape = mean_absolute_percentage_error(
                    y_true_passengers[non_zero_mask], 
                    y_pred_passengers[non_zero_mask]
                ) * 100
            else:
                mape = float('inf')
            
            print(f"\n  {model_key} Test Performance:")
            print(f"    MAE:  {mae:.0f} passengers")
            print(f"    RMSE: {rmse:.0f} passengers")
            print(f"    R²:   {r2:.4f}")
            print(f"    MAPE: {mape:.1f}% (on non-zero hours)")
        else:
            mae, rmse, r2, mape = None, None, None, None
            print(f"\n  {model_key} Test Performance: No test data")
        
        # Save
        model.save(f'{MODELS_PATH}/{model_key}_lstm_v10_plus.keras')
        with open(f'{MODELS_PATH}/{model_key}_feature_scaler.pkl', 'wb') as f:
            pickle.dump(feature_scaler, f)
        with open(f'{MODELS_PATH}/{model_key}_target_scaler.pkl', 'wb') as f:
            pickle.dump(target_scaler, f)
        
        all_results.append({
            'station': model_key,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        })
        
        del model, history
        gc.collect()

# ========== SUMMARY ==========
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

if all_results:
    results_df = pd.DataFrame(all_results)
    print(f"\n📊 Average Test Results:")
    print(f"    MAE:  {results_df['mae'].mean():.0f} passengers")
    print(f"    RMSE: {results_df['rmse'].mean():.0f} passengers")
    print(f"    R²:   {results_df['r2'].mean():.4f}")
    print(f"    MAPE: {results_df['mape'].mean():.1f}% (on non-zero hours)")
    print(f"\n📁 Models saved to: {MODELS_PATH}/")
    
    with open(f'{MODELS_PATH}/feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)

print("\n✅ DONE!")