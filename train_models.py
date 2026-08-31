#train_local_full_optimized.py 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pandas as pd
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
from datetime import datetime
import warnings
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ==========CONFIGURATION FOR 3M ROWS ==========
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

data_folder = 'data (2022-2024)'
files = ['2023.csv', '2024.csv']    
MODELS_PATH = 'models_2023-2024'

# ========== HOLIDAYS 2022-2024 ==========
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
    """Infer direction from entry and exit station numbers"""
    entry = row['StationEntry']
    exit_station = row['StationExit']
    
    if entry < exit_station:  
        return 'Southbound'
    elif entry > exit_station:   
        return 'Northbound'
    else:
        return 'Unknown'



def smart_data_cleaner(df):
    """
     clean the data based on MRT-3 operating schedule
    Keeps legitimate late-night records but flags maintenance entries
    """
    time_decimal = df['time_decimal']
    passenger_count = df['TotalPassenger']
    
    # Flag maintenance records (1-5 AM with very low passenger counts)
    df['is_maintenance_record'] = (
        (time_decimal < 5.0) & 
        (passenger_count < 10)
    ).astype(np.int8)
    
    # For maintenance records, set congestion to 0
    # This teaches the model that these hours should have no passengers
    df.loc[df['is_maintenance_record'] == 1, 'congestion'] = 0
    
    # Flag extended hours records (legitimate late-night passengers)
    df['is_extended_hours'] = (
        (time_decimal >= 22.0) & 
        (time_decimal < 23.0) &
        (passenger_count >= 10)
    ).astype(np.int8)
    
    return df

# ========== CYCLICAL TIME FEATURES ==========
def add_cyclical_time_features(df):
    """
    Add cyclical (sin/cos) encodings for time features.
    This helps the LSTM understand that 23:00 and 00:00 are actually close in time.
    """
    # Hour of day (0-23) -> cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week (0-6) -> cyclical encoding
    df['dow_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    # Month (1-12) -> cyclical encoding (for seasonal patterns)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    
    # Convert to decimal hours for precise comparison
    df['time_decimal'] = df['hour'] + df['datetime'].dt.minute / 60
    df['is_operating_hour'] = (
        (df['time_decimal'] >= 4.5) &  # 4:30 AM opening
        (df['time_decimal'] < 23.0)     # 11:00 PM closing (last train departs ~10:30, but station open until 11)
    ).astype(np.int8)
    
    # Minute within hour (for more granular time)
    df['minute_normalized'] = df['datetime'].dt.minute / 60.0
    
    return df

def add_smart_operating_flags(df):
    """
    Add flags for different time periods to help model distinguish patterns
    Uses ACTUAL MRT-3 schedule
    """
    hour = df['hour']
    minute = df['datetime'].dt.minute
    time_decimal = df['time_decimal']
    
    # Rush hour flags (more granular)
    df['is_morning_rush'] = ((time_decimal >= 7.0) & (time_decimal <= 9.0)).astype(np.int8)
    df['is_evening_rush'] = ((time_decimal >= 17.0) & (time_decimal <= 19.0)).astype(np.int8)
    df['is_noon'] = ((time_decimal >= 12.0) & (time_decimal <= 13.0)).astype(np.int8)
    
    # Pre-opening and post-closing periods
    df['is_pre_opening'] = ((time_decimal >= 4.5) & (time_decimal < 5.0)).astype(np.int8)
    df['is_post_closing'] = ((time_decimal >= 22.5) & (time_decimal < 23.0)).astype(np.int8)
    
    # Time-based features for transition periods
    # Minutes until closing (negative during closed hours, set to 0)
    minutes_until = (23.0 - time_decimal) * 60
    df['minutes_until_closing'] = minutes_until.clip(lower=0).astype(np.float32)
    
    # Minutes since opening
    minutes_since = (time_decimal - 4.5) * 60
    df['minutes_since_opening'] = minutes_since.clip(lower=0).astype(np.float32)
    
    # Normalized time of day (0-1) for additional context
    df['time_normalized'] = (time_decimal - 4.5) / (23.0 - 4.5)
    df['time_normalized'] = df['time_normalized'].clip(0, 1)
    
    return df


def create_sequences(features, target, seq_length=SEQ_LENGTH):
    """Create sequences for LSTM - optimized with numpy"""
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
    """Evaluate model performance with correct MAPE"""
    y_pred_scaled = model.predict(X_val, verbose=0, batch_size=BATCH_SIZE)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_true = target_scaler.inverse_transform(y_val.reshape(-1, 1))
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
  
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # Also calculate sMAPE (Symmetric MAPE) which handles zeros better
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + epsilon)) * 100
    
    print(f"\n{model_name} Performance:")
    print(f"   MAE: {mae:.2f}% | RMSE: {rmse:.2f}% | MAPE: {mape:.2f}% | sMAPE: {smape:.2f}% | R2: {r2:.4f}")
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'smape': smape, 'r2': r2}

def plot_predictions(model, X_val, y_val, target_scaler, model_key, models_path):
    """Plot predictions vs actual"""
    y_pred_scaled = model.predict(X_val, verbose=0, batch_size=BATCH_SIZE)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_true = target_scaler.inverse_transform(y_val.reshape(-1, 1))
    
    n_plot = min(500, len(y_true))
    
    plt.figure(figsize=(15, 5))
    plt.plot(y_true[:n_plot], label='Actual', alpha=0.7, linewidth=1)
    plt.plot(y_pred[:n_plot], label='Predicted', alpha=0.7, linewidth=1)
    plt.title(f'{model_key} - Predicted vs Actual Congestion (First {n_plot} samples)')
    plt.xlabel('Time Step')
    plt.ylabel('Congestion (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{models_path}/{model_key}_predictions.png', dpi=100)
    plt.close()

def plot_all_metrics(results_df, models_path):
    """Create a single figure with all 4 metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    stations = results_df['station'].tolist()
    
    # 1. MAE
    axes[0,0].bar(stations, results_df['mae'], color='skyblue')
    axes[0,0].axhline(y=results_df['mae'].mean(), color='red', linestyle='--', label=f'Avg: {results_df["mae"].mean():.2f}%')
    axes[0,0].set_ylabel('MAE (%)')
    axes[0,0].set_title('1. Mean Absolute Error (Lower is Better)')
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. RMSE
    axes[0,1].bar(stations, results_df['rmse'], color='lightgreen')
    axes[0,1].axhline(y=results_df['rmse'].mean(), color='red', linestyle='--', label=f'Avg: {results_df["rmse"].mean():.2f}%')
    axes[0,1].set_ylabel('RMSE (%)')
    axes[0,1].set_title('2. Root Mean Square Error (Lower is Better)')
    axes[0,1].legend()
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. MAPE
    axes[1,0].bar(stations, results_df['mape'], color='orange')
    axes[1,0].axhline(y=results_df['mape'].mean(), color='red', linestyle='--', label=f'Avg: {results_df["mape"].mean():.2f}%')
    axes[1,0].set_ylabel('MAPE (%)')
    axes[1,0].set_title('3. Mean Absolute Percentage Error (Lower is Better)')
    axes[1,0].legend()
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. R²
    axes[1,1].bar(stations, results_df['r2'], color='purple')
    axes[1,1].axhline(y=results_df['r2'].mean(), color='red', linestyle='--', label=f'Avg: {results_df["r2"].mean():.3f}')
    axes[1,1].set_ylabel('R² Score')
    axes[1,1].set_title('4. R² Score (Closer to 1 is Better)')
    axes[1,1].legend()
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{models_path}/all_4_metrics_summary.png', dpi=150)
    plt.close()
    print(f"Saved: {models_path}/all_4_metrics_summary.png")

def plot_training_history(history, model_key, models_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=1.5)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=1.5)
    axes[0].set_title(f'{model_key} - Loss (MSE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=1.5)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=1.5)
    axes[1].set_title(f'{model_key} - MAE (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{models_path}/{model_key}_training_history.png', dpi=100)
    plt.close()
    
def build_lstm_model(input_shape):
    """Build LSTM model - configurable"""
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    
    if USE_BIDIRECTIONAL:
        #di yan magwowork. set as FALSE si bi sa taas
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
    
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),  # Add gradient clipping
        loss='mse', 
        metrics=['mae', rmse]
    )
    return model

# ========== LOAD & ENHANCE DATA IN CHUNKS ==========
print("="*60)

chunks = []
total_rows = 0
start_load = time.time()

for file in files:
    file_path = os.path.join(data_folder, file)
    print(f"Reading {file_path} in chunks of {CHUNK_SIZE:,} rows...")
    
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
        
        # ========== SMART DATA CLEANING ==========
        chunk = smart_data_cleaner(chunk)
        
        chunk['is_weekend'] = (chunk['datetime'].dt.weekday >= 5).astype(np.int8)
        chunk['is_holiday'] = chunk['datetime'].dt.date.astype(str).isin(holidays).astype(np.int8)
        chunk['is_special_event'] = chunk['datetime'].dt.date.astype(str).isin(special_events.keys()).astype(np.int8)
        chunk['is_christmas_season'] = chunk['datetime'].apply(is_christmas_season).astype(np.int8)
        chunk['is_payday'] = chunk['datetime'].apply(is_payday).astype(np.int8)
        chunk['is_friday'] = chunk['datetime'].apply(is_friday).astype(np.int8)
        chunk['is_rush_hour'] = ((chunk['hour'].between(7, 9)) | (chunk['hour'].between(17, 19))).astype(np.int8)
        
        chunk['Direction'] = chunk.apply(infer_direction, axis=1)
        
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
print("Available columns:", df.columns.tolist())
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Unique stations: {df['StationEntry'].unique()}")
print(f"Missing values: {df.isnull().sum().sum()}")

if 'TotalPassenger' in df.columns:
    max_passengers = df['TotalPassenger'].quantile(0.99)
    df['congestion'] = (df['TotalPassenger'] / max_passengers * 100).clip(0, 100)
    print(f"Congestion calculated from TotalPassenger. Max (99th percentile): {max_passengers:.0f}")
elif 'passenger_count' in df.columns:
    max_passengers = df['passenger_count'].quantile(0.99)
    df['congestion'] = (df['passenger_count'] / max_passengers * 100).clip(0, 100)
    print(f"Congestion calculated from passenger_count. Max: {max_passengers:.0f}")
else:
    print("WARNING: No passenger count column! Using entry frequency as congestion proxy.")
    entries_per_hour = df.groupby(['StationEntry', df['datetime'].dt.hour]).size()
    max_entries = entries_per_hour.max()
    df['congestion'] = df.groupby(['StationEntry', df['datetime'].dt.hour])['StationEntry'].transform('count') / max_entries * 100
    df['congestion'] = df['congestion'].clip(0, 100)

print(f"Congestion range: {df['congestion'].min():.1f}% - {df['congestion'].max():.1f}%")

os.makedirs(MODELS_PATH, exist_ok=True)
print(f"Models will save to: {MODELS_PATH}")

# ========== FEATURES ==========

feature_cols = [
    # Basic time features
    'hour', 'weekday', 'month',
    
    # Cyclical time features (for circular time understanding)
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'month_sin', 'month_cos',
    
    # MRT operating hour flags
    'is_operating_hour',
    'is_morning_rush', 'is_evening_rush', 'is_noon',
    'is_pre_opening', 'is_post_closing',
    
    # Time transition features
    'minutes_until_closing', 'minutes_since_opening',
    'time_normalized', 'minute_normalized',
    
    # Calendar features
    'is_weekend', 'is_holiday', 'is_special_event',
    'is_christmas_season', 'is_payday', 'is_friday',
    
    # Legacy rush hour (keeping for compatibility)
    'is_rush_hour',
    
    # Data quality flags
    'is_maintenance_record', 'is_extended_hours',
    
    # Target (will be predicted)
    'congestion'
]

print(f"\n{len(feature_cols)} ENHANCED FEATURES (Cyclical Time + MRT Schedule):")
for i, f in enumerate(feature_cols, 1):
    print(f"   {i}. {f}")


# ========== PRE-FILTER AND PRE-PROCESS (PER-DIRECTION MODELS) ==========
print("\n" + "="*60)
print(f"PRE-PROCESSING {len(STATIONS)} STATIONS (Per-Direction Models)")
print("="*60)

station_data_dict = {}
scaler_type = RobustScaler if USE_ROBUST_SCALER else MinMaxScaler

for station_idx, station in enumerate(STATIONS):
    station_num = STATION_NUMBERS[station]
    print(f"\nProcessing {station} ({station_idx+1}/{len(STATIONS)})...")
    
    mask = (df['StationEntry'] == station_num) | (df['StationExit'] == station_num)
    station_df = df[mask].copy()
    
    #filter to only valid directions (Direction column already exists from chunk loading)
    station_df = station_df[station_df['Direction'] != 'Unknown']
    
    # 1. Floor the datetime to the nearest hour
    station_df['hour_timestamp'] = station_df['datetime'].dt.floor('h')
    
    # 2. Aggregate: Sum passengers by hour AND direction
    station_df = station_df.groupby(['hour_timestamp', 'Direction']).agg({
        'TotalPassenger': 'sum',
        'hour': 'first',
        'weekday': 'first',
        'month': 'first',
        'is_weekend': 'first',
        'is_holiday': 'first',
        'is_special_event': 'first',
        'is_christmas_season': 'first',
        'is_payday': 'first',
        'is_friday': 'first',
        'is_rush_hour': 'first',
        'hour_sin': 'first',
        'hour_cos': 'first',
        'dow_sin': 'first',
        'dow_cos': 'first',
        'month_sin': 'first',
        'month_cos': 'first',
        'is_operating_hour': 'first',
        'is_morning_rush': 'first',
        'is_evening_rush': 'first',
        'is_noon': 'first',
        'is_pre_opening': 'first',
        'is_post_closing': 'first',
        'minutes_until_closing': 'first',
        'minutes_since_opening': 'first',
        'time_normalized': 'first',
        'minute_normalized': 'first',
        'is_maintenance_record': 'first',
        'is_extended_hours': 'first'
    }).reset_index()
    
    # 3. SPLIT BY DIRECTION - Train separate models for each
    for direction in ['Northbound', 'Southbound']:
        dir_df = station_df[station_df['Direction'] == direction].sort_values('hour_timestamp')
        
        if len(dir_df) < SEQ_LENGTH + 10:
            print(f"  [{direction}] Insufficient data: {len(dir_df)} records (need {SEQ_LENGTH + 10})")
            continue
        
        print(f"  [{direction}] Records: {len(dir_df):,}")
        
        # Recalculate congestion for this platform's hourly totals
        max_hourly = dir_df['TotalPassenger'].quantile(0.99)
        dir_df['congestion'] = (dir_df['TotalPassenger'] / max_hourly * 100).clip(0, 100)
        
        feature_data = dir_df[feature_cols].values.astype(np.float32)
        
        feature_scaler = scaler_type()
        features_scaled = feature_scaler.fit_transform(feature_data).astype(np.float32)
        
        target_scaler = MinMaxScaler()
        targets_raw = dir_df['congestion'].values.reshape(-1, 1).astype(np.float32)
        targets_scaled = target_scaler.fit_transform(targets_raw).flatten()
        
        X, y = create_sequences(features_scaled, targets_scaled, seq_length=SEQ_LENGTH)
        
        if len(X) == 0:
            print(f"  [{direction}] No sequences created")
            continue
        
        print(f"  [{direction}] Sequences: {len(X):,}")
        
        split_idx = int(len(X) * 0.8)
        
        # Use naming convention: Station_Direction (e.g., "Guadalupe_Northbound")
        model_key = f"{station}_{direction}"
        station_data_dict[model_key] = {
            'X_train': X[:split_idx],
            'X_val': X[split_idx:],
            'y_train': y[:split_idx],
            'y_val': y[split_idx:],
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'station_num': station_num,
            'n_records': len(dir_df),
            'n_sequences': len(X),
            'max_hourly': max_hourly   
        }
        
        print(f"    Train: {split_idx:,} | Val: {len(X) - split_idx:,}")

print(f"\nPre-processed {len(station_data_dict)} directional models")

# Save per-direction max passengers for consistent testing
per_direction_max = {key: data['max_hourly'] for key, data in station_data_dict.items()}
with open(f'{MODELS_PATH}/per_direction_max_passengers.pkl', 'wb') as f:
    pickle.dump(per_direction_max, f)
print(f"Saved per-direction max passengers for {len(per_direction_max)} models.")


del df
gc.collect()

# ========== TRAIN ALL DIRECTIONAL MODELS ==========
all_results = []
start_time = time.time()

for model_idx, (model_key, data) in enumerate(station_data_dict.items()):
    model_start = time.time()
    
    print(f"\n{'='*60}")
    print(f"Training: {model_key} ({model_idx+1}/{len(station_data_dict)})")
    print('='*60)
    
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    feature_scaler = data['feature_scaler']
    target_scaler = data['target_scaler']
    
    print(f"Train samples: {len(X_train):,} | Val samples: {len(X_val):,}")
    print(f"   Batch size: {BATCH_SIZE} -> {len(X_train)//BATCH_SIZE} steps/epoch")
    
    input_shape = (SEQ_LENGTH, len(feature_cols))
    model = build_lstm_model(input_shape)
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=PATIENCE_EARLY, 
        restore_best_weights=True, 
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=PATIENCE_LR, 
        min_lr=0.00001, 
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        f'{MODELS_PATH}/{model_key}_best.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
    
    print(f"\nTraining (batch_size={BATCH_SIZE}, epochs={EPOCHS})...")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )
    
    eval_metrics = evaluate_model(model, X_val, y_val, target_scaler, model_key)
    
    plot_predictions(model, X_val, y_val, target_scaler, model_key, MODELS_PATH)
    plot_training_history(history, model_key, MODELS_PATH)
    
    model.save(f'{MODELS_PATH}/{model_key}_lstm_enhanced.keras')
    print(f"Saved: {MODELS_PATH}/{model_key}_lstm_enhanced.keras")
    
    with open(f'{MODELS_PATH}/{model_key}_feature_scaler.pkl', 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open(f'{MODELS_PATH}/{model_key}_target_scaler.pkl', 'wb') as f:
        pickle.dump(target_scaler, f)
    
    model_time = time.time() - model_start
    all_results.append({
        'station': model_key,
        'epochs': len(history.history['loss']),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'mae': eval_metrics['mae'],
        'rmse': eval_metrics['rmse'],
        'mape': eval_metrics['mape'],
        'r2': eval_metrics['r2'],
        'time_min': model_time / 60
    })
    
    print(f"\n{model_key} complete in {model_time/60:.1f} min | MAE: {eval_metrics['mae']:.2f}%")
     
    if eval_metrics['mae'] > 5.0:
        print(f"⚠️ WARNING: {model_key} MAE = {eval_metrics['mae']:.2f}% (>5%)")
        print(f"   Consider retraining this specific station with more epochs")
    
    del model, history, X_train, X_val, y_train, y_val
    gc.collect()

# ========== FINAL SUMMARY ==========
total_time = time.time() - start_time

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"\nTotal training time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")

results_df = pd.DataFrame(all_results)
print("\nRESULTS SUMMARY (Per-Direction Models):")
print(results_df[['station', 'mae', 'rmse', 'mape', 'r2', 'time_min']].to_string(index=False))

results_df.to_csv(f'{MODELS_PATH}/training_summary_directional.csv', index=False)

plot_all_metrics(results_df, MODELS_PATH)

print("\nAGGREGATE STATISTICS:")
print(f"   Average MAE: {results_df['mae'].mean():.2f}%")
print(f"   Average RMSE: {results_df['rmse'].mean():.2f}%")
print(f"   Average MAPE: {results_df['mape'].mean():.2f}%")
print(f"   Average R2: {results_df['r2'].mean():.4f}")

# Create summary plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

ax1 = axes[0, 0]
stations_names = results_df['station'].tolist()
maes = results_df['mae'].tolist()
bars = ax1.bar(stations_names, maes, color='skyblue')
ax1.axhline(y=np.mean(maes), color='r', linestyle='--', label=f'Avg: {np.mean(maes):.1f}%')
ax1.set_ylabel('MAE (%)')
ax1.set_title('Mean Absolute Error by Model')
ax1.set_xticklabels(stations_names, rotation=45, ha='right')
ax1.legend()
for bar, mae in zip(bars, maes):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{mae:.1f}%', ha='center', va='bottom', fontsize=8)

ax2 = axes[0, 1]
r2_scores = results_df['r2'].tolist()
bars = ax2.bar(stations_names, r2_scores, color='lightgreen')
ax2.axhline(y=np.mean(r2_scores), color='red', linestyle='--', label=f'Avg: {np.mean(r2_scores):.3f}')
ax2.set_ylabel('R2 Score')
ax2.set_title('R2 by Model')
ax2.set_xticklabels(stations_names, rotation=45, ha='right')
ax2.legend()
for bar, r2 in zip(bars, r2_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{r2:.3f}', ha='center', va='bottom', fontsize=8)

ax3 = axes[1, 0]
times = results_df['time_min'].tolist()
bars = ax3.bar(stations_names, times, color='orange')
ax3.set_ylabel('Time (minutes)')
ax3.set_title('Training Time by Model')
ax3.set_xticklabels(stations_names, rotation=45, ha='right')
for bar, t in zip(bars, times):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{t:.1f}m', ha='center', va='bottom', fontsize=8)

ax4 = axes[1, 1]
epochs = results_df['epochs'].tolist()
bars = ax4.bar(stations_names, epochs, color='purple')
ax4.axhline(y=EPOCHS, color='gray', linestyle='--', label=f'Max: {EPOCHS}')
ax4.set_ylabel('Epochs')
ax4.set_title('Training Epochs Used (Early Stopping)')
ax4.set_xticklabels(stations_names, rotation=45, ha='right')
ax4.legend()
for bar, e in zip(bars, epochs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{e}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(f'{MODELS_PATH}/directional_models_comparison.png', dpi=150)
plt.close()

print("\n" + "="*60)
print("TRAINING COMPLETE - PER DIRECTION MODELS")
print(f"Results saved to: {MODELS_PATH}/")
print("="*60)
print("\nSAVED FILES:")
for model_key in station_data_dict.keys():
    print(f"   - {model_key}_lstm_enhanced.keras")
    print(f"   - {model_key}_feature_scaler.pkl")
    print(f"   - {model_key}_target_scaler.pkl") 
