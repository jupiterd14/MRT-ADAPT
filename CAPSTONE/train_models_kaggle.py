# ============================================
# KAGGLE NOTEBOOK - RUN YOUR EXACT TRAINING SCRIPT
# ============================================

import tensorflow as tf
import os
import gc

print("="*60)
print("KAGGLE GPU CHECK")
print("="*60)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU FOUND: {gpus[0]}")
    print(f"   GPU Name: {tf.test.gpu_device_name()}")
    print(f"   Training will be FAST (3-5 hours)")
else:
    print("❌ NO GPU DETECTED!")
    print("   Go to: Settings → Accelerator → GPU T4 x2")
    print("   Then restart this notebook")
    print("="*60)
    
print("="*60)

# Create data folder
os.makedirs("data (2022-2024)", exist_ok=True)

print("\n" + "="*60)
print("HOW TO ADD YOUR CSV FILES:")
print("="*60)
print("1. Click the 'Add Data' button on the RIGHT side of this notebook")
print("2. Click 'Upload'")
print("3. Upload your 3 CSV files:")
print("   - 2022.csv")
print("   - 2023.csv")  
print("   - 2024.csv")
print("4. Wait for upload to complete")
print("5. Run this cell again")
print("="*60)

# Check for uploaded files
print("\n📁 Scanning for uploaded CSV files...")

# Check multiple possible locations
found_files = []
possible_paths = ['/kaggle/input', '/kaggle/working', '.']

for path in possible_paths:
    if os.path.exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                for f in os.listdir(item_path):
                    if f.endswith('.csv'):
                        src = os.path.join(item_path, f)
                        dst = f"data (2022-2024)/{f}"
                        !cp "{src}" "{dst}"
                        found_files.append(f)
                        print(f"  ✓ Found and copied: {f}")
            elif item.endswith('.csv'):
                src = item_path
                dst = f"data (2022-2024)/{item}"
                !cp "{src}" "{dst}"
                found_files.append(item)
                print(f"  ✓ Found and copied: {item}")

# Also check current directory
for f in os.listdir('.'):
    if f.endswith('.csv'):
        src = f
        dst = f"data (2022-2024)/{f}"
        !cp "{src}" "{dst}"
        if f not in found_files:
            found_files.append(f)
            print(f"  ✓ Found and copied: {f}")

print(f"\n📁 Files in data folder: {os.listdir('data (2022-2024)')}")

required_files = ['2022.csv', '2023.csv', '2024.csv']
missing = [f for f in required_files if f not in os.listdir('data (2022-2024)')]

if missing:
    print(f"\n⚠️ MISSING FILES: {missing}")
    print("Please upload them using the 'Add Data' button")
else:
    print("\n✅ ALL FILES READY! Starting training...")

# ============================================
# CREATE YOUR EXACT TRAINING SCRIPT
# ============================================

script_content = r'''#train_local_full_optimized.py 
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

# ========== SETUP DATA PATH ==========
# Auto-detect the correct data folder
data_folder = None
files = ['2022.csv', '2023.csv', '2024.csv']
MODELS_PATH = 'models_2022-2024_v4'

print("\n" + "="*60)
print("SETTING UP DATA PATH")
print("="*60)

# Check for files in Kaggle input first
if os.path.exists('/kaggle/input'):
    for root, dirs, filenames in os.walk('/kaggle/input'):
        for file in files:
            if file in filenames:
                data_folder = root
                print(f"✅ Found data in: {data_folder}")
                break
        if data_folder:
            break

# If not found in Kaggle input, use local folder
if not data_folder:
    data_folder = 'data (2022-2024)'
    print(f"📁 Using local data folder: {data_folder}")

# Verify all files exist
print("\nVerifying files:")
for file in files:
    file_path = os.path.join(data_folder, file)
    if os.path.exists(file_path):
        print(f"  ✅ {file} found")
    else:
        print(f"  ❌ {file} NOT found at {file_path}")

# Create models directory
os.makedirs(MODELS_PATH, exist_ok=True)
print(f"\n📁 Models will be saved to: {MODELS_PATH}")

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
    entry = row['StationEntry']
    exit_station = row['StationExit']
    if entry < exit_station:  
        return 'Southbound'
    elif entry > exit_station:   
        return 'Northbound'
    else:
        return 'Unknown'

def smart_data_cleaner(df):
    time_decimal = df['time_decimal']
    passenger_count = df['TotalPassenger']
    df['is_maintenance_record'] = ((time_decimal < 5.0) & (passenger_count < 10)).astype(np.int8)
    df.loc[df['is_maintenance_record'] == 1, 'congestion'] = 0
    df['is_extended_hours'] = ((time_decimal >= 22.0) & (time_decimal < 23.0) & (passenger_count >= 10)).astype(np.int8)
    return df

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
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + epsilon)) * 100
    print(f"\n{model_name} Performance:")
    print(f"   MAE: {mae:.2f}% | RMSE: {rmse:.2f}% | MAPE: {mape:.2f}% | sMAPE: {smape:.2f}% | R2: {r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'smape': smape, 'r2': r2}

def plot_predictions(model, X_val, y_val, target_scaler, model_key, models_path):
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    stations = results_df['station'].tolist()
    axes[0,0].bar(stations, results_df['mae'], color='skyblue')
    axes[0,0].axhline(y=results_df['mae'].mean(), color='red', linestyle='--', label=f'Avg: {results_df["mae"].mean():.2f}%')
    axes[0,0].set_ylabel('MAE (%)')
    axes[0,0].set_title('1. Mean Absolute Error (Lower is Better)')
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,1].bar(stations, results_df['rmse'], color='lightgreen')
    axes[0,1].axhline(y=results_df['rmse'].mean(), color='red', linestyle='--', label=f'Avg: {results_df["rmse"].mean():.2f}%')
    axes[0,1].set_ylabel('RMSE (%)')
    axes[0,1].set_title('2. Root Mean Square Error (Lower is Better)')
    axes[0,1].legend()
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[1,0].bar(stations, results_df['mape'], color='orange')
    axes[1,0].axhline(y=results_df['mape'].mean(), color='red', linestyle='--', label=f'Avg: {results_df["mape"].mean():.2f}%')
    axes[1,0].set_ylabel('MAPE (%)')
    axes[1,0].set_title('3. Mean Absolute Percentage Error (Lower is Better)')
    axes[1,0].legend()
    axes[1,0].tick_params(axis='x', rotation=45)
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


print(f"\n📊 DATA VERIFICATION:")
print(f"   TotalPassenger range: {df['TotalPassenger'].min():.0f} - {df['TotalPassenger'].max():.0f}")
print(f"   Records with congestion=0: {(df['congestion'] == 0).sum():,} ({(df['congestion'] == 0).mean()*100:.1f}%)")


os.makedirs(MODELS_PATH, exist_ok=True)
print(f"Models will save to: {MODELS_PATH}")

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

print(f"\n{len(feature_cols)} ENHANCED FEATURES (Cyclical Time + MRT Schedule):")
for i, f in enumerate(feature_cols, 1):
    print(f"   {i}. {f}")

# ========== PRE-FILTER AND PRE-PROCESS ==========
print("\n" + "="*60)
print(f"PRE-PROCESSING {len(STATIONS)} STATIONS (Per-Direction Models)")
print("="*60)

station_data_dict = {}
scaler_type = RobustScaler if USE_ROBUST_SCALER else MinMaxScaler

for station_idx, station in enumerate(STATIONS):
    station_num = STATION_NUMBERS[station]
    print(f"\nProcessing {station} ({station_idx+1}/{len(STATIONS)})...")
    
    for direction in ['Northbound', 'Southbound']:
        if direction == 'Northbound':
            station_df = df[df['StationExit'] == station_num].copy()
            flow_type = "EXITS"
        else:
            station_df = df[df['StationEntry'] == station_num].copy()
            flow_type = "ENTRIES"
        
        if len(station_df) == 0:
            print(f"  [{direction}] No data")
            continue
        
        station_df['inferred_direction'] = station_df.apply(infer_direction, axis=1)
        station_df = station_df[station_df['inferred_direction'] == direction]
        
        if len(station_df) < SEQ_LENGTH + 10:
            print(f"  [{direction}] Insufficient data: {len(station_df)} records (need {SEQ_LENGTH + 10})")
            continue
        
        print(f"  [{direction}] Records: {len(station_df):,} ({flow_type})")
        
        station_df['hour_timestamp'] = station_df['datetime'].dt.floor('h')
        dir_df = station_df.groupby('hour_timestamp').agg({
            'TotalPassenger': 'sum',
            'hour': 'first', 'weekday': 'first', 'month': 'first',
            'is_weekend': 'first', 'is_holiday': 'first', 'is_special_event': 'first',
            'is_christmas_season': 'first', 'is_payday': 'first', 'is_friday': 'first',
            'is_rush_hour': 'first', 'hour_sin': 'first', 'hour_cos': 'first',
            'dow_sin': 'first', 'dow_cos': 'first', 'month_sin': 'first', 'month_cos': 'first',
            'is_operating_hour': 'first', 'is_morning_rush': 'first', 'is_evening_rush': 'first',
            'is_noon': 'first', 'is_pre_opening': 'first', 'is_post_closing': 'first',
            'minutes_until_closing': 'first', 'minutes_since_opening': 'first',
            'time_normalized': 'first', 'minute_normalized': 'first',
            'is_maintenance_record': 'first', 'is_extended_hours': 'first'
        }).reset_index()
        
        dir_df = dir_df.sort_values('hour_timestamp')
        max_hourly = dir_df['TotalPassenger'].quantile(0.99)
        if max_hourly == 0:
            max_hourly = 1
        dir_df['congestion'] = (dir_df['TotalPassenger'] / max_hourly * 100).clip(0, 100)
        
        # ========== CORRECT ORDER: Split BEFORE scaling ==========
        # 1. Split temporally (preserve time order)
        n = len(dir_df)
        train_size = int(n * 0.8)
        train_df = dir_df.iloc[:train_size].copy()
        val_df = dir_df.iloc[train_size:].copy()
        
        print(f"    Train period: {train_df['hour_timestamp'].min()} to {train_df['hour_timestamp'].max()}")
        print(f"    Val period: {val_df['hour_timestamp'].min()} to {val_df['hour_timestamp'].max()}")
        
        # 2. Fit scalers ONLY on training data
        feature_scaler = scaler_type()
        feature_scaler.fit(train_df[feature_cols])
        
        target_scaler = MinMaxScaler()
        target_scaler.fit(train_df[['congestion']])

        # 3. Transform features & targets
        train_features = feature_scaler.transform(train_df[feature_cols])
        train_targets = target_scaler.transform(train_df[['congestion']]).flatten()
        
        val_features = feature_scaler.transform(val_df[feature_cols])
        val_targets = target_scaler.transform(val_df[['congestion']]).flatten()
        
        # 4. Create sequences
        X_train, y_train = create_sequences(train_features, train_targets, seq_length=SEQ_LENGTH)
        X_val, y_val = create_sequences(val_features, val_targets, seq_length=SEQ_LENGTH)
        
        if len(X_train) == 0 or len(X_val) == 0:
            print(f"  [{direction}] Not enough sequences after split")
            continue
        
        model_key = f"{station}_{direction}"
        station_data_dict[model_key] = {
            'X_train': X_train, 'X_val': X_val,
            'y_train': y_train, 'y_val': y_val,
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'station_num': station_num,
            'n_records': len(dir_df),
            'n_sequences': len(X_train) + len(X_val),
            'max_hourly': max_hourly,
            'flow_type': flow_type
        }
        print(f"    Train sequences: {len(X_train):,} | Val sequences: {len(X_val):,}")
        print(f"    Max passengers (99th percentile): {max_hourly:.0f} {flow_type}/hour")

print(f"\nPre-processed {len(station_data_dict)} directional models")

per_direction_max = {key: data['max_hourly'] for key, data in station_data_dict.items()}
with open(f'{MODELS_PATH}/per_direction_max_passengers.pkl', 'wb') as f:
    pickle.dump(per_direction_max, f)
print(f"Saved per-direction max passengers for {len(per_direction_max)} models.")

del df
gc.collect()

all_results = []
start_time = time.time()

# ========== CRASH RECOVERY: CHECK EXISTING MODELS ==========
trained_models = []
if os.path.exists(MODELS_PATH):
    for f in os.listdir(MODELS_PATH):
        if f.endswith('_lstm_enhanced.keras'):
            model_name = f.replace('_lstm_enhanced.keras', '')
            trained_models.append(model_name)
            print(f"✓ Found already trained: {model_name}")
print(f"\n📊 Recovery: {len(trained_models)} models already trained\n")
# ========================================================

for model_idx, (model_key, data) in enumerate(station_data_dict.items()):
    # ========== SKIP IF ALREADY TRAINED ==========
    if model_key in trained_models:
        print(f"\n⏭️ SKIPPING: {model_key} ({model_idx+1}/{len(station_data_dict)}) - already trained")
        continue
        
    model_start = time.time()
    print(f"\n{'='*60}")
    print(f"Training: {model_key} ({model_idx+1}/{len(station_data_dict)})")
    print('='*60)
    
    input_shape = (SEQ_LENGTH, len(feature_cols))
    model = build_lstm_model(input_shape)
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE_EARLY, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE_LR, min_lr=0.00001, verbose=1)
    checkpoint = ModelCheckpoint(f'{MODELS_PATH}/{model_key}_best.keras', monitor='val_loss', save_best_only=True, verbose=0)
    
    print(f"Train samples: {len(data['X_train']):,} | Val samples: {len(data['X_val']):,}")
    print(f"Training (batch_size={BATCH_SIZE}, epochs={EPOCHS})...")
    
    history = model.fit(data['X_train'], data['y_train'], epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(data['X_val'], data['y_val']), callbacks=[early_stop, reduce_lr, checkpoint], verbose=1)
    
    eval_metrics = evaluate_model(model, data['X_val'], data['y_val'], data['target_scaler'], model_key)
    plot_predictions(model, data['X_val'], data['y_val'], data['target_scaler'], model_key, MODELS_PATH)
    plot_training_history(history, model_key, MODELS_PATH)
    
    model.save(f'{MODELS_PATH}/{model_key}_lstm_enhanced.keras')
    print(f"Saved: {MODELS_PATH}/{model_key}_lstm_enhanced.keras")
    
    with open(f'{MODELS_PATH}/{model_key}_feature_scaler.pkl', 'wb') as f:
        pickle.dump(data['feature_scaler'], f)
    with open(f'{MODELS_PATH}/{model_key}_target_scaler.pkl', 'wb') as f:
        pickle.dump(data['target_scaler'], f)
    
    model_time = time.time() - model_start
    all_results.append({'station': model_key, 'epochs': len(history.history['loss']),
                        'train_samples': len(data['X_train']), 'val_samples': len(data['X_val']),
                        'mae': eval_metrics['mae'], 'rmse': eval_metrics['rmse'],
                        'mape': eval_metrics['mape'], 'r2': eval_metrics['r2'],
                        'time_min': model_time / 60})
    print(f"\n{model_key} complete in {model_time/60:.1f} min | MAE: {eval_metrics['mae']:.2f}%")
    
    del model, history
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
stations_names = results_df['station'].tolist()
maes = results_df['mae'].tolist()
bars = axes[0,0].bar(stations_names, maes, color='skyblue')
axes[0,0].axhline(y=np.mean(maes), color='r', linestyle='--', label=f'Avg: {np.mean(maes):.1f}%')
axes[0,0].set_ylabel('MAE (%)')
axes[0,0].set_title('Mean Absolute Error by Model')
axes[0,0].set_xticklabels(stations_names, rotation=45, ha='right')
axes[0,0].legend()
for bar, mae in zip(bars, maes):
    axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{mae:.1f}%', ha='center', va='bottom', fontsize=8)

r2_scores = results_df['r2'].tolist()
bars = axes[0,1].bar(stations_names, r2_scores, color='lightgreen')
axes[0,1].axhline(y=np.mean(r2_scores), color='red', linestyle='--', label=f'Avg: {np.mean(r2_scores):.3f}')
axes[0,1].set_ylabel('R2 Score')
axes[0,1].set_title('R2 by Model')
axes[0,1].set_xticklabels(stations_names, rotation=45, ha='right')
axes[0,1].legend()
for bar, r2 in zip(bars, r2_scores):
    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{r2:.3f}', ha='center', va='bottom', fontsize=8)

times = results_df['time_min'].tolist()
bars = axes[1,0].bar(stations_names, times, color='orange')
axes[1,0].set_ylabel('Time (minutes)')
axes[1,0].set_title('Training Time by Model')
axes[1,0].set_xticklabels(stations_names, rotation=45, ha='right')
for bar, t in zip(bars, times):
    axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{t:.1f}m', ha='center', va='bottom', fontsize=8)

epochs = results_df['epochs'].tolist()
bars = axes[1,1].bar(stations_names, epochs, color='purple')
axes[1,1].axhline(y=EPOCHS, color='gray', linestyle='--', label=f'Max: {EPOCHS}')
axes[1,1].set_ylabel('Epochs')
axes[1,1].set_title('Training Epochs Used (Early Stopping)')
axes[1,1].set_xticklabels(stations_names, rotation=45, ha='right')
axes[1,1].legend()
for bar, e in zip(bars, epochs):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{e}', ha='center', va='bottom', fontsize=8)

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
'''

# Write and run
with open('train_local_full_optimized.py', 'w') as f:
    f.write(script_content)

print("\n" + "="*60)
print("STARTING TRAINING WITH YOUR EXACT SCRIPT")
print("="*60)
print("This will take 3-5 hours on GPU")
print("Keep this tab open")
print("="*60 + "\n")

!python train_local_full_optimized.py