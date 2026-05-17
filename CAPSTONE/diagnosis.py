# TESTING SCRIPT
"""

How to use?

--open 2025.csv(raw data from mrt)
--input STATION_NAME, DIRECTION, target-time

"""

import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
import os
from datetime import datetime, timedelta

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# ========== CONFIG ==========
MODEL_PATH = 'models_2023-2024'         
DATA_FOLDER = 'data (2022-2024)'
DATA_FILE = os.path.join(DATA_FOLDER, '2025.csv')

"""       MODIFY HERE   """
STATION_NAME = "North Ave"
DIRECTION = "Northbound"
target_time = pd.to_datetime('2025-09-29 15:00:00')
MODEL_KEY = f"{STATION_NAME}_{DIRECTION}"
SEQ_LENGTH = 24

STATION_NUMBERS = {
    "North Ave": 1, "Quezon Ave": 2, "Kamuning": 3, "Cubao": 4,
    "Santolan": 5, "Ortigas": 6, "Shaw Blvd": 7, "Boni Ave": 8,
    "Guadalupe": 9, "Buendia": 10, "Ayala Ave": 11, "Magallanes": 12,
    "Taft": 13
}

# ========== FEATURE ENGINEERING FUNCTIONS (same as training) ==========
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
    df['time_normalized'] = ((time_decimal - 4.5) / (23.0 - 4.5)).clip(0, 1)
    return df

def smart_data_cleaner(df):
    time_decimal = df['time_decimal']
    passenger_count = df['TotalPassenger']
    df['is_maintenance_record'] = ((time_decimal < 5.0) & (passenger_count < 10)).astype(np.int8)
    df['is_extended_hours'] = ((time_decimal >= 22.0) & (time_decimal < 23.0) & (passenger_count >= 10)).astype(np.int8)
    df.loc[df['is_maintenance_record'] == 1, 'congestion'] = 0
    return df

def is_christmas_season(date):
    month_day = date.strftime('%m-%d')
    return (month_day >= '12-15') or (month_day <= '01-05')

def is_payday(date):
    return date.day in [15, 30, 31]

def is_friday(date):
    return date.weekday() == 4

def infer_direction(row):
    entry = row['StationEntry']
    exit_st = row['StationExit']
    if entry < exit_st:
        return 'Southbound'
    elif entry > exit_st:
        return 'Northbound'
    else:
        return 'Unknown'

# ========== LOAD RAW DATA ==========
print("="*60)
print(" DIAGNOSIS SCRIPT - 2023-2024 MODELS (consistent scaling)")
print("="*60)
print(f" Loading: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
print(f" Loaded {len(df):,} rows, {df['datetime'].min()} to {df['datetime'].max()}")

# ========== CREATE ALL FEATURES ==========
print("\n Engineering features...")
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day

df = add_cyclical_time_features(df)
df = add_smart_operating_flags(df)

df['is_weekend'] = (df['datetime'].dt.weekday >= 5).astype(np.int8)
df['is_holiday'] = 0
df['is_special_event'] = 0
df['is_christmas_season'] = df['datetime'].apply(is_christmas_season).astype(np.int8)
df['is_payday'] = df['datetime'].apply(is_payday).astype(np.int8)
df['is_friday'] = df['datetime'].apply(is_friday).astype(np.int8)
df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(np.int8)
df['Direction'] = df.apply(infer_direction, axis=1)

# Initial congestion (will be replaced per station)
max_passengers = df['TotalPassenger'].quantile(0.99)
df['congestion'] = (df['TotalPassenger'] / max_passengers * 100).clip(0, 100)
df = smart_data_cleaner(df)

print(f" Features created: {len(df.columns)} columns")

# ========== LOAD MODEL ==========
print(f"\n Loading model: {MODEL_KEY}...")
print(f"   From: {MODEL_PATH}")

model = tf.keras.models.load_model(
    f'{MODEL_PATH}/{MODEL_KEY}_lstm_enhanced.keras',
    custom_objects={'rmse': rmse}
)
with open(f'{MODEL_PATH}/{MODEL_KEY}_feature_scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)
with open(f'{MODEL_PATH}/{MODEL_KEY}_target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)
print(" Model loaded")

# ========== LOAD PER-DIRECTION MAX PASSENGERS ==========
with open(f'{MODEL_PATH}/per_direction_max_passengers.pkl', 'rb') as f:
    per_direction_max = pickle.load(f)
print(f" Loaded per-direction max passengers for {len(per_direction_max)} models.")

# ========== FILTER & AGGREGATE ==========
station_num = STATION_NUMBERS[STATION_NAME]
print(f"\n Filtering: {STATION_NAME} (Station #{station_num}) {DIRECTION}")

station_df = df[(df['StationEntry'] == station_num) | (df['StationExit'] == station_num)]
station_df = station_df[station_df['Direction'] == DIRECTION].sort_values('datetime')

station_df['hour_timestamp'] = station_df['datetime'].dt.floor('h')
station_df = station_df.groupby('hour_timestamp').agg({
    'TotalPassenger': 'sum',
    'hour': 'first', 'weekday': 'first', 'month': 'first',
    'hour_sin': 'first', 'hour_cos': 'first',
    'dow_sin': 'first', 'dow_cos': 'first',
    'month_sin': 'first', 'month_cos': 'first',
    'time_decimal': 'first',
    'is_operating_hour': 'first',
    'minute_normalized': 'first',
    'is_morning_rush': 'first', 'is_evening_rush': 'first', 'is_noon': 'first',
    'is_pre_opening': 'first', 'is_post_closing': 'first',
    'minutes_until_closing': 'first', 'minutes_since_opening': 'first',
    'time_normalized': 'first',
    'is_weekend': 'first', 'is_holiday': 'first', 'is_special_event': 'first',
    'is_christmas_season': 'first', 'is_payday': 'first', 'is_friday': 'first',
    'is_rush_hour': 'first',
    'is_maintenance_record': 'first', 'is_extended_hours': 'first'
}).reset_index()

# Use the saved per-direction max for consistent scaling
key = f"{STATION_NAME}_{DIRECTION}"
global_max = per_direction_max[key]
station_df['congestion'] = (station_df['TotalPassenger'] / global_max * 100).clip(0, 100)

print(f" {len(station_df)} hourly records (using training max: {global_max:.0f})")

# ========== TARGET TIME ==========
actual_row = station_df[station_df['hour_timestamp'] == target_time]

if actual_row.empty:
    print(f"\n❌ No data for {target_time}")
    print(f"Available: {station_df['hour_timestamp'].min()} to {station_df['hour_timestamp'].max()}")
    exit()

actual_congestion = actual_row['congestion'].iloc[0]
actual_passengers = actual_row['TotalPassenger'].iloc[0]

# ========== GET 24-HOUR HISTORY ==========
idx = station_df[station_df['hour_timestamp'] == target_time].index[0]
if idx < SEQ_LENGTH:
    print(f"❌ Need {SEQ_LENGTH} hours history, only have {idx}")
    exit()

history_df = station_df.iloc[idx-SEQ_LENGTH:idx]

print("\n" + "="*60)
print(f" {STATION_NAME} {DIRECTION} — {target_time}")
print("="*60)
print(f"Last 6 hours:")
for _, row in history_df.tail(6).iterrows():
    print(f"  {row['hour_timestamp']}: {int(row['TotalPassenger']):4d} pax ({row['congestion']:.1f}%)")
print(f"\n Target: {actual_passengers} pax ({actual_congestion:.1f}%)")

# ========== PREDICT ==========
FEATURE_COLS = [
    'hour', 'weekday', 'month',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'is_operating_hour', 'is_morning_rush', 'is_evening_rush', 'is_noon',
    'is_pre_opening', 'is_post_closing',
    'minutes_until_closing', 'minutes_since_opening', 'time_normalized', 'minute_normalized',
    'is_weekend', 'is_holiday', 'is_special_event', 'is_christmas_season', 'is_payday', 'is_friday',
    'is_rush_hour', 'is_maintenance_record', 'is_extended_hours', 'congestion'
]

features = history_df[FEATURE_COLS].values.astype(np.float32)
features_scaled = feature_scaler.transform(features)
features_scaled = features_scaled.reshape(1, SEQ_LENGTH, len(FEATURE_COLS))

pred_scaled = model.predict(features_scaled, verbose=0)
pred_congestion = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
pred_congestion = np.clip(pred_congestion, 0, 100)

# ========== RESULTS ==========
diff = abs(actual_congestion - pred_congestion)
error_pct = diff / max(actual_congestion, 0.1) * 100

print(f"\n{'='*60}")
print(f" RESULT: {MODEL_KEY} (2023-2024 Model)")
print(f"{'='*60}")
print(f"  Predicted: {pred_congestion:.1f}%")
print(f"  Actual:    {actual_congestion:.1f}%")
print(f"  Error:     {diff:.1f}% ({error_pct:.1f}%)")

if error_pct < 10:
    print(f"\n EXCELLENT")
elif error_pct < 20:
    print(f"\n GOOD!")
elif error_pct < 30:
    print(f"\n OKAY")
else:
    print(f"\n Check model - error is high")

print(f" Model error: {error_pct:.1f}%")
print("\n" + "="*60)
print(" DIAGNOSIS COMPLETE")
print("="*60)