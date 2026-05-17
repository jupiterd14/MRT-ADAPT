# FORDA TESTING NG 2025 raw data

import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
import os

# ========== DEFINE CUSTOM METRICS ==========
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def mape(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + 0.01))) * 100

# ========== ADVISORY FUNCTION ==========
def get_advisory(congestion):
    if congestion < 20:
        return "🟢 LIGHT"
    elif congestion < 50:
        return "🟡 MODERATE"
    elif congestion < 75:
        return "🟠 HEAVY"
    else:
        return "🔴 SEVERE"

# ========== FEATURE EXTRACTION FUNCTION ==========
def extract_features(row):
    """Extract features from a row of data (same as training)"""
    dt = row['datetime']
    
    features = {
        'hour': dt.hour,
        'weekday': dt.weekday(),
        'month': dt.month,
        'is_weekend': 1 if dt.weekday() >= 5 else 0,
        'is_holiday': 1 if dt.strftime('%Y-%m-%d') in ['2025-01-01'] else 0,  # Add holidays
        'is_special_event': 0,
        'is_christmas_season': 1 if (dt.month == 12 and dt.day >= 15) or (dt.month == 1 and dt.day <= 5) else 0,
        'direction_code': row.get('direction_code', 0.5),
        'is_rush_hour': 1 if (7 <= dt.hour <= 9) or (17 <= dt.hour <= 19) else 0,
    }
    
    return np.array([[features['hour'], features['weekday'], features['month'], 
                      features['is_weekend'], features['is_holiday'], 
                      features['is_special_event'], features['is_christmas_season'],
                      features['direction_code'], features['is_rush_hour']]], dtype=np.float32)

# ========== CONFIGURATION ==========
STATION = "Shaw Blvd"
TEST_DATE = "2025-01-15"
TEST_TIME = "08:00:00"
MODELS_PATH = 'models_2022-2024'
DATA_FOLDER = 'data (2022-2024)'

STATION_NUMBERS = {
    "North Ave": 1, "Quezon Ave": 2, "Kamuning": 3, "Cubao": 4,
    "Santolan": 5, "Ortigas": 6, "Shaw Blvd": 7, "Boni Ave": 8,
    "Guadalupe": 9, "Buendia": 10, "Ayala Ave": 11, "Magallanes": 12,
    "Taft": 13
}

print("="*70)
print(f"COMPLETE TEST: {STATION} on REAL 2025 Data")
print("="*70)

# ========== LOAD DATA ==========
df = pd.read_csv(f'{DATA_FOLDER}/2025_sorted.csv')
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# Add direction code
def get_direction(row):
    entry = row['StationEntry']
    exit_station = row['StationExit']
    if entry < exit_station:
        return 1
    elif entry > exit_station:
        return 0
    else:
        return 0.5
df['direction_code'] = df.apply(get_direction, axis=1)

# Calculate congestion using same method as training
max_passengers = df['TotalPassenger'].quantile(0.99)
df['congestion'] = (df['TotalPassenger'] / max_passengers * 100).clip(0, 100)

print(f"✅ Loaded {len(df):,} records")
print(f"📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# ========== FILTER FOR SPECIFIC STATION AND TIME ==========
station_num = STATION_NUMBERS[STATION]
target_datetime = pd.to_datetime(f"{TEST_DATE} {TEST_TIME}")

station_df = df[df['StationEntry'] == station_num].copy()
station_df = station_df.sort_values('datetime')

actual_row = station_df[station_df['datetime'] == target_datetime]

if actual_row.empty:
    print(f"\n❌ {target_datetime} not found for {STATION}")
    print(f"\nAvailable times on {TEST_DATE}:")
    times = station_df[station_df['datetime'].dt.date == pd.to_datetime(TEST_DATE).date()]['Time'].unique()
    print(f"   {sorted(times)}")
    exit()

# ========== GET ACTUAL CONGESTION ==========
actual_passengers = actual_row['TotalPassenger'].iloc[0]
actual_congestion = actual_row['congestion'].iloc[0]

print(f"\n✅ Found data for {target_datetime}")
print(f"\n📊 REAL 2025 DATA:")
print(f"   Station: {STATION}")
print(f"   Date/Time: {target_datetime}")
print(f"   Total Passengers: {actual_passengers}")
print(f"   Actual Congestion: {actual_congestion:.1f}%")

# ========== LOAD MODEL ==========
print(f"\n📂 Loading trained model...")

model = tf.keras.models.load_model(
    f'{MODELS_PATH}/{STATION}_lstm_enhanced.keras',
    custom_objects={'rmse': rmse, 'mape': mape}
)

with open(f'{MODELS_PATH}/{STATION}_feature_scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)
with open(f'{MODELS_PATH}/{STATION}_target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)

print("✅ Model loaded")

# ========== GET 24 HOURS OF HISTORY ==========
idx = station_df[station_df['datetime'] == target_datetime].index[0]

if idx < 24:
    print(f"\n❌ Not enough history data (need 24 hours before)")
    print(f"   Only have {idx} records before this time")
    exit()

# Get previous 24 hours
history_df = station_df.iloc[idx-24:idx]

print(f"\n📊 Using {len(history_df)} hours of history for prediction")

# ========== PREPARE FEATURES ==========
feature_cols = ['hour', 'weekday', 'month', 'is_weekend', 'is_holiday',
                'is_special_event', 'is_christmas_season', 'direction_code']

# Extract features for each hour in history
features_list = []
for _, hist_row in history_df.iterrows():
    hist_row['datetime'] = hist_row['datetime']
    feat = extract_features(hist_row)
    features_list.append(feat[0])

features_sequence = np.stack(features_list, axis=0)
features_scaled = feature_scaler.transform(features_sequence)
features_scaled = features_scaled.reshape(1, 24, 9)

# ========== PREDICT ==========
pred_scaled = model.predict(features_scaled, verbose=0)
pred_congestion = target_scaler.inverse_transform(pred_scaled)[0][0]

# ========== RESULTS ==========
print("\n" + "="*60)
print("📊 ACTUAL vs PREDICTED on REAL 2025 DATA")
print("="*60)
print(f"\n📍 Station: {STATION}")
print(f"📅 Date/Time: {target_datetime}")
print("\n" + "="*50)
print(f"  🟢 ACTUAL Congestion:    {actual_congestion:.1f}%")
print(f"  🔵 PREDICTED Congestion: {pred_congestion:.1f}%")
print(f"  📉 DIFFERENCE:           {abs(actual_congestion - pred_congestion):.1f}%")
if actual_congestion > 0:
    print(f"  📊 PERCENTAGE ERROR:     {abs(actual_congestion - pred_congestion) / actual_congestion * 100:.1f}%")
print("="*50)

# ========== ADVISORY COMPARISON ==========
actual_adv = get_advisory(actual_congestion)
pred_adv = get_advisory(pred_congestion)

print(f"\n🚆 ADVISORY COMPARISON (What Commuters See):")
print(f"   ACTUAL Advisory:    {actual_adv}")
print(f"   PREDICTED Advisory: {pred_adv}")
if actual_adv == pred_adv:
    print(f"   ✅ MATCH! The advisory is correct!")
else:
    print(f"   ❌ MISMATCH - Model predicted different advisory level")

# ========== INTERPRETATION ==========
print(f"\n💡 INTERPRETATION:")
print(f"   On New Year's Day ({TEST_DATE}) at 8:00 AM, {STATION} had very light traffic")
print(f"   Only {actual_passengers} passengers (0.8% congestion)")
print(f"   The model predicted {pred_congestion:.1f}% which is {'accurate' if abs(actual_congestion - pred_congestion) < 5 else 'off'}")

# ========== SAVE RESULT ==========
result = {
    'station': STATION,
    'test_date': TEST_DATE,
    'test_time': TEST_TIME,
    'actual_passengers': actual_passengers,
    'actual_congestion': actual_congestion,
    'predicted_congestion': pred_congestion,
    'difference': abs(actual_congestion - pred_congestion),
    'error_percent': abs(actual_congestion - pred_congestion) / actual_congestion * 100 if actual_congestion > 0 else None,
    'actual_advisory': actual_adv,
    'predicted_advisory': pred_adv,
    'advisory_match': actual_adv == pred_adv
}

result_df = pd.DataFrame([result])
result_df.to_csv(f'{MODELS_PATH}/{STATION}_2025_test_result.csv', index=False)
print(f"\n✅ Result saved to: {MODELS_PATH}/{STATION}_2025_test_result.csv")

print("\n" + "="*60)
print("✅ TEST COMPLETE")
print("="*60)