# FORDA TESTING NG 2025 raw data - MRT-3 VERSION
# UPDATED to match model filenames

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
        'is_holiday': 1 if dt.strftime('%Y-%m-%d') in ['2025-01-01', '2025-04-09', '2025-05-01', '2025-06-12', '2025-08-21', '2025-11-30', '2025-12-25', '2025-12-30'] else 0,
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
# Use the EXACT names from your model files
MODEL_STATION_NAME = "Shaw Blvd"  # Match your model filename (Shaw Blvd_lstm_enhanced.keras)
TEST_DATE = "2025-01-15"
TEST_TIME = "08:00:00"
MODELS_PATH = 'models_2022-2024'

# Mapping between display names and model names
STATION_MAPPING = {
    "North Avenue": "North Ave",
    "Quezon Avenue": "Quezon Ave", 
    "GMA-Kamuning": "Kamuning",
    "Araneta Center-Cubao": "Cubao",
    "Santolan-Annapolis": "Santolan",
    "Ortigas": "Ortigas",
    "Shaw Boulevard": "Shaw Blvd",
    "Boni": "Boni Ave",
    "Guadalupe": "Guadalupe",
    "Buendia": "Buendia",
    "Ayala": "Ayala Ave",
    "Magallanes": "Magallanes",
    "Taft Avenue": "Taft"
}

# Reverse mapping for display
DISPLAY_NAMES = {v: k for k, v in STATION_MAPPING.items()}

# Station numbers for direction calculation
STATION_NUMBERS = {
    "North Ave": 1,
    "Quezon Ave": 2, 
    "Kamuning": 3,
    "Cubao": 4,
    "Santolan": 5,
    "Ortigas": 6,
    "Shaw Blvd": 7,
    "Boni Ave": 8,
    "Guadalupe": 9,
    "Buendia": 10,
    "Ayala Ave": 11,
    "Magallanes": 12,
    "Taft": 13
}

# Get display name for printing
DISPLAY_STATION_NAME = DISPLAY_NAMES.get(MODEL_STATION_NAME, MODEL_STATION_NAME)

print("="*70)
print(f"COMPLETE TEST: {DISPLAY_STATION_NAME} on REAL 2025 MRT-3 Data")
print("="*70)

# ========== LOAD DATA ==========
# Load your newly created MRT-3 file
df = pd.read_csv('2025_mrt3_complete.csv')
print(f"✅ Loaded MRT-3 data with {len(df):,} records")

# Create datetime column
df['datetime'] = pd.to_datetime(df['Full_Timestamp'])

# Add direction code (based on station numbers)
def get_direction(row):
    entry = row['StationEntry']
    exit_station = row['StationExit']
    if entry < exit_station:
        return 1  # Northbound
    elif entry > exit_station:
        return 0  # Southbound
    else:
        return 0.5  # Same station
df['direction_code'] = df.apply(get_direction, axis=1)

# Calculate congestion using 99th percentile as max
max_passengers = df['TotalPassenger'].quantile(0.99)
df['congestion'] = (df['TotalPassenger'] / max_passengers * 100).clip(0, 100)

print(f"📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"📊 Max passengers (99th percentile): {max_passengers:.0f}")

# ========== FILTER FOR SPECIFIC STATION ==========
station_num = STATION_NUMBERS[MODEL_STATION_NAME]
target_datetime = pd.to_datetime(f"{TEST_DATE} {TEST_TIME}")

# Filter by entry station (where people board)
station_df = df[df['StationEntry'] == station_num].copy()
station_df = station_df.sort_values('datetime')

print(f"\n📊 {DISPLAY_STATION_NAME} (Station {station_num}) has {len(station_df)} records")

# Find the actual row
actual_row = station_df[station_df['datetime'] == target_datetime]

if actual_row.empty:
    print(f"\n❌ {target_datetime} not found for {DISPLAY_STATION_NAME}")
    print(f"\nAvailable times on {TEST_DATE}:")
    times = station_df[station_df['datetime'].dt.date == pd.to_datetime(TEST_DATE).date()]['datetime'].dt.strftime('%H:%M').unique()
    print(f"   {sorted(times)[:10]}")
    exit()

# ========== GET ACTUAL CONGESTION ==========
actual_passengers = actual_row['TotalPassenger'].iloc[0]
actual_congestion = actual_row['congestion'].iloc[0]

print(f"\n✅ Found data for {target_datetime}")
print(f"\n📊 REAL 2025 MRT-3 DATA:")
print(f"   Station: {DISPLAY_STATION_NAME}")
print(f"   Date/Time: {target_datetime}")
print(f"   Total Passengers: {actual_passengers:,}")
print(f"   Actual Congestion: {actual_congestion:.1f}%")

# ========== LOAD MODEL ==========
print(f"\n📂 Loading trained model for {MODEL_STATION_NAME}...")

model_path = f'{MODELS_PATH}/{MODEL_STATION_NAME}_lstm_enhanced.keras'
if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("\nAvailable models:")
    for f in sorted(os.listdir(MODELS_PATH)):
        if f.endswith('.keras'):
            print(f"   - {f}")
    exit()

model = tf.keras.models.load_model(
    model_path,
    custom_objects={'rmse': rmse, 'mape': mape}
)

# Load scalers
feature_scaler_path = f'{MODELS_PATH}/{MODEL_STATION_NAME}_feature_scaler.pkl'
target_scaler_path = f'{MODELS_PATH}/{MODEL_STATION_NAME}_target_scaler.pkl'

if not os.path.exists(feature_scaler_path):
    print(f"❌ Feature scaler not found: {feature_scaler_path}")
    exit()
if not os.path.exists(target_scaler_path):
    print(f"❌ Target scaler not found: {target_scaler_path}")
    exit()

with open(feature_scaler_path, 'rb') as f:
    feature_scaler = pickle.load(f)
with open(target_scaler_path, 'rb') as f:
    target_scaler = pickle.load(f)

print("✅ Model and scalers loaded")

# ========== GET 24 HOURS OF HISTORY ==========
idx = station_df[station_df['datetime'] == target_datetime].index[0]

if idx < 24:
    print(f"\n❌ Not enough history data (need 24 hours before)")
    print(f"   Only have {idx} records before this time")
    exit()

# Get previous 24 hours
history_df = station_df.iloc[idx-24:idx]

print(f"\n📊 Using {len(history_df)} hours of history for prediction")
print(f"   History range: {history_df['datetime'].min()} to {history_df['datetime'].max()}")

# ========== PREPARE FEATURES ==========
# Extract features for each hour in history
features_list = []
for _, hist_row in history_df.iterrows():
    # Create a copy with datetime
    hist_row_copy = hist_row.copy()
    hist_row_copy['datetime'] = hist_row['datetime']
    feat = extract_features(hist_row_copy)
    features_list.append(feat[0])

features_sequence = np.stack(features_list, axis=0)
features_scaled = feature_scaler.transform(features_sequence)
features_scaled = features_scaled.reshape(1, 24, 9)

# ========== PREDICT ==========
pred_scaled = model.predict(features_scaled, verbose=0)
pred_congestion = target_scaler.inverse_transform(pred_scaled)[0][0]
pred_congestion = np.clip(pred_congestion, 0, 100)  # Cap at 100

# ========== RESULTS ==========
print("\n" + "="*60)
print("📊 ACTUAL vs PREDICTED on REAL 2025 MRT-3 DATA")
print("="*60)
print(f"\n📍 Station: {DISPLAY_STATION_NAME}")
print(f"📅 Date/Time: {target_datetime}")
print("\n" + "="*50)
print(f"  🟢 ACTUAL Congestion:    {actual_congestion:.1f}%")
print(f"  🔵 PREDICTED Congestion: {pred_congestion:.1f}%")
print(f"  📉 DIFFERENCE:           {abs(actual_congestion - pred_congestion):.1f}%")
if actual_congestion > 0:
    error_pct = abs(actual_congestion - pred_congestion) / actual_congestion * 100
    print(f"  📊 PERCENTAGE ERROR:     {error_pct:.1f}%")
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
    print(f"   ⚠️  MISMATCH - Model predicted different advisory level")

# ========== SAVE RESULT ==========
result = {
    'station_display': DISPLAY_STATION_NAME,
    'station_model': MODEL_STATION_NAME,
    'station_number': station_num,
    'test_date': TEST_DATE,
    'test_time': TEST_TIME,
    'actual_passengers': actual_passengers,
    'actual_congestion': actual_congestion,
    'predicted_congestion': pred_congestion,
    'difference': abs(actual_congestion - pred_congestion),
    'error_percent': error_pct if actual_congestion > 0 else None,
    'actual_advisory': actual_adv,
    'predicted_advisory': pred_adv,
    'advisory_match': actual_adv == pred_adv
}

result_df = pd.DataFrame([result])
os.makedirs('test_results', exist_ok=True)
result_df.to_csv(f'test_results/{MODEL_STATION_NAME}_2025_test_result.csv', index=False)
print(f"\n✅ Result saved to: test_results/{MODEL_STATION_NAME}_2025_test_result.csv")

print("\n" + "="*60)
print("✅ TEST COMPLETE")
print("="*60)

# ========== OPTIONAL: TEST ALL STATIONS ==========
print("\n" + "="*60)
print("TEST ALL AVAILABLE STATIONS?")
print("="*60)
run_all = input("Run test for all stations? (y/n): ").lower().strip()

if run_all == 'y':
    print("\n" + "="*60)
    print("TESTING ALL STATIONS...")
    print("="*60)
    
    all_results = []
    
    for model_name in STATION_NUMBERS.keys():
        display_name = DISPLAY_NAMES.get(model_name, model_name)
        print(f"\n--- Testing {display_name}...")
        
        model_path = f'{MODELS_PATH}/{model_name}_lstm_enhanced.keras'
        if not os.path.exists(model_path):
            print(f"   ⚠️  Model not found, skipping")
            continue
        
        # Filter data for this station
        station_num_test = STATION_NUMBERS[model_name]
        station_df_test = df[df['StationEntry'] == station_num_test].copy()
        station_df_test = station_df_test.sort_values('datetime')
        
        actual_row_test = station_df_test[station_df_test['datetime'] == target_datetime]
        
        if actual_row_test.empty:
            print(f"   ⚠️  No data for {target_datetime}, skipping")
            continue
        
        actual_pass_test = actual_row_test['TotalPassenger'].iloc[0]
        actual_cong_test = actual_row_test['congestion'].iloc[0]
        
        print(f"   Actual: {actual_cong_test:.1f}% ({actual_pass_test} passengers)")
        
        # Quick prediction without full history (simplified)
        # For full test, you'd need to implement the same prediction logic
        all_results.append({
            'station': display_name,
            'actual_congestion': actual_cong_test,
            'actual_passengers': actual_pass_test
        })
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('test_results/all_stations_summary.csv', index=False)
    print("\n✅ Summary saved to: test_results/all_stations_summary.csv")