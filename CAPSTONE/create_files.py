# create_pickle_files.py
# RUN THIS FIRST! It will create your pickle files.

import os
import numpy as np
import pandas as pd
import pickle
import glob
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🔨 CREATING PICKLE FILES FROM YOUR EXCEL DATA")
print("="*70)

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

# ========== COPY YOUR EXISTING LOADING FUNCTIONS HERE ==========
# Just copy these functions from your original app.py:

def load_hourly_data():
    """Load the Daily Hourly Excel files from Data folder"""
    all_data = []
    data_folder = 'Data'
    
    if not os.path.exists(data_folder):
        print(f"⚠️ Data folder not found")
        return None
    
    excel_files = sorted(glob.glob(os.path.join(data_folder, 'Daily Hourly *.xlsx')))
    print(f"\n📁 Found {len(excel_files)} hourly files")
    
    for file in excel_files:
        try:
            year = os.path.basename(file).replace('Daily Hourly ', '').replace('.xlsx', '')
            
            df = pd.read_excel(file, engine='openpyxl', header=None, skiprows=4)
            
            if df.shape[1] >= 14:
                df = df.iloc[:, :14]
                df.columns = ['timestamp'] + STATIONS
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                
                df['hour'] = df['timestamp'].dt.hour
                df['day'] = df['timestamp'].dt.day
                df['month'] = df['timestamp'].dt.month
                df['year'] = int(year)
                df['is_pandemic'] = (2020 <= int(year) <= 2022)
                df['source'] = f'hourly_{year}'
                
                for station in STATIONS:
                    if station in df.columns:
                        station_df = pd.DataFrame({
                            'timestamp': df['timestamp'],
                            'station': station,
                            'ridership': pd.to_numeric(df[station], errors='coerce'),
                            'hour': df['hour'],
                            'day': df['day'],
                            'month': df['month'],
                            'year': df['year'],
                            'is_pandemic': df['is_pandemic'],
                            'source': df['source']
                        })
                        station_df = station_df.dropna(subset=['ridership'])
                        if len(station_df) > 0:
                            all_data.append(station_df)
                            print(f"   ✅ {year} - {station}: {len(station_df)} rows")
        except Exception as e:
            print(f"   ❌ Error loading {file}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

def load_reports_data():
    """Load the comprehensive data from Ms. Mica Natividad.xlsx"""
    all_data = []
    reports_file = os.path.join('Data Reports', 'Ms. Mica Natividad.xlsx')
    
    if not os.path.exists(reports_file):
        print(f"\n📁 Reports file not found")
        return None
    
    print(f"\n📁 Loading reports file...")
    
    try:
        xl = pd.ExcelFile(reports_file, engine='openpyxl')
        sheet_names = xl.sheet_names
        print(f"   Sheets: {sheet_names}")
        
        # Sheet 1: 2021-2025 Hourly Data
        if '2021-2025' in sheet_names:
            print(f"   Processing hourly data 2021-2025...")
            df_hourly = pd.read_excel(reports_file, sheet_name='2021-2025', engine='openpyxl', header=None)
            
            current_year = None
            for idx, row in df_hourly.iterrows():
                row_str = str(row.iloc[0])
                
                if row_str in ['2021', '2022', '2023', '2024', '2025']:
                    current_year = int(row_str)
                    continue
                
                if ':' in row_str and '-' in row_str:
                    time_str = row_str.split('-')[0].strip()
                    try:
                        hour = int(time_str.split(':')[0])
                        
                        for station_idx, station in enumerate(STATIONS):
                            if station_idx + 1 < len(row):
                                value = row.iloc[station_idx + 1]
                                if pd.notna(value) and value != 0:
                                    timestamp = pd.Timestamp(year=current_year, month=1, day=1, hour=hour)
                                    all_data.append({
                                        'timestamp': timestamp,
                                        'station': station,
                                        'ridership': float(value),
                                        'hour': hour,
                                        'year': current_year,
                                        'source': 'reports_hourly'
                                    })
                    except:
                        pass
        
        # Sheet 2: 2016-2025 by station (Monthly data)
        if '2016-2025 by station' in sheet_names:
            print(f"   Processing monthly data 2016-2025...")
            df_monthly = pd.read_excel(reports_file, sheet_name='2016-2025 by station', engine='openpyxl', header=None)
            
            month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            
            for idx, row in df_monthly.iterrows():
                row_str = str(row.iloc[0])
                
                if row_str in [str(y) for y in range(2016, 2026)]:
                    current_year = int(row_str)
                    
                    if current_year <= 2020:
                        start_col = 2
                    else:
                        start_col = 18
                    
                    for offset in range(1, 13):
                        if idx + offset < len(df_monthly):
                            month_row = df_monthly.iloc[idx + offset]
                            month_name = str(month_row.iloc[0])
                            
                            if month_name in month_names:
                                month_num = month_names.index(month_name) + 1
                                for station_idx, station in enumerate(STATIONS):
                                    value_col = start_col + station_idx
                                    if value_col < len(month_row):
                                        value = month_row.iloc[value_col]
                                        if pd.notna(value) and value != 0:
                                            timestamp = pd.Timestamp(year=current_year, month=month_num, day=15)
                                            all_data.append({
                                                'timestamp': timestamp,
                                                'station': station,
                                                'ridership': float(value),
                                                'year': current_year,
                                                'month': month_num,
                                                'source': 'reports_monthly'
                                            })
        
        # Sheet 3: 2016-2025 (Daily data)
        if '2016-2025' in sheet_names:
            print(f"   Processing daily data 2016-2025...")
            df_daily = pd.read_excel(reports_file, sheet_name='2016-2025', engine='openpyxl', header=None)
            
            current_year = None
            current_month = None
            month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            
            for idx, row in df_daily.iterrows():
                row_str = str(row.iloc[0])
                
                if row_str in [str(y) for y in range(2016, 2026)]:
                    current_year = int(row_str)
                    continue
                
                if row_str in month_names:
                    current_month = row_str
                    month_num = month_names.index(current_month) + 1
                    continue
                
                try:
                    day = int(row_str)
                    if 1 <= day <= 31 and current_year and current_month:
                        if current_year <= 2020:
                            start_col = 2
                        else:
                            start_col = 18
                        
                        for station_idx, station in enumerate(STATIONS):
                            value_col = start_col + station_idx
                            if value_col < len(row):
                                value = row.iloc[value_col]
                                if pd.notna(value) and value != 0:
                                    timestamp = pd.Timestamp(year=current_year, month=month_num, day=day)
                                    all_data.append({
                                        'timestamp': timestamp,
                                        'station': station,
                                        'ridership': float(value),
                                        'year': current_year,
                                        'month': month_num,
                                        'day': day,
                                        'source': 'reports_daily'
                                    })
                except:
                    pass
        
        if all_data:
            df = pd.DataFrame(all_data)
            print(f"   ✅ Loaded {len(df)} rows from reports")
            return df
        
    except Exception as e:
        print(f"   ❌ Error loading reports: {e}")
    
    return None

# ========== LOAD ALL DATA ==========
print("\n📊 LOADING YOUR EXCEL FILES...")

all_data_frames = []

hourly_data = load_hourly_data()
if hourly_data is not None and len(hourly_data) > 0:
    all_data_frames.append(hourly_data)
    print(f"\n✅ Data folder: {len(hourly_data):,} rows")

reports_data = load_reports_data()
if reports_data is not None and len(reports_data) > 0:
    all_data_frames.append(reports_data)
    print(f"✅ Data Reports: {len(reports_data):,} rows")

if not all_data_frames:
    print("\n❌ NO DATA FOUND! Make sure your Excel files are in:")
    print("   - Data/ folder")
    print("   - Data Reports/ folder")
    exit()

mrt_data = pd.concat(all_data_frames, ignore_index=True)
mrt_data = mrt_data.sort_values('timestamp')

print(f"\n🎉 TOTAL DATA: {len(mrt_data):,} ROWS")

# ========== FIX DATA IMBALANCE ==========
print("\n📊 FIXING DATA IMBALANCE...")

# Convert daily totals to hourly
north_ave_mask = mrt_data['station'] == 'North Ave'
if north_ave_mask.any():
    mrt_data.loc[north_ave_mask, 'ridership'] = mrt_data.loc[north_ave_mask, 'ridership'] / 18
    print("✅ Converted North Ave from DAILY to HOURLY")

# ========== CREATE AND SAVE PICKLE FILES ==========
print("\n💾 CREATING LIGHTWEIGHT PICKLE FILES...")

# 1. Station Capacities
station_capacities = {}
for station in STATIONS:
    station_data = mrt_data[mrt_data['station'] == station]['ridership']
    if len(station_data) > 0:
        station_capacities[station] = int(station_data.quantile(0.95))
    else:
        station_capacities[station] = 5000  # Default value

with open('station_capacities.pkl', 'wb') as f:
    pickle.dump(station_capacities, f)
print("✅ Created station_capacities.pkl")

# 2. Hourly Averages
hourly_averages = {}
for station in STATIONS:
    station_data = mrt_data[mrt_data['station'] == station]
    if len(station_data) > 0:
        hourly_avg = station_data.groupby('hour')['ridership'].mean().to_dict()
        hourly_averages[station] = hourly_avg
    else:
        hourly_averages[station] = {h: 2000 for h in range(24)}

with open('hourly_averages.pkl', 'wb') as f:
    pickle.dump(hourly_averages, f)
print("✅ Created hourly_averages.pkl")

# 3. Time Series Data (last 1000 points per station)
time_series_data = {}
for station in STATIONS:
    station_df = mrt_data[mrt_data['station'] == station].copy()
    if len(station_df) > 0:
        station_df = station_df.sort_values('timestamp')
        if len(station_df) > 1000:
            station_df = station_df.tail(1000)
        # Keep only needed columns
        time_series_data[station] = station_df[['timestamp', 'hour', 'ridership']].to_dict('records')
    else:
        time_series_data[station] = []

with open('time_series_data.pkl', 'wb') as f:
    pickle.dump(time_series_data, f)
print("✅ Created time_series_data.pkl")

# 4. Train and Save LSTM Models (optional - skip if too slow)
print("\n🤖 TRAINING LSTM MODELS (this may take a while)...")
os.makedirs('models', exist_ok=True)

models_trained = 0
for station in STATIONS:
    print(f"   Training {station}...")
    series = mrt_data[mrt_data['station'] == station]['ridership']
    
    if len(series) < 100:
        continue
    
    values = series.values[-2000:].reshape(-1, 1)
    
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
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, input_shape=(24, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    
    model.save(f'models/{station}_lstm.h5')
    with open(f'models/{station}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    models_trained += 1
    print(f"   ✅ Saved {station} model")

print(f"\n✅ Trained {models_trained} models")

print("\n" + "="*70)
print("🎉 ALL DONE! Pickle files created successfully!")
print("="*70)
print("\nFiles created:")
print("  - station_capacities.pkl")
print("  - hourly_averages.pkl")  
print("  - time_series_data.pkl")
print("  - models/*.h5 (if training completed)")
print("\nNow your app.py will load instantly!")