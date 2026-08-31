# process_data.py (FIXED)
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

STATION_ID_MAP = {
    1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
    5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
    9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
}

def process_all_data():
    print("📊 Processing historical data...")
    
    historical_entry_counts = {station: [] for station in STATIONS}
    historical_exit_counts = {station: [] for station in STATIONS}
    direction_counts = {'northbound': 0, 'southbound': 0}
    hourly_patterns = {}
    day_of_week_patterns = {}
    
    data_folder = 'data_new_2025'
    total_records = 0
    
    if not os.path.exists(data_folder):
        print(f"❌ Folder '{data_folder}' not found!")
        return None
    
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    print(f"📁 Found {len(csv_files)} CSV files")
    
    for file in csv_files:
        file_path = os.path.join(data_folder, file)
        print(f"📖 Reading {file}...")
        
        try:
            df = pd.read_csv(file_path)
            total_records += len(df)
            
            # Your CSV columns are: TotalPassenger, Time, Date, StationEntry, StationExit
            print(f"   Columns: {list(df.columns)}")  # Debug: see column order
            
            # Process each row using correct column names
            for _, row in df.iterrows():
                try:
                    # Use the actual column names from your CSV
                    passengers = int(row['TotalPassenger'])
                    time_str = str(row['Time'])
                    date_str = str(row['Date'])
                    entry_id = int(row['StationEntry'])
                    exit_id = int(row['StationExit'])
                    
                    entry_station = STATION_ID_MAP.get(entry_id)
                    exit_station = STATION_ID_MAP.get(exit_id)
                    
                    if entry_station and exit_station and entry_station != exit_station:
                        # Determine direction
                        if entry_id < exit_id:
                            direction = 'southbound'
                        else:
                            direction = 'northbound'
                        
                        direction_counts[direction] += passengers
                        
                        # Count entry and exit
                        historical_entry_counts[entry_station].append(passengers)
                        historical_exit_counts[exit_station].append(passengers)
                        
                        # Parse time for hourly patterns
                        try:
                            hour = int(time_str.split(':')[0]) if ':' in time_str else int(time_str)
                            if hour not in hourly_patterns:
                                hourly_patterns[hour] = {'entry': [], 'exit': []}
                            hourly_patterns[hour]['entry'].append(passengers)
                            hourly_patterns[hour]['exit'].append(passengers)
                        except:
                            pass
                        
                        # Parse date for day of week patterns
                        try:
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                            day = date_obj.weekday()
                            if day not in day_of_week_patterns:
                                day_of_week_patterns[day] = {'entry': [], 'exit': []}
                            day_of_week_patterns[day]['entry'].append(passengers)
                            day_of_week_patterns[day]['exit'].append(passengers)
                        except:
                            pass
                            
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"   ❌ Error reading {file}: {e}")
    
    # Calculate averages
    print("\n📊 Calculating averages...")
    
    historical_entry = {}
    historical_exit = {}
    
    for station in STATIONS:
        historical_entry[station] = np.mean(historical_entry_counts[station]) if historical_entry_counts[station] else 0
        historical_exit[station] = np.mean(historical_exit_counts[station]) if historical_exit_counts[station] else 0
    
    # Calculate hourly averages
    hourly_avg_entry = {}
    hourly_avg_exit = {}
    for hour in range(24):
        if hour in hourly_patterns:
            hourly_avg_entry[hour] = np.mean(hourly_patterns[hour]['entry']) if hourly_patterns[hour]['entry'] else 0
            hourly_avg_exit[hour] = np.mean(hourly_patterns[hour]['exit']) if hourly_patterns[hour]['exit'] else 0
        else:
            hourly_avg_entry[hour] = 0
            hourly_avg_exit[hour] = 0
    
    # Calculate day of week averages
    dow_avg_entry = {}
    dow_avg_exit = {}
    for day in range(7):
        if day in day_of_week_patterns:
            dow_avg_entry[day] = np.mean(day_of_week_patterns[day]['entry']) if day_of_week_patterns[day]['entry'] else 0
            dow_avg_exit[day] = np.mean(day_of_week_patterns[day]['exit']) if day_of_week_patterns[day]['exit'] else 0
        else:
            dow_avg_entry[day] = 0
            dow_avg_exit[day] = 0
    
    # Save to cache file
    cache_data = {
        'historical_entry': historical_entry,
        'historical_exit': historical_exit,
        'direction_counts': direction_counts,
        'hourly_avg_entry': hourly_avg_entry,
        'hourly_avg_exit': hourly_avg_exit,
        'dow_avg_entry': dow_avg_entry,
        'dow_avg_exit': dow_avg_exit,
        'total_records': total_records
    }
    
    with open('historical_data_cache.pkl', 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"\n✅ Processed {total_records} total records")
    print(f"📊 Direction Flow:")
    total = direction_counts['northbound'] + direction_counts['southbound']
    if total > 0:
        print(f"   Northbound: {direction_counts['northbound']:,} ({direction_counts['northbound']/total*100:.1f}%)")
        print(f"   Southbound: {direction_counts['southbound']:,} ({direction_counts['southbound']/total*100:.1f}%)")
    
    print("\n📊 Historical averages per station (average passengers per trip):")
    for station in STATIONS:
        print(f"  📍 {station}: Entry={historical_entry[station]:.0f}, Exit={historical_exit[station]:.0f}")
    
    return cache_data

if __name__ == '__main__':
    process_all_data()