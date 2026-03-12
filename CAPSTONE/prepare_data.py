# prepare.py
import pandas as pd
import glob
import os
import numpy as np
from datetime import datetime

def process_all_data():
    path = 'data_training'  # Your folder with CSV files
    all_files = glob.glob(os.path.join(path, "*.csv"))
    
    stations = ['North Ave', 'Quezon Ave', 'GMA Kamuning', 'Cubao', 'Santolan', 
                'Ortigas', 'Shaw Blvd', 'Boni Ave', 'Guadalupe', 'Buendia', 
                'Ayala Ave', 'Magallanes', 'Taft']
    
    # Map to your app's station names
    station_name_map = {
        'GMA Kamuning': 'Kamuning',
        'North Ave': 'North Ave',
        'Quezon Ave': 'Quezon Ave',
        'Cubao': 'Cubao',
        'Santolan': 'Santolan',
        'Ortigas': 'Ortigas',
        'Shaw Blvd': 'Shaw Blvd',
        'Boni Ave': 'Boni Ave',
        'Guadalupe': 'Guadalupe',
        'Buendia': 'Buendia',
        'Ayala Ave': 'Ayala Ave',
        'Magallanes': 'Magallanes',
        'Taft': 'Taft'
    }
    
    
    master_list = []
    skipped_files = 0

    print(f"🔍 Found {len(all_files)} files. Starting cleanup...")
    print("="*70)

    for filename in all_files:
        try:
            # Get year from filename if possible
            file_year = None
            for year in range(2016, 2026):
                if str(year) in filename:
                    file_year = year
                    break
            
            # Skip DOTr header rows
            df = pd.read_csv(filename, skiprows=6)
            
            # Check if columns exist
            expected_cols = ['Unnamed: 0', 'TIME'] + stations
            if not all(col in df.columns for col in ['Unnamed: 0', 'TIME']):
                print(f"⚠️  {os.path.basename(filename)}: Missing required columns")
                skipped_files += 1
                continue
                
            df = df.rename(columns={'Unnamed: 0': 'Date'})
            df = df.dropna(subset=['Date', 'TIME'])
            
            file_rows = 0
            for _, row in df.iterrows():
                try:
                    # Extract starting hour (e.g., "05:00 - 05:59" -> 5)
                    time_str = str(row['TIME']).strip()
                    if ' - ' in time_str:
                        hour = int(time_str.split(':')[0])
                    else:
                        continue  # Skip if time format is wrong
                    
                    date_val = row['Date']
                    
                    # Create timestamp
                    if file_year:
                        # Use year from filename
                        timestamp = pd.to_datetime(f"{file_year}-{date_val} {hour}:00:00")
                    else:
                        # Try to parse date as is
                        try:
                            timestamp = pd.to_datetime(f"{date_val} {hour}:00:00")
                        except:
                            continue
                    
                    # Combine Entry and Exit for each station
                    for station in stations:
                        if station in df.columns:
                            station_idx = df.columns.get_loc(station)
                            
                            # Get entry value
                            entry_val = pd.to_numeric(row.iloc[station_idx], errors='coerce')
                            
                            # Get exit value (next column)
                            if station_idx + 1 < len(row):
                                exit_val = pd.to_numeric(row.iloc[station_idx + 1], errors='coerce')
                            else:
                                exit_val = 0
                            
                            # Handle NaN
                            entry_val = 0 if pd.isna(entry_val) else entry_val
                            exit_val = 0 if pd.isna(exit_val) else exit_val
                            
                            total_volume = entry_val + exit_val
                            
                            if total_volume > 0:  # Only save non-zero entries
                                master_list.append({
                                    'timestamp': timestamp,
                                    'station': station_name_map[station],
                                    'ridership': total_volume,
                                    'hour': hour,
                                    'day': timestamp.day,
                                    'month': timestamp.month,
                                    'year': timestamp.year,
                                    'day_of_week': timestamp.dayofweek,
                                    'is_weekend': 1 if timestamp.dayofweek >= 5 else 0
                                })
                                file_rows += 1
                except Exception as e:
                    continue  # Skip problematic rows
                    
            print(f"✅ {os.path.basename(filename)}: Processed {file_rows} rows")
                    
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(filename)}: {str(e)[:50]}")
            skipped_files += 1

    print("="*70)
    print(f"📊 Total files processed: {len(all_files) - skipped_files}/{len(all_files)}")
    print(f"📈 Total rows collected: {len(master_list):,}")

    # Save the cleaned data
    if master_list:
        final_df = pd.DataFrame(master_list)
        final_df = final_df.sort_values(by='timestamp')
        
        # Save as CSV
        final_df.to_csv('mrt_cleaned_master.csv', index=False)
        print(f"💾 Saved to 'mrt_cleaned_master.csv' ({len(final_df):,} rows)")
        
        # Also save as pickle for faster loading later
        final_df.to_pickle('mrt_cleaned_master.pkl')
        print(f"💾 Saved to 'mrt_cleaned_master.pkl' for faster loading")
        
        # Show summary
        print("\n📊 DATA SUMMARY:")
        print(f"   Years: {sorted(final_df['year'].unique())}")
        print(f"   Stations: {final_df['station'].nunique()}")
        print(f"   Date range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")
        
        # Show sample
        print("\n📋 Sample data:")
        print(final_df.head(3).to_string())
        
    else:
        print("❌ No data collected!")

if __name__ == "__main__":
    process_all_data()