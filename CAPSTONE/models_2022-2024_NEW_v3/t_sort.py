import pandas as pd
import numpy as np

# Load the file
df = pd.read_csv('data (2022-2024)/2025.csv')

# Create timestamp
df['Full_Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# Sort chronologically
df_sorted = df.sort_values('Full_Timestamp').reset_index(drop=True)

# OFFICIAL MRT-3 STATION NAMES (North to South)
station_names = {
    1: 'North Avenue',
    2: 'Quezon Avenue',
    3: 'GMA-Kamuning',
    4: 'Araneta Center-Cubao',
    5: 'Santolan-Annapolis',
    6: 'Ortigas',
    7: 'Shaw Boulevard',
    8: 'Boni',
    9: 'Guadalupe',
    10: 'Buendia',
    11: 'Ayala',
    12: 'Magallanes',
    13: 'Taft Avenue'
}

# IMPORTANT: Fix the station names in your data
# Your output shows 'Gil Puyat' and 'Vito Cruz' which are LRT stations!
# Map the incorrect names to correct MRT-3 names
name_fixes = {
    'Gil Puyat': 'Buendia',  # Gil Puyat is actually Buendia station
    'Vito Cruz': 'Buendia',   # Vito Cruz doesn't exist on MRT-3
}

# Apply fixes if needed
df_sorted['StationEntry_Name'] = df_sorted['StationEntry'].map(station_names)
df_sorted['StationExit_Name'] = df_sorted['StationExit'].map(station_names)

# Replace any incorrect names
df_sorted['StationEntry_Name'] = df_sorted['StationEntry_Name'].replace(name_fixes)
df_sorted['StationExit_Name'] = df_sorted['StationExit_Name'].replace(name_fixes)

# Add direction (Northbound if going to higher number station)
df_sorted['Direction'] = np.where(
    df_sorted['StationEntry'] < df_sorted['StationExit'], 'Northbound',
    np.where(df_sorted['StationEntry'] > df_sorted['StationExit'], 'Southbound', 'Same_Station')
)

# Add hour
df_sorted['Hour'] = df_sorted['Full_Timestamp'].dt.hour

# Create ONE simple file - hourly totals by ALL dimensions
final_df = df_sorted.groupby([
    'Full_Timestamp', 'Date', 'Time', 'Hour',
    'StationEntry', 'StationEntry_Name',
    'StationExit', 'StationExit_Name', 
    'Direction'
])['TotalPassenger'].sum().reset_index()

# Sort
final_df = final_df.sort_values('Full_Timestamp').reset_index(drop=True)

# Save with a NEW filename (since old one is locked)
output_file = '2025_mrt3_complete.csv'
final_df.to_csv(output_file, index=False)

print("="*60)
print("MRT-3 DATA PROCESSING - COMPLETE")
print("="*60)
print(f"\n✅ DONE! FILE SAVED: {output_file}")
print(f"   Records: {len(final_df):,}")
print(f"   Total Passengers: {final_df['TotalPassenger'].sum():,.0f}")

print("\n" + "="*60)
print("FIRST 15 ROWS (MRT-3 Official Stations):")
print("="*60)
print(final_df[['Full_Timestamp', 'StationEntry_Name', 'StationExit_Name', 'Direction', 'TotalPassenger']].head(15))

print("\n" + "="*60)
print("OFFICIAL MRT-3 STATIONS (North to South):")
print("="*60)
for num, name in station_names.items():
    # Get passenger counts
    entry_count = final_df[final_df['StationEntry_Name'] == name]['TotalPassenger'].sum()
    exit_count = final_df[final_df['StationExit_Name'] == name]['TotalPassenger'].sum()
    print(f"  {num}: {name:<25} | Entry: {entry_count:>12,} | Exit: {exit_count:>12,}")

# Top origins/destinations
print("\n" + "="*60)
print("TOP 5 BUSIEST ENTRY STATIONS:")
print("="*60)
top_entry = final_df.groupby('StationEntry_Name')['TotalPassenger'].sum().sort_values(ascending=False).head(5)
for station, passengers in top_entry.items():
    print(f"  {station}: {passengers:,.0f} passengers")

print("\n" + "="*60)
print("TOP 5 BUSIEST EXIT STATIONS:")
print("="*60)
top_exit = final_df.groupby('StationExit_Name')['TotalPassenger'].sum().sort_values(ascending=False).head(5)
for station, passengers in top_exit.items():
    print(f"  {station}: {passengers:,.0f} passengers")

print("\n" + "="*60)
print("TAFT AVENUE (Station 13) - Northbound Traffic:")
print("="*60)
taft_north = final_df[(final_df['StationEntry_Name'] == 'Taft Avenue') & (final_df['Direction'] == 'Northbound')]
taft_hourly = taft_north.groupby('Hour')['TotalPassenger'].sum().reset_index()
for _, row in taft_hourly.head(10).iterrows():
    print(f"  {row['Hour']:02d}:00 - {row['TotalPassenger']:,.0f} passengers")

print("\n✅ File saved successfully! Use: 2025_mrt3_complete.csv")