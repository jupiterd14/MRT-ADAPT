# test_congestion.py
import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv('data (2022-2024)/2025.csv')
print(f"Loaded {len(df)} rows")

# Capacity dict
capacities = {
    "North Ave": 1142, "Quezon Ave": 1195, "Kamuning": 1364, "Cubao": 1747,
    "Santolan": 1306, "Ortigas": 1331, "Shaw Blvd": 1619, "Boni Ave": 1417,
    "Guadalupe": 1301, "Buendia": 1645, "Ayala Ave": 1222, "Magallanes": 1202,
    "Taft": 720
}

# Test a few stations
for station, capacity in capacities.items():
    station_num = {"North Ave":1, "Quezon Ave":2, "Kamuning":3, "Cubao":4, 
                   "Santolan":5, "Ortigas":6, "Shaw Blvd":7, "Boni Ave":8,
                   "Guadalupe":9, "Buendia":10, "Ayala Ave":11, "Magallanes":12,
                   "Taft":13}[station]
    
    station_df = df[df['StationEntry'] == station_num]
    if len(station_df) > 0:
        max_pass = station_df['TotalPassenger'].max()
        max_cong = (max_pass / capacity * 100)
        print(f"{station}: Capacity={capacity}, Max Passenger={max_pass}, Max Congestion={max_cong:.1f}%")
        