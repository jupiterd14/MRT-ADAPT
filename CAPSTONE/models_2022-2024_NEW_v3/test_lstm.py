import pandas as pd
from datetime import datetime

# Check your training data for odd hours
for year in [2022, 2023, 2024]:
    df = pd.read_csv(f'data (2022-2024)/{year}.csv')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['hour'] = df['datetime'].dt.hour
    
    # Count records by hour
    hour_counts = df.groupby('hour').size()
    print(f"\n{year} - Records by hour:")
    print(hour_counts)
    
    # Check passenger counts at odd hours
    odd_hours = df[df['hour'].between(0, 4) | df['hour'].between(23, 24)]
    if len(odd_hours) > 0:
        print(f"\n⚠️ Found {len(odd_hours)} records between 11PM-4AM in {year}")
        print(odd_hours[['datetime', 'TotalPassenger', 'StationEntry', 'StationExit']].head(10))
