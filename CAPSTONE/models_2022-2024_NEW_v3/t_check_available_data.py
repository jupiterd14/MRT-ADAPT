# check_available_dates.py
import pandas as pd

df = pd.read_csv('data (2022-2024)/2025_sorted.csv')
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

print("Available dates in 2025 data:")
print(f"  First date: {df['datetime'].min()}")
print(f"  Last date: {df['datetime'].max()}")

# Show sample of dates
print("\nSample of dates available:")
sample_dates = df['Date'].unique()[:20]
for date in sorted(sample_dates):
    print(f"  {date}")