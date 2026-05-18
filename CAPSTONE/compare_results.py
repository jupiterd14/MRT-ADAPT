import pandas as pd
import glob

# Load all result files
files = glob.glob('diagnosis_results_*.csv')

all_results = []
for f in files:
    df = pd.read_csv(f)
    df['source_file'] = f
    all_results.append(df)

combined = pd.concat(all_results, ignore_index=True)

print("\n========== ALL TEST RESULTS ==========")
print(combined[['timestamp', 'station', 'direction', 'absolute_error', 'percentage_error', 'verdict']].to_string())

print("\n========== AVERAGE BY STATION ==========")
avg_by_station = combined.groupby('station')['absolute_error'].mean().sort_values()
print(avg_by_station)

print(f"\n OVERALL AVERAGE ABSOLUTE ERROR: {combined['absolute_error'].mean():.1f} points")
print(f" BEST PERFORMANCE: {combined['absolute_error'].min():.1f} points")
print(f" WORST PERFORMANCE: {combined['absolute_error'].max():.1f} points")