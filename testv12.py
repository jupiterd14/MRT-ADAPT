"""
diagnose_sb_h8.py — Why does SB hour 8 fail so badly?
Run: python diagnose_sb_h8.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error

df = pd.read_csv('manual_test_2026_results.csv')

print("=" * 70)
print("SB HOUR 8 DEEP DIVE (the worst hour)")
print("=" * 70)

# Hour 8 SB
h8_sb = df[(df['direction'] == 'Southbound') & (df['hour'] == 8)]

print(f"\nSB hour 8:")
print(f"  Samples: {len(h8_sb):,}")
print(f"  Accuracy: {accuracy_score(h8_sb['actual_cat'], h8_sb['predicted_cat'])*100:.1f}%")
print(f"  MAE: {mean_absolute_error(h8_sb['actual'], h8_sb['predicted']):.2f}%")
print(f"  Pred mean: {h8_sb['predicted'].mean():.1f}%")
print(f"  Actual mean: {h8_sb['actual'].mean():.1f}%")
print(f"  Bias: {(h8_sb['predicted'] - h8_sb['actual']).mean():+.1f}%")

# Per-station breakdown
print(f"\n  Per-station in hour 8 SB:")
print(f"  {'Station':<15} {'Samples':>8} {'Accuracy':>10} {'Pred':>8} {'Actual':>8} {'Bias':>8}")
print(f"  {'-'*60}")
for station in sorted(h8_sb['station'].unique()):
    sub = h8_sb[h8_sb['station'] == station]
    if len(sub) == 0:
        continue
    acc = accuracy_score(sub['actual_cat'], sub['predicted_cat'])
    pred = sub['predicted'].mean()
    actual = sub['actual'].mean()
    bias = pred - actual
    print(f"  {station:<15} {len(sub):>8} {acc*100:>9.1f}% {pred:>7.1f}% {actual:>7.1f}% {bias:>+7.1f}%")

# Distribution at hour 8 SB
print(f"\n  Hour 8 SB actual distribution:")
print(f"  {h8_sb['actual_cat'].value_counts()}")
print(f"\n  Hour 8 SB predicted distribution:")
print(f"  {h8_sb['predicted_cat'].value_counts()}")

# Compare to hour 8 NB
print("\n" + "=" * 70)
print("COMPARE HOUR 8: NB vs SB")
print("=" * 70)

h8_nb = df[(df['direction'] == 'Northbound') & (df['hour'] == 8)]
print(f"\nHour 8 NB:")
print(f"  Accuracy: {accuracy_score(h8_nb['actual_cat'], h8_nb['predicted_cat'])*100:.1f}%")
print(f"  Pred mean: {h8_nb['predicted'].mean():.1f}%, Actual mean: {h8_nb['actual'].mean():.1f}%")

# Compare to hour 8 in 2025 training data (from Parquets)
print("\n" + "=" * 70)
print("COMPARE: 2025 hour 8 vs 2026 hour 8 (SB)")
print("=" * 70)

import os
total_2025_h8 = 0
total_2026_h8 = 0

for station in sorted(df['station'].unique()):
    parquet_path = f"services/data_cache/{station}_Southbound.parquet"
    if os.path.exists(parquet_path):
        pdf = pd.read_parquet(parquet_path)
        h8_2025 = pdf[(pdf.index.hour == 8)]['TotalPassenger']
        h8_2025_mean = h8_2025.mean() if len(h8_2025) > 0 else 0
        
        h8_2026 = h8_sb[h8_sb['station'] == station]['actual']
        h8_2026_mean = h8_2026.mean() if len(h8_2026) > 0 else 0
        
        if h8_2025_mean > 0 or h8_2026_mean > 0:
            change = ((h8_2026_mean - h8_2025_mean) / max(h8_2025_mean, 0.1)) * 100
            print(f"  {station:<15} 2025: {h8_2025_mean:>6.1f}%  2026: {h8_2026_mean:>6.1f}%  Δ: {change:>+6.1f}%")