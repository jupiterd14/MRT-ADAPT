"""
Standalone test for upload pipeline + prediction debugging.
Run: python testv12.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta


def main():
    print("=" * 70)
    print("UPLOAD PIPELINE + PREDICTION DEBUG TEST")
    print("=" * 70)

    # ============================================================
    # 1. Load full CSV (no row cap)
    # ============================================================
    csv_path = 'uploads/2025.csv'
    df = pd.read_csv(csv_path)
    print(f"\n[1] Loaded {len(df):,} rows")
    print(f"[1] Columns: {list(df.columns)}")

    # ============================================================
    # 2. Parse datetime
    # ============================================================
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    print(f"[2] After datetime parse: {len(df):,} rows")

    # ============================================================
    # 3. Hour distribution
    # ============================================================
    hour_dist = df['datetime'].dt.hour.value_counts().sort_index()
    print(f"[3] Hour distribution (first 5 / last 5):")
    print(hour_dist.head())
    print(hour_dist.tail())

    # ============================================================
    # 4. Direction filter
    # ============================================================
    def infer_dir(row):
        if row['StationEntry'] < row['StationExit']:
            return 'Southbound'
        elif row['StationEntry'] > row['StationExit']:
            return 'Northbound'
        return 'Unknown'

    df['direction'] = df.apply(infer_dir, axis=1)
    df = df[df['direction'] != 'Unknown']
    print(f"[4] After direction filter: {len(df):,} rows")

    # ============================================================
    # 5. Station mapping
    # ============================================================
    STATION_NUMBER_MAP = {
        1: "North Ave", 2: "Quezon Ave", 3: "Kamuning", 4: "Cubao",
        5: "Santolan", 6: "Ortigas", 7: "Shaw Blvd", 8: "Boni Ave",
        9: "Guadalupe", 10: "Buendia", 11: "Ayala Ave", 12: "Magallanes", 13: "Taft"
    }

    def get_station_name_fixed(row):
        entry = row['StationEntry']
        exit_station = row['StationExit']
        if entry == 1 or exit_station == 1:
            return "North Ave"
        if entry == 13 or exit_station == 13:
            return "Taft"
        return STATION_NUMBER_MAP.get(exit_station)

    df['StationName'] = df.apply(get_station_name_fixed, axis=1)
    df = df.dropna(subset=['StationName'])
    print(f"[5] After station mapping: {len(df):,} rows")

    # ============================================================
    # 6. Group by hour
    # ============================================================
    df['Hour'] = df['datetime'].dt.floor('h')
    grouped = df.groupby(['StationName', 'direction', 'Hour']).agg({
        'TotalPassenger': 'sum',
        'datetime': 'first'
    }).reset_index()
    print(f"[6] Grouped hourly records: {len(grouped):,}")

    # ============================================================
    # 7. Compute actual congestion via P90
    # ============================================================
    from routes.api_predict import get_p90_percentile

    p90_cache = {}

    def calc_congestion(row):
        key = f"{row['StationName']}_{row['direction']}"
        if key not in p90_cache:
            p90_cache[key] = get_p90_percentile(row['StationName'], row['direction'])
        p90 = p90_cache[key]
        if p90 <= 0:
            p90 = 1000
        return round(min((row['TotalPassenger'] / p90) * 100, 100), 1)

    grouped['actual_congestion'] = grouped.apply(calc_congestion, axis=1)
    print(f"[7] Computed actual_congestion: {len(grouped):,} rows")

    # ============================================================
    # 8. Rename columns
    # ============================================================
    df2 = grouped[['StationName', 'direction', 'Hour', 'actual_congestion']].copy()
    df2.columns = ['station', 'direction', 'datetime', 'actual_congestion']
    df2['datetime'] = df2['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df2['actual_congestion'] = df2['actual_congestion'].round(1)
    print(f"[8] After rename: {len(df2):,} rows")

    # ============================================================
    # 9. Operating-hours filter
    # ============================================================
    df2['datetime'] = pd.to_datetime(df2['datetime'], errors='coerce')
    df2 = df2.dropna(subset=['datetime'])

    def is_operating_hour(dt):
        hour = dt.hour + dt.minute / 60
        return 4.5 <= hour <= 22.5

    df2 = df2[df2['datetime'].apply(is_operating_hour)]
    df2['hour'] = df2['datetime'].dt.hour
    print(f"[9] After operating-hours filter: {len(df2):,} rows")
    print(f"[9] Unique hours: {sorted(df2['hour'].unique().tolist())}")

    # ============================================================
    # 10. Sample 3 per hour
    # ============================================================
    all_hours = list(range(5, 23))
    SAMPLES_PER_HOUR = 3
    sampled_dfs = []

    for hour in all_hours:
        hour_df = df2[df2['hour'] == hour]
        if len(hour_df) == 0:
            print(f"  [X] Hour {hour}: 0 rows")
            continue
        print(f"  [+] Hour {hour}: {len(hour_df):,} rows")

        if len(hour_df) <= SAMPLES_PER_HOUR:
            sampled_dfs.append(hour_df)
        else:
            pairs = hour_df.groupby(['station', 'direction']).size().reset_index(name='count')
            pair_samples = []
            for _, pair_row in pairs.iterrows():
                pair_df = hour_df[
                    (hour_df['station'] == pair_row['station']) &
                    (hour_df['direction'] == pair_row['direction'])
                ]
                if len(pair_df) > 0:
                    pair_samples.append(pair_df.sample(n=1, random_state=42))

            if pair_samples:
                combined = pd.concat(pair_samples, ignore_index=True)
                if len(combined) < SAMPLES_PER_HOUR:
                    remaining = hour_df.drop(index=combined.index, errors='ignore')
                    extra_n = min(SAMPLES_PER_HOUR - len(combined), len(remaining))
                    if extra_n > 0:
                        combined = pd.concat(
                            [combined, remaining.sample(n=extra_n, random_state=42)],
                            ignore_index=True
                        )
                sampled_dfs.append(combined)

    print(f"\n[10] sampled_dfs count: {len(sampled_dfs)}")

    if not sampled_dfs:
        print("[10] FAIL: NO SAMPLES")
        return

    df_sampled = pd.concat(sampled_dfs, ignore_index=True)
    print(f"[10] Total sampled rows: {len(df_sampled)}")
    print(f"[10] Hours covered: {sorted(df_sampled['hour'].unique().tolist())}")

    # ============================================================
    # 10b. P90 values
    # ============================================================
    print(f"\n[10b] P90 values (used to compute congestion):")
    test_stations = ["Ayala Ave", "Boni Ave", "Buendia", "North Ave"]
    for station in test_stations:
        for direction in ["Northbound", "Southbound"]:
            p90 = get_p90_percentile(station, direction)
            print(f"  {station:12} {direction:11}: p90 = {p90:8.0f}")

    # ============================================================
    # 10c. Model raw output for 5 AM (all 7 days)
    # ============================================================
    print(f"\n[10c] Model raw output at 5 AM for Buendia Northbound:")
    from services.feature_engineering import get_scaled_feature_sequence
    from services.model_loader import directional_models, directional_scalers

    station, direction = "Buendia", "Northbound"
    model_key = f"{station}_{direction}"

    model = directional_models.get(model_key)
    target_scaler = directional_scalers.get(f"{model_key}_target")
    p90_buendia = get_p90_percentile(station, direction)

    if model is not None and target_scaler is not None:
        for dow, label in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
            base_dt = datetime(2025, 1, 6) + timedelta(days=dow, hours=5)
            features = get_scaled_feature_sequence(station, direction, base_dt)
            if features is None:
                print(f"  {label} 5AM: features=None")
                continue
            inp = features.reshape(1, 24, -1)
            pred_scaled = model.predict(inp, verbose=0)[0][0]
            pred_passengers = float(target_scaler.inverse_transform([[pred_scaled]])[0][0])
            cong = min((pred_passengers / p90_buendia) * 100, 100)
            print(f"  {label} 5AM: raw={pred_scaled:7.4f}  passengers={pred_passengers:8.0f}  "
                  f"p90={p90_buendia:6.0f}  congestion={cong:5.1f}%")
    else:
        print(f"  ⚠️ Model or scaler missing for {model_key}")

    # ============================================================
    # 10d. Actual Buendia Northbound data at 5 AM
    # ============================================================
    print(f"\n[10d] Actual Buendia Northbound 5 AM congestion (from uploaded file):")
    buendia_data = df2[
        (df2['station'] == 'Buendia') &
        (df2['direction'] == 'Northbound') &
        (df2['hour'] == 5)
    ].copy()
    if len(buendia_data) > 0:
        buendia_data['dow'] = pd.to_datetime(buendia_data['datetime']).dt.weekday
        for dow, label in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
            subset = buendia_data[buendia_data['dow'] == dow]
            if len(subset) > 0:
                print(f"  {label} 5AM: avg={subset['actual_congestion'].mean():5.1f}%  "
                      f"min={subset['actual_congestion'].min():5.1f}%  "
                      f"max={subset['actual_congestion'].max():5.1f}%  "
                      f"n={len(subset)}")
    else:
        print(f"  (no Buendia Northbound 5AM data)")

    # ============================================================
    # 11. Load prediction cache
    # ============================================================
    print(f"\n[11] Loading prediction cache...")
    from routes.api_predict import _PREDICTION_CACHE

    cache_file = 'cache/cached_predictions.pkl'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        _PREDICTION_CACHE.update(data)
        print(f"[11] Prediction cache loaded: {len(_PREDICTION_CACHE)} entries")
    else:
        print(f"[11] WARNING: No prediction cache at {cache_file}")

    # ============================================================
    # 11b. Inspect the cache for 5 AM values
    # ============================================================
    print(f"\n[11b] Prediction cache entries for hour=5:")
    for station in test_stations:
        for direction in ["Northbound", "Southbound"]:
            for dow in range(7):
                key = f"{station}_{direction}_{dow}_5"
                val = _PREDICTION_CACHE.get(key)
                if val is not None:
                    print(f"  {key}: {val:.1f}%")

    # ============================================================
    # 12. Test real predictions
    # ============================================================
    from routes.api_predict import get_directional_prediction

    print(f"\n[12] Testing predictions on 5 sampled rows:")
    for _, row in df_sampled.head(5).iterrows():
        station = str(row['station']).strip()
        direction = str(row['direction']).strip().capitalize()
        target_time = pd.to_datetime(row['datetime'])
        actual = float(row['actual_congestion'])

        pred = get_directional_prediction(station, direction, target_time)
        error = abs(pred - actual) if pred is not None else None

        if pred is not None:
            print(f"     {station:12} {direction:11} @ {target_time}  "
                  f"actual={actual:5.1f}%  predicted={pred:5.1f}%  err={error:4.1f}%")
        else:
            print(f"     {station} {direction} @ {target_time}  FAILED")

    # ============================================================
    # 12b. Test same rows with a direct model call (bypass cache)
    # ============================================================
    print(f"\n[12b] Same rows with DIRECT model call (bypass cache):")
    for _, row in df_sampled.head(5).iterrows():
        station = str(row['station']).strip()
        direction = str(row['direction']).strip().capitalize()
        target_time = pd.to_datetime(row['datetime'])
        actual = float(row['actual_congestion'])
        model_key = f"{station}_{direction}"

        model = directional_models.get(model_key)
        target_scaler = directional_scalers.get(f"{model_key}_target")

        if model is None or target_scaler is None:
            print(f"     {station} {direction} @ {target_time}  MODEL MISSING")
            continue

        features = get_scaled_feature_sequence(station, direction, target_time)
        if features is None:
            print(f"     {station} {direction} @ {target_time}  FEATURES None")
            continue

        inp = features.reshape(1, 24, -1)
        pred_scaled = model.predict(inp, verbose=0)[0][0]
        pred_passengers = float(target_scaler.inverse_transform([[pred_scaled]])[0][0])
        p90 = get_p90_percentile(station, direction)
        pred_cong = min((pred_passengers / p90) * 100, 100)
        err = abs(pred_cong - actual)

        print(f"     {station:12} {direction:11} @ {target_time}  "
              f"actual={actual:5.1f}%  direct_pred={pred_cong:5.1f}%  err={err:4.1f}%  "
              f"(passengers={pred_passengers:.0f}, p90={p90:.0f})")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    from app import app
    with app.app_context():
        main()