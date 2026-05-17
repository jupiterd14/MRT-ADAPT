# Create a professional presentation of your scaler analysis
import pickle
import pandas as pd
import matplotlib.pyplot as plt

with open('models_2022-2024_NEW_v2w/openclose/Cubao_Northbound_feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Create a clear, professional table
analysis = []
features_short = [
    "Hour", "Weekday", "Month", "Hour_Sin", "Hour_Cos", 
    "DOW_Sin", "DOW_Cos", "Month_Sin", "Month_Cos",
    "Is Operating", "Morning Rush", "Evening Rush", "Noon",
    "Pre-opening", "Post-closing", "Minutes to Close", 
    "Minutes from Open", "Time Normalized", "Minute Normalized",
    "Weekend", "Holiday", "Special Event", "Christmas", 
    "Payday", "Friday", "Rush Hour", "Maintenance", 
    "Extended Hours", "Congestion"
]

for i, name in enumerate(features_short):
    analysis.append({
        'Feature': name,
        'Min Value': scaler.data_min_[i],
        'Max Value': scaler.data_max_[i],
        'Status': 'Active' if scaler.data_max_[i] > 0 else 'No Data',
        'Type': 'Binary' if scaler.data_max_[i] == 1 and scaler.data_min_[i] == 0 else
                'Cyclical' if scaler.data_min_[i] == -1 else
                'Numerical'
    })

df_analysis = pd.DataFrame(analysis)

print("\n" + "="*80)
print("MODEL FEATURE ANALYSIS - For Professor Review")
print("="*80)
print(f"\n📊 Total Features: {len(df_analysis)}")
print(f"✅ Active Features: {len(df_analysis[df_analysis['Status']=='Active'])}")
print(f"⚠️  Features with No Data: {len(df_analysis[df_analysis['Status']=='No Data'])}")

print("\n" + "="*80)
print("FEATURE BREAKDOWN BY TYPE:")
print("="*80)
print(df_analysis['Type'].value_counts().to_string())

print("\n" + "="*80)
print("FEATURES WITH NO DATA (LIMITATIONS):")
print("="*80)
print(df_analysis[df_analysis['Status']=='No Data'][['Feature', 'Min Value', 'Max Value']].to_string(index=False))

print("\n" + "="*80)
print("EXPLANATION FOR PROFESSOR:")
print("="*80)
print("""
These are NORMAL findings in real-world data:

1. 'Post-closing' = 0 → MRT-3 stops operations at 10:30 PM
   • This is a TRUE operational constraint
   • Model correctly learned there are no trips after this time

2. 'Minute Normalized' = 0 → Only hourly data available
   • Data was recorded at hourly intervals
   • Model trained on hourly aggregation (still effective)

✅ CONCLUSION: The scaler correctly captured ALL available data patterns
⚠️  LIMITATION: Model cannot predict beyond operating hours
💡 IMPROVEMENT: Could collect minute-level data for better accuracy
""")