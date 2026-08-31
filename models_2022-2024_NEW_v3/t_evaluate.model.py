"""
evaluate_model.py - Evaluate trained LSTM models
"""
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ========== CONFIGURATION ==========
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

print("="*70)
print("LSTM MODEL EVALUATION - USING VALIDATION DATA")
print("="*70)

results = []

for station in STATIONS:
    print(f"\n📊 Evaluating: {station}")
    
    # Check if validation data exists
    val_file = f'models/{station}_validation_data.pkl'
    model_file = f'models/{station}_lstm.h5'
    
    if not os.path.exists(val_file):
        print(f"  ⚠️ No validation data found for {station}")
        print(f"  💡 Re-run training to save validation data")
        continue
        
    if not os.path.exists(model_file):
        print(f"  ❌ Model not found for {station}")
        continue
    
    try:
        # Load validation data
        with open(val_file, 'rb') as f:
            val_data = pickle.load(f)
        
        X_val = val_data['X_val']
        y_val = val_data['y_val']
        scaler = val_data['scaler']
        
        # Load model
        model = load_model(model_file)
        
        # Make predictions
        y_pred = model.predict(X_val, verbose=0)
        
        # Flatten predictions (in case they're 2D)
        y_pred = y_pred.flatten()
        y_val = y_val.flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        
        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((y_val - y_pred) / (y_val + 0.001))) * 100
        
        # Store results
        results.append({
            'station': station,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': f"{mape:.2f}%",
            'samples': len(y_val)
        })
        
        print(f"  ✅ MAE: {mae:.4f}")
        print(f"  ✅ RMSE: {rmse:.4f}")
        print(f"  ✅ R²: {r2:.4f}")
        print(f"  ✅ MAPE: {mape:.2f}%")
        print(f"  📊 Validation samples: {len(y_val):,}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

# ========== PRINT SUMMARY ==========
print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)

if results:
    results_df = pd.DataFrame(results)
    
    # Sort by MAE (best first)
    results_df = results_df.sort_values('mae')
    
    print("\n📊 Best performing stations (lowest MAE):")
    print(results_df[['station', 'mae', 'rmse', 'r2', 'mape']].head().to_string(index=False))
    
    print("\n📊 Worst performing stations (highest MAE):")
    print(results_df[['station', 'mae', 'rmse', 'r2', 'mape']].tail().to_string(index=False))
    
    # Overall statistics
    print("\n📈 OVERALL STATISTICS:")
    print(f"   Average MAE: {results_df['mae'].mean():.4f}")
    print(f"   Average RMSE: {results_df['rmse'].mean():.4f}")
    print(f"   Average R²: {results_df['r2'].mean():.4f}")
    print(f"   Best MAE: {results_df['mae'].min():.4f} ({results_df.loc[results_df['mae'].idxmin(), 'station']})")
    print(f"   Worst MAE: {results_df['mae'].max():.4f} ({results_df.loc[results_df['mae'].idxmax(), 'station']})")
    
    # Save to CSV
    results_df.to_csv('evaluation_results.csv', index=False)
    print(f"\n💾 Results saved to 'evaluation_results.csv'")
else:
    print("\n❌ No evaluation results. Re-run training first!")

print("\n" + "="*70)