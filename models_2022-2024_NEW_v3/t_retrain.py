# train_lstm.py - Fixed with proper scaling
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import matplotlib.pyplot as plt

# ========== CONFIGURATION ==========
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

STATION_NUMBERS = {
    "North Ave": 1, "Quezon Ave": 2, "Kamuning": 3, "Cubao": 4,
    "Santolan": 5, "Ortigas": 6, "Shaw Blvd": 7, "Boni Ave": 8,
    "Guadalupe": 9, "Buendia": 10, "Ayala Ave": 11, "Magallanes": 12,
    "Taft": 13
}

def create_sequences(features, target, seq_length=24):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

# Load data
print("Loading data...")
df = pd.read_csv('historical_data_3years.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"Loaded {len(df)} records")

# Add time features
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday
df['month'] = df['datetime'].dt.month

# Calculate congestion (0-100 scale)
max_entry = df['StationEntry'].max()
df['congestion'] = (df['StationEntry'] / max_entry * 100).clip(0, 100)

print(f"✅ Congestion range: {df['congestion'].min():.1f}% - {df['congestion'].max():.1f}%")

# Create models directory
os.makedirs('models', exist_ok=True)

# Store results
all_results = []

for station in STATIONS:
    print(f"\n{'='*60}")
    print(f"Training Stacked LSTM for: {station}")
    print('='*60)
    
    # Filter data for this station
    station_num = STATION_NUMBERS[station]
    station_df = df[(df['StationEntry'] == station_num) | (df['StationExit'] == station_num)].copy()
    station_df = station_df.sort_values('datetime')
    
    if len(station_df) < 100:
        print(f"Not enough data for {station} (only {len(station_df)} records)")
        continue
    
    # ========== CORRECT SCALING ==========
    feature_cols = ['hour', 'weekday', 'congestion']
    feature_data = station_df[feature_cols].values
    
    # Scale features (0-1)
    feature_scaler = MinMaxScaler()
    features_scaled = feature_scaler.fit_transform(feature_data)
    
    # Scale target separately (0-1)
    target_scaler = MinMaxScaler()
    targets_raw = station_df['congestion'].values.reshape(-1, 1)
    targets_scaled = target_scaler.fit_transform(targets_raw).flatten()
    
    print(f"✅ Features scaled: {features_scaled.shape}")
    print(f"✅ Targets scaled: min={targets_scaled.min():.4f}, max={targets_scaled.max():.4f}")
    
    # Create sequences
    X, y = create_sequences(features_scaled, targets_scaled, seq_length=24)
    
    if len(X) == 0:
        print(f"Not enough sequences for {station}")
        continue
    
    # Split: 80% train, 20% validation (NO SHUFFLE for time series)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"📊 Training samples: {len(X_train):,}")
    print(f"📊 Validation samples: {len(X_val):,}")
    
    # Build Stacked LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(24, len(feature_cols))),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Output is scaled congestion (0-1 scale)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train
    print("\n🔄 Training Stacked LSTM (max 50 epochs)...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # ========== SAVE EVERYTHING ==========
    
    # Save model
    model.save(f'models/{station}_lstm.h5')
    print(f"✅ Saved model: models/{station}_lstm.h5")
    
    # Save BOTH scalers (feature scaler AND target scaler)
    with open(f'models/{station}_feature_scaler.pkl', 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open(f'models/{station}_target_scaler.pkl', 'wb') as f:
        pickle.dump(target_scaler, f)
    print(f"✅ Saved feature scaler and target scaler")
    
    # Save validation data for evaluation
    val_data = {
        'X_val': X_val,
        'y_val': y_val,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_names': feature_cols
    }
    with open(f'models/{station}_validation_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    print(f"✅ Saved validation data: models/{station}_validation_data.pkl")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{station} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{station} - MAE (scaled)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'models/{station}_training_history.png')
    plt.close()
    print(f"✅ Saved plot: models/{station}_training_history.png")
    
    # Store results
    epochs_done = len(history.history['loss'])
    all_results.append({
        'station': station,
        'epochs_completed': epochs_done,
        'stopped_by': 'Early stopping' if epochs_done < 50 else 'Max epochs (50)',
        'final_train_loss': history.history['loss'][-1],
        'final_train_mae': history.history['mae'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_val_mae': history.history['val_mae'][-1]
    })
    
    print(f"\n✅ {station} complete!")
    print(f"   Epochs completed: {epochs_done}/50")
    print(f"   Final Validation MAE (scaled): {history.history['val_mae'][-1]:.4f}")
    print(f"   This equals ~{history.history['val_mae'][-1] * 100:.1f}% congestion error")

# Summary
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))
results_df.to_csv('training_summary.csv', index=False)

print("\n✅ All models saved successfully!")
print("\n📁 Files saved in 'models/' folder:")
print("   • *_lstm.h5 - Model weights")
print("   • *_feature_scaler.pkl - Feature scaler (hour, weekday)")
print("   • *_target_scaler.pkl - Target scaler (congestion)")
print("   • *_validation_data.pkl - Validation data for evaluation")
print("   • *_training_history.png - Training plots")
print("\n🚀 Next step: python evaluate_model.py")