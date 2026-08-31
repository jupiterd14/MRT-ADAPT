#!/usr/bin/env python3
"""Test if directional models load correctly"""

import os
import tensorflow as tf
import pickle

# Configuration
MODELS_PATH = 'models_2022-2024_NEW'
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

print("="*60)
print("TESTING DIRECTIONAL MODEL LOADING")
print("="*60)

loaded_count = 0
failed_count = 0

for station in STATIONS:
    for direction in ['Northbound', 'Southbound']:
        model_key = f"{station}_{direction}"
        model_path = f'{MODELS_PATH}/{model_key}_lstm_enhanced.keras'
        feature_scaler_path = f'{MODELS_PATH}/{model_key}_feature_scaler.pkl'
        target_scaler_path = f'{MODELS_PATH}/{model_key}_target_scaler.pkl'
        
        model_exists = os.path.exists(model_path)
        feature_exists = os.path.exists(feature_scaler_path)
        target_exists = os.path.exists(target_scaler_path)
        
        if model_exists and feature_exists and target_exists:
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                with open(feature_scaler_path, 'rb') as f:
                    feature_scaler = pickle.load(f)
                with open(target_scaler_path, 'rb') as f:
                    target_scaler = pickle.load(f)
                
                print(f"✅ {model_key:30s} - Model: {model.input_shape}, Feature scaler: {feature_scaler.scale_.shape}")
                loaded_count += 1
                del model  # Free memory
            except Exception as e:
                print(f"❌ {model_key:30s} - ERROR: {str(e)[:50]}")
                failed_count += 1
        else:
            missing = []
            if not model_exists: missing.append("model")
            if not feature_exists: missing.append("feature_scaler")
            if not target_exists: missing.append("target_scaler")
            print(f"⚠️ {model_key:30s} - Missing: {', '.join(missing)}")
            failed_count += 1

print("\n" + "="*60)
print(f"RESULTS: {loaded_count} loaded, {failed_count} failed")
print(f"Expected: {len(STATIONS) * 2} directional models")
print("="*60)