"""
Model Registry Verification Script
Checks that all 13 stations have valid model + scaler pairs
"""

import os
import pickle
import tensorflow as tf
from datetime import datetime

STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

def verify_model_registry(models_dir='models/'):
    """
    Verify that every station has both .keras and .pkl files
    """
    print("="*70)
    print("📦 MODEL REGISTRY VERIFICATION")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {
        'valid': [],
        'missing_model': [],
        'missing_scaler': [],
        'corrupted': []
    }
    
    for station in STATIONS:
        model_path = f'{models_dir}{station}_lstm.keras'
        scaler_path = f'{models_dir}{station}_scaler.pkl'
        
        model_exists = os.path.exists(model_path)
        scaler_exists = os.path.exists(scaler_path)
        
        if not model_exists and not scaler_exists:
            results['missing_model'].append(station)
            results['missing_scaler'].append(station)
            continue
        
        if not model_exists:
            results['missing_model'].append(station)
            continue
            
        if not scaler_exists:
            results['missing_scaler'].append(station)
            continue
        
        # Try to load and verify
        try:
            model = tf.keras.models.load_model(model_path)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Get file sizes
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            scaler_size = os.path.getsize(scaler_path) / 1024  # KB
            
            results['valid'].append({
                'station': station,
                'model_size': f'{model_size:.2f} MB',
                'scaler_size': f'{scaler_size:.1f} KB',
                'model_ok': True
            })
            
        except Exception as e:
            results['corrupted'].append({
                'station': station,
                'error': str(e)
            })
    
    # Print results
    print("\n✅ VALID MODELS:")
    for item in results['valid']:
        print(f"   📍 {item['station']}: {item['model_size']} + {item['scaler_size']}")
    
    if results['missing_model']:
        print("\n❌ MISSING MODEL FILES:")
        for station in results['missing_model']:
            print(f"   • {station}_lstm.keras")
    
    if results['missing_scaler']:
        print("\n❌ MISSING SCALER FILES:")
        for station in results['missing_scaler']:
            print(f"   • {station}_scaler.pkl")
    
    if results['corrupted']:
        print("\n⚠️ CORRUPTED FILES:")
        for item in results['corrupted']:
            print(f"   • {item['station']}: {item['error']}")
    
    # Summary
    total_valid = len(results['valid'])
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"   Valid models: {total_valid}/{len(STATIONS)}")
    print(f"   Missing models: {len(results['missing_model'])}")
    print(f"   Missing scalers: {len(results['missing_scaler'])}")
    print(f"   Corrupted: {len(results['corrupted'])}")
    
    if total_valid == len(STATIONS):
        print("\n🎉 ALL SYSTEMS GO! Ready for deployment.")
    else:
        print(f"\n⚠️ WARNING: {len(STATIONS) - total_valid} stations need attention!")
    
    return results


def generate_model_manifest(results, output_file='model_manifest.txt'):
    """
    Generate a manifest file listing all models with metadata
    """
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MRT-3 LSTM MODEL MANIFEST\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        for item in results['valid']:
            f.write(f"Station: {item['station']}\n")
            f.write(f"  Model: {item['station']}_lstm.keras ({item['model_size']})\n")
            f.write(f"  Scaler: {item['station']}_scaler.pkl ({item['scaler_size']})\n")
            f.write(f"  Status: ACTIVE\n\n")
        
        f.write("="*70 + "\n")
        f.write(f"Total Models: {len(results['valid'])}/13\n")
        f.write("="*70 + "\n")
    
    print(f"\n✅ Manifest saved to: {output_file}")


if __name__ == "__main__":
    results = verify_model_registry()
    
    if len(results['valid']) == len(STATIONS):
        generate_model_manifest(results)
    
    # Exit with error code if models are missing
    if len(results['missing_model']) > 0 or len(results['corrupted']) > 0:
        exit(1)