# routes/model_cache.py
import os
import pickle
import joblib
from pathlib import Path
from datetime import datetime  # ← ADD THIS IMPORT

MODEL_CACHE_DIR = "models_cache"

def ensure_cache_dir():
    """Create cache directory if it doesn't exist"""
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def save_models_to_cache(models, scalers, metadata=None):
    """
    Save loaded models to disk cache for instant loading on next startup
    """
    ensure_cache_dir()
    
    cache_file = os.path.join(MODEL_CACHE_DIR, "models_cache.pkl")
    
    try:
        # Save scalers and metadata (model paths, etc)
        cache_data = {
            'model_keys': list(models.keys()),
            'scalers': scalers,
            'metadata': metadata or {'version': '1.0', 'timestamp': str(datetime.now())}
        }
        
        # Save scalers and metadata
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Save each model using joblib (better for TF models)
        saved_count = 0
        for key, model in models.items():
            model_path = os.path.join(MODEL_CACHE_DIR, f"{key}.joblib")
            try:
                joblib.dump(model, model_path)
                saved_count += 1
                print(f"  💾 Cached model: {key}")
            except Exception as e:
                print(f"  ⚠️ Could not cache {key}: {e}")
        
        print(f"✅ Cached {saved_count}/{len(models)} models to {MODEL_CACHE_DIR}/")
        return True
        
    except Exception as e:
        print(f"❌ Failed to cache models: {e}")
        return False

def load_models_from_cache():
    """
    Load models from disk cache - MUCH faster than loading from .h5
    Returns: (models, scalers) or (None, None) if cache doesn't exist
    """
    cache_file = os.path.join(MODEL_CACHE_DIR, "models_cache.pkl")
    
    if not os.path.exists(cache_file):
        print("ℹ️ No model cache found")
        return None, None
    
    try:
        # Load scalers and metadata
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        model_keys = cache_data['model_keys']
        scalers = cache_data['scalers']
        
        # Load each model from joblib files
        models = {}
        loaded_count = 0
        
        for key in model_keys:
            model_path = os.path.join(MODEL_CACHE_DIR, f"{key}.joblib")
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    models[key] = model
                    loaded_count += 1
                    print(f"  ✅ Loaded cached model: {key}")
                except Exception as e:
                    print(f"  ⚠️ Could not load cached {key}: {e}")
        
        print(f"✅ Loaded {loaded_count}/{len(model_keys)} models from cache")
        return models, scalers
        
    except Exception as e:
        print(f"❌ Failed to load from cache: {e}")
        return None, None

def clear_model_cache():
    """Clear the model cache"""
    import shutil
    if os.path.exists(MODEL_CACHE_DIR):
        shutil.rmtree(MODEL_CACHE_DIR)
        print("🗑️ Model cache cleared")
        return True
    return False