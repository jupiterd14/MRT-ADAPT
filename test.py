"""
Measure peak memory and CPU time for cache‑only startup.
Run this after you've generated cache/ files.
"""

import psutil
import os
import gc
import time
import sys

sys.path.insert(0, os.path.dirname(__file__))

def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

print(f"🔹 Initial Memory: {get_mem():.2f} MB")
gc.collect()

start_time = time.time()

# Import the app – this will run the cache‑only startup
from app import app

# Force the app context and load caches (the startup block will run)
with app.app_context():
    # The caches are already loaded during import, but we need to ensure
    # the global variables are populated. Just accessing them is enough.
    from routes.api_predict import _PREDICTION_CACHE, _P90_CACHE
    print(f"   Predictions loaded: {len(_PREDICTION_CACHE)}")
    print(f"   P90 values loaded: {len(_P90_CACHE)}")

gc.collect()
time.sleep(1)  # let memory settle

peak_mem = get_mem()
duration = time.time() - start_time

print("=" * 40)
print(f"✅ Peak Memory Usage: {peak_mem:.2f} MB ({peak_mem / 1024:.2f} GB)")
print(f"⏱️  Startup Duration: {duration:.2f} seconds")
print("=" * 40)