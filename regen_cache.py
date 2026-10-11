"""Regenerate prediction cache with corrected typical patterns.

Run this OFFLINE (dev machine) whenever:
- Models are retrained
- Correction factors change
- The cache is corrupted with fallback values

The resulting cache/cached_predictions.pkl is committed to git so Render
can run in cache-only mode without loading models.
"""

from app import app, preload_all_models
from routes.api_predict import precompute_all_predictions, _PREDICTION_CACHE
import pickle
import os

with app.app_context():
    print("=" * 60)
    print("REGENERATING PREDICTION CACHE")
    print("=" * 60)

    # ── Step 1: Clear any existing in-memory cache ──
    _PREDICTION_CACHE.clear()
    print(f"Cleared in-memory cache")

    # ── Step 2: Force-load models into the app ──
    preload_all_models()

    # ── Step 3: Verify models are actually in app.config ──
    from flask import current_app
    from services.model_loader import directional_models, directional_scalers

    print(f"\nModel verification:")
    print(f"  services.directional_models: {len(directional_models)}")
    print(f"  services.directional_scalers: {len(directional_scalers)}")
    print(f"  app.config DIRECTIONAL_MODELS: {len(current_app.config.get('DIRECTIONAL_MODELS', {}))}")
    print(f"  app.config DIRECTIONAL_SCALERS: {len(current_app.config.get('DIRECTIONAL_SCALERS', {}))}")

    if not current_app.config.get('DIRECTIONAL_MODELS'):
        print("\n❌ ABORT: app.config['DIRECTIONAL_MODELS'] is empty!")
        print("   precompute_all_predictions() would run with no models and produce fallback values.")
        print("   Check that preload_all_models() actually populated app.config.")
        raise SystemExit(1)

    # ── Step 4: Precompute ──
    print(f"\n🔄 Running precompute_all_predictions()...")
    precompute_all_predictions()

    total = len(_PREDICTION_CACHE)
    expected = 13 * 2 * 7 * 24   # 13 stations × 2 directions × 7 days × 24 hours
    print(f"\nCache size: {total} (expected: {expected})")

    if total == 0:
        print("\n❌ ABORT: Cache is empty. Precompute failed.")
        raise SystemExit(1)

    # ── Step 5: Detect fallback poisoning ──
    # Fallback values are exact multiples of 5 in a narrow set
    fallback_signatures = {65.0, 45.0, 25.0, 10.0, 0.0}
    fallback_hits = sum(1 for v in _PREDICTION_CACHE.values() if v in fallback_signatures)
    fallback_pct = (fallback_hits / total) * 100

    print(f"\nFallback signature check:")
    print(f"  Entries matching fallback values: {fallback_hits}/{total} ({fallback_pct:.1f}%)")

    if fallback_pct > 50:
        print("\n❌ WARNING: Cache is dominated by fallback values!")
        print("   This means precompute ran without models loaded.")
        print("   Check that preload_all_models() succeeded.")
        raise SystemExit(1)
    elif fallback_pct > 20:
        print(f"\n⚠️ CAUTION: {fallback_pct:.1f}% of values match fallback pattern.")
        print("   Verify before committing.")
    else:
        print(f"  ✅ Cache looks healthy")

    # ── Step 6: Sanity check specific entries ──
    print(f"\nSanity check — North Ave Northbound:")
    checks = [(0, 8, "Mon 08:00"), (0, 12, "Mon 12:00"),
              (0, 18, "Mon 18:00"), (0, 21, "Mon 21:00"),
              (5, 8, "Sat 08:00"), (6, 12, "Sun 12:00")]
    for dow, hr, label in checks:
        key = f"North Ave_Northbound_{dow}_{hr}"
        v = _PREDICTION_CACHE.get(key)
        if v is not None:
            flag = "⚠️ FALLBACK" if v in fallback_signatures else "✅"
            print(f"  {label}: {v:.1f}%  {flag}")
        else:
            print(f"  {label}: MISSING")

    print(f"\nSanity check — Buendia Northbound:")
    for dow in range(7):
        key = f"Buendia_Northbound_{dow}_5"
        v = _PREDICTION_CACHE.get(key)
        if v is not None:
            print(f"  {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]} 05:00: {v:.1f}%")

    print(f"\nSanity check — Taft Southbound:")
    for dow, hr in [(0, 8), (0, 18), (6, 12)]:
        key = f"Taft_Southbound_{dow}_{hr}"
        v = _PREDICTION_CACHE.get(key)
        if v is not None:
            print(f"  DOW={dow} {hr:02d}:00: {v:.1f}%")

    # ── Step 7: Persist ──
    os.makedirs('cache', exist_ok=True)
    with open('cache/cached_predictions.pkl', 'wb') as f:
        pickle.dump(dict(_PREDICTION_CACHE), f)
    file_size = os.path.getsize('cache/cached_predictions.pkl') / 1024
    print(f"\n✅ Saved {total} entries to cache/cached_predictions.pkl ({file_size:.1f} KB)")

    # ── Step 8: Verify the file reads back ──
    with open('cache/cached_predictions.pkl', 'rb') as f:
        reloaded = pickle.load(f)
    print(f"✅ Verified: reloaded {len(reloaded)} entries from disk")

    print("\n" + "=" * 60)
    print("DONE — commit cache/cached_predictions.pkl to git")
    print("=" * 60)