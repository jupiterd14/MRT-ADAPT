import pickle

with open('cache/cached_predictions.pkl', 'rb') as f:
    c = pickle.load(f)

# Clamp counter
at_100 = sum(1 for v in c.values() if v >= 99.99)
print(f"Cache: {len(c)} entries")
print(f"Clamped at 100%: {at_100} ({at_100/len(c)*100:.2f}%)")
print()

# Full MRT-3 curve comparison
stations_to_check = [
    ("North Ave", "Northbound"),
    ("Cubao", "Northbound"),
    ("Ayala Ave", "Southbound"),
    ("Taft", "Southbound"),
]

for station, direction in stations_to_check:
    print("=" * 60)
    print(f"{station} {direction} — Monday")
    print("=" * 60)
    for hr in range(5, 23):
        v = c.get(f"{station}_{direction}_0_{hr}")
        if v is not None:
            bar = "█" * int(v / 2)
            print(f"  {hr:02d}:00 → {v:5.1f}%  {bar}")
    print()