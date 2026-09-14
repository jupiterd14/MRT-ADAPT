import sqlite3
con = sqlite3.connect('mrt.db')
cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cur.fetchall()]
print("Tables:", tables)
print()
for t in tables:
    try:
        n = cur.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        print(f"{t:30s} {n}")
    except Exception as e:
        print(f"{t:30s} ERROR: {e}")
print()
print("integrity_check:", cur.execute("PRAGMA integrity_check").fetchone()[0])
