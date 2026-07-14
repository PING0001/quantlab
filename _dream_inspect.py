"""Read-only inspection of mimocode.db for dream consolidation."""
import sqlite3
import json
import os

DB = r"C:\Users\cui\.local\share\mimocode\mimocode.db"
conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
c = conn.cursor()

# 1. Tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print("=== TABLES ===")
print(tables)

# 2. Schema for key tables
for t in tables:
    c.execute(f"PRAGMA table_info({t})")
    cols = [(r[1], r[2]) for r in c.fetchall()]
    print(f"\n=== SCHEMA: {t} ===")
    for name, typ in cols:
        print(f"  {name} ({typ})")

# 3. List all sessions
if "session" in tables:
    c.execute("SELECT * FROM session ORDER BY time_created DESC")
    rows = c.fetchall()
    c.execute("PRAGMA table_info(session)")
    session_cols = [r[1] for r in c.fetchall()]
    print(f"\n=== SESSIONS ({len(rows)} rows) ===")
    print(f"Columns: {session_cols}")
    for row in rows:
        d = dict(zip(session_cols, row))
        # Try to parse data JSON
        if 'data' in d and isinstance(d['data'], str):
            try:
                data_json = json.loads(d['data'])
                print(json.dumps(data_json, indent=2, ensure_ascii=False)[:600])
            except:
                print(str(d['data'])[:300])
        else:
            print(d)

conn.close()
