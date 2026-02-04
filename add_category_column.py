import sqlite3

db_path = r"D:\My projects\photo-organizer-ai\data\photo_index.sqlite"

conn = sqlite3.connect(db_path)

try:
    conn.execute("ALTER TABLE ai_tags ADD COLUMN category TEXT;")
    conn.commit()
    print("✅ category column added")
except Exception as e:
    print("ℹ️ category column already exists or error:", e)

# Show final table structure
rows = conn.execute("PRAGMA table_info(ai_tags);").fetchall()
print("\nCurrent ai_tags columns:")
for r in rows:
    print(r)

conn.close()
