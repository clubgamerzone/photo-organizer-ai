import sqlite3
conn = sqlite3.connect(r'D:\My projects\photo-organizer-ai\data\photo_index.sqlite')

print("Tables:")
for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    print(" -", row[0])

print("\nphotos schema:")
for col in conn.execute("PRAGMA table_info(photos)"):
    print(" ", col)

print("\nCount in photos:", conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0])
print("Count in ai_tags:", conn.execute("SELECT COUNT(*) FROM ai_tags").fetchone()[0])

# Check if any from Work folder
print("\nPhotos from Work folder:")
count = conn.execute("SELECT COUNT(*) FROM photos WHERE path LIKE '%Pictures\\Work%'").fetchone()[0]
print(" Count:", count)

conn.close()
