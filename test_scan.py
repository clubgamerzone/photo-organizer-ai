import sqlite3
from pathlib import Path

DB_PATH = Path(r"D:\My projects\photo-organizer-ai\data\photo_index.sqlite")
conn = sqlite3.connect(str(DB_PATH))

# add your folder path here
folder_path = r"C:\Users\jay\Pictures\Work"
include_subfolders = True
max_files = 20  # test with just 20

root = Path(folder_path)
iterator = root.rglob("*") if include_subfolders else root.glob("*")
allowed_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

added = 0
checked = 0

for p in iterator:
    if not p.is_file():
        continue
    if p.suffix.lower() not in allowed_ext:
        continue
    
    checked += 1
    if max_files and checked > max_files:
        break
    
    try:
        stat = p.stat()
        size_bytes = int(stat.st_size)
        mtime = int(stat.st_mtime)
    except Exception as e:
        print(f"Stat failed: {p} - {e}")
        continue
    
    # Insert with ext column
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO photos(path, filename, ext, size_bytes, mtime)
        VALUES (?, ?, ?, ?, ?)
        """,
        (str(p), p.name, p.suffix.lower(), size_bytes, mtime),
    )
    
    if cur.rowcount == 1:
        conn.execute(
            "INSERT OR IGNORE INTO ai_tags(path, caption, objects, category) VALUES (?, NULL, NULL, NULL)",
            (str(p),),
        )
        added += 1
        print(f"Added: {p.name}")

conn.commit()
print(f"\n✅ Added {added} new images (checked {checked})")

# Verify
count = conn.execute("SELECT COUNT(*) FROM photos WHERE path LIKE '%Pictures\\\\Work%'").fetchone()[0]
print(f"Total Work photos in DB: {count}")

conn.close()
