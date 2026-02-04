import hashlib
from pathlib import Path

from app.config import default_config
from app.db import connect


def ensure_hash_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS file_hashes (
            path TEXT PRIMARY KEY,
            sha256 TEXT,
            size_bytes INTEGER
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_hashes_sha256 ON file_hashes(sha256);")
    conn.commit()


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()

    # Read file in chunks so it works for big files
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def main():
    cfg = default_config()
    conn = connect(cfg.db_path)
    ensure_hash_table(conn)

    # Only hash files we haven't hashed yet
    rows = conn.execute(
        """
        SELECT photos.path, photos.size_bytes
        FROM photos
        LEFT JOIN file_hashes ON file_hashes.path = photos.path
        WHERE file_hashes.path IS NULL
        """
    ).fetchall()

    print("Files to hash:", len(rows))

    done = 0
    for path, size_bytes in rows:
        p = Path(path)
        if not p.exists():
            continue

        try:
            digest = sha256_of_file(path)
            conn.execute(
                "INSERT INTO file_hashes(path, sha256, size_bytes) VALUES (?, ?, ?)",
                (path, digest, size_bytes),
            )
            done += 1

            if done % 200 == 0:
                conn.commit()
                print(f"Hashed {done}...")

        except Exception as e:
            print("❌ Failed:", path, "|", e)

    conn.commit()
    conn.close()
    print("✅ Done. Hashed", done, "files.")


if __name__ == "__main__":
    main()
