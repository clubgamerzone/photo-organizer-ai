from app.config import default_config
from app.db import connect


def main():
    cfg = default_config()
    conn = connect(cfg.db_path)

    # hashes that appear more than once
    dup_hashes = conn.execute(
        """
        SELECT sha256, COUNT(*) as c
        FROM file_hashes
        GROUP BY sha256
        HAVING c > 1
        ORDER BY c DESC
        """
    ).fetchall()

    print("Duplicate groups:", len(dup_hashes))

    # show top 20 groups
    for i, (sha256, count) in enumerate(dup_hashes[:20], start=1):
        print(f"\n=== Group {i}: {count} exact duplicates ===")
        rows = conn.execute(
            "SELECT path, size_bytes FROM file_hashes WHERE sha256 = ? ORDER BY size_bytes DESC",
            (sha256,),
        ).fetchall()

        for (path, size_bytes) in rows:
            mb = size_bytes / (1024 * 1024)
            print(f'  {mb:.2f} MB  "{path}"')

    conn.close()


if __name__ == "__main__":
    main()
