from app.config import default_config
from app.db import connect


def search_by_name(conn, text: str, limit: int = 20):
    """
    Find photos where the filename contains some text.
    Example: text="IMG" finds IMG_001.jpg etc.
    """
    rows = conn.execute(
        """
        SELECT path, size_bytes, width, height
        FROM photos
        WHERE filename LIKE ?
        ORDER BY size_bytes DESC
        LIMIT ?
        """,
        (f"%{text}%", limit),
    ).fetchall()

    return rows


def biggest_photos(conn, limit: int = 20):
    """
    Show the biggest files by size.
    """
    rows = conn.execute(
        """
        SELECT path, size_bytes, width, height
        FROM photos
        ORDER BY size_bytes DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    return rows


def newest_photos(conn, limit: int = 20):
    """
    Show most recently modified photos.
    """
    rows = conn.execute(
        """
        SELECT path, size_bytes, width, height, mtime
        FROM photos
        ORDER BY mtime DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    return rows


def print_rows(rows):
    """
    Pretty-print results.
    """
    for i, r in enumerate(rows, start=1):
        # r is a tuple, depending on query:
        # (path, size_bytes, width, height) or (path, size_bytes, width, height, mtime)
        path = r[0]
        size_bytes = r[1]
        width = r[2]
        height = r[3]

        size_mb = size_bytes / (1024 * 1024)

        print(f"{i}. {size_mb:.2f} MB | {width}x{height} | {path}")


def main():
    cfg = default_config()
    conn = connect(cfg.db_path)

    print("\nChoose an option:")
    print("1) Search by filename text")
    print("2) Show biggest photos")
    print("3) Show newest photos")

    choice = input("Type 1, 2, or 3: ").strip()

    if choice == "1":
        text = input("Type text to search (example: IMG, Screenshot): ").strip()
        rows = search_by_name(conn, text, limit=30)
        print(f"\nFound {len(rows)} results (showing up to 30):\n")
        print_rows(rows)

    elif choice == "2":
        rows = biggest_photos(conn, limit=30)
        print("\nBiggest 30 photos:\n")
        print_rows(rows)

    elif choice == "3":
        rows = newest_photos(conn, limit=30)
        print("\nNewest 30 photos:\n")
        print_rows(rows)

    else:
        print("Invalid choice. Please run again and type 1, 2, or 3.")

    conn.close()


if __name__ == "__main__":
    main()
