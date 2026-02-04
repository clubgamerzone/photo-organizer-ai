from app.config import default_config
from app.db import connect


def main():
    cfg = default_config()
    conn = connect(cfg.db_path)

    category = input("Category to show (People / Pets / Game Assets / Screenshots / Other): ").strip()
    if not category:
        print("Please type a category name.")
        return

    limit_text = input("How many results to show? (default 30): ").strip()
    limit = int(limit_text) if limit_text else 30

    rows = conn.execute(
        """
        SELECT photos.filename, ai_tags.category, ai_tags.objects, ai_tags.caption, photos.path
        FROM ai_tags
        JOIN photos ON photos.path = ai_tags.path
        WHERE ai_tags.category = ?
        ORDER BY photos.mtime DESC
        LIMIT ?
        """,
        (category, limit),
    ).fetchall()

    print(f"\nShowing {len(rows)} items in category: {category}\n")

    for i, (filename, cat, objects_csv, caption, path) in enumerate(rows, start=1):
        print(f"{i}. {filename}")
        print(f"   🏷️ category: {cat}")
        print(f"   🧠 objects: {objects_csv or ''}")
        print(f"   ✍️ caption: {caption or ''}")
        print(f'   📍 "{path}"\n')  # quotes help with spaces

    conn.close()


if __name__ == "__main__":
    main()
