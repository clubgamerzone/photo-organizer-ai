from app.config import default_config
from app.db import connect


def main():
    cfg = default_config()
    conn = connect(cfg.db_path)

    rows = conn.execute(
        """
        SELECT photos.filename, ai_tags.category, ai_tags.objects, ai_tags.caption, photos.path
        FROM ai_tags
        JOIN photos ON photos.path = ai_tags.path
        WHERE ai_tags.category IS NOT NULL
        ORDER BY photos.mtime DESC
        LIMIT 30
        """
    ).fetchall()

    for i, (filename, category, objects_csv, caption, path) in enumerate(rows, start=1):
        print(f"{i}. {filename}")
        print(f"   🏷️ category: {category}")
        print(f"   🧠 objects: {objects_csv or ''}")
        print(f"   ✍️ caption: {caption or ''}")
        print(f'   📍 "{path}"\n')  # quotes fix spaces

    conn.close()


if __name__ == "__main__":
    main()
