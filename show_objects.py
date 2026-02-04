from app.config import default_config
from app.db import connect


def main():
    cfg = default_config()
    conn = connect(cfg.db_path)

    rows = conn.execute(
        """
        SELECT photos.filename, ai_tags.objects, photos.path
        FROM ai_tags
        JOIN photos ON photos.path = ai_tags.path
        WHERE ai_tags.objects IS NOT NULL
        ORDER BY photos.mtime DESC
        LIMIT 20
        """
    ).fetchall()

    for i, (filename, objects, path) in enumerate(rows, start=1):
        print(f"{i}. {filename}")
        print(f"   🧠 objects: {objects}")
        print(f'   📍 "{path}"\n')

    conn.close()


if __name__ == "__main__":
    main()
