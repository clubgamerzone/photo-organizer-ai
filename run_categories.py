from app.config import default_config
from app.db import connect, init_ai_tables


def ensure_category_column(conn):
    # Adds the column once. If it already exists, SQLite throws an error, we ignore it.
    try:
        conn.execute("ALTER TABLE ai_tags ADD COLUMN category TEXT;")
        conn.commit()
        print("✅ Added category column to ai_tags")
    except Exception:
        print("ℹ️ category column already exists")


def normalize_csv(s):
    if not s:
        return set()
    return {x.strip().lower() for x in s.split(",") if x.strip()}


def has_any(text: str, keywords):
    t = text.lower()
    return any(k in t for k in keywords)


def decide_category(filename: str, caption, objects_csv) -> str:
    filename_l = filename.lower()
    caption_l = (caption or "").lower()
    objs = normalize_csv(objects_csv)

    # 1) Screenshots (highest priority)
    screenshot_keywords = ["screenshot", "screen shot", "screenrecord", "screen record", "chatgpt image"]
    if has_any(filename_l, screenshot_keywords):
        return "Screenshots"

    # 2) Game Assets (your art/UI files)
    asset_keywords = [
        "pixel art", "sprite", "spritesheet", "icon", "ui", "button", "panel",
        "frame", "banner", "logo", "texture", "tileset", "hud", "health bar",
        "papyrus", "background", "border"
    ]
    if has_any(caption_l, asset_keywords) or has_any(filename_l, asset_keywords):
        return "Game Assets"

    # 3) Pets (YOLO objects)
    pet_objects = {"dog", "cat", "bird", "horse", "sheep", "cow", "bear"}
    if objs.intersection(pet_objects):
        return "Pets"

    # 4) People (YOLO objects)
    if "person" in objs:
        return "People"

    return "Other"


def main():
    cfg = default_config()
    conn = connect(cfg.db_path)
    init_ai_tables(conn)
    ensure_category_column(conn)

    rows = conn.execute(
        """
        SELECT photos.path, photos.filename, ai_tags.caption, ai_tags.objects
        FROM photos
        JOIN ai_tags ON ai_tags.path = photos.path
        WHERE ai_tags.caption IS NOT NULL OR ai_tags.objects IS NOT NULL
        """
    ).fetchall()

    updated = 0
    for path, filename, caption, objects_csv in rows:
        category = decide_category(filename, caption, objects_csv)
        conn.execute(
            "UPDATE ai_tags SET category = ? WHERE path = ?",
            (category, path),
        )
        updated += 1

    conn.commit()
    conn.close()
    print(f"✅ Categorized {updated} items.")


if __name__ == "__main__":
    main()
