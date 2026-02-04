from app.config import default_config
from app.db import connect, init_db
from app.indexer import scan_and_save_photos
from app.db import init_ai_tables

def main():
    cfg = default_config()

    print("📁 Scanning:", cfg.photos_root)
    print("🗄️ Database:", cfg.db_path)

    conn = connect(cfg.db_path)
    init_db(conn)
    init_ai_tables(conn)

    total = scan_and_save_photos(cfg.photos_root, conn)

    conn.close()
    print(f"✅ Done! Indexed {total} images.")


if __name__ == "__main__":
    main()
