from app.config import default_config

cfg = default_config()
print("Photos folder:", cfg.photos_root)
print("DB path:", cfg.db_path)
print("Folder exists?", cfg.photos_root.exists())
