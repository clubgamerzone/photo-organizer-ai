from dataclasses import dataclass
from pathlib import Path

# A "dataclass" is just an easy way to store some values together.
# Think of it like a small box with labels.
@dataclass(frozen=True)
class AppConfig:
    photos_root: Path  # where your photos are stored
    db_path: Path      # where we will save the database file

def default_config() -> AppConfig:
    # ✅ CHANGE THIS to your real photos folder (example below):
    photos_root = Path(r"C:\Users\jay\Downloads")

    # This creates: project_folder/data/photo_index.sqlite
    db_path = Path(__file__).resolve().parents[1] / "data" / "photo_index.sqlite"

    # Make sure the "data" folder exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    return AppConfig(photos_root=photos_root, db_path=db_path)
