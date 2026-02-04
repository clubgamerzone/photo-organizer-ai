from pathlib import Path
from PIL import Image
from tqdm import tqdm

# These are the image file types we care about
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def is_image_file(path: Path) -> bool:
    """
    Returns True if the file has an extension like .jpg or .png
    """
    return path.suffix.lower() in IMAGE_EXTS


def get_image_size(path: Path):
    """
    Tries to read the image width/height.
    If the image is broken or unreadable, returns (None, None).
    """
    try:
        with Image.open(path) as img:
            width, height = img.size
            return width, height
    except Exception:
        return None, None


def scan_and_save_photos(root_folder: Path, conn) -> int:
    """
    Walks through root_folder and all subfolders.
    For every image it finds, it saves info into the database.
    Returns how many images it saved.
    """
    from .db import save_photo  # import here so the file stays simple

    count = 0

    # root_folder.rglob("*") means: "give me EVERYTHING inside this folder (including subfolders)"
    all_paths = list(root_folder.rglob("*"))

    # tqdm adds a progress bar while we loop
    for p in tqdm(all_paths, desc="Scanning files"):
        if not p.is_file():
            continue  # skip folders

        if not is_image_file(p):
            continue  # skip non-image files

        stat = p.stat()  # file info: size, modified time, etc.
        width, height = get_image_size(p)

        data = (
            str(p.resolve()),     # full path as text
            p.name,               # filename only
            p.suffix.lower(),     # extension
            stat.st_size,         # file size in bytes
            stat.st_mtime,        # modified time
            width,
            height
        )

        save_photo(conn, data)
        count += 1

    conn.commit()
    return count
