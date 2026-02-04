# ui_app.py
# A local UI for browsing photos, editing tags, and managing exact duplicates.
# Tech: Streamlit + SQLite + Pillow + send2trash (safe delete to Recycle Bin)
from datetime import datetime

import os
import sqlite3
from pathlib import Path

from run_categories import decide_category
import streamlit as st
from PIL import Image
from send2trash import send2trash
from app.ai_yolo import detect_objects

# =========================
# 1) CONFIG
# =========================
# Path to your SQLite database file (created by your indexer scripts)
DB_PATH = Path(r"D:\My projects\photo-organizer-ai\data\photo_index.sqlite")


# =========================
# 2) IMAGE HELPERS
# =========================
def fit_image(path: Path, size=(256, 256)) -> Image.Image:
    """
    Makes all thumbnails look consistent in the grid.

    - Opens image
    - Resizes it to fit inside `size` while keeping aspect ratio
    - Places it centered on a transparent canvas

    This avoids weird layouts when images have different sizes.
    """
    img = Image.open(path).convert("RGBA")
    img.thumbnail(size, Image.LANCZOS)

    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas

# =========================
# YOLO (objects detection)
# =========================
@st.cache_resource



def open_containing_folder(path: str):
    """
    Opens the folder that contains the file in Windows Explorer.
    """
    folder = str(Path(path).parent)
    os.startfile(folder)


def open_all_unique_folders(paths: list[str]):
    """
    If duplicates exist in multiple folders, this opens each unique folder.
    WARNING: can open multiple Explorer windows.
    """
    folders = sorted(set(str(Path(p).parent) for p in paths))
    for f in folders:
        os.startfile(f)


# =========================
# 3) DB HELPERS
# =========================
def connect_db():
    """
    Connects to SQLite.
    check_same_thread=False allows Streamlit to reuse connection across reruns.
    """
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)

def fetch_ignored_groups(conn, limit_groups: int, offset: int):
    """
    Returns ignored duplicate groups (sha256 + note + date), with pagination.
    """
    return conn.execute(
        """
        SELECT sha256, note, created_at
        FROM duplicate_ignores
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        (limit_groups, offset),
    ).fetchall()


def unignore_group(conn, sha256: str):
    """
    Removes a sha256 from the ignore list so it shows again in Duplicates.
    """
    conn.execute("DELETE FROM duplicate_ignores WHERE sha256 = ?", (sha256,))
    conn.commit()


def ensure_categories_table(conn):
    """
    Stores a list of available categories so we can show them in a dropdown.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            name TEXT PRIMARY KEY
        );
    """)
    conn.commit()


def ensure_hash_table(conn):
    """
    Stores exact-file hashes for duplicate detection.
    These hashes are created by your script: run_file_hashes.py
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS file_hashes (
            path TEXT PRIMARY KEY,
            sha256 TEXT,
            size_bytes INTEGER
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_hashes_sha256 ON file_hashes(sha256);")
    conn.commit()


def ensure_scan_history_table(conn):
    """
    Tracks scan history: which folders were scanned, when, and how many images found.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_path TEXT NOT NULL,
            scanned_at TEXT DEFAULT (datetime('now')),
            total_images INTEGER DEFAULT 0,
            new_images INTEGER DEFAULT 0,
            include_subfolders INTEGER DEFAULT 1
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scan_history_folder ON scan_history(folder_path);")
    conn.commit()


def get_scan_history(conn, limit: int = 10):
    """
    Returns recent scan history.
    """
    return conn.execute("""
        SELECT folder_path, scanned_at, total_images, new_images, include_subfolders
        FROM scan_history
        ORDER BY scanned_at DESC
        LIMIT ?
    """, (limit,)).fetchall()


def get_folder_stats(conn, folder_path: str):
    """
    Returns stats for a specific folder: last scan date, total images, previous new count.
    """
    row = conn.execute("""
        SELECT scanned_at, total_images, new_images
        FROM scan_history
        WHERE folder_path = ?
        ORDER BY scanned_at DESC
        LIMIT 1
    """, (folder_path,)).fetchone()
    return row


def count_images_in_folder(conn, folder_path: str) -> int:
    """
    Counts how many images from this folder are already in the database.
    """
    row = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE path LIKE ?",
        (folder_path + "%",)
    ).fetchone()
    return row[0] if row else 0


def record_scan(conn, folder_path: str, total_images: int, new_images: int, include_subfolders: bool):
    """
    Records a scan in history.
    """
    conn.execute("""
        INSERT INTO scan_history (folder_path, total_images, new_images, include_subfolders)
        VALUES (?, ?, ?, ?)
    """, (folder_path, total_images, new_images, 1 if include_subfolders else 0))
    conn.commit()


def ensure_duplicate_ignore_table(conn):
    """
    Stores sha256 hashes of duplicate groups that you want to ignore.

    Example:
    - you *want* the same file in multiple project folders,
      so you mark that group as "not duplicate" and it disappears from the list.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS duplicate_ignores (
            sha256 TEXT PRIMARY KEY,
            note TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()


def seed_categories_from_ai_tags(conn):
    """
    Takes any categories already assigned in ai_tags and adds them into categories table.
    Safe to run repeatedly.
    """
    rows = conn.execute(
        "SELECT DISTINCT category FROM ai_tags WHERE category IS NOT NULL AND TRIM(category) != ''"
    ).fetchall()

    for (cat,) in rows:
        conn.execute("INSERT OR IGNORE INTO categories(name) VALUES (?)", (cat,))

    conn.commit()


def fetch_categories(conn) -> list[str]:
    """
    Returns categories sorted alphabetically (for dropdown).
    """
    rows = conn.execute("SELECT name FROM categories ORDER BY name").fetchall()
    return [r[0] for r in rows]


def add_category(conn, name: str):
    """
    Adds a new category to the categories table if it doesn't already exist.
    """
    name = name.strip()
    if not name:
        return
    conn.execute("INSERT OR IGNORE INTO categories(name) VALUES (?)", (name,))
    conn.commit()


def fetch_rows(conn, category_filter: str, search: str, limit: int, offset: int):
    """
    Returns photos + ai_tags for the gallery, with filters, using LIMIT/OFFSET pagination.
    Automatically removes missing files from the database.
    """
    sql = """
    SELECT photos.path, photos.filename, ai_tags.category, ai_tags.caption, ai_tags.objects
    FROM photos
    JOIN ai_tags ON ai_tags.path = photos.path
    WHERE 1=1
    """
    params = []

    if category_filter != "All":
        sql += " AND ai_tags.category = ?"
        params.append(category_filter)

    if search:
        sql += " AND (photos.filename LIKE ? OR ai_tags.caption LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])

    sql += " ORDER BY photos.mtime DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(sql, params).fetchall()
    
    # Filter out missing files and clean them from DB
    valid_rows = []
    for row in rows:
        path = row[0]
        if Path(path).exists():
            valid_rows.append(row)
        else:
            # File was deleted manually, remove from DB
            conn.execute("DELETE FROM photos WHERE path=?", (path,))
            conn.execute("DELETE FROM ai_tags WHERE path=?", (path,))
            conn.execute("DELETE FROM file_hashes WHERE path=?", (path,))
    
    if len(valid_rows) != len(rows):
        conn.commit()  # Save the cleanup
    
    return valid_rows


def update_tags(conn, path: str, caption: str, objects_csv: str, category: str):
    """
    Saves edited caption, objects list, and category into ai_tags.
    """
    conn.execute(
        "UPDATE ai_tags SET caption=?, objects=?, category=? WHERE path=?",
        (caption, objects_csv, category, path),
    )
    conn.commit()


def rename_file(conn, old_path: str, new_filename: str) -> str:
    """
    Renames the file on disk AND updates database references.

    Important:
    - photos.path and ai_tags.path are used as identifiers,
      so we update them so the UI doesn't break.
    - Automatically preserves the original extension if user doesn't provide one.
    """
    old = Path(old_path)
    new_name = Path(new_filename)
    
    # If user didn't include an extension, preserve the original one
    if not new_name.suffix:
        new_filename = new_filename + old.suffix
    
    new_path = old.with_name(new_filename)

    if new_path.exists():
        raise FileExistsError(f"File already exists: {new_path}")

    old.rename(new_path)

    # Update DB references
    conn.execute(
        "UPDATE photos SET path=?, filename=? WHERE path=?",
        (str(new_path), new_path.name, old_path),
    )
    conn.execute(
        "UPDATE ai_tags SET path=? WHERE path=?",
        (str(new_path), old_path),
    )
    conn.execute(
        "UPDATE file_hashes SET path=? WHERE path=?",
        (str(new_path), old_path),
    )
    conn.commit()

    return str(new_path)


def delete_file(conn, path: str):
    """
    Safe delete:
    - sends file to Recycle Bin (send2trash)
    - removes records from DB tables
    """
    send2trash(path)
    conn.execute("DELETE FROM photos WHERE path=?", (path,))
    conn.execute("DELETE FROM ai_tags WHERE path=?", (path,))
    conn.execute("DELETE FROM file_hashes WHERE path=?", (path,))
    conn.commit()


def fetch_duplicate_groups(conn, limit_groups: int, offset: int):
    """
    Returns duplicate groups by sha256, excluding ignored groups.
    Pagination is supported via LIMIT/OFFSET.
    """
    return conn.execute(
        """
        SELECT fh.sha256, COUNT(*) as c
        FROM file_hashes fh
        LEFT JOIN duplicate_ignores di ON di.sha256 = fh.sha256
        WHERE di.sha256 IS NULL
        GROUP BY fh.sha256
        HAVING c > 1
        ORDER BY c DESC
        LIMIT ? OFFSET ?
        """,
        (limit_groups, offset),
    ).fetchall()


def fetch_group_files(conn, sha256: str):
    """
    Returns the files that belong to one duplicate group.
    Automatically removes missing files from the database.
    """
    rows = conn.execute(
        """
        SELECT photos.path, photos.filename, photos.size_bytes
        FROM file_hashes
        JOIN photos ON photos.path = file_hashes.path
        WHERE file_hashes.sha256 = ?
        ORDER BY photos.size_bytes DESC
        """,
        (sha256,),
    ).fetchall()
    
    # Filter out missing files and clean them from DB
    valid_files = []
    for path, filename, size_bytes in rows:
        if Path(path).exists():
            valid_files.append((path, filename, size_bytes))
        else:
            # File was deleted manually, remove from DB
            conn.execute("DELETE FROM photos WHERE path=?", (path,))
            conn.execute("DELETE FROM ai_tags WHERE path=?", (path,))
            conn.execute("DELETE FROM file_hashes WHERE path=?", (path,))
    
    if len(valid_files) != len(rows):
        conn.commit()  # Save the cleanup
    
    return valid_files

def fetch_one_file_for_hash(conn, sha256: str):
    """
    Returns ONE file path for a given hash.
    Used to show a preview thumbnail in Unignore page.
    """
    row = conn.execute(
        """
        SELECT photos.path
        FROM file_hashes
        JOIN photos ON photos.path = file_hashes.path
        WHERE file_hashes.sha256 = ?
        LIMIT 1
        """,
        (sha256,),
    ).fetchone()

    return row[0] if row else None

def scan_and_ai_tag_folder(conn, folder_path: str, include_subfolders: bool, max_files: int = 0) -> tuple[int, int]:
    """
    1) Scans folder and inserts NEW photos into DB
    2) Runs AI (YOLO objects + BLIP caption + category) ONLY for new photos
    Returns: (added_count, tagged_count)
    """
    # Step A: scan into DB
    added = scan_folder_into_db(conn, folder_path, include_subfolders, max_files)

    # Get all paths in this folder that still need tags
    # (caption or objects or category missing)
    root = str(Path(folder_path))
    rows = conn.execute(
        """
        SELECT photos.path, photos.filename
        FROM photos
        LEFT JOIN ai_tags ON ai_tags.path = photos.path
        WHERE photos.path LIKE ?
          AND (
              ai_tags.path IS NULL OR
              ai_tags.caption IS NULL OR ai_tags.caption = '' OR
              ai_tags.objects IS NULL OR ai_tags.objects = '' OR
              ai_tags.category IS NULL OR ai_tags.category = ''
          )
        """,
        (root + "%",),
    ).fetchall()

    paths_to_tag = [r[0] for r in rows]

    return added, len(paths_to_tag)

def scan_folder_into_db(conn, folder_path: str, include_subfolders: bool, max_files: int = 0) -> tuple[int, int]:
    """
    Scans a folder (and optionally subfolders) and inserts any image files into the `photos` table.
    Returns (new_images_added, total_images_found).

    Noob explanation:
    - We walk through files in the folder
    - We only keep image extensions
    - For each image, we store path + filename + size + modified time
    - 'INSERT OR IGNORE' prevents duplicates by path (so existing files are skipped!)
    """
    root = Path(folder_path)

    if not root.exists() or not root.is_dir():
        raise ValueError("Folder does not exist or is not a folder.")

    # Choose how we iterate files
    iterator = root.rglob("*") if include_subfolders else root.glob("*")

    allowed_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

    added = 0
    total_found = 0

    for p in iterator:
        if not p.is_file():
            continue

        if p.suffix.lower() not in allowed_ext:
            continue

        total_found += 1
        if max_files and total_found > max_files:
            break

        try:
            stat = p.stat()
            size_bytes = int(stat.st_size)
            mtime = int(stat.st_mtime)
        except Exception:
            continue

        # Insert into photos table
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO photos(path, filename, ext, size_bytes, mtime)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(p), p.name, p.suffix.lower(), size_bytes, mtime),
        )

        # If a row was inserted, also ensure ai_tags row exists
        if cur.rowcount == 1:
            conn.execute(
                "INSERT OR IGNORE INTO ai_tags(path, caption, objects, category) VALUES (?, NULL, NULL, NULL)",
                (str(p),),
            )
            added += 1

    conn.commit()
    
    # Record this scan in history
    record_scan(conn, folder_path, total_found, added, include_subfolders)
    
    return added, total_found


# =========================
# 4) STREAMLIT UI SETUP
# =========================
st.set_page_config(page_title="Photo Organizer AI — Local UI", layout="wide")
st.title("📸 Photo Organizer AI — Local UI")

# CSS: align buttons by forcing fixed height for filenames (optional but nice)
st.markdown(
    """
<style>
.file-name {
  height: 42px;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  color: #9aa4b2;
  font-size: 0.85rem;
  margin-top: 6px;
  margin-bottom: 8px;
}
div.stButton > button {
  width: 100%;
}

/* Make sidebar wider to fit editor */
section[data-testid="stSidebar"] {
  width: 380px !important;
}
section[data-testid="stSidebar"] > div {
  width: 380px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# Connect DB and ensure tables exist
conn = connect_db()
ensure_categories_table(conn)
ensure_hash_table(conn)
ensure_duplicate_ignore_table(conn)
ensure_scan_history_table(conn)
seed_categories_from_ai_tags(conn)

# Session state (Streamlit reruns the script often; session_state keeps UI state)
if "selected_path" not in st.session_state:
    st.session_state.selected_path = None

if "gallery_page" not in st.session_state:
    st.session_state.gallery_page = 0

if "dupe_page" not in st.session_state:
    st.session_state.dupe_page = 0

if "ignore_page" not in st.session_state:
    st.session_state.ignore_page = 0
    
if "busy" not in st.session_state:
    st.session_state.busy = False
# =========================
# 5) SIDEBAR (filters + mode + pagination controls + EDITOR)
# =========================
with st.sidebar:
    st.header("Navigation")

    mode = st.radio("Mode", ["Gallery", "Duplicates", "Unignore"])

    st.divider()
    st.header("Scan / Index")

    # 1) Folder path (user pastes it)
    default_scan_folder = str(Path.home() / "Downloads")  # fallback
    scan_folder = st.text_input("Folder to scan", value=default_scan_folder)

    # Helper: open the folder in Explorer so you can copy the path easily
    if st.button("📂 Open this folder in Explorer"):
        try:
            os.startfile(scan_folder)
        except Exception as e:
            st.error(f"Can't open folder: {e}")

    include_subfolders = st.checkbox("Include subfolders", value=True)
    scan_limit = st.number_input("Max files per scan (0 = no limit)", min_value=0, value=0, step=1000)
    
    # Show current folder stats
    folder_stats = get_folder_stats(conn, scan_folder)
    images_in_db = count_images_in_folder(conn, scan_folder)
    
    if folder_stats:
        last_scan, prev_total, prev_new = folder_stats
        st.info(f"📊 **This folder:**\n- In database: **{images_in_db}** images\n- Last scan: {last_scan}\n- Found {prev_total} images, {prev_new} were new")
    elif images_in_db > 0:
        st.info(f"📊 **{images_in_db}** images from this folder already in database")
    else:
        st.caption("📂 This folder has not been scanned yet")
    
    if st.button("🧠 Scan + AI Tag this folder", disabled=st.session_state.busy):
        st.session_state.busy = True

        try:
            # 1) figure out what needs tagging
            added_count, to_tag_count = scan_and_ai_tag_folder(
                conn,
                scan_folder,
                include_subfolders,
                int(scan_limit),
            )

            st.success(f"✅ Added {added_count} new images. Now tagging {to_tag_count} images...")

            # 2) Load AI models ONCE (important for speed)
            # NOTE: These imports must exist in your project already (YOLO + BLIP)
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            from PIL import Image

            # If you already have YOLO model code in another file, we’ll call it here.
            # For now assume you have a function: yolo_detect_objects(image_path) -> "person,dog,..."
            # If you don't yet, tell me and I’ll paste the YOLO function.

            st.info("Loading BLIP model...")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            # 3) progress UI
            progress = st.progress(0)
            status = st.empty()

            # 4) Pull the paths again (so we can loop)
            root = str(Path(scan_folder))
            rows = conn.execute(
                """
                SELECT photos.path, photos.filename,
                    COALESCE(ai_tags.caption,''), COALESCE(ai_tags.objects,''), COALESCE(ai_tags.category,'')
                FROM photos
                LEFT JOIN ai_tags ON ai_tags.path = photos.path
                WHERE photos.path LIKE ?
                """,
                (root + "%",),
            ).fetchall()

            # filter only ones missing anything
            todo = []
            for path, filename, caption, objects_csv, category in rows:
                if (not caption) or (not objects_csv) or (not category):
                    todo.append((path, filename))

            total = len(todo)

            # 5) loop and tag
            for i, (path, filename) in enumerate(todo, start=1):
                status.write(f"Tagging {i}/{total}: {filename}")

                # --- BLIP caption ---
                caption_text = ""
                try:
                    img = Image.open(path).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    out = model.generate(**inputs, max_new_tokens=30)
                    caption_text = processor.decode(out[0], skip_special_tokens=True)
                except Exception:
                    caption_text = ""

                # --- YOLO objects ---
                objects_text = ""
                try:
                    objects_text = detect_objects(path)
                except Exception:
                    objects_text = ""

                # --- category ---
                category_text = decide_category(filename, caption_text, objects_text)

                # Ensure ai_tags row exists
                conn.execute("INSERT OR IGNORE INTO ai_tags(path) VALUES (?)", (path,))

                # Save tags
                conn.execute(
                    "UPDATE ai_tags SET caption=?, objects=?, category=? WHERE path=?",
                    (caption_text, objects_text, category_text, path),
                )
                conn.commit()

                progress.progress(int(i / total * 100))

            status.success("✅ Done tagging this folder!")
            st.session_state.busy = False
            st.rerun()

        except Exception as e:
            st.session_state.busy = False
            st.error(str(e))

    
    if st.button("🚀 Scan / Index this folder now", disabled=st.session_state.busy):
        try:
            new_added, total_found = scan_folder_into_db(conn, scan_folder, include_subfolders, int(scan_limit))
            if new_added > 0:
                st.success(f"✅ Found {total_found} images, **{new_added} NEW** added to database!")
            else:
                st.info(f"📂 Found {total_found} images, all already in database (0 new)")
            st.rerun()
        except Exception as e:
            st.error(str(e))
    
    # Show recent scan history
    with st.expander("📜 Recent Scan History"):
        history = get_scan_history(conn, limit=10)
        if history:
            for folder_path, scanned_at, total_img, new_img, subfolders in history:
                folder_short = folder_path if len(folder_path) < 30 else "..." + folder_path[-27:]
                sub_icon = "📁" if subfolders else "📄"
                st.caption(f"{sub_icon} **{folder_short}**")
                st.caption(f"   {scanned_at} — {total_img} found, {new_img} new")
        else:
            st.caption("No scans yet")
            
    st.divider()
    st.header("Filters")

    all_cats = fetch_categories(conn)
    category_filter = st.selectbox("Category filter", ["All"] + all_cats)

    search = st.text_input("Search (filename or caption)", "")

    # Page size controls (pagination)
    # Page size controls (pagination)
    if mode == "Gallery":
        page_size = st.selectbox("Gallery page size", [60, 120, 240], index=1)
        if st.button("Reset gallery to page 1"):
            st.session_state.gallery_page = 0

    elif mode == "Duplicates":
        groups_page_size = st.selectbox("Duplicate groups per page", [25, 50, 100, 200], index=2)
        if st.button("Reset duplicates to page 1"):
            st.session_state.dupe_page = 0

    else:  # Unignore
        ignored_page_size = st.selectbox("Ignored groups per page", [25, 50, 100, 200], index=1)
        if st.button("Reset ignored to page 1"):
            st.session_state.ignore_page = 0

    st.divider()

    # =========================
    # EDITOR (now in sidebar - always sticky!)
    # =========================
    st.subheader("🛠️ Editor")
    sel = st.session_state.selected_path

    if not sel:
        st.info("Select a photo in Gallery to edit it here.")
    else:
        row = conn.execute(
            """
            SELECT photos.path, photos.filename, ai_tags.category, ai_tags.caption, ai_tags.objects
            FROM photos
            JOIN ai_tags ON ai_tags.path = photos.path
            WHERE photos.path = ?
            """,
            (sel,),
        ).fetchone()

        if not row:
            st.error("Selected photo not found in DB (maybe deleted).")
        else:
            path, filename, cat, caption, objects_csv = row
            p = Path(path)

            if p.exists():
                st.image(str(p), use_container_width=True)

            st.markdown(f"**{filename}**")
            st.caption(f'📍 "{path}"')

            # edit fields
            caption_val = st.text_area("Caption", caption or "", height=90)
            objects_val = st.text_input("Objects (comma-separated)", objects_csv or "")

            # category dropdown + add new
            categories = fetch_categories(conn)
            if cat and cat not in categories:
                categories = [cat] + categories

            add_new_label = "➕ Add new category..."
            cat_choice = st.selectbox(
                "Category",
                options=categories + [add_new_label],
                index=(categories.index(cat) if cat in categories else 0) if categories else 0,
            )

            final_category = cat_choice
            if cat_choice == add_new_label:
                new_category_name = st.text_input("New category name", "")
                final_category = new_category_name.strip()

            c1, c2 = st.columns(2)
            with c1:
                if st.button("💾 Save tags"):
                    if not final_category:
                        st.error("Category can't be empty.")
                    else:
                        add_category(conn, final_category)
                        update_tags(conn, path, caption_val, objects_val, final_category)
                        st.success("Saved!")

            with c2:
                if st.button("📂 Open folder", disabled=st.session_state.busy):
                    open_containing_folder(path)

            st.divider()

            # rename
            new_name = st.text_input("Rename to (filename)", filename)
            if st.button("✏️ Rename file", disabled=st.session_state.busy):
                try:
                    new_path = rename_file(conn, path, new_name)
                    st.session_state.selected_path = new_path
                    st.success("Renamed!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

            # delete
            st.warning("Delete sends the file to Recycle Bin.")
            if st.button("🗑️ Delete (Recycle Bin)", disabled=st.session_state.busy):
                try:
                    delete_file(conn, path)
                    st.session_state.selected_path = None
                    st.warning("Deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))


# Layout: full width for main content (editor is now in sidebar)


# =========================
# 6) MAIN PANEL
# =========================
if mode == "Gallery":
    st.subheader("🖼️ Gallery")

    # Pagination math
    offset = st.session_state.gallery_page * page_size
    rows = fetch_rows(conn, category_filter, search, page_size, offset)

    # Pager buttons
    p1, p2, p3 = st.columns([1, 1, 2])
    with p1:
        if st.button("⬅️ Prev", disabled=st.session_state.gallery_page == 0):
            st.session_state.gallery_page -= 1
            st.rerun()
    with p2:
        # if rows < page_size, it likely means this is the last page
        if st.button("Next ➡️", disabled=len(rows) < page_size):
            st.session_state.gallery_page += 1
            st.rerun()
    with p3:
        st.write(f"Page: **{st.session_state.gallery_page + 1}**")

    st.write(f"Showing **{len(rows)}** results")

    # Grid
    cols_per_row = 5
    grid_cols = st.columns(cols_per_row)

    for idx, (path, filename, cat, caption, objects_csv) in enumerate(rows):
        p = Path(path)
        col = grid_cols[idx % cols_per_row]

        with col:
            if p.exists():
                st.image(fit_image(p, size=(256, 256)), use_container_width=True)
            else:
                st.error("Missing")

            safe_name = (filename or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(f"<div class='file-name'>{safe_name}</div>", unsafe_allow_html=True)

            # Buttons row: Select | Delete
            b1, b2 = st.columns(2)

            with b1:
                if st.button("Select", key=f"sel_{path}"):
                    st.session_state.selected_path = path
                    st.rerun()

            with b2:
                if st.button("🗑️", key=f"del_{path}", help="Delete (Recycle Bin)"):
                    try:
                        delete_file(conn, path)
                        if st.session_state.selected_path == path:
                            st.session_state.selected_path = None
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

elif mode == "Duplicates":
    st.subheader("🧬 Exact Duplicates")

    # Pagination math for duplicate groups
    dupe_offset = st.session_state.dupe_page * groups_page_size
    groups = fetch_duplicate_groups(conn, groups_page_size, dupe_offset)

    # Pager buttons for groups
    d1, d2, d3 = st.columns([1, 1, 2])
    with d1:
        if st.button("⬅️ Prev groups", disabled=st.session_state.dupe_page == 0):
            st.session_state.dupe_page -= 1
            st.rerun()
    with d2:
        if st.button("Next groups ➡️", disabled=len(groups) < groups_page_size):
            st.session_state.dupe_page += 1
            st.rerun()
    with d3:
        st.write(f"Groups page: **{st.session_state.dupe_page + 1}**")

    st.write(f"Found **{len(groups)}** duplicate groups on this page (ignored groups hidden).")

    if not groups:
        st.info("No exact duplicates found on this page. Make sure you ran: py run_file_hashes.py")
    else:
        # Create readable labels to choose a group
        group_labels = [f"{count} files  |  {sha[:12]}..." for (sha, count) in groups]

        selected_index = st.selectbox(
            "Pick a duplicate group",
            range(len(group_labels)),
            format_func=lambda i: group_labels[i],
        )

        # IMPORTANT: set selected_sha BEFORE using it
        selected_sha, selected_count = groups[selected_index]
        files = fetch_group_files(conn, selected_sha)

        st.subheader(f"Group: {selected_count} exact duplicates")

        # Choose which file to keep
        keep_path = st.selectbox(
            "✅ Keep this one (we delete the others to Recycle Bin)",
            [f[0] for f in files],
            format_func=lambda x: Path(x).name,
        )

        # Folder tools
        paths_in_group = [f[0] for f in files]
        cA, cB = st.columns(2)
        with cA:
            if st.button("📂 Open folder (kept file)"):
                open_containing_folder(keep_path)
        with cB:
            if st.button("📂 Open ALL folders in this group"):
                st.info("This may open multiple Explorer windows.")
                open_all_unique_folders(paths_in_group)

        # Ignore tool (mark this group as not duplicates)
        ignore_note = st.text_input("Ignore note (optional)", "")
        if st.button("🙈 Mark this group as NOT duplicates (ignore)"):
            conn.execute(
                "INSERT OR REPLACE INTO duplicate_ignores(sha256, note) VALUES (?, ?)",
                (selected_sha, ignore_note.strip()),
            )
            conn.commit()
            st.success("Ignored. This group will no longer show up.")
            st.rerun()

        # Show images
        cols = st.columns(4)
        for i, (path, filename, size_bytes) in enumerate(files):
            p = Path(path)
            with cols[i % 4]:
                if p.exists():
                    st.image(str(p), use_container_width=True)
                else:
                    st.error("Missing")
                st.caption(filename)
                st.caption(f"{(size_bytes or 0) / (1024 * 1024):.2f} MB")
                st.code(f'"{path}"', language=None)

        # Delete all except kept file
        st.warning("Delete is SAFE: it sends files to Recycle Bin.")
        
        del_col1, del_col2 = st.columns(2)
        
        with del_col1:
            if st.button("🗑️ Delete duplicates (keep 1)"):
                deleted = 0

                for (path, filename, size_bytes) in files:
                    if path == keep_path:
                        continue

                    try:
                        send2trash(path)
                    except Exception as e:
                        st.error(f"Failed to delete {path}: {e}")
                        continue

                    # Remove deleted file from DB
                    conn.execute("DELETE FROM photos WHERE path=?", (path,))
                    conn.execute("DELETE FROM ai_tags WHERE path=?", (path,))
                    conn.execute("DELETE FROM file_hashes WHERE path=?", (path,))
                    deleted += 1

                conn.commit()
                st.success(f"✅ Deleted {deleted} duplicates (kept 1).")
                st.rerun()
        
        with del_col2:
            if st.button("☠️ Delete ALL (keep none)", type="primary"):
                deleted = 0

                for (path, filename, size_bytes) in files:
                    try:
                        send2trash(path)
                    except Exception as e:
                        st.error(f"Failed to delete {path}: {e}")
                        continue

                    # Remove deleted file from DB
                    conn.execute("DELETE FROM photos WHERE path=?", (path,))
                    conn.execute("DELETE FROM ai_tags WHERE path=?", (path,))
                    conn.execute("DELETE FROM file_hashes WHERE path=?", (path,))
                    deleted += 1

                conn.commit()
                st.success(f"✅ Deleted ALL {deleted} files in this group.")
                st.rerun()

else:
    st.subheader("🧹 Unignore duplicate groups")

    # Pagination math for ignored groups
    ig_offset = st.session_state.ignore_page * ignored_page_size
    ignored = fetch_ignored_groups(conn, ignored_page_size, ig_offset)

    # Pager buttons
    u1, u2, u3 = st.columns([1, 1, 2])
    with u1:
        if st.button("⬅️ Prev ignored", disabled=st.session_state.ignore_page == 0):
            st.session_state.ignore_page -= 1
            st.rerun()
    with u2:
        if st.button("Next ignored ➡️", disabled=len(ignored) < ignored_page_size):
            st.session_state.ignore_page += 1
            st.rerun()
    with u3:
        st.write(f"Ignored page: **{st.session_state.ignore_page + 1}**")

    st.write(f"Showing **{len(ignored)}** ignored groups on this page.")

    if not ignored:
        st.info("No ignored groups. Ignore a group in Duplicates mode first.")
    else:
        # Show each ignored group with an Unignore button
        for sha256, note, created_at in ignored:
            with st.container(border=True):

                # --- NEW: preview image ---
                preview_path = fetch_one_file_for_hash(conn, sha256)

                cols = st.columns([1, 3])
                with cols[0]:
                    if preview_path and Path(preview_path).exists():
                        try:
                            st.image(
                                fit_image(Path(preview_path), size=(160, 160)),
                                use_container_width=True,
                            )
                        except Exception:
                            st.error("Preview failed")
                    else:
                        st.info("No image found")

                with cols[1]:
                    st.markdown(f"**{sha256[:12]}...**")
                    st.caption(f"🕒 {created_at}")

                    if note:
                        st.write(f"📝 {note}")

                    count_row = conn.execute(
                        "SELECT COUNT(*) FROM file_hashes WHERE sha256 = ?",
                        (sha256,),
                    ).fetchone()
                    st.write(f"📦 Files in group: **{count_row[0]}**")
                    # NEW: open the folder of the preview file
                    if preview_path and Path(preview_path).exists():
                        if st.button("📂 Open folder", key=f"open_ig_{sha256}"):
                            open_containing_folder(preview_path)
                    else:
                        st.caption("No valid preview path to open.")
                    if st.button(
                        "✅ Unignore (show again in Duplicates)",
                        key=f"unig_{sha256}",
                    ):
                        unignore_group(conn, sha256)
                        st.success("Unignored! It will appear again in Duplicates.")
                        st.rerun()

# Close DB connection at the end of the script
conn.close()
