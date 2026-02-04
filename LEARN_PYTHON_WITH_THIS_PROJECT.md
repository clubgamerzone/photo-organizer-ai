# 🐍 Learn Python by Building a Local AI Photo Organizer

**A beginner-friendly guide using real code from this project**

This guide teaches Python fundamentals through the actual code in your `photo-organizer-ai` project. Every concept is explained with examples from the real files you've created.

---

## 📚 Table of Contents

1. [Project Overview](#-project-overview)
2. [Python Basics](#-python-basics)
   - [Variables and Data Types](#variables-and-data-types)
   - [Functions (`def`)](#functions-def)
   - [Type Hints (`: str`, `-> bool`)](#type-hints)
   - [Imports](#imports)
3. [Data Structures](#-data-structures)
   - [Lists, Sets, Tuples, Dictionaries](#lists-sets-tuples-dictionaries)
4. [Control Flow](#-control-flow)
   - [If/Else, Loops, Comprehensions](#ifelse-loops-comprehensions)
5. [Object-Oriented Python](#-object-oriented-python)
   - [Classes and Dataclasses](#classes-and-dataclasses)
   - [Decorators (`@`)](#decorators)
6. [Working with Files](#-working-with-files)
   - [The `pathlib` Module](#the-pathlib-module)
7. [Databases with SQLite](#-databases-with-sqlite)
   - [Connection (`conn`)](#connection-conn)
   - [SQL Queries](#sql-queries)
8. [AI Integration](#-ai-integration)
   - [YOLO Object Detection](#yolo-object-detection)
   - [BLIP Image Captioning](#blip-image-captioning)
9. [Building a Web UI with Streamlit](#-building-a-web-ui-with-streamlit)
10. [Project Architecture](#-project-architecture)

---

## 🎯 Project Overview

This project is a **local photo organizer** that:
1. **Scans** folders for images
2. **Stores** metadata in a SQLite database
3. **Uses AI** to detect objects (YOLO) and generate captions (BLIP)
4. **Categorizes** photos automatically
5. **Provides a UI** to browse, edit, and manage photos

### Files in the Project

| File | Purpose |
|------|---------|
| `app/config.py` | Configuration settings |
| `app/db.py` | Database connection and table creation |
| `app/indexer.py` | Scans folders and saves photo info |
| `app/ai_yolo.py` | Object detection using YOLO |
| `run_caption_blip.py` | Generates captions using BLIP AI |
| `run_categories.py` | Auto-categorizes photos |
| `ui_app.py` | The main Streamlit web interface |

---

## 🔤 Python Basics

### Variables and Data Types

```python
# From ui_app.py
DB_PATH = Path(r"D:\My projects\photo-organizer-ai\data\photo_index.sqlite")
```

**Explanation:**
- `DB_PATH` is a **variable** — a name that stores a value
- `Path(...)` creates a **Path object** (more on this later)
- `r"..."` is a **raw string** — the `r` means "don't interpret backslashes specially"
  - Without `r`: `"C:\new"` would interpret `\n` as a newline
  - With `r`: `r"C:\new"` keeps the backslash as-is

**Common Data Types:**

```python
# Strings (text)
filename = "photo.jpg"
caption = "a dog playing in the park"

# Numbers
size_bytes = 1024000        # Integer (whole number)
confidence = 0.85           # Float (decimal number)

# Booleans (True/False)
include_subfolders = True
is_duplicate = False

# None (represents "no value")
width = None
```

---

### Functions (`def`)

A **function** is a reusable block of code that does a specific task.

```python
# From app/indexer.py
def is_image_file(path: Path) -> bool:
    """
    Returns True if the file has an extension like .jpg or .png
    """
    return path.suffix.lower() in IMAGE_EXTS
```

**Breaking it down:**

| Part | Meaning |
|------|---------|
| `def` | "I'm defining a function" |
| `is_image_file` | The function's name (you choose this) |
| `(path: Path)` | **Parameter** — input the function needs |
| `: Path` | **Type hint** — tells you `path` should be a Path object |
| `-> bool` | **Return type hint** — this function returns True or False |
| `"""..."""` | **Docstring** — explains what the function does |
| `return ...` | Sends a value back to whoever called the function |

**Using the function:**

```python
result = is_image_file(Path("photo.jpg"))  # result = True
result = is_image_file(Path("document.pdf"))  # result = False
```

**More Examples from Your Project:**

```python
# From app/db.py - A function with no return value
def init_db(conn):
    """
    Creates the photos table if it doesn't exist.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photos (...)
    """)
    conn.commit()

# From ui_app.py - A function that returns a list
def fetch_categories(conn) -> list[str]:
    rows = conn.execute("SELECT name FROM categories ORDER BY name").fetchall()
    return [r[0] for r in rows]
```

---

### Type Hints

Type hints tell you (and your code editor) what type of data is expected.

```python
# From app/ai_yolo.py
def detect_objects(image_path: str, conf: float = 0.25, max_objects: int = 10) -> str:
```

| Hint | Meaning |
|------|---------|
| `image_path: str` | `image_path` should be a string |
| `conf: float = 0.25` | `conf` should be a float, defaults to 0.25 |
| `max_objects: int = 10` | `max_objects` should be an integer, defaults to 10 |
| `-> str` | This function returns a string |

**Common type hints:**

```python
name: str              # String
count: int             # Integer
price: float           # Decimal number
is_valid: bool         # True or False
items: list            # A list
items: list[str]       # A list of strings
data: dict             # A dictionary
data: dict[str, int]   # Dict with string keys, int values
value: str | None      # Either a string OR None
```

---

### Imports

Imports let you use code from other files or libraries.

```python
# From ui_app.py

# Built-in Python modules (come with Python)
import os                    # Operating system functions
import sqlite3               # Database
from pathlib import Path     # File path handling
from datetime import datetime  # Date/time handling

# Third-party libraries (installed via pip)
import streamlit as st       # Web UI framework
from PIL import Image        # Image processing (Pillow)
from send2trash import send2trash  # Safe file deletion

# Your own project files
from app.config import default_config
from app.db import connect, init_ai_tables
from app.ai_yolo import detect_objects
```

**Import styles:**

```python
import os                      # Import whole module, use as: os.startfile()
from pathlib import Path       # Import specific thing, use as: Path()
import streamlit as st         # Import with nickname, use as: st.button()
from app.db import connect, init_ai_tables  # Import multiple things
```

---

## 📦 Data Structures

### Lists, Sets, Tuples, Dictionaries

**Lists** — ordered, changeable collections

```python
# From app/indexer.py
all_paths = list(root_folder.rglob("*"))  # Convert to list

# From run_caption_blip.py
paths = get_sample_paths(conn, limit=200)
for p in paths:  # Loop through each item
    process(p)
```

**Sets** — unordered, unique items only

```python
# From app/indexer.py
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

# From run_categories.py
def normalize_csv(s):
    return {x.strip().lower() for x in s.split(",") if x.strip()}
    # {curly braces} with a loop = set comprehension

# Sets are great for checking membership:
if ".jpg" in IMAGE_EXTS:  # Very fast lookup!
    print("It's an image")
```

**Tuples** — ordered, unchangeable (immutable)

```python
# From app/indexer.py
data = (
    str(p.resolve()),     # full path
    p.name,               # filename
    p.suffix.lower(),     # extension
    stat.st_size,         # file size
    stat.st_mtime,        # modified time
    width,
    height
)
# Tuples are often used for database rows or function returns
```

**Dictionaries** — key-value pairs

```python
# From app/ai_yolo.py
name = model.names.get(int(cid))  # model.names is a dict

# Example dictionary:
photo_info = {
    "filename": "beach.jpg",
    "size": 1024000,
    "tags": ["beach", "sunset", "ocean"]
}

# Access values:
print(photo_info["filename"])  # "beach.jpg"
print(photo_info.get("caption", "No caption"))  # "No caption" (default if key missing)
```

---

## 🔀 Control Flow

### If/Else, Loops, Comprehensions

**If/Else Statements:**

```python
# From run_categories.py
def decide_category(filename: str, caption, objects_csv) -> str:
    filename_l = filename.lower()
    
    # Check multiple conditions in order
    if has_any(filename_l, screenshot_keywords):
        return "Screenshots"
    
    if has_any(caption_l, asset_keywords):
        return "Game Assets"
    
    if "person" in objs:
        return "People"
    
    return "Other"  # Default if nothing matched
```

**For Loops:**

```python
# From app/indexer.py
for p in tqdm(all_paths, desc="Scanning files"):
    if not p.is_file():
        continue  # Skip to next iteration
    
    if not is_image_file(p):
        continue
    
    # Process the file...
    count += 1
```

**While Loops:**

```python
# Keep doing something until a condition is false
while not finished:
    process_next_item()
```

**List Comprehensions** — compact way to create lists:

```python
# From ui_app.py
def fetch_categories(conn) -> list[str]:
    rows = conn.execute("SELECT name FROM categories ORDER BY name").fetchall()
    return [r[0] for r in rows]  # List comprehension!

# This is equivalent to:
result = []
for r in rows:
    result.append(r[0])
return result

# More examples:
numbers = [1, 2, 3, 4, 5]
squares = [n * n for n in numbers]  # [1, 4, 9, 16, 25]
evens = [n for n in numbers if n % 2 == 0]  # [2, 4]
```

---

## 🏗️ Object-Oriented Python

### Classes and Dataclasses

**Dataclass** — a simple way to create classes that hold data:

```python
# From app/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AppConfig:
    photos_root: Path  # where your photos are stored
    db_path: Path      # where we will save the database file
```

| Part | Meaning |
|------|---------|
| `@dataclass` | Decorator that auto-generates `__init__`, `__repr__`, etc. |
| `(frozen=True)` | Makes instances immutable (can't be changed) |
| `class AppConfig:` | Defines a new type called `AppConfig` |
| `photos_root: Path` | An attribute that stores a Path |

**Using the dataclass:**

```python
def default_config() -> AppConfig:
    return AppConfig(
        photos_root=Path(r"C:\Users\jay\Downloads"),
        db_path=Path("data/photo_index.sqlite")
    )

# Usage:
cfg = default_config()
print(cfg.photos_root)  # Access the attribute
print(cfg.db_path)
```

### Decorators (`@`)

Decorators modify or enhance functions/classes.

```python
# From app/config.py
@dataclass(frozen=True)  # Adds automatic methods to the class
class AppConfig:
    ...

# From ui_app.py
@st.cache_resource  # Streamlit decorator: cache expensive operations
def load_model():
    return YOLO("yolov8n.pt")
```

**How decorators work conceptually:**

```python
# This:
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

# Is equivalent to:
def load_model():
    return YOLO("yolov8n.pt")
load_model = st.cache_resource(load_model)
```

---

## 📁 Working with Files

### The `pathlib` Module

`pathlib` is the modern way to work with file paths in Python.

```python
from pathlib import Path

# Creating paths
p = Path(r"C:\Users\jay\Pictures")
p = Path(__file__)  # Path to current script

# Path operations (from your project)
p.exists()          # Does this file/folder exist?
p.is_file()         # Is it a file?
p.is_dir()          # Is it a directory?
p.name              # "photo.jpg" (filename only)
p.suffix            # ".jpg" (extension)
p.suffix.lower()    # ".jpg" (lowercase extension)
p.parent            # Parent folder
p.resolve()         # Full absolute path
p.stat()            # File info (size, modified time)
p.stat().st_size    # File size in bytes
p.stat().st_mtime   # Modified timestamp

# Iterating files
p.glob("*")         # All files in folder
p.glob("*.jpg")     # All .jpg files in folder
p.rglob("*")        # All files including subfolders (recursive)

# Building paths
db_path = Path(__file__).resolve().parents[1] / "data" / "photo_index.sqlite"
#         current file    absolute    go up 1    add "data"   add filename
```

**Real example from app/indexer.py:**

```python
def scan_and_save_photos(root_folder: Path, conn) -> int:
    # rglob = recursive glob (searches subfolders too)
    all_paths = list(root_folder.rglob("*"))
    
    for p in all_paths:
        if not p.is_file():
            continue  # Skip directories
        
        if p.suffix.lower() not in IMAGE_EXTS:
            continue  # Skip non-images
        
        stat = p.stat()
        data = (
            str(p.resolve()),  # Full path as string
            p.name,            # Just the filename
            p.suffix.lower(),  # Extension
            stat.st_size,      # Size in bytes
            stat.st_mtime,     # Modified time
        )
```

---

## 🗄️ Databases with SQLite

### Connection (`conn`)

`conn` is short for "connection" — it's your link to the database.

```python
# From app/db.py
import sqlite3

def connect(db_path: Path):
    """Opens (or creates) the database file."""
    conn = sqlite3.connect(str(db_path))
    return conn
```

**The connection object lets you:**
- Execute SQL commands: `conn.execute("SELECT * FROM photos")`
- Save changes: `conn.commit()`
- Close the connection: `conn.close()`

### SQL Queries

**Creating Tables:**

```python
# From app/db.py
def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            ext TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime REAL NOT NULL,
            width INTEGER,
            height INTEGER
        );
    """)
    conn.commit()  # Save the changes!
```

| SQL Part | Meaning |
|----------|---------|
| `CREATE TABLE IF NOT EXISTS` | Create table only if it doesn't exist |
| `id INTEGER PRIMARY KEY AUTOINCREMENT` | Auto-incrementing ID |
| `path TEXT UNIQUE NOT NULL` | Text field, must be unique, can't be empty |
| `width INTEGER` | Optional integer (can be NULL) |

**Inserting Data:**

```python
# From app/db.py
conn.execute("""
    INSERT OR REPLACE INTO photos
    (path, filename, ext, size_bytes, mtime, width, height)
    VALUES (?, ?, ?, ?, ?, ?, ?)
""", data)  # data is a tuple with 7 values
```

- `?` are **placeholders** — they get replaced with values from `data`
- This prevents **SQL injection** attacks
- `INSERT OR REPLACE` updates if the record exists

**Selecting Data:**

```python
# From ui_app.py
rows = conn.execute("""
    SELECT photos.path, photos.filename, ai_tags.caption
    FROM photos
    JOIN ai_tags ON ai_tags.path = photos.path
    WHERE ai_tags.category = ?
    ORDER BY photos.mtime DESC
    LIMIT ? OFFSET ?
""", (category, limit, offset)).fetchall()
```

| Part | Meaning |
|------|---------|
| `SELECT ... FROM photos` | Get these columns from photos table |
| `JOIN ai_tags ON ...` | Combine with ai_tags table where paths match |
| `WHERE ai_tags.category = ?` | Filter by category |
| `ORDER BY photos.mtime DESC` | Sort by modified time, newest first |
| `LIMIT ? OFFSET ?` | Pagination (get X items starting from Y) |
| `.fetchall()` | Get all matching rows as a list |
| `.fetchone()` | Get just one row |

**Updating Data:**

```python
# From ui_app.py
conn.execute(
    "UPDATE ai_tags SET caption=?, objects=?, category=? WHERE path=?",
    (caption, objects, category, path)
)
conn.commit()
```

**Deleting Data:**

```python
conn.execute("DELETE FROM photos WHERE path=?", (path,))
conn.commit()
```

---

## 🤖 AI Integration

### YOLO Object Detection

YOLO (You Only Look Once) detects objects in images.

```python
# From app/ai_yolo.py
from ultralytics import YOLO

_model = None  # Global variable to store the model

def get_model():
    global _model  # Use the global variable
    if _model is None:  # Load only once (lazy loading)
        _model = YOLO("yolov8n.pt")
    return _model

def detect_objects(image_path: str, conf: float = 0.25, max_objects: int = 10) -> str:
    model = get_model()
    results = model.predict(source=image_path, conf=conf, verbose=False)
    r = results[0]
    
    labels = []
    if r.boxes is not None and len(r.boxes) > 0:
        for cid in r.boxes.cls.tolist():
            name = model.names.get(int(cid))
            if name:
                labels.append(name)
    
    # Remove duplicates while preserving order
    seen = set()
    uniq = []
    for x in labels:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    
    return ",".join(uniq[:max_objects])  # "person,dog,car"
```

### BLIP Image Captioning

BLIP generates natural language descriptions of images.

```python
# From run_caption_blip.py
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model (downloads on first run)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)  # Move to GPU if available

# Generate a caption
img = Image.open("photo.jpg").convert("RGB")
inputs = processor(images=img, return_tensors="pt").to(device)
output_ids = model.generate(**inputs, max_new_tokens=30)
caption = processor.decode(output_ids[0], skip_special_tokens=True)
# caption = "a dog playing in the park"
```

---

## 🖥️ Building a Web UI with Streamlit

Streamlit lets you create web apps with pure Python.

```python
# From ui_app.py
import streamlit as st

# Page config (must be first Streamlit command)
st.set_page_config(page_title="Photo Organizer AI", layout="wide")

# Title
st.title("📸 Photo Organizer AI")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    mode = st.radio("Mode", ["Gallery", "Duplicates", "Unignore"])
    
    st.divider()
    
    search = st.text_input("Search", "")
    category = st.selectbox("Category", ["All", "People", "Pets"])

# Main content
if mode == "Gallery":
    st.subheader("🖼️ Gallery")
    
    # Create columns for grid layout
    cols = st.columns(5)  # 5 columns
    
    for idx, photo in enumerate(photos):
        with cols[idx % 5]:  # Distribute across columns
            st.image(photo.path)
            st.caption(photo.filename)
            
            if st.button("Select", key=f"sel_{photo.path}"):
                st.session_state.selected = photo.path
                st.rerun()  # Refresh the page
```

**Common Streamlit Elements:**

```python
# Display
st.title("Big Title")
st.header("Header")
st.subheader("Subheader")
st.write("Any text or data")
st.markdown("**Bold** and *italic*")
st.image("photo.jpg")
st.error("Error message")
st.success("Success!")
st.info("Info message")
st.warning("Warning!")

# Input
name = st.text_input("Your name", "default value")
bio = st.text_area("Bio", height=100)
age = st.number_input("Age", min_value=0, max_value=120)
agree = st.checkbox("I agree")
option = st.selectbox("Choose", ["A", "B", "C"])
options = st.multiselect("Choose many", ["A", "B", "C"])

# Buttons
if st.button("Click me"):
    st.write("Button clicked!")

# Layout
col1, col2 = st.columns(2)
with col1:
    st.write("Left side")
with col2:
    st.write("Right side")

# Session state (remembers values between reruns)
if "counter" not in st.session_state:
    st.session_state.counter = 0

if st.button("Increment"):
    st.session_state.counter += 1
    st.rerun()

st.write(f"Count: {st.session_state.counter}")
```

---

## 🏛️ Project Architecture

### How the Pieces Fit Together

```
photo-organizer-ai/
│
├── app/                    # Core library code
│   ├── __init__.py        # Makes it a Python package
│   ├── config.py          # Configuration settings
│   ├── db.py              # Database functions
│   ├── indexer.py         # File scanning logic
│   └── ai_yolo.py         # Object detection
│
├── data/                   # Database storage
│   └── photo_index.sqlite # SQLite database file
│
├── run_*.py               # Standalone scripts
│   ├── run_index.py       # Scan and index photos
│   ├── run_caption_blip.py # Generate AI captions
│   ├── run_categories.py  # Auto-categorize
│   └── run_file_hashes.py # Find duplicates
│
├── ui_app.py              # Main Streamlit UI
│
└── requirements.txt       # Python dependencies
```

### Data Flow

```
1. SCAN: Folder → indexer.py → photos table
2. AI TAG: photos → YOLO + BLIP → ai_tags table  
3. CATEGORIZE: ai_tags → run_categories.py → categories
4. BROWSE: Database → ui_app.py → Web Browser
```

### The `__name__ == "__main__"` Pattern

```python
# From run_categories.py
def main():
    # All the work happens here
    cfg = default_config()
    conn = connect(cfg.db_path)
    # ...

if __name__ == "__main__":
    main()
```

**What this means:**
- When you run `python run_categories.py`, `__name__` is `"__main__"`, so `main()` runs
- When you `import run_categories`, `__name__` is `"run_categories"`, so `main()` does NOT run
- This lets you import functions from a file without executing the whole script

---

## 🎓 Quick Reference

### Essential Python Syntax

```python
# Variables
name = "value"
count = 42
is_valid = True

# Functions
def function_name(param1: type, param2: type = default) -> return_type:
    """Docstring explaining the function."""
    # code here
    return result

# Conditionals
if condition:
    do_this()
elif other_condition:
    do_that()
else:
    do_default()

# Loops
for item in collection:
    process(item)

while condition:
    keep_going()

# List comprehension
squares = [x*x for x in range(10)]

# Dictionary comprehension  
counts = {word: len(word) for word in words}

# Try/except (error handling)
try:
    risky_operation()
except Exception as e:
    print(f"Error: {e}")
finally:
    cleanup()

# Context managers (with statement)
with open("file.txt") as f:
    content = f.read()
# File is automatically closed after the with block

# Classes
class MyClass:
    def __init__(self, value):
        self.value = value
    
    def method(self):
        return self.value * 2
```

### Common String Operations

```python
text = "Hello, World!"
text.lower()           # "hello, world!"
text.upper()           # "HELLO, WORLD!"
text.strip()           # Remove whitespace from ends
text.split(",")        # ["Hello", " World!"]
",".join(["a", "b"])   # "a,b"
text.replace("o", "0") # "Hell0, W0rld!"
f"Name: {name}"        # f-string formatting
```

---

## 🚀 Next Steps

Now that you understand the code, try these exercises:

1. **Add a new category** — Edit `run_categories.py` to detect "Food" photos
2. **Add a new filter** — Modify `ui_app.py` to filter by file extension
3. **Export feature** — Create a function that exports photos by category to folders
4. **Statistics page** — Add a dashboard showing photo counts by category

---

**Happy Learning! 🐍✨**

*This guide was created based on your actual `photo-organizer-ai` project code.*
