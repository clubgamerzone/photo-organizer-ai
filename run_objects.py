from ultralytics import YOLO

from app.config import default_config
from app.db import connect, init_ai_tables


def get_sample_paths(conn, limit: int = 200):
    rows = conn.execute(
        "SELECT path FROM photos ORDER BY mtime DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [r[0] for r in rows]


def save_objects(conn, path: str, objects_csv: str):
    conn.execute(
        """
        INSERT INTO ai_tags(path, caption, objects)
        VALUES (?, NULL, ?)
        ON CONFLICT(path) DO UPDATE SET objects=excluded.objects;
        """,
        (path, objects_csv),
    )


def main():
    cfg = default_config()
    conn = connect(cfg.db_path)
    init_ai_tables(conn)

    print("Loading YOLO model (first time downloads)...")
    model = YOLO("yolov8n.pt")  # small + fast model

    paths = get_sample_paths(conn, limit=200)
    print("Detecting objects in", len(paths), "photos...")

    done = 0
    for p in paths:
        try:
            results = model.predict(p, verbose=False)
            r = results[0]

            names = r.names  # class id -> class name
            if r.boxes is None or len(r.boxes) == 0:
                objects = []
            else:
                class_ids = r.boxes.cls.tolist()
                objects = sorted(set(names[int(i)] for i in class_ids))

            objects_csv = ",".join(objects) if objects else ""
            save_objects(conn, p, objects_csv)
            done += 1

            if done % 20 == 0:
                print(f"{done}/200 ✅ example objects:", objects_csv)

        except Exception as e:
            print("❌ Failed:", p, "|", e)

    conn.commit()
    conn.close()
    print("✅ Done. Objects saved for", done, "photos.")


if __name__ == "__main__":
    main()
