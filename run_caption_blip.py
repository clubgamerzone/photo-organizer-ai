from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

from app.config import default_config
from app.db import connect, init_ai_tables


def get_sample_paths(conn, limit: int = 200):
    # newest photos first
    rows = conn.execute(
        "SELECT path FROM photos ORDER BY mtime DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [r[0] for r in rows]


def save_caption(conn, path: str, caption: str):
    conn.execute(
        """
        INSERT INTO ai_tags(path, caption, objects)
        VALUES (?, ?, NULL)
        ON CONFLICT(path) DO UPDATE SET caption=excluded.caption;
        """,
        (path, caption),
    )


def main():
    cfg = default_config()
    conn = connect(cfg.db_path)
    init_ai_tables(conn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("Loading BLIP caption model (first run downloads files)...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)

    paths = get_sample_paths(conn, limit=200)
    print("Captioning", len(paths), "photos...")

    done = 0
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            output_ids = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(output_ids[0], skip_special_tokens=True)

            save_caption(conn, p, caption)
            done += 1

            # print occasionally so the console isn't spammed
            if done % 20 == 0:
                print(f"{done}/200 ✅ example:", caption)

        except Exception as e:
            print("❌ Failed:", p, "|", e)

    conn.commit()
    conn.close()
    print("✅ Done. Captions saved for", done, "photos.")


if __name__ == "__main__":
    main()
