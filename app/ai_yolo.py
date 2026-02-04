from ultralytics import YOLO

_model = None

def get_model():
    global _model
    if _model is None:
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

    # unique preserve order
    seen = set()
    uniq = []
    for x in labels:
        if x not in seen:
            seen.add(x)
            uniq.append(x)

    return ",".join(uniq[:max_objects])
