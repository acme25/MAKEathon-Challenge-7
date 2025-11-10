# backend/services/yolo_service.py

from ultralytics import YOLO
from pathlib import Path
import config  

CUSTOM_MODEL_PATH = Path("data/yolo_models/yolov8s.pt")

if CUSTOM_MODEL_PATH.exists():
    print(f"🔍 Lade custom YOLO-Modell aus {CUSTOM_MODEL_PATH}")
    _yolo_model = YOLO(str(CUSTOM_MODEL_PATH))
else:
    print(f"⚠️ Kein custom YOLO-Modell gefunden – verwende '{config.YOLO_MODEL_NAME}'")
    _yolo_model = YOLO(config.YOLO_MODEL_NAME)
    

def detect_objects(image_path: str, conf: float = 0.25):
    results = _yolo_model(image_path, conf=conf)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = _yolo_model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
        })

    return detections

