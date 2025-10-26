# usingyolobasics.py
from ultralytics import YOLO
import cv2

# Pfad zu deinem trainierten Modell
model = YOLO("yolov8n.pt")   # oder dein Dateiname, z. B. runs/segment/train/weights/best.pt
model = YOLO("weights/legend_model.pt")

# Testbild (dein Plan oder ein Tile)
img_path = os.path.join("Data", "raw_tiles", "tile_00001.png")

# Modell anwenden
results = model(img_path, conf=0.4, imgsz=1024)

# Ergebnisse anzeigen und speichern
vis = results[0].plot()
cv2.imwrite("result", vis)

print("✅ Ergebnis gespeichert unter: result/prediction.png")