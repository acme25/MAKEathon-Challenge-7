import numpy as np
import cv2
from ultralytics import YOLO
import random
from pdf2image import convert_from_path
import os
 
# # === Klassen laden ===
with open("Quartierpark Thurgauerstrasse_TEXT.txt", "r") as f:
    class_list = f.read().split("\n")
 
# Zufällige Farben für jede Klasse
detection_colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) 
                    for _ in class_list]
 
# === YOLO Modell laden ===
model = YOLO("weights/yolov8n.pt", "v8")
 
# === PDF einlesen ===
PDF_PATH = "Quartierpark Thurgauerstrasse.pdf"     # deine PDF-Datei
OUTPUT_DIR = "result"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# PDF-Seiten als PIL-Bilder (z.B. 300 DPI)
pages = convert_from_path(PDF_PATH, dpi=300)
 
print(f"PDF geladen: {len(pages)} Seiten gefunden.")
 
for page_num, page in enumerate(pages, start=1):
    print(f"➡️  Verarbeite Seite {page_num}...")
 
    # PIL → OpenCV-Format
    frame = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
 
    # YOLO-Vorhersage
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    DP = detect_params[0].numpy()
 
    # Zeichne Boxen falls etwas erkannt wird
    if len(DP) != 0:
        boxes = detect_params[0].boxes
        for i in range(len(boxes)):
            box = boxes[i]
            clsID = int(box.cls.numpy()[0])
            conf = float(box.conf.numpy()[0])
            bb = box.xyxy.numpy()[0]
 
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[clsID],
                3
            )
 
            cv2.putText(
                frame,
                f"{class_list[clsID]} {round(conf,2)}",
                (int(bb[0]), int(bb[1])-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
 
    # Seite speichern
    out_path = os.path.join(OUTPUT_DIR, f"page_{page_num:03d}.jpg")
    cv2.imwrite(out_path, frame)
    print(f"✅  Ergebnisse gespeichert: {out_path}")
 
print("Fertig! Alle Seiten analysiert.")