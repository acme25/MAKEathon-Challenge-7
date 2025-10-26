import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pdf2image import convert_from_path
from PIL import Image
 
# === EINSTELLUNGEN ===
MODEL_PATH = "weights/yolov8n.pt"       # YOLO-Modell für eure Legenden
CLASS_FILE = "utils/legend_classes.txt"      # Klassenliste
PLAN_DIR = "Quartierpark Thurgauerstrasse.pdf" # Input-Ordner mit PDFs oder Bildern
OUTPUT_DIR = "output"                        # Ergebnisse hierhin
CONFIDENCE = 0.4                             # Erkennungsschwelle
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# === Klassen einlesen ===
with open(CLASS_FILE, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]
 
# === YOLO laden ===
model = YOLO(MODEL_PATH)
 
# === Hilfsfunktion: PDF in Bilder umwandeln ===
def pdf_to_images(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi)
    images = []
    for i, page in enumerate(pages):
        img_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page{i+1}.jpg")
        page.save(img_path, "JPEG")
        images.append(img_path)
    return images
 
# === Alle Pläne verarbeiten ===
all_detections = []
 
for file in os.listdir(PLAN_DIR):
    path = os.path.join(PLAN_DIR, file)
    images = []
 
    if file.lower().endswith(".pdf"):
        print(f"📄 PDF erkannt → konvertiere: {file}")
        images = pdf_to_images(path)
    elif file.lower().endswith((".png", ".jpg", ".jpeg")):
        images = [path]
    else:
        print(f"⚠️ Überspringe {file} (kein unterstütztes Format)")
        continue
 
    # Jede Seite / jedes Bild analysieren
    for img_path in images:
        print(f"🔍 Analysiere: {img_path}")
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"❌ Fehler beim Laden von {img_path}")
            continue
 
        results = model.predict(source=[frame], conf=CONFIDENCE, save=False)
        boxes = results[0].boxes
 
        for box in boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
 
            # Erkannten Eintrag speichern
            all_detections.append({
                "plan": os.path.basename(img_path),
                "object": label,
                "confidence": round(conf, 3),
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)
            })
 
        # Optional: Boxen auf Bild zeichnen und speichern
        annotated = results[0].plot()
        out_img = os.path.join(OUTPUT_DIR, f"annotated_{os.path.basename(img_path)}")
        cv2.imwrite(out_img, annotated)
        print(f"✅ Ergebnis gespeichert: {out_img}")
 
# === Zusammenfassung als Tabelle ===
if not all_detections:
    print("⚠️ Keine Objekte erkannt.")
else:
    df = pd.DataFrame(all_detections)
    summary = df.groupby("object").size().reset_index(name="count")
    summary_path = os.path.join(OUTPUT_DIR, "materials_summary.xlsx")
 
    with pd.ExcelWriter(summary_path) as writer:
        df.to_excel(writer, sheet_name="Detections", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
 
    print(f"\n📋 Erkennung abgeschlossen!")
    print(f"Gespeichert unter: {summary_path}\n")
    print("=== Material-Zusammenfassung ===")
    print(summary)