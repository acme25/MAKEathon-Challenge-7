#!/usr/bin/env python3
# Versucht automatisch zusammenhängende Flächen in Bauplänen zu erkennen
# und als farbiges Overlay zu visualisieren. (Output in ./output/)
"""
autolabel_regions.py

Automatisch zusammenhängende Flächen in Bauplänen (PDF/JPG/PNG) erkennen
und als visuelles Ergebnisbild farbig hervorheben.

💡 Was es macht
- Liest alle PDFs/JPGs/PNGs aus einem Eingabeordner
- PDFs werden zuerst nach PNG konvertiert (pro Seite ein Bild)
- Führt Farbreduktion via K-Means durch (macht homogene Regionen homogener)
- Zerlegt pro Farb-Cluster die zusammenhängenden Regionen (Connected Components)
- Filtert sehr kleine Regionen heraus
- Rendert eine farbige Overlay-Maske + Konturen über dem Originalplan
- Speichert Ergebnisbilder unter ./output/

⚙️ Abhängigkeiten
- Python 3.9+ empfohlen
- pip install opencv-python numpy pdf2image pillow
  (Optional: PyMuPDF als Fallback für PDF → Bild: pip install pymupdf)

📦 Zusätzliche System-Abhängigkeit (nur falls pdf2image genutzt wird):
- Poppler (https://poppler.freedesktop.org/)
  macOS: brew install poppler
  Ubuntu/Debian: sudo apt-get install poppler-utils
  Windows: Poppler-Binary entpacken und den Pfad zu bin/ in PATH aufnehmen

▶️ Nutzung
    python autolabel_regions.py --input ./plans --output ./output \
        --dpi 300 --clusters 8 --min-area 1500

Parameter:
  --input       Eingabeordner mit PDF/JPG/PNG (Default: ./plans)
  --output      Ausgabeordner für Ergebnisbilder (Default: ./output)
  --dpi         DPI bei PDF-Konvertierung (Default: 300)
  --clusters    Anzahl der K-Means-Farbcluster (Default: 8)
  --min-area    Minimale Regionfläche (Pixel) (Default: 1500)
  --max-size    Max. Breite/Höhe fürs Processing (zur Beschleunigung), 0 = aus (Default: 2500)
  --outline     Konturbreite in Pixel (Default: 2)

💬 Hinweis
- Dieses Skript klassifiziert keine „Straße/Grünfläche/Gebäude“. Es gruppiert nur
  zusammenhängende Flächen. Das ist ideal als Vorlabeling, das du in CVAT
  visuell prüfen und ggf. manuell verfeinern kannst.
"""

import os
import sys
import glob
import math
import argparse
import random
from typing import List, Tuple

import numpy as np
import cv2

# Optionaler Import für PDF -> Bild
PDF_BACKENDS = []

try:
    from pdf2image import convert_from_path  # type: ignore
    PDF_BACKENDS.append("pdf2image")
except Exception:
    pass

try:
    import fitz  # PyMuPDF  # type: ignore
    PDF_BACKENDS.append("pymupdf")
except Exception:
    pass


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_image(path: str) -> np.ndarray:
    """Liest ein Bild (PNG/JPG) mit OpenCV ein (BGR)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Konnte Bild nicht lesen: {path}")
    return img


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
    """Konvertiert ein PDF in eine Liste von BGR-Images (OpenCV)."""
    if "pdf2image" in PDF_BACKENDS:
        pages = convert_from_path(pdf_path, dpi=dpi)
        imgs = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]
        return imgs
    elif "pymupdf" in PDF_BACKENDS:
        doc = fitz.open(pdf_path)  # type: ignore
        imgs = []
        for page in doc:
            mat = fitz.Matrix(dpi/72.0, dpi/72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            imgs.append(img)
        return imgs
    else:
        raise RuntimeError(
            "Kein PDF-Backend gefunden. Bitte installiere entweder 'pdf2image' (plus Poppler) "
            "oder 'pymupdf' (fitz)."
        )


def resize_for_processing(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Skaliert das Bild so, dass die längere Seite <= max_side ist. Gibt Scaling-Faktor zurück."""
    if max_side <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    scale = 1.0
    long_side = max(h, w)
    if long_side > max_side:
        scale = max_side / float(long_side)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def kmeans_color_quantization(img: np.ndarray, k: int = 8, attempts: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Reduziert Farben via K-Means; gibt (labels, centers) zurück."""
    Z = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    ret, labels, centers = cv2.kmeans(Z, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    quant = centers[labels].reshape(img.shape)
    return labels.reshape(img.shape[:2]), centers


def connected_components_per_cluster(labels2d: np.ndarray, cluster_id: int, min_area: int) -> Tuple[np.ndarray, int]:
    """Connected components auf einer binären Maske eines Farbclusters."""
    mask = (labels2d == cluster_id).astype(np.uint8) * 255
    # Kleine Löcher schließen / Kanten glätten
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, ccmap = cv2.connectedComponents(mask)
    # Fläche filtern
    for lab in range(1, num_labels):
        area = int((ccmap == lab).sum())
        if area < min_area:
            ccmap[ccmap == lab] = 0
    # Reindex
    unique = sorted([u for u in np.unique(ccmap) if u != 0])
    remap = np.zeros_like(ccmap, dtype=np.int32)
    for new_id, u in enumerate(unique, start=1):
        remap[ccmap == u] = new_id
    return remap, len(unique)


def draw_overlay(original: np.ndarray, region_maps: List[np.ndarray], outline: int = 2, alpha: float = 0.35) -> np.ndarray:
    """Zeichnet farbiges Overlay und Konturen der Regionen auf das Originalbild."""
    h, w = original.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    rng = random.Random(42)
    color_pool = []

    total_regions = 0
    for rmap in region_maps:
        if rmap is None:
            continue
        ids = [u for u in np.unique(rmap) if u != 0]
        total_regions += len(ids)
        for rid in ids:
            # Zufallsfarbe (nicht zu dunkel)
            if len(color_pool) <= rid:
                color_pool.append((rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255)))
            color = color_pool[rid % len(color_pool)]
            overlay[rmap == rid] = color

    blended = cv2.addWeighted(original, 1.0, overlay, alpha, 0)

    # Konturen zeichnen
    edges = cv2.Canny(overlay, 50, 150)
    if outline > 0:
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (outline, outline)), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 0, 0), 1)

    # Info-Badge
    info = f"{total_regions} Regionen"
    cv2.rectangle(blended, (10, 10), (10 + 8*len(info) + 16, 40), (255, 255, 255), -1)
    cv2.putText(blended, info, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)

    return blended


def process_image(img: np.ndarray, clusters: int, min_area: int, max_side: int, outline: int) -> np.ndarray:
    """Kompletter Pipeline-Lauf für ein einzelnes Bild."""
    work, scale = resize_for_processing(img, max_side=max_side)

    # Leichtes edge-aware Smoothing, damit K-Means bessere Cluster findet
    try:
        smoothed = cv2.pyrMeanShiftFiltering(work, sp=21, sr=15)
    except Exception:
        smoothed = cv2.bilateralFilter(work, 9, 50, 50)

    labels2d, centers = kmeans_color_quantization(smoothed, k=clusters, attempts=3)

    region_maps: List[np.ndarray] = []
    for cid in range(clusters):
        rmap, count = connected_components_per_cluster(labels2d, cid, min_area=int(min_area * (scale**2)))
        region_maps.append(rmap)

    # Overlay auf Originalgröße zurückprojizieren (falls skaliert wurde)
    vis = draw_overlay(work, region_maps, outline=outline, alpha=0.35)
    if scale != 1.0:
        vis = cv2.resize(vis, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return vis


def collect_files(input_dir: str) -> List[str]:
    exts = ["*.pdf", "*.png", "*.jpg", "*.jpeg"]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(input_dir, e)))
    paths = sorted(paths)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="./data/plans", help="Eingabeordner mit PDF/JPG/PNG")
    ap.add_argument("--output", type=str, default="./output", help="Ausgabeordner")
    ap.add_argument("--dpi", type=int, default=300, help="DPI für PDF-Konvertierung")
    ap.add_argument("--clusters", type=int, default=8, help="Anzahl KMeans-Farbcluster")
    ap.add_argument("--min-area", type=int, default=1500, help="minimale Regionfläche (Pixel)")
    ap.add_argument("--max-size", type=int, default=2500, help="Max. Bildkante für Processing (0=aus)")
    ap.add_argument("--outline", type=int, default=2, help="Konturbreite (px) in der Visualisierung")
    args = ap.parse_args()

    ensure_dir(args.output)

    files = collect_files(args.input)
    if not files:
        print(f"Keine Dateien gefunden in: {args.input}")
        sys.exit(1)

    print(f"Gefundene Dateien: {len(files)}")
    for path in files:
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)

        try:
            if ext.lower() == ".pdf":
                pages = pdf_to_images(path, dpi=args.dpi)
                for idx, img in enumerate(pages, start=1):
                    vis = process_image(img, clusters=args.clusters, min_area=args.min_area,
                                        max_side=args.max_size, outline=args.outline)
                    out_name = f"{name}_page{idx:02d}_segmented.png"
                    out_path = os.path.join(args.output, out_name)
                    cv2.imwrite(out_path, vis)
                    print(f"[OK] {out_path}")
            else:
                img = read_image(path)
                vis = process_image(img, clusters=args.clusters, min_area=args.min_area,
                                    max_side=args.max_size, outline=args.outline)
                out_name = f"{name}_segmented.png"
                out_path = os.path.join(args.output, out_name)
                cv2.imwrite(out_path, vis)
                print(f"[OK] {out_path}")
        except Exception as e:
            print(f"[FEHLER] {basename}: {e}")

    print("Fertig. Ergebnisse liegen im Ausgabeordner.")
    

if __name__ == "__main__":
    main()
