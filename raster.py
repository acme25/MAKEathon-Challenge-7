# pdf_to_tiles.py
import fitz, os, cv2, math, numpy as np

PDF_PATH = "Quartierpark Thurgauerstrasse.pdf"
OUT_DIR  = "bites"
DPI = 300           # Auflösung
TILE = 1024         # Kachelgröße
OVERLAP = 128       # Überlappung für bessere Kontexteinfänge

os.makedirs(OUT_DIR, exist_ok=True)
doc = fitz.open(PDF_PATH)
page = doc[0]
pix = page.get_pixmap(dpi=DPI, alpha=False)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
h, w = img.shape[:2]

idx = 0
for y in range(0, h, TILE - OVERLAP):
    for x in range(0, w, TILE - OVERLAP):
        tile = img[y:min(y+TILE,h), x:min(x+TILE,w)]
        if tile.shape[0] < TILE or tile.shape[1] < TILE:
            pad = np.zeros((TILE, TILE, 3), dtype=np.uint8) + 255
            pad[:tile.shape[0], :tile.shape[1]] = tile
            tile = pad
        cv2.imwrite(os.path.join(OUT_DIR, f"tile_{idx:05d}.png"), tile)
        idx += 1
print("Tiles:", idx)
