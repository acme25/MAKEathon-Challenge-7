# backend/services/pdf_service.py

import fitz  # PyMuPDF
import tempfile
import os

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[str]:
    """
    Nimmt ein PDF und rendert jede Seite als temporäres PNG-Bild.
    Rückgabe: Liste von Pfaden zu den erzeugten PNGs.
    Wichtig: Wir benutzen mkstemp, damit die Datei NICHT geöffnet bleibt.
    """
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        zoom = dpi / 72  # 72 dpi ist PDF-Standard
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Sicheren temporären Dateinamen erzeugen, aber Handle gleich wieder schließen
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)            # Handle schließen, damit nix gesperrt bleibt

        # Jetzt kann PyMuPDF gefahrlos in die Datei schreiben
        pix.save(tmp_path)
        pix = None

        image_paths.append(tmp_path)

    doc.close()
    return image_paths
