# import fitz  # PyMuPDF

# # PDF öffnen
# pdf_path = "Wettbewerb Saint-Louis-Park_Montauk_PLAN 2_A3.pdf"
# doc = fitz.open(pdf_path)

# # Alle Seiten durchgehen
# for page_number, page in enumerate(doc, start=1):
#     text = page.get_text("text")
#     print(f"--- Seite {page_number} ---")
#     print(text)



import fitz  # PyMuPDF
import io
from PIL import Image
import pytesseract

PDF_PATH = r"Wettbewerb Saint-Louis-Park_Montauk_PLAN 2_A3.pdf"
LANG = "deu+eng"

def extract_text_plain(page: fitz.Page) -> str:
    """Nur menschenlesbaren Text zurückgeben (kein XML)."""
    pieces = []

    # 1) Normal formatiert
    t1 = page.get_text("text") or ""
    if t1.strip():
        pieces.append(t1)

    # 2) Wörter -> Zeilen (robust bei Rotation/Layouts)
    words = page.get_text("words") or []  # (x0,y0,x1,y1,word, block, line, word_no)
    if words:
        words.sort(key=lambda w: (round(w[1], 1), w[0]))  # top->down, left->right
        line_y = None
        line, lines = [], []
        for w in words:
            y = round(w[1], 1)
            if line_y is None or abs(y - line_y) <= 2:  # gleiche Zeile
                line.append(w[4])
                line_y = y if line_y is None else line_y
            else:
                lines.append(" ".join(line))
                line = [w[4]]
                line_y = y
        if line:
            lines.append(" ".join(line))
        t2 = "\n".join(lines)
        # Nur nehmen, wenn länger / sinnvoll
        if len(t2) > len(t1):
            pieces.append(t2)

    # Beste menschenlesbare Variante
    return max(pieces, key=len) if pieces else ""

def ocr_page(page: fitz.Page, dpi=450) -> str:
    """Fallback-OCR, falls kaum Text gefunden wurde."""
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang=LANG)

def extract_text_hybrid(pdf_path: str, ocr_threshold: int = 120) -> str:
    doc = fitz.open(pdf_path)
    out = []
    for i, page in enumerate(doc, start=1):
        plain = extract_text_plain(page).strip()
        if len(plain) < ocr_threshold:  # zu wenig -> OCR
            ocr = ocr_page(page)
            text_final = ocr if len(ocr.strip()) > len(plain) else plain
        else:
            text_final = plain
        out.append(f"--- Seite {i} ---\n{text_final}")
    return "\n\n".join(out)

if __name__ == "__main__":
    text_all = extract_text_hybrid(PDF_PATH)
    out_path = PDF_PATH.rsplit(".", 1)[0] + "_TEXT.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text_all)
    print(f"Fertig. Export: {out_path}")
