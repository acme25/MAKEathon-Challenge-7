import re
import unicodedata
from pathlib import Path
import pandas as pd
import PyPDF2

# ------------------------------------------------------------
# Einstellungen
# ------------------------------------------------------------
RESULT_DIR = Path("result")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULT_DIR / "fundstellen.csv"
OUT_TXT = RESULT_DIR / "fundstellen.txt"

# Minimale Begrifflänge (zu kurze Tokens wie "m2", "DN" etc. rausfiltern)
MIN_TERM_LEN = 3

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def normalize_text(s: str) -> str:
    """Sanfte Normalisierung: Klein, Umlaute flach, Satzzeichen vereinfachen, Mehrfachspaces entfernen."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # Erlaube Buchstaben, Zahlen und einige Trennzeichen; Rest -> Space
    keep = "abcdefghijklmnopqrstuvwxyz0123456789 äöüß-_/.,:+()"
    s = "".join(ch if ch in keep else " " for ch in s)
    s = s.replace("ß", "ss")
    s = s.replace(".", " ").replace(",", " ")
    s = " ".join(s.split())
    return s


def word_boundary_regex(term: str) -> re.Pattern:
    """Erzeugt ein Regex mit Wortgrenzen (groß/klein egal)."""
    esc = re.escape(term)
    pattern = r"(?<!\w)" + esc + r"(?!\w)"
    return re.compile(pattern, flags=re.IGNORECASE)


def load_excel_terms(excel_file: str, sheet_name=0) -> list[str]:
    """Excel ohne Header einlesen und alle Spaltenwerte als Liste zurückgeben."""
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, engine="openpyxl")
    except Exception:
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)

    values = []
    for col in df.columns:
        vals = df[col].dropna().astype(str).str.strip().tolist()
        values.extend(vals)

    # Duplikate entfernen & filtern
    values = list({v for v in values if len(normalize_text(v)) >= MIN_TERM_LEN})
    return values


# ------------------------------------------------------------
# PDF
# ------------------------------------------------------------
def extract_text_from_pdf(pdf_file: str) -> str:
    """Extrahiert Text aus PDF (einfach, robust)."""
    text = ""
    with open(pdf_file, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            text += "\n" + t
    return text


# ------------------------------------------------------------
# Matching (nur exact / contains)
# ------------------------------------------------------------
def compare_text_with_terms(extracted_text: str, excel_terms: list[str]) -> pd.DataFrame:
    text_raw = extracted_text or ""
    text_norm = normalize_text(text_raw)

    results = []
    for term in excel_terms:
        term_raw = term
        term_norm = normalize_text(term_raw)

        if not term_norm or len(term_norm) < MIN_TERM_LEN:
            continue

        # 1) Exakt mit Wortgrenzen
        rex = word_boundary_regex(term_norm)
        if rex.search(text_norm):
            results.append({
                "excel_term": term_raw,
                "found": term_raw,
                "method": "exact"
            })
            continue

        # 2) Weiche Teilstring-Suche
        if term_norm in text_norm:
            results.append({
                "excel_term": term_raw,
                "found": term_raw,
                "method": "contains"
            })
            continue

    if not results:
        return pd.DataFrame(columns=["excel_term", "found", "method"])

    df = pd.DataFrame(results).sort_values(["method", "excel_term"], ascending=[True, True])
    return df


# ------------------------------------------------------------
# Hauptworkflow
# ------------------------------------------------------------
def bauplan_analyse(pdf_file: str, excel_file: str, sheet_name=0) -> pd.DataFrame:
    print("=" * 60)
    print("BAUPLAN ANALYSE (ohne Fuzzy)")
    print("=" * 60)

    print("\n[1/3] PDF lesen …")
    text = extract_text_from_pdf(pdf_file)
    print(f"   ✔ Textlänge: {len(text)} Zeichen")

    print("\n[2/3] Excel-Begriffe laden … (ohne Header)")
    terms = load_excel_terms(excel_file, sheet_name=sheet_name)
    print(f"   ✔ {len(terms)} eindeutige Begriffe geladen")

    print("\n[3/3] Abgleich durchführen (exact / contains) …")
    df_hits = compare_text_with_terms(text, terms)
    print(f"   ✔ Treffer: {len(df_hits)}")

    # Outputs speichern
    df_hits.to_csv(OUT_CSV, index=False, encoding="utf-8")
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for _, r in df_hits.iterrows():
            f.write(f"{r['excel_term']} → {r['found']} ({r['method']})\n")

    print("\nErgebnisse gespeichert:")
    print(f"  • {OUT_CSV}")
    print(f"  • {OUT_TXT}")

    return df_hits


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    pdf_datei = "Quartierpark Thurgauerstrasse.pdf"
    excel_datei = "Materialmatrix.xlsx"
    sheet = 0

    hits = bauplan_analyse(pdf_datei, excel_datei, sheet_name=sheet)

    print("\n" + "=" * 60)
    print("TREFFER (Top 30):")
    print("=" * 60)
    for i, row in hits.head(30).iterrows():
        print(f"{i+1}. {row['excel_term']} → {row['found']}  [{row['method']}]")
