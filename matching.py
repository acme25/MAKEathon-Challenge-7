import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import process, fuzz

# ----------------------- Pfade anpassen -----------------------
XLSX_PATH = r"Materialmatrix.xlsx"   # deine hierarchische Excel
TXT_PATH  = r"Wettbewerb Saint-Louis-Park_Montauk_PLAN 2_A3_TEXT.txt"  # Text aus dem PDF
# -------------------------------------------------------------

# einfache Normalisierung (ohne externe Pakete)
def norm(s: str) -> str:
    if s is None: return ""
    s = s.lower()
    for a, b in [("ä","ae"),("ö","oe"),("ü","ue"),("ß","ss")]:
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9\s\-_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Attribute-Überschriften, die in deiner Excel als "Äste" vorkommen
ATTRIBUTE_KEYS = [
    "farbe", "oberflaechenbearbeitung", "oberflaeche",
    "einstreu", "geruestkorn", "koernung",
    "steindicke", "dicke"
]
# optionale Priorität, in welcher Reihenfolge der Bot fragen soll
ATTRIBUTE_ASK_ORDER = ["farbe", "oberflaechenbearbeitung", "oberflaeche", "koernung", "einstreu", "geruestkorn", "steindicke", "dicke"]

def read_hierarchical_excel(xlsx_path: str, sheet="Tabelle1"):
    """
    Liest die hierarchische Matrix (mehrere "Unnamed:"-Spalten) und rekonstruiert Pfade.
    Jeder Eintrag ist eine Liste path=[Knoten_1, Knoten_2, ..., Blatt].
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    rows = []
    for _, r in df.iterrows():
        seq = []
        for j, c in enumerate(df.columns):
            v = r[c]
            if isinstance(v, str): v = v.strip()
            if pd.isna(v) or v == "": continue
            seq.append((j, str(v)))
        if seq:
            rows.append(seq)

    # Pfad-Stack nach Spaltentiefe (j)
    entries = []
    stack = {}
    for idx, seq in enumerate(rows):
        depth, value = seq[0]
        stack[depth] = value
        for k in list(stack.keys()):
            if k > depth:
                del stack[k]
        next_depth = rows[idx+1][0][0] if idx+1 < len(rows) else -1
        # Leaf-Heuristik: wenn der nächste Knoten gleich flach oder flacher -> aktueller Knoten ist Blatt
        if next_depth <= depth:
            path = [stack[k] for k in sorted(stack)]
            entries.append(path)
    return entries

####### NUR DASSSSSSSS
def build_catalog(entries):
    """
    Aus den Pfaden einen Katalog bauen:
    - base_keys: Kandidaten für Material-/Produktnamen (flache Ebenen, keine Attributknoten)
    - attribute_values: für jeden Basis-Knoten -> { attribut -> {werte} }
    """
    base_candidates = set()
    attribute_values = defaultdict(lambda: defaultdict(set))

    def is_attribute_node(x: str) -> bool:
        n = norm(x)
        return any(ak in n for ak in ATTRIBUTE_KEYS)

    for path in entries:
        # Basis-Kandidaten: frühe Knoten (Tiefe <= 3) ohne Attributbezug
        for node in path[:4]:
            if not is_attribute_node(node):
                base_candidates.add(node)

        # Attribute-Werte extrahieren: wenn ein Knoten ein Attributtitel ist,
        # dann ist der nächste Knoten dessen Wert
        for i, node in enumerate(path):
            if is_attribute_node(node) and i + 1 < len(path):
                attr_key = next((ak for ak in ATTRIBUTE_KEYS if ak in norm(node)), None)
                val = path[i+1]
                # "Werte" sind keine weiteren Attributtitel
                if not is_attribute_node(val) and attr_key:
                    # als Basis verwenden wir den ersten nicht-Attribut-Knoten der Kette (am besten den "Material"-Knoten)
                    base_node = next((p for p in path[:4] if not is_attribute_node(p)), None)
                    if base_node:
                        attribute_values[base_node][attr_key].add(val)

    # Normalisierte Lookups für Matching
    base_index = {norm(b): b for b in base_candidates}

    return base_index, attribute_values

def read_text(txt_path: str) -> str:
    return Path(txt_path).read_text(encoding="utf-8", errors="ignore")

def find_material_mentions(text: str, base_index: dict, fuzzy_threshold=88):
    """
    Sucht Materialbegriffe aus dem Basisindex im PDF-Text.
    1) Exakt als Wort
    2) Fuzzy fallback (gegen OCR-Fehler)
    """
    text_n = norm(text)
    found = set()

    # (1) exakt
    for key_n, original in base_index.items():
        if not key_n:
            continue
        if re.search(rf"\b{re.escape(key_n)}\b", text_n):
            found.add(original)

    # (2) fuzzy: nur wenn nichts gefunden oder du mehr Robustheit willst
    if not found:
        keys = list(base_index.keys())
        # lange Texte beschneiden (Performance)
        sample = " ".join(text_n.split()[:4000])
        for k, score, _ in process.extract(sample, keys, scorer=fuzz.WRatio, limit=25):
            if score >= fuzzy_threshold:
                found.add(base_index[k])

    return sorted(found)

def build_questions(materials, attribute_values):
    """
    Erzeuge Chat-Fragen pro erkanntem Material, basierend auf den vorhandenen Attributen.
    """
    questions = []
    for mat in materials:
        attrs = attribute_values.get(mat, {})
        # frage in definierter Reihenfolge
        for ak in ATTRIBUTE_ASK_ORDER:
            values = sorted(attrs.get(ak, []))
            if values:
                # hübscher Attributname
                label = {
                    "farbe": "Farbe",
                    "oberflaechenbearbeitung": "Oberflächenbearbeitung",
                    "oberflaeche": "Oberfläche",
                    "einstreu": "Einstreu",
                    "geruestkorn": "Gerüstkorn",
                    "koernung": "Körnung",
                    "steindicke": "Steindicke",
                    "dicke": "Dicke"
                }.get(ak, ak.capitalize())
                # kurze, natürliche Frage
                opts = " oder ".join(values[:6])  # beschränken, falls viele
                questions.append(f"Ich habe {mat} im Dokument gefunden. {label}: {opts}?")
                break  # pro Material erstmal nur eine Frage stellen (kannst du aufheben)
        else:
            # kein bekanntes Attribut -> generische Nachfrage
            questions.append(f"Ich habe {mat} im Dokument gefunden. Möchtest du eine Variante / Spezifikation angeben?")
    return questions

if __name__ == "__main__":
    entries = read_hierarchical_excel(XLSX_PATH, sheet="Tabelle1")
    base_index, attribute_values = build_catalog(entries)
    text = read_text(TXT_PATH)

    materials = find_material_mentions(text, base_index, fuzzy_threshold=88)
    print("Erkannte Materialien:", materials, "\n")

    for q in build_questions(materials, attribute_values):
        print(q)