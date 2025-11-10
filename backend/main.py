# uvicorn backend.main:app --reload

# Kinder eines Knotens abrufen
# /kg/children/{name}

# Gesamt Baum abrufen
# /kg/subtree/{name}

#Semantische Zuordnung eines Begriffs von eingelesenen Begriffe
# /kg/map_term/{name}

#Semantische Zuordnung eines Begriffs zu Oberklassen
# /kg/map_root?term={name}

from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import tempfile
import traceback

from backend.services.pdf_picture_service import pdf_to_images
from backend.services.yolo_services import detect_objects
from backend.services.root_mapper import best_roots_for_term
from backend.services.neo4j_service import neo4j_db

from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import tempfile
import traceback
from backend.services.pdf_picture_service import pdf_to_images

#app = FastAPI(title="Knowledge Graph API", version="1.0")
app = FastAPI()

@app.get("/")
def root():
    return {"message": "🚀 Knowledge Graph API läuft!"}

@app.get("/kg/children/{name}")
def get_children(name: str):
    """Gibt alle direkten Unterkategorien eines Knotens zurück"""
    children = neo4j_db.get_children(name)
    return {"node": name, "children": children}

@app.get("/kg/subtree/{name}")
def get_subtree(name: str, depth: int = 4):
    """Gibt alle Knoten bis zur angegebenen Tiefe zurück"""
    query = f"""
    MATCH p = (:Kategorie {{name:$name}})-[:HAT_UNTERKATEGORIE*1..{depth}]->(c)
    RETURN DISTINCT c.name AS child
    ORDER BY child
    """
    results = [r["child"] for r in neo4j_db.run_query(query, {"name": name})]
    return {"root": name, "depth": depth, "children": results}

@app.get("/kg/map_term")
def map_term(term: str, top_k: int = 3):
    """
    Semantische Zuordnung eines Begriffs (YOLO-Label oder User-Input)
    zu Kategorien aus dem Knowledge Graph.
    """
    matches = best_categories_for_term(term, top_k=top_k)
    return {
        "term": term,
        "matches": [
            {"category": name, "score": score}
            for name, score in matches
        ]
    }

@app.get("/kg/map_root")
def map_root(term: str, top_k: int = 2):
    """
    Mappt einen Begriff (z.B. 'Ententeich') auf deine Oberklassen
    (Beläge, Grünflächen, Wasserflächen, Treppenanlagen).
    """
    matches = best_roots_for_term(term, top_k=top_k)
    return {
        "term": term,
        "roots": [
            {"root": name, "score": score}
            for name, score in matches
        ]
    }

@app.post("/plan/analyze")
async def analyze_plan(
    file: UploadFile = File(...),
    depth: int = 3,
    top_k_roots: int = 2,
):
    """
    Nimmt einen Plan als PDF oder Bild.
    - Bei PDF: rendert jede Seite als Bild
    - Bei Bild: nutzt direkt YOLO
    Für jedes erkannte Objekt:
      - Root-Klasse via Embedding bestimmen
      - Kinder im Knowledge Graph holen
    """
    try:
        # 1. Datei temporär speichern
        suffix = os.path.splitext(file.filename)[1].lower()
        if not suffix:
            suffix = ".pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        image_paths: list[str] = []

        # 2. PDF -> Bilder oder direkt Bild verwenden
        if suffix == ".pdf" or file.content_type == "application/pdf":
            image_paths = pdf_to_images(tmp_path)
        else:
            image_paths = [tmp_path]

        all_results = []

        # 3. YOLO auf jede Seite / jedes Bild
        for page_idx, img_path in enumerate(image_paths):
            detections = detect_objects(img_path)

            enriched = []
            for det in detections:
                term = det["label"]

                # 4. Root-Klasse bestimmen
                root_candidates = best_roots_for_term(term, top_k=top_k_roots)
                best_root = root_candidates[0][0] if root_candidates else None

                # 5. Kinder aus dem Knowledge Graph holen
                children = neo4j_db.get_children(best_root) if best_root else []

                enriched.append({
                    "label": term,
                    "confidence": det["confidence"],
                    "bbox": det["bbox"],
                    "root": best_root,
                    "root_candidates": [
                        {"root": r, "score": s} for r, s in root_candidates
                    ],
                    "kg_children": children,
                })

            all_results.append({
                "page_index": page_idx,
                "image_path": img_path,
                "objects": enriched,
            })

        return {"pages": all_results}

    except Exception as e:
        # Stacktrace im Terminal
        traceback.print_exc()
        # Fehlermeldung im HTTP-Response
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Temp-Datei aufräumen
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/plan/debug")
async def debug_plan(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        if not suffix:
            suffix = ".pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        if suffix == ".pdf" or file.content_type == "application/pdf":
            image_paths = pdf_to_images(tmp_path)
        else:
            image_paths = [tmp_path]

        return {
            "suffix": suffix,
            "content_type": file.content_type,
            "image_paths": image_paths,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plan/debug_yolo")
async def debug_yolo(file: UploadFile = File(...)):
    """
    Nimmt ein PDF oder Bild, rendert erste Seite und führt YOLO-Erkennung durch.
    Zeigt erkannte Labels und Confidence-Werte.
    """
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        if not suffix:
            suffix = ".pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # PDF -> Bild
        if suffix == ".pdf" or file.content_type == "application/pdf":
            image_paths = pdf_to_images(tmp_path)
        else:
            image_paths = [tmp_path]

        first_image = image_paths[0]

        detections = detect_objects(first_image, conf=0.1)

        return {
            "file": file.filename,
            "pages": len(image_paths),
            "image_used": first_image,
            "detections": detections,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
