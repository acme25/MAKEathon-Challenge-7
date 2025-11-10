# backend/services/semantic_mapper.py

import json
import numpy as np
from pathlib import Path
from backend.services.embedding_service import get_embedding, cosine_sim

EMB_PATH = Path("data/category_embeddings.json")

with EMB_PATH.open("r", encoding="utf-8") as f:
    CATEGORY_INDEX = json.load(f)

# Embeddings in numpy konvertieren für schnellere Similarity-Berechnung
for item in CATEGORY_INDEX:
    item["embedding"] = np.array(item["embedding"], dtype=np.float32)


def best_categories_for_term(term: str, top_k: int = 3):
    """
    Nimmt z.B. 'Kletterpark' oder 'Skatepark' und liefert die besten Kategorien aus dem KG zurück.
    """
    term_emb = np.array(get_embedding(term), dtype=np.float32)
    scores = []
    for item in CATEGORY_INDEX:
        sim = cosine_sim(term_emb, item["embedding"])
        scores.append((item["name"], sim))

    # nach Ähnlichkeit sortieren, höchste zuerst
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
