# backend/services/root_mapper.py

from backend.services.embedding_service import get_embedding, cosine_sim
import numpy as np

# Anzeige-Name (id) und Text, der für das Embedding benutzt wird
ROOT_DEFS = [
    {
        "id": "Beläge",
        "text": "Beläge, Wege, Plätze, Asphalt, Pflaster, Platten, Verkehrsflächen"
    },
    {
        "id": "Grünflächen",
        "text": "Grünflächen, Wiese, Rasen, Park, Bäume, Sträucher, Spielwiese"
    },
    {
        "id": "Wasserflächen",
        "text": "Wasserflächen, Teich, Ententeich, See, Pool, Schwimmbecken, Wasserbecken, Schwimmbad, Schwimmteich"
    },
    {
        "id": "Treppenanlagen",
        "text": "Treppenanlagen, Treppe, Stufen, Rampen, Aufgänge, Treppenlauf"
    },
]

# Embeddings einmal vorrechnen
ROOT_EMBS = {}
for root in ROOT_DEFS:
    emb = get_embedding(root["text"])
    ROOT_EMBS[root["id"]] = np.array(emb, dtype=np.float32)


def best_roots_for_term(term: str, top_k: int = 2):
    term_emb = np.array(get_embedding(term), dtype=np.float32)
    scores = []
    for root_id, emb in ROOT_EMBS.items():
        score = cosine_sim(term_emb.tolist(), emb.tolist())
        scores.append((root_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
