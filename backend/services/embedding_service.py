# backend/services/embedding_service.py

import os
import numpy as np
import config

EMBEDDING_BACKEND = config.EMBEDDING_BACKEND
EMBEDDING_MODEL = config.EMBEDDING_MODEL

if EMBEDDING_BACKEND == "huggingface":
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(EMBEDDING_MODEL)
else:
    raise ValueError(f"Unsupported EMBEDDING_BACKEND: {EMBEDDING_BACKEND}")


def get_embedding(text: str) -> list[float]:
    """
    Gibt einen Embedding-Vektor für den Text zurück.
    """
    text = text.strip()
    if not text:
        return []
    # sentence-transformers kann direkt Listen encoden
    emb = _model.encode([text])[0]   # shape (768,) als 1D-Array
    return emb.tolist()


def cosine_sim(a: list[float], b: list[float]) -> float:
    """
    Cosine Similarity zweier Vektoren.
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))