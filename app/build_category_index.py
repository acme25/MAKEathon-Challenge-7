# python -m apps.build_category_index

import json
from pathlib import Path
from backend.services.neo4j_service import neo4j_db
from backend.services.embedding_service import get_embedding

OUTPUT_PATH = Path("data/category_embeddings.json")

# 1. Kategorien aus Neo4j holen
query = """
MATCH (k:Kategorie)
RETURN DISTINCT k.name AS name
"""
rows = neo4j_db.run_query(query)
names = sorted({row["name"] for row in rows})

print(f"📊 {len(names)} Kategorien gefunden – Embeddings werden berechnet...")

index = []
for i, name in enumerate(names, start=1):
    emb = get_embedding(name)
    index.append({"name": name, "embedding": emb})
    if i % 20 == 0:
        print(f"  -> {i} / {len(names)} fertig")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(index, f)

print(f"✅ Embedding-Index gespeichert unter {OUTPUT_PATH}")
neo4j_db.close()