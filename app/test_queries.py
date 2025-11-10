# python apps/test_queries.py
from backend.services.neo4j_service import neo4j_db

print("Kinder von Grünflächen:", neo4j_db.get_children("Grünflächen"))
print("Kinder von Beläge:", neo4j_db.get_children("Beläge"))

neo4j_db.close()