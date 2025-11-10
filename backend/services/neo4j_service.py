from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

class Neo4jService:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASS")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query, params=None):
        with self.driver.session() as session:
            return list(session.run(query, params or {}))

    def get_children(self, name: str):
        query = """
        MATCH (p:Kategorie {name:$name})-[:HAT_UNTERKATEGORIE]->(c)
        RETURN DISTINCT c.name AS child
        ORDER BY child
        """
        return [r["child"] for r in self.run_query(query, {"name": name})]

    def get_subtree(self, root: str, depth: int = 5):
        query = f"""
        MATCH p = (:Kategorie {{name:$root}})-[:HAT_UNTERKATEGORIE*1..{depth}]->(c)
        RETURN p
        """
        return self.run_query(query, {"root": root})

neo4j_db = Neo4jService()
