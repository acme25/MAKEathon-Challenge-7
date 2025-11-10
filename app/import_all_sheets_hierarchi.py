# python apps/import_all_sheets_hierarchi.py

import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# === .env laden ===
load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

EXCEL_PATH = "data/dreimaterial.xlsx"

# Alle Sheets der Datei öffnen
xls = pd.ExcelFile(EXCEL_PATH)
print("📘 Gefundene Sheets:", xls.sheet_names)


def create_path(tx, sheet_name, nodes):
    """
    nodes = Werte aus EINER Zeile (linke nach rechte Ebene)
    sheet_name = Name des Worksheets (z.B. 'Beläge', 'Grünflächen' ...)
    Wir erzeugen eindeutige uids: <Sheet>/<Pfad> damit Kontexte getrennt bleiben.
    """
    for i in range(len(nodes) - 1):
        parent = nodes[i]
        child  = nodes[i + 1]
        if not parent or not child or parent == child:
            continue

        # eindeutige IDs: Sheet + Pfad
        parent_uid = f"{sheet_name}/" + "/".join(nodes[:i+1])
        child_uid  = f"{sheet_name}/" + "/".join(nodes[:i+2])

        tx.run("""
            MERGE (p:Kategorie {uid:$pid})
              ON CREATE SET p.name = $pname, p.sheet = $sheet
            MERGE (c:Kategorie {uid:$cid})
              ON CREATE SET c.name = $cname, c.sheet = $sheet
            MERGE (p)-[:HAT_UNTERKATEGORIE]->(c)
        """, pid=parent_uid, pname=parent,
             cid=child_uid,  cname=child,
             sheet=sheet_name)


# === optional: DB leeren ===
with driver.session() as s:
    s.run("MATCH (n) DETACH DELETE n")

# === alle Sheets importieren ===
total_paths = 0

with driver.session() as session:
    for sheet_name in xls.sheet_names:
        print(f"\n🌳 Importiere Sheet: {sheet_name}")
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)

        # Nur leere Zellen zu "" machen – KEIN ffill(), damit er nichts „dazu erfindet“
        df = df.fillna("")

        for _, row in df.iterrows():
            # echte Werte dieser Zeile, von links nach rechts
            nodes = [str(x).strip() for x in row if str(x).strip()]
            if len(nodes) > 1:
                session.execute_write(create_path, sheet_name, nodes)
                total_paths += 1

print(f"\n✅ Insgesamt {total_paths} Pfade aus {len(xls.sheet_names)} Sheets importiert.")
driver.close()
