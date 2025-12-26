# KI-gestützte Materialerkennung aus Landschaftsplänen
MAKEathon Challenge 7

## Projektkontext & Auftrag
Im Rahmen der MAKEathon Challenge besteht die Aufgabe darin, den Grundstein für eine intelligente Plattform zur nachhaltigen Stadt- und Landschaftsplanung zu legen.

Heute verbringen Planner:innen viel Zeit damit, Pläne manuell zu analysieren und Materialien zu identifizieren. Dieses Projekt zeigt, wie dieser Prozess mithilfe von Computer Vision, KI und semantischer Modellierung automatisiert werden kann.

## Ziel der Anwendung
Die entwickelt Anwendung ermöglicht es:
- Landschafts- oder Stadtpläne als PDF hochzuladen
- Relevante Objekte automatisch mithilfe eines YOLOv8-Modells zu erkennen
- Erkannte Objekte semantisch auf Material-Oberklassen abzubilden
- Diese Klassen über einen Knowledge Graph (Neo4j) weiter zu strukturieren

## Funktionsübersicht
### Plan-Analyse
- Upload eines PDFs
- PDFs werden seitenweise in Bilder gerendet
- YOLOv8 erkennt Objekte pro Seite
- Ausgabe strukturierter Erkennungsergebnisse (Label, Confidence, Bounding Box)

### Semantische Einordnung
- YOLO-Labels werden mithilfe von Embeddings auf Oberklassen (Roots) gemappt
- Über einen Neo4j Knowledge Graph können:
  - ganze Teilbäume
  - Material-Hierarchien abgefragt werden
### Debug & Entwicklung
- Separate Debug-Endpunkte zum:
  - Prüfen des PDF-Renderings
  - Testen der YOLO-Erkennung auf einer einzelnen Seite

## Technischer Stack
- Backend: FastAPI (Python)
- Computer Vision: YOLOv8 (Ultralytics)
- Bilderverarbeitung: OpenCV
- PDF-Verarbeitung: pdf2image
- Semantik: Sentence Embeddings
- Knowledge Graph: Neo4j
- Frontend: HTML, CSS, JavaScript (Drag & Drop Upload)

#### How to use
### Voraussetzungen:
- Python 3.10 oder 3.11
- Installierte Abhängigkeiten (inkl. YOLOv8, FastAPI)
- Optional: laufende Neo4j-Instanz für Knowledge-Graph-Funktionen

### Weboberfläche verwenden
1. Öffne die Weboberfläche im Browser
2. Ziehe einen PDF-Plan per Drag & Drop in das Upload-Feld oder wähle eine Datei über den Dateidialog aus.
3. Die Datei wird automatisch hochgeladen und verarbeitet
   - PDFs werden seitenweise in Bilder umgewandelt
   - YOLOv8 erkennt relevante Objekte im Plan
   - Erkannte Objekte werden semantisch zugeordnet
4. Das Ergebnis wird:
   - Im Frontend angezeigt (Bild)
   - Oder als Download bereitgestellt (je nach Konfiguration)

## Ausblick
Mögliche Erweiterungen:
- Training eines domänenspezifischen YOLO-Modells
- Generierung eines annotierten PDFs
- Integration von:
    - Nachhaltigkeitskennzahlen
    - Lieferanten- und Materialdatenbanken
- Erweiterte Web-UI mit Projekt-Historie

