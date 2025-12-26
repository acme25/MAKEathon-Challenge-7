# PlanMind
Automatische Datenextraktion und Materialkategorisierung aus Bausplänen
MAKEathon Challenge 7

## Projektkontext & Auftrag
Architek:innen und Planer:innen arbeiten häufig mit komplexen Entwurfsplänen im PDF-Format, die eine grosse Vielfalt an Baumaterialien enthalten.
Die manuelle Extraktion und Kategorisierung dieser Materialien ist zeitaufwendig, fehleranfällig und verzögert Kostenkalkulation sowie Projektplanung.

## Ziel der Anwendung
Durch die Kombination von Computer Vision (OpenCV), LLM-basierter Semantik und einem Wissensgraphen verfolgt PlanMind folgende Ziele:
- Identifikation aller relevanten Materialien und Komponenten direkt aus PDF-Plänen
- Verständnis der Beziehungen zwischen verschiedenen Materialtypen
- Erstellung einer strukturierten, kategorisierten Materialliste
- Reduktion manueller Arbeit und Erhöhung der Genauigkeit
- Unterstützung einer datengetriebenen Planung in Architektur- und Designprozessen

## Zielsetzung 
Durch die Kombination von Computer Vision (OpenCV), LLM-basierter Semantik und einem Wissensgraphen verfolgt PlanMind folgende Ziele:
- Identifikation aller relevanter Materialien und Komponenten direkt aus PDF-Plänen
- Verständnis der Beziehungen zwischen verschiedenen Materialtypen
- Erstellung einer strukturierten, kategorisierten Materiallisten
- Reduktion manueller Arbeit und Erhöhung der Genauigkeit

## Technologischer Ansatz
- Computer Vision (OpenCV)
  - Erkennung zusammenhängender Flächen (Segmentierung)
  - Kontur- und Mustererkennung
  - Farbige Overlays
  - K-Means-Farbclustering zur Materialunterscheidung
- LLM-basierte Semantik
- Wissensgraph (Neo4j)

## Systemprozess
1. Upload eines Entwurfsplans
2. Analyse des Plans mittels OpenCV
3. Erkennung von Formen und Flächen
4. Semantische Zuordnung der erkannten Strukturen zu Materialien
5. Abgleich mit Excel-Materialliste
6. Ausgabe einer kategorisierten Materialliste

## How to use
1. Backend starten
2. Browser öffen
3. Bauplan (PDF) hochladen
4. Automatische Analyse & Materialkategorisierung abwarten
5. Strukturierte Ergebnisse weiterverwenden




