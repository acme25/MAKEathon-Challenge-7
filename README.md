# Make-it-great
Trying out the makeathon challenge 7 by my own

## Use Case
Planners need to manually read and write down the materials from the construction plans. Zoë wants to automatize this process. Therefore we create a solution which is shown in this repository.

## How to use
### Voraussetzungen:
- VS Code oder Pycharm als Applikation
- `requirements.txt` ausführen während der Installation (siehe Schritt 3)
- Eigenen API Key -> Wir empfehlen CEREBRAS

1. ### Virtuelle Umgebung erstellen und aktivieren
```
python -m venv ./.venv
````
.\.venv\Scripts\Activate
pip install -r .\requirements.txt
```

4. ### .env Datei erstellen

----------------------------------------------
# File Autolabel.py
# Auto-Vorlabel für Baupläne (zusammenhängende Flächen)

Dieses Skript erkennt **zusammenhängende Flächen** in deinen Bauplänen (PDF/JPG/PNG) und erzeugt **visuelle Ergebnisbilder** mit farbigen Overlays. Es klassifiziert **nicht** (z. B. „Straße/Grünfläche“), sondern gruppiert nur Flächen – perfekt als Vorlabel, das du später in CVAT noch feingranular anpassen kannst.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install opencv-python numpy pdf2image pillow
# (Optional) Fallback für PDF -> Bild:
pip install pymupdf
```

**Poppler** (nur wenn `pdf2image` genutzt wird) installieren:
- macOS: `brew install poppler`
- Ubuntu/Debian: `sudo apt-get install poppler-utils`
- Windows: Poppler herunterladen und den `bin/`-Pfad zur `PATH`-Umgebungsvariablen hinzufügen.

## Nutzung

Lege deine Pläne in einen Ordner, z. B. `./plans`, und starte dann:

```bash
python autolabel_regions.py --input ./plans --output ./output --dpi 300 --clusters 8 --min-area 1500
```

Parameter (wichtigste):
- `--clusters` = Anzahl Farbcluster für K‑Means (mehr = feiner, langsamer)
- `--min-area` = minimale Regionfläche in Pixel (kleine Artefakte wegfiltern)
- `--max-size` = längste Bildkante zum Beschleunigen (0 = Originalgröße)
- `--outline` = Konturstärke

Die Ergebnisbilder liegen danach unter `./output/` und heißen z. B. `PlanXY_page01_segmented.png`.

## Tipps
- Wenn sehr viele winzige Regionen entstehen, **erhöhe `--min-area`** (z. B. 4000–8000).
- Wenn Regionen zu grob sind, **erhöhe `--clusters`** (z. B. 10–12).
- Für sehr große PDFs kannst du **`--max-size`** auf 1800–2200 setzen, das beschleunigt.

## Autolabel.py
- konvertiert automatisch alle deine PDF-Pläne zu Bildern,
- erkennt zusammenhängende Flächen (Regionen),
- zeichnet diese farbig und nummeriert sie,
- speichert fertige Bilder z. B. als
- plan_page01_segmented.png im Ordner output/.
