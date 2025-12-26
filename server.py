from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import uuid
import shutil
import subprocess
from pathlib import Path

app = FastAPI()

# ✅ Frontend-Dateien (CSS/JS/PNGs) unter /static ausliefern
#    -> /static/style.css, /static/app.js, /static/tree_round.png ...
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ✅ Startseite: index.html aus dem frontend-Ordner ausliefern
@app.get("/")
def index():
    return FileResponse("frontend/index.html")


@app.post("/process")
async def process(file: UploadFile = File(...)):
    """
    1) Upload speichern
    2) autolabel.py aufrufen (input-dir -> output-dir)
    3) Erstes erzeugtes PNG zurückgeben
    """
    try:
        Path("tmp").mkdir(exist_ok=True)
        Path("output").mkdir(exist_ok=True)

        run_id = str(uuid.uuid4())
        in_dir = Path("tmp") / run_id
        out_dir = Path("output") / run_id
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Upload speichern
        ext = (Path(file.filename).suffix or ".bin").lower()
        in_path = in_dir / f"upload{ext}"

        with open(in_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # autolabel.py ausführen
        cmd = [
            "python",
            "autolabel.py",
            "--input", str(in_dir),
            "--output", str(out_dir),
            "--dpi", "300",
            "--clusters", "8",
            "--min-area", "1500",
            "--max-size", "2500",
            "--outline", "2",
        ]
        subprocess.run(cmd, check=True)

        # Erstes PNG finden
        pngs = sorted([p for p in out_dir.iterdir() if p.suffix.lower() == ".png"])
        if not pngs:
            return JSONResponse(
                {"error": "Kein PNG erzeugt. Prüfe output/ und Server-Logs."},
                status_code=500,
            )

        return FileResponse(str(pngs[0]), media_type="image/png")

    except subprocess.CalledProcessError as e:
        return JSONResponse(
            {"error": f"autolabel.py ist abgestürzt: {e}"},
            status_code=500,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
