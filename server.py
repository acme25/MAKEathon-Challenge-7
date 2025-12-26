from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import subprocess
import uuid
import os
import glob

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name="static")

INPUT_DIR = "./tmp"
OUTPUT_DIR = "./output"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def ui():
    with open("frontend/index.html") as f:
        return f.read()


@app.post("/process")
async def process(file: UploadFile):
    job_id = str(uuid.uuid4())
    input_path = os.path.join(INPUT_DIR, f"{job_id}_{file.filename}")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    subprocess.run([
        "python3",
        "autolabel.py",
        "--input", INPUT_DIR,
        "--output", OUTPUT_DIR
    ], check=True)

    result_files = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "*_segmented.png")),
        key=os.path.getmtime,
        reverse=True
    )

    return FileResponse(result_files[0])
