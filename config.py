import os
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()
 
# === Datenpfade ===
# PDF-Ordner nach Modul getrennt
DATA_DIR = "data"
PLANS_DIR = f"{DATA_DIR}/plans"
YOLO_DIR = f"{DATA_DIR}/yolo_datasets"
 
# === Modelleinstellungen ===
EMBEDDING_BACKEND = "huggingface"  # nur zur Info
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "openai/gpt-oss-20b"  # kann später auch was anderes sein
API_KEY_GROQ = os.environ.get("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"
YOLO_MODEL_NAME = "yolov8s.pt"
