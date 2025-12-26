const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const resultImg = document.getElementById("result");
const loading = document.getElementById("loading");

// ✅ Backend-URL fest verdrahten (damit es immer passt)
const API_BASE = "http://127.0.0.1:8000";

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", (e) => {
  handleFile(e.target.files[0]);
});

async function handleFile(file) {
  if (!file) return;

  loading.classList.remove("hidden");
  resultImg.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${API_BASE}/process`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      alert(`Server-Fehler (${response.status}): ${text}`);
      loading.classList.add("hidden");
      return;
    }

    const blob = await response.blob();
    resultImg.src = URL.createObjectURL(blob);
    resultImg.classList.remove("hidden");
  } catch (err) {
    alert("Backend nicht erreichbar. Starte den Server mit: python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000");
  } finally {
    loading.classList.add("hidden");
  }
}
