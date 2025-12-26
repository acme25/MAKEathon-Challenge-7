const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const resultImg = document.getElementById("result");
const loading = document.getElementById("loading");

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

  const response = await fetch("/process", {
    method: "POST",
    body: formData
  });

  const blob = await response.blob();
  resultImg.src = URL.createObjectURL(blob);

  loading.classList.add("hidden");
  resultImg.classList.remove("hidden");
}
