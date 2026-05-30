"""
BlurArroba no Modal.com — GPU serverless.

Como funciona:
  - A classe `Blurrer` roda em um container com GPU. O EasyOCR é carregado uma
    única vez em `@modal.enter()` e o container fica "quente" por alguns minutos.
  - O web app (FastAPI) serve o front e os endpoints de job. Ele NÃO usa GPU
    (é barato e escala sozinho).
  - Ao subir o vídeo, o web chama `Blurrer().process.spawn(...)`, que dispara o
    cold-start do container GPU (a GPU "acorda"). O processamento roda em
    background e o status/progresso vai para um `modal.Dict`.
  - O resultado é gravado num `modal.Volume`; o usuário faz polling e baixa.
  - `/wake` permite acordar a GPU já na seleção do arquivo (pré-aquecimento),
    encurtando o tempo percebido.

Deploy:
  pip install modal
  modal token new            # autentica (uma vez)
  modal serve modal_app.py   # dev: URL temporária com hot-reload
  modal deploy modal_app.py  # produção: URL permanente
"""
import os
import uuid
import tempfile

import modal

app = modal.App("blur-arroba")

# Imagem: ffmpeg (remux do áudio) + libs de runtime do OpenCV + deps Python.
# No Linux, `pip install torch` (via easyocr) já traz a build com CUDA, então
# o EasyOCR usa a GPU sem passos extras.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "opencv-python-headless",
        "easyocr",
        "numpy",
        "fastapi[standard]",
        "python-multipart",
    )
    # leva os módulos locais para dentro do container
    .add_local_python_source("detector", "processor")
    # serve o front a partir do arquivo local
    .add_local_file("frontend.html", "/assets/index.html")
)

# Estado compartilhado entre o web e a GPU.
status = modal.Dict.from_name("blur-arroba-status", create_if_missing=True)
outputs = modal.Volume.from_name("blur-arroba-outputs", create_if_missing=True)
OUT_DIR = "/outputs"

ALLOWED = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


@app.cls(
    image=image,
    gpu="T4",                       # GPU mais barata; suficiente para o OCR
    volumes={OUT_DIR: outputs},
    scaledown_window=300,           # mantém a GPU quente 5 min após o último uso
)
class Blurrer:
    @modal.enter()
    def load(self):
        from detector import AtDetector

        # carrega o modelo uma vez por container (cold-start paga isso só aqui)
        self.detector = AtDetector(langs=("en", "pt"), gpu=True)

    @modal.method()
    def ping(self):
        """Acorda/aquece o container GPU sem fazer trabalho."""
        return "ready"

    @modal.method()
    def process(self, job_id: str, video_bytes: bytes, ext: str, opts: dict):
        from processor import process_video

        def _set(**kw):
            cur = status.get(job_id, {})
            cur.update(kw)
            status[job_id] = cur

        _set(status="processing", progress=0.0)

        def on_progress(done, total):
            if total:
                _set(progress=round(done / total, 4))

        src = tempfile.mktemp(suffix=ext)
        out = os.path.join(OUT_DIR, f"{job_id}.mp4")
        with open(src, "wb") as f:
            f.write(video_bytes)

        try:
            process_video(src, out, self.detector, progress=on_progress, **opts)
            outputs.commit()                       # persiste o arquivo no volume
            _set(status="done", progress=1.0)
        except Exception as e:
            _set(status="error", error=str(e))
        finally:
            if os.path.exists(src):
                os.remove(src)


@app.function(image=image, volumes={OUT_DIR: outputs})
@modal.asgi_app()
def web():
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import HTMLResponse, FileResponse

    api = FastAPI(title="BlurArroba")

    with open("/assets/index.html", encoding="utf-8") as f:
        INDEX_HTML = f.read()

    @api.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @api.post("/wake")
    def wake():
        # dispara o cold-start da GPU enquanto o usuário ainda vai enviar
        Blurrer().ping.spawn()
        return {"status": "warming"}

    @api.post("/jobs")
    async def create_job(
        file: UploadFile = File(...),
        mode: str = Form("pixelate"),
        strength: int = Form(14),
        detect_scale: float = Form(1.0),
        min_conf: float = Form(0.30),
        sample_interval: int = Form(0),
    ):
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED:
            raise HTTPException(400, f"extensão não suportada: {ext}")

        data = await file.read()
        job_id = uuid.uuid4().hex
        status[job_id] = {"status": "queued", "progress": 0.0, "error": None}

        opts = {
            "mode": mode,
            "strength": strength,
            "detect_scale": detect_scale,
            "min_conf": min_conf,
            "sample_interval": sample_interval or None,
        }
        # acorda a GPU (se ainda não estiver) e processa em background
        Blurrer().process.spawn(job_id, data, ext, opts)
        return {"job_id": job_id, "status": "queued"}

    @api.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = status.get(job_id)
        if not job:
            raise HTTPException(404, "job não encontrado")
        return {k: v for k, v in job.items()}

    @api.get("/jobs/{job_id}/download")
    def download(job_id: str):
        job = status.get(job_id)
        if not job:
            raise HTTPException(404, "job não encontrado")
        if job.get("status") != "done":
            raise HTTPException(409, f"job não pronto: {job.get('status')}")
        outputs.reload()                            # vê o arquivo recém-commitado
        path = os.path.join(OUT_DIR, f"{job_id}.mp4")
        if not os.path.exists(path):
            raise HTTPException(404, "arquivo expirou ou não existe")
        return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")

    return api
