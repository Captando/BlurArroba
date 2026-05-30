import os
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from detector import AtDetector
from processor import process_video

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/atblur/in")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/atblur/out")
USE_GPU = os.environ.get("USE_GPU", "0") == "1"
ALLOWED = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

state = {"detector": None}
jobs = {}
lock = threading.Lock()
# Produção: trocar por arq/Celery + Redis e N workers. max_workers=1 serializa
# os jobs (1 modelo na memória, sem disputa de VRAM).
executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app):
    state["detector"] = AtDetector(langs=("en", "pt"), gpu=USE_GPU)
    yield
    executor.shutdown(wait=False)


app = FastAPI(title="At Blur API", lifespan=lifespan)


def _set(job_id, **kw):
    with lock:
        jobs[job_id].update(kw)


def _run(job_id, src, out, opts):
    _set(job_id, status="processing", progress=0.0)

    def on_progress(done, total):
        if total:
            _set(job_id, progress=round(done / total, 4))

    try:
        process_video(src, out, state["detector"], progress=on_progress, **opts)
        _set(job_id, status="done", progress=1.0)
    except Exception as e:
        _set(job_id, status="error", error=str(e))
    finally:
        if os.path.exists(src):
            os.remove(src)


@app.post("/jobs")
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
        raise HTTPException(400, f"unsupported extension: {ext}")

    job_id = uuid.uuid4().hex
    src = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    out = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")

    with open(src, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    opts = {
        "mode": mode,
        "strength": strength,
        "detect_scale": detect_scale,
        "min_conf": min_conf,
        "sample_interval": sample_interval or None,
    }
    with lock:
        jobs[job_id] = {"status": "queued", "progress": 0.0, "out": out, "error": None}

    executor.submit(_run, job_id, src, out, opts)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    with lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {k: v for k, v in job.items() if k != "out"}


@app.get("/jobs/{job_id}/download")
async def download(job_id: str):
    with lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"job not ready: {job['status']}")
    return FileResponse(job["out"], media_type="video/mp4", filename=f"{job_id}.mp4")
