import os
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.filter import apply_score_mask
from pipeline.aligner import align_score_to_audio
from pipeline.score_parser import extract_piano_roll, parse_score
from pipeline.separator import separate_vocals

app = FastAPI(title="PartIsolator")

# Serve frontend at /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

jobs: dict[str, dict] = {}

# Rate limiter: { ip: { "count": int, "month": "2024-01" } }
_rate_limits: dict[str, dict] = {}
_rate_lock = threading.Lock()
FREE_LIMIT = 3


def _check_rate_limit(ip: str) -> bool:
    """Returns True if the request is allowed; False if limit exceeded."""
    now_month = datetime.now().strftime("%Y-%m")
    with _rate_lock:
        entry = _rate_limits.get(ip)
        if entry is None or entry["month"] != now_month:
            _rate_limits[ip] = {"count": 1, "month": now_month}
            return True
        if entry["count"] >= FREE_LIMIT:
            return False
        entry["count"] += 1
        return True


# ---------------------------------------------------------------------------
# Background cleanup
# ---------------------------------------------------------------------------

def _cleanup_stale_jobs():
    """Remove jobs and temp files older than 1 hour."""
    while True:
        time.sleep(3600)
        cutoff = time.time() - 3600
        to_delete = [
            jid for jid, j in list(jobs.items())
            if j.get("created_at", 0) < cutoff
        ]
        for jid in to_delete:
            tmpdir = jobs[jid].get("tmpdir")
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
            jobs.pop(jid, None)


threading.Thread(target=_cleanup_stale_jobs, daemon=True).start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/parts")
async def get_parts(score: UploadFile = File(...)):
    """Return the list of part names in the uploaded score."""
    ext = os.path.splitext(score.filename or "")[1].lower()
    if ext not in (".xml", ".mxl", ".musicxml"):
        raise HTTPException(400, "Score must be .xml, .mxl, or .musicxml")

    content = await score.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "Score file exceeds 10 MB limit")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        parts = parse_score(tmp_path)
        return {"parts": parts}
    except Exception as e:
        raise HTTPException(422, f"Could not parse score: {e}")
    finally:
        os.unlink(tmp_path)


@app.post("/isolate")
async def isolate(
    request: Request,
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    score: UploadFile = File(...),
    part_name: str = Form(...),
):
    """Accept an audio file + score, kick off background processing."""
    ip = request.headers.get("X-Forwarded-For", request.client.host or "unknown")
    ip = ip.split(",")[0].strip()

    if not _check_rate_limit(ip):
        raise HTTPException(
            429,
            "Free tier limit reached (3 isolations/month). "
            "Upgrade to Pro for unlimited access.",
        )

    audio_ext = os.path.splitext(audio.filename or "")[1].lower()
    score_ext = os.path.splitext(score.filename or "")[1].lower()

    if audio_ext not in (".mp3", ".wav", ".flac", ".m4a"):
        raise HTTPException(400, "Audio must be MP3, WAV, FLAC, or M4A")
    if score_ext == ".pdf":
        raise HTTPException(
            400,
            "PDF conversion coming soon — please upload MusicXML (.xml or .mxl)",
        )
    if score_ext not in (".xml", ".mxl", ".musicxml"):
        raise HTTPException(400, "Score must be .xml, .mxl, or .musicxml")
    if not part_name.strip():
        raise HTTPException(400, "'part_name' is required")

    audio_content = await audio.read()
    score_content = await score.read()

    if len(audio_content) > 50 * 1024 * 1024:
        raise HTTPException(413, "Audio file exceeds 50 MB limit")
    if len(score_content) > 10 * 1024 * 1024:
        raise HTTPException(413, "Score file exceeds 10 MB limit")

    tmpdir = tempfile.mkdtemp(prefix="partiso_")
    audio_path = os.path.join(tmpdir, f"audio{audio_ext}")
    score_path = os.path.join(tmpdir, f"score{score_ext}")

    with open(audio_path, "wb") as f:
        f.write(audio_content)
    with open(score_path, "wb") as f:
        f.write(score_content)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "pct": 0,
        "msg": "Queued — starting soon",
        "error": None,
        "result_path": None,
        "tmpdir": tmpdir,
        "created_at": time.time(),
    }

    background_tasks.add_task(
        _run_pipeline, job_id, audio_path, score_path, part_name.strip(), tmpdir
    )

    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return {
        "status": job["status"],
        "pct": job["pct"],
        "msg": job["msg"],
        "error": job["error"],
    }


@app.get("/download/{job_id}")
async def download(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(202, "Not ready yet")
    result_path = job.get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(500, "Result file missing")
    return FileResponse(
        result_path,
        media_type="audio/wav",
        filename="isolated_part.wav",
    )


# ---------------------------------------------------------------------------
# Pipeline background task
# ---------------------------------------------------------------------------

def _update_job(job_id: str, status: str, pct: int, msg: str, error: Optional[str] = None):
    jobs[job_id].update({"status": status, "pct": pct, "msg": msg, "error": error})


def _run_pipeline(
    job_id: str,
    audio_path: str,
    score_path: str,
    part_name: str,
    tmpdir: str,
):
    try:
        # Step 1 — Demucs vocal separation (~60% of total wall time)
        _update_job(job_id, "running", 5,
                    "Separating vocals with Demucs (1–4 min depending on file length)…")
        sep_dir = os.path.join(tmpdir, "separated")
        os.makedirs(sep_dir, exist_ok=True)
        try:
            vocals_path = separate_vocals(audio_path, sep_dir)
        except RuntimeError as e:
            vocals_path = audio_path
            _update_job(job_id, "running", 35,
                        f"Demucs unavailable — using raw audio. ({str(e)[:120]})")

        # Step 2 — Parse score and build piano roll
        _update_job(job_id, "running", 35, "Parsing score and building piano roll…")
        piano_roll = extract_piano_roll(score_path, part_name)

        # Step 3 — Align score to audio via DTW
        _update_job(job_id, "running", 50, "Aligning score to audio timing (DTW)…")
        piano_roll_aligned = align_score_to_audio(
            piano_roll=piano_roll,
            audio_path=vocals_path,
        )

        # Step 4 — Apply time-varying frequency mask
        _update_job(job_id, "running", 75, "Applying time-varying frequency mask…")
        result_path = os.path.join(tmpdir, "result.wav")
        apply_score_mask(
            vocals_path=vocals_path,
            piano_roll_aligned=piano_roll_aligned,
            output_path=result_path,
        )

        jobs[job_id].update({
            "status": "done",
            "pct": 100,
            "msg": "Done! Your isolated part is ready.",
            "result_path": result_path,
        })

    except ValueError as e:
        _update_job(job_id, "error", 0, str(e), error=str(e))
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        msg = f"Unexpected error: {str(e)}"
        _update_job(job_id, "error", 0, msg, error=msg)
        shutil.rmtree(tmpdir, ignore_errors=True)
