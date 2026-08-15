from __future__ import annotations

import hashlib
import io
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from pydantic import BaseModel, Field

from . import RELEASE_ID
from .store import JobStore


STATE_ROOT = Path(os.environ.get("SCENEPROOF_API_STATE_ROOT", "api_state")).resolve()
INPUT_ROOT = STATE_ROOT / "inputs"
ARTIFACT_ROOT = STATE_ROOT / "artifacts"
INPUT_ROOT.mkdir(parents=True, exist_ok=True)
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
store = JobStore(STATE_ROOT / "jobs.sqlite3")
app = FastAPI(title="SceneProof API", version="1.0.0")
UI_PATH = Path(__file__).with_name("ui.html")


@app.get("/", response_class=HTMLResponse)
def user_interface() -> str:
    return UI_PATH.read_text(encoding="utf-8")
    # Legacy inline UI retained below for source-package compatibility.
    return """<!doctype html><html><head><meta charset='utf-8'><title>SceneProof</title>
<style>body{font:16px system-ui;max-width:760px;margin:48px auto;padding:0 20px}button,select,input{font:inherit;margin:8px 0;padding:8px}pre{background:#f4f4f4;padding:12px;white-space:pre-wrap}img{max-width:100%}</style></head>
<body><h1>SceneProof</h1><p>Upload a 1024×1024 PNG/JPEG and select a quality profile.</p>
<form id='f'><input name='image' type='file' accept='image/png,image/jpeg' required><br>
<select name='profile'><option value='fast'>V5-fast — frozen Fix61</option><option value='medium' selected>V5-medium — Fix61 + visual-safe cleanup</option></select><br>
<button>Generate scene</button></form><pre id='s'>Ready</pre><div id='o'></div>
<script>const f=document.querySelector('#f'),s=document.querySelector('#s'),o=document.querySelector('#o');f.onsubmit=async e=>{e.preventDefault();o.innerHTML='';let r=await fetch('/v1/jobs',{method:'POST',body:new FormData(f)}),b=await r.json();if(!r.ok){s.textContent=JSON.stringify(b,null,2);return}let id=b.job.job_id;s.textContent='Queued '+id;let t=setInterval(async()=>{let j=await (await fetch('/v1/jobs/'+id)).json();s.textContent=JSON.stringify(j,null,2);if(j.state==='succeeded'){clearInterval(t);o.innerHTML=`<img src='/v1/jobs/${id}/artifacts/render.png'><p><a href='/v1/jobs/${id}/artifacts/sceneproof-result.zip'>Download result package</a></p>`}else if(['failed','cancelled'].includes(j.state))clearInterval(t)},3000)};</script></body></html>"""


class ClaimRequest(BaseModel):
    worker_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")
    lease_seconds: float = Field(default=180.0, ge=30.0, le=900.0)


class HeartbeatRequest(ClaimRequest):
    job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    stage: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")
    progress: float = Field(ge=0.0, le=1.0)


class FinishRequest(BaseModel):
    job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    worker_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")
    succeeded: bool
    result: dict | None = None
    error: str | None = Field(default=None, max_length=8192)


def _authorize_worker(token: str | None) -> None:
    expected = os.environ.get("SCENEPROOF_WORKER_TOKEN")
    if not expected or token != expected:
        raise HTTPException(status_code=401, detail="invalid worker token")


def _public_job(job: dict) -> dict:
    return {
        key: value for key, value in job.items()
        if key not in {"input_path", "artifact_dir", "worker_id", "lease_expires_at"}
    }


@app.get("/healthz")
def health() -> dict:
    return {"ok": True, "release_id": RELEASE_ID}


@app.get("/v1/releases/current")
def current_release() -> dict:
    return {
        "release_id": RELEASE_ID,
        "pipeline": [
            "deepsearch",
            "scenelm_fix61",
            "optional_sceneproof_fix114",
            "conservative_dominant_true_mesh_support_fix140_runnerfix1",
        ],
        "profiles": {
            "fast": "frozen Fix61; no online pose mutation",
            "medium": "Fix61 + presentation-only floor fallback and bounded render suppression",
        },
        "input": {"formats": ["image/png", "image/jpeg"], "size": [1024, 1024]},
        "camera_policy": "source_s3_scene_camera_locked",
        "execution": "full_s0_s4_for_new_inputs; frozen_fix61_cache_for_identical_inputs",
        "failure_policy": "bounded_retry_then_explicit_pipeline_failed",
    }


@app.post("/v1/jobs", status_code=202)
async def create_job(
    image: UploadFile = File(...),
    profile: str = Form(default="medium"),
    force_rerun: bool = Form(default=False),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict:
    if image.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(status_code=415, detail="PNG or JPEG required")
    payload = await image.read()
    if not payload or len(payload) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="image must be 1 byte to 25 MiB")
    try:
        with Image.open(io.BytesIO(payload)) as decoded:
            decoded.verify()
        with Image.open(io.BytesIO(payload)) as decoded:
            if decoded.size != (1024, 1024):
                raise HTTPException(status_code=422, detail="image must be 1024x1024")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="invalid image") from exc
    if profile not in {"fast", "medium"}:
        raise HTTPException(status_code=422, detail="profile must be fast or medium")
    digest = hashlib.sha256(payload).hexdigest()
    rerun_id = uuid.uuid4().hex if force_rerun else None
    key = idempotency_key or (
        f"sha256:{digest}:release:{RELEASE_ID}:profile:{profile}:rerun:{rerun_id}"
        if rerun_id else
        f"sha256:{digest}:release:{RELEASE_ID}:profile:{profile}"
    )
    provisional = STATE_ROOT / "pending" / f"{digest}.image"
    provisional.parent.mkdir(parents=True, exist_ok=True)
    provisional.write_bytes(payload)
    job, created = store.create(
        release_id=RELEASE_ID,
        input_path=str(provisional),
        artifact_dir=str(
            ARTIFACT_ROOT / RELEASE_ID / digest / profile
            / (rerun_id or "canonical")
        ),
        idempotency_key=key,
        profile=profile,
    )
    return {"job": _public_job(job), "created": created}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _public_job(job)


@app.post("/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if not store.cancel(job_id):
        raise HTTPException(status_code=409, detail="job is already terminal")
    return _public_job(store.get(job_id))


@app.get("/v1/jobs/{job_id}/artifacts/{name}")
def get_artifact(job_id: str, name: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job["state"] != "succeeded":
        raise HTTPException(status_code=409, detail="job is not complete")
    if Path(name).name != name:
        raise HTTPException(status_code=400, detail="invalid artifact name")
    path = Path(job["artifact_dir"]) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path)


@app.post("/internal/claim")
def claim(request: ClaimRequest, x_worker_token: str | None = Header(default=None)):
    _authorize_worker(x_worker_token)
    return {"job": store.claim(worker_id=request.worker_id,
                               lease_seconds=request.lease_seconds)}


@app.get("/internal/jobs/{job_id}/input")
def worker_input(
    job_id: str,
    x_worker_token: str | None = Header(default=None),
    x_worker_id: str | None = Header(default=None),
):
    _authorize_worker(x_worker_token)
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if not x_worker_id or not store.owns_active_lease(
        job_id=job_id, worker_id=x_worker_id
    ):
        raise HTTPException(status_code=409, detail="lease not owned")
    return FileResponse(job["input_path"], media_type="application/octet-stream")


@app.post("/internal/jobs/{job_id}/artifacts/{name}")
async def worker_artifact(
    job_id: str,
    name: str,
    artifact: UploadFile = File(...),
    x_worker_token: str | None = Header(default=None),
    x_worker_id: str | None = Header(default=None),
):
    _authorize_worker(x_worker_token)
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if not x_worker_id or not store.owns_active_lease(
        job_id=job_id, worker_id=x_worker_id
    ):
        raise HTTPException(status_code=409, detail="lease not owned")
    allowed = {"placement.json", "render.png", "evaluation.json", "result.json", "sceneproof-result.zip"}
    if name not in allowed:
        raise HTTPException(status_code=400, detail="unsupported artifact")
    payload = await artifact.read()
    if len(payload) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="artifact too large")
    directory = Path(job["artifact_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{name}.upload"
    temporary.write_bytes(payload)
    temporary.replace(directory / name)
    return {"ok": True, "bytes": len(payload)}


@app.post("/internal/heartbeat")
def heartbeat(request: HeartbeatRequest,
              x_worker_token: str | None = Header(default=None)):
    _authorize_worker(x_worker_token)
    ok = store.heartbeat(**request.model_dump())
    if not ok:
        raise HTTPException(status_code=409, detail="lease not owned")
    return {"ok": True}


@app.post("/internal/finish")
def finish(request: FinishRequest,
           x_worker_token: str | None = Header(default=None)):
    _authorize_worker(x_worker_token)
    if request.succeeded:
        job = store.get(request.job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        required = {"placement.json", "render.png", "evaluation.json", "result.json", "sceneproof-result.zip"}
        available = {
            path.name for path in Path(job["artifact_dir"]).glob("*") if path.is_file()
        }
        if not required.issubset(available):
            raise HTTPException(status_code=409, detail="required artifacts missing")
    ok = store.finish(**request.model_dump())
    if not ok:
        raise HTTPException(status_code=409, detail="lease not owned")
    return {"ok": True}
