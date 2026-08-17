from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import time
import uuid
import zipfile
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
PUBLIC_ARTIFACTS = (
    "placement.json", "geometry.json", "render.png", "evaluation.json",
    "result.json", "sceneproof-result.zip",
)


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
            "best": "three independent V5-fast cold starts + GT-free high selector",
        },
        "input": {"formats": ["image/png", "image/jpeg"], "size": [1024, 1024]},
        "camera_policy": "source_s3_scene_camera_locked",
        "execution": "full_s0_s4_for_new_inputs; frozen_fix61_cache_for_identical_inputs; explicit_force_cold_rerun_supported",
        "failure_policy": "bounded_retry_then_explicit_pipeline_failed",
    }


@app.post("/v1/jobs", status_code=202)
async def create_job(
    image: UploadFile = File(...),
    profile: str = Form(default="medium"),
    force_rerun: bool = Form(default=False),
    force_cold_rerun: bool = Form(default=False),
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
    if profile not in {"fast", "medium", "best"}:
        raise HTTPException(status_code=422, detail="profile must be fast, medium or best")
    digest = hashlib.sha256(payload).hexdigest()
    rerun_id = uuid.uuid4().hex if (force_rerun or force_cold_rerun) else None
    rerun_kind = "cold" if force_cold_rerun else "profile"
    if rerun_id:
        key_prefix = idempotency_key or (
            f"sha256:{digest}:release:{RELEASE_ID}:profile:{profile}"
        )
        key = f"{key_prefix}:rerun:{rerun_kind}:{rerun_id}"
    else:
        key = idempotency_key or (
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
        initial_state="waiting" if profile == "best" else "queued",
    )
    if created and profile == "best":
        for trial_index in range(3):
            store.create(
                release_id=RELEASE_ID,
                input_path=str(provisional),
                artifact_dir=str(Path(job["artifact_dir"]) / "trials" / f"trial_{trial_index}"),
                idempotency_key=(
                    f"{key}:trial:{trial_index}:rerun:cold:{uuid.uuid4().hex}"
                ),
                profile="fast",
                parent_job_id=job["job_id"],
                trial_index=trial_index,
            )
    public = _public_job(job)
    if profile == "best":
        public["trials"] = [_public_job(child) for child in store.children(job["job_id"])]
    return {"job": public, "created": created}


def _select_best(parent: dict, children: list[dict]) -> None:
    from sceneproof_cold_start_selector import high_measure, rank

    selector_started = time.monotonic()
    rows = []
    for child in children:
        artifact = Path(child["artifact_dir"])
        candidate = {
            "candidate_id": f"trial_{child['trial_index']}",
            "geometry_path": str(artifact / "geometry.json"),
            "placement_path": str(artifact / "placement.json"),
        }
        evaluation = json.loads((artifact / "evaluation.json").read_text(encoding="utf-8"))
        row = {
            "candidate_id": candidate["candidate_id"],
            "job_id": child["job_id"],
            "trial_index": child["trial_index"],
            "certificate_passed": bool(evaluation.get("passed")),
            "unresolved_count": len(evaluation.get("unresolved_object_ids", [])),
            "high": high_measure(candidate),
        }
        base_rank = rank(row, "high")
        row["rank"] = [
            int(row["certificate_passed"]),
            -row["unresolved_count"],
            *base_rank,
            -int(child["trial_index"]),
        ]
        rows.append(row)
    winner = max(rows, key=lambda item: tuple(item["rank"]))
    selected = children[winner["trial_index"]]
    source = Path(selected["artifact_dir"])
    target = Path(parent["artifact_dir"])
    target.mkdir(parents=True, exist_ok=True)
    for name in PUBLIC_ARTIFACTS:
        if name != "sceneproof-result.zip":
            shutil.copy2(source / name, target / name)
    selector = {
        "schema_version": "sceneproof_v5_best_selector_v1",
        "gt_free": True,
        "policy": "certificate_then_unresolved_then_high_physical_v1",
        "selected_candidate_id": winner["candidate_id"],
        "selected_job_id": winner["job_id"],
        "candidates": rows,
    }
    evaluation_path = target / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["v5_best_selector"] = selector
    evaluation_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    result_path = target / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    selected_trial_timing = result.get("timing_seconds", {})
    trial_seconds = [
        float((child.get("result") or {}).get("timing_seconds", {}).get("end_to_end", 0.0))
        for child in children
    ]
    result.update({
        "job_id": parent["job_id"],
        "profile": "best",
        "selection_policy": selector["policy"],
        "selected_trial_job_id": winner["job_id"],
        "trial_job_ids": [child["job_id"] for child in children],
        "eligible_for_paper_metrics": True,
        "selected_trial_timing_seconds": selected_trial_timing,
        "timing_seconds": {
            "trial_end_to_end_seconds": trial_seconds,
            "useful_gpu_seconds": sum(trial_seconds),
            "two_a10_wall_seconds": time.time() - parent["created_at"],
            "selector_seconds": time.monotonic() - selector_started,
        },
    })
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(target / "sceneproof-result.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in PUBLIC_ARTIFACTS:
            if name != "sceneproof-result.zip":
                archive.write(target / name, arcname=name)
    if not store.finish_selection(
        parent_job_id=parent["job_id"], succeeded=True, result=result,
    ):
        raise RuntimeError("best parent selection lease lost")


def _refresh_best(job: dict) -> dict:
    if job.get("profile") != "best" or job["state"] not in {"waiting", "selecting"}:
        return job
    children = store.children(job["job_id"])
    if job["state"] == "selecting" and time.time() - job["updated_at"] > 300:
        try:
            _select_best(job, children)
        except Exception as error:
            store.finish_selection(
                parent_job_id=job["job_id"], succeeded=False,
                error=repr(error),
            )
    elif job["state"] == "waiting" and len(children) == 3:
        states = {child["state"] for child in children}
        if states.issubset({"succeeded", "failed", "cancelled"}):
            if all(child["state"] == "succeeded" for child in children):
                if store.begin_selection(job["job_id"]):
                    parent = store.get(job["job_id"])
                    try:
                        _select_best(parent, children)
                    except Exception as error:
                        store.finish_selection(
                            parent_job_id=job["job_id"], succeeded=False,
                            error=repr(error),
                        )
            elif store.begin_selection(job["job_id"]):
                store.finish_selection(
                    parent_job_id=job["job_id"], succeeded=False,
                    error="one or more V5-best cold trials failed",
                )
    return store.get(job["job_id"])


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    job = _refresh_best(job)
    public = _public_job(job)
    if job.get("profile") == "best":
        children = store.children(job_id)
        public["trials"] = [_public_job(child) for child in children]
        if job["state"] == "waiting" and children:
            public["progress"] = sum(child["progress"] for child in children) / len(children) * 0.96
            public["stage"] = "best_cold_trials"
    return public


@app.post("/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("profile") == "best":
        for child in store.children(job_id):
            store.cancel(child["job_id"])
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
    allowed = set(PUBLIC_ARTIFACTS)
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
        required = set(PUBLIC_ARTIFACTS)
        available = {
            path.name for path in Path(job["artifact_dir"]).glob("*") if path.is_file()
        }
        if not required.issubset(available):
            raise HTTPException(status_code=409, detail="required artifacts missing")
    ok = store.finish(**request.model_dump())
    if not ok:
        raise HTTPException(status_code=409, detail="lease not owned")
    return {"ok": True}
