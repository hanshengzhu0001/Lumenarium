from __future__ import annotations

import argparse
import json
import os
import signal
import re
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import requests


ARTIFACTS = ("placement.json", "render.png", "evaluation.json", "result.json", "sceneproof-result.zip")
STAGE_PATTERN = re.compile(
    r"SCENEPROOF_API_STAGE=([A-Za-z0-9_.:-]+)\s+PROGRESS=([0-9.]+)"
)


class Worker:
    def __init__(self, *, api_url: str, token: str, gpu: int, worker_id: str,
                 project_root: Path, poll_seconds: float = 5.0):
        self.api_url = api_url.rstrip("/")
        self.headers = {"X-Worker-Token": token, "X-Worker-ID": worker_id}
        self.gpu = gpu
        self.worker_id = worker_id
        self.project_root = project_root
        self.poll_seconds = poll_seconds
        self.session = requests.Session()

    def _post(self, path: str, **kwargs):
        response = self.session.post(
            self.api_url + path, headers=self.headers, timeout=60, **kwargs
        )
        response.raise_for_status()
        return response

    def claim(self):
        return self._post(
            "/internal/claim",
            json={"worker_id": self.worker_id, "lease_seconds": 180},
        ).json()["job"]

    def heartbeat(
        self, job_id: str, stop: threading.Event, lease_lost: threading.Event,
        progress_state: dict, progress_lock: threading.Lock,
    ) -> None:
        while not stop.wait(30):
            try:
                with progress_lock:
                    stage = progress_state["stage"]
                    progress = progress_state["progress"]
                self._post(
                    "/internal/heartbeat",
                    json={
                        "job_id": job_id,
                        "worker_id": self.worker_id,
                        "stage": stage,
                        "progress": progress,
                        "lease_seconds": 180,
                    },
                )
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 409:
                    lease_lost.set()
                    return
            except Exception:
                # The main process still owns the authoritative outcome; a
                # transient coordinator outage must not kill Blender mid-write.
                pass

    def run_job(self, job: dict) -> None:
        job_id = job["job_id"]
        with tempfile.TemporaryDirectory(prefix=f"sceneproof-{job_id}-") as tmp:
            temporary = Path(tmp)
            input_path = temporary / "input.image"
            artifact_dir = temporary / "artifacts"
            artifact_dir.mkdir()
            response = self.session.get(
                f"{self.api_url}/internal/jobs/{job_id}/input",
                headers=self.headers,
                timeout=120,
            )
            response.raise_for_status()
            input_path.write_bytes(response.content)
            log_path = temporary / "worker.log"
            stop = threading.Event()
            lease_lost = threading.Event()
            progress_state = {"stage": "claimed", "progress": 0.01}
            progress_lock = threading.Lock()
            heartbeat = threading.Thread(
                target=self.heartbeat,
                args=(job_id, stop, lease_lost, progress_state, progress_lock),
                daemon=True,
            )
            heartbeat.start()
            try:
                command = [
                    "bash",
                    "scripts/run_sceneproof_frozen_single_job_fix115.sh",
                    job_id,
                    str(input_path),
                    str(artifact_dir),
                    str(self.gpu),
                    job.get("profile", "medium"),
                ]
                with log_path.open("wb") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=self.project_root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    deadline = time.monotonic() + float(
                        os.environ.get("SCENEPROOF_JOB_TIMEOUT", "21600")
                    )
                    consumed = 0
                    while process.poll() is None:
                        log.flush()
                        data = log_path.read_text(errors="replace")
                        if len(data) > consumed:
                            scan_from = max(0, consumed - 128)
                            for match in STAGE_PATTERN.finditer(data, scan_from):
                                with progress_lock:
                                    progress_state["stage"] = match.group(1)
                                    progress_state["progress"] = float(match.group(2))
                            consumed = len(data)
                        if lease_lost.is_set() or time.monotonic() >= deadline:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                            break
                        time.sleep(2)
                    return_code = process.returncode
                if lease_lost.is_set():
                    return
                if return_code != 0:
                    tail = log_path.read_text(errors="replace")[-8000:]
                    self._post(
                        "/internal/finish",
                        json={"job_id": job_id, "worker_id": self.worker_id,
                              "succeeded": False, "error": tail},
                    )
                    return
                for name in ARTIFACTS:
                    path = artifact_dir / name
                    if not path.is_file():
                        raise RuntimeError(f"missing artifact {name}")
                    with path.open("rb") as stream:
                        self._post(
                            f"/internal/jobs/{job_id}/artifacts/{name}",
                            files={"artifact": (name, stream)},
                        )
                result = json.loads((artifact_dir / "result.json").read_text())
                self._post(
                    "/internal/finish",
                    json={"job_id": job_id, "worker_id": self.worker_id,
                          "succeeded": True, "result": result},
                )
            except Exception as exc:
                self._post(
                    "/internal/finish",
                    json={"job_id": job_id, "worker_id": self.worker_id,
                          "succeeded": False, "error": repr(exc)},
                )
            finally:
                stop.set()
                heartbeat.join(timeout=2)

    def serve_forever(self) -> None:
        while True:
            try:
                job = self.claim()
                if job is None:
                    time.sleep(self.poll_seconds)
                    continue
                self.run_job(job)
            except requests.RequestException:
                time.sleep(self.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id")
    parser.add_argument("--project-root", default=str(Path(__file__).parents[1]))
    args = parser.parse_args()
    token = os.environ.get("SCENEPROOF_WORKER_TOKEN")
    if not token:
        raise SystemExit("SCENEPROOF_WORKER_TOKEN is required")
    worker_id = args.worker_id or f"{socket.gethostname()}:gpu{args.gpu}"
    Worker(
        api_url=args.api_url,
        token=token,
        gpu=args.gpu,
        worker_id=worker_id,
        project_root=Path(args.project_root).resolve(),
    ).serve_forever()


if __name__ == "__main__":
    main()
