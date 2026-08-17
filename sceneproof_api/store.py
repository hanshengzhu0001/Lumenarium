from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


class JobStore:
    """Coordinator-owned SQLite job store.

    GPU hosts claim work through the coordinator HTTP API; they never share or
    open this database directly. This keeps leases safe across two hosts.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    release_id TEXT NOT NULL,
                    idempotency_key TEXT UNIQUE,
                    state TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    artifact_dir TEXT NOT NULL,
                    worker_id TEXT,
                    lease_expires_at REAL,
                    progress REAL NOT NULL DEFAULT 0,
                    error TEXT,
                    result_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_state_created
                    ON jobs(state, created_at);
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
            if "profile" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN profile TEXT NOT NULL DEFAULT 'medium'")
            if "parent_job_id" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN parent_job_id TEXT")
            if "trial_index" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN trial_index INTEGER")

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        value = dict(row)
        value["result"] = (
            json.loads(value.pop("result_json"))
            if value.get("result_json") else None
        )
        return value

    def create(
        self,
        *,
        release_id: str,
        input_path: str,
        artifact_dir: str,
        idempotency_key: str | None,
        profile: str = "medium",
        initial_state: str = "queued",
        parent_job_id: str | None = None,
        trial_index: int | None = None,
    ) -> tuple[dict[str, Any], bool]:
        if initial_state not in {"queued", "waiting"}:
            raise ValueError(f"invalid initial state: {initial_state}")
        now = time.time()
        job_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                prior = connection.execute(
                    "SELECT * FROM jobs WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if prior is not None:
                    connection.execute("COMMIT")
                    return self._decode(prior), False
            connection.execute(
                """INSERT INTO jobs(
                    job_id, release_id, idempotency_key, state, stage,
                    input_path, artifact_dir, created_at, updated_at, profile,
                    parent_job_id, trial_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, release_id, idempotency_key, initial_state,
                 initial_state, input_path, artifact_dir, now, now, profile,
                 parent_job_id, trial_index),
            )
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            connection.execute("COMMIT")
        return self._decode(row), True

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            return self._decode(
                connection.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
            )

    def children(self, parent_job_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE parent_job_id=? ORDER BY trial_index",
                (parent_job_id,),
            ).fetchall()
            return [self._decode(row) for row in rows]

    def begin_selection(self, parent_job_id: str) -> bool:
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET state='selecting', stage='best_selecting',
                   progress=0.97, updated_at=?
                   WHERE job_id=? AND state='waiting'""",
                (now, parent_job_id),
            )
            return cursor.rowcount == 1

    def finish_selection(
        self, *, parent_job_id: str, succeeded: bool,
        result: dict[str, Any] | None = None, error: str | None = None,
    ) -> bool:
        now = time.time()
        state = "succeeded" if succeeded else "failed"
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET state=?, stage=?, progress=?, result_json=?,
                   error=?, updated_at=? WHERE job_id=? AND state='selecting'""",
                (state, state, 1.0 if succeeded else 0.0,
                 json.dumps(result) if result is not None else None,
                 error, now, parent_job_id),
            )
            return cursor.rowcount == 1

    def owns_active_lease(self, *, job_id: str, worker_id: str) -> bool:
        now = time.time()
        with self._connect() as connection:
            row = connection.execute(
                """SELECT 1 FROM jobs WHERE job_id=? AND worker_id=?
                   AND state='running' AND lease_expires_at>=?""",
                (job_id, worker_id, now),
            ).fetchone()
            return row is not None

    def cancel(self, job_id: str) -> bool:
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET state='cancelled', stage='cancelled',
                   lease_expires_at=NULL, updated_at=?
                   WHERE job_id=? AND state IN ('queued','running','waiting','selecting')""",
                (now, job_id),
            )
            return cursor.rowcount == 1

    def claim(self, *, worker_id: str, lease_seconds: float) -> dict[str, Any] | None:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT * FROM jobs
                   WHERE state='queued'
                      OR (state='running' AND lease_expires_at < ?)
                   ORDER BY created_at LIMIT 1""",
                (now,),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            connection.execute(
                """UPDATE jobs SET state='running', stage='claimed', worker_id=?,
                   lease_expires_at=?, updated_at=? WHERE job_id=?""",
                (worker_id, now + lease_seconds, now, row["job_id"]),
            )
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)
            ).fetchone()
            connection.execute("COMMIT")
        return self._decode(claimed)

    def heartbeat(
        self,
        *,
        job_id: str,
        worker_id: str,
        stage: str,
        progress: float,
        lease_seconds: float,
    ) -> bool:
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET stage=?, progress=?, lease_expires_at=?, updated_at=?
                   WHERE job_id=? AND worker_id=? AND state='running'""",
                (stage, max(0.0, min(1.0, progress)), now + lease_seconds, now,
                 job_id, worker_id),
            )
            return cursor.rowcount == 1

    def finish(
        self,
        *,
        job_id: str,
        worker_id: str,
        succeeded: bool,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        now = time.time()
        state = "succeeded" if succeeded else "failed"
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET state=?, stage=?, progress=?, result_json=?,
                   error=?, lease_expires_at=NULL, updated_at=?
                   WHERE job_id=? AND worker_id=? AND state='running'""",
                (state, state, 1.0 if succeeded else 0.0,
                 json.dumps(result) if result is not None else None,
                 error, now, job_id, worker_id),
            )
            return cursor.rowcount == 1
