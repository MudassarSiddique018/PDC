"""SQLite persistence for job history (survives coordinator restarts)."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from minicluster.coordinator.jobs.base import FAILED, FINAL_STATES, Job

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL,
    spec        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL,
    progress    REAL NOT NULL DEFAULT 0,
    phase       TEXT,
    workers     TEXT,
    history     TEXT,
    result      TEXT,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""


class JobStore:
    def __init__(self, path: Path):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def save(self, job: Job) -> None:
        row = (
            job.id, job.type, job.name, job.status, json.dumps(job.spec), job.created_at, job.started_at,
            job.finished_at, job.progress, job.phase, json.dumps(job.workers), json.dumps(job.history),
            json.dumps(job.result) if job.result is not None else None, job.error,
        )
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
            self._db.commit()

    def load_all(self, limit: int = 500) -> list[Job]:
        with self._lock:
            cur = self._db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        jobs = []
        for r in rows:
            job = Job(type=r[1], spec=json.loads(r[4]), name=r[2], id=r[0], status=r[3], created_at=r[5],
                      started_at=r[6], finished_at=r[7], progress=r[8], phase=r[9] or "",
                      workers=json.loads(r[10] or "{}"), history=json.loads(r[11] or "[]"),
                      result=json.loads(r[12]) if r[12] else None, error=r[13])
            if job.status not in FINAL_STATES:  # the coordinator died while it was queued/running
                job.status, job.error, job.phase = FAILED, "interrupted: coordinator restarted", "Interrupted"
            jobs.append(job)
        return jobs

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._db.commit()

    def counts(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._db.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())

    def close(self) -> None:
        with self._lock:
            self._db.close()
