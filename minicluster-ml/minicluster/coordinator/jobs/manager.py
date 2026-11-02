"""Job queue: accepts submissions, runs them FIFO and persists their state."""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Any

from minicluster.coordinator.cluster import ClusterManager, TaskFailed
from minicluster.coordinator.config import Settings
from minicluster.coordinator.events import EventBus
from minicluster.coordinator.jobs.base import (
    CANCELLED,
    COMPLETED,
    FAILED,
    FINAL_STATES,
    QUEUED,
    RUNNING,
    Job,
    JobCancelled,
    JobContext,
    NoWorkersAvailable,
)
from minicluster.coordinator.jobs.benchmark import run_benchmark
from minicluster.coordinator.jobs.compute import run_compute
from minicluster.coordinator.jobs.training import run_training
from minicluster.coordinator.store import JobStore

log = logging.getLogger("minicluster.jobs")

RUNNERS = {"compute": run_compute, "training": run_training, "benchmark": run_benchmark}


class JobManager:
    def __init__(self, settings: Settings, cluster: ClusterManager, bus: EventBus, store: JobStore, datasets):
        self.settings = settings
        self.cluster = cluster
        self.bus = bus
        self.store = store
        self.datasets = datasets
        self.jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._runners: list[asyncio.Task] = []

    async def start(self) -> None:
        for job in reversed(self.store.load_all()):
            self.jobs[job.id] = job
            self.store.save(job)  # persist "interrupted" status fix-ups
        for i in range(max(1, self.settings.max_concurrent_jobs)):
            self._runners.append(asyncio.create_task(self._runner(i)))

    async def stop(self) -> None:
        for t in self._runners:
            t.cancel()
        for job in self.jobs.values():
            if job.status == RUNNING:
                job.cancel_requested = True

    # ------------------------------------------------------------- public
    def submit(self, job_type: str, spec: dict[str, Any], name: str = "") -> Job:
        job = Job(type=job_type, spec=spec, name=name or _default_name(job_type, spec))
        self.jobs[job.id] = job
        self.store.save(job)
        self._queue.put_nowait(job.id)
        self.bus.log(f"Job {job.id} queued: {job.name}", job_id=job.id)
        self.bus.publish("job", job.detail())
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        if job.status in FINAL_STATES:
            return job
        job.cancel_requested = True
        if job.status == QUEUED:
            self._finish(job, CANCELLED, error="cancelled by user")
        return job

    def delete(self, job_id: str) -> None:
        job = self.jobs[job_id]
        if job.status not in FINAL_STATES:
            raise ValueError("cancel the job before deleting it")
        del self.jobs[job_id]
        self.store.delete(job_id)
        self.bus.publish("job_deleted", {"id": job_id})

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        jobs = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.summary() for j in jobs[:limit]]

    def stats(self) -> dict[str, int]:
        out = {s: 0 for s in (QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED)}
        for j in self.jobs.values():
            out[j.status] = out.get(j.status, 0) + 1
        out["total"] = len(self.jobs)
        return out

    # ------------------------------------------------------------ runner
    async def _runner(self, slot: int) -> None:
        while True:
            job_id = await self._queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status != QUEUED:
                continue
            await self._run(job)

    async def _run(self, job: Job) -> None:
        ctx = JobContext(job, self.cluster, self.bus, self.settings, self.datasets)
        job.status, job.started_at, job.phase = RUNNING, time.time(), "Starting"
        self.store.save(job)
        ctx.log(f"Job {job.id} started: {job.name}")
        ctx.update(force=True)
        try:
            job.result = await RUNNERS[job.type](ctx, job.spec)
            job.progress = 1.0
            self._finish(job, COMPLETED, phase="Completed")
            ctx.log(f"Job {job.id} completed - {job.result.get('headline', '')}")
        except JobCancelled:
            self._finish(job, CANCELLED, error="cancelled by user")
            ctx.log(f"Job {job.id} cancelled", level="warn")
        except (NoWorkersAvailable, TaskFailed, ValueError, KeyError) as exc:
            self._finish(job, FAILED, error=str(exc))
            ctx.log(f"Job {job.id} failed: {exc}", level="error")
        except Exception as exc:  # unexpected bug: keep the coordinator alive
            log.error("job %s crashed:\n%s", job.id, traceback.format_exc())
            self._finish(job, FAILED, error=f"internal error: {type(exc).__name__}: {exc}")
            ctx.log(f"Job {job.id} crashed: {exc}", level="error")

    def _finish(self, job: Job, status: str, error: str | None = None, phase: str | None = None) -> None:
        job.status = status
        job.error = error
        job.finished_at = time.time()
        job.phase = phase or status.capitalize()
        for stat in job.workers.values():
            if stat.get("state") == "busy":
                stat["state"] = "idle"
        self.store.save(job)
        self.bus.publish("job", job.detail())


def _default_name(job_type: str, spec: dict[str, Any]) -> str:
    if job_type == "compute":
        return f"{spec.get('kernel', 'compute')} n={int(spec.get('n', 0)):,} on {spec.get('workers', '?')} worker(s)"
    if job_type == "training":
        rows = f" {int(spec['rows']):,} rows" if spec.get("dataset", "student_performance") == "student_performance" else ""
        return f"{spec.get('model')} on {spec.get('dataset')}{rows} ({spec.get('workers', '?')} workers)"
    if job_type == "benchmark":
        base = spec.get("base", {})
        what = (f"{base.get('kernel')} n={int(base.get('n', 0)):,}" if base.get("type") == "compute"
                else f"{base.get('model')} {int(base.get('rows', 0)):,} rows x {base.get('epochs')} epochs")
        return f"Speed-up benchmark: {what}"
    return job_type
