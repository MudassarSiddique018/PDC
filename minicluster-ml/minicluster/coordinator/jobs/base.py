"""Job model and the fault-tolerant task dispatcher shared by all job types."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Hashable, Sequence

from minicluster.coordinator.cluster import ClusterManager, TaskFailed, WorkerConnection, WorkerLost
from minicluster.coordinator.config import Settings
from minicluster.coordinator.events import EventBus

QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED = "queued", "running", "completed", "failed", "cancelled"
FINAL_STATES = {COMPLETED, FAILED, CANCELLED}


class JobCancelled(Exception):
    pass


class NoWorkersAvailable(Exception):
    pass


@dataclass
class Job:
    type: str
    spec: dict[str, Any]
    name: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    progress: float = 0.0
    phase: str = "Waiting in queue"
    workers: dict[str, dict[str, Any]] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = False

    def check_cancelled(self) -> None:
        if self.cancel_requested:
            raise JobCancelled("cancelled by user")

    @property
    def duration(self) -> float | None:
        if self.started_at is None:
            return None
        return (self.finished_at or time.time()) - self.started_at

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": self.duration,
            "progress": round(self.progress, 4),
            "phase": self.phase,
            "error": self.error,
            "headline": (self.result or {}).get("headline"),
        }

    def detail(self) -> dict[str, Any]:
        return {**self.summary(), "spec": self.spec, "workers": self.workers, "history": self.history, "result": self.result}


@dataclass
class JobContext:
    job: Job
    cluster: ClusterManager
    bus: EventBus
    settings: Settings
    datasets: Any  # DatasetRegistry (avoid an import cycle)
    _last_push: float = 0.0

    def log(self, message: str, level: str = "info", worker: str | None = None) -> None:
        self.bus.log(message, level=level, job_id=self.job.id, worker=worker)

    def update(self, force: bool = False, **fields: Any) -> None:
        for k, v in fields.items():
            setattr(self.job, k, v)
        now = time.time()
        if force or now - self._last_push > 0.25:  # throttle WebSocket traffic
            self._last_push = now
            self.bus.publish("job", self.job.detail())

    def worker_stat(self, worker_id: str) -> dict[str, Any]:
        return self.job.workers.setdefault(
            worker_id, {"tasks": 0, "items": 0, "compute_time": 0.0, "progress": 0.0, "state": "idle", "lost": False}
        )


Item = Hashable


class Dispatcher:
    """Runs a list of items on a pool of workers with automatic fail-over.

    * Every idle worker pulls the next item, giving dynamic load balancing:
      fast PCs simply process more items.
    * Items may have a *home* worker (e.g. the PC that already holds a data
      shard). A worker prefers its own items and only takes orphans whose home
      is gone, so data stays put between training epochs.
    * If a worker disconnects, its in-flight item goes back to the queue and
      another worker picks it up (and becomes its new home). Lost workers are
      replaced by idle online workers when available.
    """

    def __init__(self, ctx: JobContext, want: int):
        self.ctx = ctx
        self.want = max(1, want)
        self.pool: list[WorkerConnection] = []
        self.home: dict[Item, str] = {}
        self.ever_used: dict[str, WorkerConnection] = {}

    def recruit(self) -> list[WorkerConnection]:
        self.pool = [w for w in self.pool if w.online]
        for w in self.ctx.cluster.online_workers():
            if len(self.pool) >= self.want:
                break
            if all(p is not w for p in self.pool):
                self.pool.append(w)
                self.ever_used[w.session] = w
                w.current_job = self.ctx.job.id
                self.ctx.worker_stat(w.id)
        return self.pool

    async def ensure_pool(self) -> list[WorkerConnection]:
        if not self.recruit():
            self.ctx.update(phase="Waiting for workers to come online", force=True)
            if not await self.ctx.cluster.wait_for_worker(self.ctx.settings.no_worker_timeout):
                raise NoWorkersAvailable("no online workers - start at least one worker PC")
            self.recruit()
        return self.pool

    def release(self) -> None:
        for w in self.ever_used.values():
            if w.current_job == self.ctx.job.id:
                w.current_job = None

    def _pick(self, worker: WorkerConnection, pending: list[Item]) -> Item | None:
        pool_ids = {w.id for w in self.pool}
        for item in pending:
            if self.home.get(item) == worker.id:
                return item
        for item in pending:
            if self.home.get(item) not in pool_ids:
                return item
        return None

    async def map(
        self,
        items: Sequence[Item],
        fn: Callable[[WorkerConnection, Item], Awaitable[Any]],
        describe: Callable[[Item], str] = lambda i: f"task #{i}",
        on_done: Callable[[Item, WorkerConnection, Any], None] | None = None,
    ) -> dict[Item, Any]:
        job = self.ctx.job
        pending: list[Item] = list(items)
        results: dict[Item, Any] = {}
        running: dict[asyncio.Task, tuple[WorkerConnection, Item]] = {}
        lost_items: set[Item] = set()
        try:
            while len(results) < len(items):
                job.check_cancelled()
                if not self.recruit() and not running:
                    await self.ensure_pool()
                busy = {id(w) for w, _ in running.values()}
                for w in self.pool:
                    if id(w) in busy:
                        continue
                    item = self._pick(w, pending)
                    if item is None:
                        continue
                    pending.remove(item)
                    if item in lost_items:
                        lost_items.discard(item)
                        self.ctx.log(f"Task reassignment: {w.id} <- {describe(item)}", level="warn", worker=w.id)
                    self.home[item] = w.id
                    stat = self.ctx.worker_stat(w.id)
                    stat["state"] = "busy"
                    running[asyncio.create_task(fn(w, item))] = (w, item)
                if not running:
                    await asyncio.sleep(0.1)
                    continue
                done, _ = await asyncio.wait(running.keys(), timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    w, item = running.pop(task)
                    stat = self.ctx.worker_stat(w.id)
                    stat["state"] = "idle"
                    exc = task.exception()
                    if exc is None:
                        results[item] = task.result()
                        stat["tasks"] += 1
                        if on_done:
                            on_done(item, w, results[item])
                    elif isinstance(exc, WorkerLost):
                        stat["state"], stat["lost"] = "offline", True
                        self.home.pop(item, None)
                        pending.insert(0, item)
                        lost_items.add(item)
                        self.ctx.log(f"{w.id} OFFLINE during {describe(item)} - reassigning", level="warn", worker=w.id)
                    else:
                        raise exc
                self.ctx.update()
        finally:
            for task in running:
                task.cancel()
        return results


async def call(ctx: JobContext, w: WorkerConnection, kind: str, params: dict[str, Any], payload: bytes = b""):
    """Thin wrapper over ``cluster.call`` that keeps per-worker job stats."""
    data, out, elapsed = await ctx.cluster.call(w, kind, params, payload)
    ctx.worker_stat(w.id)["compute_time"] = ctx.worker_stat(w.id)["compute_time"] + elapsed
    return data, out, elapsed


__all__ = [
    "CANCELLED",
    "COMPLETED",
    "Dispatcher",
    "FAILED",
    "FINAL_STATES",
    "Job",
    "JobCancelled",
    "JobContext",
    "NoWorkersAvailable",
    "QUEUED",
    "RUNNING",
    "TaskFailed",
    "WorkerLost",
    "call",
]
