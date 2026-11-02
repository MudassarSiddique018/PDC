"""Cluster membership and remote task execution.

The coordinator runs an asyncio TCP server. Each worker PC opens one
persistent connection, registers, and then streams heartbeats while the
coordinator pushes tasks down the same socket. Every task is tracked by an
``asyncio.Future`` that resolves when the matching RESULT frame arrives, or
fails with :class:`WorkerLost` when the connection drops or heartbeats stop.
"""

from __future__ import annotations

import asyncio
import hmac
import itertools
import logging
import time
import uuid
from typing import Any

from minicluster.common.protocol import (
    PROTOCOL_VERSION,
    MsgType,
    ProtocolError,
    async_recv,
    async_send,
)
from minicluster.coordinator.config import Settings
from minicluster.coordinator.events import EventBus

log = logging.getLogger("minicluster.cluster")


class WorkerLost(Exception):
    """The worker disconnected (or went silent) before finishing a task."""


class TaskFailed(Exception):
    """The worker ran the task but it raised an exception."""


class WorkerConnection:
    def __init__(self, worker_id: str, info: dict[str, Any], reader, writer, peer: str):
        self.id = worker_id
        self.session = uuid.uuid4().hex  # changes on every reconnect
        self.info = info
        self.reader = reader
        self.writer = writer
        self.peer = peer
        self.online = True
        self.connected_at = time.time()
        self.last_seen = time.time()
        self.stats: dict[str, Any] = {"cpu": 0.0, "ram": 0.0, "per_cpu": []}
        self.pending: dict[str, asyncio.Future] = {}
        self.tasks_done = 0
        self.tasks_failed = 0
        self.busy_seconds = 0.0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.current_job: str | None = None
        self._send_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self.id

    async def send(self, header: dict[str, Any], payload: bytes = b"") -> None:
        async with self._send_lock:
            await async_send(self.writer, header, payload)
        self.bytes_sent += len(payload)

    def fail_pending(self, reason: str) -> None:
        for fut in self.pending.values():
            if not fut.done():
                fut.set_exception(WorkerLost(reason))
        self.pending.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": "online" if self.online else "offline",
            "address": self.peer,
            "hostname": self.info.get("hostname"),
            "platform": self.info.get("platform"),
            "version": self.info.get("version"),
            "cores": self.info.get("cores"),
            "cpu_count": self.info.get("cpu_count"),
            "ram_total": self.info.get("ram_total"),
            "gpu": self.info.get("gpu"),
            "cpu": self.stats.get("cpu", 0.0),
            "per_cpu": self.stats.get("per_cpu", []),
            "ram": self.stats.get("ram", 0.0),
            "running": len(self.pending),
            "tasks_done": self.tasks_done,
            "tasks_failed": self.tasks_failed,
            "busy_seconds": round(self.busy_seconds, 3),
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "current_job": self.current_job,
            "connected_at": self.connected_at,
            "last_seen": self.last_seen,
        }


class ClusterManager:
    def __init__(self, settings: Settings, bus: EventBus):
        self.settings = settings
        self.bus = bus
        self.workers: dict[str, WorkerConnection] = {}
        self._server: asyncio.base_events.Server | None = None
        self._monitor: asyncio.Task | None = None
        self._task_ids = itertools.count(1)
        self.tasks_dispatched = 0
        self._worker_joined = asyncio.Event()

    # ----------------------------------------------------------- lifecycle
    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_connection, self.settings.host, self.settings.worker_port)
        self._monitor = asyncio.create_task(self._monitor_loop())
        self.bus.log(f"Coordinator listening for workers on TCP {self.settings.host}:{self.settings.worker_port}")

    async def stop(self) -> None:
        if self._monitor:
            self._monitor.cancel()
        # Workers are only disconnected; they keep running and reconnect when
        # the coordinator comes back.
        for w in self.online_workers():
            w.writer.close()
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    # ------------------------------------------------------------- queries
    def online_workers(self) -> list[WorkerConnection]:
        return sorted((w for w in self.workers.values() if w.online), key=lambda w: w.id)

    def snapshot(self) -> dict[str, Any]:
        workers = sorted(self.workers.values(), key=lambda w: w.id)
        online = [w for w in workers if w.online]
        return {
            "workers": [w.snapshot() for w in workers],
            "online": len(online),
            "total": len(workers),
            "cores": sum(w.info.get("cores") or 0 for w in online),
            "running_tasks": sum(len(w.pending) for w in online),
            "tasks_dispatched": self.tasks_dispatched,
        }

    async def wait_for_worker(self, timeout: float) -> bool:
        if self.online_workers():
            return True
        self._worker_joined.clear()
        try:
            await asyncio.wait_for(self._worker_joined.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return bool(self.online_workers())

    # ------------------------------------------------------------- tasks
    async def call(
        self, worker: WorkerConnection, kind: str, params: dict[str, Any] | None = None, payload: bytes = b""
    ) -> tuple[dict[str, Any], bytes, float]:
        """Run one task on ``worker`` and return ``(data, payload, elapsed)``."""
        if not worker.online:
            raise WorkerLost(f"{worker.id} is offline")
        task_id = f"T{next(self._task_ids):06d}"
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        worker.pending[task_id] = fut
        self.tasks_dispatched += 1
        try:
            await worker.send({"type": MsgType.TASK, "task_id": task_id, "kind": kind, "params": params or {}}, payload)
        except (ConnectionError, OSError) as exc:
            worker.pending.pop(task_id, None)
            raise WorkerLost(f"{worker.id}: {exc}") from exc
        header, result_payload = await fut
        elapsed = float(header.get("elapsed", 0.0))
        worker.busy_seconds += elapsed
        if not header.get("ok"):
            worker.tasks_failed += 1
            raise TaskFailed(f"{worker.id}: {header.get('error', 'unknown error')}")
        worker.tasks_done += 1
        return header.get("data", {}), result_payload, elapsed

    # -------------------------------------------------------- connections
    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        peer_s = f"{peer[0]}:{peer[1]}" if peer else "?"
        try:
            hello = await asyncio.wait_for(async_recv(reader), timeout=10)
        except (asyncio.TimeoutError, ConnectionError, ProtocolError) as exc:
            log.warning("handshake with %s failed: %s", peer_s, exc)
            writer.close()
            return

        reason = self._validate_hello(hello.header)
        if reason:
            self.bus.log(f"Rejected worker from {peer_s}: {reason}", level="warn")
            try:
                await async_send(writer, {"type": MsgType.REJECT, "reason": reason})
            finally:
                writer.close()
            return

        worker_id = self._allocate_id(str(hello.header.get("name") or peer_s))
        worker = WorkerConnection(worker_id, hello.header, reader, writer, peer_s)
        previous = self.workers.get(worker_id)
        self.workers[worker_id] = worker
        await worker.send({"type": MsgType.WELCOME, "worker_id": worker_id, "protocol": PROTOCOL_VERSION})
        self.bus.log(
            f"{worker_id} {'RECONNECTED' if previous else 'ONLINE'} from {peer_s} "
            f"({worker.info.get('cores')} cores, {worker.info.get('platform')})",
            worker=worker_id,
        )
        self._worker_joined.set()
        self.bus.publish("cluster", self.snapshot())

        try:
            while True:
                msg = await async_recv(reader)
                worker.last_seen = time.time()
                if msg.type == MsgType.HEARTBEAT:
                    worker.stats = msg.header
                elif msg.type == MsgType.RESULT:
                    worker.bytes_received += len(msg.payload)
                    fut = worker.pending.pop(msg.header.get("task_id"), None)
                    if fut and not fut.done():
                        fut.set_result((msg.header, msg.payload))
        except (ConnectionError, ProtocolError, OSError) as exc:
            log.debug("worker %s connection ended: %s", worker_id, exc)
        finally:
            self._mark_offline(worker, "connection closed")

    def _validate_hello(self, header: dict[str, Any]) -> str | None:
        if header.get("type") != MsgType.REGISTER:
            return "expected a register message"
        if header.get("protocol") != PROTOCOL_VERSION:
            return f"protocol mismatch (worker {header.get('protocol')}, coordinator {PROTOCOL_VERSION})"
        secret = self.settings.cluster_secret
        if secret and not hmac.compare_digest(str(header.get("secret", "")), secret):
            return "invalid cluster secret"
        return None

    def _allocate_id(self, name: str) -> str:
        existing = self.workers.get(name)
        if existing is None or not existing.online:
            return name
        for i in itertools.count(2):
            candidate = f"{name}-{i}"
            w = self.workers.get(candidate)
            if w is None or not w.online:
                return candidate
        raise RuntimeError("unreachable")

    def _mark_offline(self, worker: WorkerConnection, reason: str) -> None:
        if not worker.online:
            return
        worker.online = False
        lost = len(worker.pending)
        worker.fail_pending(f"{worker.id} went offline ({reason})")
        try:
            worker.writer.close()
        except Exception:
            pass
        if self.workers.get(worker.id) is worker:
            extra = f" - {lost} in-flight task(s) will be reassigned" if lost else ""
            self.bus.log(f"{worker.id} OFFLINE ({reason}){extra}", level="warn", worker=worker.id)
            self.bus.publish("cluster", self.snapshot())

    def forget(self, worker_id: str) -> bool:
        w = self.workers.get(worker_id)
        if w is None or w.online:
            return False
        del self.workers[worker_id]
        return True

    async def _monitor_loop(self) -> None:
        """Detect silent workers and push a cluster snapshot every second."""
        while True:
            await asyncio.sleep(1.0)
            now = time.time()
            for w in self.online_workers():
                if now - w.last_seen > self.settings.heartbeat_timeout:
                    self._mark_offline(w, f"no heartbeat for {self.settings.heartbeat_timeout:.0f}s")
            self.bus.publish("cluster", self.snapshot())
