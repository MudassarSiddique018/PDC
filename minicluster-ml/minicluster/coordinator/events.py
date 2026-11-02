"""In-process publish/subscribe bus that feeds the dashboard WebSockets."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

log = logging.getLogger("minicluster.events")


class EventBus:
    def __init__(self, history: int = 300):
        self._subscribers: set[asyncio.Queue] = set()
        self.recent_logs: deque[dict[str, Any]] = deque(maxlen=history)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def publish(self, kind: str, data: Any) -> None:
        event = {"type": kind, "ts": time.time(), "data": data}
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # slow client: drop it rather than block the cluster
                self._subscribers.discard(q)

    def log(self, message: str, level: str = "info", job_id: str | None = None, worker: str | None = None) -> None:
        entry = {"ts": time.time(), "level": level, "message": message, "job_id": job_id, "worker": worker}
        self.recent_logs.append(entry)
        getattr(log, "warning" if level == "warn" else level, log.info)(message)
        self.publish("log", entry)
