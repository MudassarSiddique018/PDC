"""Worker node: runs on every lab PC and executes tasks sent by the coordinator.

Concurrency model (each piece maps to a lab-manual topic):

* **Socket programming** - one persistent TCP connection to the coordinator.
* **Multithreading**     - a receiver thread, a heartbeat thread and executor
                           thread(s) share that socket (sends are serialised
                           with a lock).
* **Multiprocessing**    - CPU-heavy work inside a task is split across the
                           PC's cores with a ``multiprocessing.Pool``; training
                           shards live in ``shared_memory`` so the processes
                           read them without copying.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import platform
import queue
import shutil
import signal
import socket
import subprocess
import threading
import time
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import psutil

from minicluster import __version__
from minicluster.common import compute, ml
from minicluster.common.protocol import (
    PROTOCOL_VERSION,
    Message,
    MsgType,
    pack_arrays,
    recv_message,
    send_message,
    unpack_arrays,
)

log = logging.getLogger("minicluster.worker")


# --------------------------------------------------------------------------- #
# Functions executed inside pool processes (must be module level to pickle)
# --------------------------------------------------------------------------- #
_ATTACHED: dict[str, tuple[shared_memory.SharedMemory, np.ndarray]] = {}


def _attach(spec: tuple[str, tuple[int, ...], str]) -> np.ndarray:
    name, shape, dtype = spec
    if name not in _ATTACHED:
        # Pool processes share the parent's resource tracker, so the (idempotent)
        # re-registration on attach is harmless; the parent unlinks the block.
        shm = shared_memory.SharedMemory(name=name)
        _ATTACHED[name] = (shm, np.ndarray(shape, dtype=dtype, buffer=shm.buf))
    return _ATTACHED[name][1]


def _grad_chunk(args):
    model, xspec, yspec, W, a, b = args
    X, y = _attach(xspec), _attach(yspec)
    return ml.loss_and_grad_sum(model, X[a:b], y[a:b], W)


def _kernel_chunk(args):
    kernel, a, b = args
    return compute.KERNELS[kernel](a, b)


def _pool_init() -> None:
    # Forked children inherit the parent's SIGTERM handler, which would make
    # Pool.terminate() unable to stop them; Ctrl+C is handled by the parent.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# --------------------------------------------------------------------------- #
# Shard storage
# --------------------------------------------------------------------------- #
class Shard:
    """A slice of a training dataset, optionally backed by shared memory."""

    def __init__(self, X: np.ndarray, y: np.ndarray, shared: bool):
        self.blocks: list[shared_memory.SharedMemory] = []
        if shared:
            self.X, self.xspec = self._to_shared(X)
            self.y, self.yspec = self._to_shared(y)
        else:
            self.X, self.y = X, y
            self.xspec = self.yspec = None

    def _to_shared(self, arr: np.ndarray):
        arr = np.ascontiguousarray(arr)
        shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        view[...] = arr
        self.blocks.append(shm)
        return view, (shm.name, arr.shape, arr.dtype.str)

    def close(self) -> None:
        self.X = self.y = None  # drop views before closing the buffers
        for shm in self.blocks:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        self.blocks.clear()


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def detect_gpu() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


class RegistrationRejected(Exception):
    """The coordinator refused this worker (bad secret, protocol mismatch...)."""


class WorkerNode:
    def __init__(
        self,
        host: str,
        port: int,
        name: str | None = None,
        cores: int | None = None,
        secret: str = "",
        heartbeat_interval: float = 1.0,
        reconnect: bool = True,
        slowdown: float = 1.0,
    ):
        self.host = host
        self.port = port
        self.name = name or socket.gethostname()
        self.cores = max(1, cores or os.cpu_count() or 1)
        self.secret = secret
        self.heartbeat_interval = heartbeat_interval
        self.reconnect = reconnect
        self.slowdown = max(1.0, slowdown)

        self.worker_id: str | None = None
        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._tasks: queue.Queue[Message | None] = queue.Queue()
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._running_tasks = 0
        self._completed = 0
        self._shards: dict[tuple[str, int], Shard] = {}
        self._pool: Any = None
        self._gpu = detect_gpu()
        psutil.cpu_percent(None)  # prime the CPU counter

    # ----------------------------------------------------------------- pool
    def _get_pool(self):
        if self._pool is None and self.cores > 1:
            self._pool = mp.get_context().Pool(self.cores, initializer=_pool_init)
        return self._pool

    def _reset_pool(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    # ------------------------------------------------------------- lifecycle
    def run(self) -> None:
        """Connect, serve tasks, and reconnect with back-off until stopped."""
        threading.Thread(target=self._executor_loop, name="executor", daemon=True).start()
        threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True).start()
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._connect()
                backoff = 1.0
                self._receive_loop()
            except RegistrationRejected as exc:
                log.error("coordinator rejected this worker: %s", exc)
                break
            except (ConnectionError, OSError) as exc:
                if self._stop.is_set():
                    break
                log.warning("connection lost: %s", exc)
            finally:
                self._disconnect()
            if not self.reconnect or self._stop.is_set():
                break
            log.info("reconnecting in %.0fs ...", backoff)
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 15.0)
        self.close()

    def stop(self) -> None:
        self._stop.set()
        self._disconnect()

    def close(self) -> None:
        self._tasks.put(None)
        for shard in self._shards.values():
            shard.close()
        self._shards.clear()
        self._reset_pool()

    def _connect(self) -> None:
        log.info("connecting to coordinator %s:%s", self.host, self.port)
        sock = socket.create_connection((self.host, self.port), timeout=10)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        mem = psutil.virtual_memory()
        send_message(
            sock,
            {
                "type": MsgType.REGISTER,
                "protocol": PROTOCOL_VERSION,
                "version": __version__,
                "name": self.name,
                "secret": self.secret,
                "hostname": socket.gethostname(),
                "platform": f"{platform.system()} {platform.release()}",
                "cores": self.cores,
                "cpu_count": os.cpu_count(),
                "ram_total": mem.total,
                "gpu": self._gpu,
            },
        )
        reply = recv_message(sock)
        if reply.type == MsgType.REJECT:
            sock.close()
            raise RegistrationRejected(reply.header.get("reason", "rejected"))
        if reply.type != MsgType.WELCOME:
            sock.close()
            raise ConnectionError(f"unexpected reply {reply.type}")
        sock.settimeout(None)
        self.worker_id = reply.header["worker_id"]
        self._sock = sock
        # A fresh session: the coordinator no longer knows about old shards.
        for shard in self._shards.values():
            shard.close()
        self._shards.clear()
        self._connected.set()
        log.info("registered as %s (%d cores%s)", self.worker_id, self.cores, f", GPU {self._gpu}" if self._gpu else "")

    def _disconnect(self) -> None:
        self._connected.clear()
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _send(self, header: dict[str, Any], payload: bytes = b"") -> None:
        sock = self._sock
        if sock is None:
            raise ConnectionError("not connected")
        with self._send_lock:
            send_message(sock, header, payload)

    # --------------------------------------------------------------- threads
    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            msg = recv_message(self._sock)  # type: ignore[arg-type]
            if msg.type == MsgType.TASK:
                self._tasks.put(msg)
            elif msg.type == MsgType.SHUTDOWN:
                log.info("coordinator requested shutdown")
                self._stop.set()
                return

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            if self._connected.wait(0.5):
                mem = psutil.virtual_memory()
                try:
                    self._send(
                        {
                            "type": MsgType.HEARTBEAT,
                            "cpu": psutil.cpu_percent(None),
                            "per_cpu": psutil.cpu_percent(None, percpu=True),
                            "ram": mem.percent,
                            "ram_used": mem.used,
                            "running": self._running_tasks,
                            "completed": self._completed,
                            "shards": len(self._shards),
                        }
                    )
                except (ConnectionError, OSError):
                    pass
                self._stop.wait(self.heartbeat_interval)

    def _executor_loop(self) -> None:
        while True:
            msg = self._tasks.get()
            if msg is None:
                return
            task_id = msg.header["task_id"]
            kind = msg.header.get("kind")
            started = time.perf_counter()
            self._running_tasks += 1
            try:
                data, payload = self.handle(kind, msg.header.get("params", {}), msg.payload)
                header = {"type": MsgType.RESULT, "task_id": task_id, "ok": True, "data": data}
            except Exception as exc:  # report the failure, keep serving
                log.exception("task %s (%s) failed", task_id, kind)
                header = {"type": MsgType.RESULT, "task_id": task_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                payload = b""
            finally:
                self._running_tasks -= 1
            elapsed = time.perf_counter() - started
            if self.slowdown > 1.0:  # simulate a slower PC for load-balancing demos
                time.sleep(elapsed * (self.slowdown - 1.0))
                elapsed = time.perf_counter() - started
            header["elapsed"] = elapsed
            self._completed += 1
            try:
                self._send(header, payload)
            except (ConnectionError, OSError):
                log.warning("could not deliver result for task %s", task_id)

    # ---------------------------------------------------------------- tasks
    def handle(self, kind: str, params: dict[str, Any], payload: bytes) -> tuple[dict[str, Any], bytes]:
        if kind == "compute":
            return self._task_compute(params), b""
        if kind == "load_shard":
            return self._task_load_shard(params, payload), b""
        if kind == "gradient":
            return self._task_gradient(params, payload)
        if kind == "local_train":
            return self._task_local_train(params, payload)
        if kind == "release":
            return self._task_release(params), b""
        if kind == "ping":
            return {"pong": True}, b""
        raise ValueError(f"unknown task kind: {kind}")

    def _task_compute(self, p: dict[str, Any]) -> dict[str, Any]:
        kernel, start, end = p["kernel"], int(p["start"]), int(p["end"])
        if kernel not in compute.KERNELS:
            raise ValueError(f"unknown kernel {kernel}")
        pool = self._get_pool()
        if pool is None or end - start < 2 * self.cores:
            return {"parts": [compute.KERNELS[kernel](start, end)], "processes": 1}
        bounds = np.linspace(start, end, self.cores + 1, dtype=np.int64)
        jobs = [(kernel, int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        return {"parts": pool.map(_kernel_chunk, jobs), "processes": len(jobs)}

    def _task_load_shard(self, p: dict[str, Any], payload: bytes) -> dict[str, Any]:
        arrays = unpack_arrays(payload)
        key = (p["job_id"], int(p["shard_id"]))
        if key in self._shards:
            self._shards.pop(key).close()
        self._shards[key] = Shard(arrays["X"], arrays["y"], shared=self.cores > 1)
        return {"rows": int(arrays["X"].shape[0]), "bytes": len(payload)}

    def _shard(self, p: dict[str, Any]) -> Shard:
        key = (p["job_id"], int(p["shard_id"]))
        if key not in self._shards:
            raise KeyError(f"shard {key} not loaded on this worker")
        return self._shards[key]

    def _task_gradient(self, p: dict[str, Any], payload: bytes) -> tuple[dict[str, Any], bytes]:
        shard = self._shard(p)
        W = unpack_arrays(payload)["W"]
        model = p["model"]
        pool = self._get_pool()
        if pool is not None and shard.xspec is not None and shard.X.shape[0] >= 2 * self.cores:
            def map_fn(pieces):
                return pool.map(_grad_chunk, [(model, shard.xspec, shard.yspec, W, a, b) for a, b in pieces])

            loss, grad, n = ml.chunked_loss_and_grad(model, shard.X, shard.y, W, self.cores, map_fn)
        else:
            loss, grad, n = ml.loss_and_grad_sum(model, shard.X, shard.y, W)
        return {"loss": loss, "n": n}, pack_arrays(grad=grad)

    def _task_local_train(self, p: dict[str, Any], payload: bytes) -> tuple[dict[str, Any], bytes]:
        shard = self._shard(p)
        W = unpack_arrays(payload)["W"]
        W_new, loss = ml.local_sgd(
            p["model"],
            shard.X,
            shard.y,
            W,
            lr=float(p["lr"]),
            batch_size=int(p["batch_size"]),
            local_epochs=int(p.get("local_epochs", 1)),
            l2=float(p.get("l2", 0.0)),
            seed=int(p.get("seed", 0)),
        )
        return {"loss": loss, "n": int(shard.X.shape[0])}, pack_arrays(W=W_new)

    def _task_release(self, p: dict[str, Any]) -> dict[str, Any]:
        released = [k for k in self._shards if k[0] == p["job_id"]]
        for key in released:
            self._shards.pop(key).close()
        if released and self._pool is not None:
            self._reset_pool()  # pool processes may still map the freed shared memory
        return {"released": len(released)}
