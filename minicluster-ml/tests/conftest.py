import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveCluster:
    """A real coordinator + worker processes on random local ports."""

    def __init__(self, tmp: Path, token: str = ""):
        self.http, self.tcp = free_port(), free_port()
        self.base = f"http://127.0.0.1:{self.http}"
        self.token = token
        self.env = {**os.environ, "MINICLUSTER_DATA_DIR": str(tmp), "PYTHONPATH": str(ROOT),
                    "MINICLUSTER_API_TOKEN": token, "MINICLUSTER_HEARTBEAT_TIMEOUT": "4",
                    "MINICLUSTER_NO_WORKER_TIMEOUT": "5"}
        self.workers: dict[str, subprocess.Popen] = {}
        self.coord = subprocess.Popen(
            [sys.executable, "-m", "minicluster.coordinator", "--http-port", str(self.http),
             "--worker-port", str(self.tcp), "--log-level", "WARNING"], cwd=ROOT, env=self.env)
        self._wait(lambda: self.get("/healthz")["status"] == "ok")

    def _wait(self, cond, timeout=20.0):
        end = time.time() + timeout
        while time.time() < end:
            try:
                if cond():
                    return
            except (OSError, urllib.error.URLError, KeyError):
                pass
            time.sleep(0.2)
        raise TimeoutError("condition not met")

    def request(self, path, body=None, method=None):
        req = urllib.request.Request(self.base + path, method=method or ("POST" if body is not None else "GET"))
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        data = None
        if body is not None:
            req.add_header("Content-Type", "application/json")
            data = json.dumps(body).encode()
        with urllib.request.urlopen(req, data, timeout=30) as r:
            return json.loads(r.read())

    def get(self, path):
        return self.request(path)

    def start_worker(self, name: str, cores: int = 1, slowdown: float = 1.0):
        self.workers[name] = subprocess.Popen(
            [sys.executable, "-m", "minicluster.worker", "--port", str(self.tcp), "--name", name, "--cores", str(cores),
             "--slowdown", str(slowdown), "--log-level", "WARNING"], cwd=ROOT, env=self.env)

    def wait_workers(self, n: int):
        self._wait(lambda: self.get("/api/cluster")["online"] == n)

    def kill_worker(self, name: str):
        self.workers.pop(name).kill()

    def submit(self, spec, name=""):
        return self.request("/api/jobs", {"name": name, "spec": spec})["id"]

    def wait_job(self, job_id, timeout=120.0):
        end = time.time() + timeout
        while time.time() < end:
            job = self.get(f"/api/jobs/{job_id}")
            if job["status"] in ("completed", "failed", "cancelled"):
                return job
            time.sleep(0.25)
        raise TimeoutError(job_id)

    def close(self):
        for p in [*self.workers.values(), self.coord]:
            p.terminate()
        for p in [*self.workers.values(), self.coord]:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()


@pytest.fixture
def cluster(tmp_path):
    c = LiveCluster(tmp_path)
    yield c
    c.close()
