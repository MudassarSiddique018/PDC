"""FastAPI application: REST API, live WebSocket feed and the web dashboard."""

from __future__ import annotations

import asyncio
import hmac
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from minicluster import __version__
from minicluster.common import compute, ml
from minicluster.common.datasets import csv_columns
from minicluster.coordinator.cluster import ClusterManager
from minicluster.coordinator.config import Settings
from minicluster.coordinator.datasets import DatasetRegistry
from minicluster.coordinator.events import EventBus
from minicluster.coordinator.jobs.base import COMPLETED
from minicluster.coordinator.jobs.manager import JobManager
from minicluster.coordinator.jobs.training import predict
from minicluster.coordinator.schemas import JobRequest, PredictRequest
from minicluster.coordinator.store import JobStore

log = logging.getLogger("minicluster.api")
WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_dirs()
    bus = EventBus()
    cluster = ClusterManager(settings, bus)
    store = JobStore(settings.db_path)
    datasets = DatasetRegistry(settings)
    jobs = JobManager(settings, cluster, bus, store, datasets)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await cluster.start()
        await jobs.start()
        bus.log(f"MiniCluster ML v{__version__} coordinator ready - dashboard on port {settings.http_port}")
        yield
        await jobs.stop()
        await cluster.stop()
        store.close()

    app = FastAPI(
        title="MiniCluster ML",
        version=__version__,
        description="Web-based parallel & distributed machine-learning platform for lab PCs.",
        lifespan=lifespan,
    )
    app.state.settings, app.state.cluster, app.state.jobs, app.state.bus = settings, cluster, jobs, bus
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])

    # ------------------------------------------------------------- auth
    def _token_ok(token: str | None) -> bool:
        return not settings.api_token or (token is not None and hmac.compare_digest(token, settings.api_token))

    def require_token(request: Request) -> None:
        auth = request.headers.get("authorization", "")
        token = auth[7:] if auth.lower().startswith("bearer ") else request.headers.get("x-api-token")
        if not _token_ok(token):
            raise HTTPException(401, "missing or invalid API token", headers={"WWW-Authenticate": "Bearer"})

    auth = [Depends(require_token)]

    def get_job(job_id: str):
        job = jobs.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"job {job_id} not found")
        return job

    # ----------------------------------------------------------- system
    @app.get("/healthz", tags=["system"])
    def healthz():
        return {"status": "ok", "version": __version__, "workers_online": len(cluster.online_workers())}

    @app.get("/api/info", tags=["system"])
    def info():
        return {
            "version": __version__,
            "auth_required": bool(settings.api_token),
            "worker_port": settings.worker_port,
            "kernels": compute.KERNEL_LABELS,
            "models": ml.MODELS,
        }

    @app.get("/metrics", tags=["system"], response_class=PlainTextResponse, dependencies=auth)
    def metrics():
        snap = cluster.snapshot()
        lines = [
            "# HELP minicluster_workers_online Workers currently connected",
            "# TYPE minicluster_workers_online gauge",
            f"minicluster_workers_online {snap['online']}",
            "# TYPE minicluster_cluster_cores gauge",
            f"minicluster_cluster_cores {snap['cores']}",
            "# TYPE minicluster_tasks_dispatched_total counter",
            f"minicluster_tasks_dispatched_total {snap['tasks_dispatched']}",
            "# TYPE minicluster_jobs gauge",
        ]
        lines += [f'minicluster_jobs{{status="{k}"}} {v}' for k, v in jobs.stats().items() if k != "total"]
        lines += ["# TYPE minicluster_worker_cpu_percent gauge", "# TYPE minicluster_worker_ram_percent gauge",
                  "# TYPE minicluster_worker_tasks_done_total counter"]
        for w in snap["workers"]:
            lbl = f'worker="{w["id"]}",status="{w["status"]}"'
            lines += [f"minicluster_worker_cpu_percent{{{lbl}}} {w['cpu']}",
                      f"minicluster_worker_ram_percent{{{lbl}}} {w['ram']}",
                      f"minicluster_worker_tasks_done_total{{{lbl}}} {w['tasks_done']}"]
        return "\n".join(lines) + "\n"

    # ---------------------------------------------------------- cluster
    @app.get("/api/cluster", tags=["cluster"], dependencies=auth)
    def get_cluster():
        return {**cluster.snapshot(), "jobs": jobs.stats()}

    @app.delete("/api/cluster/workers/{worker_id}", tags=["cluster"], dependencies=auth)
    def forget_worker(worker_id: str):
        if not cluster.forget(worker_id):
            raise HTTPException(409, "only offline workers can be removed")
        return {"removed": worker_id}

    @app.get("/api/events", tags=["cluster"], dependencies=auth)
    def get_events(limit: int = Query(100, ge=1, le=300)):
        return list(bus.recent_logs)[-limit:]

    # ------------------------------------------------------------- jobs
    @app.get("/api/jobs", tags=["jobs"], dependencies=auth)
    def list_jobs(limit: int = Query(100, ge=1, le=500)):
        return jobs.list(limit)

    @app.post("/api/jobs", tags=["jobs"], status_code=201, dependencies=auth)
    def submit_job(req: JobRequest):
        spec = req.spec.model_dump()
        job_type = spec.pop("type")
        if job_type == "benchmark":
            spec["base"] = req.spec.base.model_dump()
        return jobs.submit(job_type, spec, req.name).detail()

    @app.get("/api/jobs/{job_id}", tags=["jobs"], dependencies=auth)
    def job_detail(job_id: str):
        return get_job(job_id).detail()

    @app.post("/api/jobs/{job_id}/cancel", tags=["jobs"], dependencies=auth)
    def cancel_job(job_id: str):
        get_job(job_id)
        return jobs.cancel(job_id).summary()

    @app.delete("/api/jobs/{job_id}", tags=["jobs"], dependencies=auth)
    def delete_job(job_id: str):
        get_job(job_id)
        try:
            jobs.delete(job_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"deleted": job_id}

    @app.post("/api/jobs/{job_id}/predict", tags=["jobs"], dependencies=auth)
    def predict_job(job_id: str, req: PredictRequest):
        job = get_job(job_id)
        if job.type != "training" or job.status != COMPLETED or not job.result:
            raise HTTPException(409, "predictions need a completed training job")
        try:
            return {"predictions": predict(job.result["model"], req.rows)}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    # --------------------------------------------------------- datasets
    @app.get("/api/datasets", tags=["datasets"], dependencies=auth)
    def list_datasets():
        return datasets.list()

    @app.post("/api/datasets", tags=["datasets"], status_code=201, dependencies=auth)
    async def upload_dataset(file: UploadFile = File(...)):
        if not (file.filename or "").lower().endswith(".csv"):
            raise HTTPException(415, "only .csv files are supported")
        dest = datasets.upload_path(file.filename or "dataset.csv")
        limit = settings.max_upload_mb * 1024 * 1024
        size = 0
        tmp = dest.with_suffix(".part")
        try:
            with tmp.open("wb") as fh:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"file larger than {settings.max_upload_mb} MB")
                    fh.write(chunk)
            columns = csv_columns(tmp)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        bus.log(f"Dataset uploaded: {dest.name} ({size / 1e6:.1f} MB, {len(columns)} columns)")
        return {"id": f"upload:{dest.name}", "name": dest.name, "columns": columns, "size_bytes": size}

    # -------------------------------------------------------- websocket
    @app.websocket("/ws")
    async def ws_feed(ws: WebSocket, token: str | None = None):
        if not _token_ok(token):
            await ws.close(code=4401)
            return
        await ws.accept()
        q = bus.subscribe()
        try:
            await ws.send_json({"type": "hello", "ts": time.time(), "data": {
                "cluster": {**cluster.snapshot(), "jobs": jobs.stats()},
                "jobs": jobs.list(50),
                "logs": list(bus.recent_logs)[-100:],
            }})
            while True:
                event = await q.get()
                if event["type"] == "cluster":
                    event["data"] = {**event["data"], "jobs": jobs.stats()}
                await ws.send_json(event)
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            bus.unsubscribe(q)

    # -------------------------------------------------------- dashboard
    if WEB_DIR.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(WEB_DIR / "index.html")

    return app
