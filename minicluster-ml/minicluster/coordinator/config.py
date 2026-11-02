"""Coordinator settings, read from ``MINICLUSTER_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(f"MINICLUSTER_{name}", default)


@dataclass
class Settings:
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    http_port: int = field(default_factory=lambda: int(_env("HTTP_PORT", "8000")))
    worker_port: int = field(default_factory=lambda: int(_env("WORKER_PORT", "9000")))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")).resolve())
    api_token: str = field(default_factory=lambda: _env("API_TOKEN", ""))
    cluster_secret: str = field(default_factory=lambda: _env("CLUSTER_SECRET", ""))
    heartbeat_timeout: float = field(default_factory=lambda: float(_env("HEARTBEAT_TIMEOUT", "6")))
    no_worker_timeout: float = field(default_factory=lambda: float(_env("NO_WORKER_TIMEOUT", "30")))
    max_concurrent_jobs: int = field(default_factory=lambda: int(_env("MAX_CONCURRENT_JOBS", "1")))
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "200")))
    max_dataset_rows: int = field(default_factory=lambda: int(_env("MAX_DATASET_ROWS", "5000000")))
    cors_origins: list[str] = field(
        default_factory=lambda: [o.strip() for o in _env("CORS_ORIGINS", "").split(",") if o.strip()]
    )
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "minicluster.db"

    def ensure_dirs(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
