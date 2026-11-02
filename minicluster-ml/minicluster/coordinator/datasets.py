"""Dataset catalogue: the built-in generator plus CSV files uploaded via the API."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from minicluster.common import datasets as ds
from minicluster.coordinator.config import Settings

BUILTIN = "student_performance"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


class DatasetRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings

    def safe_name(self, filename: str) -> str:
        stem = _SAFE_NAME.sub("_", Path(filename).name).strip("._") or "dataset"
        return stem if stem.lower().endswith(".csv") else f"{stem}.csv"

    def upload_path(self, name: str) -> Path:
        path = (self.settings.uploads_dir / self.safe_name(name)).resolve()
        if path.parent != self.settings.uploads_dir.resolve():
            raise ValueError("invalid dataset name")
        return path

    def list(self) -> list[dict[str, Any]]:
        items = [{
            "id": BUILTIN,
            "name": "Student Performance (synthetic)",
            "builtin": True,
            "columns": ds.STUDENT_FEATURES + list(ds.STUDENT_TARGETS),
            "features": ds.STUDENT_FEATURES,
            "targets": ds.STUDENT_TARGETS,
            "description": "University scenario: predict pass/fail, grade category or final score from academic history. "
                           "Any size from 1,000 to millions of rows, generated identically on demand.",
        }]
        for path in sorted(self.settings.uploads_dir.glob("*.csv")):
            try:
                columns = ds.csv_columns(path)
            except Exception:
                continue
            items.append({
                "id": f"upload:{path.name}",
                "name": path.name,
                "builtin": False,
                "columns": columns,
                "size_bytes": path.stat().st_size,
            })
        return items

    def load(self, spec: dict[str, Any]) -> ds.Dataset:
        dataset_id = spec.get("dataset", BUILTIN)
        if dataset_id == BUILTIN:
            rows = int(spec.get("rows", 100_000))
            if not 100 <= rows <= self.settings.max_dataset_rows:
                raise ValueError(f"rows must be between 100 and {self.settings.max_dataset_rows:,}")
            return ds.student_dataset(rows, spec.get("target", "passed"), int(spec.get("seed", 42)))
        if dataset_id.startswith("upload:"):
            path = self.upload_path(dataset_id.split(":", 1)[1])
            if not path.exists():
                raise ValueError(f"dataset {dataset_id} not found")
            target = spec.get("target")
            if not target:
                raise ValueError("choose a target column for an uploaded dataset")
            return ds.load_csv(path, target)
        raise ValueError(f"unknown dataset {dataset_id}")
