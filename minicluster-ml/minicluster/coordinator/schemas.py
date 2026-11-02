"""Request/response schemas (validated by FastAPI / pydantic)."""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from minicluster.common import ml


class ComputeSpec(BaseModel):
    type: Literal["compute"] = "compute"
    kernel: Literal["square_sum", "prime_count", "monte_carlo_pi"] = "square_sum"
    n: int = Field(10_000, ge=1, le=2_000_000_000, description="How many numbers / darts to process")
    workers: int = Field(3, ge=1, le=256)
    chunks_per_worker: int = Field(4, ge=1, le=100, description="More chunks = finer load balancing")


class TrainingSpec(BaseModel):
    type: Literal["training"] = "training"
    dataset: str = "student_performance"
    rows: int = Field(100_000, ge=100, description="Rows to generate (built-in dataset only)")
    target: str = "passed"
    model: Literal["linear_regression", "logistic_regression", "softmax_regression"] = "logistic_regression"
    strategy: Literal["gradient_sync", "model_averaging"] = "gradient_sync"
    workers: int = Field(3, ge=1, le=256)
    epochs: int = Field(30, ge=1, le=5000)
    lr: float = Field(0.5, gt=0, le=100)
    momentum: float = Field(0.9, ge=0, lt=1)
    batch_size: int = Field(256, ge=1, le=1_000_000)
    l2: float = Field(0.0, ge=0, le=10)
    test_fraction: float = Field(0.2, gt=0, lt=0.9)
    seed: int = 42

    @model_validator(mode="after")
    def _model_matches_target(self):
        if self.dataset == "student_performance":
            from minicluster.common.datasets import STUDENT_TARGETS

            if self.target not in STUDENT_TARGETS:
                raise ValueError(f"target must be one of {list(STUDENT_TARGETS)}")
            task = STUDENT_TARGETS[self.target]
            need = ml.MODELS[self.model]["task"]
            if not (need == task or (need == "multiclass" and task == "binary")):
                raise ValueError(f"{ml.MODELS[self.model]['label']} cannot be trained on target '{self.target}' ({task})")
        return self


class BenchmarkSpec(BaseModel):
    type: Literal["benchmark"] = "benchmark"
    base: Union[ComputeSpec, TrainingSpec] = Field(discriminator="type")
    max_workers: Optional[int] = Field(None, ge=1, le=256, description="Default: every online worker")


class JobRequest(BaseModel):
    name: str = Field("", max_length=120)
    spec: Union[ComputeSpec, TrainingSpec, BenchmarkSpec] = Field(discriminator="type")


class PredictRequest(BaseModel):
    rows: list[dict[str, float]] = Field(min_length=1, max_length=10_000)
