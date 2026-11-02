"""Built-in datasets and CSV loading.

The flagship dataset is a synthetic *Student Performance* table modelled on
the lab scenario: a university wants to predict whether a student will pass
from their academic history. It is generated deterministically from a seed,
so every PC in the lab gets exactly the same data for a given size.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STUDENT_FEATURES = [
    "matric_marks",
    "inter_marks",
    "entry_test",
    "attendance",
    "prev_gpa",
    "study_hours",
    "assignments_done",
]
STUDENT_TARGETS = {
    "final_score": "regression",
    "passed": "binary",
    "category": "multiclass",
}
CATEGORY_NAMES = ["Fail", "Pass", "Distinction"]


@dataclass
class Dataset:
    name: str
    feature_names: list[str]
    X: np.ndarray
    y: np.ndarray
    target: str
    task: str
    class_names: list[str] | None = None

    @property
    def n_classes(self) -> int:
        return len(self.class_names) if self.class_names else 0


def generate_student_performance(n_rows: int, seed: int = 42) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return ``(features, targets)`` for ``n_rows`` synthetic students."""
    rng = np.random.default_rng(seed)
    ability = rng.normal(0.0, 1.0, n_rows)  # hidden "talent" variable
    effort = rng.normal(0.0, 1.0, n_rows)

    def clip(a, lo, hi):
        return np.clip(a, lo, hi)

    matric = clip(70 + 10 * ability + rng.normal(0, 6, n_rows), 33, 100)
    inter = clip(66 + 11 * ability + 3 * effort + rng.normal(0, 6, n_rows), 33, 100)
    entry = clip(55 + 14 * ability + rng.normal(0, 9, n_rows), 0, 100)
    attendance = clip(80 + 8 * effort + rng.normal(0, 7, n_rows), 30, 100)
    prev_gpa = clip(2.8 + 0.45 * ability + 0.25 * effort + rng.normal(0, 0.3, n_rows), 0, 4)
    study_hours = clip(14 + 6 * effort + rng.normal(0, 4, n_rows), 0, 45)
    assignments = clip(78 + 10 * effort + rng.normal(0, 8, n_rows), 0, 100)

    final = (
        0.10 * matric
        + 0.12 * inter
        + 0.15 * entry
        + 0.18 * attendance
        + 9.0 * prev_gpa
        + 0.55 * study_hours
        + 0.10 * assignments
        - 20.0
        + rng.normal(0, 4.5, n_rows)
    )
    final = clip(final, 0, 100)

    X = np.column_stack([matric, inter, entry, attendance, prev_gpa, study_hours, assignments]).astype(np.float64)
    category = np.digitize(final, [50, 75]).astype(np.int64)
    targets = {
        "final_score": final.astype(np.float64),
        "passed": (final >= 50).astype(np.int64),
        "category": category,
    }
    return X, targets


def student_dataset(n_rows: int, target: str = "passed", seed: int = 42) -> Dataset:
    if target not in STUDENT_TARGETS:
        raise ValueError(f"target must be one of {list(STUDENT_TARGETS)}")
    X, targets = generate_student_performance(n_rows, seed)
    task = STUDENT_TARGETS[target]
    classes = {"binary": ["Fail", "Pass"], "multiclass": CATEGORY_NAMES}.get(task)
    return Dataset("student_performance", list(STUDENT_FEATURES), X, targets[target], target, task, classes)


def write_student_csv(path: Path, n_rows: int, seed: int = 42) -> None:
    X, targets = generate_student_performance(n_rows, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(STUDENT_FEATURES + list(STUDENT_TARGETS))
        for i in range(n_rows):
            row = [round(float(v), 3) for v in X[i]]
            row += [round(float(targets["final_score"][i]), 2), int(targets["passed"][i]), int(targets["category"][i])]
            writer.writerow(row)


def csv_columns(path: Path) -> list[str]:
    with path.open(newline="") as fh:
        return next(csv.reader(fh))


def load_csv(path: Path, target: str, task: str | None = None, max_classes: int = 20) -> Dataset:
    """Load a numeric CSV. Non-numeric target columns are label-encoded."""
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if r and any(c.strip() for c in r)]
    if target not in header:
        raise ValueError(f"target column '{target}' not found; columns are {header}")
    t_idx = header.index(target)
    feature_names = [h for i, h in enumerate(header) if i != t_idx]

    try:
        X = np.array([[float(c) for i, c in enumerate(r) if i != t_idx] for r in rows], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"all feature columns must be numeric: {exc}") from exc

    raw_y = [r[t_idx].strip() for r in rows]
    class_names = None
    try:
        y_num = np.array([float(v) for v in raw_y])
        numeric = True
    except ValueError:
        numeric = False

    if task is None:
        if not numeric:
            task = "multiclass"
        else:
            uniq = np.unique(y_num)
            is_int = np.all(np.equal(np.mod(uniq, 1), 0))
            task = "regression" if (not is_int or len(uniq) > max_classes) else ("binary" if len(uniq) == 2 else "multiclass")

    if task == "regression":
        if not numeric:
            raise ValueError("regression target must be numeric")
        y = y_num.astype(np.float64)
    else:
        labels = sorted(set(raw_y), key=lambda v: (float(v) if numeric else 0, v))
        class_names = labels
        lookup = {v: i for i, v in enumerate(labels)}
        y = np.array([lookup[v] for v in raw_y], dtype=np.int64)
        if task == "binary" and len(labels) != 2:
            task = "multiclass"
    return Dataset(path.stem, feature_names, X, y, target, task, class_names)
