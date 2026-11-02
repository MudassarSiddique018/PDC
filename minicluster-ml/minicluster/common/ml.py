"""Tiny NumPy machine-learning engine used by both workers and coordinator.

All models are linear models with a weight matrix ``W`` of shape
``(n_features + 1, n_outputs)`` (the extra row is the bias - callers add a
column of ones with :func:`add_bias`).

Workers return *sums* (not means) of loss and gradient so the coordinator can
aggregate them exactly, independent of how unevenly the data was split:

    global_grad = sum(worker_grad_sums) / sum(worker_counts)

That makes synchronous distributed gradient descent mathematically identical
to single-machine full-batch gradient descent.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

LINEAR = "linear_regression"
LOGISTIC = "logistic_regression"
SOFTMAX = "softmax_regression"

MODELS = {
    LINEAR: {"task": "regression", "label": "Linear Regression"},
    LOGISTIC: {"task": "binary", "label": "Logistic Regression"},
    SOFTMAX: {"task": "multiclass", "label": "Softmax Regression"},
}

_EPS = 1e-12


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])


def n_outputs(model: str, n_classes: int) -> int:
    return n_classes if model == SOFTMAX else 1


def init_weights(n_features_with_bias: int, outputs: int) -> np.ndarray:
    return np.zeros((n_features_with_bias, outputs), dtype=np.float64)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(model: str, X: np.ndarray, W: np.ndarray) -> np.ndarray:
    z = X @ W
    if model == LOGISTIC:
        return _sigmoid(z)
    if model == SOFTMAX:
        return _softmax(z)
    return z


def loss_and_grad_sum(model: str, X: np.ndarray, y: np.ndarray, W: np.ndarray) -> tuple[float, np.ndarray, int]:
    """Return ``(loss_sum, grad_sum, n_samples)`` for one block of data."""
    n = X.shape[0]
    if n == 0:
        return 0.0, np.zeros_like(W), 0
    out = forward(model, X, W)
    if model == LINEAR:
        target = y.reshape(-1, 1)
        err = out - target
        loss = 0.5 * float(np.sum(err**2))
    elif model == LOGISTIC:
        target = y.reshape(-1, 1).astype(np.float64)
        err = out - target
        loss = -float(np.sum(target * np.log(out + _EPS) + (1 - target) * np.log(1 - out + _EPS)))
    elif model == SOFTMAX:
        onehot = np.zeros_like(out)
        onehot[np.arange(n), y.astype(np.int64)] = 1.0
        err = out - onehot
        loss = -float(np.sum(onehot * np.log(out + _EPS)))
    else:
        raise ValueError(f"unknown model: {model}")
    return loss, X.T @ err, n


def predict(model: str, X: np.ndarray, W: np.ndarray) -> np.ndarray:
    out = forward(model, X, W)
    if model == LOGISTIC:
        return (out[:, 0] >= 0.5).astype(np.int64)
    if model == SOFTMAX:
        return out.argmax(axis=1)
    return out[:, 0]


def evaluate(model: str, X: np.ndarray, y: np.ndarray, W: np.ndarray) -> dict[str, float]:
    loss, _, n = loss_and_grad_sum(model, X, y, W)
    metrics = {"loss": loss / max(n, 1)}
    pred = predict(model, X, W)
    if model == LINEAR:
        mse = float(np.mean((pred - y) ** 2))
        var = float(np.var(y)) or 1.0
        metrics.update(rmse=mse**0.5, r2=1.0 - mse / var)
    else:
        metrics["accuracy"] = float(np.mean(pred == y))
    return metrics


def local_sgd(
    model: str,
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    lr: float,
    batch_size: int,
    local_epochs: int,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Mini-batch SGD on a single shard (used by the model-averaging strategy)."""
    rng = np.random.default_rng(seed)
    W = W.copy()
    n = X.shape[0]
    loss_total = 0.0
    for _ in range(local_epochs):
        order = rng.permutation(n)
        loss_total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            loss, grad, m = loss_and_grad_sum(model, X[idx], y[idx], W)
            loss_total += loss
            W -= lr * (grad / m + l2 * W)
    return W, loss_total


def chunked_loss_and_grad(
    model: str,
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    n_chunks: int,
    map_fn: Callable | None = None,
) -> tuple[float, np.ndarray, int]:
    """Split a shard into ``n_chunks`` and reduce the partial sums.

    ``map_fn`` lets the worker plug in a multiprocessing pool; by default the
    chunks are processed sequentially (useful for tests).
    """
    bounds = np.linspace(0, X.shape[0], n_chunks + 1, dtype=np.int64)
    pieces = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    if map_fn is None:
        parts = [loss_and_grad_sum(model, X[a:b], y[a:b], W) for a, b in pieces]
    else:
        parts = map_fn(pieces)
    loss = sum(p[0] for p in parts)
    grad = sum((p[1] for p in parts), np.zeros_like(W))
    n = sum(p[2] for p in parts)
    return loss, grad, n
