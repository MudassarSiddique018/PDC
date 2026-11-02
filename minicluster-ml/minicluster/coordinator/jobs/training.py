"""Level 2 - distributed machine-learning training (data parallelism).

Two synchronisation strategies are offered so students can compare them:

``gradient_sync`` (parameter server / synchronous data-parallel GD)
    Every epoch the coordinator broadcasts the current weights, each worker
    computes the loss & gradient *sums* on its shard, the coordinator adds
    them up and takes one optimisation step. The result is exactly the same
    model a single PC would have produced on the whole dataset.

``model_averaging`` (local SGD / FedAvg-style)
    Each worker runs a full epoch of mini-batch SGD on its own shard, then
    the coordinator averages the resulting weights (weighted by shard size).
    Far less communication, but only an approximation of centralised SGD.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from minicluster.common import ml
from minicluster.common.datasets import Dataset
from minicluster.common.protocol import pack_arrays, unpack_arrays
from minicluster.coordinator.jobs.base import Dispatcher, JobContext, call

GRADIENT_SYNC = "gradient_sync"
MODEL_AVERAGING = "model_averaging"


def prepare(ds: Dataset, test_fraction: float, seed: int):
    """Shuffle, split and standardise. Returns train/test arrays and the scaler."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(ds.X.shape[0])
    X, y = ds.X[order], ds.y[order]
    n_test = int(round(len(y) * test_fraction))
    X_tr, y_tr, X_te, y_te = X[n_test:], y[n_test:], X[:n_test], y[:n_test]
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std == 0] = 1.0
    X_tr = ml.add_bias((X_tr - mean) / std)
    X_te = ml.add_bias((X_te - mean) / std)
    return X_tr, y_tr, X_te, y_te, mean, std


async def run_training(ctx: JobContext, spec: dict[str, Any], workers: int | None = None) -> dict[str, Any]:
    job = ctx.job
    model = spec["model"]
    strategy = spec.get("strategy", GRADIENT_SYNC)
    epochs = int(spec["epochs"])
    lr = float(spec["lr"])
    l2 = float(spec.get("l2", 0.0))
    momentum = float(spec.get("momentum", 0.0))
    batch_size = int(spec.get("batch_size", 256))
    seed = int(spec.get("seed", 42))
    want = workers or int(spec.get("workers", 3))
    t0 = time.perf_counter()

    # ---------------------------------------------------------------- data
    ctx.update(phase="Loading dataset", force=True)
    ds: Dataset = await asyncio.to_thread(ctx.datasets.load, spec)
    expected = ml.MODELS[model]["task"]
    if expected != ds.task and not (expected == "multiclass" and ds.task == "binary"):
        raise ValueError(f"{ml.MODELS[model]['label']} needs a {expected} target, but '{ds.target}' is {ds.task}")
    X_tr, y_tr, X_te, y_te, mean, std = await asyncio.to_thread(prepare, ds, float(spec.get("test_fraction", 0.2)), seed)
    k = ml.n_outputs(model, max(ds.n_classes, 2))
    W = ml.init_weights(X_tr.shape[1], k)
    velocity = np.zeros_like(W)
    ctx.log(f"Dataset '{ds.name}' ready: {X_tr.shape[0]:,} train / {X_te.shape[0]:,} test rows, "
            f"{len(ds.feature_names)} features, target '{ds.target}' ({ds.task})")

    dispatcher = Dispatcher(ctx, want)
    pool = await dispatcher.ensure_pool()
    n_shards = len(pool)
    bounds = np.linspace(0, X_tr.shape[0], n_shards + 1, dtype=np.int64)
    shards = {i: (int(bounds[i]), int(bounds[i + 1])) for i in range(n_shards)}
    for i, w in enumerate(pool):
        dispatcher.home[i] = w.id
    loaded: set[tuple[str, int]] = set()  # (worker session, shard id)

    async def ensure_loaded(w, shard_id):
        if (w.session, shard_id) in loaded:
            return 0.0
        a, b = shards[shard_id]
        payload = await asyncio.to_thread(pack_arrays, X=X_tr[a:b], y=y_tr[a:b])
        t = time.perf_counter()
        await call(ctx, w, "load_shard", {"job_id": job.id, "shard_id": shard_id}, payload)
        loaded.add((w.session, shard_id))
        stat = ctx.worker_stat(w.id)
        stat["items"] = stat.get("items", 0) + (b - a)
        stat["shard"] = "; ".join(filter(None, [stat.get("shard"), f"rows {a + 1:,}-{b:,}"]))
        ctx.log(f"{w.id} <- shard #{shard_id} (rows {a + 1:,}-{b:,}, {len(payload) / 1e6:.1f} MB)", worker=w.id)
        return time.perf_counter() - t

    def describe(i):
        return f"shard #{i}"

    # ----------------------------------------------------- scatter shards
    ctx.update(phase=f"Scatter: sending {n_shards} data shard(s) to workers", force=True)
    t_scatter = time.perf_counter()
    try:
        await dispatcher.map(list(shards), ensure_loaded, describe)
        scatter_seconds = time.perf_counter() - t_scatter

        # -------------------------------------------------------- epochs
        compute_seconds = 0.0
        comm_seconds = 0.0
        for epoch in range(1, epochs + 1):
            job.check_cancelled()
            ctx.update(phase=f"Epoch {epoch}/{epochs} - {strategy.replace('_', ' ')}")
            W_payload = pack_arrays(W=W)
            round_start = time.perf_counter()
            elapsed_per_worker: list[float] = []

            if strategy == GRADIENT_SYNC:
                async def step(w, shard_id):
                    await ensure_loaded(w, shard_id)
                    params = {"job_id": job.id, "shard_id": shard_id, "model": model}
                    data, out, el = await call(ctx, w, "gradient", params, W_payload)
                    elapsed_per_worker.append(el)
                    return data["loss"], unpack_arrays(out)["grad"], data["n"]
            else:
                async def step(w, shard_id):
                    await ensure_loaded(w, shard_id)
                    params = {"job_id": job.id, "shard_id": shard_id, "model": model, "lr": lr, "batch_size": batch_size,
                              "l2": l2, "local_epochs": 1, "seed": seed * 1000 + epoch * 10 + shard_id}
                    data, out, el = await call(ctx, w, "local_train", params, W_payload)
                    elapsed_per_worker.append(el)
                    return data["loss"], unpack_arrays(out)["W"], data["n"]

            def on_done(shard_id, w, res):
                ctx.worker_stat(w.id)["progress"] = epoch / epochs
                ctx.worker_stat(w.id)["epochs"] = epoch

            results = await dispatcher.map(list(shards), step, describe, on_done)
            round_seconds = time.perf_counter() - round_start
            slowest = max(elapsed_per_worker, default=0.0)
            compute_seconds += slowest
            comm_seconds += max(0.0, round_seconds - slowest)

            # -------------------------------------------- reduce / update
            total_n = sum(r[2] for r in results.values())
            train_loss = sum(r[0] for r in results.values()) / max(total_n, 1)
            if strategy == GRADIENT_SYNC:
                grad = sum((r[1] for r in results.values()), np.zeros_like(W)) / total_n + l2 * W
                velocity = momentum * velocity - lr * grad
                W = W + velocity
            else:
                W = sum((r[1] * r[2] for r in results.values()), np.zeros_like(W)) / total_n

            metrics = ml.evaluate(model, X_te, y_te, W)
            point = {"epoch": epoch, "train_loss": train_loss, "seconds": round(time.perf_counter() - t0, 3), **metrics}
            job.history.append(point)
            ctx.update(progress=epoch / epochs, force=epoch == epochs)
    finally:
        await _release(ctx, dispatcher)
        dispatcher.release()

    # ------------------------------------------------------------- result
    total = time.perf_counter() - t0
    final = ml.evaluate(model, X_te, y_te, W)
    train_metrics = ml.evaluate(model, X_tr, y_tr, W)
    if model == ml.LINEAR:
        headline = f"R2 {final['r2']:.3f}, RMSE {final['rmse']:.2f} in {total:.2f}s"
    else:
        headline = f"Accuracy {final['accuracy'] * 100:.1f}% in {total:.2f}s"
    ctx.log(f"Training finished: {headline}")
    return {
        "kind": "training",
        "headline": headline,
        "seconds": total,
        "timing": {
            "total": total,
            "scatter": scatter_seconds,
            "compute": compute_seconds,
            "communication": comm_seconds,
        },
        "metrics": {"test": final, "train": train_metrics},
        "dataset": {
            "name": ds.name,
            "rows": int(ds.X.shape[0]),
            "train_rows": int(X_tr.shape[0]),
            "test_rows": int(X_te.shape[0]),
            "features": ds.feature_names,
            "target": ds.target,
            "task": ds.task,
        },
        "workers_used": sorted(job.workers),
        "model": {
            "type": model,
            "label": ml.MODELS[model]["label"],
            "strategy": strategy,
            "features": ds.feature_names,
            "classes": ds.class_names,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "weights": W.tolist(),
        },
    }


async def _release(ctx: JobContext, dispatcher: Dispatcher) -> None:
    """Best-effort: free the shard memory on every worker that held one."""
    for w in list(dispatcher.ever_used.values()):
        if w.online:
            try:
                await asyncio.wait_for(ctx.cluster.call(w, "release", {"job_id": ctx.job.id}), timeout=10)
            except Exception:
                pass


def predict(model_info: dict[str, Any], rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    """Score raw feature rows with a trained model stored in a job result."""
    feats = model_info["features"]
    missing = [f for f in feats if any(f not in r for r in rows)]
    if missing:
        raise ValueError(f"missing feature(s): {sorted(set(missing))}")
    X = np.array([[float(r[f]) for f in feats] for r in rows], dtype=np.float64)
    X = ml.add_bias((X - np.array(model_info["mean"])) / np.array(model_info["std"]))
    W = np.array(model_info["weights"])
    model = model_info["type"]
    out = ml.forward(model, X, W)
    preds = ml.predict(model, X, W)
    classes = model_info.get("classes")
    results = []
    for i in range(len(rows)):
        if model == ml.LINEAR:
            results.append({"prediction": float(preds[i])})
            continue
        probs = [float(1 - out[i, 0]), float(out[i, 0])] if model == ml.LOGISTIC else [float(p) for p in out[i]]
        label = classes[int(preds[i])] if classes else int(preds[i])
        names = classes or [str(c) for c in range(len(probs))]
        results.append({"prediction": label, "probabilities": dict(zip(names, probs))})
    return results
