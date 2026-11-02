"""Level 1 - classic distributed computing (scatter / compute / reduce)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from minicluster.common import compute
from minicluster.coordinator.jobs.base import Dispatcher, JobContext, call


async def run_compute(ctx: JobContext, spec: dict[str, Any], workers: int | None = None) -> dict[str, Any]:
    kernel = spec["kernel"]
    n = int(spec["n"])
    want = workers or int(spec.get("workers", 3))
    started = time.perf_counter()

    dispatcher = Dispatcher(ctx, want)
    pool = await dispatcher.ensure_pool()
    n_chunks = max(1, len(pool) * int(spec.get("chunks_per_worker", 4)))
    bounds = np.linspace(1, n + 1, n_chunks + 1, dtype=np.int64)
    chunks = {i: (int(a), int(b)) for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])) if b > a}
    ctx.log(f"Scatter: numbers 1..{n:,} split into {len(chunks)} chunks across {len(pool)} worker(s)")
    ctx.update(phase=f"Computing on {len(pool)} worker(s)", force=True)

    async def run_chunk(w, idx):
        a, b = chunks[idx]
        data, _, elapsed = await call(ctx, w, "compute", {"kernel": kernel, "start": a, "end": b})
        return data["parts"], elapsed, b - a, data.get("processes", 1)

    done_count = 0

    def on_done(idx, w, res):
        nonlocal done_count
        done_count += 1
        stat = ctx.worker_stat(w.id)
        stat["items"] += res[2]
        stat["processes"] = res[3]
        stat["range"] = f"{stat.get('range', '')} {chunks[idx][0]}-{chunks[idx][1] - 1}".strip()[-80:]
        ctx.job.progress = done_count / len(chunks)
        ctx.job.history.append({"chunk": idx, "worker": w.id, "start": chunks[idx][0], "end": chunks[idx][1] - 1,
                                "seconds": round(res[1], 4)})
        total = sum(1 for h in ctx.job.history if h.get("worker") == w.id)
        stat["progress"] = total / max(1, len(chunks))

    try:
        def describe(i):
            return f"chunk #{i} ({chunks[i][0]}-{chunks[i][1] - 1})"

        results = await dispatcher.map(list(chunks), run_chunk, describe, on_done)
    finally:
        dispatcher.release()

    ctx.update(phase="Reduce: combining partial results", force=True)
    parts = [p for idx in sorted(results) for p in results[idx][0]]
    reduced = compute.reduce_results(kernel, parts)
    elapsed = time.perf_counter() - started
    used = sorted({h["worker"] for h in ctx.job.history})
    return {
        "kind": "compute",
        "kernel": kernel,
        "kernel_label": compute.KERNEL_LABELS.get(kernel, kernel),
        "n": n,
        "chunks": len(chunks),
        "workers_used": used,
        "seconds": elapsed,
        "output": reduced,
        "headline": _headline(kernel, reduced, elapsed),
    }


def _headline(kernel: str, out: dict[str, Any], seconds: float) -> str:
    if kernel == "prime_count":
        return f"{out['primes_found']:,} primes found in {seconds:.2f}s"
    if kernel == "monte_carlo_pi":
        return f"pi ~ {out['pi_estimate']:.6f} in {seconds:.2f}s"
    return f"Squared {out['numbers']:,} numbers in {seconds:.2f}s"
