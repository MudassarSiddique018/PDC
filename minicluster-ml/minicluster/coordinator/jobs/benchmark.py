"""Level 3 - speed-up experiment: run the same job on 1, 2, ... N workers.

    Speedup(p)    = T(1) / T(p)
    Efficiency(p) = Speedup(p) / p
"""

from __future__ import annotations

from typing import Any

from minicluster.coordinator.jobs.base import JobContext
from minicluster.coordinator.jobs.compute import run_compute
from minicluster.coordinator.jobs.training import run_training


async def run_benchmark(ctx: JobContext, spec: dict[str, Any]) -> dict[str, Any]:
    base = spec["base"]
    runner = run_compute if base["type"] == "compute" else run_training
    online = len(ctx.cluster.online_workers())
    max_workers = min(int(spec.get("max_workers") or online or 1), max(online, 1))
    counts = list(range(1, max_workers + 1))
    rows: list[dict[str, Any]] = []
    ctx.log(f"Benchmark: running the same {base['type']} job on {counts} worker(s)")

    for i, p in enumerate(counts):
        ctx.job.check_cancelled()
        ctx.update(phase=f"Run {i + 1}/{len(counts)}: {p} worker(s)", force=True)
        ctx.job.history = []
        ctx.job.workers = {}
        res = await runner(ctx, base, workers=p)
        t = res["seconds"]
        t1 = rows[0]["seconds"] if rows else t
        speedup = t1 / t if t else 0.0
        rows.append({
            "workers": p,
            "workers_used": len(res.get("workers_used", [])),
            "seconds": round(t, 4),
            "speedup": round(speedup, 3),
            "efficiency": round(speedup / p, 3),
            "headline": res.get("headline"),
            "timing": res.get("timing"),
        })
        ctx.log(f"Benchmark: {p} worker(s) -> {t:.2f}s (speed-up x{speedup:.2f})")
        ctx.update(progress=(i + 1) / len(counts), force=True)

    ctx.job.history = rows
    best = max(rows, key=lambda r: r["speedup"])
    return {
        "kind": "benchmark",
        "base": base,
        "runs": rows,
        "seconds": sum(r["seconds"] for r in rows),
        "headline": f"Best speed-up x{best['speedup']:.2f} with {best['workers']} worker(s)",
    }
