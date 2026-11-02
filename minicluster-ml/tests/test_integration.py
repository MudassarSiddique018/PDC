"""End-to-end tests: a real coordinator and real worker processes over TCP."""

import time

import pytest


def test_distributed_compute_is_correct(cluster):
    for name in ("PC-01", "PC-02", "PC-03"):
        cluster.start_worker(name)
    cluster.wait_workers(3)
    job = cluster.wait_job(cluster.submit({"type": "compute", "kernel": "prime_count", "n": 100_000, "workers": 3}))
    assert job["status"] == "completed", job["error"]
    assert job["result"]["output"]["primes_found"] == 9592  # pi(100,000)
    assert set(job["result"]["workers_used"]) == {"PC-01", "PC-02", "PC-03"}


def test_multiprocessing_worker(cluster):
    cluster.start_worker("PC-MP", cores=2)
    cluster.wait_workers(1)
    job = cluster.wait_job(cluster.submit({"type": "compute", "kernel": "square_sum", "n": 20_000, "workers": 1}))
    n = 20_000
    assert int(job["result"]["output"]["sum_of_squares"]) == n * (n + 1) * (2 * n + 1) // 6
    assert job["workers"]["PC-MP"]["processes"] == 2
    tr = cluster.wait_job(cluster.submit({"type": "training", "rows": 20_000, "workers": 1, "epochs": 20}))
    assert tr["status"] == "completed", tr["error"]
    assert tr["result"]["metrics"]["test"]["accuracy"] > 0.85


@pytest.mark.parametrize("strategy", ["gradient_sync", "model_averaging"])
def test_distributed_training_and_prediction(cluster, strategy):
    cluster.start_worker("PC-01")
    cluster.start_worker("PC-02")
    cluster.wait_workers(2)
    spec = {"type": "training", "rows": 30_000, "workers": 2, "epochs": 25, "strategy": strategy}
    if strategy == "model_averaging":
        spec.update(lr=0.05, epochs=5)
    job = cluster.wait_job(cluster.submit(spec))
    assert job["status"] == "completed", job["error"]
    assert job["result"]["metrics"]["test"]["accuracy"] > 0.85
    assert len(job["history"]) == spec["epochs"]
    row = {"matric_marks": 90, "inter_marks": 88, "entry_test": 85, "attendance": 95, "prev_gpa": 3.8,
           "study_hours": 25, "assignments_done": 98}
    pred = cluster.request(f"/api/jobs/{job['id']}/predict", {"rows": [row]})["predictions"][0]
    assert pred["prediction"] == "Pass"


def test_gradient_sync_is_independent_of_worker_count(cluster):
    """Same data, same seed: 1 PC and 3 PCs must produce the same model."""
    for name in ("PC-01", "PC-02", "PC-03"):
        cluster.start_worker(name)
    cluster.wait_workers(3)
    base = {"type": "training", "rows": 9_000, "epochs": 10, "model": "softmax_regression", "target": "category"}
    one = cluster.wait_job(cluster.submit({**base, "workers": 1}))
    three = cluster.wait_job(cluster.submit({**base, "workers": 3}))
    w1, w3 = one["result"]["model"]["weights"], three["result"]["model"]["weights"]
    for r1, r3 in zip(w1, w3):
        assert r1 == pytest.approx(r3, rel=1e-9, abs=1e-12)


def test_worker_failure_during_compute_is_recovered(cluster):
    for name in ("PC-01", "PC-02", "PC-03"):
        cluster.start_worker(name)
    cluster.wait_workers(3)
    spec = {"type": "compute", "kernel": "prime_count", "n": 1_500_000, "workers": 3, "chunks_per_worker": 6}
    job_id = cluster.submit(spec)
    time.sleep(1.0)
    cluster.kill_worker("PC-02")
    job = cluster.wait_job(job_id)
    assert job["status"] == "completed", job["error"]
    assert job["result"]["output"]["primes_found"] == 114155  # pi(1,500,000)
    assert job["workers"]["PC-02"]["lost"] is True
    events = [e["message"] for e in cluster.get("/api/events?limit=50")]
    assert any("Task reassignment" in m for m in events)


def test_worker_failure_during_training_is_recovered(cluster):
    for name in ("PC-01", "PC-02", "PC-03"):
        cluster.start_worker(name)
    cluster.wait_workers(3)
    job_id = cluster.submit({"type": "training", "rows": 400_000, "workers": 3, "epochs": 80})
    cluster._wait(lambda: cluster.get(f"/api/jobs/{job_id}")["progress"] > 0.1, timeout=30)
    cluster.kill_worker("PC-03")
    job = cluster.wait_job(job_id)
    assert job["status"] == "completed", job["error"]
    assert job["result"]["metrics"]["test"]["accuracy"] > 0.85
    assert job["workers"]["PC-03"]["lost"] is True


def test_benchmark_reports_speedup(cluster):
    cluster.start_worker("PC-01")
    cluster.start_worker("PC-02")
    cluster.wait_workers(2)
    base = {"type": "compute", "kernel": "prime_count", "n": 300_000}
    job = cluster.wait_job(cluster.submit({"type": "benchmark", "base": base}))
    runs = job["result"]["runs"]
    assert [r["workers"] for r in runs] == [1, 2]
    assert runs[0]["speedup"] == 1.0
    assert all(r["efficiency"] == pytest.approx(r["speedup"] / r["workers"], abs=1e-3) for r in runs)


def test_wrong_cluster_secret_is_rejected(tmp_path):
    from conftest import LiveCluster

    c = LiveCluster(tmp_path)
    c.coord.terminate()
    c.coord.wait(5)
    c.env["MINICLUSTER_CLUSTER_SECRET"] = "right"
    import subprocess
    import sys

    from conftest import ROOT

    c.coord = subprocess.Popen([sys.executable, "-m", "minicluster.coordinator", "--http-port", str(c.http),
                                "--worker-port", str(c.tcp), "--log-level", "WARNING"], cwd=ROOT, env=c.env)
    try:
        c._wait(lambda: c.get("/healthz")["status"] == "ok")
        bad = subprocess.Popen([sys.executable, "-m", "minicluster.worker", "--port", str(c.tcp), "--name", "EVIL",
                                "--secret", "wrong", "--log-level", "ERROR"], cwd=ROOT, env=c.env)
        assert bad.wait(15) == 0  # rejected workers exit instead of retrying
        c.env["MINICLUSTER_CLUSTER_SECRET"] = "right"
        c.start_worker("GOOD")
        c.wait_workers(1)
        assert [w["id"] for w in c.get("/api/cluster")["workers"]] == ["GOOD"]
    finally:
        c.close()


def test_multiprocessing_worker_exits_on_sigterm_mid_task(cluster):
    """Pool children must not inherit the worker's SIGTERM handler, or Pool.terminate() hangs."""
    cluster.start_worker("PC-MP", cores=2)
    cluster.wait_workers(1)
    cluster.submit({"type": "compute", "kernel": "prime_count", "n": 3_000_000, "workers": 1})
    cluster._wait(lambda: cluster.get("/api/cluster")["running_tasks"] > 0)
    time.sleep(0.5)
    proc = cluster.workers.pop("PC-MP")
    proc.terminate()
    try:
        assert proc.wait(10) == 0
    finally:
        proc.kill()
