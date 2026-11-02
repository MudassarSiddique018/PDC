"""Regenerate the README screenshots in docs/images/.

Starts a throw-away local cluster (1 coordinator + 3 workers), drives a set of
demo jobs - including killing a worker mid-job to show fail-over - and
captures the dashboard with Playwright.

    pip install playwright && playwright install chromium
    python scripts/capture_screenshots.py            # or --chrome /usr/bin/google-chrome
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
HTTP, TCP = 8799, 9799
BASE = f"http://127.0.0.1:{HTTP}"


def api(path: str, body: dict | None = None):
    req = urllib.request.Request(BASE + path, method="POST" if body is not None else "GET")
    data = None
    if body is not None:
        req.add_header("Content-Type", "application/json")
        data = json.dumps(body).encode()
    with urllib.request.urlopen(req, data, timeout=30) as r:
        return json.loads(r.read())


def wait_job(job_id: str, timeout: float = 300) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        job = api(f"/api/jobs/{job_id}")
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise TimeoutError(job_id)


def submit(spec: dict, name: str = "") -> str:
    return api("/api/jobs", {"name": name, "spec": spec})["id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome", default=None, help="path to a Chrome/Chromium binary (default: Playwright's)")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="minicluster-shots-")
    env = {**os.environ, "MINICLUSTER_DATA_DIR": tmp, "PYTHONPATH": str(ROOT)}
    py = sys.executable
    procs: dict[str, subprocess.Popen] = {}
    procs["coord"] = subprocess.Popen([py, "-m", "minicluster.coordinator", "--http-port", str(HTTP), "--worker-port", str(TCP),
                                       "--log-level", "WARNING"], cwd=ROOT, env=env)

    def start_worker(name: str) -> None:
        procs[name] = subprocess.Popen([py, "-m", "minicluster.worker", "--port", str(TCP), "--name", name, "--cores", "1",
                                        "--log-level", "WARNING"], cwd=ROOT, env=env)

    try:
        time.sleep(2.5)
        for n in ("PC-01", "PC-02", "PC-03"):
            start_worker(n)
        time.sleep(2.5)

        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=args.chrome) if args.chrome else p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})

            def shot(hash_: str, name: str, wait: float = 1.5, full: bool = False, setup=None):
                page.goto(f"{BASE}/{hash_}")
                page.wait_for_timeout(int(wait * 1000))
                if setup:
                    setup()
                page.evaluate("document.getElementById('toast').hidden = true")
                page.screenshot(path=str(OUT / name), full_page=full)
                print("saved", name)

            # Level 1 + Level 2 results
            sq = submit({"type": "compute", "kernel": "square_sum", "n": 10000, "workers": 3}, "Lab 1 - square 10,000 numbers")
            wait_job(sq)
            pr = submit({"type": "compute", "kernel": "prime_count", "n": 2_000_000, "workers": 3, "chunks_per_worker": 6},
                        "Count primes up to 2 million")
            wait_job(pr)
            tr = submit({"type": "training", "rows": 300_000, "workers": 3, "epochs": 40, "model": "logistic_regression",
                         "target": "passed"}, "Smart Admission Predictor (pass / fail)")
            wait_job(tr)
            sm = submit({"type": "training", "rows": 200_000, "workers": 3, "epochs": 15, "model": "softmax_regression",
                         "target": "category", "strategy": "model_averaging", "lr": 0.05, "batch_size": 256},
                        "Grade category - model averaging")
            wait_job(sm)

            # A long job captured while running, then a worker is killed for the fail-over shot
            live = submit({"type": "training", "rows": 1_500_000, "workers": 3, "epochs": 120, "model": "logistic_regression",
                           "target": "passed"}, "1.5M students - live training")
            time.sleep(4)
            shot("#/overview", "dashboard-overview.png", wait=2.5)
            procs.pop("PC-03").kill()
            time.sleep(3)
            shot("#/overview", "fault-tolerance.png", wait=2.0)
            shot("#/logs", "event-log.png", wait=1.0)
            wait_job(live)
            start_worker("PC-03")
            time.sleep(3)

            bm = submit({"type": "benchmark", "base": {"type": "compute", "kernel": "prime_count", "n": 3_000_000,
                                                       "workers": 3, "chunks_per_worker": 4}}, "Speed-up: primes up to 3M")
            wait_job(bm)

            shot(f"#/jobs/{tr}", "training-result.png", wait=2.0, full=True)
            shot(f"#/jobs/{pr}", "compute-result.png", wait=2.0, full=True)
            shot(f"#/jobs/{bm}", "benchmark-speedup.png", wait=2.0, full=True)
            shot(f"#/jobs/{live}", "failover-job.png", wait=2.0, full=True)
            shot("#/jobs", "jobs-list.png")

            def tab(name):
                def go():
                    page.click(f"#job-tabs button[data-tab={name}]")
                    page.wait_for_timeout(400)
                return go

            shot("#/new", "new-job-compute.png", setup=tab("compute"))
            shot("#/new", "new-job-training.png", setup=tab("training"))
            shot("#/new", "new-job-benchmark.png", setup=tab("benchmark"))

            def predict():
                page.select_option("#predict-job", tr)
                page.wait_for_timeout(500)
                page.click("#form-predict button[type=submit]")
                page.wait_for_timeout(800)

            shot("#/predict", "predict.png", setup=predict)
            try:  # Swagger UI assets come from a CDN; skip the shot when offline
                page.goto(f"{BASE}/docs", wait_until="networkidle", timeout=60000)
                page.wait_for_timeout(1500)
                page.screenshot(path=str(OUT / "api-docs.png"))
                print("saved api-docs.png")
            except Exception as exc:
                print("skipped api-docs.png:", exc)

            mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
            mobile.goto(f"{BASE}/#/overview")
            mobile.wait_for_timeout(2000)
            mobile.screenshot(path=str(OUT / "mobile.png"))
            print("saved mobile.png")
            browser.close()
    finally:
        for proc in procs.values():
            proc.terminate()
        for proc in procs.values():
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
