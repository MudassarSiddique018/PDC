"""Drive MiniCluster ML from Python instead of the browser (REST API example).

    python scripts/client_example.py --url http://192.168.1.10:8000

Runs a Level-1 job, a Level-2 training job, prints the results and uses the
trained model for a prediction. Only the standard library is needed.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


class Client:
    def __init__(self, url: str, token: str = ""):
        self.url = url.rstrip("/")
        self.token = token

    def _req(self, method: str, path: str, body: dict | None = None):
        req = urllib.request.Request(self.url + path, method=method)
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        data = None
        if body is not None:
            req.add_header("Content-Type", "application/json")
            data = json.dumps(body).encode()
        with urllib.request.urlopen(req, data, timeout=60) as r:
            return json.loads(r.read())

    def submit(self, spec: dict, name: str = "") -> str:
        return self._req("POST", "/api/jobs", {"name": name, "spec": spec})["id"]

    def wait(self, job_id: str) -> dict:
        while True:
            job = self._req("GET", f"/api/jobs/{job_id}")
            print(f"\r  {job['status']:<10} {job['progress'] * 100:5.1f}%  {job['phase'][:60]:<60}", end="", flush=True)
            if job["status"] in ("completed", "failed", "cancelled"):
                print()
                return job
            time.sleep(0.5)

    def predict(self, job_id: str, rows: list[dict]) -> list[dict]:
        return self._req("POST", f"/api/jobs/{job_id}/predict", {"rows": rows})["predictions"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default="")
    args = ap.parse_args()
    c = Client(args.url, args.token)

    cluster = c._req("GET", "/api/cluster")
    print(f"Cluster: {cluster['online']}/{cluster['total']} workers online, {cluster['cores']} cores")

    print("Level 1 - square 10,000 numbers")
    job = c.wait(c.submit({"type": "compute", "kernel": "square_sum", "n": 10_000, "workers": 3}))
    print("  ", job["result"]["headline"] if job["result"] else job["error"])

    print("Level 2 - train the Smart Admission Predictor")
    job = c.wait(c.submit({"type": "training", "rows": 200_000, "epochs": 30, "workers": 3}))
    if job["status"] != "completed":
        raise SystemExit(job["error"])
    print("  ", job["result"]["headline"])

    student = {"matric_marks": 72, "inter_marks": 65, "entry_test": 48, "attendance": 70,
               "prev_gpa": 2.4, "study_hours": 8, "assignments_done": 60}
    print("Prediction for a borderline student:", c.predict(job["id"], [student])[0])


if __name__ == "__main__":
    main()
