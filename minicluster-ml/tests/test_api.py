"""API-level tests using FastAPI's TestClient (no worker processes needed)."""

import pytest
from fastapi.testclient import TestClient

from conftest import free_port
from minicluster.coordinator.app import create_app
from minicluster.coordinator.config import Settings


def make_client(tmp_path, **overrides):
    settings = Settings()
    settings.data_dir = tmp_path
    settings.host = "127.0.0.1"
    settings.worker_port = free_port()
    settings.no_worker_timeout = 0.5
    for k, v in overrides.items():
        setattr(settings, k, v)
    return TestClient(create_app(settings))


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as c:
        yield c


def test_health_and_info(client):
    assert client.get("/healthz").json()["status"] == "ok"
    info = client.get("/api/info").json()
    assert "prime_count" in info["kernels"] and info["auth_required"] is False
    assert client.get("/").status_code == 200


def test_validation_rejects_bad_specs(client):
    bad = [
        {"spec": {"type": "compute", "kernel": "rm -rf", "n": 10}},
        {"spec": {"type": "compute", "n": 0}},
        {"spec": {"type": "training", "model": "linear_regression", "target": "passed"}},
        {"spec": {"type": "training", "target": "nope"}},
        {"spec": {"type": "unknown"}},
    ]
    for body in bad:
        assert client.post("/api/jobs", json=body).status_code == 422, body


def test_job_without_workers_fails_cleanly(client):
    job = client.post("/api/jobs", json={"spec": {"type": "compute", "n": 100}}).json()
    assert job["status"] == "queued"
    import time

    for _ in range(50):
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state["status"] == "failed":
            break
        time.sleep(0.1)
    assert state["status"] == "failed" and "no online workers" in state["error"]
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_cancel_queued_job(client):
    first = client.post("/api/jobs", json={"spec": {"type": "compute", "n": 100}}).json()
    second = client.post("/api/jobs", json={"spec": {"type": "compute", "n": 100}}).json()
    res = client.post(f"/api/jobs/{second['id']}/cancel").json()
    assert res["status"] == "cancelled"
    assert client.get(f"/api/jobs/{first['id']}").status_code == 200


def test_dataset_upload_and_listing(client):
    csv = b"x1,x2,label\n1,2,0\n3,4,1\n"
    res = client.post("/api/datasets", files={"file": ("../../evil name.csv", csv, "text/csv")})
    assert res.status_code == 201
    assert res.json()["id"] == "upload:evil_name.csv"  # path traversal stripped
    ids = [d["id"] for d in client.get("/api/datasets").json()]
    assert "student_performance" in ids and "upload:evil_name.csv" in ids
    assert client.post("/api/datasets", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 415


def test_metrics_endpoint(client):
    text = client.get("/metrics").text
    assert "minicluster_workers_online 0" in text


def test_predict_requires_completed_training(client):
    job = client.post("/api/jobs", json={"spec": {"type": "compute", "n": 100}}).json()
    assert client.post(f"/api/jobs/{job['id']}/predict", json={"rows": [{"a": 1}]}).status_code == 409


def test_api_token_is_enforced(tmp_path):
    with make_client(tmp_path, api_token="s3cret") as c:
        assert c.get("/healthz").status_code == 200  # health stays open for load balancers
        assert c.get("/api/cluster").status_code == 401
        assert c.get("/api/cluster", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/api/cluster", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/api/info").json()["auth_required"] is True
