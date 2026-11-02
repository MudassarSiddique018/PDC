# REST API Reference

Interactive OpenAPI docs are served by the coordinator at **`/docs`**
(Swagger UI) and **`/redoc`**; the raw schema is at `/openapi.json`.

If `MINICLUSTER_API_TOKEN` is set, send `Authorization: Bearer <token>` with
every request (the WebSocket takes `?token=<token>`). `/healthz` and
`/api/info` are always public.

## System

| Method | Path | Description |
|---|---|---|
| GET | `/healthz` | liveness probe → `{"status":"ok","version":"1.0.0","workers_online":3}` |
| GET | `/api/info` | version, whether auth is required, worker port, kernels, models |
| GET | `/metrics` | Prometheus text format (workers online, cores, per-worker CPU/RAM, jobs by status) |

## Cluster

| Method | Path | Description |
|---|---|---|
| GET | `/api/cluster` | all workers with live stats + job counters |
| DELETE | `/api/cluster/workers/{id}` | forget an **offline** worker |
| GET | `/api/events?limit=100` | recent event log entries |
| WS | `/ws` | live feed: `hello` (initial snapshot), `cluster` (every second), `job`, `job_deleted`, `log` |

## Jobs

| Method | Path | Description |
|---|---|---|
| GET | `/api/jobs?limit=100` | job summaries, newest first |
| POST | `/api/jobs` | submit a job (body below) → `201` with the job |
| GET | `/api/jobs/{id}` | full job: spec, per-worker stats, history, result |
| POST | `/api/jobs/{id}/cancel` | cancel a queued or running job |
| DELETE | `/api/jobs/{id}` | delete a finished job |
| POST | `/api/jobs/{id}/predict` | score rows with a completed training job's model |

### Job specs

**Level 1 — compute**
```json
{
  "name": "Lab 1 - squares",
  "spec": {"type": "compute", "kernel": "square_sum", "n": 10000, "workers": 3, "chunks_per_worker": 4}
}
```
`kernel`: `square_sum` | `prime_count` | `monte_carlo_pi`.

**Level 2 — training**
```json
{
  "spec": {
    "type": "training",
    "dataset": "student_performance",
    "rows": 300000,
    "target": "passed",
    "model": "logistic_regression",
    "strategy": "gradient_sync",
    "workers": 3,
    "epochs": 30,
    "lr": 0.5,
    "momentum": 0.9,
    "batch_size": 256,
    "l2": 0.0,
    "test_fraction": 0.2,
    "seed": 42
  }
}
```
* `model`: `linear_regression` | `logistic_regression` | `softmax_regression`
* `strategy`: `gradient_sync` | `model_averaging` (`batch_size` is used by model averaging)
* For an uploaded CSV use `"dataset": "upload:<file>.csv"` and set `target` to a column name.

**Level 3 — benchmark**
```json
{
  "spec": {
    "type": "benchmark",
    "base": {"type": "compute", "kernel": "prime_count", "n": 2000000},
    "max_workers": 3
  }
}
```
Result: `result.runs[] = {workers, seconds, speedup, efficiency, headline, timing}`.

### Prediction
```bash
curl -X POST http://localhost:8000/api/jobs/<id>/predict \
  -H 'Content-Type: application/json' \
  -d '{"rows":[{"matric_marks":82,"inter_marks":76,"entry_test":64,"attendance":88,
               "prev_gpa":3.1,"study_hours":18,"assignments_done":90}]}'
# {"predictions":[{"prediction":"Pass","probabilities":{"Fail":0.003,"Pass":0.997}}]}
```

## Datasets

| Method | Path | Description |
|---|---|---|
| GET | `/api/datasets` | built-in + uploaded datasets with their columns |
| POST | `/api/datasets` | `multipart/form-data` upload of a `.csv` (`file` field) |

CSV rules: header row; all feature columns numeric; the target may be numeric
or text (text labels are encoded automatically). The task type is inferred:
2 distinct integer values → binary, ≤ 20 → multi-class, otherwise regression.

## Errors

| Status | When |
|---|---|
| 401 | missing/invalid API token |
| 404 | unknown job |
| 409 | e.g. deleting a running job, predicting with a non-training job |
| 413 | upload larger than `MINICLUSTER_MAX_UPLOAD_MB` |
| 415 | upload is not a `.csv` |
| 422 | validation error (body explains which field) |

## Python client

See [`scripts/client_example.py`](../scripts/client_example.py) — a
dependency-free client that submits jobs, waits for them and calls predict.
