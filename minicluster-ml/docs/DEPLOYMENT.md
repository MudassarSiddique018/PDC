# Deployment & Operations

## Configuration

The coordinator reads `MINICLUSTER_*` environment variables (CLI flags
override the first five).

| Variable | Default | Description |
|---|---|---|
| `MINICLUSTER_HOST` | `0.0.0.0` | bind address for HTTP and worker TCP |
| `MINICLUSTER_HTTP_PORT` | `8000` | dashboard + REST API |
| `MINICLUSTER_WORKER_PORT` | `9000` | TCP port workers connect to |
| `MINICLUSTER_DATA_DIR` | `./data` | SQLite job history + uploaded CSVs |
| `MINICLUSTER_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, … |
| `MINICLUSTER_API_TOKEN` | *(empty)* | if set, required as Bearer token for the API, metrics and WebSocket |
| `MINICLUSTER_CLUSTER_SECRET` | *(empty)* | if set, workers must present it to register |
| `MINICLUSTER_HEARTBEAT_TIMEOUT` | `6` | seconds of silence before a worker is declared offline |
| `MINICLUSTER_NO_WORKER_TIMEOUT` | `30` | how long a job waits for at least one worker before failing |
| `MINICLUSTER_MAX_CONCURRENT_JOBS` | `1` | jobs executed in parallel (1 keeps benchmark timings clean) |
| `MINICLUSTER_MAX_UPLOAD_MB` | `200` | CSV upload limit |
| `MINICLUSTER_MAX_DATASET_ROWS` | `5000000` | upper bound for the generated dataset |
| `MINICLUSTER_CORS_ORIGINS` | *(empty)* | comma-separated origins allowed to call the API from other sites |

Worker variables: `MINICLUSTER_COORDINATOR`, `MINICLUSTER_WORKER_PORT`,
`MINICLUSTER_WORKER_NAME`, `MINICLUSTER_WORKER_CORES`,
`MINICLUSTER_CLUSTER_SECRET`, `MINICLUSTER_LOG_LEVEL`.

## Docker

```bash
docker compose up --build                       # coordinator + 3 workers
docker compose up --build --scale worker=6      # more workers
MINICLUSTER_CLUSTER_SECRET=s3cret MINICLUSTER_API_TOKEN=t0ken docker compose up -d
```

Real lab PCs can join the containerised coordinator too, because port 9000
is published: `python -m minicluster.worker --coordinator <docker-host-ip> --secret s3cret`.

## Running as a service (Linux, systemd)

`/etc/systemd/system/minicluster-worker.service`:

```ini
[Unit]
Description=MiniCluster ML worker
After=network-online.target
Wants=network-online.target

[Service]
User=lab
WorkingDirectory=/opt/minicluster-ml
Environment=MINICLUSTER_CLUSTER_SECRET=change-me
ExecStart=/opt/minicluster-ml/.venv/bin/python -m minicluster.worker --coordinator 192.168.1.10 --name %H
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now minicluster-worker
```

Create the same for the coordinator with
`ExecStart=… -m minicluster.coordinator`.

On Windows, `scripts\start_worker.bat` can be placed in the Startup folder
(`shell:startup`) or registered with Task Scheduler.

## TLS / reverse proxy

The dashboard and API can be put behind nginx (WebSocket upgrade required):

```nginx
server {
    listen 443 ssl;
    server_name minicluster.lab.local;
    ssl_certificate     /etc/ssl/minicluster.crt;
    ssl_certificate_key /etc/ssl/minicluster.key;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        client_max_body_size 200m;
    }
}
```

Bind the coordinator to localhost for HTTP (`--host 127.0.0.1` would also
bind the worker port to localhost, so in that case keep `0.0.0.0` and firewall
port 8000 instead).

## Monitoring

* **Health:** `GET /healthz` (used by the Docker `HEALTHCHECK`).
* **Prometheus:** scrape `GET /metrics`:

```yaml
scrape_configs:
  - job_name: minicluster
    static_configs: [{targets: ["192.168.1.10:8000"]}]
    authorization: {credentials: "<api token if set>"}
```

Exposed series: `minicluster_workers_online`, `minicluster_cluster_cores`,
`minicluster_tasks_dispatched_total`, `minicluster_jobs{status}`,
`minicluster_worker_cpu_percent{worker}`, `minicluster_worker_ram_percent{worker}`,
`minicluster_worker_tasks_done_total{worker}`.

## Backup

Everything persistent is in `MINICLUSTER_DATA_DIR`: copy
`minicluster.db` (SQLite, WAL mode — copy while stopped or use
`sqlite3 minicluster.db ".backup backup.db"`) and the `uploads/` folder.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Worker prints `connection refused` | coordinator not running, wrong IP, or firewall blocking port 9000 |
| Worker appears then goes OFFLINE every few seconds | network drops heartbeats — raise `MINICLUSTER_HEARTBEAT_TIMEOUT` |
| `coordinator rejected this worker: invalid cluster secret` | `--secret` on the worker must match the coordinator |
| Dashboard says *Reconnecting…* | WebSocket blocked by a proxy — enable upgrade headers (see nginx above) |
| Job fails with `no online workers` | start at least one worker within `NO_WORKER_TIMEOUT` seconds |
| `/docs` page is blank | Swagger UI loads from a CDN; without internet use `docs/API.md` |
| Speed-up lower than expected | check *Time breakdown*: small datasets are communication bound; running many workers on one PC is limited by its physical cores |
