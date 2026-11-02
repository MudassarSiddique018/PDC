# Changelog

## [1.0.0] — 2026-11-01

First release.

### Added
- Coordinator (FastAPI + asyncio) with REST API, WebSocket live feed, Prometheus metrics and health check.
- Worker node with TCP sockets, multithreading, multiprocessing pool and shared-memory data shards.
- **Level 1** distributed compute: square numbers, prime counting, Monte Carlo π.
- **Level 2** distributed training: linear / logistic / softmax regression with gradient-sync (parameter server) and model-averaging strategies.
- **Level 3** speed-up benchmark with speed-up and efficiency charts.
- Fault tolerance: heartbeat failure detection, automatic task and shard reassignment, worker auto-reconnect.
- Dynamic load balancing with data affinity.
- Synthetic Student Performance dataset generator and CSV upload.
- Prediction API and dashboard page.
- SQLite job history, API token and cluster-secret authentication.
- Docker image, docker-compose lab simulation, GitHub Actions CI.
- Documentation: architecture, lab guide with 8 exercises, API reference, deployment guide.
