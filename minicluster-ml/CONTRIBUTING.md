# Contributing

Thanks for helping improve MiniCluster ML!

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
make test          # or: pytest -q
make lint          # or: ruff check minicluster tests scripts
make dev           # coordinator + 3 local workers
```

## Guidelines

* Keep the coordinator **single event loop**: never block it — use
  `asyncio.to_thread` for CPU work.
* Workers must only execute the fixed task kinds in `WorkerNode.handle`.
  Never send code or pickles over the network.
* Every new job type should go through `Dispatcher` so it gets load
  balancing and fail-over for free.
* The dashboard is dependency-free vanilla JS on purpose (labs are often
  offline). Escape every server-provided string with `esc()`.
* Add tests: unit tests for pure functions, `tests/test_integration.py` for
  anything that crosses the network.
* Regenerate README screenshots after UI changes:
  `pip install playwright && playwright install chromium && python scripts/capture_screenshots.py`.

## Pull requests

1. Fork and create a feature branch.
2. `make lint test` must pass (CI runs both on Python 3.10–3.12 plus a Docker build).
3. Describe *what* and *why*; include a screenshot for UI changes.
