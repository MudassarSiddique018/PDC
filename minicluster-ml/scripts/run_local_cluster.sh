#!/usr/bin/env bash
# Run a complete demo cluster on ONE machine: 1 coordinator + N workers.
#   ./scripts/run_local_cluster.sh          # 3 workers
#   WORKERS=4 CORES=2 ./scripts/run_local_cluster.sh
# Press Ctrl+C to stop everything.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=${WORKERS:-3}
CORES=${CORES:-1}
HTTP_PORT=${HTTP_PORT:-8000}
WORKER_PORT=${WORKER_PORT:-9000}
PY=${PYTHON:-python3}
[ -x .venv/bin/python ] && PY=.venv/bin/python

pids=()
cleanup() { echo; echo "Stopping cluster..."; kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

"$PY" -m minicluster.coordinator --http-port "$HTTP_PORT" --worker-port "$WORKER_PORT" &
pids+=($!)
sleep 2

for i in $(seq 1 "$WORKERS"); do
  "$PY" -m minicluster.worker --port "$WORKER_PORT" --name "$(printf 'PC-%02d' "$i")" --cores "$CORES" &
  pids+=($!)
done

echo
echo "  MiniCluster ML is running with $WORKERS worker(s)"
echo "  Dashboard:  http://localhost:$HTTP_PORT"
echo "  API docs:   http://localhost:$HTTP_PORT/docs"
echo
wait
