"""``python -m minicluster.worker`` - start a worker on this PC."""

from __future__ import annotations

import os

# One BLAS thread per process: parallelism is controlled explicitly with
# --cores (multiprocessing), which keeps speed-up measurements honest.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import signal  # noqa: E402

from minicluster.worker.node import WorkerNode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniCluster ML worker node")
    parser.add_argument("--coordinator", default=os.getenv("MINICLUSTER_COORDINATOR", "127.0.0.1"),
                        help="coordinator host or IP (default: %(default)s)")
    parser.add_argument("--port", type=int, default=int(os.getenv("MINICLUSTER_WORKER_PORT", "9000")),
                        help="coordinator worker port (default: %(default)s)")
    parser.add_argument("--name", default=os.getenv("MINICLUSTER_WORKER_NAME"),
                        help="display name, e.g. PC-01 (default: hostname)")
    parser.add_argument("--cores", type=int, default=int(os.getenv("MINICLUSTER_WORKER_CORES", "0")) or None,
                        help="CPU cores to use with multiprocessing (default: all)")
    parser.add_argument("--secret", default=os.getenv("MINICLUSTER_CLUSTER_SECRET", ""),
                        help="shared cluster secret, must match the coordinator")
    parser.add_argument("--slowdown", type=float, default=1.0,
                        help="artificially slow this worker down by a factor (load-balancing demos)")
    parser.add_argument("--no-reconnect", action="store_true", help="exit instead of reconnecting")
    parser.add_argument("--log-level", default=os.getenv("MINICLUSTER_LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s")
    node = WorkerNode(
        args.coordinator,
        args.port,
        name=args.name,
        cores=args.cores,
        secret=args.secret,
        reconnect=not args.no_reconnect,
        slowdown=args.slowdown,
    )
    signal.signal(signal.SIGTERM, lambda *_: node.stop())
    try:
        node.run()
    except KeyboardInterrupt:
        node.stop()
        node.close()


if __name__ == "__main__":
    main()
