"""``python -m minicluster.coordinator`` - start the coordinator + dashboard."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from minicluster.coordinator.app import create_app
from minicluster.coordinator.config import Settings


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="MiniCluster ML coordinator (web dashboard + worker server)")
    parser.add_argument("--host", default=settings.host, help="bind address (default: %(default)s)")
    parser.add_argument("--http-port", type=int, default=settings.http_port, help="dashboard/API port (default: %(default)s)")
    parser.add_argument("--worker-port", type=int, default=settings.worker_port,
                        help="TCP port workers connect to (default: %(default)s)")
    parser.add_argument("--data-dir", default=str(settings.data_dir), help="job history & uploads (default: %(default)s)")
    parser.add_argument("--log-level", default=settings.log_level)
    args = parser.parse_args()

    settings.host, settings.http_port, settings.worker_port = args.host, args.http_port, args.worker_port
    settings.data_dir = Path(args.data_dir).resolve()
    settings.log_level = args.log_level

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.http_port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
