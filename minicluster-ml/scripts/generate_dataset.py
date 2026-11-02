"""Write the synthetic Student Performance dataset to a CSV file.

    python scripts/generate_dataset.py --rows 100000 --out data/student_data.csv

The CSV can be opened in Excel for discussion in class, or uploaded back into
the dashboard ("Upload your own CSV dataset") to train on it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minicluster.common.datasets import write_student_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/student_data.csv"))
    args = ap.parse_args()
    write_student_csv(args.out, args.rows, args.seed)
    print(f"wrote {args.rows:,} rows to {args.out}")


if __name__ == "__main__":
    main()
