"""Produce scores first, then independently evaluate the completed run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import evaluate, print_table
from .pipeline import run
from .prep import DEFAULT_DATA, DEFAULT_RUN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()
    run(args.data, args.out)
    results = evaluate(args.data, args.out, not args.skip_ablations)
    (args.out/"metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    print_table(results)
    sys.exit(0 if all(check["pass"] for check in results["acceptance"].values()) else 1)


if __name__ == "__main__":
    main()
