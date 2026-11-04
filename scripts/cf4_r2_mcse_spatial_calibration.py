#!/usr/bin/env python3
"""Run the bounded synthetic spatial-field MCSE calibration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cf4_r2_mcse_spatial_calibration import run_spatial_calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--draws", type=int, default=512)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--batch-sizes", type=int, nargs=2, default=(32, 64))
    parser.add_argument("--minimum-batches", type=int, default=8)
    parser.add_argument("--correlation-lengths", type=float, nargs="+",
        default=(0., 1., 2.))
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--output", type=Path,
        help="write to a new JSON file; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        report = run_spatial_calibration(size=args.grid_size,
            draws=args.draws, chains=args.chains, replicates=args.replicates,
            seed=args.seed, batch_sizes=tuple(args.batch_sizes),
            minimum_batches=args.minimum_batches,
            correlation_lengths=tuple(args.correlation_lengths),
            threshold=args.threshold)
        report["implementation_commit"] = os.environ.get(
            "CF4_EXPECTED_COMMIT")
        report["runtime_versions"] = dict(python=sys.version.split()[0],
            numpy=np.__version__, scipy=scipy.__version__)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(payload)
        else:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(payload)
    except (OSError, ValueError, FloatingPointError) as error:
        print(f"spatial field-MCSE calibration refused: {error}",
            file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
