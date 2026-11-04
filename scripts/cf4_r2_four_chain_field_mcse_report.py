"""Pool completed-chain field-MCSE statistics without reading held-out data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cf4_r2_four_chain_field_mcse import evaluate_four_chain_field_mcse


DEFAULT_SPLIT = Path("/gpfs/kjhan/CF4/z0_density/r2_sky_closed_split_v6/split.npz")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs=4, type=Path,
        help="four completed independent-prior chain result.json files")
    parser.add_argument("--training-split", type=Path, default=DEFAULT_SPLIT,
        help="frozen split NPZ; only the train_keys member is read")
    parser.add_argument("--output", type=Path,
        help="new JSON path; an existing file is never overwritten")
    args = parser.parse_args(argv)
    try:
        with np.load(args.training_split, allow_pickle=False) as split:
            train_keys = np.asarray(split["train_keys"], dtype=np.int64)
        result = evaluate_four_chain_field_mcse(args.reports,
            train_keys=train_keys)
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(payload)
        else:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(payload)
    except (OSError, KeyError, json.JSONDecodeError, TypeError, ValueError,
            FloatingPointError) as error:
        print(f"four-chain field-MCSE report refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
