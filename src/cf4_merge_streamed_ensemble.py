#!/usr/bin/env python3
"""Rebuild one all-data CR manifest from streamed, restart-safe sample files."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

try:
    from cf4_linear_cr import field_statistics
except ModuleNotFoundError:  # package-style import in tests and diagnostics
    from src.cf4_linear_cr import field_statistics


SAMPLE_RE = re.compile(
    r"\[sample (?P<seed>\d+)\] cg_rel=(?P<cg>[0-9.eE+-]+) "
    r"sec=(?P<sec>[0-9.]+)"
)


def log_diagnostics(paths: list[Path]) -> dict[int, tuple[float, float]]:
    found: dict[int, tuple[float, float]] = {}
    for path in paths:
        for match in SAMPLE_RE.finditer(path.read_text(errors="replace")):
            found[int(match["seed"])] = (
                float(match["cg"]), float(match["sec"])
            )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.base_manifest.read_text())
    seeds = list(range(args.seed_start, args.seed_end + 1))
    diagnostics = log_diagnostics(args.log)
    samples = []
    outputs = []
    for index, seed in enumerate(seeds, start=1):
        path = args.outdir / f"cf4_linear_cr_{args.tag}_s{seed}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        if seed not in diagnostics:
            raise RuntimeError(f"no CG diagnostic found for seed {seed}")
        with np.load(path, allow_pickle=False) as data:
            stored_seed = int(data["sample_seed"])
            if stored_seed != seed:
                raise RuntimeError(f"{path}: stored seed {stored_seed} != {seed}")
            field = data["s_out"].astype(np.float32)
            nuisance = data["nuisance_q"].astype(np.float64).tolist()
            box_size = float(data["L"])
        cg_rel, seconds = diagnostics[seed]
        samples.append({
            "seed": seed,
            "cg_rel": cg_rel,
            "seconds": seconds,
            "q": nuisance,
            **field_statistics(field, box_size),
        })
        outputs.append(str(path))
        if index == 1 or index % 16 == 0 or index == len(seeds):
            print(f"[merge] {index}/{len(seeds)} seed={seed}", flush=True)

    manifest["configuration"]["sample_seeds"] = seeds
    manifest["samples"] = samples
    manifest["outputs"] = outputs
    manifest["streamed_restart_merge"] = {
        "base_manifest": str(args.base_manifest.resolve()),
        "logs": [str(path.resolve()) for path in args.log],
        "reason": "bounded batches prevent JAX allocator growth; every seed and gate is unchanged",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(args.out)
    print(f"[merge] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
