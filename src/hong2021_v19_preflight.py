#!/usr/bin/env python
"""Hard preflight for the frozen V19-E7 experiment."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from hong2021_v14_freeze import astrid_files
from hong2021_v15_edm import git_state
from hong2021_v18_init import EXPECTED_MODE_COUNTS_BY_GRID
from hong2021_v19_edm import (
    FROZEN_REGISTRY_SHA256,
    P_MEAN,
    P_STD,
    derive_band_anchored_noise,
    load_frozen_registry,
)


SCHEMA = "hong2021-v19-hard-preflight-v1"
KS_SAMPLES = 1_000_000
KS_SEED = 190071
KS_MAXIMUM = 0.002


def fixed_lognormal_ks() -> float:
    generator = torch.Generator(device="cpu").manual_seed(KS_SEED)
    normal = torch.randn(KS_SAMPLES, generator=generator, dtype=torch.float64)
    sample = torch.exp(normal * P_STD + P_MEAN)
    ordered = sample.sort().values
    standardized = (torch.log(ordered) - P_MEAN) / P_STD
    cdf = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
    upper = torch.arange(1, KS_SAMPLES + 1, dtype=torch.float64) / KS_SAMPLES
    lower = torch.arange(0, KS_SAMPLES, dtype=torch.float64) / KS_SAMPLES
    return float(torch.maximum((upper - cdf).max(), (cdf - lower).max()))


def rng_stream_self_check() -> bool:
    first = torch.Generator(device="cpu").manual_seed(190072)
    second = torch.Generator(device="cpu").manual_seed(190072)
    for _ in range(7):
        z_first = torch.randn(6, generator=first)
        _ = torch.exp(z_first * 1.2 + 0.18549829030348333)
        _ = torch.randn((6, 1, 4, 4, 4), generator=first)
        z_second = torch.randn(6, generator=second)
        _ = torch.exp(z_second * P_STD + P_MEAN)
        _ = torch.randn((6, 1, 4, 4, 4), generator=second)
    return bool(torch.equal(first.get_state(), second.get_state()))


def run_preflight(
    *, repo: Path, registry_path: Path, astrid_root: Path,
    forbidden_outputs: list[Path],
) -> dict:
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V19 hard preflight requires a clean worktree")
    registry = load_frozen_registry(registry_path.resolve(), repo.resolve())
    if astrid_files(astrid_root.resolve()):
        raise RuntimeError("Astrid is not unopened at V19 hard preflight")
    present = [str(path.resolve()) for path in forbidden_outputs if path.exists()]
    if present:
        raise RuntimeError(f"V19 output already exists: {present}")
    noise = registry["e7_band_anchored_noise_retrain"]["training_noise"]
    p_mean, p_std, _ = derive_band_anchored_noise(
        noise["source_balanced_per_band_mode_variance"]
    )
    relative = max(abs(p_mean - P_MEAN) / abs(P_MEAN), abs(p_std - P_STD) / abs(P_STD))
    if relative > 1.0e-12:
        raise RuntimeError("V19 derived noise constants failed preflight")
    ks = fixed_lognormal_ks()
    if ks >= KS_MAXIMUM:
        raise RuntimeError("V19 fixed lognormal draw KS failed preflight")
    if not rng_stream_self_check():
        raise RuntimeError("V19 training-noise RNG stream differs from E3")
    expected_counts = {
        "64": [146, 3596, 25296, 233105],
        "80": [250, 6872, 49928, 454949],
    }
    actual_counts = {
        str(grid): list(values) for grid, values in EXPECTED_MODE_COUNTS_BY_GRID.items()
    }
    if actual_counts != expected_counts:
        raise RuntimeError("V19 inherited initializer mode counts changed")
    return {
        "schema": SCHEMA,
        "registry": str(registry_path.resolve()),
        "registry_sha256": FROZEN_REGISTRY_SHA256,
        "code_commit": commit,
        "worktree_clean": True,
        "astrid_files": 0,
        "historical_eagle_accessed": False,
        "derived_noise_maximum_relative_difference": relative,
        "fixed_lognormal_ks": {
            "samples": KS_SAMPLES, "seed": KS_SEED,
            "value": ks, "maximum": KS_MAXIMUM, "pass": True,
        },
        "rng_stream_state_equal_to_e3": True,
        "mode_counts_by_grid": actual_counts,
        "forbidden_outputs_absent": [str(path.resolve()) for path in forbidden_outputs],
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--astrid-root", type=Path, required=True)
    parser.add_argument("--forbid-output", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"V19 refuses an existing preflight report: {args.out}")
    report = run_preflight(
        repo=args.repo, registry_path=args.registry, astrid_root=args.astrid_root,
        forbidden_outputs=args.forbid_output,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
