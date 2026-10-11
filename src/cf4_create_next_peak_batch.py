#!/usr/bin/env python3
"""Pre-register a fresh unchanged-likelihood LG morphology search batch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--failed-preview", type=Path, required=True)
    p.add_argument("--seed-start", type=int, required=True)
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--label", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    preview = json.loads(a.failed_preview.read_text())
    if preview.get("passing_seeds"):
        p.error("fallback batch may only follow a zero-survivor preview")
    config = json.loads(a.base.read_text())
    seeds = list(range(a.seed_start, a.seed_start + a.count))
    config["schema"] = f"ouruniv-explicit-lg-peak-likelihood-{a.label}"
    config["frozen_date"] = "2026-08-04"
    config["supersedes"] = str(a.base.resolve())
    config["supersedes_sha256"] = sha256_file(a.base)
    config["activation_condition"] = (
        f"The preceding recentered screen had zero survivors; draw {a.count} "
        "fresh preregistered samples without changing the likelihood or any gate.")
    config["proposal_seeds"] = seeds
    config["geometry_seeds"] = [seed + 1100 for seed in seeds]
    config["likelihood_noise_seeds"] = [seed + 2200 for seed in seeds]
    config["selection_policy"]["failed_preview"] = str(a.failed_preview.resolve())
    config["selection_policy"]["failed_preview_sha256"] = sha256_file(a.failed_preview)
    base_dir = "/gpfs/kjhan/CF4/recon/linear_cr"
    config["storage"] = {
        "proposal_directory": f"{base_dir}/v3_bgc_lg_peak_proposals_{a.label}",
        "parent_projection_directory":
            f"{base_dir}/v3_bgc_lg_peak_parent_projections_{a.label}",
        "p1_directory": f"{base_dir}/v3_bgc_lg_peak_p1_{a.label}",
        "screen_directory": f"{base_dir}/v3_bgc_lg_peak_p2_{a.label}",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"[next-batch] frozen {a.label}: {seeds[0]}..{seeds[-1]}")
    print(a.out)


if __name__ == "__main__":
    main()
