#!/usr/bin/env python3
"""Freeze a P2 config from preregistered proposals that survive parent P1.

This is a deterministic bridge between the N192 P1 validation and the N576
P2 forward.  It changes no physical threshold: the resulting config inherits
the SHA-pinned v11 P2 gates and replaces only the proposal inputs and the list
of seeds selected by the already-frozen P1 policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-manifest", type=Path, required=True)
    parser.add_argument("--conditioned-p1-result", type=Path, required=True)
    parser.add_argument(
        "--base-config", type=Path,
        default=ROOT / "config/p2_lg_targets_v11_bgc_inverse_peak.json")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.proposal_manifest.read_text())
    p1 = json.loads(args.conditioned_p1_result.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("proposal manifest is not complete")
    if p1.get("status") != "complete":
        raise RuntimeError("conditioned P1 result is not complete")
    proposal_seeds = sorted(int(row["proposal_seed"])
                            for row in manifest["entries"])
    member_seeds = sorted(int(row["seed"]) for row in p1["members"])
    if proposal_seeds != member_seeds:
        raise RuntimeError("P1 members do not exactly match proposal seeds")
    passing = sorted(int(seed) for seed in p1["passing_seeds"])
    if not passing:
        raise RuntimeError("no proposal survived the frozen parent P1 gates")

    result = {
        "schema": "cf4-p2-lg-targets-v12-recentered-search-derived",
        "frozen_before_high_resolution_forwarding": True,
        "frozen_date": date.today().isoformat(),
        "extends": str(args.base_config.resolve()),
        "extends_sha256": sha256_file(args.base_config),
        "change_from_v11": (
            "Use fresh preregistered v4 morphology proposals that passed the "
            "unchanged parent-centred P1 gates. All inherited P2 thresholds "
            "remain unchanged. Every evolved screen pair is subsequently "
            "subject to the unchanged five P1 gates at its own midpoint."),
        "paired_small_scale_seeds": passing,
        "input": {
            "conditioned_proposal_manifest":
                str(args.proposal_manifest.resolve()),
            "conditioned_proposal_manifest_sha256":
                sha256_file(args.proposal_manifest),
            "conditioned_p1_result":
                str(args.conditioned_p1_result.resolve()),
            "conditioned_p1_result_sha256":
                sha256_file(args.conditioned_p1_result),
        },
        "selection_policy_addendum": {
            "parent_p1_survivors": passing,
            "recentered_p1_gate_required_before_ramses": True,
            "evaluate_every_p2_screen_pair_not_only_morphology_rank_one": True,
            "P1_and_P2_physical_thresholds_changed_from_v11": False,
            "no_retuning_after_forward_results": True,
        },
        "storage": {"output_directory": str(args.outdir.resolve())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[freeze] P1 survivors: {passing}", flush=True)
    print(f"[freeze] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
