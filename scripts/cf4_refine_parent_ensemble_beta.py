#!/usr/bin/env python3
"""Refine beta around the first passing ensemble-conditioned parent proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--reference-beta", type=float, default=0.25)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    args = parser.parse_args()

    coarse = json.loads(args.coarse_manifest.read_text())
    reference_row = next(
        row for row in coarse["proposals"]
        if np.isclose(float(row["beta"]), args.reference_beta)
    )
    with np.load(args.base, allow_pickle=False) as data:
        base = np.asarray(data["s_out"], dtype=np.float64)
        common = {
            key: data[key]
            for key in ("N", "spacing", "L", "hh", "Om", "Ob", "A_s_1e9", "ns")
        }
        base_seed = int(data["sample_seed"])
    with np.load(reference_row["path"], allow_pickle=False) as data:
        reference = np.asarray(data["s_out"], dtype=np.float64)
    correction = (reference - base) / args.reference_beta

    args.outdir.mkdir(parents=True, exist_ok=True)
    proposals = []
    for beta in args.betas:
        field = (base + beta * correction).astype(np.float32)
        label = f"b{base_seed}_beta{beta:.2f}".replace(".", "p")
        path = args.outdir / f"cf4_parent_ensemble_conditioned_{label}.npz"
        np.savez(
            path,
            **common,
            s_out=field,
            sample_seed=np.int64(base_seed),
            source_base_seed=np.int64(base_seed),
            conditioning_beta=np.float64(beta),
        )
        proposals.append({
            "label": label,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "base_seed": base_seed,
            "beta": beta,
            "base_field_rms": float(base.std()),
            "correction_rms": float(correction.std()),
            "proposal_rms": float(field.std()),
            "proposal_base_correlation": float(np.corrcoef(base.ravel(), field.ravel())[0, 1]),
        })

    result = {
        "schema": "ouruniv-cf4-parent-ensemble-conditioning-pilot-v1",
        "status": "complete_proposals_require_forward_validation",
        "method": "beta refinement from the SHA-pinned beta=0.25 correction",
        "limitations": coarse["limitations"],
        "inputs": {
            "coarse_manifest": str(args.coarse_manifest.resolve()),
            "coarse_manifest_sha256": sha256_file(args.coarse_manifest),
            "base": str(args.base.resolve()),
            "reference_proposal": reference_row["path"],
            "reference_proposal_sha256": reference_row["sha256"],
        },
        "ensemble_size": coarse["ensemble_size"],
        "feature_names": coarse["feature_names"],
        "feature_mean": coarse["feature_mean"],
        "feature_target_by_base_seed": coarse["feature_target_by_base_seed"],
        "observation_sigma": coarse["observation_sigma"],
        "k_localization_h_Mpc": coarse["k_localization_h_Mpc"],
        "base_seeds": [base_seed],
        "betas": args.betas,
        "proposals": proposals,
        "decision": "FORWARD_VALIDATE_ONLY",
    }
    (args.outdir / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "n_proposals": len(proposals)}, sort_keys=True))


if __name__ == "__main__":
    main()
