#!/usr/bin/env python3
"""Low-rank ensemble-conditioning proposal for missing z=0 structures.

This is deliberately a proposal pilot, not a production posterior.  It uses
the 256 CF4 velocity-conditioned parent draws to estimate cross-covariances
between the whitened IC and five z=0 environment summaries.  Corrections are
restricted to k <= 0.6 h/Mpc so the parent's smaller-scale realization is not
silently replaced.  Every proposal must be re-forwarded through PM+FoF and the
full P1 environment scorer before it can be considered further.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mass_feature(value: float | None, scale: float = 1.0e14) -> float:
    return float(np.log1p((0.0 if value is None else value) / scale))


def smooth_lowpass(field: np.ndarray, box: float, full: float, zero: float) -> np.ndarray:
    n = field.shape[0]
    fk = np.fft.rfftn(field)
    kxy = 2.0 * np.pi * np.fft.fftfreq(n, d=box / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=box / n)
    kmag = np.sqrt(
        kxy[:, None, None] ** 2 + kxy[None, :, None] ** 2 + kz[None, None, :] ** 2
    )
    weight = np.ones_like(kmag)
    transition = (kmag > full) & (kmag < zero)
    x = (kmag[transition] - full) / (zero - full)
    weight[transition] = 1.0 - x * x * (3.0 - 2.0 * x)
    weight[kmag >= zero] = 0.0
    return np.fft.irfftn(fk * weight, s=field.shape).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--p1-result", type=Path, required=True)
    parser.add_argument("--pm-fof-result", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--base-seeds", type=int, nargs="+", default=[3292])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--k-full", type=float, default=0.3)
    parser.add_argument("--k-zero", type=float, default=0.6)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    p1 = json.loads(args.p1_result.read_text())
    pmfof = json.loads(args.pm_fof_result.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)

    if pmfof.get("legacy_p1_and_both_pass_seeds"):
        result = {
            "schema": "ouruniv-cf4-parent-ensemble-conditioning-pilot-v1",
            "status": "skipped_existing_parent_available",
            "existing_seeds": pmfof["legacy_p1_and_both_pass_seeds"],
            "proposals": [],
        }
        (args.outdir / "manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, sort_keys=True))
        return

    source_by_seed = {}
    for path_text in manifest["outputs"]:
        path = Path(path_text)
        with np.load(path, allow_pickle=False) as data:
            source_by_seed[int(data["sample_seed"])] = path
    p1_by_seed = {int(row["seed"]): row for row in p1["members"]}
    fof_by_seed = {int(row["seed"]): row for row in pmfof["rows"]}
    seeds = sorted(set(source_by_seed) & set(p1_by_seed) & set(fof_by_seed))
    if len(seeds) != 256:
        raise RuntimeError(f"expected 256 aligned ensemble members, found {len(seeds)}")
    if not set(args.base_seeds) <= set(seeds):
        raise RuntimeError("one or more base seeds are absent")

    probe_names = [row["name"] for row in p1_by_seed[seeds[0]]["local_void"]["probes"]]
    feature_names = [
        "virgo_log1p_mass_over_1e14",
        "coma_log1p_mass_over_1e14",
        *[f"local_void_{name}_mean_delta" for name in probe_names],
        "bootes_R24_mean_delta",
        "observer_R8_mean_delta",
    ]
    features = []
    for seed in seeds:
        p = p1_by_seed[seed]
        f = fof_by_seed[seed]
        probe_by_name = {row["name"]: row for row in p["local_void"]["probes"]}
        features.append([
            mass_feature(f["anchors"]["Virgo"]["maximum_mass_msun_h"]),
            mass_feature(f["anchors"]["Coma"]["maximum_mass_msun_h"]),
            *[float(probe_by_name[name]["mean_delta"]) for name in probe_names],
            float(p["bootes_void"]["mean_delta_profile"]["24.0"]),
            float(p["observer_environment"]["spheres"]["8.0"]["mean_delta"]),
        ])
    y = np.asarray(features, dtype=np.float64)
    y_mean = y.mean(axis=0)
    y_anomaly = y - y_mean
    observation_sigma = np.array(
        [0.35, 0.35, *([0.15] * len(probe_names)), 0.15, 0.15], dtype=np.float64
    )
    covariance = y_anomaly.T @ y_anomaly / (len(seeds) - 1)
    regularized = covariance + np.diag(observation_sigma**2)

    # Member-space Kalman coefficients for each retained base realization.
    coefficient = {}
    target_by_base = {}
    for base in args.base_seeds:
        current = y[seeds.index(base)]
        target = current.copy()
        target[0] = max(current[0], mass_feature(1.0e14))
        target[1] = max(current[1], mass_feature(5.0e14))
        target[2:2 + len(probe_names)] = np.minimum(
            current[2:2 + len(probe_names)], -0.10
        )
        target[-2] = min(current[-2], -0.25)
        target[-1] = current[-1]
        target_by_base[base] = target
        innovation = target - current
        coefficient[base] = (
            y_anomaly @ np.linalg.solve(regularized, innovation) / (len(seeds) - 1)
        )

    with np.load(source_by_seed[seeds[0]], allow_pickle=False) as first:
        shape = first["s_out"].shape
        n = int(first["N"])
        box = float(first["L"])
    mean = np.zeros(shape, dtype=np.float64)
    for number, seed in enumerate(seeds, 1):
        with np.load(source_by_seed[seed], allow_pickle=False) as data:
            mean += np.asarray(data["s_out"], dtype=np.float64) / len(seeds)
        if number % 64 == 0:
            print(f"[ensemble] mean {number}/{len(seeds)}", flush=True)

    corrections = {base: np.zeros(shape, dtype=np.float64) for base in args.base_seeds}
    for number, seed in enumerate(seeds, 1):
        with np.load(source_by_seed[seed], allow_pickle=False) as data:
            anomaly = np.asarray(data["s_out"], dtype=np.float64) - mean
        for base in args.base_seeds:
            corrections[base] += coefficient[base][seeds.index(seed)] * anomaly
        if number % 64 == 0:
            print(f"[ensemble] gain {number}/{len(seeds)}", flush=True)

    proposals = []
    for base in args.base_seeds:
        correction = smooth_lowpass(
            corrections[base], box, args.k_full, args.k_zero
        ).astype(np.float64)
        with np.load(source_by_seed[base], allow_pickle=False) as data:
            base_field = np.asarray(data["s_out"], dtype=np.float64)
            common = {
                key: data[key]
                for key in ("N", "spacing", "L", "hh", "Om", "Ob", "A_s_1e9", "ns")
            }
        for beta in args.betas:
            proposal = (base_field + beta * correction).astype(np.float32)
            label = f"b{base}_beta{beta:.2f}".replace(".", "p")
            path = args.outdir / f"cf4_parent_ensemble_conditioned_{label}.npz"
            np.savez(
                path,
                **common,
                s_out=proposal,
                sample_seed=np.int64(base),
                source_base_seed=np.int64(base),
                conditioning_beta=np.float64(beta),
            )
            proposals.append(
                {
                    "label": label,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "base_seed": base,
                    "beta": beta,
                    "base_field_rms": float(base_field.std()),
                    "correction_rms": float(correction.std()),
                    "proposal_rms": float(proposal.std()),
                    "proposal_base_correlation": float(
                        np.corrcoef(base_field.ravel(), proposal.ravel())[0, 1]
                    ),
                }
            )

    result = {
        "schema": "ouruniv-cf4-parent-ensemble-conditioning-pilot-v1",
        "status": "complete_proposals_require_forward_validation",
        "method": "regularized ensemble linear regression in whitened-IC space",
        "limitations": [
            "proposal mechanism is an approximate nonlinear-observable update, not a final calibrated posterior",
            "no proposal is promotable without PM+FoF, P1, and power validation",
        ],
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "p1_result": str(args.p1_result.resolve()),
            "pm_fof_result": str(args.pm_fof_result.resolve()),
        },
        "ensemble_size": len(seeds),
        "feature_names": feature_names,
        "feature_mean": y_mean.tolist(),
        "feature_target_by_base_seed": {
            str(base): target_by_base[base].tolist() for base in args.base_seeds
        },
        "observation_sigma": observation_sigma.tolist(),
        "feature_covariance": covariance.tolist(),
        "k_localization_h_Mpc": {"full_below": args.k_full, "zero_above": args.k_zero},
        "base_seeds": args.base_seeds,
        "betas": args.betas,
        "proposals": proposals,
        "decision": "FORWARD_VALIDATE_ONLY",
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": result["status"], "n_proposals": len(proposals)}, sort_keys=True))


if __name__ == "__main__":
    main()
