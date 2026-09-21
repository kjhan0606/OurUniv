#!/usr/bin/env python3
"""Audit a conditioned parent against the original linear CF4 likelihood."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cf4_linear_cr as linear  # noqa: E402


def profile_nuisance(signal: np.ndarray, data: dict, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    residual_without_q = data["vobs"][train] - signal
    variance = data["variance"][train]
    design = data["B"][train]
    prior_precision = np.diag(1.0 / data["q_std"] ** 2)
    normal = design.T @ (design / variance[:, None]) + prior_precision
    rhs = design.T @ (residual_without_q / variance)
    q = np.linalg.solve(normal, rhs)
    residual = residual_without_q - design @ q
    chi2 = float(np.sum(residual**2 / variance))
    return q, residual, chi2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.parent_manifest.read_text())
    model = dict(manifest["configuration"])
    model["sample_seeds"] = [int(x) for x in model["sample_seeds"]]
    namespace = Namespace(**model)
    data = linear.prepare_catalog(namespace)
    train = ~data["holdout"]
    A, _, _, dtype = linear.build_forward(data["pos"][train], data["rhat"][train], namespace)

    with np.load(args.base, allow_pickle=False) as source:
        base = np.asarray(source["s_out"], dtype=np.float32)
    with np.load(args.candidate, allow_pickle=False) as source:
        candidate = np.asarray(source["s_out"], dtype=np.float32)
    if candidate.shape != base.shape:
        raise RuntimeError("candidate/base shape mismatch")

    base_signal = np.asarray(A(jnp.asarray(base, dtype=dtype)), dtype=np.float64)
    candidate_signal = np.asarray(A(jnp.asarray(candidate, dtype=dtype)), dtype=np.float64)
    base_q, base_residual, base_chi2 = profile_nuisance(base_signal, data, train)
    candidate_q, candidate_residual, candidate_chi2 = profile_nuisance(candidate_signal, data, train)

    correction = candidate.astype(np.float64) - base.astype(np.float64)
    delta_signal = candidate_signal - base_signal
    variance = data["variance"][train]
    design = data["B"][train]
    prior_precision = np.diag(1.0 / data["q_std"] ** 2)
    normal = design.T @ (design / variance[:, None]) + prior_precision
    delta_q = np.linalg.solve(normal, -design.T @ (delta_signal / variance))
    profiled_delta_signal = delta_signal + design @ delta_q
    field_term = float(np.sum(correction**2))
    likelihood_term = float(np.sum(profiled_delta_signal**2 / variance))
    nuisance_term = float(delta_q @ prior_precision @ delta_q)
    mahalanobis2 = field_term + likelihood_term + nuisance_term

    result = {
        "schema": "ouruniv-cf4-parent-candidate-likelihood-audit-v1",
        "status": "complete_read_only_statistical_audit",
        "parent_manifest": str(args.parent_manifest.resolve()),
        "base": str(args.base.resolve()),
        "candidate": str(args.candidate.resolve()),
        "n_velocity_rows": int(train.sum()),
        "profiled_likelihood": {
            "base_normalized_residual_rms": float(np.sqrt(np.mean(base_residual**2 / data["variance"][train]))),
            "candidate_normalized_residual_rms": float(np.sqrt(np.mean(candidate_residual**2 / data["variance"][train]))),
            "base_chi2": base_chi2,
            "candidate_chi2": candidate_chi2,
            "delta_chi2": candidate_chi2 - base_chi2,
            "base_profiled_nuisance_q": base_q.tolist(),
            "candidate_profiled_nuisance_q": candidate_q.tolist(),
        },
        "posterior_displacement": {
            "field_prior_term": field_term,
            "velocity_likelihood_term_after_nuisance_profile": likelihood_term,
            "nuisance_prior_term": nuisance_term,
            "mahalanobis_squared": mahalanobis2,
            "mahalanobis": float(np.sqrt(mahalanobis2)),
            "profiled_delta_q": delta_q.tolist(),
            "correction_rms_per_voxel": float(correction.std()),
            "candidate_base_correlation": float(np.corrcoef(base.ravel(), candidate.ravel())[0, 1]),
        },
        "interpretation": (
            "The Mahalanobis distance is measured in the original CF4 linear-Gaussian "
            "posterior metric after profiling the four nuisance modes. It is a compatibility "
            "diagnostic, not the normalization of the added structure likelihood."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
