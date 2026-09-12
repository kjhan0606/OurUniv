"""Development-only latent-mode coverage diagnosis for B1.

Unlike the legacy per-voxel diagnostic, this computes coverage as the
frequency over independent development seeds that the true eight-mode
coefficient lies inside each member's posterior interval.  It is diagnostic
only and does not alter the frozen promotion gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

import cf4_b1_integrated_joint_calibration_v3 as integrated


TARGET_68 = 0.6826894921370859
TARGET_95 = 0.9544997361036416


def analytic_posterior_sigma(estimate: np.ndarray, arm: str) -> np.ndarray:
    """Return the exact Laplace posterior sigma used by the coverage metric."""

    precision = (
        np.eye(integrated.MODE_COUNT) / integrated.PRIOR_SIGMA**2
        + integrated._count_fisher(estimate, arm)
        + integrated.MARK_FISHER
    )
    covariance = np.linalg.pinv(precision) * integrated.ARM_WIDTH_SCALE[arm] ** 2
    return np.sqrt(np.maximum(np.diag(covariance), 0.0))


def run_mode_coverage() -> dict[str, object]:
    arms = ("A", "B", "C", "D")
    rows = []
    for index in range(integrated.MOCK_COUNT):
        arm = arms[index // 16]
        member = integrated.run_mock(index, arm)
        truth = np.asarray(member["_truth_coeff"], dtype=np.float64)
        estimate = np.asarray(member["_estimate_coeff"], dtype=np.float64)
        draws = np.asarray(member["_draws_coeff"], dtype=np.float64)
        posterior_sigma = analytic_posterior_sigma(estimate, arm)
        rows.append({
            "index": index,
            "arm": arm,
            "seed": member["seed"],
            "coverage68_by_mode": (np.abs(estimate - truth) <= posterior_sigma).tolist(),
            "coverage95_by_mode": (np.abs(estimate - truth) <= 1.96 * posterior_sigma).tolist(),
            "truth_coeff": truth.tolist(),
            "estimate_coeff": estimate.tolist(),
            "posterior_sigma": posterior_sigma.tolist(),
        })

    def frequencies(subset: list[dict[str, object]], key: str) -> list[float]:
        matrix = np.asarray([row[key] for row in subset], dtype=bool)
        return np.mean(matrix, axis=0).astype(float).tolist()

    by_arm: dict[str, object] = {}
    for arm in arms:
        subset = [row for row in rows if row["arm"] == arm]
        f68 = frequencies(subset, "coverage68_by_mode")
        f95 = frequencies(subset, "coverage95_by_mode")
        by_arm[arm] = {
            "member_count": len(subset),
            "coverage68_frequency_by_mode": f68,
            "coverage95_frequency_by_mode": f95,
            "coverage68_frequency_mean": float(np.mean(f68)),
            "coverage95_frequency_mean": float(np.mean(f95)),
            "coverage68_target": TARGET_68,
            "coverage95_target": TARGET_95,
        }
    all68 = frequencies(rows, "coverage68_by_mode")
    all95 = frequencies(rows, "coverage95_by_mode")
    implementation = Path(__file__).resolve()
    return {
        "schema": "ouruniv-cf4-b1-mode-coverage-diagnosis-result-v3",
        "status": "COMPLETE_DEVELOPMENT_ONLY_NO_SCIENCE_CLAIM",
        "source": "src/cf4_b1_integrated_joint_calibration_v3.py",
        "source_artifact": {
            "path": str(Path(integrated.__file__).resolve()),
            "bytes": Path(integrated.__file__).stat().st_size,
            "sha256": hashlib.sha256(Path(integrated.__file__).read_bytes()).hexdigest(),
        },
        "implementation_artifact": {
            "path": str(implementation),
            "bytes": implementation.stat().st_size,
            "sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
        },
        "dependency_artifact": {
            "path": str(Path(integrated.base.__file__).resolve()),
            "bytes": Path(integrated.base.__file__).stat().st_size,
            "sha256": hashlib.sha256(Path(integrated.base.__file__).read_bytes()).hexdigest(),
        },
        "posterior_width_calibration": {
            "arm_scale": {key: float(value) for key, value in integrated.ARM_WIDTH_SCALE.items()},
            "D_overdispersion_phi": float(integrated.D_OVERDISPERSION_PHI),
            "mean_model_changed": False,
        },
        "metric_definition": {
            "coverage_unit": "independent development seed",
            "parameter_unit": "latent coefficient mode",
            "interval68": "posterior mean +/- analytic sqrt(diag(Laplace covariance))",
            "interval95": "posterior mean +/- 1.96 analytic sqrt(diag(Laplace covariance))",
            "sigma_source": "analytic_sqrt_diag_laplace_covariance",
            "posterior_draws_retained_for_cross_check": integrated.POSTERIOR_DRAWS,
            "legacy_per_voxel_metric_replaced_for_diagnosis": True,
            "strict_promotion_gate_changed": False,
        },
        "seed_firewall": {
            "development": [2026083000, 2026083064],
            "development_auxiliary_streams": {
                "counts": [2026400000, 2026400064],
                "count_split": [2026401000, 2026401064],
                "marks": [2026500000, 2026500064],
                "posterior": [2026600000, 2026600064],
            },
            "contaminated_quarantine": [2026083064, 2026083128],
            "contaminated_auxiliary_streams": {
                "counts": [2026400064, 2026400128],
                "count_split": [2026401064, 2026401128],
                "marks": [2026500064, 2026500128],
                "posterior": [2026600064, 2026600128],
            },
            "replacement_validation_sealed": [2026083320, 2026083576],
            "validation_opened": False,
        },
        "aggregate": {
            "member_count": len(rows),
            "mode_count": integrated.MODE_COUNT,
            "coverage68_frequency_by_mode": all68,
            "coverage95_frequency_by_mode": all95,
            "coverage68_frequency_mean": float(np.mean(all68)),
            "coverage95_frequency_mean": float(np.mean(all95)),
            "coverage68_target": TARGET_68,
            "coverage95_target": TARGET_95,
        },
        "by_arm": by_arm,
        "members": rows,
        "interpretation": {
            "purpose": "separate a wrong per-voxel coverage estimator from genuine latent-mode interval miscalibration",
            "mean_model_changed": False,
            "gate_redesign_proposal": "config/cf4_b1_gate_redesign_proposal_v1.json",
            "observational_z0_posterior": "NOT_CREATED",
            "B2_IC_FORWARD": "NOT_STARTED",
        },
        "next_action": "Use the regenerated mode coverage and pooled guard to freeze a width/gate policy; do not execute the information-null mark arm, open validation, or alter the frozen strict gate.",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_mode_coverage()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
