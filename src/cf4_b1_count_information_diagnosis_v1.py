"""Development-only count-side expected-vs-empirical information diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

import cf4_b1_integrated_joint_calibration_v3 as integrated


ARM_ORDER = ("A", "B", "C", "D")
SAMPLE_INDICES = tuple(index for start in (0, 16, 32, 48) for index in range(start, start + 4))
FINITE_DIFFERENCE_STEP = 1.0e-4


def _count_derivatives(coeff: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray]:
    centre_field = integrated._field_from_coeff(coeff)
    centre = (
        integrated.base._positive_intensity(
            centre_field, arm, integrated._velocity_from_field(centre_field)
        ) * integrated.base.TRAIN_FRACTION
    )
    derivatives = []
    for mode in range(integrated.MODE_COUNT):
        direction = np.eye(integrated.MODE_COUNT, dtype=np.float64)[mode] * FINITE_DIFFERENCE_STEP
        plus_field = integrated._field_from_coeff(coeff + direction)
        minus_field = integrated._field_from_coeff(coeff - direction)
        plus = (
            integrated.base._positive_intensity(
                plus_field, arm, integrated._velocity_from_field(plus_field)
            ) * integrated.base.TRAIN_FRACTION
        )
        minus = (
            integrated.base._positive_intensity(
                minus_field, arm, integrated._velocity_from_field(minus_field)
            ) * integrated.base.TRAIN_FRACTION
        )
        derivatives.append(((plus - minus) / (2.0 * FINITE_DIFFERENCE_STEP)).ravel())
    return centre.ravel(), np.asarray(derivatives, dtype=np.float64)


def diagnose_member(index: int, arm: str) -> dict[str, object]:
    member = integrated.run_mock(index, arm)
    seeds = member["seed"]
    truth = integrated._field_from_coeff(member["_truth_coeff"])
    all_counts = integrated.base._draw_counts(
        integrated.base._positive_intensity(
            truth, arm, integrated._velocity_from_field(truth)
        ),
        arm,
        seeds["counts"],
    )
    train, _ = integrated.base._split_counts(all_counts, seeds["count_split"])
    centre, derivatives = _count_derivatives(member["_estimate_coeff"], arm)
    train_flat = train.ravel().astype(np.float64)
    standardized_residual = (train_flat - centre) / np.maximum(centre, 1.0e-12)
    score = derivatives * standardized_residual[None, :]
    empirical_score_covariance = score @ score.T
    expected_information = integrated._count_fisher(member["_estimate_coeff"], arm)
    inverse_expected = np.linalg.pinv(expected_information)
    sandwich = inverse_expected @ empirical_score_covariance @ inverse_expected
    expected_sigma = np.sqrt(np.maximum(np.diag(inverse_expected), 0.0))
    sandwich_sigma = np.sqrt(np.maximum(np.diag(sandwich), 0.0))
    ratio = sandwich_sigma / np.maximum(expected_sigma, 1.0e-15)
    return {
        "index": index,
        "arm": arm,
        "seed": seeds,
        "expected_information_trace": float(np.trace(expected_information)),
        "empirical_score_covariance_trace": float(np.trace(empirical_score_covariance)),
        "sandwich_to_expected_sigma_by_mode": ratio.tolist(),
        "sandwich_to_expected_sigma_median": float(np.median(ratio)),
        "sandwich_to_expected_sigma_min": float(np.min(ratio)),
        "sandwich_to_expected_sigma_max": float(np.max(ratio)),
    }


def run_diagnosis() -> dict[str, object]:
    rows = [diagnose_member(index, ARM_ORDER[index // 16]) for index in SAMPLE_INDICES]
    by_arm = {}
    for arm in ARM_ORDER:
        values = np.asarray(
            [row["sandwich_to_expected_sigma_by_mode"] for row in rows if row["arm"] == arm],
            dtype=np.float64,
        )
        by_arm[arm] = {
            "member_count": int(values.shape[0]),
            "mode_count": int(values.shape[1]),
            "median_by_mode": np.median(values, axis=0).tolist(),
            "median_over_modes_and_members": float(np.median(values)),
            "fraction_modes_above_1p1": float(np.mean(values > 1.1)),
            "fraction_modes_below_0p9": float(np.mean(values < 0.9)),
        }
    source = Path(integrated.__file__).resolve()
    return {
        "schema": "ouruniv-cf4-b1-count-information-diagnosis-result-v1",
        "status": "COMPLETE_DEVELOPMENT_ONLY_NO_SCIENCE_CLAIM",
        "source_artifact": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "definition": {
            "expected_information": "declared count Fisher at MAP",
            "empirical_information": "outer product of per-voxel count scores at MAP",
            "sandwich_covariance": "A^-1 B A^-1",
            "finite_difference_step": FINITE_DIFFERENCE_STEP,
            "sample_indices": list(SAMPLE_INDICES),
            "member_count": len(rows),
            "validation_opened": False,
        },
        "by_arm": by_arm,
        "members": rows,
        "interpretation": {
            "purpose": "test whether A/C overcoverage is a count-side expected-information mismatch before adding mark-side misspecification",
            "mean_model_changed": False,
            "observational_z0_posterior": "NOT_CREATED",
            "B2_IC_FORWARD": "NOT_STARTED",
        },
        "next_action": "Use the arm-stratified sandwich ratios to decide whether a count-side covariance repair is justified; keep validation sealed.",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_diagnosis()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["by_arm"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
