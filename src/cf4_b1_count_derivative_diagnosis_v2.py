"""Development-only independent count-information and z-covariance diagnosis."""

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
DERIVATIVE_STEPS = (1.0e-3, 1.0e-4, 1.0e-5)
MODE_RESULT = Path("config/cf4_b1_mode_coverage_diagnosis_result_v3.json")


def _count_derivative_fisher(coeff: np.ndarray, arm: str, step: float) -> np.ndarray:
    centre_field = integrated._field_from_coeff(coeff)
    centre = integrated.base._positive_intensity(
        centre_field, arm, integrated._velocity_from_field(centre_field)
    ) * integrated.base.TRAIN_FRACTION
    derivatives = []
    for mode in range(integrated.MODE_COUNT):
        direction = np.eye(integrated.MODE_COUNT, dtype=np.float64)[mode] * step
        plus_field = integrated._field_from_coeff(coeff + direction)
        minus_field = integrated._field_from_coeff(coeff - direction)
        plus = integrated.base._positive_intensity(
            plus_field, arm, integrated._velocity_from_field(plus_field)
        ) * integrated.base.TRAIN_FRACTION
        minus = integrated.base._positive_intensity(
            minus_field, arm, integrated._velocity_from_field(minus_field)
        ) * integrated.base.TRAIN_FRACTION
        derivatives.append(((plus - minus) / (2.0 * step)).ravel())
    derivative = np.asarray(derivatives)
    variance = centre if arm != "D" else centre + integrated.D_OVERDISPERSION_PHI * centre**2
    weighted = derivative / np.sqrt(np.maximum(variance.ravel(), 1.0e-12))[None, :]
    return weighted @ weighted.T


def diagnose_member(index: int, arm: str) -> dict[str, object]:
    member = integrated.run_mock(index, arm)
    truth = integrated._field_from_coeff(member["_truth_coeff"])
    seeds = member["seed"]
    all_counts = integrated.base._draw_counts(
        integrated.base._positive_intensity(truth, arm, integrated._velocity_from_field(truth)),
        arm,
        seeds["counts"],
    )
    train, _ = integrated.base._split_counts(all_counts, seeds["count_split"])
    coeff = member["_estimate_coeff"]
    centre_field = integrated._field_from_coeff(coeff)
    centre = (
        integrated.base._positive_intensity(
            centre_field, arm, integrated._velocity_from_field(centre_field)
        ) * integrated.base.TRAIN_FRACTION
    ).ravel()
    variance = centre if arm != "D" else centre + integrated.D_OVERDISPERSION_PHI * centre**2
    deriv_fishers = {str(step): _count_derivative_fisher(coeff, arm, step) for step in DERIVATIVE_STEPS}
    empirical_derivative = []
    step = DERIVATIVE_STEPS[1]
    for mode in range(integrated.MODE_COUNT):
        direction = np.eye(integrated.MODE_COUNT, dtype=np.float64)[mode] * step
        plus_field = integrated._field_from_coeff(coeff + direction)
        minus_field = integrated._field_from_coeff(coeff - direction)
        plus = integrated.base._positive_intensity(
            plus_field, arm, integrated._velocity_from_field(plus_field)
        ) * integrated.base.TRAIN_FRACTION
        minus = integrated.base._positive_intensity(
            minus_field, arm, integrated._velocity_from_field(minus_field)
        ) * integrated.base.TRAIN_FRACTION
        empirical_derivative.append(((plus - minus) / (2.0 * step)).ravel())
    derivative = np.asarray(empirical_derivative)
    standardized_residual = (train.ravel().astype(np.float64) - centre) / np.sqrt(np.maximum(variance, 1.0e-12))
    score = derivative / np.sqrt(np.maximum(variance, 1.0e-12))[None, :] * standardized_residual[None, :]
    empirical_score_covariance = score @ score.T
    expected = deriv_fishers[str(step)]
    inv_expected = np.linalg.pinv(expected)
    sandwich = inv_expected @ empirical_score_covariance @ inv_expected
    expected_sigma = np.sqrt(np.maximum(np.diag(inv_expected), 0.0))
    sandwich_sigma = np.sqrt(np.maximum(np.diag(sandwich), 0.0))
    return {
        "index": index,
        "arm": arm,
        "seed": seeds,
        "derivative_fisher_trace_by_step": {key: float(np.trace(value)) for key, value in deriv_fishers.items()},
        "relative_fisher_difference_1e-3_vs_1e-4": float(
            np.max(np.abs(deriv_fishers["0.001"] - deriv_fishers["0.0001"]))
            / np.maximum(np.max(np.abs(deriv_fishers["0.0001"])), 1.0e-30)
        ),
        "relative_fisher_difference_1e-5_vs_1e-4": float(
            np.max(np.abs(deriv_fishers["1e-05"] - deriv_fishers["0.0001"]))
            / np.maximum(np.max(np.abs(deriv_fishers["0.0001"])), 1.0e-30)
        ),
        "sandwich_to_expected_sigma_by_mode": (
            sandwich_sigma / np.maximum(expected_sigma, 1.0e-15)
        ).tolist(),
    }


def run_diagnosis() -> dict[str, object]:
    rows = [diagnose_member(index, ARM_ORDER[index // 16]) for index in SAMPLE_INDICES]
    by_arm = {}
    for arm in ARM_ORDER:
        subset = [row for row in rows if row["arm"] == arm]
        ratios = np.asarray([row["sandwich_to_expected_sigma_by_mode"] for row in subset])
        by_arm[arm] = {
            "member_count": len(subset),
            "median_sandwich_to_expected_sigma": float(np.median(ratios)),
            "max_step_ladder_relative_difference": float(max(
                max(row["relative_fisher_difference_1e-3_vs_1e-4"], row["relative_fisher_difference_1e-5_vs_1e-4"])
                for row in subset
            )),
        }
    mode_path = Path(MODE_RESULT)
    mode_payload = json.loads(mode_path.read_text(encoding="utf-8"))
    z = {}
    for arm in ARM_ORDER:
        subset = [row for row in mode_payload["members"] if row["arm"] == arm]
        errors = np.asarray([row["estimate_coeff"] for row in subset]) - np.asarray(
            [row["truth_coeff"] for row in subset]
        )
        sigmas = np.asarray([row["posterior_sigma"] for row in subset])
        z[arm] = {
            "member_count": len(subset),
            "z_sd_by_mode": np.std(errors / sigmas, axis=0, ddof=1).tolist(),
            "z_sd_mean": float(np.mean(np.std(errors / sigmas, axis=0, ddof=1))),
        }
    source = Path(integrated.__file__).resolve()
    return {
        "schema": "ouruniv-cf4-b1-count-derivative-diagnosis-result-v2",
        "status": "COMPLETE_DEVELOPMENT_ONLY_NO_SCIENCE_CLAIM",
        "source_artifact": {"path": str(source), "bytes": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "dependency_artifact": {"path": str(Path(integrated.base.__file__).resolve()), "bytes": Path(integrated.base.__file__).stat().st_size, "sha256": hashlib.sha256(Path(integrated.base.__file__).read_bytes()).hexdigest()},
        "mode_result_input": {"path": str(mode_path), "bytes": mode_path.stat().st_size, "sha256": hashlib.sha256(mode_path.read_bytes()).hexdigest()},
        "definition": {
            "purpose": "test derivative-scale stability and direct member z-covariance before mark-side stress",
            "derivative_steps": list(DERIVATIVE_STEPS),
            "empirical_variance": "count centre + D phi*centre^2 in score standardization",
            "sample_indices": list(SAMPLE_INDICES),
            "validation_opened": False,
        },
        "by_arm": by_arm,
        "direct_member_z_covariance": z,
        "members": rows,
        "interpretation": {
            "mean_model_changed": False,
            "mark_side_arm_executed": False,
            "mark_side_arm_status": "REJECTED_INFORMATION_NULL_BY_OPUS_V14",
            "observational_z0_posterior": "NOT_CREATED",
            "B2_IC_FORWARD": "NOT_STARTED",
        },
        "next_action": "Use derivative stability and direct z covariance to set a frozen width policy; do not run the information-null mark arm.",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_diagnosis()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"by_arm": result["by_arm"], "direct_member_z_covariance": result["direct_member_z_covariance"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
