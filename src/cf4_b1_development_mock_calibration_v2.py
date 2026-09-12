"""Corrected B1 development calibration with a complete local joint harness.

Version 1 is retained as an immutable count-only smoke diagnostic.  This
version fixes train-conditioned holdout scaling and obtains coverage from the
declared finite posterior-draw count after applying the same fixed correlated
prior smoothing to every draw.  The joint interface probes run once before
the 64 development members.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter

import cf4_b1_development_mock_calibration as base
from cf4_b1_joint_development_harness import (
    run_joint_harness,
    source_bound_joint_factor_probe,
    source_bound_joint_score,
)


POSTERIOR_DRAWS = 16
PRIOR_SMOOTHING_SIGMA_CELLS = 1.25


def posterior_draws(
    posterior_mean: np.ndarray, local_variance: np.ndarray, seed: int
) -> np.ndarray:
    """Draw a finite Gaussian technical posterior with correlated noise."""

    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(POSTERIOR_DRAWS,) + posterior_mean.shape)
    scale = np.sqrt(np.maximum(local_variance, 1.0e-12))
    draws = np.empty_like(raw)
    for index in range(POSTERIOR_DRAWS):
        noise = gaussian_filter(raw[index] * scale, sigma=PRIOR_SMOOTHING_SIGMA_CELLS, mode="wrap")
        draw = posterior_mean + noise
        draws[index] = draw - np.mean(draw)
    if not np.all(np.isfinite(draws)):
        raise base.MockCalibrationError("posterior draws are not finite")
    return draws


def _estimate_eta(counts: np.ndarray, exposure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Poisson voxel MLE with a fixed, truth-blind correlated-prior proxy."""

    observed = counts.astype(np.float64)
    eta_hat = np.zeros(observed.shape[1:], dtype=np.float64)
    base_rate = exposure * np.exp(base.ALPHA[:, None, None, None])
    for _ in range(12):
        intensity = base_rate * np.exp(base.BIAS[:, None, None, None] * eta_hat[None, ...])
        gradient = np.sum(base.BIAS[:, None, None, None] * (observed - intensity), axis=0)
        information = np.sum(base.BIAS[:, None, None, None] ** 2 * intensity, axis=0)
        eta_hat += gradient / np.maximum(information, 1.0e-12)
        eta_hat = np.clip(eta_hat, -5.0, 5.0)
    intensity = base_rate * np.exp(base.BIAS[:, None, None, None] * eta_hat[None, ...])
    local_variance = 1.0 / np.maximum(
        np.sum(base.BIAS[:, None, None, None] ** 2 * intensity, axis=0), 1.0e-12
    )
    eta_hat = gaussian_filter(eta_hat, sigma=PRIOR_SMOOTHING_SIGMA_CELLS, mode="wrap")
    eta_hat -= np.mean(eta_hat)
    return eta_hat, local_variance


def _fit_population_nuisance(
    counts: np.ndarray, exposure: np.ndarray, eta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Profile six alpha and six positive-bias nuisances on the train split."""

    observed = counts.astype(np.float64)
    alpha = base.ALPHA.copy()
    logbias = np.log(base.BIAS)
    for population in range(6):
        n = observed[population]
        e = exposure[population]
        theta = np.array([alpha[population], logbias[population]], dtype=np.float64)
        for _ in range(8):
            bias = float(np.exp(theta[1]))
            lam = e * np.exp(theta[0] + bias * eta)
            residual = n - lam
            g = np.array([np.sum(residual), np.sum(residual * bias * eta)])
            h00 = -np.sum(lam)
            h01 = -np.sum(lam * bias * eta)
            h11 = np.sum(residual * bias * eta) - np.sum(lam * (bias * eta) ** 2)
            hessian = np.array([[h00, h01], [h01, h11]], dtype=np.float64)
            try:
                step = np.linalg.solve(hessian, g)
            except np.linalg.LinAlgError:
                break
            theta -= step
            theta[0] = np.clip(theta[0], -8.0, 4.0)
            theta[1] = np.clip(theta[1], -2.0, 2.0)
            if np.max(np.abs(step)) < 1.0e-7:
                break
        alpha[population], logbias[population] = theta
    if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(logbias)):
        raise base.MockCalibrationError("profiled nuisance parameters are not finite")
    return alpha, np.exp(logbias), 12


def _metrics(
    truth: np.ndarray,
    posterior_mean: np.ndarray,
    draws: np.ndarray,
) -> dict[str, float | bool]:
    mask = base._LOW_K_MASK
    truth_modes = np.fft.fftn(truth, norm="ortho")[mask]
    mean_modes = np.fft.fftn(posterior_mean, norm="ortho")[mask]
    truth_vector = np.concatenate((truth_modes.real, truth_modes.imag))
    mean_vector = np.concatenate((mean_modes.real, mean_modes.imag))
    correlation = float(np.corrcoef(truth_vector, mean_vector)[0, 1]) if np.std(mean_vector) > 0 else 0.0
    response = float(np.dot(mean_vector, truth_vector) / max(np.dot(truth_vector, truth_vector), 1.0e-30))
    residual = float(np.sum((mean_vector - truth_vector) ** 2) / max(np.sum(truth_vector**2), 1.0e-30))
    sample_std = np.std(draws, axis=0, ddof=1)
    coverage68 = float(np.mean(np.abs(posterior_mean - truth) <= sample_std))
    coverage95 = float(np.mean(np.abs(posterior_mean - truth) <= 1.96 * sample_std))
    return {
        "response": response,
        "correlation_r": correlation,
        "residual_power_ratio": residual,
        "coverage68": coverage68,
        "coverage95": coverage95,
        "posterior_draw_count": int(draws.shape[0]),
        "strict_low_k_gate": bool(
            0.8 <= response <= 1.2
            and correlation >= 0.7
            and residual <= 0.5
            and abs(coverage68 - 0.6826894921370859) <= 0.05
            and abs(coverage95 - 0.9544997361036416) <= 0.025
        ),
    }


def run_mock(index: int, arm: str) -> dict[str, object]:
    seeds = base.seed_schedule(index)
    eta, velocity = base._truth_field(seeds["truth"])
    intensity = base._positive_intensity(eta, arm, velocity)
    counts = base._draw_counts(intensity, arm, seeds["counts"])
    train, holdout = base._split_counts(counts, seeds["selection"])
    posterior_mean, local_variance = _estimate_eta(train, base._EXPOSURE * base.TRAIN_FRACTION)
    draws = posterior_draws(posterior_mean, local_variance, seeds["posterior"])
    metrics = _metrics(eta, posterior_mean, draws)
    fitted_alpha, fitted_bias, nuisance_parameter_count = _fit_population_nuisance(
        train, base._EXPOSURE * base.TRAIN_FRACTION, posterior_mean
    )
    # A thinned Poisson holdout has expected intensity (1-p)*lambda.  The
    # previous smoke incorrectly scored it against the full lambda.
    holdout_exposure = base._EXPOSURE * (1.0 - base.TRAIN_FRACTION)
    fitted = holdout_exposure * np.exp(
        fitted_alpha[:, None, None, None] + fitted_bias[:, None, None, None] * posterior_mean
    )
    prior = holdout_exposure * np.exp(base.ALPHA[:, None, None, None])
    metrics["heldout_log_score_improvement"] = base._poisson_log_score(holdout, fitted) - base._poisson_log_score(holdout, prior)
    metrics["profiled_nuisance_parameter_count"] = nuisance_parameter_count
    metrics["profiled_alpha_median_abs_error"] = float(np.median(np.abs(fitted_alpha - base.ALPHA)))
    metrics["profiled_bias_median_abs_error"] = float(np.median(np.abs(fitted_bias - base.BIAS)))
    joint_intensity = base._EXPOSURE * base.TRAIN_FRACTION * np.exp(
        fitted_alpha[:, None, None, None] + fitted_bias[:, None, None, None] * posterior_mean
    )
    joint_score = source_bound_joint_score(train, joint_intensity, seeds["posterior"])
    metrics["joint_log_likelihood_finite"] = bool(joint_score["finite"])
    metrics["joint_secure_rows"] = int(joint_score["secure_rows"])
    metrics["count_total"] = int(np.sum(counts))
    metrics["positive_support_fraction"] = float(np.mean(intensity > 0.0))
    return {"index": index, "arm": arm, "seed": seeds, "metrics": metrics}


def run_calibration() -> dict[str, object]:
    joint_harness = run_joint_harness()
    if joint_harness["status"] != "PASS":
        raise base.MockCalibrationError("joint development harness did not pass")
    arms = ("A", "B", "C", "D")
    members = [run_mock(index, arms[index // 16]) for index in range(base.MOCK_COUNT)]
    by_arm: dict[str, dict[str, object]] = {}
    names = (
        "response", "correlation_r", "residual_power_ratio", "coverage68",
        "coverage95", "heldout_log_score_improvement", "count_total", "positive_support_fraction",
        "profiled_alpha_median_abs_error", "profiled_bias_median_abs_error",
        "joint_secure_rows",
    )
    for arm in arms:
        rows = [row for row in members if row["arm"] == arm]
        arrays = {name: np.asarray([row["metrics"][name] for row in rows], dtype=np.float64) for name in names}
        by_arm[arm] = {
            "member_count": len(rows),
            "seed_range": [base.SEED_START + 16 * arms.index(arm), base.SEED_START + 16 * (arms.index(arm) + 1) - 1],
            "metric_median": {name: float(np.median(values)) for name, values in arrays.items()},
            "metric_min": {name: float(np.min(values)) for name, values in arrays.items()},
            "metric_max": {name: float(np.max(values)) for name, values in arrays.items()},
            "strict_low_k_gate_pass_count": int(sum(bool(row["metrics"]["strict_low_k_gate"]) for row in rows)),
            "heldout_positive_count": int(np.count_nonzero(arrays["heldout_log_score_improvement"] > 0.0)),
            "posterior_draw_count": POSTERIOR_DRAWS,
            "nuisance_parameter_count": 12,
        }
    # The source-bound joint factor is evaluated on the same canonical
    # crossmatch once per bundle.  It is deliberately a score/interface gate,
    # not an observational CF4 posterior.
    joint_factor = source_bound_joint_factor_probe()
    return {
        "schema": "ouruniv-cf4-b1-development-mock-calibration-result-v2",
        "status": "COMPLETE_64_DEVELOPMENT_MOCKS_JOINT_HARNESS_TECHNICAL_NO_SCIENCE_CLAIM",
        "predecessor": "config/cf4_bundle_b1_development_mock_calibration_result_v1.json",
        "grid": {"N": base.GRID, "box_size_cMpc_h": base.BOX_SIZE, "cell_size_cMpc_h": base.CELL_SIZE},
        "seed_firewall": {
            "development_count": base.MOCK_COUNT,
            "seed_start_inclusive": base.SEED_START,
            "seed_stop_exclusive": base.SEED_STOP,
            "untouched_validation_seed_start": 2026083320,
            "untouched_validation_seed_stop_exclusive": 2026083576,
            "validation_opened": False,
        },
        "joint_harness": joint_harness,
        "joint_factor_score_probe": joint_factor,
        "inference_semantics": {
            "holdout_score": "train-conditioned thinned-Poisson intensity with (1-training_fraction) exposure",
            "posterior_draws": POSTERIOR_DRAWS,
            "draw_noise": "fixed correlated-prior smoothing applied to every draw; sample std drives coverage",
            "truth_blind_smoothing": True,
        },
        "arms": by_arm,
        "aggregate": {
            "member_count": base.MOCK_COUNT,
            "strict_low_k_gate_pass_count": sum(item["strict_low_k_gate_pass_count"] for item in by_arm.values()),
            "all_members_positive_support": all(item["metric_min"]["positive_support_fraction"] == 1.0 for item in by_arm.values()),
            "all_arms_have_heldout_gain": all(item["heldout_positive_count"] > 0 for item in by_arm.values()),
        },
        "scientific_disposition": {
            "development_calibration": "TECHNICAL_RESULT_ONLY",
            "observational_z0_posterior": "NOT_CREATED",
            "KF_EXPAND": "NOT_AUTHORIZED",
            "untouched_256_validation": "NOT_RUN",
            "frontier_promotion": "NOT_ALLOWED",
            "0p3_cMpc_h_claim": "NOT_ALLOWED",
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_calibration()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
