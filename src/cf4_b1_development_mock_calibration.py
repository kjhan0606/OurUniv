"""Local, deterministic B1 count-likelihood development calibration.

This runner is deliberately self-contained and CPU-only.  It exercises the
frozen six-population positive-count contract on synthetic N32 fields; it does
not read CF4/2M++ data, GPFS, Slurm, or validation seeds.  The output is a
development diagnostic, never an observational posterior or a resolution
claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates


GRID = 32
BOX_SIZE = 384.0
CELL_SIZE = BOX_SIZE / GRID
POPULATIONS = 6
MOCK_COUNT = 64
SEED_START = 2026083000
SEED_STOP = SEED_START + MOCK_COUNT
TRAIN_FRACTION = 0.8
POSTERIOR_DRAWS = 16
HUBBLE = 74.6
LITTLE_H = 0.746

# Published 2M++ nuisance reference values are used only as a synthetic prior
# centre.  They are not fitted to observed totals in this runner.
ALPHA = np.log(np.array([0.0112, 0.086, 0.124, 0.0104, 0.086, 0.137]))
BIAS = np.array([1.70, 1.20, 1.15, 1.74, 1.21, 1.00])
FOG_SIGMA_CELLS = np.array([0.35, 0.45, 0.55, 0.40, 0.50, 0.60])


class MockCalibrationError(ValueError):
    """A synthetic calibration contract violation."""


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = (np.arange(GRID, dtype=np.float64) + 0.5) * CELL_SIZE - BOX_SIZE / 2.0
    return np.meshgrid(axis, axis, axis, indexing="ij")


def _selection_exposure() -> np.ndarray:
    """Positive synthetic exposure with angular and radial structure."""

    xx, yy, zz = _grid()
    radius = np.sqrt(xx * xx + yy * yy + zz * zz)
    radial = np.exp(-0.5 * (radius / 185.0) ** 2)
    angular = 0.82 + 0.10 * np.sin(np.arctan2(yy, xx)) + 0.08 * (zz / (radius + 1.0))
    response = np.clip(0.25 + 0.75 * radial * angular, 0.05, None)
    return np.stack([response * (0.92 + 0.025 * population) for population in range(POPULATIONS)])


def _low_k_filter() -> np.ndarray:
    frequencies = np.fft.fftfreq(GRID, d=CELL_SIZE)
    kx, ky, kz = np.meshgrid(frequencies, frequencies, frequencies, indexing="ij")
    k = 2.0 * np.pi * np.sqrt(kx * kx + ky * ky + kz * kz)
    filt = np.exp(-0.5 * (k / 0.16) ** 2)
    filt[0, 0, 0] = 0.0
    return filt


_FILTER = _low_k_filter()
_EXPOSURE = _selection_exposure()
_XX, _YY, _ZZ = _grid()
_RELATIVE = np.stack((_XX, _YY, _ZZ), axis=-1)
_RADIUS = np.linalg.norm(_RELATIVE, axis=-1)
_RHAT = _RELATIVE / np.maximum(_RADIUS[..., None], 1.0)


def seed_schedule(index: int) -> dict[str, int]:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < MOCK_COUNT:
        raise MockCalibrationError("mock index is outside the frozen development range")
    return {
        "truth": SEED_START + index,
        "counts": 2026100000 + index,
        "selection": 2026200000 + index,
        "posterior": 2026300000 + index,
    }


def _truth_field(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    white = rng.normal(size=(GRID, GRID, GRID))
    eta = np.fft.ifftn(np.fft.fftn(white) * _FILTER).real
    eta -= np.mean(eta)
    eta /= max(float(np.std(eta)), np.finfo(np.float64).tiny)
    eta *= 0.45
    # A potential-flow velocity proxy with the same latent phase.
    grad = np.gradient(eta, CELL_SIZE, edge_order=2)
    velocity = -110.0 * np.stack(grad, axis=-1)
    return eta.astype(np.float64), velocity.astype(np.float64)


def _spherical_rsd_field(eta: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Sample eta at observer-centred spherical coherent-RSD source positions."""

    radial_velocity = np.sum(velocity * _RHAT, axis=-1)
    displacement = LITTLE_H * radial_velocity / HUBBLE
    shifted = _RELATIVE + displacement[..., None] * _RHAT
    coordinates = ((shifted + BOX_SIZE / 2.0) / CELL_SIZE - 0.5) % GRID
    return map_coordinates(eta, np.moveaxis(coordinates, -1, 0), order=1, mode="wrap")


def _positive_intensity(eta: np.ndarray, arm: str, velocity: np.ndarray) -> np.ndarray:
    if arm == "A":
        field = eta
    elif arm in ("B", "C", "D"):
        field = _spherical_rsd_field(eta, velocity)
    else:
        raise MockCalibrationError(f"unknown stress arm {arm}")
    intensity = np.exp(ALPHA[:, None, None, None] + BIAS[:, None, None, None] * field)
    intensity *= _EXPOSURE
    if arm in ("C", "D"):
        broadened = np.empty_like(intensity)
        for population, sigma in enumerate(FOG_SIGMA_CELLS):
            broadened[population] = gaussian_filter1d(
                intensity[population], sigma=float(sigma), axis=2, mode="wrap"
            )
        intensity = broadened
    if arm == "D":
        discrepancy = gaussian_filter(eta, sigma=2.0, mode="wrap")
        discrepancy -= np.mean(discrepancy)
        intensity *= np.exp(0.12 * discrepancy)[None, ...]
        intensity *= np.exp(0.08 * gaussian_filter(eta, sigma=3.0, mode="wrap"))[None, ...]
    if not np.all(np.isfinite(intensity)) or np.any(intensity <= 0.0):
        raise MockCalibrationError("synthetic intensity is not finite and positive")
    return intensity


def _draw_counts(intensity: np.ndarray, arm: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if arm != "D":
        return rng.poisson(intensity).astype(np.int64)
    # Gamma-Poisson mixture: predeclared overdispersed stress arm.
    phi = 0.35
    rate = rng.gamma(shape=1.0 / phi, scale=intensity * phi)
    return rng.poisson(rate).astype(np.int64)


def _split_counts(counts: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Poisson thinning equivalent to a row split before voxelization."""

    rng = np.random.default_rng(seed)
    train = rng.binomial(counts, TRAIN_FRACTION).astype(np.int64)
    return train, counts - train


def _estimate_eta(counts: np.ndarray, exposure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Voxelwise Poisson maximum-likelihood eta and observed-information proxy."""

    observed = counts.astype(np.float64)
    # For lambda_p=E_p exp(alpha_p+b_p eta), Newton's update is analytic and
    # uses the same positive Poisson factor as the frozen contract.  The
    # bounded iterate prevents a rare zero-count cell from leaving the finite
    # log-domain contract while retaining the unconstrained eta semantics.
    eta_hat = np.zeros(observed.shape[1:], dtype=np.float64)
    base = exposure * np.exp(ALPHA[:, None, None, None])
    for _ in range(12):
        intensity = base * np.exp(BIAS[:, None, None, None] * eta_hat[None, ...])
        gradient = np.sum(BIAS[:, None, None, None] * (observed - intensity), axis=0)
        information = np.sum(BIAS[:, None, None, None] ** 2 * intensity, axis=0)
        eta_hat += gradient / np.maximum(information, 1.0e-12)
        eta_hat = np.clip(eta_hat, -5.0, 5.0)
    intensity = base * np.exp(BIAS[:, None, None, None] * eta_hat[None, ...])
    variance = 1.0 / np.maximum(
        np.sum(BIAS[:, None, None, None] ** 2 * intensity, axis=0), 1.0e-12
    )
    # The z=0 prior is spatially correlated; a cellwise MLE would overfit the
    # sub-unity count regime.  This fixed, truth-blind smoothing is the local
    # technical proxy for that prior and is not a tunable science cutoff.
    eta_hat = gaussian_filter(eta_hat, sigma=1.25, mode="wrap")
    eta_hat -= np.mean(eta_hat)
    return eta_hat, variance


def _low_k_mask() -> np.ndarray:
    frequencies = np.fft.fftfreq(GRID, d=CELL_SIZE)
    kx, ky, kz = np.meshgrid(frequencies, frequencies, frequencies, indexing="ij")
    k = 2.0 * np.pi * np.sqrt(kx * kx + ky * ky + kz * kz)
    return (k > 0.0) & (k <= 0.10)


_LOW_K_MASK = _low_k_mask()


def _mode_metrics(truth: np.ndarray, estimate: np.ndarray, variance: np.ndarray) -> dict[str, float | bool]:
    truth_modes = np.fft.fftn(truth, norm="ortho")[_LOW_K_MASK]
    estimate_modes = np.fft.fftn(estimate, norm="ortho")[_LOW_K_MASK]
    truth_real = np.concatenate((truth_modes.real, truth_modes.imag))
    estimate_real = np.concatenate((estimate_modes.real, estimate_modes.imag))
    if truth_real.size == 0 or np.std(truth_real) == 0.0 or np.std(estimate_real) == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(truth_real, estimate_real)[0, 1])
    residual_power = float(np.sum((estimate_real - truth_real) ** 2) / max(np.sum(truth_real**2), 1e-30))
    response = float(np.dot(estimate_real, truth_real) / max(np.dot(truth_real, truth_real), 1e-30))
    sigma = np.sqrt(np.maximum(variance, 1e-12))
    coverage68 = float(np.mean(np.abs(estimate - truth) <= sigma))
    coverage95 = float(np.mean(np.abs(estimate - truth) <= 1.96 * sigma))
    return {
        "response": response,
        "correlation_r": correlation,
        "residual_power_ratio": residual_power,
        "coverage68": coverage68,
        "coverage95": coverage95,
        "strict_low_k_gate": bool(
            0.8 <= response <= 1.2
            and correlation >= 0.7
            and residual_power <= 0.5
            and abs(coverage68 - 0.6826894921370859) <= 0.05
            and abs(coverage95 - 0.9544997361036416) <= 0.025
        ),
    }


def _poisson_log_score(counts: np.ndarray, intensity: np.ndarray) -> float:
    from scipy.special import gammaln

    log_intensity = np.log(np.maximum(intensity, np.finfo(np.float64).tiny))
    return float(np.sum(counts * log_intensity - intensity - gammaln(counts + 1.0)))


def run_mock(index: int, arm: str) -> dict[str, object]:
    seeds = seed_schedule(index)
    eta, velocity = _truth_field(seeds["truth"])
    intensity = _positive_intensity(eta, arm, velocity)
    counts = _draw_counts(intensity, arm, seeds["counts"])
    train, holdout = _split_counts(counts, seeds["selection"])
    estimate, variance = _estimate_eta(train, _EXPOSURE * TRAIN_FRACTION)
    metrics = _mode_metrics(eta, estimate, variance)
    fitted_intensity = _EXPOSURE * np.exp(ALPHA[:, None, None, None] + BIAS[:, None, None, None] * estimate)
    prior_intensity = _EXPOSURE * np.exp(ALPHA[:, None, None, None])
    metrics["heldout_log_score_improvement"] = _poisson_log_score(holdout, fitted_intensity) - _poisson_log_score(holdout, prior_intensity)
    metrics["count_total"] = int(np.sum(counts))
    metrics["positive_support_fraction"] = float(np.mean(intensity > 0.0))
    return {"index": index, "arm": arm, "seed": seeds, "metrics": metrics}


def run_calibration() -> dict[str, object]:
    arms = ("A", "B", "C", "D")
    members = [run_mock(index, arms[index // 16]) for index in range(MOCK_COUNT)]
    by_arm: dict[str, dict[str, object]] = {}
    for arm in arms:
        rows = [row for row in members if row["arm"] == arm]
        metrics = {name: np.asarray([row["metrics"][name] for row in rows], dtype=np.float64) for name in (
            "response", "correlation_r", "residual_power_ratio", "coverage68", "coverage95", "heldout_log_score_improvement", "count_total", "positive_support_fraction"
        )}
        by_arm[arm] = {
            "member_count": len(rows),
            "seed_range": [SEED_START + 16 * arms.index(arm), SEED_START + 16 * (arms.index(arm) + 1) - 1],
            "metric_median": {name: float(np.median(values)) for name, values in metrics.items()},
            "metric_min": {name: float(np.min(values)) for name, values in metrics.items()},
            "metric_max": {name: float(np.max(values)) for name, values in metrics.items()},
            "strict_low_k_gate_pass_count": int(sum(bool(row["metrics"]["strict_low_k_gate"]) for row in rows)),
            "heldout_positive_count": int(np.count_nonzero(metrics["heldout_log_score_improvement"] > 0.0)),
        }
    strict_pass = sum(item["strict_low_k_gate_pass_count"] for item in by_arm.values())
    return {
        "schema": "ouruniv-cf4-b1-development-mock-calibration-result-v1",
        "status": "COMPLETE_64_DEVELOPMENT_MOCKS_TECHNICAL_NO_SCIENCE_CLAIM",
        "grid": {"N": GRID, "box_size_cMpc_h": BOX_SIZE, "cell_size_cMpc_h": CELL_SIZE},
        "seed_firewall": {
            "development_count": MOCK_COUNT,
            "seed_start_inclusive": SEED_START,
            "seed_stop_exclusive": SEED_STOP,
            "untouched_validation_seed_start": 2026083064,
            "untouched_validation_seed_stop_exclusive": 2026083320,
            "validation_opened": False,
        },
        "arms": by_arm,
        "aggregate": {
            "member_count": MOCK_COUNT,
            "strict_low_k_gate_pass_count": strict_pass,
            "all_members_positive_support": all(
                item["metric_min"]["positive_support_fraction"] == 1.0 for item in by_arm.values()
            ),
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
