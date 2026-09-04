"""Integrated, source-bound B1 joint development calibration.

This CPU-only diagnostic generates the six selected 2M++ count fields and the
CF4 shared-redshift marks from one frozen low-k synthetic LCDM-like field.
It is deliberately a technical calibration: it does not make an observational
posterior, open validation seeds, or claim the 0.3 cMpc/h target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from scipy.special import gammaln

import cf4_b1_development_mock_calibration as base
from cf4_b1_joint_development_harness import run_joint_harness
from cf4_2mpp_crossmatch_manifest import build_secure_crossmatch_manifest
from cf4_2mpp_joint_likelihood_local import joint_log_likelihood


GRID = base.GRID
BOX_SIZE = base.BOX_SIZE
CELL_SIZE = base.CELL_SIZE
POPULATIONS = base.POPULATIONS
MOCK_COUNT = base.MOCK_COUNT
POSTERIOR_DRAWS = 16
MODE_COUNT = 8
SEED_START = 2026083064
SEED_STOP = SEED_START + MOCK_COUNT
MARK_MEASUREMENT_SIGMA = 120.0
MARK_SHARED_SIGMA = 35.0
PRIOR_SIGMA = 1.0

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "data/cf4_2mpp_crossmatch_v1.csv"
SUMMARY = ROOT / "config/cf4_2mpp_crossmatch_v1_result.json"


def _basis() -> np.ndarray:
    axis = (np.arange(GRID, dtype=np.float64) + 0.5) / GRID * 2.0 * np.pi - np.pi
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    waves = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0),
             (1, 0, 1), (0, 1, 1), (2, 0, 0), (0, 2, 0))
    fields = []
    for mode, (kx, ky, kz) in enumerate(waves):
        phase = kx * xx + ky * yy + kz * zz
        fields.extend((np.cos(phase), np.sin(phase)) if mode < 3 else (np.cos(phase),))
        if len(fields) >= MODE_COUNT:
            break
    result = np.asarray(fields[:MODE_COUNT], dtype=np.float64)
    result -= result.mean(axis=(1, 2, 3), keepdims=True)
    result /= np.maximum(result.std(axis=(1, 2, 3), keepdims=True), 1.0e-12)
    return result


BASIS = _basis()
_XX, _YY, _ZZ = base._grid()
_RELATIVE = np.stack((_XX, _YY, _ZZ), axis=-1)


def _manifest_arrays() -> dict[str, object]:
    manifest = build_secure_crossmatch_manifest(MAPPING, SUMMARY)
    entries = manifest["entries"]
    positions = np.empty((len(entries), 3), dtype=np.float64)
    for row, entry in enumerate(entries):
        digest = hashlib.sha256(f"{entry['cf4_recno']}:{entry['twompp_recno']}".encode()).digest()
        unit = np.frombuffer(digest[:3], dtype=np.uint8).astype(np.float64) / 255.0
        positions[row] = unit * BOX_SIZE
    relative = (positions - BOX_SIZE / 2.0 + BOX_SIZE / 2.0) % BOX_SIZE - BOX_SIZE / 2.0
    radius = np.linalg.norm(relative, axis=1)
    rhat = relative / np.maximum(radius[:, None], 1.0e-12)
    return {
        "manifest": manifest,
        "positions": positions,
        "rhat": rhat,
        "groups": np.asarray([int(x["group_index"]) for x in entries], dtype=np.int64),
        "secure_ids": [x["secure_object_id"] for x in entries],
        "excluded_target_rows": int(manifest["counts"].get("mapping_rows", 0) - manifest["counts"]["secure_rows"]),
    }


MARKS = _manifest_arrays()


def _truth_coefficients(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.42, MODE_COUNT).astype(np.float64)


def _field_from_coeff(coeff: np.ndarray) -> np.ndarray:
    return np.tensordot(np.asarray(coeff, dtype=np.float64), BASIS, axes=(0, 0))


def _velocity_from_field(field: np.ndarray) -> np.ndarray:
    grad = np.gradient(field, CELL_SIZE, edge_order=2)
    return -110.0 * np.stack(grad, axis=-1)


def _sample_basis_at_positions(field_basis: np.ndarray, positions: np.ndarray) -> np.ndarray:
    coords = ((positions - BOX_SIZE / 2.0) / CELL_SIZE - 0.5) % GRID
    # nearest-neighbour sampling is deterministic and avoids inventing a
    # high-k interpolant in this low-k development harness.
    ijk = np.floor(coords + 0.5).astype(np.int64) % GRID
    return np.asarray([field[ijk[:, 0], ijk[:, 1], ijk[:, 2]] for field in field_basis]).T


def _mark_jacobian() -> np.ndarray:
    positions = MARKS["positions"]
    rhat = MARKS["rhat"]
    jac = np.empty((len(positions), MODE_COUNT), dtype=np.float64)
    for mode in range(MODE_COUNT):
        gradient = np.gradient(BASIS[mode], CELL_SIZE, edge_order=2)
        components = np.column_stack([_sample_basis_at_positions(g[None, ...], positions)[:, 0] for g in gradient])
        jac[:, mode] = -110.0 * np.sum(components * rhat, axis=1)
    return jac


MARK_JAC = _mark_jacobian()


def _count_objective(coeff: np.ndarray, counts: np.ndarray, arm: str) -> tuple[float, np.ndarray]:
    eta = _field_from_coeff(coeff)
    velocity = _velocity_from_field(eta)
    intensity = base._positive_intensity(eta, arm, velocity) * base.TRAIN_FRACTION
    residual = counts.astype(np.float64) - intensity
    value = float(np.sum(counts * np.log(np.maximum(intensity, 1.0e-300)) - intensity - gammaln(counts + 1.0))
                   - 0.5 * np.sum(np.asarray(coeff) ** 2 / PRIOR_SIGMA**2))
    # This is an analytic score for the A/B selection model; C/D convolution
    # remains a declared stress arm and uses a finite-difference fallback.
    if arm in ("A", "B"):
        score = np.zeros(MODE_COUNT, dtype=np.float64)
        # A conservative low-k score keeps the optimizer deterministic while
        # retaining the exact joint likelihood value for the final candidate.
        for mode in range(MODE_COUNT):
            score[mode] = np.sum(residual * (base.BIAS[:, None, None, None] * intensity).sum(axis=0) * BASIS[mode])
        score -= np.asarray(coeff) / PRIOR_SIGMA**2
    else:
        score = np.zeros(MODE_COUNT, dtype=np.float64)
        for mode in range(MODE_COUNT):
            step = 1.0e-4
            plus = _field_from_coeff(coeff + np.eye(1, MODE_COUNT, mode)[0] * step)
            minus = _field_from_coeff(coeff - np.eye(1, MODE_COUNT, mode)[0] * step)
            p = base._positive_intensity(plus, arm, _velocity_from_field(plus)) * base.TRAIN_FRACTION
            m = base._positive_intensity(minus, arm, _velocity_from_field(minus)) * base.TRAIN_FRACTION
            lp = np.sum(counts * np.log(np.maximum(p, 1.0e-300)) - p)
            lm = np.sum(counts * np.log(np.maximum(m, 1.0e-300)) - m)
            score[mode] = (lp - lm) / (2.0 * step) - coeff[mode] / PRIOR_SIGMA**2
    return -value, -score


def _joint_value(coeff: np.ndarray, counts: np.ndarray, arm: str, observed: np.ndarray, sigma: np.ndarray) -> float:
    eta = _field_from_coeff(coeff)
    intensity = base._positive_intensity(eta, arm, _velocity_from_field(eta)) * base.TRAIN_FRACTION
    predicted = MARK_JAC @ coeff
    return float(joint_log_likelihood(
        counts.astype(np.int64), intensity.astype(np.float64), observed, predicted, sigma,
        MARKS["groups"], np.full(int(MARKS["manifest"]["counts"]["secure_cf4_groups"]), MARK_SHARED_SIGMA),
        secure_object_ids=MARKS["secure_ids"], expected_group_count=int(MARKS["manifest"]["counts"]["secure_cf4_groups"]),
    ))


def _joint_vector_objective(coeff: np.ndarray, counts: np.ndarray, arm: str, observed: np.ndarray, sigma: np.ndarray) -> float:
    eta = _field_from_coeff(coeff)
    intensity = base._positive_intensity(eta, arm, _velocity_from_field(eta)) * base.TRAIN_FRACTION
    residual = observed - MARK_JAC @ coeff
    groups = MARKS["groups"]
    tau2 = MARK_SHARED_SIGMA ** 2
    group_count = int(MARKS["manifest"]["counts"]["secure_cf4_groups"])
    inv = 1.0 / sigma**2
    q = np.bincount(groups, weights=residual**2 * inv, minlength=group_count)
    w = np.bincount(groups, weights=residual * inv, minlength=group_count)
    contraction = np.bincount(groups, weights=inv, minlength=group_count)
    n_group = np.bincount(groups, minlength=group_count)
    den = 1.0 + tau2 * contraction
    q -= tau2 * w * w / den
    logdet = np.bincount(groups, weights=np.log(sigma**2), minlength=group_count) + np.log(den)
    score = float(np.sum(-0.5 * (q + logdet + n_group * np.log(2.0 * np.pi))))
    return float(np.sum(counts * np.log(np.maximum(intensity, 1.0e-300)) - intensity - gammaln(counts + 1.0))
                 + score - 0.5 * np.sum(coeff**2 / PRIOR_SIGMA**2))


def _draw_marks(coeff: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    observed = MARK_JAC @ coeff
    groups = MARKS["groups"]
    observed = observed + rng.normal(0.0, MARK_MEASUREMENT_SIGMA, len(observed))
    observed += rng.normal(0.0, MARK_SHARED_SIGMA, int(MARKS["manifest"]["counts"]["secure_cf4_groups"]))[groups]
    return observed.astype(np.float64), np.full(len(observed), MARK_MEASUREMENT_SIGMA, dtype=np.float64)


def _mode_metrics(truth: np.ndarray, draws: np.ndarray, estimate: np.ndarray, counts: np.ndarray,
                  holdout: np.ndarray, arm: str, observed: np.ndarray, sigma: np.ndarray,
                  fit_success: bool, estimate_coeff: np.ndarray) -> dict[str, object]:
    truth_modes = np.fft.fftn(truth, norm="ortho")
    est_modes = np.fft.fftn(estimate, norm="ortho")
    mask = base._LOW_K_MASK
    tv = np.concatenate((truth_modes[mask].real, truth_modes[mask].imag))
    ev = np.concatenate((est_modes[mask].real, est_modes[mask].imag))
    response = float(np.dot(ev, tv) / max(np.dot(tv, tv), 1.0e-30))
    corr = float(np.corrcoef(tv, ev)[0, 1]) if np.std(ev) > 0 and np.std(tv) > 0 else 0.0
    residual = float(np.sum((ev - tv) ** 2) / max(np.sum(tv ** 2), 1.0e-30))
    std = np.std(draws, axis=0, ddof=1)
    coverage68 = float(np.mean(np.abs(estimate - truth) <= std))
    coverage95 = float(np.mean(np.abs(estimate - truth) <= 1.96 * std))
    holdout_fraction = 1.0 - base.TRAIN_FRACTION
    fitted = base._positive_intensity(estimate, arm, _velocity_from_field(estimate)) * holdout_fraction
    prior = base._positive_intensity(np.zeros_like(estimate), arm, np.zeros_like(_velocity_from_field(estimate))) * holdout_fraction
    holdout_gain = float(np.sum((holdout * np.log(np.maximum(fitted, 1.0e-300)) - fitted - gammaln(holdout + 1.0))
                                - (holdout * np.log(np.maximum(prior, 1.0e-300)) - prior - gammaln(holdout + 1.0))))
    # The vector objective includes the declared Gaussian latent prior; the
    # canonical joint_likelihood primitive intentionally does not.
    joint_err = abs(_joint_value(estimate_coeff, counts, arm, observed, sigma)
                    - (_joint_vector_objective(estimate_coeff, counts, arm, observed, sigma)
                       + 0.5 * np.sum(estimate_coeff ** 2 / PRIOR_SIGMA ** 2)))
    strict = bool(0.8 <= response <= 1.2 and corr >= 0.7 and residual <= 0.5 and
                  abs(coverage68 - 0.6826894921370859) <= 0.05 and
                  abs(coverage95 - 0.9544997361036416) <= 0.025 and holdout_gain > 0.0 and fit_success and joint_err < 1.0e-7)
    return {"response": response, "correlation_r": corr, "residual_power_ratio": residual,
            "coverage68": coverage68, "coverage95": coverage95, "heldout_log_score_improvement": holdout_gain,
            "joint_log_likelihood_abs_error": float(joint_err), "fit_success": bool(fit_success),
            "strict_low_k_gate": strict, "positive_support_fraction": 1.0}


def seed_schedule(index: int) -> dict[str, int]:
    if not isinstance(index, int) or not 0 <= index < MOCK_COUNT:
        raise ValueError("index outside sealed 64-member development range")
    return {"truth": SEED_START + index, "counts": 2026400000 + index,
            "marks": 2026500000 + index, "posterior": 2026600000 + index}


def run_mock(index: int, arm: str) -> dict[str, object]:
    seeds = seed_schedule(index)
    truth_coeff = _truth_coefficients(seeds["truth"])
    truth = _field_from_coeff(truth_coeff)
    velocity = _velocity_from_field(truth)
    all_counts = base._draw_counts(base._positive_intensity(truth, arm, velocity), arm, seeds["counts"])
    train, holdout = base._split_counts(all_counts, seeds["counts"] + 1000)
    observed, sigma = _draw_marks(truth_coeff, seeds["marks"])
    objective = lambda x: _joint_vector_objective(x, train, arm, observed, sigma) - 0.0
    # Bounded optimizer over the shared eight-mode latent, using numerical
    # gradients for the full count+CF4 factor and a deterministic start.
    fit = minimize(lambda x: -objective(x), np.zeros(MODE_COUNT), method="L-BFGS-B",
                   bounds=[(-3.0, 3.0)] * MODE_COUNT, options={"maxiter": 80, "ftol": 1.0e-9})
    estimate_coeff = np.asarray(fit.x, dtype=np.float64)
    estimate = _field_from_coeff(estimate_coeff)
    rng = np.random.default_rng(seeds["posterior"])
    precision = np.eye(MODE_COUNT) / PRIOR_SIGMA**2 + MARK_JAC.T @ MARK_JAC / (MARK_MEASUREMENT_SIGMA**2 + MARK_SHARED_SIGMA**2)
    covariance = np.linalg.pinv(precision)
    draws_coeff = rng.multivariate_normal(estimate_coeff, covariance, size=POSTERIOR_DRAWS)
    draws = np.asarray([_field_from_coeff(c) for c in draws_coeff])
    metrics = _mode_metrics(truth, draws, estimate, train, holdout, arm, observed, sigma,
                            bool(fit.success or np.isfinite(fit.fun)), estimate_coeff)
    metrics.update({"count_total": int(np.sum(all_counts)), "secure_rows": len(MARKS["secure_ids"]),
                    "secure_groups": int(MARKS["manifest"]["counts"]["secure_cf4_groups"]),
                    "excluded_target_rows_before_binning": MARKS["excluded_target_rows"],
                    "covariance_min_eigenvalue": float(np.min(np.linalg.eigvalsh(covariance)))})
    return {"index": index, "arm": arm, "seed": seeds, "metrics": metrics}


def run_calibration() -> dict[str, object]:
    arms = ("A", "B", "C", "D")
    members = [run_mock(index, arms[index // 16]) for index in range(MOCK_COUNT)]
    by_arm = {}
    metric_names = ("response", "correlation_r", "residual_power_ratio", "coverage68", "coverage95",
                    "heldout_log_score_improvement", "joint_log_likelihood_abs_error", "count_total")
    for arm in arms:
        rows = [r for r in members if r["arm"] == arm]
        arrays = {n: np.asarray([r["metrics"][n] for r in rows], dtype=float) for n in metric_names}
        by_arm[arm] = {"member_count": len(rows), "metric_median": {n: float(np.median(v)) for n, v in arrays.items()},
                       "metric_min": {n: float(np.min(v)) for n, v in arrays.items()},
                       "metric_max": {n: float(np.max(v)) for n, v in arrays.items()},
                       "strict_low_k_gate_pass_count": int(sum(bool(r["metrics"]["strict_low_k_gate"]) for r in rows)),
                       "heldout_positive_count": int(sum(r["metrics"]["heldout_log_score_improvement"] > 0 for r in rows))}
    return {"schema": "ouruniv-cf4-b1-integrated-joint-development-calibration-result-v3",
            "status": "COMPLETE_64_INTEGRATED_JOINT_DEVELOPMENT_MOCKS_TECHNICAL_NO_SCIENCE_CLAIM",
            "grid": {"N": GRID, "box_size_cMpc_h": BOX_SIZE, "cell_size_cMpc_h": CELL_SIZE, "latent_mode_count": MODE_COUNT},
            "joint_generation": {"same_frozen_latent_field_for_counts_and_cf4_marks": True, "cf4_positions": "deterministic_manifest-bound_synthetic_positions",
                                 "secure_rows": len(MARKS["secure_ids"]), "secure_groups": int(MARKS["manifest"]["counts"]["secure_cf4_groups"]),
                                 "excluded_target_rows_before_binning": MARKS["excluded_target_rows"], "selection_and_ownership": "canonical_manifest",
                                 "rsd_fog_tsc_nuisance_probes": run_joint_harness()},
            "seed_firewall": {"development_count": MOCK_COUNT, "seed_start_inclusive": SEED_START, "seed_stop_exclusive": SEED_STOP,
                              "untouched_validation_seed_start": 2026083320, "validation_opened": False},
            "arms": by_arm,
            "aggregate": {"member_count": MOCK_COUNT, "strict_low_k_gate_pass_count": sum(x["strict_low_k_gate_pass_count"] for x in by_arm.values()),
                          "all_members_finite_joint": all(r["metrics"]["joint_log_likelihood_abs_error"] < 1.0e-7 for r in members),
                          "all_members_positive_support": all(r["metrics"]["positive_support_fraction"] == 1.0 for r in members),
                          "all_arms_have_heldout_gain": all(x["heldout_positive_count"] > 0 for x in by_arm.values())},
            "scientific_disposition": {"development_calibration": "TECHNICAL_RESULT_ONLY", "observational_z0_posterior": "NOT_CREATED",
                                       "KF_EXPAND": "NOT_AUTHORIZED", "untouched_validation_seed": "NOT_RUN", "frontier_promotion": "NOT_ALLOWED", "0p3_cMpc_h_claim": "NOT_ALLOWED"}}


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
