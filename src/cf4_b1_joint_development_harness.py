"""Fail-closed probes for the active B1 CF4+2M++ joint contract.

The harness is local and synthetic.  It validates the interfaces needed by
the 64-mock development run without reading an observational posterior or
opening the untouched validation seeds.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from cf4_2mpp_crossmatch_manifest import build_secure_crossmatch_manifest
from cf4_2mpp_joint_likelihood_local import (
    LikelihoodInputError,
    joint_log_likelihood,
    poisson_log_likelihood,
    predict_selected_intensity,
    tsc_deposit,
    validate_factor_ownership,
)


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
MAPPING = ROOT / "data/cf4_2mpp_crossmatch_v1.csv"
SUMMARY = ROOT / "config/cf4_2mpp_crossmatch_v1_result.json"
_MANIFEST_CACHE: dict[str, object] | None = None


def _canonical_manifest() -> dict[str, object]:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        _MANIFEST_CACHE = build_secure_crossmatch_manifest(MAPPING, SUMMARY)
    return _MANIFEST_CACHE


def validate_selection_support(
    counts: np.ndarray, exposure: np.ndarray
) -> dict[str, object]:
    """Require finite raw exposure and reject positive counts off support."""

    counts = np.asarray(counts)
    exposure = np.asarray(exposure, dtype=np.float64)
    if counts.shape != exposure.shape or counts.ndim != 4 or counts.shape[0] != 6:
        raise LikelihoodInputError("counts/exposure must share shape (6,N,N,N)")
    if counts.dtype != np.dtype(np.int64):
        raise LikelihoodInputError("counts must have exact int64 dtype")
    if not np.all(np.isfinite(exposure)) or np.any(exposure < 0.0):
        raise LikelihoodInputError("selection exposure must be finite and non-negative")
    bad = int(np.count_nonzero((counts > 0) & (exposure <= 0.0)))
    if bad:
        raise LikelihoodInputError("positive observed count has zero selection exposure")
    return {
        "shape": list(counts.shape),
        "positive_count_zero_exposure": bad,
        "raw_exposure_not_count_normalized": True,
    }


def _tsc_adjoint(cotangent: np.ndarray, positions: np.ndarray, box_size: float) -> np.ndarray:
    """Analytic transpose of the periodic TSC scatter used by the oracle."""

    cotangent = np.asarray(cotangent, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    grid_size = cotangent.shape[0]
    spacing = box_size / grid_size
    wrapped = positions % box_size
    cell = wrapped / spacing - 0.5
    nearest = np.floor(cell + 0.5).astype(np.int64)
    offset = cell - nearest

    def weights(component: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            0.5 * (0.5 - component) ** 2,
            0.75 - component**2,
            0.5 * (0.5 + component) ** 2,
        )

    wx, wy, wz = (weights(offset[:, axis]) for axis in range(3))
    result = np.zeros(positions.shape[0], dtype=np.float64)
    for ix, dx in enumerate((-1, 0, 1)):
        for iy, dy in enumerate((-1, 0, 1)):
            for iz, dz in enumerate((-1, 0, 1)):
                result += (
                    wx[ix]
                    * wy[iy]
                    * wz[iz]
                    * cotangent[
                        (nearest[:, 0] + dx) % grid_size,
                        (nearest[:, 1] + dy) % grid_size,
                        (nearest[:, 2] + dz) % grid_size,
                    ]
                )
    return result


def tsc_adjoint_probe() -> dict[str, object]:
    positions = np.array(
        [[0.11, 0.23, 0.31], [5.91, 0.17, 5.73], [2.30, 4.70, 1.20], [3.77, 1.55, 4.33]],
        dtype=np.float64,
    )
    masses = np.array([0.4, 1.2, 0.7, 2.1], dtype=np.float64)
    cotangent = np.arange(8**3, dtype=np.float64).reshape(8, 8, 8) / 100.0
    deposited = tsc_deposit(positions, masses, 8, 6.0)
    left = float(np.sum(deposited * cotangent))
    right = float(np.dot(masses, _tsc_adjoint(cotangent, positions, 6.0)))
    error = abs(left - right)
    if error > 1.0e-12:
        raise AssertionError("TSC scatter/adjoint inner-product identity failed")
    return {"status": "PASS", "inner_product_abs_error": error, "mass_conserved": bool(np.sum(deposited) == np.sum(masses))}


def spherical_rsd_fog_probe() -> dict[str, object]:
    positions = np.array(
        [[0.2, 0.4, 0.7], [2.1, 1.3, 4.8], [5.2, 5.5, 0.9], [1.8, 4.3, 2.4]],
        dtype=np.float64,
    )
    velocities = np.array(
        [[30.0, -10.0, 5.0], [-20.0, 4.0, 11.0], [3.0, 14.0, -8.0], [8.0, -4.0, 13.0]],
        dtype=np.float64,
    )
    masses = np.full((6, len(positions)), 0.5, dtype=np.float64)
    exposure = np.ones((6, 8, 8, 8), dtype=np.float64)
    intensity = predict_selected_intensity(
        positions,
        velocities,
        masses,
        exposure,
        observer=np.array([3.0, 3.0, 3.0]),
        box_size_cMpc_h=6.0,
        hubble_km_s_Mpc=100.0,
        little_h=0.746,
        scale_factor=1.0,
        sigma_fog_km_s=np.full(6, 20.0),
        sigma_redshift_km_s=np.full(6, 10.0),
        quadrature_order=3,
    )
    totals = np.sum(intensity, axis=(1, 2, 3))
    expected = np.sum(masses, axis=1)
    relative_error = float(np.max(np.abs(totals - expected) / expected))
    if relative_error > 1.0e-12:
        raise AssertionError("spherical RSD/FoG mass conservation failed")
    return {"status": "PASS", "mass_relative_error_max": relative_error, "quadrature_order": 3}


def source_bound_joint_factor_probe() -> dict[str, object]:
    manifest = _canonical_manifest()
    entries = manifest["entries"]
    secure_ids = [entry["secure_object_id"] for entry in entries]
    twompp_ids = [entry["twompp_object_id"] for entry in entries]
    groups = np.asarray([entry["group_index"] for entry in entries], dtype=np.int64)
    observed = np.zeros(len(entries), dtype=np.float64)
    predicted = np.zeros(len(entries), dtype=np.float64)
    sigma = np.full(len(entries), 120.0, dtype=np.float64)
    shared = np.full(manifest["counts"]["secure_cf4_groups"], 35.0, dtype=np.float64)
    counts = np.zeros((6, 2, 2, 2), dtype=np.int64)
    intensity = np.ones_like(counts, dtype=np.float64)
    value = joint_log_likelihood(
        counts,
        intensity,
        observed,
        predicted,
        sigma,
        groups,
        shared,
        secure_object_ids=secure_ids,
        expected_group_count=manifest["counts"]["secure_cf4_groups"],
        independent_twompp_redshift_ids=(),
    )
    if not np.isfinite(value):
        raise AssertionError("source-bound joint likelihood is not finite")
    try:
        validate_factor_ownership(
            secure_ids,
            groups,
            independent_twompp_redshift_ids=[twompp_ids[0]],
        )
    except LikelihoodInputError:
        duplicate_rejection = True
    else:
        duplicate_rejection = False
    if not duplicate_rejection:
        raise AssertionError("independent 2M++ redshift factor was not rejected")
    return {
        "status": "PASS",
        "secure_rows": len(entries),
        "secure_groups": int(manifest["counts"]["secure_cf4_groups"]),
        "joint_log_likelihood_finite": True,
        "independent_redshift_rejected": duplicate_rejection,
    }


def source_bound_joint_score(
    counts: np.ndarray, intensity: np.ndarray, seed: int
) -> dict[str, float | bool | int]:
    """Evaluate one synthetic joint count+CF4 factor with canonical ownership."""

    manifest = _canonical_manifest()
    entries = manifest["entries"]
    secure_ids = [entry["secure_object_id"] for entry in entries]
    groups = np.asarray([entry["group_index"] for entry in entries], dtype=np.int64)
    shared = np.full(manifest["counts"]["secure_cf4_groups"], 35.0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    predicted = rng.normal(0.0, 100.0, size=len(entries)).astype(np.float64)
    observed = predicted + rng.normal(0.0, 120.0, size=len(entries)).astype(np.float64)
    sigma = np.full(len(entries), 120.0, dtype=np.float64)
    value = joint_log_likelihood(
        np.asarray(counts, dtype=np.int64),
        np.asarray(intensity, dtype=np.float64),
        observed,
        predicted,
        sigma,
        groups,
        shared,
        secure_object_ids=secure_ids,
        expected_group_count=manifest["counts"]["secure_cf4_groups"],
    )
    if not np.isfinite(value):
        raise AssertionError("synthetic source-bound joint score is not finite")
    return {
        "finite": True,
        "joint_log_likelihood": float(value),
        "secure_rows": len(entries),
        "secure_groups": int(manifest["counts"]["secure_cf4_groups"]),
    }


def nuisance_identifiability_probe() -> dict[str, object]:
    eta = np.linspace(-0.5, 0.5, 256, dtype=np.float64)
    exposure = np.linspace(0.2, 1.0, 256, dtype=np.float64)
    rows = []
    for population in range(6):
        lam = exposure * np.exp(np.log([0.0112, 0.086, 0.124, 0.0104, 0.086, 0.137][population]) + [1.7, 1.2, 1.15, 1.74, 1.21, 1.0][population] * eta)
        weight = np.sqrt(lam)
        row = np.zeros((256, 12), dtype=np.float64)
        row[:, population] = weight
        row[:, 6 + population] = weight * eta
        rows.append(row)
    design = np.concatenate(rows, axis=0)
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design, tol=1.0e-10))
    return {
        "status": "PASS" if rank == 12 else "FAIL",
        "rank": rank,
        "parameter_count": 12,
        "smallest_singular_value": float(np.min(singular)),
        "finite": bool(np.all(np.isfinite(singular))),
    }


def run_joint_harness() -> dict[str, object]:
    exposure = np.ones((6, 2, 2, 2), dtype=np.float64)
    counts = np.zeros((6, 2, 2, 2), dtype=np.int64)
    selection = validate_selection_support(counts, exposure)
    probes = {
        "selection_support": selection,
        "tsc_adjoint": tsc_adjoint_probe(),
        "spherical_rsd_fog": spherical_rsd_fog_probe(),
        "source_bound_joint_factor": source_bound_joint_factor_probe(),
        "nuisance_identifiability": nuisance_identifiability_probe(),
    }
    return {
        "status": "PASS" if all(item.get("status") in ("PASS", None) for item in probes.values()) else "FAIL",
        "probes": probes,
        "external_data_read": False,
        "observational_posterior": False,
        "validation_seeds_opened": False,
    }


__all__ = [
    "run_joint_harness",
    "source_bound_joint_factor_probe",
    "source_bound_joint_score",
    "spherical_rsd_fog_probe",
    "tsc_adjoint_probe",
    "validate_selection_support",
]
