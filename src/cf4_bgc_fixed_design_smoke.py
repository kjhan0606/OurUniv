#!/usr/bin/env python3
"""One frozen grouped-CF4 fixed-design linear-Gaussian implementation smoke.

The observed grouped-CF4 catalog fixes positions, errors, and the BGc design.
Its observed peculiar velocities are never posterior data: the datum is drawn
as ``u = A s_truth + B q_truth + epsilon``.  This single development smoke is
not a population-selection mock, validation ensemble, frontier, or science
claim.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

import cf4_linear_cr as linear
from cf4_kf_bin_manifest import canonical_json_bytes, validate_manifest_envelope
from cf4_mock_calibration import (
    compute_development_smoke_metrics,
    development_upstream_gate_schema,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = {
    "data/cf4_groups.csv": "bfdc0cfc0f172b48468e3a8fd05e87978c1ec68c341fb2d929fc1200f0123334",
    "data/cf4_clean.npz": "38cd2a22268c203ddc7ef708eeaa683ff2f55084ebf3983dc3cbffa56fe44b37",
    "src/cf4_ingest.py": "de0df0c427a935e82cd76904ee522145cc0c9c5b64ed8494b6ca3b8478af1b26",
    "src/cf4_bgc.py": "9ffbc8d07ca07ee05cba53f58d3e3cca5598bd27fffa8610c0c856bba043a90d",
    "src/cf4_linear_cr.py": "bee7924e64777c9ad5a0052ed4f8ec9b85fa4a932f6f87c92a0627bb3535c2ed",
}

N = 32
BOX_SIZE = 384.0
OMEGA_M = 0.31
OMEGA_B = 0.05
H = 0.746
A_S_1E9 = 1.63
N_S = 0.96
TRUTH_SEED = 2026083000
NUISANCE_TRUTH_SEED = 2026090100
NOISE_SEED = 2026090200
POSTERIOR_DRAW_SEEDS = (2026090300, 2026090301, 2026090302, 2026090303)
PRECONDITIONER_SEED = 2026090400
ADJOINT_SEED = 2026090500
CG_TOL = 3.0e-5
CG_MAXITER = 500
PRECONDITIONER_PROBES = 4
ADJOINT_MAX = 5.0e-5
CG_RESIDUAL_MAX = 1.0e-4
RADIAL_FORWARD_MAX_RELATIVE_ERROR = 5.0e-8
THETA_NON_NYQUIST_MAX_RELATIVE_ERROR = 1.0e-12
EXPECTED_OUTPUT_FILES = {"fields.npz", "result.json", "manifest.json", "COMPLETE"}


class SmokeError(ValueError):
    """The frozen implementation-smoke contract failed closed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_args(catalog: str | Path) -> SimpleNamespace:
    """Namespace consumed by the unchanged ``cf4_linear_cr`` helpers."""

    return SimpleNamespace(
        catalog=str(Path(catalog).resolve()),
        velocity_estimator="bgc",
        N=N,
        box_size=BOX_SIZE,
        Om=OMEGA_M,
        Ob=OMEGA_B,
        h=H,
        A_s_1e9=A_S_1E9,
        ns=N_S,
        bgc_window=801,
        bgc_cz_min=1500.0,
        bgc_cz_max=18000.0,
        bgc_pool_cz_min=500.0,
        bgc_pool_cz_max=30000.0,
        vmax=6000.0,
        edm_floor=0.04343,
        error_scale=0.9,
        sigma_nl=0.0,
        radial_fraction=0.95,
        bulk_prior=150.0,
        h0_prior=3.0,
        local_distance_max=0.0,
        holdout=0.2,
        holdout_by_raw_index_hash=True,
        split_seed=20260823,
        precond_probes=PRECONDITIONER_PROBES,
        cg_tol=CG_TOL,
        cg_maxiter=CG_MAXITER,
        float64=True,
    )


def verify_frozen_provenance(catalog: str | Path) -> dict[str, dict[str, str]]:
    """Bind the actual catalog and its construction/likelihood sources."""

    catalog_path = Path(catalog).resolve()
    expected_catalog = (ROOT / "data/cf4_clean.npz").resolve()
    if catalog_path != expected_catalog:
        raise SmokeError("catalog path is not the frozen data/cf4_clean.npz")
    rows: dict[str, dict[str, str]] = {}
    for relative, expected in EXPECTED_SHA256.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise SmokeError(f"frozen provenance SHA256 mismatch: {relative}")
        rows[relative] = {"path": str(path), "sha256": actual}
    return rows


def fixed_design_from_prepared(prepared: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Copy only fixed design arrays; deliberately discard observed ``vobs``."""

    required = (
        "raw_idx",
        "pgc",
        "cz",
        "pos",
        "rhat",
        "sig_measure",
        "variance",
        "likelihood_kind",
        "B",
        "q_std",
        "holdout",
    )
    missing = [key for key in required if key not in prepared]
    if missing:
        raise SmokeError(f"prepared BGc design is missing keys: {missing}")
    design = {key: np.array(prepared[key], copy=True) for key in required}
    count = design["raw_idx"].size
    if count == 0:
        raise SmokeError("fixed BGc design is empty")
    for key in ("pgc", "cz", "sig_measure", "variance", "likelihood_kind", "holdout"):
        if design[key].shape != (count,):
            raise SmokeError(f"fixed-design shape mismatch: {key}")
    for key in ("pos", "rhat"):
        if design[key].shape != (count, 3):
            raise SmokeError(f"fixed-design shape mismatch: {key}")
    if design["B"].shape != (count, 4) or design["q_std"].shape != (4,):
        raise SmokeError("fixed-design nuisance shape mismatch")
    if design["holdout"].dtype != np.dtype(bool):
        raise SmokeError("holdout mask must have exact boolean dtype")
    for key, value in design.items():
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise SmokeError(f"fixed design contains nonfinite values: {key}")
    if np.any(design["variance"] <= 0.0):
        raise SmokeError("fixed-design noise variance must be positive")
    if np.any(design["likelihood_kind"] != 2):
        raise SmokeError("fixed design contains a non-BGc likelihood row")
    if not np.any(design["holdout"]) or np.all(design["holdout"]):
        raise SmokeError("fixed design must contain train and holdout rows")
    return design


def prepare_fixed_design(catalog: str | Path) -> dict[str, np.ndarray]:
    """Prepare the observed-grouped-CF4-conditioned BGc geometry only."""

    verify_frozen_provenance(catalog)
    args = frozen_args(catalog)
    with np.load(catalog, allow_pickle=False) as source:
        prepared = linear.prepare_bgc_catalog(args, source)
    return fixed_design_from_prepared(prepared)


def generate_mock_datum(
    forward: Callable[[np.ndarray], object],
    design: Mapping[str, np.ndarray],
    shape: tuple[int, int, int] = (N, N, N),
) -> dict[str, np.ndarray]:
    """Draw truth, nuisance, and noise from separate frozen RNG streams."""

    s_truth = np.random.default_rng(TRUTH_SEED).standard_normal(shape)
    q_truth = (
        np.random.default_rng(NUISANCE_TRUTH_SEED).standard_normal(4)
        * np.asarray(design["q_std"], dtype=np.float64)
    )
    epsilon = (
        np.random.default_rng(NOISE_SEED).standard_normal(design["raw_idx"].size)
        * np.sqrt(np.asarray(design["variance"], dtype=np.float64))
    )
    signal = np.asarray(forward(s_truth), dtype=np.float64)
    if signal.shape != epsilon.shape or not np.all(np.isfinite(signal)):
        raise SmokeError("forward truth signal has invalid shape or values")
    datum = signal + np.asarray(design["B"], dtype=np.float64) @ q_truth + epsilon
    return {
        "s_truth": s_truth,
        "q_truth": q_truth,
        "epsilon": epsilon,
        "signal": signal,
        "u_mock": datum,
    }


def build_density_transfer(args: SimpleNamespace) -> tuple[np.ndarray, float]:
    """Return the full-FFT white-to-z=0-density multiplier used by A."""

    import jax.numpy as jnp
    from pmwd import Configuration, SimpleLCDM, boltzmann, growth
    from pmwd.boltzmann import linear_power

    spacing = args.box_size / args.N
    conf = Configuration(
        ptcl_spacing=float(spacing),
        ptcl_grid_shape=(args.N,) * 3,
        mesh_shape=1,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    cosmo = boltzmann(
        SimpleLCDM(
            conf,
            Omega_m=args.Om,
            Omega_b=args.Ob,
            h=args.h,
            A_s_1e9=args.A_s_1e9,
            n_s=args.ns,
        ),
        conf,
    )
    d1 = growth(1.0, cosmo, conf, order=1, deriv=0)
    dd1 = growth(1.0, cosmo, conf, order=1, deriv=1)
    growth_rate = float(dd1 / d1)
    frequency = 2.0 * np.pi * np.fft.fftfreq(args.N, d=spacing)
    kx, ky, kz = np.meshgrid(frequency, frequency, frequency, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    amplitude = np.sqrt(
        np.asarray(
            linear_power(jnp.asarray(kmag, dtype=jnp.float64), 1.0, cosmo, conf)
        )
        * args.box_size**3
    )
    transfer = amplitude / (math.sqrt(args.N**3) * spacing**3)
    transfer[0, 0, 0] = 0.0
    if not np.all(np.isfinite(transfer)) or np.any(transfer < 0.0):
        raise SmokeError("linear density transfer is nonfinite or negative")
    return transfer, growth_rate


def white_to_delta(white: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    """Apply the frozen white-to-z=0 linear-density transfer."""

    white = np.asarray(white, dtype=np.float64)
    transfer = np.asarray(transfer, dtype=np.float64)
    if white.shape != transfer.shape or white.ndim != 3:
        raise SmokeError("white field and density transfer shapes must match")
    if not np.all(np.isfinite(white)) or not np.all(np.isfinite(transfer)):
        raise SmokeError("white field or transfer contains nonfinite values")
    delta_k = np.fft.fftn(white, norm="ortho") * transfer
    return np.fft.ifftn(delta_k, norm="ortho").real


def delta_to_velocity(
    delta: np.ndarray,
    growth_rate: float,
    box_size: float = BOX_SIZE,
) -> np.ndarray:
    """Return the full linear velocity vector using the forward-model kernel."""

    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 3 or len(set(delta.shape)) != 1 or not np.all(np.isfinite(delta)):
        raise SmokeError("delta must be a finite cubic 3D field")
    if not math.isfinite(growth_rate) or growth_rate <= 0.0:
        raise SmokeError("growth rate must be finite and positive")
    grid_size = delta.shape[0]
    spacing = box_size / grid_size
    frequency = 2.0 * np.pi * np.fft.fftfreq(grid_size, d=spacing)
    radial_frequency = 2.0 * np.pi * np.fft.rfftfreq(grid_size, d=spacing)
    kx, ky, kz = np.meshgrid(
        frequency, frequency, radial_frequency, indexing="ij"
    )
    k2 = kx**2 + ky**2 + kz**2
    delta_k = np.fft.rfftn(delta)
    safe = np.where(k2 > 0.0, k2, 1.0)
    velocity = []
    for component in (kx, ky, kz):
        velocity_k = 1j * 100.0 * growth_rate * component / safe * delta_k
        velocity_k[k2 == 0.0] = 0.0
        velocity.append(np.fft.irfftn(velocity_k, s=delta.shape, axes=(0, 1, 2)))
    result = np.stack(velocity)
    if not np.all(np.isfinite(result)):
        raise SmokeError("velocity field contains nonfinite values")
    return result


def velocity_to_normalized_divergence(
    velocity: np.ndarray,
    growth_rate: float,
    box_size: float = BOX_SIZE,
) -> np.ndarray:
    """Compute theta=-div(v)/(100 f) from the stored discrete velocity grid."""

    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 4 or velocity.shape[0] != 3:
        raise SmokeError("velocity must have shape (3,N,N,N)")
    grid_size = velocity.shape[1]
    if velocity.shape != (3, grid_size, grid_size, grid_size):
        raise SmokeError("velocity grid must be cubic")
    if not np.all(np.isfinite(velocity)):
        raise SmokeError("velocity grid contains nonfinite values")
    if not math.isfinite(growth_rate) or growth_rate <= 0.0:
        raise SmokeError("growth rate must be finite and positive")
    spacing = box_size / grid_size
    frequency = 2.0 * np.pi * np.fft.fftfreq(grid_size, d=spacing)
    radial_frequency = 2.0 * np.pi * np.fft.rfftfreq(grid_size, d=spacing)
    kx, ky, kz = np.meshgrid(
        frequency, frequency, radial_frequency, indexing="ij"
    )
    divergence_k = 1j * (
        kx * np.fft.rfftn(velocity[0])
        + ky * np.fft.rfftn(velocity[1])
        + kz * np.fft.rfftn(velocity[2])
    )
    theta = -np.fft.irfftn(
        divergence_k, s=(grid_size,) * 3, axes=(0, 1, 2)
    ) / (100.0 * growth_rate)
    if not np.all(np.isfinite(theta)):
        raise SmokeError("normalized velocity divergence contains nonfinite values")
    return theta


def non_nyquist_mode_mask(grid_size: int) -> np.ndarray:
    """Modes where a real-grid spectral derivative has no Nyquist ambiguity."""

    if grid_size <= 0 or grid_size % 2 != 0:
        raise SmokeError("non-Nyquist mask requires a positive even grid size")
    indices = np.arange(grid_size)
    ix, iy, iz = np.meshgrid(indices, indices, indices, indexing="ij")
    return (ix != grid_size // 2) & (iy != grid_size // 2) & (iz != grid_size // 2)


def non_nyquist_delta_theta_relative_error(
    delta: np.ndarray, theta: np.ndarray
) -> float:
    """Relative Fourier-space delta/theta error outside Nyquist planes."""

    delta = np.asarray(delta, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if delta.shape != theta.shape or delta.ndim != 3 or len(set(delta.shape)) != 1:
        raise SmokeError("delta/theta fields must be matching cubic grids")
    mask = non_nyquist_mode_mask(delta.shape[0])
    delta_modes = np.fft.fftn(delta, norm="ortho")[mask]
    theta_modes = np.fft.fftn(theta, norm="ortho")[mask]
    return float(
        np.linalg.norm(theta_modes - delta_modes)
        / max(np.linalg.norm(delta_modes), 1.0e-30)
    )


def cic_sample_radial_velocity(
    velocity: np.ndarray,
    positions: np.ndarray,
    radial_unit_vectors: np.ndarray,
    box_size: float = BOX_SIZE,
) -> np.ndarray:
    """NumPy CIC sampling matching ``cf4_linear_cr.build_forward``."""

    velocity = np.asarray(velocity, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    radial = np.asarray(radial_unit_vectors, dtype=np.float64)
    if velocity.ndim != 4 or velocity.shape[0] != 3:
        raise SmokeError("velocity must have shape (3,N,N,N)")
    grid_size = velocity.shape[1]
    if velocity.shape != (3, grid_size, grid_size, grid_size):
        raise SmokeError("velocity grid must be cubic")
    if positions.ndim != 2 or positions.shape[1] != 3 or radial.shape != positions.shape:
        raise SmokeError("positions/radial vectors must have matching shape (rows,3)")
    spacing = box_size / grid_size
    coordinate = (positions % box_size) / spacing
    base = np.floor(coordinate).astype(np.int64)
    fraction = coordinate - base
    sampled = np.zeros((positions.shape[0], 3), dtype=np.float64)
    for dx in (0, 1):
        wx = fraction[:, 0] if dx else 1.0 - fraction[:, 0]
        for dy in (0, 1):
            wy = fraction[:, 1] if dy else 1.0 - fraction[:, 1]
            for dz in (0, 1):
                wz = fraction[:, 2] if dz else 1.0 - fraction[:, 2]
                weight = wx * wy * wz
                ii = (base[:, 0] + dx) % grid_size
                jj = (base[:, 1] + dy) % grid_size
                kk = (base[:, 2] + dz) % grid_size
                sampled += weight[:, None] * velocity[:, ii, jj, kk].T
    return np.sum(sampled * radial, axis=1)


def load_bin_manifest(path: str | Path) -> tuple[dict[str, object], str, str]:
    payload = Path(path).read_bytes()
    try:
        envelope = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SmokeError("cannot parse bin manifest") from exc
    body = validate_manifest_envelope(envelope)
    if payload != canonical_json_bytes(envelope):
        raise SmokeError("bin manifest is not canonical JSON")
    body_sha = envelope["manifest_body_sha256"]
    return body, body_sha, _sha256(payload)


def global_merged_mode_plan(
    manifest_body: Mapping[str, object],
    *,
    grid_size: int = N,
    box_size: float = BOX_SIZE,
) -> dict[str, object]:
    """Map canonical independent-real modes into all 33 manifest merges."""

    native = manifest_body.get("native_bins")
    merged = manifest_body.get("merged_bins")
    if not isinstance(native, list) or len(native) != 38:
        raise SmokeError("manifest does not contain 38 native bins")
    if not isinstance(merged, list) or len(merged) != 33:
        raise SmokeError("manifest does not contain 33 frozen merged bins")
    if grid_size <= 0 or grid_size % 2 != 0:
        raise SmokeError("canonical signed-DFT plan requires a positive even grid size")
    spacing = box_size / grid_size
    signed_frequency_index = np.rint(
        np.fft.fftfreq(grid_size) * grid_size
    ).astype(np.int64)
    qx, qy, qz = np.meshgrid(
        signed_frequency_index,
        signed_frequency_index,
        signed_frequency_index,
        indexing="ij",
    )
    frequency = 2.0 * np.pi * signed_frequency_index / box_size
    kx, ky, kz = np.meshgrid(frequency, frequency, frequency, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2).ravel()
    n32_nyquist = np.pi / spacing
    qx_flat = qx.ravel()
    qy_flat = qy.ravel()
    qz_flat = qz.ravel()
    nyquist_index = -(grid_size // 2)

    def conjugate(component: np.ndarray) -> np.ndarray:
        opposite = -component
        return np.where(opposite == grid_size // 2, nyquist_index, opposite)

    cx = conjugate(qx_flat)
    cy = conjugate(qy_flat)
    cz = conjugate(qz_flat)
    canonical_representative = (
        (qx_flat < cx)
        | ((qx_flat == cx) & (qy_flat < cy))
        | ((qx_flat == cx) & (qy_flat == cy) & (qz_flat <= cz))
    )
    analysis = (
        (kmag > 0.0)
        & (kmag <= n32_nyquist * (1.0 + 4.0e-15))
        & canonical_representative
    )
    flat_indices = np.flatnonzero(analysis)
    kval = kmag[flat_indices]
    native_assignment = np.full(kval.size, -1, dtype=np.int64)
    for item in native:
        index = int(item["index"])
        lower = float(item["lower_h_Mpc"])
        upper = float(item["upper_h_Mpc"])
        inside = (kval >= lower) & (
            (kval < upper)
            | (bool(item["terminal_upper_inclusive"]) & (kval <= upper))
        )
        if np.any(native_assignment[inside] != -1):
            raise SmokeError("N32 modes map to overlapping native bins")
        native_assignment[inside] = index
    if np.any(native_assignment < 0):
        raise SmokeError("an N32 isotropic mode is absent from the manifest bins")
    native_to_merged = np.full(38, -1, dtype=np.int64)
    for item in merged:
        merged_index = int(item["merged_bin_index"])
        native_to_merged[np.asarray(item["native_bin_indices"], dtype=int)] = merged_index
    if np.any(native_to_merged < 0):
        raise SmokeError("manifest merged bins omit native bins")
    merged_assignment = native_to_merged[native_assignment]
    availability = []
    for item in merged:
        index = int(item["merged_bin_index"])
        count = int(np.count_nonzero(merged_assignment == index))
        availability.append(
            {
                "merged_bin_index": index,
                "native_bin_indices": item["native_bin_indices"],
                "N32_canonical_independent_real_mode_count": count,
                "status": (
                    "AVAILABLE_DEVELOPMENT_SMOKE"
                    if count > 0
                    else "NOT_EVALUATED_NO_N32_CANONICAL_INDEPENDENT_REAL_MODES"
                ),
            }
        )
    available_ids = np.array(
        [
            item["merged_bin_index"]
            for item in availability
            if item["N32_canonical_independent_real_mode_count"] > 0
        ],
        dtype=np.int64,
    )
    return {
        "flat_independent_field_indices": flat_indices,
        "mode_merged_bin_index": merged_assignment,
        "available_merged_bin_ids": available_ids,
        "availability": availability,
        "N32_isotropic_nyquist_h_Mpc": n32_nyquist,
        "canonical_independent_real_analysis_mode_count": int(flat_indices.size),
        "canonical_representative_rule": (
            "signed_DFT_indices_[-N/2,N/2-1];_lexicographically_smaller_of_"
            "(q,-q_mod_DFT)_or_self_conjugate_once"
        ),
    }


def evaluate_delta_theta_metrics(
    truth_white: np.ndarray,
    posterior_white: np.ndarray,
    posterior_mean_white: np.ndarray,
    transfer: np.ndarray,
    growth_rate: float,
    plan: Mapping[str, object],
    manifest_body_sha256: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Evaluate only available global merged bins; all strict gates stay false."""

    posterior_white = np.asarray(posterior_white, dtype=np.float64)
    if posterior_white.shape != (4,) + truth_white.shape:
        raise SmokeError("posterior white ensemble must contain exactly four matching draws")
    posterior_mean_white = np.asarray(posterior_mean_white, dtype=np.float64)
    if posterior_mean_white.shape != truth_white.shape or not np.all(
        np.isfinite(posterior_mean_white)
    ):
        raise SmokeError("analytic posterior mean white field must match finite truth")
    truth_delta = white_to_delta(truth_white, transfer)
    mean_delta = white_to_delta(posterior_mean_white, transfer)
    draw_delta = []
    for draw in posterior_white:
        draw_delta.append(white_to_delta(draw, transfer))
    draw_delta_array = np.stack(draw_delta)
    truth_velocity = delta_to_velocity(truth_delta, growth_rate)
    mean_velocity = delta_to_velocity(mean_delta, growth_rate)
    draw_velocity_array = np.stack(
        [delta_to_velocity(field, growth_rate) for field in draw_delta_array]
    )
    truth_theta = velocity_to_normalized_divergence(truth_velocity, growth_rate)
    mean_theta = velocity_to_normalized_divergence(mean_velocity, growth_rate)
    draw_theta_array = np.stack(
        [
            velocity_to_normalized_divergence(field, growth_rate)
            for field in draw_velocity_array
        ]
    )
    flat = np.asarray(plan["flat_independent_field_indices"], dtype=np.int64)
    assignment = np.asarray(plan["mode_merged_bin_index"], dtype=np.int64)
    available = np.asarray(plan["available_merged_bin_ids"], dtype=np.int64)
    truth_delta_modes = np.fft.fftn(truth_delta, norm="ortho").ravel()[flat]
    draw_delta_modes = np.stack(
        [np.fft.fftn(field, norm="ortho").ravel()[flat] for field in draw_delta_array]
    )
    mean_delta_modes = np.fft.fftn(mean_delta, norm="ortho").ravel()[flat]
    delta_prior_variance = np.asarray(transfer, dtype=np.float64).ravel()[flat] ** 2
    if np.any(delta_prior_variance <= 0.0):
        raise SmokeError("available N32 modes have zero prior density variance")
    delta_upstream = development_upstream_gate_schema(
        np.zeros(available.size, dtype=bool),
        np.zeros(available.size, dtype=bool),
    )
    delta_metrics = compute_development_smoke_metrics(
        truth_delta_modes[None, :],
        draw_delta_modes[None, :, :],
        delta_prior_variance,
        assignment,
        available,
        np.ones(available.size, dtype=bool),
        delta_upstream,
        domain_id="global_z0_density_delta",
        bin_manifest_body_sha256=manifest_body_sha256,
        posterior_mean=mean_delta_modes[None, :],
    )

    grid_size = truth_delta.shape[0]
    independent_grid_indices = np.unravel_index(flat, (grid_size,) * 3)
    theta_keep = np.logical_and.reduce(
        [axis != grid_size // 2 for axis in independent_grid_indices]
    )
    theta_flat = flat[theta_keep]
    theta_assignment = assignment[theta_keep]
    theta_available = np.unique(theta_assignment)
    theta_availability = []
    for item in plan["availability"]:
        merged_index = int(item["merged_bin_index"])
        count = int(np.count_nonzero(theta_assignment == merged_index))
        theta_availability.append(
            {
                "merged_bin_index": merged_index,
                "native_bin_indices": item["native_bin_indices"],
                "N32_non_nyquist_canonical_independent_real_mode_count": count,
                "status": (
                    "AVAILABLE_DEVELOPMENT_SMOKE"
                    if count > 0
                    else "NOT_EVALUATED_NO_N32_NON_NYQUIST_MODES"
                ),
            }
        )
    truth_theta_modes = np.fft.fftn(truth_theta, norm="ortho").ravel()[theta_flat]
    draw_theta_modes = np.stack(
        [np.fft.fftn(field, norm="ortho").ravel()[theta_flat] for field in draw_theta_array]
    )
    mean_theta_modes = np.fft.fftn(mean_theta, norm="ortho").ravel()[theta_flat]
    theta_prior_variance = (
        np.asarray(transfer, dtype=np.float64).ravel()[theta_flat] ** 2
    )
    theta_upstream = development_upstream_gate_schema(
        np.zeros(theta_available.size, dtype=bool),
        np.zeros(theta_available.size, dtype=bool),
    )
    theta_metrics = compute_development_smoke_metrics(
        truth_theta_modes[None, :],
        draw_theta_modes[None, :, :],
        theta_prior_variance,
        theta_assignment,
        theta_available,
        np.ones(theta_available.size, dtype=bool),
        theta_upstream,
        domain_id="global_discrete_normalized_velocity_divergence_theta",
        bin_manifest_body_sha256=manifest_body_sha256,
        posterior_mean=mean_theta_modes[None, :],
    )
    consistency_errors = [
        non_nyquist_delta_theta_relative_error(truth_delta, truth_theta),
        non_nyquist_delta_theta_relative_error(mean_delta, mean_theta),
        *[
            non_nyquist_delta_theta_relative_error(delta, theta)
            for delta, theta in zip(draw_delta_array, draw_theta_array)
        ],
    ]
    return delta_metrics, theta_metrics, {
        "truth_delta": truth_delta,
        "truth_theta": truth_theta,
        "posterior_mean_delta": mean_delta,
        "posterior_mean_theta": mean_theta,
        "posterior_delta": draw_delta_array,
        "posterior_theta": draw_theta_array,
        "truth_velocity": truth_velocity,
        "posterior_mean_velocity": mean_velocity,
        "posterior_draw_velocity": draw_velocity_array,
        "theta_global_merged_bin_availability": theta_availability,
        "theta_non_nyquist_analysis_mode_count": int(theta_flat.size),
        "delta_theta_non_nyquist_relative_errors": consistency_errors,
    }


def heldout_mock_predictive(
    forward_all: Callable[[np.ndarray], object],
    design: Mapping[str, np.ndarray],
    u_mock: np.ndarray,
    posterior_mean: np.ndarray,
    mean_q: np.ndarray,
    posterior_draws: np.ndarray,
    draw_q: np.ndarray,
) -> dict[str, object]:
    """Held-out mock diagnostic only; it cannot satisfy a strict gate."""

    hold = np.asarray(design["holdout"], dtype=bool)
    latent_mean = np.asarray(forward_all(posterior_mean), dtype=float)[hold]
    latent_mean += np.asarray(design["B"])[hold] @ mean_q
    latent_draws = np.stack(
        [
            np.asarray(forward_all(field), dtype=float)[hold]
            + np.asarray(design["B"])[hold] @ nuisance
            for field, nuisance in zip(posterior_draws, draw_q)
        ]
    )
    latent_variance = np.var(latent_draws, axis=0, ddof=1)
    noise_variance = np.asarray(design["variance"], dtype=float)[hold]
    predictive_variance = noise_variance + latent_variance
    residual = np.asarray(u_mock, dtype=float)[hold] - latent_mean
    z = residual / np.sqrt(predictive_variance)
    logp = -0.5 * np.sum(
        np.log(2.0 * np.pi * predictive_variance)
        + residual**2 / predictive_variance
    )
    return {
        "status": "DIAGNOSTIC_ONLY_NOT_A_HELDOUT_IMPROVEMENT_GATE",
        "n": int(np.count_nonzero(hold)),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "mock_log_predictive_density": float(logp),
        "strict_heldout_improvement_evaluated": False,
        "science_claim_allowed": False,
    }


def _numerical_gate(
    adjoint: float, mean_cg: float, sample_cg: Sequence[float]
) -> dict[str, object]:
    values = np.asarray(sample_cg, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise SmokeError("exactly four finite sample CG residuals are required")
    gate = {
        "adjoint_relative_error": float(adjoint),
        "adjoint_max_inclusive": ADJOINT_MAX,
        "adjoint_pass": bool(adjoint <= ADJOINT_MAX),
        "mean_cg_relative_residual": float(mean_cg),
        "sample_cg_relative_residuals": values.tolist(),
        "mean_and_sample_cg_max_inclusive": CG_RESIDUAL_MAX,
        "mean_cg_pass": bool(mean_cg <= CG_RESIDUAL_MAX),
        "all_sample_cg_pass": bool(np.all(values <= CG_RESIDUAL_MAX)),
    }
    gate["all_pass"] = bool(
        gate["adjoint_pass"] and gate["mean_cg_pass"] and gate["all_sample_cg_pass"]
    )
    return gate


def no_claim_policy() -> dict[str, bool]:
    """Frozen fail-closed policy for this one-mock implementation smoke."""

    return {
        "coverage68_gate_evaluated": False,
        "coverage95_gate_evaluated": False,
        "heldout_improvement_gate_evaluated": False,
        "frontier_evaluated": False,
        "development_64_mock_execution_performed": False,
        "untouched_validation_256_mock_execution_performed": False,
        "science_metric_or_claim_allowed": False,
        "KF_EXPAND_authorized": False,
    }


def calculate(
    catalog: str | Path,
    bin_manifest_path: str | Path,
    implementation_commit: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run the single authorized numerical implementation smoke."""

    if re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None:
        raise SmokeError("implementation commit must be lowercase 40-hex")
    provenance = verify_frozen_provenance(catalog)
    body, body_sha, manifest_file_sha = load_bin_manifest(bin_manifest_path)
    args = frozen_args(catalog)
    design = prepare_fixed_design(catalog)
    train = ~design["holdout"]
    hold = design["holdout"]

    import jax
    import jax.numpy as jnp

    forward_all, adjoint_all, growth_rate, dtype = linear.build_forward(
        design["pos"], design["rhat"], args
    )
    train_indices = np.flatnonzero(train)
    hold_indices = np.flatnonzero(hold)
    train_indices_jax = jnp.asarray(train_indices)

    A_train = jax.jit(lambda field: forward_all(field)[train_indices_jax])

    @jax.jit
    def AT_train(values):
        expanded = jnp.zeros(design["raw_idx"].size, dtype=dtype)
        return adjoint_all(expanded.at[train_indices_jax].set(values))

    mock = generate_mock_datum(forward_all, design)
    scale = jnp.asarray(np.sqrt(design["variance"][train]), dtype=dtype)
    Bn = jnp.asarray(design["B"][train], dtype=dtype) / scale[:, None]
    qvar = jnp.asarray(design["q_std"] ** 2, dtype=dtype)
    dnorm = jnp.asarray(mock["u_mock"][train], dtype=dtype) / scale
    An = jax.jit(lambda field: A_train(field) / scale)
    ATn = jax.jit(lambda values: AT_train(values / scale))

    @jax.jit
    def Cnorm(values):
        return values + An(ATn(values)) + Bn @ (qvar * (Bn.T @ values))

    adjoint_rng = np.random.default_rng(ADJOINT_SEED)
    sx = jnp.asarray(adjoint_rng.standard_normal((N, N, N)), dtype=dtype)
    dy = jnp.asarray(adjoint_rng.standard_normal(train_indices.size), dtype=dtype)
    lhs = float(jnp.vdot(An(sx), dy))
    rhs = float(jnp.vdot(sx, ATn(dy)))
    adjoint_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-30)

    probe_rng = np.random.default_rng(PRECONDITIONER_SEED)
    probe_power = np.zeros(train_indices.size, dtype=np.float64)
    for _ in range(PRECONDITIONER_PROBES):
        probe = jnp.asarray(probe_rng.standard_normal((N, N, N)), dtype=dtype)
        probe_power += np.asarray(An(probe), dtype=np.float64) ** 2
    probe_power /= PRECONDITIONER_PROBES
    nuisance_diag = np.sum(
        np.asarray(Bn, dtype=np.float64) ** 2
        * design["q_std"][None, :] ** 2,
        axis=1,
    )
    preconditioner = jnp.asarray(1.0 + probe_power + nuisance_diag, dtype=dtype)

    alpha_mean, mean_rel, _mean_seconds = linear.cg_solve(
        Cnorm, dnorm, preconditioner, args
    )
    mean_white_jax = ATn(alpha_mean)
    mean_white = np.asarray(mean_white_jax, dtype=np.float64)
    mean_q = np.asarray(qvar * (Bn.T @ alpha_mean), dtype=np.float64)

    posterior_white = []
    posterior_q = []
    sample_cg = []
    for seed in POSTERIOR_DRAW_SEEDS:
        rng = np.random.default_rng(seed)
        xi = jnp.asarray(rng.standard_normal((N, N, N)), dtype=dtype)
        q0 = jnp.asarray(rng.standard_normal(4) * design["q_std"], dtype=dtype)
        eps0 = jnp.asarray(rng.standard_normal(train_indices.size), dtype=dtype)
        rhs_sample = dnorm - An(xi) - Bn @ q0 - eps0
        alpha, relative, _seconds = linear.cg_solve(
            Cnorm, rhs_sample, preconditioner, args
        )
        posterior_white.append(np.asarray(xi + ATn(alpha), dtype=np.float64))
        posterior_q.append(
            np.asarray(q0 + qvar * (Bn.T @ alpha), dtype=np.float64)
        )
        sample_cg.append(relative)
    posterior_white_array = np.stack(posterior_white)
    posterior_q_array = np.stack(posterior_q)
    gates = _numerical_gate(adjoint_error, mean_rel, sample_cg)

    transfer, transfer_growth_rate = build_density_transfer(args)
    if abs(transfer_growth_rate - growth_rate) > 1.0e-12:
        raise SmokeError("forward and density-transfer growth rates disagree")
    plan = global_merged_mode_plan(body)
    delta_metrics, theta_metrics, physical_fields = evaluate_delta_theta_metrics(
        mock["s_truth"],
        posterior_white_array,
        mean_white,
        transfer,
        growth_rate,
        plan,
        body_sha,
    )
    posterior_mean_delta = physical_fields["posterior_mean_delta"]
    posterior_mean_theta = physical_fields["posterior_mean_theta"]
    truth_velocity = physical_fields["truth_velocity"]
    posterior_mean_velocity = physical_fields["posterior_mean_velocity"]
    posterior_draw_velocity = physical_fields["posterior_draw_velocity"]
    truth_radial_cic = cic_sample_radial_velocity(
        truth_velocity, design["pos"], design["rhat"], BOX_SIZE
    )
    radial_difference = truth_radial_cic - mock["signal"]
    radial_relative_error = float(
        np.linalg.norm(radial_difference)
        / max(np.linalg.norm(mock["signal"]), 1.0e-30)
    )
    gates["truth_radial_forward_CIC_relative_error"] = radial_relative_error
    gates["truth_radial_forward_CIC_max_abs_error_km_s"] = float(
        np.max(np.abs(radial_difference))
    )
    gates["truth_radial_forward_CIC_max_inclusive"] = (
        RADIAL_FORWARD_MAX_RELATIVE_ERROR
    )
    gates["truth_radial_forward_CIC_pass"] = bool(
        radial_relative_error <= RADIAL_FORWARD_MAX_RELATIVE_ERROR
    )
    theta_errors = np.asarray(
        physical_fields["delta_theta_non_nyquist_relative_errors"], dtype=float
    )
    gates["delta_theta_non_nyquist_relative_errors"] = theta_errors.tolist()
    gates["delta_theta_non_nyquist_max_relative_error"] = float(
        np.max(theta_errors)
    )
    gates["delta_theta_non_nyquist_max_inclusive"] = (
        THETA_NON_NYQUIST_MAX_RELATIVE_ERROR
    )
    gates["delta_theta_non_nyquist_pass"] = bool(
        np.all(theta_errors <= THETA_NON_NYQUIST_MAX_RELATIVE_ERROR)
    )
    gates["all_pass"] = bool(
        gates["all_pass"]
        and gates["truth_radial_forward_CIC_pass"]
        and gates["delta_theta_non_nyquist_pass"]
    )
    if not gates["all_pass"]:
        raise SmokeError("adjoint, CG, radial-forward, or theta numerical gate failed")
    heldout = heldout_mock_predictive(
        forward_all,
        design,
        mock["u_mock"],
        mean_white,
        mean_q,
        posterior_white_array,
        posterior_q_array,
    )
    implementation_path = Path(__file__)
    result = {
        "schema": "ouruniv-cf4-bgc-fixed-design-single-mock-smoke-result-v2",
        "status": "COMPLETE_IMPLEMENTATION_SMOKE_NO_SCIENCE_CLAIM",
        "selection_semantics": "observed_grouped_CF4_fixed_design_conditioned",
        "population_selection_mock": False,
        "fixed_design_catalog_and_source_provenance_validated": True,
        "population_selection_function_validation_performed": False,
        "observed_catalog_vobs_used_as_posterior_datum": False,
        "mock_datum_formula": "u_mock=A*s_truth+B*q_truth+epsilon",
        "development_truth_seed_consumed": TRUTH_SEED,
        "development_truth_seed_count_consumed": 1,
        "seeds": {
            "truth_white": TRUTH_SEED,
            "truth_nuisance": NUISANCE_TRUTH_SEED,
            "likelihood_noise": NOISE_SEED,
            "posterior_draws": list(POSTERIOR_DRAW_SEEDS),
            "preconditioner": PRECONDITIONER_SEED,
            "adjoint": ADJOINT_SEED,
        },
        "frozen_configuration": vars(args),
        "provenance": provenance,
        "bin_manifest": {
            "path": str(Path(bin_manifest_path).resolve()),
            "file_sha256": manifest_file_sha,
            "manifest_body_sha256": body_sha,
        },
        "implementation": {
            "path": "src/cf4_bgc_fixed_design_smoke.py",
            "sha256": sha256_file(implementation_path),
            "commit": implementation_commit,
        },
        "catalog_design": {
            "selected_rows": int(design["raw_idx"].size),
            "train_rows": int(np.count_nonzero(train)),
            "holdout_rows": int(hold_indices.size),
            "raw_index_hash_split": True,
            "fixed_design_uses_observed_positions_errors_and_BGc_selection": True,
            "real_vobs_retained_in_fixed_design": False,
        },
        "growth_rate": growth_rate,
        "numerical_gates": gates,
        "global_merged_bin_availability": plan["availability"],
        "theta_global_merged_bin_availability": physical_fields[
            "theta_global_merged_bin_availability"
        ],
        "N32_canonical_independent_real_analysis_mode_count": plan[
            "canonical_independent_real_analysis_mode_count"
        ],
        "N32_non_nyquist_theta_analysis_mode_count": physical_fields[
            "theta_non_nyquist_analysis_mode_count"
        ],
        "delta_metrics": delta_metrics,
        "theta_metrics": theta_metrics,
        "delta_theta_normalization": {
            "definition": "theta=-discrete_spectral_div(stored_v)/(100*f)",
            "stored_theta_semantics": "reconstructed_from_stored_velocity_not_copied_from_delta",
            "continuum_relation": "theta=delta_outside_real_grid_Nyquist_planes",
            "Nyquist_plane_modes_excluded_from_theta_metrics": True,
            "truth_max_abs_difference": float(
                np.max(np.abs(physical_fields["truth_delta"] - physical_fields["truth_theta"]))
            ),
            "posterior_max_abs_difference": float(
                np.max(
                    np.abs(
                        physical_fields["posterior_delta"]
                        - physical_fields["posterior_theta"]
                    )
                )
            ),
            "non_nyquist_relative_errors": theta_errors.tolist(),
            "non_nyquist_max_relative_error": float(np.max(theta_errors)),
            "non_nyquist_max_inclusive": THETA_NON_NYQUIST_MAX_RELATIVE_ERROR,
            "non_nyquist_consistency_pass": gates[
                "delta_theta_non_nyquist_pass"
            ],
        },
        "velocity_posterior_product": {
            "linear_kernel": "v_k=i*100*f*k/k^2*delta_k_with_DC_zero",
            "stored_arrays": [
                "truth_velocity",
                "posterior_mean_velocity",
                "posterior_draw_velocity",
            ],
            "component_order": ["x", "y", "z"],
            "per_cell_posterior_mean_vector_preserved": True,
            "full_four_draw_vector_ensemble_preserved": True,
            "per_cell_full_3x3_covariance_reconstructable_from_draw_axis": True,
            "covariance_reconstruction": "sample_covariance_across_four_draws_ddof_1",
            "scalar_sigma_v_substitution_allowed": False,
        },
        "heldout_mock_predictive": heldout,
        "ROI_metrics": {
            roi_id: "NOT_EVALUATED_SINGLE_GLOBAL_N32_IMPLEMENTATION_SMOKE"
            for roi_id in (
                "Local_Group",
                "Virgo",
                "Coma",
                "Local_Void",
                "Bootes_Void",
                "observer_environment",
            )
        },
        "missing_high_k_semantics": "NOT_EVALUATED_FAIL_CLOSED_NOT_ZERO_INFORMATION_AND_NOT_A_PASS",
        **no_claim_policy(),
    }
    arrays = {
        "truth_white": mock["s_truth"],
        "posterior_mean_white": mean_white,
        "posterior_draws_white": posterior_white_array,
        "truth_nuisance_q": mock["q_truth"],
        "posterior_mean_nuisance_q": mean_q,
        "posterior_draws_nuisance_q": posterior_q_array,
        "mock_datum": mock["u_mock"],
        "truth_forward_radial_signal": mock["signal"],
        "truth_velocity_CIC_radial_signal": truth_radial_cic,
        "truth_delta": physical_fields["truth_delta"],
        "truth_theta": physical_fields["truth_theta"],
        "posterior_mean_delta": posterior_mean_delta,
        "posterior_mean_theta": posterior_mean_theta,
        "posterior_delta": physical_fields["posterior_delta"],
        "posterior_theta": physical_fields["posterior_theta"],
        "truth_velocity": truth_velocity,
        "posterior_mean_velocity": posterior_mean_velocity,
        "posterior_draw_velocity": posterior_draw_velocity,
        "train_raw_idx": design["raw_idx"][train],
        "holdout_raw_idx": design["raw_idx"][hold],
    }
    return result, arrays


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Timestamp-free, key-sorted, uncompressed NPZ bytes."""

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for key in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue())
    return output.getvalue()


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def publish(
    output_path: str | Path,
    result: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Build a sibling staging directory, then publish by one plain rename."""

    if result.get("status") != "COMPLETE_IMPLEMENTATION_SMOKE_NO_SCIENCE_CLAIM":
        raise SmokeError("only a complete no-claim implementation smoke may publish")
    if result.get("numerical_gates", {}).get("all_pass") is not True:
        raise SmokeError("numerical gates did not pass")
    output = Path(output_path)
    if not output.parent.is_dir():
        raise SmokeError("output parent directory must already exist")
    stage = output.parent / f".{output.name}.staging"
    if os.path.lexists(output):
        raise FileExistsError(f"refusing overwrite of {output}") from None
    try:
        stage.mkdir(mode=0o700)
    except FileExistsError:
        raise FileExistsError(f"refusing existing staging directory {stage}") from None
    stage_stat = stage.stat()
    stage_identity = (stage_stat.st_dev, stage_stat.st_ino)
    published = False
    try:
        fields_payload = deterministic_npz_bytes(arrays)
        result_payload = canonical_json_bytes(result)
        _write_exclusive(stage / "fields.npz", fields_payload)
        _write_exclusive(stage / "result.json", result_payload)
        artifact = {
            "schema": "ouruniv-cf4-bgc-fixed-design-single-mock-smoke-artifact-manifest-v2",
            "status": result["status"],
            "implementation_commit": result["implementation"]["commit"],
            "bin_manifest_body_sha256": result["bin_manifest"]["manifest_body_sha256"],
            "payloads": {
                "fields.npz": {"sha256": _sha256(fields_payload), "bytes": len(fields_payload)},
                "result.json": {"sha256": _sha256(result_payload), "bytes": len(result_payload)},
            },
        }
        artifact_payload = canonical_json_bytes(artifact)
        _write_exclusive(stage / "manifest.json", artifact_payload)
        complete = {
            "schema": "ouruniv-cf4-bgc-fixed-design-single-mock-smoke-complete-v2",
            "status": result["status"],
            "manifest_sha256": _sha256(artifact_payload),
            "implementation_commit": result["implementation"]["commit"],
            "bin_manifest_body_sha256": result["bin_manifest"]["manifest_body_sha256"],
            "COMPLETE_written_last": True,
        }
        complete_payload = canonical_json_bytes(complete)
        directory_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _write_exclusive(stage / "COMPLETE", complete_payload)
        directory_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if os.path.lexists(output):
            raise FileExistsError(f"refusing raced overwrite of {output}")
        os.rename(stage, output)
        published = True
    finally:
        if not published:
            try:
                current = stage.stat()
            except FileNotFoundError:
                current = None
            if current is not None and (current.st_dev, current.st_ino) == stage_identity:
                shutil.rmtree(stage)


def validate_output(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {item.name for item in root.iterdir()} != EXPECTED_OUTPUT_FILES:
        raise SmokeError("output directory/file set is not exact")
    result_payload = (root / "result.json").read_bytes()
    fields_payload = (root / "fields.npz").read_bytes()
    manifest_payload = (root / "manifest.json").read_bytes()
    complete_payload = (root / "COMPLETE").read_bytes()
    result = json.loads(result_payload)
    artifact = json.loads(manifest_payload)
    complete = json.loads(complete_payload)
    if result_payload != canonical_json_bytes(result):
        raise SmokeError("result JSON is not canonical")
    if manifest_payload != canonical_json_bytes(artifact):
        raise SmokeError("artifact manifest JSON is not canonical")
    if complete_payload != canonical_json_bytes(complete):
        raise SmokeError("COMPLETE JSON is not canonical")
    if result.get("schema") != "ouruniv-cf4-bgc-fixed-design-single-mock-smoke-result-v2":
        raise SmokeError("result schema mismatch")
    if result.get("status") != "COMPLETE_IMPLEMENTATION_SMOKE_NO_SCIENCE_CLAIM":
        raise SmokeError("result status mismatch")
    if (
        artifact.get("schema")
        != "ouruniv-cf4-bgc-fixed-design-single-mock-smoke-artifact-manifest-v2"
    ):
        raise SmokeError("artifact manifest schema mismatch")
    if (
        complete.get("schema")
        != "ouruniv-cf4-bgc-fixed-design-single-mock-smoke-complete-v2"
    ):
        raise SmokeError("COMPLETE schema mismatch")
    if artifact.get("payloads") != {
        "fields.npz": {"sha256": _sha256(fields_payload), "bytes": len(fields_payload)},
        "result.json": {"sha256": _sha256(result_payload), "bytes": len(result_payload)},
    }:
        raise SmokeError("artifact payload bindings mismatch")
    if complete.get("manifest_sha256") != _sha256(manifest_payload):
        raise SmokeError("COMPLETE does not bind artifact manifest")
    if complete.get("COMPLETE_written_last") is not True:
        raise SmokeError("COMPLETE-last contract is absent")
    if complete.get("status") != result.get("status") or artifact.get(
        "status"
    ) != result.get("status"):
        raise SmokeError("artifact status bindings mismatch")
    if complete.get("implementation_commit") != result.get("implementation", {}).get("commit"):
        raise SmokeError("COMPLETE implementation binding mismatch")
    if artifact.get("implementation_commit") != result.get("implementation", {}).get(
        "commit"
    ):
        raise SmokeError("artifact implementation binding mismatch")
    if complete.get("bin_manifest_body_sha256") != result.get("bin_manifest", {}).get(
        "manifest_body_sha256"
    ):
        raise SmokeError("COMPLETE bin-manifest binding mismatch")
    if artifact.get("bin_manifest_body_sha256") != result.get("bin_manifest", {}).get(
        "manifest_body_sha256"
    ):
        raise SmokeError("artifact bin-manifest binding mismatch")
    if result.get("selection_semantics") != "observed_grouped_CF4_fixed_design_conditioned":
        raise SmokeError("fixed-design selection semantics mismatch")
    if result.get("mock_datum_formula") != "u_mock=A*s_truth+B*q_truth+epsilon":
        raise SmokeError("mock datum formula mismatch")
    if result.get("population_selection_mock") is not False:
        raise SmokeError("output incorrectly claims a population-selection mock")
    if result.get("population_selection_function_validation_performed") is not False:
        raise SmokeError("output incorrectly claims population-selection validation")
    if result.get("observed_catalog_vobs_used_as_posterior_datum") is not False:
        raise SmokeError("output used or ambiguously labels observed vobs")
    if result.get("development_truth_seed_consumed") != TRUTH_SEED or result.get(
        "development_truth_seed_count_consumed"
    ) != 1:
        raise SmokeError("single development truth-seed contract mismatch")
    seeds = result.get("seeds", {})
    if seeds != {
        "truth_white": TRUTH_SEED,
        "truth_nuisance": NUISANCE_TRUTH_SEED,
        "likelihood_noise": NOISE_SEED,
        "posterior_draws": list(POSTERIOR_DRAW_SEEDS),
        "preconditioner": PRECONDITIONER_SEED,
        "adjoint": ADJOINT_SEED,
    }:
        raise SmokeError("RNG stream contract mismatch")
    if result.get("numerical_gates", {}).get("all_pass") is not True:
        raise SmokeError("output numerical gates are not all passing")
    if result.get("numerical_gates", {}).get("delta_theta_non_nyquist_pass") is not True:
        raise SmokeError("output delta/theta non-Nyquist gate is not passing")
    if result.get("N32_canonical_independent_real_analysis_mode_count") != 8538:
        raise SmokeError("N32 canonical independent-real mode count mismatch")
    if result.get("N32_non_nyquist_theta_analysis_mode_count") != 8535:
        raise SmokeError("N32 non-Nyquist theta mode count mismatch")
    availability = result.get("global_merged_bin_availability")
    if not isinstance(availability, list) or len(availability) != 33:
        raise SmokeError("all 33 merged-bin availability records are required")
    if [item.get("merged_bin_index") for item in availability] != list(range(33)):
        raise SmokeError("merged-bin availability indices are not canonical")
    if sum(
        item.get("N32_canonical_independent_real_mode_count", -1)
        for item in availability
    ) != 8538:
        raise SmokeError("density merged-bin mode counts do not sum to 8538")
    theta_availability = result.get("theta_global_merged_bin_availability")
    if not isinstance(theta_availability, list) or len(theta_availability) != 33:
        raise SmokeError("all 33 theta merged-bin availability records are required")
    if [item.get("merged_bin_index") for item in theta_availability] != list(range(33)):
        raise SmokeError("theta merged-bin availability indices are not canonical")
    if sum(
        item.get("N32_non_nyquist_canonical_independent_real_mode_count", -1)
        for item in theta_availability
    ) != 8535:
        raise SmokeError("theta merged-bin mode counts do not sum to 8535")
    normalization = result.get("delta_theta_normalization", {})
    if normalization.get("stored_theta_semantics") != (
        "reconstructed_from_stored_velocity_not_copied_from_delta"
    ) or normalization.get("non_nyquist_consistency_pass") is not True or normalization.get(
        "Nyquist_plane_modes_excluded_from_theta_metrics"
    ) is not True:
        raise SmokeError("stored theta/divergence semantics or consistency gate mismatch")
    for key, expected in no_claim_policy().items():
        if result.get(key) is not expected:
            raise SmokeError(f"output no-claim flag mismatch: {key}")
    for domain in ("delta_metrics", "theta_metrics"):
        metrics = result.get(domain, {})
        if metrics.get("posterior_mean_source") != (
            "explicit_analytic_posterior_mean"
        ):
            raise SmokeError(f"{domain} does not use the analytic posterior mean")
        if metrics.get("mock_count") != 1 or metrics.get("posterior_draw_count") != 4:
            raise SmokeError(f"{domain} is not the one-mock/four-draw smoke")
    velocity_contract = result.get("velocity_posterior_product", {})
    if (
        velocity_contract.get("per_cell_full_3x3_covariance_reconstructable_from_draw_axis")
        is not True
        or velocity_contract.get("scalar_sigma_v_substitution_allowed") is not False
    ):
        raise SmokeError("full-vector velocity posterior contract is absent")
    with np.load(io.BytesIO(fields_payload), allow_pickle=False) as fields:
        expected_arrays = {
            "truth_white",
            "posterior_mean_white",
            "posterior_draws_white",
            "truth_nuisance_q",
            "posterior_mean_nuisance_q",
            "posterior_draws_nuisance_q",
            "mock_datum",
            "truth_forward_radial_signal",
            "truth_velocity_CIC_radial_signal",
            "truth_delta",
            "truth_theta",
            "posterior_mean_delta",
            "posterior_mean_theta",
            "posterior_delta",
            "posterior_theta",
            "truth_velocity",
            "posterior_mean_velocity",
            "posterior_draw_velocity",
            "train_raw_idx",
            "holdout_raw_idx",
        }
        if "sigma_v" in fields.files or any(name.endswith("sigma_v") for name in fields.files):
            raise SmokeError("scalar sigma_v may not replace the vector ensemble")
        if set(fields.files) != expected_arrays:
            missing = sorted(expected_arrays - set(fields.files))
            extra = sorted(set(fields.files) - expected_arrays)
            raise SmokeError(
                f"stored field-array set is not exact; missing={missing}, extra={extra}"
            )
        required_shapes = {
            "truth_white": (N, N, N),
            "posterior_mean_white": (N, N, N),
            "posterior_draws_white": (4, N, N, N),
            "truth_delta": (N, N, N),
            "truth_theta": (N, N, N),
            "posterior_mean_delta": (N, N, N),
            "posterior_mean_theta": (N, N, N),
            "posterior_delta": (4, N, N, N),
            "posterior_theta": (4, N, N, N),
            "truth_velocity": (3, N, N, N),
            "posterior_mean_velocity": (3, N, N, N),
            "posterior_draw_velocity": (4, 3, N, N, N),
        }
        for name, shape in required_shapes.items():
            if name not in fields.files or fields[name].shape != shape:
                raise SmokeError(f"required field shape mismatch: {name}")
            if not np.all(np.isfinite(fields[name])):
                raise SmokeError(f"required field is nonfinite: {name}")
        selected_rows = result.get("catalog_design", {}).get("selected_rows")
        train_rows = result.get("catalog_design", {}).get("train_rows")
        holdout_rows = result.get("catalog_design", {}).get("holdout_rows")
        if not all(isinstance(value, int) and value > 0 for value in (
            selected_rows,
            train_rows,
            holdout_rows,
        )) or train_rows + holdout_rows != selected_rows:
            raise SmokeError("catalog train/holdout row-count contract mismatch")
        row_shapes = {
            "mock_datum": (selected_rows,),
            "truth_forward_radial_signal": (selected_rows,),
            "truth_velocity_CIC_radial_signal": (selected_rows,),
            "train_raw_idx": (train_rows,),
            "holdout_raw_idx": (holdout_rows,),
            "truth_nuisance_q": (4,),
            "posterior_mean_nuisance_q": (4,),
            "posterior_draws_nuisance_q": (4, 4),
        }
        for name, shape in row_shapes.items():
            if fields[name].shape != shape or not np.all(np.isfinite(fields[name])):
                raise SmokeError(f"row/nuisance field shape or finiteness mismatch: {name}")
        growth_rate = result.get("growth_rate")
        if (
            not isinstance(growth_rate, (int, float))
            or isinstance(growth_rate, bool)
            or not math.isfinite(growth_rate)
            or growth_rate <= 0.0
        ):
            raise SmokeError("result growth rate is not finite and positive")
        reconstructed = {
            "truth_theta": velocity_to_normalized_divergence(
                fields["truth_velocity"], growth_rate
            ),
            "posterior_mean_theta": velocity_to_normalized_divergence(
                fields["posterior_mean_velocity"], growth_rate
            ),
        }
        reconstructed["posterior_theta"] = np.stack(
            [
                velocity_to_normalized_divergence(velocity, growth_rate)
                for velocity in fields["posterior_draw_velocity"]
            ]
        )
        for name, expected in reconstructed.items():
            if not np.array_equal(fields[name], expected):
                raise SmokeError(f"stored theta is not reconstructed from velocity: {name}")
        consistency = [
            non_nyquist_delta_theta_relative_error(
                fields["truth_delta"], fields["truth_theta"]
            ),
            non_nyquist_delta_theta_relative_error(
                fields["posterior_mean_delta"], fields["posterior_mean_theta"]
            ),
            *[
                non_nyquist_delta_theta_relative_error(delta, theta)
                for delta, theta in zip(
                    fields["posterior_delta"], fields["posterior_theta"]
                )
            ],
        ]
        if np.max(consistency) > THETA_NON_NYQUIST_MAX_RELATIVE_ERROR:
            raise SmokeError("stored delta/theta non-Nyquist consistency gate failed")
        for source, label in (
            (normalization.get("non_nyquist_relative_errors"), "normalization"),
            (
                result.get("numerical_gates", {}).get(
                    "delta_theta_non_nyquist_relative_errors"
                ),
                "numerical gate",
            ),
        ):
            try:
                recorded = np.asarray(source, dtype=float)
            except (TypeError, ValueError) as exc:
                raise SmokeError(f"{label} delta/theta errors are invalid") from exc
            if recorded.shape != (6,) or not np.array_equal(recorded, consistency):
                raise SmokeError(f"{label} delta/theta errors do not bind stored fields")
    return {
        "status": "PASS",
        "directory": str(root),
        "result_sha256": _sha256(result_payload),
        "fields_sha256": _sha256(fields_payload),
        "manifest_sha256": _sha256(manifest_payload),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--catalog", type=Path, default=ROOT / "data/cf4_clean.npz")
    run.add_argument(
        "--bin-manifest",
        type=Path,
        default=ROOT / "config/cf4_kf_bin_manifest_v1.json",
    )
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--implementation-commit", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--directory", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            result, arrays = calculate(
                args.catalog, args.bin_manifest, args.implementation_commit
            )
            publish(args.output, result, arrays)
            report = validate_output(args.output)
        else:
            report = validate_output(args.directory)
    except (OSError, ValueError, SmokeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
