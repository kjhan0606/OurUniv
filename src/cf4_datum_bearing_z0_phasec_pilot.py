#!/usr/bin/env python3
"""Phase-C eight-mock calibration pilot for the datum-bearing z=0 route.

The program is mock-only.  It constructs an N64 PMWD truth whose non-Nyquist
N32 Fourier modes are exactly inherited from the predeclared development seed,
generates one of four frozen six-tracer stress arms, and samples the N32 joint
count-plus-CF4-velocity model with four HMC chains.  It never reads the actual
2M++ count arrays or the actual CF4 radial-velocity datum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

import cf4_bgc_fixed_design_smoke as fixed
import cf4_linear_cr as linear


SCHEMA = "ouruniv-cf4-datum-bearing-z0-phasec-program-v1"
AMENDMENT_SCHEMA = "ouruniv-cf4-datum-bearing-z0-phasec-execution-amendment-v2"
TASK_SCHEMA = "ouruniv-cf4-datum-bearing-z0-phasec-task-result-v1"
AGGREGATE_SCHEMA = "ouruniv-cf4-datum-bearing-z0-phasec-aggregate-result-v1"
TASK_FILES = {"posterior_summary.npz", "diagnostics.npz", "result.json", "manifest.json", "COMPLETE"}
AGGREGATE_FILES = {"aggregate.json", "manifest.json", "COMPLETE"}
PREFLIGHT_FILES = {"result.json", "manifest.json", "COMPLETE"}


class PhaseCError(ValueError):
    """The fail-closed Phase-C contract was violated."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def tagged_rng(seed: int, tag: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(tag)]))


def task_name(index: int, seed: int, arm: str) -> str:
    return f"mock_{index:02d}_seed_{seed}_arm_{arm}"


def load_program(path: str | Path) -> tuple[dict[str, object], str]:
    payload = Path(path).read_bytes()
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PhaseCError("cannot parse Phase-C program") from exc
    if document.get("schema") == AMENDMENT_SCHEMA:
        for binding_name in ("base_program", "V1_infrastructure_failure"):
            binding = document.get(binding_name, {})
            binding_path = Path(str(binding.get("path", "")))
            if not binding_path.is_file() or sha256_file(binding_path) != binding.get("sha256"):
                raise PhaseCError(f"Phase-C V2 amendment binding mismatch: {binding_name}")
        if "prior_preflight_failure" in document:
            binding = document["prior_preflight_failure"]
            binding_path = Path(str(binding.get("path", "")))
            if not binding_path.is_file() or sha256_file(binding_path) != binding.get("sha256"):
                raise PhaseCError("Phase-C amendment prior-preflight binding mismatch")
        if document.get("authorization", {}).get("execution_only_retry_after_pre_science_failure") is not True:
            raise PhaseCError("Phase-C V2 execution-only repair is unauthorized")
        program = json.loads(Path(document["base_program"]["path"]).read_text())
        program["status"] = "AUTHORIZED_V2_EXECUTION_ONLY_REPAIR"
        program["authorization"]["execution_only_retry_after_pre_science_failure"] = True
        program["input_bindings"]["V1_program"] = dict(document["base_program"])
        program["input_bindings"]["V1_infrastructure_failure"] = dict(
            document["V1_infrastructure_failure"]
        )
        if "prior_preflight_failure" in document:
            program["input_bindings"]["prior_preflight_failure"] = dict(
                document["prior_preflight_failure"]
            )
        program["source_bindings"]["Phase_C_implementation"] = dict(
            document["Phase_C_implementation"]
        )
        program["execution"].update(document["execution_override"])
        program["environment"].update(document["environment_override"])
        program["execution_amendment"] = {
            "schema": AMENDMENT_SCHEMA,
            "path": str(Path(path).resolve()),
            "science_contract_change": False,
        }
        if "cross_device_growth_relative_tolerance" in document["environment_override"]:
            program["execution_amendment"]["cross_device_growth_relative_tolerance"] = float(
                document["environment_override"]["cross_device_growth_relative_tolerance"]
            )
    else:
        program = document
    if program.get("schema") != SCHEMA:
        raise PhaseCError("Phase-C program schema mismatch")
    authorization = program.get("authorization", {})
    for key in (
        "Phase_C_eight_mock_pilot",
        "Slurm_GPU_array",
        "Slurm_CPU_aggregate",
        "GPFS_read",
        "GPFS_write_new_output_only",
    ):
        if authorization.get(key) is not True:
            raise PhaseCError(f"missing Phase-C authorization: {key}")
    for key in (
        "actual_observational_field_inference",
        "actual_2Mpp_count_read",
        "actual_CF4_velocity_datum_used",
        "validation_seed_access",
        "Phase_D_or_later",
        "IC_PM_HOP_RAMSES",
    ):
        if authorization.get(key) is not False:
            raise PhaseCError(f"forbidden Phase-C authorization enabled: {key}")
    assignments = program.get("mock_assignments")
    expected = [
        {"index": i, "seed": 2026083000 + i, "arm": "ABCD"[i // 2]}
        for i in range(8)
    ]
    if assignments != expected:
        raise PhaseCError("the exact eight-mock balanced assignment changed")
    grid = program.get("grid", {})
    if grid != {
        "box_size_cMpc_h": 384.0,
        "inference_N": 32,
        "inference_cell_size_cMpc_h": 12.0,
        "truth_N": 64,
        "truth_cell_size_cMpc_h": 6.0,
    }:
        raise PhaseCError("Phase-C grid contract changed")
    if program.get("sampler", {}).get("chain_count") != 4:
        raise PhaseCError("Phase-C requires exactly four chains per mock")
    if program.get("sampler", {}).get("posterior_draws_per_chain", 0) < 128:
        raise PhaseCError("Phase-C posterior chains are too short for the declared diagnostics")
    for section in ("input_bindings", "source_bindings"):
        records = program.get(section)
        if not isinstance(records, Mapping) or not records:
            raise PhaseCError(f"{section} is absent")
        for record in records.values():
            source = Path(str(record["path"]))
            if sha256_file(source) != record["sha256"]:
                raise PhaseCError(f"SHA256 mismatch: {source}")
    return program, hashlib.sha256(payload).hexdigest()


def nested_white_fields(
    seed: int,
    coarse_n: int,
    fine_n: int,
    high_k_tag: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Generate a fine white field with exact inherited coarse low modes.

    The coarse Nyquist planes are excluded because their signed-frequency
    representation is ambiguous on an even grid.  Every other coarse Fourier
    coefficient is inserted into the same signed mode of the fine orthonormal
    FFT.  Both members of each Hermitian pair are inserted together.
    """

    if fine_n <= coarse_n or fine_n % coarse_n != 0 or coarse_n % 2:
        raise PhaseCError("nested grid sizes must be even integer refinements")
    coarse = np.random.default_rng(int(seed)).standard_normal((coarse_n,) * 3)
    fine_base = tagged_rng(seed, high_k_tag).standard_normal((fine_n,) * 3)
    coarse_k = np.fft.fftn(coarse, norm="ortho")
    fine_k = np.fft.fftn(fine_base, norm="ortho")

    coarse_modes = np.rint(np.fft.fftfreq(coarse_n) * coarse_n).astype(int)
    keep = np.abs(coarse_modes) < coarse_n // 2
    coarse_idx = np.flatnonzero(keep)
    signed = coarse_modes[coarse_idx]
    fine_idx = np.where(signed >= 0, signed, fine_n + signed).astype(int)
    fine_k[np.ix_(fine_idx, fine_idx, fine_idx)] = coarse_k[
        np.ix_(coarse_idx, coarse_idx, coarse_idx)
    ]
    fine_complex = np.fft.ifftn(fine_k, norm="ortho")
    imaginary_max = float(np.max(np.abs(fine_complex.imag)))
    if imaginary_max > 2.0e-12:
        raise PhaseCError("nested field lost Hermitian symmetry")
    fine = fine_complex.real
    check_k = np.fft.fftn(fine, norm="ortho")
    inherited_error = float(
        np.max(
            np.abs(
                check_k[np.ix_(fine_idx, fine_idx, fine_idx)]
                - coarse_k[np.ix_(coarse_idx, coarse_idx, coarse_idx)]
            )
        )
    )
    if inherited_error > 5.0e-12:
        raise PhaseCError("nested coarse low modes are not exact")
    return fine, coarse, {
        "inherited_mode_count": int(coarse_idx.size**3),
        "coarse_non_nyquist_mode_count": int((coarse_n - 1) ** 3),
        "max_inherited_complex_coefficient_error": inherited_error,
        "max_inverse_fft_imaginary_part": imaginary_max,
        "fine_white_mean": float(fine.mean()),
        "fine_white_std": float(fine.std()),
    }


def block_sum(field: np.ndarray, coarse_n: int) -> np.ndarray:
    field = np.asarray(field)
    fine_n = field.shape[0]
    if field.shape[:3] != (fine_n,) * 3 or fine_n % coarse_n:
        raise PhaseCError("invalid block-sum shape")
    ratio = fine_n // coarse_n
    trailing = field.shape[3:]
    shaped = field.reshape(
        coarse_n, ratio, coarse_n, ratio, coarse_n, ratio, *trailing
    )
    return shaped.sum(axis=(1, 3, 5))


def cic_read_vector(grid: np.ndarray, positions: np.ndarray, box_size: float) -> np.ndarray:
    """Periodic trilinear read of a vector-valued mesh."""

    grid = np.asarray(grid, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    n = grid.shape[0]
    if grid.shape != (n, n, n, 3) or positions.ndim != 2 or positions.shape[1] != 3:
        raise PhaseCError("invalid CIC-read input")
    x = (positions % box_size) / (box_size / n)
    i0 = np.floor(x).astype(np.int64)
    frac = x - i0
    out = np.zeros((positions.shape[0], 3), dtype=np.float64)
    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            for dz in (0, 1):
                wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                out += (wx * wy * wz)[:, None] * grid[
                    (i0[:, 0] + dx) % n,
                    (i0[:, 1] + dy) % n,
                    (i0[:, 2] + dz) % n,
                ]
    return out


def tsc_deposit_numpy(
    masses: np.ndarray,
    positions: np.ndarray,
    n: int,
    box_size: float,
) -> np.ndarray:
    """Periodic conservative TSC deposit for one scalar source field."""

    mass = np.asarray(masses, dtype=np.float64).reshape(-1)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if mass.size != pos.shape[0] or np.any(mass < 0) or not np.all(np.isfinite(mass)):
        raise PhaseCError("invalid truth-deposition input")
    spacing = box_size / n
    cell = (pos % box_size) / spacing - 0.5
    nearest = np.floor(cell + 0.5).astype(np.int64)
    offset = cell - nearest

    def weights(component: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            0.5 * (0.5 - component) ** 2,
            0.75 - component**2,
            0.5 * (0.5 + component) ** 2,
        )

    wx, wy, wz = (weights(offset[:, axis]) for axis in range(3))
    out = np.zeros(n**3, dtype=np.float64)
    for ix, dx in enumerate((-1, 0, 1)):
        for iy, dy in enumerate((-1, 0, 1)):
            for iz, dz in enumerate((-1, 0, 1)):
                ii = (nearest[:, 0] + dx) % n
                jj = (nearest[:, 1] + dy) % n
                kk = (nearest[:, 2] + dz) % n
                flat = (ii * n + jj) * n + kk
                out += np.bincount(
                    flat,
                    weights=mass * wx[ix] * wy[iy] * wz[iz],
                    minlength=n**3,
                )
    return out.reshape((n, n, n))


def _load_selection(program: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    datum_path = program["input_bindings"]["Phase_A_datum"]["path"]
    quadrature_path = program["input_bindings"]["quadrature_arrays"]["path"]
    with np.load(datum_path, allow_pickle=False) as datum:
        response6 = np.asarray(datum["raw_selection_exposure"], dtype=np.float64)
    with np.load(quadrature_path, allow_pickle=False) as quad:
        response4 = np.asarray(quad["raw_selection_order4"], dtype=np.float64)
        response6_check = np.asarray(quad["raw_selection_order6"], dtype=np.float64)
    expected = (6, fixed.N, fixed.N, fixed.N)
    if response6.shape != expected or response4.shape != expected:
        raise PhaseCError("selection response shape mismatch")
    if not np.array_equal(response6, response6_check):
        raise PhaseCError("Phase-A and quadrature order-6 responses differ")
    for response in (response6, response4):
        if not np.all(np.isfinite(response)) or np.min(response) < -2e-14:
            raise PhaseCError("selection response is invalid")
    return np.clip(response6, 0.0, 1.0), np.clip(response4, 0.0, 1.0)


def _published_prior_arrays(program: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    prior = program["external_population_prior"]
    original_cell = float(prior["published_cell_size_cMpc_h"])
    ratio = (fixed.BOX_SIZE / fixed.N / original_cell) ** 3
    nbar = np.asarray(prior["published_mean_count_per_original_voxel"], dtype=np.float64) * ratio
    bias = np.asarray(prior["published_bias"], dtype=np.float64)
    if nbar.shape != (6,) or bias.shape != (6,) or np.any(nbar <= 0) or np.any(bias <= 0):
        raise PhaseCError("external population prior arrays are invalid")
    return nbar, bias


def selection_bases(n: int, box_size: float) -> tuple[np.ndarray, np.ndarray]:
    spacing = box_size / n
    axis = (np.arange(n) + 0.5) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    rel = np.stack((x, y, z), axis=-1) - box_size / 2.0
    radius = np.linalg.norm(rel, axis=-1)
    radial = np.clip((radius - 90.0) / 90.0, -1.0, 1.0)
    angular = np.where(radius > 0, rel[..., 2] / radius, 0.0)
    radial -= radial.mean()
    angular -= angular.mean()
    return radial, angular


def build_pmwd_truth(
    fine_white: np.ndarray,
    program: Mapping[str, object],
) -> dict[str, np.ndarray | float]:
    import jax
    import jax.numpy as jnp
    from pmwd import Configuration, SimpleLCDM, boltzmann, linear_modes, lpt, nbody, scatter

    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise PhaseCError("Phase-C truth generation requires the allocated Slurm GPU")
    n = int(program["grid"]["truth_N"])
    spacing = float(program["grid"]["truth_cell_size_cMpc_h"])
    cosmology = program["cosmology"]
    conf = Configuration(
        ptcl_spacing=spacing,
        ptcl_grid_shape=(n,) * 3,
        mesh_shape=1,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    cosmo = boltzmann(
        SimpleLCDM(
            conf,
            Omega_m=float(cosmology["Om"]),
            Omega_b=float(cosmology["Ob"]),
            h=float(cosmology["h"]),
            A_s_1e9=float(cosmology["A_s_1e9"]),
            n_s=float(cosmology["ns"]),
        ),
        conf,
    )

    @jax.jit
    def forward(white):
        modes = linear_modes(white, cosmo, conf)
        particles, observables = lpt(modes, cosmo, conf)
        particles, observables = nbody(particles, observables, cosmo, conf)
        density = scatter(particles, conf)
        momentum = scatter(particles, conf, val=particles.vel * 100.0)
        return density, momentum

    density_j, momentum_j = forward(jnp.asarray(fine_white, dtype=jnp.float64))
    density = np.asarray(density_j, dtype=np.float64)
    momentum = np.asarray(momentum_j, dtype=np.float64)
    if density.shape != (n, n, n) or momentum.shape != (n, n, n, 3):
        raise PhaseCError("PMWD truth output shape mismatch")
    if not np.all(np.isfinite(density)) or not np.all(np.isfinite(momentum)):
        raise PhaseCError("PMWD truth contains nonfinite values")
    if np.any(density < 0.0) or not math.isclose(float(density.mean()), 1.0, rel_tol=2e-12):
        raise PhaseCError("PMWD density is not a conservative nonnegative mass field")
    velocity = np.divide(
        momentum,
        density[..., None],
        out=np.zeros_like(momentum),
        where=density[..., None] > 1.0e-10,
    )
    coarse_mass = block_sum(density, fixed.N)
    coarse_momentum = block_sum(momentum, fixed.N)
    coarse_velocity = np.divide(
        coarse_momentum,
        coarse_mass[..., None],
        out=np.zeros_like(coarse_momentum),
        where=coarse_mass[..., None] > 1.0e-10,
    )
    ratio = n // fixed.N
    coarse_density = coarse_mass / ratio**3
    return {
        "fine_density": density,
        "fine_velocity": velocity,
        "coarse_delta": coarse_density - 1.0,
        "coarse_velocity": coarse_velocity,
        "density_min": float(density.min()),
        "density_max": float(density.max()),
        "empty_velocity_cell_count": int(np.count_nonzero(density <= 1.0e-10)),
    }


def truth_intensity(
    arm: str,
    truth: Mapping[str, object],
    response6: np.ndarray,
    response4: np.ndarray,
    nbar: np.ndarray,
    bias: np.ndarray,
    program: Mapping[str, object],
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Generate the frozen high-resolution truth intensity for one stress arm."""

    fine_density = np.asarray(truth["fine_density"], dtype=np.float64)
    fine_velocity = np.asarray(truth["fine_velocity"], dtype=np.float64)
    fine_n = fine_density.shape[0]
    coarse_n = response6.shape[1]
    box_size = float(program["grid"]["box_size_cMpc_h"])
    fine_spacing = box_size / fine_n
    axis = (np.arange(fine_n) + 0.5) * fine_spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    centres = np.stack((x, y, z), axis=-1)
    relative = centres - box_size / 2.0
    radius = np.linalg.norm(relative, axis=-1)
    rhat = relative / radius[..., None]
    radial_velocity = np.sum(fine_velocity * rhat, axis=-1)
    eta = np.log(np.maximum(fine_density, float(program["truth_model"]["density_floor"])))
    eta -= eta.mean()
    radial_basis, angular_basis = selection_bases(coarse_n, box_size)

    if arm == "D":
        discrepancy_rng = tagged_rng(seed, int(program["rng_tags"]["smooth_discrepancy"]))
        noise = discrepancy_rng.standard_normal((fine_n,) * 3)
        freq = 2.0 * np.pi * np.fft.fftfreq(fine_n, d=fine_spacing)
        kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")
        smooth = np.exp(-0.5 * (kx**2 + ky**2 + kz**2) * float(program["stress"]["D_smooth_scale_cMpc_h"]) ** 2)
        discrepancy = np.fft.ifftn(np.fft.fftn(noise) * smooth).real
        discrepancy /= max(float(discrepancy.std()), np.finfo(float).tiny)
    else:
        discrepancy = np.zeros_like(eta)

    coherent = arm in {"B", "C", "D"}
    fog = np.asarray(program["stress"]["truth_FoG_km_s"], dtype=np.float64)
    zerr = np.asarray(program["stress"]["truth_redshift_error_km_s"], dtype=np.float64)
    use_broadening = arm in {"C", "D"}
    response = response6
    if arm == "D":
        selection_log = (
            float(program["stress"]["D_radial_selection_amplitude"]) * radial_basis
            + float(program["stress"]["D_angular_selection_amplitude"]) * angular_basis
        )
        response = np.clip(response4 * np.exp(selection_log)[None, ...], 0.0, 1.0)

    intensity = np.empty_like(response6)
    quadrature_offsets = np.asarray(program["stress"]["Gaussian_radial_quadrature_offsets_sigma"], dtype=np.float64)
    quadrature_weights = np.asarray(program["stress"]["Gaussian_radial_quadrature_weights"], dtype=np.float64)
    if not math.isclose(float(quadrature_weights.sum()), 1.0, rel_tol=0.0, abs_tol=2e-15):
        raise PhaseCError("truth radial quadrature is not normalized")
    total_fine_cells = fine_n**3
    for population in range(6):
        log_mass = bias[population] * eta
        if arm == "D":
            log_mass += float(program["stress"]["D_discrepancy_log_amplitude"]) * discrepancy
        mass = np.exp(np.clip(log_mass, -30.0, 30.0))
        mass *= (nbar[population] * coarse_n**3) / mass.sum()
        base_displacement = radial_velocity / 100.0 if coherent else np.zeros_like(radial_velocity)
        if use_broadening:
            sigma = math.sqrt(fog[population] ** 2 + zerr[population] ** 2) / 100.0
            deposited = np.zeros((coarse_n,) * 3, dtype=np.float64)
            for node, weight in zip(quadrature_offsets, quadrature_weights, strict=True):
                positions = (centres + (base_displacement + node * sigma)[..., None] * rhat) % box_size
                deposited += weight * tsc_deposit_numpy(mass, positions, coarse_n, box_size)
        elif coherent:
            positions = (centres + base_displacement[..., None] * rhat) % box_size
            deposited = tsc_deposit_numpy(mass, positions, coarse_n, box_size)
        else:
            deposited = block_sum(mass, coarse_n)
        if not math.isclose(float(deposited.sum()), float(mass.sum()), rel_tol=2e-13):
            raise PhaseCError("truth count deposition is not conservative")
        intensity[population] = response[population] * deposited

    if not np.all(np.isfinite(intensity)) or np.any(intensity < 0.0):
        raise PhaseCError("truth intensity is invalid")
    return intensity, {
        "fine_source_cell_count": total_fine_cells,
        "coherent_RSD": coherent,
        "population_FoG_and_redshift_error": use_broadening,
        "negative_binomial_overdispersion": arm == "D",
        "selection_order": 4 if arm == "D" else 6,
        "selection_perturbed": arm == "D",
        "smooth_log_intensity_discrepancy": arm == "D",
        "expected_population_totals_after_selection": intensity.sum(axis=(1, 2, 3)).tolist(),
    }


def generate_mock_data(
    arm: str,
    seed: int,
    intensity: np.ndarray,
    truth: Mapping[str, object],
    design: Mapping[str, np.ndarray],
    program: Mapping[str, object],
) -> dict[str, np.ndarray]:
    count_rng = tagged_rng(seed, int(program["rng_tags"]["count_noise"]))
    if arm == "D":
        dispersion = float(program["stress"]["D_negative_binomial_shape"])
        gamma_factor = count_rng.gamma(shape=dispersion, scale=1.0 / dispersion, size=intensity.shape)
    else:
        gamma_factor = np.ones_like(intensity)
    realized_intensity = intensity * gamma_factor
    train_fraction = float(program["heldout"]["training_fraction"])
    holdout_fraction = 1.0 - train_fraction
    counts_train = count_rng.poisson(train_fraction * realized_intensity).astype(np.int64)
    counts_holdout = count_rng.poisson(holdout_fraction * realized_intensity).astype(np.int64)

    fine_velocity = np.asarray(truth["fine_velocity"], dtype=np.float64)
    sampled = cic_read_vector(fine_velocity, np.asarray(design["pos"]), fixed.BOX_SIZE)
    radial_signal = np.sum(sampled * np.asarray(design["rhat"]), axis=1)
    q_rng = tagged_rng(seed, int(program["rng_tags"]["velocity_nuisance"]))
    noise_rng = tagged_rng(seed, int(program["rng_tags"]["velocity_noise"]))
    q_truth = q_rng.standard_normal(4) * np.asarray(design["q_std"], dtype=np.float64)
    noise = noise_rng.standard_normal(radial_signal.size) * np.sqrt(np.asarray(design["variance"]))
    velocity_mock = radial_signal + np.asarray(design["B"]) @ q_truth + noise
    return {
        "counts_train": counts_train,
        "counts_holdout": counts_holdout,
        "truth_count_intensity": intensity,
        "realized_count_intensity": realized_intensity,
        "velocity_mock": velocity_mock,
        "truth_velocity_radial_signal": radial_signal,
        "truth_velocity_nuisance_q": q_truth,
        "truth_velocity_noise": noise,
    }


def build_roi_weights(program: Mapping[str, object]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Integrate the frozen raised-cosine ROI windows over N32 voxels."""

    roi_path = Path(program["input_bindings"]["ROI_design"]["path"])
    payload = json.loads(roi_path.read_text())
    rois = payload["ROI_geometry"]["ROIs"]
    n = fixed.N
    box_size = fixed.BOX_SIZE
    spacing = box_size / n
    q = int(program["diagnostics"]["ROI_subpoints_per_axis"])
    offsets = (np.arange(q) + 0.5) / q
    weights = []
    names = []

    def sphere_window(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
        displacement = (points - center + box_size / 2.0) % box_size - box_size / 2.0
        r = np.linalg.norm(displacement, axis=-1)
        out = np.zeros_like(r)
        inner = r <= 0.75 * radius
        taper = (r > 0.75 * radius) & (r <= radius)
        out[inner] = 1.0
        out[taper] = 0.5 * (1.0 + np.cos(np.pi * (r[taper] - 0.75 * radius) / (0.25 * radius)))
        return out

    cell_index = np.indices((n, n, n), dtype=np.float64).transpose(1, 2, 3, 0)
    for roi in rois:
        accumulator = np.zeros((n, n, n), dtype=np.float64)
        for ox in offsets:
            for oy in offsets:
                for oz in offsets:
                    points = (cell_index + np.array([ox, oy, oz])) * spacing
                    if roi["geometry"] == "sphere":
                        sample = sphere_window(
                            points,
                            np.asarray(roi["center_cMpc_h"], dtype=np.float64),
                            float(roi["radius_cMpc_h"]),
                        )
                    elif roi["geometry"] == "union_max_of_spheres":
                        sample = np.zeros((n, n, n), dtype=np.float64)
                        radius = float(roi["component_radius_cMpc_h"])
                        for component in roi["component_centers_from_observer_plus_source_offsets_cMpc_h"]:
                            sample = np.maximum(
                                sample,
                                sphere_window(points, np.asarray(component["center_cMpc_h"]), radius),
                            )
                    else:
                        raise PhaseCError(f"unsupported frozen ROI geometry: {roi['geometry']}")
                    accumulator += sample
        accumulator /= q**3
        if accumulator.sum() <= 0.0:
            raise PhaseCError(f"frozen ROI has zero N32 quadrature weight: {roi['id']}")
        names.append(str(roi["id"]))
        weights.append(accumulator)
    return names, np.asarray(weights), np.asarray([weight.sum() for weight in weights])


def _jax_tsc_deposit_one(mass, positions, n: int, spacing: float):
    import jax.numpy as jnp

    flat_mass = mass.reshape(-1)
    flat_pos = positions.reshape((-1, 3))
    cell = flat_pos / spacing - 0.5
    nearest = jnp.floor(cell + 0.5).astype(jnp.int32)
    offset = cell - nearest

    def weights(component):
        return (
            0.5 * (0.5 - component) ** 2,
            0.75 - component**2,
            0.5 * (0.5 + component) ** 2,
        )

    wx, wy, wz = (weights(offset[:, axis]) for axis in range(3))
    out = jnp.zeros((n, n, n), dtype=mass.dtype)
    for ix, dx in enumerate((-1, 0, 1)):
        for iy, dy in enumerate((-1, 0, 1)):
            for iz, dz in enumerate((-1, 0, 1)):
                ii = (nearest[:, 0] + dx) % n
                jj = (nearest[:, 1] + dy) % n
                kk = (nearest[:, 2] + dz) % n
                out = out.at[ii, jj, kk].add(flat_mass * wx[ix] * wy[iy] * wz[iz])
    return out


def build_inference_model(
    program: Mapping[str, object],
    response6_np: np.ndarray,
    mock: Mapping[str, np.ndarray],
    design: Mapping[str, np.ndarray],
):
    import jax
    import jax.numpy as jnp

    args = fixed.frozen_args(program["input_bindings"]["CF4_catalog"]["path"])
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise PhaseCError("no JAX CPU device is available for deterministic transfer construction")
    # The transfer is a small, one-time cosmology table.  Construct it on the
    # CPU so GPU kernel autotuning cannot alter or block this deterministic
    # source calculation; the likelihood and every gradient remain on GPU.
    with jax.default_device(cpu_devices[0]):
        transfer_np, growth_rate = fixed.build_density_transfer(args)
    growth_relative_difference = 0.0
    growth_tolerance = 0.0
    n = fixed.N
    box_size = fixed.BOX_SIZE
    spacing = box_size / n
    field_size = n**3
    transfer = jnp.asarray(transfer_np)
    response6 = jnp.asarray(response6_np)
    counts_train = jnp.asarray(mock["counts_train"])
    velocity_mock = jnp.asarray(mock["velocity_mock"])
    train_rows = jnp.asarray(~np.asarray(design["holdout"], dtype=bool))
    B = jnp.asarray(design["B"], dtype=jnp.float64)
    q_std_np = np.asarray(design["q_std"], dtype=np.float64)
    q_std = jnp.asarray(q_std_np)
    variance = jnp.asarray(design["variance"], dtype=jnp.float64)
    nbar_np, bias_np = _published_prior_arrays(program)
    alpha_mean_np = np.log(nbar_np)
    logbias_mean_np = np.log(bias_np)
    logfog_mean_np = np.log(np.asarray(program["inference_model"]["FoG_prior_median_km_s"], dtype=np.float64))
    alpha_mean = jnp.asarray(alpha_mean_np)
    logbias_mean = jnp.asarray(logbias_mean_np)
    logfog_mean = jnp.asarray(logfog_mean_np)
    radial_basis_np, angular_basis_np = selection_bases(n, box_size)
    selection_basis = jnp.asarray(np.stack((radial_basis_np, angular_basis_np)))
    selection_amplitude = float(program["inference_model"]["selection_basis_log_amplitude"])
    redshift_error = jnp.asarray(program["inference_model"]["fixed_redshift_error_km_s"], dtype=jnp.float64)
    quadrature_offsets = jnp.asarray(program["inference_model"]["Gaussian_radial_quadrature_offsets_sigma"], dtype=jnp.float64)
    quadrature_weights = jnp.asarray(program["inference_model"]["Gaussian_radial_quadrature_weights"], dtype=jnp.float64)
    train_fraction = float(program["heldout"]["training_fraction"])

    axis = (jnp.arange(n, dtype=jnp.float64) + 0.5) * spacing
    gx, gy, gz = jnp.meshgrid(axis, axis, axis, indexing="ij")
    centres = jnp.stack((gx, gy, gz), axis=-1)
    relative = centres - box_size / 2.0
    radius = jnp.linalg.norm(relative, axis=-1)
    rhat_cells = relative / radius[..., None]

    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=spacing)
    rfreq = 2.0 * np.pi * np.fft.rfftfreq(n, d=spacing)
    kx_np, ky_np, kz_np = np.meshgrid(freq, freq, rfreq, indexing="ij")
    kvec = tuple(jnp.asarray(value) for value in (kx_np, ky_np, kz_np))
    k2 = kvec[0] ** 2 + kvec[1] ** 2 + kvec[2] ** 2
    k2_safe = jnp.where(k2 > 0.0, k2, 1.0)

    def white_to_delta(white):
        delta_k = jnp.fft.fftn(white.reshape((n, n, n)), norm="ortho") * transfer
        return jnp.fft.ifftn(delta_k, norm="ortho").real

    def delta_to_velocity(delta):
        delta_k = jnp.fft.rfftn(delta)
        pieces = []
        for component in kvec:
            velocity_k = 1j * 100.0 * growth_rate * component / k2_safe * delta_k
            velocity_k = jnp.where(k2 > 0.0, velocity_k, 0.0)
            pieces.append(jnp.fft.irfftn(velocity_k, s=(n, n, n)))
        return jnp.stack(pieces)

    # Use the same CPU-sourced transfer and growth scalar for both the count
    # and fixed-geometry CF4 operators.  The evaluation itself remains on GPU.
    catalog_positions = jnp.asarray(design["pos"], dtype=jnp.float64)
    catalog_rhat = jnp.asarray(design["rhat"], dtype=jnp.float64)
    catalog_cell = (catalog_positions % box_size) / spacing
    catalog_i0 = jnp.floor(catalog_cell).astype(jnp.int32)
    catalog_frac = catalog_cell - catalog_i0

    def cic_read_catalog(grid):
        values = jnp.zeros(catalog_positions.shape[0], dtype=grid.dtype)
        for dx in (0, 1):
            wx = catalog_frac[:, 0] if dx else 1.0 - catalog_frac[:, 0]
            for dy in (0, 1):
                wy = catalog_frac[:, 1] if dy else 1.0 - catalog_frac[:, 1]
                for dz in (0, 1):
                    wz = catalog_frac[:, 2] if dz else 1.0 - catalog_frac[:, 2]
                    values = values + wx * wy * wz * grid[
                        (catalog_i0[:, 0] + dx) % n,
                        (catalog_i0[:, 1] + dy) % n,
                        (catalog_i0[:, 2] + dz) % n,
                    ]
        return values

    def catalog_forward(white):
        velocity = delta_to_velocity(white_to_delta(white))
        components = [cic_read_catalog(velocity[component]) for component in range(3)]
        return (
            components[0] * catalog_rhat[:, 0]
            + components[1] * catalog_rhat[:, 1]
            + components[2] * catalog_rhat[:, 2]
        )

    A = jax.jit(catalog_forward)

    nuisance_size = 24

    def unpack(vector):
        white = vector[:field_size].reshape((n, n, n))
        offset = field_size
        alpha = vector[offset : offset + 6]
        offset += 6
        logbias = vector[offset : offset + 6]
        offset += 6
        logfog = vector[offset : offset + 6]
        offset += 6
        selection_unit = vector[offset : offset + 2]
        offset += 2
        q_unit = vector[offset : offset + 4]
        return white, alpha, logbias, logfog, selection_unit, q_unit

    def count_lambda(vector, response_scale=1.0):
        white, alpha, logbias, logfog, selection_unit, _q_unit = unpack(vector)
        delta = white_to_delta(white)
        eta = delta - jnp.mean(delta)
        velocity = delta_to_velocity(delta)
        velocity_last = jnp.moveaxis(velocity, 0, -1)
        radial_velocity = jnp.sum(velocity_last * rhat_cells, axis=-1)
        coherent_displacement = radial_velocity / 100.0
        response_calibration = jnp.exp(
            selection_amplitude
            * jnp.sum(selection_unit[:, None, None, None] * selection_basis, axis=0)
        )
        bias = jnp.exp(logbias)
        fog = jnp.exp(logfog)
        populations = []
        for population in range(6):
            mass = jnp.exp(alpha[population] + bias[population] * eta)
            sigma_displacement = jnp.sqrt(fog[population] ** 2 + redshift_error[population] ** 2) / 100.0
            pushed = jnp.zeros((n, n, n), dtype=mass.dtype)
            for node, weight in zip(quadrature_offsets, quadrature_weights, strict=True):
                positions = (
                    centres
                    + (coherent_displacement + node * sigma_displacement)[..., None] * rhat_cells
                ) % box_size
                pushed = pushed + weight * _jax_tsc_deposit_one(mass, positions, n, spacing)
            populations.append(response_scale * response6[population] * response_calibration * pushed)
        return jnp.stack(populations), delta, velocity

    def negative_log_posterior(vector):
        white, alpha, logbias, logfog, selection_unit, q_unit = unpack(vector)
        intensity, _delta, _velocity = count_lambda(vector, response_scale=train_fraction)
        support = response6 > 0.0
        safe = jnp.where(support, intensity, 1.0)
        count_nll = jnp.sum(jnp.where(support, intensity - counts_train * jnp.log(safe), 0.0))
        velocity_model = A(white) + B @ (q_std * q_unit)
        residual = velocity_mock - velocity_model
        velocity_nll = 0.5 * jnp.sum(jnp.where(train_rows, residual**2 / variance, 0.0))
        prior = 0.5 * jnp.sum(white**2) + 0.5 * jnp.sum(q_unit**2) + 0.5 * jnp.sum(selection_unit**2)
        prior += 0.5 * jnp.sum(((alpha - alpha_mean) / float(program["inference_model"]["alpha_log_sigma"])) ** 2)
        prior += 0.5 * jnp.sum(((logbias - logbias_mean) / float(program["inference_model"]["bias_log_sigma"])) ** 2)
        prior += 0.5 * jnp.sum(((logfog - logfog_mean) / float(program["inference_model"]["FoG_log_sigma"])) ** 2)
        return prior + count_nll + velocity_nll

    initial = np.concatenate(
        (
            np.zeros(field_size, dtype=np.float64),
            alpha_mean_np,
            logbias_mean_np,
            logfog_mean_np,
            np.zeros(2, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
        )
    )
    metadata = {
        "field_size": field_size,
        "nuisance_size": nuisance_size,
        "alpha_mean": alpha_mean_np,
        "logbias_mean": logbias_mean_np,
        "logfog_mean": logfog_mean_np,
        "transfer": transfer_np,
        "growth_rate": growth_rate,
        "CPU_transfer_growth_rate": growth_rate,
        "cross_device_growth_relative_difference": growth_relative_difference,
        "cross_device_growth_relative_tolerance": growth_tolerance,
        "q_std": q_std_np,
        "A": A,
        "B": B,
        "variance": variance,
        "train_rows": train_rows,
    }
    return negative_log_posterior, count_lambda, initial, metadata


def run_four_chains(
    negative_log_posterior: Callable,
    initial: np.ndarray,
    program: Mapping[str, object],
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray | list[float]]]:
    import blackjax
    import jax
    import jax.numpy as jnp
    from blackjax.adaptation.base import get_filter_adapt_info_fn
    from scipy.optimize import minimize

    sampler = program["sampler"]
    value_and_grad = jax.jit(jax.value_and_grad(negative_log_posterior))

    def scipy_value_gradient(vector):
        value, gradient = value_and_grad(jnp.asarray(vector))
        return float(value), np.asarray(gradient, dtype=np.float64)

    optimization = minimize(
        scipy_value_gradient,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": int(sampler["MAP_max_iterations"]),
            "maxls": int(sampler["MAP_max_line_search"]),
            "ftol": 0.0,
            "gtol": 0.0,
        },
    )
    map_position = np.asarray(optimization.x, dtype=np.float64)
    map_value, map_gradient = value_and_grad(jnp.asarray(map_position))
    if not math.isfinite(float(map_value)) or not np.all(np.isfinite(np.asarray(map_gradient))):
        raise PhaseCError("MAP state is nonfinite")

    filter_info = get_filter_adapt_info_fn(
        info_keys={"acceptance_rate", "is_divergent", "energy"},
        adapt_state_keys={"step_size"},
    )
    adaptation = blackjax.window_adaptation(
        blackjax.hmc,
        lambda vector: -negative_log_posterior(vector),
        is_mass_matrix_diagonal=True,
        initial_step_size=float(sampler["initial_step_size"]),
        target_acceptance_rate=float(sampler["target_acceptance_rate"]),
        progress_bar=False,
        adaptation_info_fn=filter_info,
        num_integration_steps=int(sampler["integration_steps"]),
    )
    warmup_steps = int(sampler["warmup_steps"])
    draw_count = int(sampler["posterior_draws_per_chain"])
    chain_count = int(sampler["chain_count"])
    chain_draws = []
    step_sizes = []
    warmup_acceptance = []
    warmup_divergences = []
    sample_acceptance = []
    sample_divergences = []
    sample_energy = []
    sample_logdensity = []
    initial_rng = tagged_rng(seed, int(program["rng_tags"]["chain_initialization"]))

    for chain in range(chain_count):
        start = map_position + float(sampler["chain_initial_jitter_std"]) * initial_rng.standard_normal(initial.size)
        warmup_key = jax.random.PRNGKey(np.uint32(seed + 1009 * (chain + 1)))
        (adapted, warmup_info) = adaptation.run(warmup_key, jnp.asarray(start), num_steps=warmup_steps)
        parameters = adapted.parameters
        kernel = blackjax.hmc(
            lambda vector: -negative_log_posterior(vector),
            step_size=parameters["step_size"],
            inverse_mass_matrix=parameters["inverse_mass_matrix"],
            num_integration_steps=int(sampler["integration_steps"]),
        )

        @jax.jit
        def sample_chain(key, state):
            keys = jax.random.split(key, draw_count)

            def one_step(current, transition_key):
                next_state, info = kernel.step(transition_key, current)
                record = (
                    next_state.position,
                    next_state.logdensity,
                    info.acceptance_rate,
                    info.is_divergent,
                    info.energy,
                )
                return next_state, record

            return jax.lax.scan(one_step, state, keys)

        sample_key = jax.random.PRNGKey(np.uint32(seed + 9173 * (chain + 1)))
        _last, records = sample_chain(sample_key, adapted.state)
        positions, logdensity, acceptance, divergent, energy = records
        positions_np = np.asarray(positions, dtype=np.float32)
        if not np.all(np.isfinite(positions_np)):
            raise PhaseCError(f"chain {chain} contains nonfinite positions")
        chain_draws.append(positions_np)
        step_sizes.append(float(parameters["step_size"]))
        warmup_acceptance.append(float(np.mean(np.asarray(warmup_info.info.acceptance_rate))))
        warmup_divergences.append(int(np.count_nonzero(np.asarray(warmup_info.info.is_divergent))))
        sample_acceptance.append(np.asarray(acceptance, dtype=np.float64))
        sample_divergences.append(np.asarray(divergent, dtype=bool))
        sample_energy.append(np.asarray(energy, dtype=np.float64))
        sample_logdensity.append(np.asarray(logdensity, dtype=np.float64))

    return np.asarray(chain_draws, dtype=np.float32), {
        "MAP_value": np.asarray(float(map_value)),
        "MAP_gradient_norm": np.asarray(float(np.linalg.norm(np.asarray(map_gradient)))),
        "MAP_iterations": np.asarray(int(optimization.nit)),
        "MAP_status": np.asarray(int(optimization.status)),
        "step_size": np.asarray(step_sizes),
        "warmup_mean_acceptance": np.asarray(warmup_acceptance),
        "warmup_divergence_count": np.asarray(warmup_divergences),
        "acceptance_rate": np.asarray(sample_acceptance),
        "is_divergent": np.asarray(sample_divergences),
        "energy": np.asarray(sample_energy),
        "logdensity": np.asarray(sample_logdensity),
    }


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    from scipy.stats import norm, rankdata

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    ranks = rankdata(flat, method="average")
    probability = (ranks - 0.375) / (flat.size + 0.25)
    return norm.ppf(probability).reshape(values.shape)


def _split_chains(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 4:
        raise PhaseCError("chain diagnostic input is invalid")
    half = values.shape[1] // 2
    return np.concatenate((values[:, :half], values[:, -half:]), axis=0)


def _basic_rhat(values: np.ndarray) -> float:
    chains = _split_chains(values)
    n = chains.shape[1]
    between = n * np.var(chains.mean(axis=1), ddof=1)
    within = np.mean(np.var(chains, axis=1, ddof=1))
    if within <= 0.0:
        return 1.0 if between <= 0.0 else float("inf")
    variance = (n - 1.0) / n * within + between / n
    return float(np.sqrt(variance / within))


def rank_normalized_rhat(values: np.ndarray) -> float:
    ranked = _rank_normalize(values)
    folded = _rank_normalize(np.abs(values - np.median(values)))
    return max(_basic_rhat(ranked), _basic_rhat(folded))


def _autocovariance_fft(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.mean()
    n = values.size
    size = 1 << (2 * n - 1).bit_length()
    transformed = np.fft.rfft(values, n=size)
    result = np.fft.irfft(transformed * np.conjugate(transformed), n=size)[:n]
    return result / np.arange(n, 0, -1)


def effective_sample_size(values: np.ndarray) -> float:
    chains = _split_chains(values)
    m, n = chains.shape
    chain_var = np.var(chains, axis=1, ddof=1)
    within = float(np.mean(chain_var))
    between = float(n * np.var(chains.mean(axis=1), ddof=1))
    variance_plus = (n - 1.0) / n * within + between / n
    if variance_plus <= 0.0:
        return float(m * n)
    autocov = np.asarray([_autocovariance_fft(chain) for chain in chains])
    rho = np.ones(n, dtype=np.float64)
    for lag in range(1, n):
        rho[lag] = 1.0 - (within - float(np.mean(autocov[:, lag]))) / variance_plus
    pair_sums = []
    for lag in range(1, n - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if pair < 0.0:
            break
        pair_sums.append(pair)
    for index in range(1, len(pair_sums)):
        pair_sums[index] = min(pair_sums[index], pair_sums[index - 1])
    tau = max(1.0, -1.0 + 2.0 * (1.0 + sum(pair_sums)))
    return float(min(m * n, m * n / tau))


def chain_diagnostics(values: np.ndarray, names: Sequence[str]) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != len(names):
        raise PhaseCError("diagnostic projection array shape mismatch")
    rows = []
    for index, name in enumerate(names):
        raw = values[:, :, index]
        ranked = _rank_normalize(raw)
        q05, q95 = np.quantile(raw, [0.05, 0.95])
        lower = (raw <= q05).astype(np.float64)
        upper = (raw >= q95).astype(np.float64)
        rows.append(
            {
                "name": str(name),
                "rank_normalized_split_Rhat": rank_normalized_rhat(raw),
                "bulk_ESS": effective_sample_size(ranked),
                "tail_ESS": min(effective_sample_size(lower), effective_sample_size(upper)),
            }
        )
    return {
        "parameters": rows,
        "max_Rhat": max(row["rank_normalized_split_Rhat"] for row in rows),
        "min_bulk_ESS": min(row["bulk_ESS"] for row in rows),
        "min_tail_ESS": min(row["tail_ESS"] for row in rows),
    }


def density_velocity_samples(
    white_draws: np.ndarray,
    transfer: np.ndarray,
    growth_rate: float,
    box_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform white draws to N32 linear density and velocity on the host."""

    shape = white_draws.shape
    n = transfer.shape[0]
    flat = white_draws.reshape((-1, n, n, n)).astype(np.float64)
    white_k = np.fft.fftn(flat, axes=(1, 2, 3), norm="ortho")
    delta_k = white_k * transfer[None, ...]
    density = np.fft.ifftn(delta_k, axes=(1, 2, 3), norm="ortho").real.astype(np.float32)
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=box_size / n)
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    safe = np.where(k2 > 0.0, k2, 1.0)
    density_k_default = np.fft.fftn(density.astype(np.float64), axes=(1, 2, 3))
    velocity = np.empty((flat.shape[0], 3, n, n, n), dtype=np.float32)
    for component, kval in enumerate((kx, ky, kz)):
        vk = 1j * 100.0 * growth_rate * kval / safe * density_k_default
        vk[:, 0, 0, 0] = 0.0
        velocity[:, component] = np.fft.ifftn(vk, axes=(1, 2, 3)).real.astype(np.float32)
    return density.reshape((*shape[:2], n, n, n)), velocity.reshape((*shape[:2], 3, n, n, n))


def k_shell_metrics(
    truth: np.ndarray,
    estimate: np.ndarray,
    samples: np.ndarray,
    transfer: np.ndarray,
    box_size: float,
    edges: np.ndarray,
) -> dict[str, np.ndarray]:
    n = truth.shape[0]
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=box_size / n)
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    truth_k = np.fft.fftn(truth, norm="ortho")
    estimate_k = np.fft.fftn(estimate, norm="ortho")
    sample_k = np.fft.fftn(samples.reshape((-1, n, n, n)), axes=(1, 2, 3), norm="ortho")
    rows = {key: [] for key in ("k_mean", "mode_count", "response", "cross_correlation", "residual_power", "posterior_variance_reduction")}
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (kmag >= lo) & (kmag < hi)
        mode_count = int(np.count_nonzero(mask))
        rows["k_mean"].append(float(kmag[mask].mean()) if mode_count else np.nan)
        rows["mode_count"].append(mode_count)
        if mode_count == 0:
            for key in ("response", "cross_correlation", "residual_power", "posterior_variance_reduction"):
                rows[key].append(np.nan)
            continue
        pt = float(np.mean(np.abs(truth_k[mask]) ** 2))
        pe = float(np.mean(np.abs(estimate_k[mask]) ** 2))
        cross = float(np.mean((estimate_k[mask] * np.conjugate(truth_k[mask])).real))
        residual = float(np.mean(np.abs(estimate_k[mask] - truth_k[mask]) ** 2))
        sample_var = float(np.mean(np.var(sample_k[:, mask], axis=0)))
        prior_var = float(np.mean(transfer[mask] ** 2))
        rows["response"].append(cross / max(pt, np.finfo(float).tiny))
        rows["cross_correlation"].append(cross / max(math.sqrt(pt * pe), np.finfo(float).tiny))
        rows["residual_power"].append(residual)
        rows["posterior_variance_reduction"].append(1.0 - sample_var / max(prior_var, np.finfo(float).tiny))
    return {key: np.asarray(value) for key, value in rows.items()}


def weighted_coverage(truth: np.ndarray, lower: np.ndarray, upper: np.ndarray, weights: np.ndarray) -> float:
    indicator = (truth >= lower) & (truth <= upper)
    return float(np.sum(weights * indicator) / np.sum(weights))


def poisson_logpmf(counts: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    counts = np.asarray(counts, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    safe = np.maximum(intensity, np.finfo(float).tiny)
    return counts * np.log(safe) - safe - gammaln(counts + 1.0)


def logmeanexp(values: np.ndarray, axis: int = 0) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(np.mean(np.exp(values - maximum), axis=axis))


def residual_spectral_diagnostics(
    fields: np.ndarray,
    box_size: float,
    k_edges: np.ndarray,
    separation_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-z-axis P0/P2 and isotropic 2PCF for six residual fields."""

    fields = np.asarray(fields, dtype=np.float64)
    if fields.ndim != 5 or fields.shape[1] != 6:
        raise PhaseCError("residual spectrum expects [draw,6,N,N,N]")
    draw_count, _, n, _, _ = fields.shape
    spacing = box_size / n
    freq = 2.0 * np.pi * np.fft.fftfreq(n, d=spacing)
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    mu = np.divide(kz, kmag, out=np.zeros_like(kmag), where=kmag > 0.0)
    legendre2 = 0.5 * (3.0 * mu**2 - 1.0)
    modes = np.fft.fftn(fields, axes=(2, 3, 4), norm="ortho")
    power = np.abs(modes) ** 2
    p0 = np.full((draw_count, 6, k_edges.size - 1), np.nan, dtype=np.float64)
    p2 = np.full_like(p0, np.nan)
    for index, (lo, hi) in enumerate(zip(k_edges[:-1], k_edges[1:], strict=True)):
        mask = (kmag >= lo) & (kmag < hi)
        if np.any(mask):
            p0[:, :, index] = np.mean(power[:, :, mask], axis=2)
            p2[:, :, index] = 5.0 * np.mean(power[:, :, mask] * legendre2[mask], axis=2)

    # Wiener-Khinchin correlation, radially averaged with minimum-image lags.
    correlation = np.fft.ifftn(power, axes=(2, 3, 4)).real / math.sqrt(n**3)
    lag = np.minimum(np.arange(n), n - np.arange(n)) * spacing
    lx, ly, lz = np.meshgrid(lag, lag, lag, indexing="ij")
    separation = np.sqrt(lx**2 + ly**2 + lz**2)
    xi = np.full((draw_count, 6, separation_edges.size - 1), np.nan, dtype=np.float64)
    for index, (lo, hi) in enumerate(zip(separation_edges[:-1], separation_edges[1:], strict=True)):
        mask = (separation >= lo) & (separation < hi)
        if np.any(mask):
            xi[:, :, index] = np.mean(correlation[:, :, mask], axis=2)
    return p0, p2, xi


def predictive_diagnostics(
    draws: np.ndarray,
    count_lambda: Callable,
    mock: Mapping[str, np.ndarray],
    response6: np.ndarray,
    design: Mapping[str, np.ndarray],
    model_meta: Mapping[str, object],
    program: Mapping[str, object],
    seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    import jax
    import jax.numpy as jnp

    flat_draws = draws.reshape((-1, draws.shape[-1]))
    predictive_count = int(program["diagnostics"]["posterior_predictive_draw_count"])
    indices = np.linspace(0, flat_draws.shape[0] - 1, predictive_count, dtype=int)
    selected = flat_draws[indices]
    holdout_fraction = float(program["heldout"]["holdout_fraction"])

    @jax.jit
    def one_lambda(vector):
        intensity, _delta, _velocity = count_lambda(vector, response_scale=holdout_fraction)
        return intensity

    lambdas = np.asarray([one_lambda(jnp.asarray(vector)) for vector in selected], dtype=np.float64)
    holdout_counts = np.asarray(mock["counts_holdout"], dtype=np.int64)
    model_cell_lpd = logmeanexp(poisson_logpmf(holdout_counts[None, ...], lambdas), axis=0)

    train_counts = np.asarray(mock["counts_train"], dtype=np.float64)
    train_fraction = float(program["heldout"]["training_fraction"])
    baseline_rate = np.empty(6, dtype=np.float64)
    baseline_lambda = np.empty_like(response6)
    for population in range(6):
        exposure_sum = float(response6[population].sum())
        baseline_rate[population] = (train_counts[population].sum() + 0.5) / (train_fraction * exposure_sum + 0.5)
        baseline_lambda[population] = holdout_fraction * baseline_rate[population] * response6[population]
    baseline_cell_lpd = poisson_logpmf(holdout_counts, baseline_lambda)
    delta_cell_lpd = model_cell_lpd - baseline_cell_lpd

    rng = tagged_rng(seed, int(program["rng_tags"]["posterior_predictive"]))
    replicated = np.asarray([rng.poisson(lam) for lam in lambdas], dtype=np.int64)
    histogram_edges = np.asarray(program["diagnostics"]["count_histogram_edges"], dtype=np.float64)
    observed_hist = np.asarray(
        [[np.histogram(holdout_counts[p].reshape(-1), bins=histogram_edges)[0] for p in range(6)]],
        dtype=np.int64,
    )[0]
    replicate_hist = np.asarray(
        [
            [np.histogram(sample[p].reshape(-1), bins=histogram_edges)[0] for p in range(6)]
            for sample in replicated
        ],
        dtype=np.int64,
    )
    observed_zero = np.mean(holdout_counts == 0, axis=(1, 2, 3))
    replicate_zero = np.mean(replicated == 0, axis=(2, 3, 4))

    n = fixed.N
    spacing = fixed.BOX_SIZE / n
    axis = (np.arange(n) + 0.5) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    radius = np.sqrt((x - 192.0) ** 2 + (y - 192.0) ** 2 + (z - 192.0) ** 2)
    radial_edges = np.asarray(program["diagnostics"]["radial_edges_cMpc_h"], dtype=np.float64)
    observed_radial = np.zeros((6, radial_edges.size - 1), dtype=np.float64)
    replicate_radial = np.zeros((predictive_count, 6, radial_edges.size - 1), dtype=np.float64)
    for radial_index, (lo, hi) in enumerate(zip(radial_edges[:-1], radial_edges[1:], strict=True)):
        mask = (radius >= lo) & (radius < hi)
        observed_radial[:, radial_index] = holdout_counts[:, mask].sum(axis=1)
        replicate_radial[:, :, radial_index] = replicated[:, :, mask].sum(axis=2)

    # Posterior-predictive spectra are defined on Pearson residual fields.  This
    # avoids interpreting the survey window itself as clustering.  The P2
    # diagnostic uses one predeclared global z axis and is not the spherical-LOS
    # likelihood approximation used by inference.
    posterior_mean_lambda = lambdas.mean(axis=0)
    observed_residual = (
        holdout_counts - posterior_mean_lambda
    ) / np.sqrt(posterior_mean_lambda + 1.0)
    replicate_residual = (replicated - lambdas) / np.sqrt(lambdas + 1.0)
    spectral_fields = np.concatenate((observed_residual[None, ...], replicate_residual), axis=0)
    k_edges = np.asarray(program["diagnostics"]["k_edges_h_Mpc"], dtype=np.float64)
    separation_edges = np.asarray(program["diagnostics"]["two_point_separation_edges_cMpc_h"], dtype=np.float64)
    residual_p0, residual_p2, residual_xi = residual_spectral_diagnostics(
        spectral_fields, fixed.BOX_SIZE, k_edges, separation_edges
    )
    radial_basis, angular_basis = selection_bases(n, fixed.BOX_SIZE)
    angular_templates = np.stack((radial_basis, angular_basis))
    observed_angular_projection = np.einsum(
        "pxyz,txyz->pt", observed_residual, angular_templates
    ) / np.sqrt(n**3)
    replicate_angular_projection = np.einsum(
        "dpxyz,txyz->dpt", replicate_residual, angular_templates
    ) / np.sqrt(n**3)

    # Velocity holdout LPD is evaluated from the same posterior subset.
    field_size = int(model_meta["field_size"])
    A = model_meta["A"]
    B = model_meta["B"]
    q_std = jnp.asarray(model_meta["q_std"])
    variance = np.asarray(model_meta["variance"])
    hold_rows = np.asarray(design["holdout"], dtype=bool)
    velocity_predictions = []
    for vector in selected:
        white = jnp.asarray(vector[:field_size].reshape((n, n, n)))
        q_unit = jnp.asarray(vector[-4:])
        velocity_predictions.append(np.asarray(A(white) + B @ (q_std * q_unit)))
    velocity_predictions = np.asarray(velocity_predictions)
    velocity_lpd_draws = -0.5 * (
        (np.asarray(mock["velocity_mock"])[None, :] - velocity_predictions) ** 2 / variance[None, :]
        + np.log(2.0 * np.pi * variance)[None, :]
    )
    velocity_holdout_lpd = logmeanexp(velocity_lpd_draws[:, hold_rows], axis=0)

    summary = {
        "predictive_draw_count": predictive_count,
        "heldout_count_model_LPD": float(model_cell_lpd.sum()),
        "heldout_count_selection_only_baseline_LPD": float(baseline_cell_lpd.sum()),
        "heldout_count_delta_LPD": float(delta_cell_lpd.sum()),
        "heldout_count_delta_LPD_by_population": delta_cell_lpd.sum(axis=(1, 2, 3)).tolist(),
        "heldout_velocity_LPD": float(velocity_holdout_lpd.sum()),
        "observed_zero_fraction_by_population": observed_zero.tolist(),
        "replicate_zero_fraction_mean_by_population": replicate_zero.mean(axis=0).tolist(),
    }
    arrays = {
        "heldout_count_delta_cell_lpd": delta_cell_lpd,
        "heldout_count_model_cell_lpd": model_cell_lpd,
        "heldout_count_baseline_cell_lpd": baseline_cell_lpd,
        "posterior_predictive_observed_histogram": observed_hist,
        "posterior_predictive_replicate_histogram": replicate_hist,
        "posterior_predictive_observed_radial_counts": observed_radial,
        "posterior_predictive_replicate_radial_counts": replicate_radial,
        "posterior_predictive_replicate_zero_fraction": replicate_zero,
        "posterior_predictive_observed_angular_template_projection": observed_angular_projection,
        "posterior_predictive_replicate_angular_template_projection": replicate_angular_projection,
        "posterior_predictive_residual_P0": residual_p0,
        "posterior_predictive_residual_P2_fixed_z_axis": residual_p2,
        "posterior_predictive_residual_2PCF": residual_xi,
        "posterior_predictive_k_edges_h_Mpc": k_edges,
        "posterior_predictive_2PCF_separation_edges_cMpc_h": separation_edges,
        "heldout_velocity_cell_lpd": velocity_holdout_lpd,
    }
    return summary, arrays


def artifact_manifest(directory: Path, schema: str) -> dict[str, object]:
    rows = []
    for path in sorted(directory.iterdir()):
        if path.name in {"manifest.json", "COMPLETE"}:
            continue
        rows.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": schema, "files": rows}


def run_preflight(
    program_path: str | Path,
    output_path: str | Path,
    implementation_commit: str,
) -> None:
    """Compile one seed-0 truth and joint value/gradient, without sampling."""

    program, program_sha = load_program(program_path)
    output = Path(output_path)
    staging = output.parent / f".{output.name}.staging"
    if output.exists() or staging.exists():
        raise PhaseCError("Phase-C preflight output or staging already exists")
    if len(implementation_commit) != 40:
        raise PhaseCError("implementation commit must be a full Git hash")
    output.parent.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise PhaseCError("Phase-C preflight requires the allocated Slurm GPU")
    assignment = program["mock_assignments"][0]
    seed = int(assignment["seed"])
    response6, response4 = _load_selection(program)
    nbar, bias = _published_prior_arrays(program)
    args = fixed.frozen_args(program["input_bindings"]["CF4_catalog"]["path"])
    design = linear.prepare_catalog(args)
    fine_white, _coarse_white, nesting = nested_white_fields(
        seed,
        int(program["grid"]["inference_N"]),
        int(program["grid"]["truth_N"]),
        int(program["rng_tags"]["high_k_white"]),
    )
    truth = build_pmwd_truth(fine_white, program)
    intensity, stress_meta = truth_intensity(
        str(assignment["arm"]), truth, response6, response4, nbar, bias, program, seed
    )
    mock = generate_mock_data(str(assignment["arm"]), seed, intensity, truth, design, program)
    negative_log_posterior, _count_lambda, initial, metadata = build_inference_model(
        program, response6, mock, design
    )
    value, gradient = jax.jit(jax.value_and_grad(negative_log_posterior))(jnp.asarray(initial))
    value_float = float(value)
    gradient_np = np.asarray(gradient, dtype=np.float64)
    if not math.isfinite(value_float) or not np.all(np.isfinite(gradient_np)):
        raise PhaseCError("Phase-C preflight value/gradient is nonfinite")
    result = {
        "schema": "ouruniv-cf4-datum-bearing-z0-phasec-preflight-result-v1",
        "status": "PASS_PHASE_C_EXECUTION_PREFLIGHT_NO_SAMPLER_NO_SCIENCE_RESULT",
        "program": {"path": str(Path(program_path).resolve()), "sha256": program_sha},
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(__file__),
            "commit": implementation_commit,
        },
        "assignment_reused": assignment,
        "nested_white": nesting,
        "stress": stress_meta,
        "initial_negative_log_posterior": value_float,
        "initial_gradient_norm": float(np.linalg.norm(gradient_np)),
        "initial_gradient_max_abs": float(np.max(np.abs(gradient_np))),
        "cross_device_growth": {
            "GPU_likelihood_growth_rate": float(metadata["growth_rate"]),
            "CPU_transfer_growth_rate": float(metadata["CPU_transfer_growth_rate"]),
            "relative_difference": float(metadata["cross_device_growth_relative_difference"]),
            "relative_tolerance": float(metadata["cross_device_growth_relative_tolerance"]),
        },
        "environment": {
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
        },
        "semantics": {
            "optimizer_run": False,
            "sampler_run": False,
            "posterior_created": False,
            "science_metric_created": False,
            "actual_observational_datum_used": False,
            "validation_seed_used": False,
            "preflight_result_may_tune_science_contract": False,
        },
    }
    staging.mkdir(mode=0o700)
    (staging / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = artifact_manifest(staging, "ouruniv-cf4-phasec-preflight-manifest-v1")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete = {
        "schema": "ouruniv-cf4-phasec-preflight-complete-v1",
        "result_sha256": sha256_file(staging / "result.json"),
        "manifest_sha256": sha256_file(staging / "manifest.json"),
        "pass": True,
    }
    (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
    os.replace(staging, output)


def validate_preflight(directory: str | Path) -> None:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != PREFLIGHT_FILES:
        raise PhaseCError("Phase-C preflight file set mismatch")
    result = json.loads((root / "result.json").read_text())
    if result.get("status") != "PASS_PHASE_C_EXECUTION_PREFLIGHT_NO_SAMPLER_NO_SCIENCE_RESULT":
        raise PhaseCError("Phase-C preflight did not pass")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("result_sha256") != sha256_file(root / "result.json"):
        raise PhaseCError("Phase-C preflight result hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise PhaseCError("Phase-C preflight manifest hash mismatch")


def run_task(
    program_path: str | Path,
    output_root: str | Path,
    task_index: int,
    implementation_commit: str,
) -> None:
    program, program_sha = load_program(program_path)
    if task_index < 0 or task_index >= 8:
        raise PhaseCError("task index outside the frozen eight-mock assignment")
    assignment = program["mock_assignments"][task_index]
    seed = int(assignment["seed"])
    arm = str(assignment["arm"])
    root = Path(output_root)
    output = root / task_name(task_index, seed, arm)
    staging = root / f".{output.name}.staging"
    if output.exists() or staging.exists():
        raise PhaseCError("Phase-C task output or staging already exists")
    if len(implementation_commit) != 40:
        raise PhaseCError("implementation commit must be a full Git hash")
    root.mkdir(parents=True, exist_ok=True)

    import jax

    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise PhaseCError("Phase-C task requires the allocated Slurm GPU")
    response6, response4 = _load_selection(program)
    nbar, bias = _published_prior_arrays(program)
    args = fixed.frozen_args(program["input_bindings"]["CF4_catalog"]["path"])
    design = linear.prepare_catalog(args)
    fine_white, coarse_white, nesting = nested_white_fields(
        seed,
        int(program["grid"]["inference_N"]),
        int(program["grid"]["truth_N"]),
        int(program["rng_tags"]["high_k_white"]),
    )
    truth = build_pmwd_truth(fine_white, program)
    intensity, stress_meta = truth_intensity(
        arm, truth, response6, response4, nbar, bias, program, seed
    )
    mock = generate_mock_data(arm, seed, intensity, truth, design, program)
    negative_log_posterior, count_lambda, initial, model_meta = build_inference_model(
        program, response6, mock, design
    )
    draws, sampler_arrays = run_four_chains(negative_log_posterior, initial, program, seed)

    field_size = int(model_meta["field_size"])
    white_draws = draws[:, :, :field_size]
    density_draws, velocity_draws = density_velocity_samples(
        white_draws,
        np.asarray(model_meta["transfer"]),
        float(model_meta["growth_rate"]),
        fixed.BOX_SIZE,
    )
    density_flat = density_draws.reshape((-1, fixed.N, fixed.N, fixed.N))
    velocity_flat = velocity_draws.reshape((-1, 3, fixed.N, fixed.N, fixed.N))
    density_mean = density_flat.mean(axis=0)
    density_quantiles = np.quantile(density_flat, [0.025, 0.16, 0.84, 0.975], axis=0).astype(np.float32)
    velocity_mean = velocity_flat.mean(axis=0)
    velocity_quantiles = np.quantile(velocity_flat, [0.025, 0.16, 0.84, 0.975], axis=0).astype(np.float32)
    truth_density = np.asarray(truth["coarse_delta"], dtype=np.float64)
    truth_velocity = np.moveaxis(np.asarray(truth["coarse_velocity"], dtype=np.float64), -1, 0)

    roi_names, roi_weights, roi_effective_cells = build_roi_weights(program)
    global_weight = np.ones((fixed.N,) * 3, dtype=np.float64)
    density_coverage = {
        "global_68": weighted_coverage(truth_density, density_quantiles[1], density_quantiles[2], global_weight),
        "global_95": weighted_coverage(truth_density, density_quantiles[0], density_quantiles[3], global_weight),
        "ROI_68": {},
        "ROI_95": {},
    }
    velocity_coverage = {"global_68": [], "global_95": [], "ROI_68": {}, "ROI_95": {}}
    for component in range(3):
        velocity_coverage["global_68"].append(
            weighted_coverage(truth_velocity[component], velocity_quantiles[1, component], velocity_quantiles[2, component], global_weight)
        )
        velocity_coverage["global_95"].append(
            weighted_coverage(truth_velocity[component], velocity_quantiles[0, component], velocity_quantiles[3, component], global_weight)
        )
    for roi_index, roi_name in enumerate(roi_names):
        weight = roi_weights[roi_index]
        density_coverage["ROI_68"][roi_name] = weighted_coverage(
            truth_density, density_quantiles[1], density_quantiles[2], weight
        )
        density_coverage["ROI_95"][roi_name] = weighted_coverage(
            truth_density, density_quantiles[0], density_quantiles[3], weight
        )
        velocity_coverage["ROI_68"][roi_name] = [
            weighted_coverage(truth_velocity[c], velocity_quantiles[1, c], velocity_quantiles[2, c], weight)
            for c in range(3)
        ]
        velocity_coverage["ROI_95"][roi_name] = [
            weighted_coverage(truth_velocity[c], velocity_quantiles[0, c], velocity_quantiles[3, c], weight)
            for c in range(3)
        ]

    nuisance = draws[:, :, field_size:].astype(np.float64)
    nuisance_names = (
        [f"alpha_{p}" for p in range(6)]
        + [f"logbias_{p}" for p in range(6)]
        + [f"logFoG_{p}" for p in range(6)]
        + ["selection_radial", "selection_angular"]
        + [f"velocity_q_{q}" for q in range(4)]
    )
    projection_parts = [nuisance]
    projection_names = list(nuisance_names)
    for roi_index, roi_name in enumerate(roi_names):
        weight = roi_weights[roi_index]
        weighted = np.sum(density_draws * weight[None, None, ...], axis=(2, 3, 4)) / weight.sum()
        projection_parts.append(weighted[:, :, None])
        projection_names.append(f"density_ROI_{roi_name}")
    projection_parts.append(np.asarray(sampler_arrays["logdensity"])[..., None])
    projection_names.append("logdensity")
    projections = np.concatenate(projection_parts, axis=2)
    convergence = chain_diagnostics(projections, projection_names)

    nuisance_flat = nuisance.reshape((-1, nuisance.shape[-1]))
    nuisance_corr = np.corrcoef(nuisance_flat, rowvar=False)
    nuisance_cov = np.cov(nuisance_flat, rowvar=False)
    nuisance_rank = int(np.linalg.matrix_rank(nuisance_cov))
    nuisance_eigenvalues = np.linalg.eigvalsh(nuisance_cov)

    k_edges = np.asarray(program["diagnostics"]["k_edges_h_Mpc"], dtype=np.float64)
    shell = k_shell_metrics(
        truth_density,
        density_mean,
        density_flat,
        np.asarray(model_meta["transfer"]),
        fixed.BOX_SIZE,
        k_edges,
    )
    frequency = 2.0 * np.pi * np.fft.fftfreq(fixed.N, d=fixed.BOX_SIZE / fixed.N)
    kx, ky, kz = np.meshgrid(frequency, frequency, frequency, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    safe_k2 = np.where(k2 > 0.0, k2, 1.0)
    velocity_shells = []
    for component, kval in enumerate((kx, ky, kz)):
        velocity_transfer = (
            np.asarray(model_meta["transfer"])
            * 100.0
            * float(model_meta["growth_rate"])
            * np.abs(kval)
            / safe_k2
        )
        velocity_transfer[0, 0, 0] = 0.0
        velocity_shells.append(
            k_shell_metrics(
                truth_velocity[component],
                velocity_mean[component],
                velocity_flat[:, component],
                velocity_transfer,
                fixed.BOX_SIZE,
                k_edges,
            )
        )
    predictive_summary, predictive_arrays = predictive_diagnostics(
        draws, count_lambda, mock, response6, design, model_meta, program, seed
    )

    sampler_checks = {
        "all_draws_finite": bool(np.all(np.isfinite(draws))),
        "all_energies_finite": bool(np.all(np.isfinite(sampler_arrays["energy"]))),
        "divergence_fraction_below_pilot_limit": float(np.mean(sampler_arrays["is_divergent"]))
        <= float(program["pilot_flags"]["divergence_fraction_max"]),
        "Rhat_below_pilot_limit": float(convergence["max_Rhat"]) <= float(program["pilot_flags"]["Rhat_max"]),
        "bulk_ESS_above_pilot_limit": float(convergence["min_bulk_ESS"]) >= float(program["pilot_flags"]["bulk_ESS_min"]),
        "tail_ESS_above_pilot_limit": float(convergence["min_tail_ESS"]) >= float(program["pilot_flags"]["tail_ESS_min"]),
    }
    mechanics_pass = bool(sampler_checks["all_draws_finite"] and sampler_checks["all_energies_finite"])
    diagnostic_pass = bool(all(sampler_checks.values()))
    result = {
        "schema": TASK_SCHEMA,
        "status": "PASS_PHASE_C_TASK_DIAGNOSTIC_FLAGS" if diagnostic_pass else (
            "COMPLETE_PHASE_C_TASK_WITH_DIAGNOSTIC_FLAGS" if mechanics_pass else "FAIL_PHASE_C_TASK_MECHANICS"
        ),
        "assignment": assignment,
        "program": {"path": str(Path(program_path).resolve()), "sha256": program_sha},
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(__file__),
            "commit": implementation_commit,
        },
        "environment": {
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "float64": bool(jax.config.jax_enable_x64),
        },
        "truth": {
            "nested_white": nesting,
            "PMWD_density_min": truth["density_min"],
            "PMWD_density_max": truth["density_max"],
            "empty_velocity_cell_count": truth["empty_velocity_cell_count"],
            "stress": stress_meta,
        },
        "mock": {
            "counts_train_total": int(np.sum(mock["counts_train"])),
            "counts_holdout_total": int(np.sum(mock["counts_holdout"])),
            "counts_train_by_population": np.sum(mock["counts_train"], axis=(1, 2, 3)).astype(int).tolist(),
            "counts_holdout_by_population": np.sum(mock["counts_holdout"], axis=(1, 2, 3)).astype(int).tolist(),
            "CF4_train_row_count": int(np.count_nonzero(~np.asarray(design["holdout"]))),
            "CF4_holdout_row_count": int(np.count_nonzero(np.asarray(design["holdout"]))),
            "actual_2Mpp_counts_read": False,
            "actual_CF4_velocity_datum_used": False,
            "validation_seed_read": False,
        },
        "sampler": {
            "chain_count": int(draws.shape[0]),
            "draws_per_chain": int(draws.shape[1]),
            "MAP_value": float(sampler_arrays["MAP_value"]),
            "MAP_gradient_norm": float(sampler_arrays["MAP_gradient_norm"]),
            "MAP_iterations": int(sampler_arrays["MAP_iterations"]),
            "MAP_status": int(sampler_arrays["MAP_status"]),
            "step_size": np.asarray(sampler_arrays["step_size"]).tolist(),
            "warmup_mean_acceptance": np.asarray(sampler_arrays["warmup_mean_acceptance"]).tolist(),
            "warmup_divergence_count": np.asarray(sampler_arrays["warmup_divergence_count"]).astype(int).tolist(),
            "sampling_mean_acceptance": float(np.mean(sampler_arrays["acceptance_rate"])),
            "sampling_divergence_fraction": float(np.mean(sampler_arrays["is_divergent"])),
            "convergence": convergence,
            "checks": sampler_checks,
        },
        "identifiability": {
            "nuisance_names": nuisance_names,
            "posterior_covariance_rank": nuisance_rank,
            "posterior_covariance_eigenvalue_min": float(nuisance_eigenvalues.min()),
            "posterior_covariance_eigenvalue_max": float(nuisance_eigenvalues.max()),
            "maximum_absolute_offdiagonal_correlation": float(
                np.max(np.abs(nuisance_corr - np.eye(nuisance_corr.shape[0])))
            ),
        },
        "predictive": predictive_summary,
        "coverage": {"density": density_coverage, "velocity_components": velocity_coverage},
        "k_shells": {key: value.tolist() for key, value in shell.items()},
        "velocity_k_shells_by_component": [
            {key: value.tolist() for key, value in component.items()}
            for component in velocity_shells
        ],
        "ROI": {
            "names": roi_names,
            "N32_effective_weighted_cell_count": roi_effective_cells.tolist(),
            "underresolved_at_N32": True,
        },
        "semantics": {
            "mock_only": True,
            "science_calibration_pilot_not_validation": True,
            "diagnostic_thresholds_are_pilot_usability_flags_not_Phase_D_promotion_gates": True,
            "stress_arms_must_be_reported_separately": True,
            "actual_present_day_posterior_created": False,
            "observational_resolution_or_frontier_claim_created": False,
            "target_0p3_cMpc_h_reached": False,
            "automatic_Phase_D_allowed": False,
        },
        "mechanics_complete": mechanics_pass,
        "all_pilot_diagnostic_flags_pass": diagnostic_pass,
    }

    staging.mkdir(mode=0o700)
    try:
        np.savez_compressed(
            staging / "posterior_summary.npz",
            truth_coarse_density=truth_density.astype(np.float32),
            truth_coarse_velocity=truth_velocity.astype(np.float32),
            posterior_density_mean=density_mean.astype(np.float32),
            posterior_density_std=density_flat.std(axis=0).astype(np.float32),
            posterior_density_quantiles=density_quantiles,
            posterior_velocity_mean=velocity_mean.astype(np.float32),
            posterior_velocity_std=velocity_flat.std(axis=0).astype(np.float32),
            posterior_velocity_quantiles=velocity_quantiles,
            posterior_white_thinned=white_draws[:, :: int(program["diagnostics"]["stored_white_thinning"]), :].astype(np.float32),
            roi_names=np.asarray(roi_names),
            roi_weights=roi_weights.astype(np.float32),
        )
        np.savez_compressed(
            staging / "diagnostics.npz",
            mock_counts_train=np.asarray(mock["counts_train"], dtype=np.int64),
            mock_counts_holdout=np.asarray(mock["counts_holdout"], dtype=np.int64),
            truth_count_intensity=np.asarray(mock["truth_count_intensity"], dtype=np.float32),
            nuisance_samples=nuisance.astype(np.float32),
            nuisance_correlation=nuisance_corr,
            projection_samples=projections.astype(np.float32),
            k_edges=k_edges,
            **{f"k_{key}": value for key, value in shell.items()},
            **{
                f"velocity_component_{component_index}_k_{key}": value
                for component_index, component in enumerate(velocity_shells)
                for key, value in component.items()
            },
            **predictive_arrays,
            sampler_acceptance_rate=np.asarray(sampler_arrays["acceptance_rate"]),
            sampler_is_divergent=np.asarray(sampler_arrays["is_divergent"]),
            sampler_energy=np.asarray(sampler_arrays["energy"]),
            sampler_logdensity=np.asarray(sampler_arrays["logdensity"]),
        )
        (staging / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        manifest = artifact_manifest(staging, "ouruniv-cf4-phasec-task-manifest-v1")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        complete = {
            "schema": "ouruniv-cf4-phasec-task-complete-v1",
            "result_sha256": sha256_file(staging / "result.json"),
            "manifest_sha256": sha256_file(staging / "manifest.json"),
            "mechanics_complete": mechanics_pass,
            "all_pilot_diagnostic_flags_pass": diagnostic_pass,
        }
        (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
        if not mechanics_pass:
            raise PhaseCError("Phase-C sampler mechanics failed")
        os.replace(staging, output)
    except Exception:
        if staging.exists() and not (staging / "result.json").exists():
            shutil.rmtree(staging)
        raise


def validate_task(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != TASK_FILES:
        raise PhaseCError("Phase-C task artifact file set mismatch")
    result = json.loads((root / "result.json").read_text())
    if result.get("schema") != TASK_SCHEMA or not result.get("mechanics_complete"):
        raise PhaseCError("Phase-C task did not complete mechanics")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("result_sha256") != sha256_file(root / "result.json"):
        raise PhaseCError("Phase-C task result hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise PhaseCError("Phase-C task manifest hash mismatch")
    return result


def aggregate(
    program_path: str | Path,
    output_root: str | Path,
    aggregate_output: str | Path,
    implementation_commit: str,
) -> None:
    program, program_sha = load_program(program_path)
    root = Path(output_root)
    output = Path(aggregate_output)
    staging = output.parent / f".{output.name}.staging"
    if output.exists() or staging.exists():
        raise PhaseCError("Phase-C aggregate output or staging already exists")
    results = []
    for assignment in program["mock_assignments"]:
        directory = root / task_name(int(assignment["index"]), int(assignment["seed"]), str(assignment["arm"]))
        result = validate_task(directory)
        if result["assignment"] != assignment:
            raise PhaseCError("Phase-C task assignment/result mismatch")
        results.append(result)
    by_arm = {}
    for arm in "ABCD":
        members = [result for result in results if result["assignment"]["arm"] == arm]
        by_arm[arm] = {
            "member_count": len(members),
            "seeds": [row["assignment"]["seed"] for row in members],
            "all_mechanics_complete": bool(all(row["mechanics_complete"] for row in members)),
            "all_pilot_diagnostic_flags_pass": bool(all(row["all_pilot_diagnostic_flags_pass"] for row in members)),
            "max_Rhat": max(row["sampler"]["convergence"]["max_Rhat"] for row in members),
            "min_bulk_ESS": min(row["sampler"]["convergence"]["min_bulk_ESS"] for row in members),
            "min_tail_ESS": min(row["sampler"]["convergence"]["min_tail_ESS"] for row in members),
            "max_divergence_fraction": max(row["sampler"]["sampling_divergence_fraction"] for row in members),
            "heldout_count_delta_LPD": [row["predictive"]["heldout_count_delta_LPD"] for row in members],
            "density_global_68_coverage": [row["coverage"]["density"]["global_68"] for row in members],
            "density_global_95_coverage": [row["coverage"]["density"]["global_95"] for row in members],
        }
    aggregate_result = {
        "schema": AGGREGATE_SCHEMA,
        "status": "COMPLETE_PHASE_C_EIGHT_MOCK_PILOT_STOP_BEFORE_PHASE_D",
        "program": {"path": str(Path(program_path).resolve()), "sha256": program_sha},
        "implementation_commit": implementation_commit,
        "task_count": len(results),
        "balanced_two_per_arm": all(by_arm[arm]["member_count"] == 2 for arm in "ABCD"),
        "by_arm": by_arm,
        "overall": {
            "all_mechanics_complete": bool(all(row["mechanics_complete"] for row in results)),
            "all_pilot_diagnostic_flags_pass": bool(all(row["all_pilot_diagnostic_flags_pass"] for row in results)),
            "actual_observational_field_inference": False,
            "validation_seed_access": False,
            "observational_resolution_claim": False,
        },
        "decision": {
            "Phase_C_is_diagnostic_not_a_promotion_gate": True,
            "Phase_D_automatic_start_allowed": False,
            "actual_observational_posterior_automatic_start_allowed": False,
            "next_step_requires_result_audit_and_user_approval": True,
        },
    }
    staging.mkdir(mode=0o700)
    (staging / "aggregate.json").write_text(json.dumps(aggregate_result, indent=2, sort_keys=True) + "\n")
    manifest = artifact_manifest(staging, "ouruniv-cf4-phasec-aggregate-manifest-v1")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete = {
        "schema": "ouruniv-cf4-phasec-aggregate-complete-v1",
        "aggregate_sha256": sha256_file(staging / "aggregate.json"),
        "manifest_sha256": sha256_file(staging / "manifest.json"),
        "task_count": len(results),
    }
    (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
    os.replace(staging, output)


def validate_aggregate(directory: str | Path) -> None:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != AGGREGATE_FILES:
        raise PhaseCError("Phase-C aggregate file set mismatch")
    result = json.loads((root / "aggregate.json").read_text())
    if result.get("schema") != AGGREGATE_SCHEMA or result.get("task_count") != 8:
        raise PhaseCError("Phase-C aggregate result is invalid")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("aggregate_sha256") != sha256_file(root / "aggregate.json"):
        raise PhaseCError("Phase-C aggregate hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise PhaseCError("Phase-C aggregate manifest hash mismatch")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("--program", required=True)
    preflight_parser.add_argument("--output", required=True)
    preflight_parser.add_argument("--implementation-commit", required=True)
    validate_preflight_parser = sub.add_parser("validate-preflight")
    validate_preflight_parser.add_argument("--directory", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--program", required=True)
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--task-index", required=True, type=int)
    run_parser.add_argument("--implementation-commit", required=True)
    validate_parser = sub.add_parser("validate-task")
    validate_parser.add_argument("--directory", required=True)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--program", required=True)
    aggregate_parser.add_argument("--output-root", required=True)
    aggregate_parser.add_argument("--aggregate-output", required=True)
    aggregate_parser.add_argument("--implementation-commit", required=True)
    validate_aggregate_parser = sub.add_parser("validate-aggregate")
    validate_aggregate_parser.add_argument("--directory", required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        run_preflight(args.program, args.output, args.implementation_commit)
    elif args.command == "validate-preflight":
        validate_preflight(args.directory)
    elif args.command == "run":
        run_task(args.program, args.output_root, args.task_index, args.implementation_commit)
    elif args.command == "validate-task":
        validate_task(args.directory)
    elif args.command == "aggregate":
        aggregate(args.program, args.output_root, args.aggregate_output, args.implementation_commit)
    elif args.command == "validate-aggregate":
        validate_aggregate(args.directory)


if __name__ == "__main__":
    main()
