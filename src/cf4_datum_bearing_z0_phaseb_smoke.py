#!/usr/bin/env python3
"""Matched-mock Phase-B smoke for the joint CF4 + 2M++ z=0 model.

This program is deliberately technical.  It reuses development truth seed
2026083000, never reads the actual 2M++ count arrays, and cannot publish an
observational field or resolution claim.  It checks a positive six-tracer
Poisson factor, observer-centred spherical coherent RSD, the existing CF4
mean-radial-velocity/sigma likelihood, gradients, an RSD mass adjoint, a short
optimizer run, and four mechanics-only HMC transitions.
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
from typing import Mapping

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

import cf4_bgc_fixed_design_smoke as fixed
import cf4_linear_cr as linear


SCHEMA = "ouruniv-cf4-datum-bearing-z0-phaseb-smoke-program-v1"
RESULT_SCHEMA = "ouruniv-cf4-datum-bearing-z0-phaseb-smoke-result-v1"
EXPECTED_FILES = {"diagnostics.npz", "result.json", "manifest.json", "COMPLETE"}


class PhaseBError(ValueError):
    """The fail-closed Phase-B smoke contract was violated."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def load_program(path: str | Path) -> tuple[dict[str, object], str]:
    payload = Path(path).read_bytes()
    try:
        program = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PhaseBError("cannot parse Phase-B program") from exc
    if program.get("schema") != SCHEMA:
        raise PhaseBError("Phase-B program schema mismatch")
    auth = program.get("authorization", {})
    required_true = {
        "Phase_B_matched_mock_smoke",
        "single_Slurm_GPU_submission",
        "GPFS_read",
        "GPFS_write_new_output_only",
    }
    required_false = {
        "actual_observational_field_inference",
        "new_truth_seed",
        "validation_seed_access",
        "Phase_C_or_later",
        "automatic_follow_on",
        "IC_PM_HOP_RAMSES",
    }
    if any(auth.get(key) is not True for key in required_true):
        raise PhaseBError("required Phase-B authorization is absent")
    if any(auth.get(key) is not False for key in required_false):
        raise PhaseBError("a forbidden Phase-B authorization is enabled")
    if program.get("mock", {}).get("truth_seed_reused") != fixed.TRUTH_SEED:
        raise PhaseBError("the single reused development truth seed changed")
    if program.get("grid", {}) != {
        "N": fixed.N,
        "box_size_cMpc_h": fixed.BOX_SIZE,
        "cell_size_cMpc_h": fixed.BOX_SIZE / fixed.N,
    }:
        raise PhaseBError("the frozen N32 grid changed")
    for section in ("input_bindings", "source_bindings"):
        records = program.get(section)
        if not isinstance(records, Mapping) or not records:
            raise PhaseBError(f"{section} is absent")
        for record in records.values():
            source = Path(str(record["path"]))
            if sha256_file(source) != record["sha256"]:
                raise PhaseBError(f"SHA256 mismatch: {source}")
    gates = program.get("gates", {})
    required_gates = {
        "radial_forward_relative_error_max",
        "rsd_mass_relative_error_max",
        "rsd_adjoint_relative_error_max",
        "directional_gradient_relative_error_max",
        "optimizer_relative_decrease_min",
        "boundary_max_displacement_cMpc_h_strict",
        "prior_dominated_information_fraction_max",
    }
    if set(gates) != required_gates:
        raise PhaseBError("Phase-B numerical gate set is not exact")
    return program, hashlib.sha256(payload).hexdigest()


def _load_selection_only(program: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Read only response arrays; actual count arrays are intentionally untouched."""

    datum_path = program["input_bindings"]["Phase_A_datum"]["path"]
    quadrature_path = program["input_bindings"]["quadrature_arrays"]["path"]
    with np.load(datum_path, allow_pickle=False) as datum:
        exposure6 = np.asarray(datum["raw_selection_exposure"], dtype=np.float64)
    with np.load(quadrature_path, allow_pickle=False) as quad:
        exposure4 = np.asarray(quad["raw_selection_order4"], dtype=np.float64)
        order6_copy = np.asarray(quad["raw_selection_order6"], dtype=np.float64)
    expected = (6, fixed.N, fixed.N, fixed.N)
    if exposure6.shape != expected or exposure4.shape != expected:
        raise PhaseBError("selection response shape mismatch")
    if not np.array_equal(exposure6, order6_copy):
        raise PhaseBError("published order-6 response differs from diagnostic")
    tolerance = 2.0e-14
    if (
        not np.all(np.isfinite(exposure6))
        or not np.all(np.isfinite(exposure4))
        or np.min(exposure6) < -tolerance
        or np.max(exposure6) > 1.0 + tolerance
    ):
        raise PhaseBError("selection response is invalid")
    return np.clip(exposure6, 0.0, 1.0), np.clip(exposure4, 0.0, 1.0)


def _load_reused_truth(program: Mapping[str, object]) -> dict[str, np.ndarray]:
    path = program["input_bindings"]["fixed_design_mock_fields"]["path"]
    names = {
        "truth_white",
        "truth_delta",
        "truth_velocity",
        "truth_forward_radial_signal",
        "truth_nuisance_q",
        "mock_datum",
    }
    with np.load(path, allow_pickle=False) as fields:
        if not names <= set(fields.files):
            raise PhaseBError("reused fixed-design mock is incomplete")
        result = {name: np.asarray(fields[name], dtype=np.float64) for name in names}
    if result["truth_white"].shape != (fixed.N,) * 3:
        raise PhaseBError("reused truth field shape mismatch")
    if result["mock_datum"].ndim != 1:
        raise PhaseBError("reused CF4 mock datum shape mismatch")
    return result


def _published_prior_arrays(program: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray, float, float]:
    prior = program["external_population_prior"]
    original_cell = float(prior["published_cell_size_cMpc_h"])
    volume_ratio = (fixed.BOX_SIZE / fixed.N / original_cell) ** 3
    nbar = np.asarray(prior["published_mean_count_per_original_voxel"], dtype=np.float64)
    bias = np.asarray(prior["published_bias"], dtype=np.float64)
    if nbar.shape != (6,) or bias.shape != (6,) or np.any(nbar <= 0) or np.any(bias <= 0):
        raise PhaseBError("external population prior arrays are invalid")
    alpha_mean = np.log(nbar * volume_ratio)
    return (
        alpha_mean,
        np.log(bias),
        float(prior["alpha_log_sigma"]),
        float(prior["bias_log_sigma"]),
    )


def _seed_from_tag(truth_seed: int, tag: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([truth_seed, tag]))


def _artifact_manifest(directory: Path) -> dict[str, object]:
    rows = []
    for path in sorted(directory.iterdir()):
        if path.name in {"manifest.json", "COMPLETE"}:
            continue
        rows.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": "ouruniv-cf4-phaseb-artifact-manifest-v1", "files": rows}


def run(program_path: str | Path, output_path: str | Path, implementation_commit: str) -> None:
    program, program_sha = load_program(program_path)
    output = Path(output_path)
    staging = output.parent / f".{output.name}.staging"
    if output.exists() or staging.exists():
        raise PhaseBError("Phase-B output or staging already exists")
    if not implementation_commit or len(implementation_commit) != 40:
        raise PhaseBError("implementation commit must be a full Git hash")

    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise PhaseBError("Phase-B smoke requires the allocated Slurm GPU")

    exposure6_np, exposure4_np = _load_selection_only(program)
    truth = _load_reused_truth(program)
    alpha_mean_np, logbias_mean_np, alpha_sigma, logbias_sigma = _published_prior_arrays(program)

    args = fixed.frozen_args(program["input_bindings"]["CF4_catalog"]["path"])
    design = linear.prepare_catalog(args)
    if truth["mock_datum"].shape != np.asarray(design["vobs"]).shape:
        raise PhaseBError("CF4 design and reused mock datum disagree")
    A, _AT, growth_rate, _dtype = linear.build_forward(design["pos"], design["rhat"], args)
    transfer_np, growth_rate_check = fixed.build_density_transfer(args)
    if not math.isclose(growth_rate, growth_rate_check, rel_tol=2e-13, abs_tol=0.0):
        raise PhaseBError("density and CF4 growth-rate kernels disagree")

    N = fixed.N
    L = fixed.BOX_SIZE
    spacing = L / N
    field_size = N**3
    transfer = jnp.asarray(transfer_np)
    exposure6 = jnp.asarray(exposure6_np)
    exposure4 = jnp.asarray(exposure4_np)
    alpha_mean = jnp.asarray(alpha_mean_np)
    logbias_mean = jnp.asarray(logbias_mean_np)
    q_std_np = np.asarray(design["q_std"], dtype=np.float64)
    q_std = jnp.asarray(q_std_np)
    B = jnp.asarray(design["B"], dtype=jnp.float64)
    velocity_variance = jnp.asarray(design["variance"], dtype=jnp.float64)
    velocity_mock = jnp.asarray(truth["mock_datum"], dtype=jnp.float64)
    train = jnp.asarray(~np.asarray(design["holdout"], dtype=bool))

    axis = (jnp.arange(N, dtype=jnp.float64) + 0.5) * spacing
    gx, gy, gz = jnp.meshgrid(axis, axis, axis, indexing="ij")
    centres = jnp.stack((gx, gy, gz), axis=-1)
    relative = centres - L / 2.0
    radius = jnp.linalg.norm(relative, axis=-1)
    rhat_cells = relative / radius[..., None]

    def white_to_delta(white):
        delta_k = jnp.fft.fftn(white.reshape((N, N, N)), norm="ortho") * transfer
        return jnp.fft.ifftn(delta_k, norm="ortho").real

    freq = 2.0 * np.pi * np.fft.fftfreq(N, d=spacing)
    rfreq = 2.0 * np.pi * np.fft.rfftfreq(N, d=spacing)
    kx_np, ky_np, kz_np = np.meshgrid(freq, freq, rfreq, indexing="ij")
    kvec = tuple(jnp.asarray(value) for value in (kx_np, ky_np, kz_np))
    k2 = kvec[0] ** 2 + kvec[1] ** 2 + kvec[2] ** 2
    k2_safe = jnp.where(k2 > 0.0, k2, 1.0)

    def delta_to_velocity(delta):
        delta_k = jnp.fft.rfftn(delta)
        pieces = []
        for component in kvec:
            velocity_k = 1j * 100.0 * growth_rate * component / k2_safe * delta_k
            velocity_k = jnp.where(k2 > 0.0, velocity_k, 0.0)
            pieces.append(jnp.fft.irfftn(velocity_k, s=(N, N, N)))
        return jnp.stack(pieces)

    def rsd_positions(velocity):
        velocity_last = jnp.moveaxis(velocity, 0, -1)
        radial_velocity = jnp.sum(velocity_last * rhat_cells, axis=-1)
        displacement = radial_velocity / 100.0
        return (centres + displacement[..., None] * rhat_cells) % L, displacement

    def cic_deposit(masses, positions):
        """Periodic, conservative, piecewise-differentiable CIC push."""
        flat_pos = positions.reshape((-1, 3))
        flat_mass = masses.reshape((masses.shape[0], -1))
        cell = flat_pos / spacing - 0.5
        lower = jnp.floor(cell).astype(jnp.int32)
        frac = cell - lower
        out = jnp.zeros((masses.shape[0], N, N, N), dtype=masses.dtype)
        for dx in (0, 1):
            wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
            for dy in (0, 1):
                wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
                for dz in (0, 1):
                    wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                    ii = (lower[:, 0] + dx) % N
                    jj = (lower[:, 1] + dy) % N
                    kk = (lower[:, 2] + dz) % N
                    out = out.at[:, ii, jj, kk].add(flat_mass * (wx * wy * wz)[None, :])
        return out

    def count_lambda(white, alpha, logbias, response):
        delta = white_to_delta(white)
        eta = delta - jnp.mean(delta)
        velocity = delta_to_velocity(delta)
        positions, displacement = rsd_positions(velocity)
        bias = jnp.exp(logbias)
        real_mass = jnp.exp(alpha[:, None, None, None] + bias[:, None, None, None] * eta)
        redshift_mass = cic_deposit(real_mass, positions)
        return response * redshift_mass, displacement, eta, positions

    truth_white = jnp.asarray(truth["truth_white"])
    truth_alpha = alpha_mean
    truth_logbias = logbias_mean
    truth_lambda6, truth_displacement, truth_eta, truth_positions = count_lambda(
        truth_white, truth_alpha, truth_logbias, exposure6
    )
    truth_lambda4, _, _, _ = count_lambda(truth_white, truth_alpha, truth_logbias, exposure4)
    truth_lambda6_np = np.asarray(truth_lambda6)
    truth_lambda4_np = np.asarray(truth_lambda4)
    if not np.all(np.isfinite(truth_lambda6_np)) or np.any(truth_lambda6_np < 0.0):
        raise PhaseBError("truth intensity is invalid")

    count_seed = program["mock"]["derived_streams"]["twoMpp_Poisson"]
    count_rng = _seed_from_tag(fixed.TRUTH_SEED, int(count_seed["tag_uint32"]))
    mock_counts_np = count_rng.poisson(truth_lambda6_np).astype(np.int64)
    if np.any((mock_counts_np > 0) & (truth_lambda6_np <= 0.0)):
        raise PhaseBError("positive mock count has nonpositive truth intensity")
    mock_counts = jnp.asarray(mock_counts_np)

    def unpack(vector):
        white = vector[:field_size].reshape((N, N, N))
        alpha = vector[field_size : field_size + 6]
        logbias = vector[field_size + 6 : field_size + 12]
        q_unit = vector[field_size + 12 : field_size + 16]
        return white, alpha, logbias, q_unit

    def negative_log_posterior(vector):
        white, alpha, logbias, q_unit = unpack(vector)
        intensity, _, _, _ = count_lambda(white, alpha, logbias, exposure6)
        positive_support = exposure6 > 0.0
        safe_intensity = jnp.where(positive_support, intensity, 1.0)
        count_nll = jnp.sum(jnp.where(positive_support, intensity - mock_counts * jnp.log(safe_intensity), 0.0))
        velocity_model = A(white) + B @ (q_std * q_unit)
        velocity_residual = velocity_mock - velocity_model
        velocity_nll = 0.5 * jnp.sum(jnp.where(train, velocity_residual**2 / velocity_variance, 0.0))
        prior = 0.5 * jnp.sum(white**2) + 0.5 * jnp.sum(q_unit**2)
        prior += 0.5 * jnp.sum(((alpha - alpha_mean) / alpha_sigma) ** 2)
        prior += 0.5 * jnp.sum(((logbias - logbias_mean) / logbias_sigma) ** 2)
        return prior + count_nll + velocity_nll

    gates = program["gates"]
    forward_truth = np.asarray(A(truth_white))
    radial_relative = float(
        np.linalg.norm(forward_truth - truth["truth_forward_radial_signal"])
        / max(np.linalg.norm(truth["truth_forward_radial_signal"]), np.finfo(float).tiny)
    )

    displacement_np = np.asarray(truth_displacement)
    max_displacement = float(np.max(np.abs(displacement_np)))
    boundary_margin = float(gates["boundary_max_displacement_cMpc_h_strict"])

    operator_rng = _seed_from_tag(fixed.TRUTH_SEED, int(program["mock"]["derived_streams"]["operator_probes"]["tag_uint32"]))
    test_mass_np = np.exp(0.1 * operator_rng.standard_normal((6, N, N, N)))
    test_cotangent_np = operator_rng.standard_normal((6, N, N, N))
    test_mass = jnp.asarray(test_mass_np)
    fixed_positions = jax.lax.stop_gradient(truth_positions)
    pushed, pullback = jax.vjp(lambda mass: cic_deposit(mass, fixed_positions), test_mass)
    pulled = pullback(jnp.asarray(test_cotangent_np))[0]
    left = float(jnp.vdot(pushed, jnp.asarray(test_cotangent_np)))
    right = float(jnp.vdot(test_mass, pulled))
    adjoint_relative = abs(left - right) / max(1.0, abs(left), abs(right))
    mass_relative = float(abs(jnp.sum(pushed) - jnp.sum(test_mass)) / jnp.sum(test_mass))

    initial_np = np.concatenate(
        (
            np.zeros(field_size, dtype=np.float64),
            alpha_mean_np,
            logbias_mean_np,
            np.zeros(4, dtype=np.float64),
        )
    )
    initial = jnp.asarray(initial_np)
    value_and_grad = jax.jit(jax.value_and_grad(negative_log_posterior))
    initial_value, initial_grad = value_and_grad(initial)
    initial_value_float = float(initial_value)
    if not math.isfinite(initial_value_float) or not np.all(np.isfinite(np.asarray(initial_grad))):
        raise PhaseBError("initial log posterior or gradient is nonfinite")

    gradient_rng = _seed_from_tag(fixed.TRUTH_SEED, int(program["mock"]["derived_streams"]["gradient_probes"]["tag_uint32"]))
    gradient_errors = []
    gradient_steps = [float(value) for value in program["mechanics"]["finite_difference_steps"]]
    automatic_gradient_np = np.asarray(initial_grad)
    for _ in range(int(program["mechanics"]["directional_probe_count"])):
        direction = gradient_rng.standard_normal(initial_np.size)
        direction /= np.linalg.norm(direction)
        automatic = float(np.dot(automatic_gradient_np, direction))
        per_step = []
        for step in gradient_steps:
            plus = float(negative_log_posterior(jnp.asarray(initial_np + step * direction)))
            minus = float(negative_log_posterior(jnp.asarray(initial_np - step * direction)))
            finite = (plus - minus) / (2.0 * step)
            per_step.append(abs(finite - automatic) / max(1.0, abs(finite), abs(automatic)))
        gradient_errors.append(min(per_step))

    def scipy_value_gradient(vector):
        value, gradient = value_and_grad(jnp.asarray(vector))
        return float(value), np.asarray(gradient, dtype=np.float64)

    optimization = minimize(
        scipy_value_gradient,
        initial_np,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": int(program["mechanics"]["optimizer_max_iterations"]),
            "maxls": int(program["mechanics"]["optimizer_max_line_search"]),
            "ftol": 0.0,
            "gtol": 0.0,
        },
    )
    optimized_np = np.asarray(optimization.x, dtype=np.float64)
    optimized_value, optimized_grad = value_and_grad(jnp.asarray(optimized_np))
    optimized_value_float = float(optimized_value)
    relative_decrease = (initial_value_float - optimized_value_float) / max(1.0, abs(initial_value_float))

    hmc_steps = int(program["mechanics"]["HMC_transition_count"])
    leapfrog_steps = int(program["mechanics"]["HMC_leapfrog_steps"])
    hmc_step_size = float(program["mechanics"]["HMC_step_size"])

    @jax.jit
    def one_hmc(position, momentum):
        start_value, start_gradient = jax.value_and_grad(negative_log_posterior)(position)
        current_momentum = momentum - 0.5 * hmc_step_size * start_gradient
        current_position = position
        current_value = start_value
        current_gradient = start_gradient
        for index in range(leapfrog_steps):
            current_position = current_position + hmc_step_size * current_momentum
            current_value, current_gradient = jax.value_and_grad(negative_log_posterior)(current_position)
            if index != leapfrog_steps - 1:
                current_momentum = current_momentum - hmc_step_size * current_gradient
        current_momentum = current_momentum - 0.5 * hmc_step_size * current_gradient
        start_energy = start_value + 0.5 * jnp.vdot(momentum, momentum)
        end_energy = current_value + 0.5 * jnp.vdot(current_momentum, current_momentum)
        log_accept = jnp.minimum(0.0, start_energy - end_energy)
        return current_position, log_accept, start_energy, end_energy

    hmc_rng = _seed_from_tag(fixed.TRUTH_SEED, int(program["mock"]["derived_streams"]["HMC_mechanics"]["tag_uint32"]))
    hmc_position = jnp.asarray(optimized_np)
    hmc_records = []
    accepted = 0
    for _ in range(hmc_steps):
        momentum_np = hmc_rng.standard_normal(optimized_np.size)
        proposal, log_accept, start_energy, end_energy = one_hmc(hmc_position, jnp.asarray(momentum_np))
        probability = float(jnp.exp(log_accept))
        uniform = float(hmc_rng.uniform())
        take = bool(uniform < probability)
        if take:
            hmc_position = proposal
            accepted += 1
        hmc_records.append(
            {
                "accept_probability": probability,
                "accepted": take,
                "start_energy": float(start_energy),
                "end_energy": float(end_energy),
            }
        )

    def nuisance_objective(nuisance):
        vector = jnp.concatenate((truth_white.reshape(-1), nuisance))
        return negative_log_posterior(vector)

    truth_q_unit = truth["truth_nuisance_q"] / q_std_np
    truth_nuisance = jnp.concatenate((truth_alpha, truth_logbias, jnp.asarray(truth_q_unit)))
    nuisance_hessian = np.asarray(jax.hessian(nuisance_objective)(truth_nuisance))
    nuisance_eigenvalues = np.linalg.eigvalsh(nuisance_hessian)
    nuisance_rank = int(np.linalg.matrix_rank(nuisance_hessian))

    homogeneous = np.exp(alpha_mean_np)[:, None, None, None] * exposure6_np
    information_proxy = np.sum(np.exp(2.0 * logbias_mean_np)[:, None, None, None] * homogeneous, axis=0)
    information_threshold = float(gates["prior_dominated_information_fraction_max"]) * float(np.max(information_proxy))
    prior_dominated = information_proxy <= information_threshold

    positive_support = exposure6_np > 0.0
    intensity_positive_on_support = bool(np.all(truth_lambda6_np[positive_support] > 0.0))
    hmc_finite = bool(
        all(
            math.isfinite(row["accept_probability"])
            and math.isfinite(row["start_energy"])
            and math.isfinite(row["end_energy"])
            and 0.0 <= row["accept_probability"] <= 1.0
            for row in hmc_records
        )
    )
    checks = {
        "radial_forward_match": radial_relative <= float(gates["radial_forward_relative_error_max"]),
        "truth_intensity_finite_nonnegative": bool(np.all(np.isfinite(truth_lambda6_np)) and np.all(truth_lambda6_np >= 0.0)),
        "truth_intensity_positive_on_exposed_support": intensity_positive_on_support,
        "positive_mock_count_has_positive_intensity": bool(not np.any((mock_counts_np > 0) & (truth_lambda6_np <= 0.0))),
        "spherical_RSD_boundary_safe": max_displacement < boundary_margin,
        "RSD_mass_conservation": mass_relative <= float(gates["rsd_mass_relative_error_max"]),
        "RSD_mass_adjoint": adjoint_relative <= float(gates["rsd_adjoint_relative_error_max"]),
        "directional_gradients": max(gradient_errors) <= float(gates["directional_gradient_relative_error_max"]),
        "optimizer_finite": bool(math.isfinite(optimized_value_float) and np.all(np.isfinite(np.asarray(optimized_grad)))),
        "optimizer_decreases_objective": relative_decrease >= float(gates["optimizer_relative_decrease_min"]),
        "HMC_mechanics_finite": hmc_finite,
        "HMC_at_least_one_accept": accepted >= 1,
        "nuisance_Hessian_full_rank": nuisance_rank == 16,
    }
    all_pass = bool(all(checks.values()))

    q46_denominator = max(float(np.sum(np.abs(truth_lambda6_np))), np.finfo(float).tiny)
    selection_q46_relative_l1 = float(np.sum(np.abs(truth_lambda6_np - truth_lambda4_np)) / q46_denominator)
    result = {
        "schema": RESULT_SCHEMA,
        "status": "PASS_PHASE_B_TECHNICAL_SMOKE_STOP_BEFORE_PHASE_C" if all_pass else "FAIL_PHASE_B_TECHNICAL_SMOKE",
        "program": {"path": str(Path(program_path)), "sha256": program_sha},
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
        "mock": {
            "truth_seed_reused": fixed.TRUTH_SEED,
            "new_truth_seed_consumed": False,
            "validation_seed_consumed": False,
            "actual_2Mpp_counts_read": False,
            "actual_CF4_velocity_datum_used": False,
            "mock_2Mpp_count_total": int(mock_counts_np.sum()),
            "mock_2Mpp_population_totals": mock_counts_np.sum(axis=(1, 2, 3)).astype(int).tolist(),
            "CF4_train_row_count": int(np.count_nonzero(np.asarray(train))),
            "CF4_holdout_row_count": int(np.count_nonzero(~np.asarray(train))),
        },
        "model": {
            "eta_spatial_mean_at_truth": float(np.mean(np.asarray(truth_eta))),
            "alpha_prior_mean": alpha_mean_np.tolist(),
            "alpha_prior_log_sigma": alpha_sigma,
            "bias_prior_mean": np.exp(logbias_mean_np).tolist(),
            "bias_prior_log_sigma": logbias_sigma,
            "full_sky_position_dependent_LOS": True,
            "plane_parallel_RSD": False,
            "coherent_RSD_only_FoG_disabled": True,
            "periodic_CIC_mass_push": True,
            "selection_applied_after_RSD": True,
            "CF4_mean_velocity_and_sigma_factor": True,
        },
        "metrics": {
            "radial_forward_relative_error": radial_relative,
            "RSD_max_abs_displacement_cMpc_h": max_displacement,
            "RSD_mass_relative_error": mass_relative,
            "RSD_adjoint_relative_error": adjoint_relative,
            "directional_gradient_best_relative_errors": gradient_errors,
            "initial_negative_log_posterior": initial_value_float,
            "optimized_negative_log_posterior": optimized_value_float,
            "optimizer_relative_decrease": relative_decrease,
            "optimizer_iterations": int(optimization.nit),
            "optimizer_status": int(optimization.status),
            "optimizer_message": str(optimization.message),
            "optimized_gradient_norm": float(np.linalg.norm(np.asarray(optimized_grad))),
            "HMC_records": hmc_records,
            "HMC_accepted": accepted,
            "nuisance_Hessian_rank": nuisance_rank,
            "nuisance_Hessian_eigenvalue_min": float(nuisance_eigenvalues.min()),
            "nuisance_Hessian_eigenvalue_max": float(nuisance_eigenvalues.max()),
            "selection_order4_to_order6_truth_lambda_relative_L1": selection_q46_relative_l1,
            "prior_dominated_information_threshold": information_threshold,
            "prior_dominated_voxel_count": int(np.count_nonzero(prior_dominated)),
        },
        "gates": {**checks, "all_pass": all_pass},
        "semantics": {
            "mechanics_only": True,
            "matched_mock_only": True,
            "present_day_observational_posterior_created": False,
            "observational_resolution_claim_created": False,
            "0p3_cMpc_h_claim_created": False,
            "seed_or_parent_ranking_allowed": False,
            "Phase_C_automatic_start_allowed": False,
            "selection_order4_order6_difference_is_diagnostic_not_a_convergence_claim": True,
            "prior_dominated_label_is_a_predeclared_exposure_information_proxy_not_a_science_frontier": True,
        },
    }

    staging.mkdir(parents=False)
    try:
        np.savez_compressed(
            staging / "diagnostics.npz",
            mock_counts=mock_counts_np,
            truth_lambda_order6=truth_lambda6_np,
            truth_lambda_order4=truth_lambda4_np,
            prior_information_proxy=information_proxy,
            prior_dominated=prior_dominated,
        )
        (staging / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        manifest = _artifact_manifest(staging)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        complete = {
            "schema": "ouruniv-cf4-phaseb-complete-v1",
            "result_sha256": sha256_file(staging / "result.json"),
            "manifest_sha256": sha256_file(staging / "manifest.json"),
            "all_pass": all_pass,
        }
        (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
        if not all_pass:
            raise PhaseBError("one or more frozen Phase-B technical gates failed")
        os.replace(staging, output)
    except Exception:
        if staging.exists() and not (staging / "result.json").exists():
            shutil.rmtree(staging)
        raise


def validate(directory: str | Path) -> None:
    root = Path(directory)
    if not root.is_dir() or {item.name for item in root.iterdir()} != EXPECTED_FILES:
        raise PhaseBError("Phase-B artifact file set mismatch")
    result = json.loads((root / "result.json").read_text())
    if result.get("schema") != RESULT_SCHEMA:
        raise PhaseBError("Phase-B result schema mismatch")
    if result.get("status") != "PASS_PHASE_B_TECHNICAL_SMOKE_STOP_BEFORE_PHASE_C":
        raise PhaseBError("Phase-B result did not pass")
    if result.get("gates", {}).get("all_pass") is not True:
        raise PhaseBError("Phase-B aggregate gate did not pass")
    semantics = result.get("semantics", {})
    forbidden_false = {
        "present_day_observational_posterior_created",
        "observational_resolution_claim_created",
        "0p3_cMpc_h_claim_created",
        "seed_or_parent_ranking_allowed",
        "Phase_C_automatic_start_allowed",
    }
    if any(semantics.get(key) is not False for key in forbidden_false):
        raise PhaseBError("Phase-B result makes a forbidden claim")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("all_pass") is not True:
        raise PhaseBError("Phase-B COMPLETE marker did not pass")
    if complete.get("result_sha256") != sha256_file(root / "result.json"):
        raise PhaseBError("Phase-B result hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise PhaseBError("Phase-B manifest hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--program", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--implementation-commit", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.program, args.output, args.implementation_commit)
    else:
        validate(args.directory)


if __name__ == "__main__":
    main()
