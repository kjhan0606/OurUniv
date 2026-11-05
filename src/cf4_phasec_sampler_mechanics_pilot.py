#!/usr/bin/env python3
"""Mock-only replacement Phase-C sampler mechanics pilot for seeds 0 and 6."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

import cf4_datum_bearing_z0_phasec_pilot as phasec_v5
import cf4_linear_cr as linear


SCHEMA = "ouruniv-cf4-phasec-sampler-mechanics-pilot-v1"
TASK_SCHEMA = "ouruniv-cf4-phasec-sampler-mechanics-task-v1"
AGGREGATE_SCHEMA = "ouruniv-cf4-phasec-sampler-mechanics-aggregate-v1"
TASK_FILES = {"diagnostics.npz", "result.json", "manifest.json", "COMPLETE"}
AGGREGATE_FILES = {"aggregate.json", "manifest.json", "COMPLETE"}


class SamplerMechanicsError(ValueError):
    """The frozen sampler-mechanics contract was violated."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_assignments() -> list[dict[str, object]]:
    return [
        {"task_index": 0, "mock_index": 0, "seed": 2026083000, "arm": "A"},
        {"task_index": 1, "mock_index": 6, "seed": 2026083006, "arm": "D"},
    ]


def load_program(path: str | Path) -> tuple[dict[str, object], str, dict[str, object]]:
    payload = Path(path).read_bytes()
    try:
        controller = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SamplerMechanicsError("cannot parse sampler-mechanics program") from exc
    if controller.get("schema") != SCHEMA:
        raise SamplerMechanicsError("sampler-mechanics schema mismatch")
    authorization = controller.get("authorization", {})
    for key in (
        "replacement_Phase_C_mock_sampler_mechanics",
        "Slurm_GPU_array_indices_0_and_6",
        "Slurm_CPU_aggregate",
        "GPFS_read_bound_mock_inputs",
        "GPFS_write_new_outputs_only",
    ):
        if authorization.get(key) is not True:
            raise SamplerMechanicsError(f"missing mechanics authorization: {key}")
    for key in (
        "actual_observational_field_inference",
        "actual_2Mpp_count_read",
        "actual_CF4_velocity_datum_used",
        "validation_seed_access",
        "Phase_D_or_later",
    ):
        if authorization.get(key) is not False:
            raise SamplerMechanicsError(f"forbidden mechanics scope enabled: {key}")
    if controller.get("assignments") != expected_assignments():
        raise SamplerMechanicsError("mechanics assignment changed")
    lineage = controller.get("lineage", {})
    expected_lineage = {
        "base_program",
        "replacement_program",
        "generator_gate_PASS_record",
        "syn06_GPU_sweep_record",
        "V5_implementation",
        "scan_generator_implementation",
        "fixed_implementation",
        "linear_implementation",
        "sampler_mechanics_implementation",
    }
    if set(lineage) != expected_lineage:
        raise SamplerMechanicsError("sampler-mechanics lineage set changed")
    for name, binding in lineage.items():
        source = Path(str(binding.get("path", "")))
        expected_hash = str(binding.get("sha256", ""))
        if not source.is_file():
            raise SamplerMechanicsError(f"missing mechanics lineage: {name}")
        if expected_hash.startswith("TO_BE_FILLED"):
            raise SamplerMechanicsError("mechanics implementation is not frozen")
        if sha256_file(source) != expected_hash:
            raise SamplerMechanicsError(f"mechanics lineage hash mismatch: {name}")

    mechanics = controller.get("mechanics", {})
    if mechanics.get("latent_dimension") != 32792:
        raise SamplerMechanicsError("standardized latent dimension changed")
    expected_map = {
        "optimizer": "SciPy L-BFGS-B with exact JAX gradient",
        "maximum_iterations": 256,
        "maximum_line_search_steps": 40,
        "objective_relative_tolerance": 1e-10,
        "finite_value_and_gradient_required": True,
    }
    if mechanics.get("MAP") != expected_map:
        raise SamplerMechanicsError("MAP mechanics changed")
    sampler = mechanics.get("sampler", {})
    expected_sampler = {
        "algorithm": "BlackJAX static Euclidean HMC",
        "chain_count": 4,
        "warmup_steps": 512,
        "posterior_draws_per_chain": 512,
        "integration_steps": 12,
        "initial_step_size": 0.02,
        "target_acceptance_rate": 0.8,
        "chain_initial_jitter_std": 0.05,
        "divergence_energy_threshold": 1000.0,
        "inverse_mass_matrix": "identity",
        "adaptation": "dual-averaging step-size only",
    }
    if sampler != expected_sampler:
        raise SamplerMechanicsError("sampler mechanics changed")
    if mechanics.get("field_probe_indices") != [0, 1, 31, 32, 1024, 4096, 16384, 32767]:
        raise SamplerMechanicsError("field convergence probes changed")
    gates = controller.get("gates", {})
    if gates != {
        "MAP_gradient_RMS_max": 0.25,
        "derived_count_intensity_max": 1000000.0,
        "rank_normalized_split_Rhat_max": 1.05,
        "bulk_ESS_min": 100.0,
        "tail_ESS_min": 100.0,
        "divergence_fraction_max": 0.01,
        "all_draws_and_energies_finite": True,
        "all_derived_count_intensities_finite_and_nonnegative": True,
        "clipping_allowed": False,
    }:
        raise SamplerMechanicsError("sampler gates changed")
    if controller.get("truth_integrator") != {
        "lpt_order": 2,
        "a_start": 0.015625,
        "a_stop": 1.0,
        "a_lpt_maxstep": 0.0078125,
        "a_nbody_maxstep": 0.00390625,
        "mesh_to_particle_ratio": 1,
        "float_dtype": "float64",
    }:
        raise SamplerMechanicsError("truth integrator changed")
    quarantine = controller.get("hardware_quarantine", {})
    if (
        quarantine.get("excluded_node") != "syn06"
        or quarantine.get("known_failing_UUID")
        != "GPU-906578dd-9007-fdbd-3c6a-a0c5821e24d6"
        or quarantine.get("allowed_partition") != "a40"
        or quarantine.get("expected_nodes") != ["syn05", "syn07"]
    ):
        raise SamplerMechanicsError("hardware quarantine changed")

    base_path = Path(controller["lineage"]["base_program"]["path"])
    base = json.loads(base_path.read_text())
    if base.get("schema") != phasec_v5.SCHEMA:
        raise SamplerMechanicsError("base mock program schema mismatch")
    for binding in base.get("input_bindings", {}).values():
        source = Path(str(binding["path"]))
        if not source.is_file() or sha256_file(source) != binding["sha256"]:
            raise SamplerMechanicsError(f"base mock input mismatch: {source}")
    expected_eight = [
        {"index": index, "seed": 2026083000 + index, "arm": "ABCD"[index // 2]}
        for index in range(8)
    ]
    if base.get("mock_assignments") != expected_eight:
        raise SamplerMechanicsError("base mock assignments changed")
    for key in (
        "actual_observational_field_inference",
        "actual_2Mpp_count_read",
        "actual_CF4_velocity_datum_used",
        "validation_seed_access",
        "Phase_D_or_later",
    ):
        if base.get("authorization", {}).get(key) is not False:
            raise SamplerMechanicsError(f"base mock forbidden scope enabled: {key}")
    return controller, hashlib.sha256(payload).hexdigest(), copy.deepcopy(base)


def task_name(assignment: Mapping[str, object]) -> str:
    return (
        f"mechanics_{int(assignment['task_index']):02d}_mock_{int(assignment['mock_index']):02d}"
        f"_seed_{int(assignment['seed'])}_arm_{assignment['arm']}"
    )


def artifact_manifest(directory: Path, schema: str) -> dict[str, object]:
    rows = []
    for path in sorted(directory.iterdir()):
        if path.name in {"manifest.json", "COMPLETE"}:
            continue
        rows.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": schema, "files": rows}


def validate_manifest(directory: Path, expected_schema: str) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != expected_schema:
        raise SamplerMechanicsError("mechanics artifact manifest schema mismatch")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise SamplerMechanicsError("mechanics artifact manifest rows are invalid")
    expected_names = {
        path.name for path in directory.iterdir() if path.name not in {"manifest.json", "COMPLETE"}
    }
    if {row.get("name") for row in rows} != expected_names:
        raise SamplerMechanicsError("mechanics artifact manifest file set mismatch")
    for row in rows:
        path = directory / str(row["name"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise SamplerMechanicsError(f"mechanics artifact file mismatch: {path.name}")


def load_device_record(
    path: str | Path, controller: Mapping[str, object]
) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise SamplerMechanicsError("physical GPU device record is missing")
    record = json.loads(source.read_text())
    if record.get("schema") != "ouruniv-cf4-phasec-same-gpu-device-record-v1":
        raise SamplerMechanicsError("physical GPU device record schema mismatch")
    quarantine = controller["hardware_quarantine"]
    host = str(record.get("host", ""))
    if host not in quarantine["expected_nodes"] or host != os.uname().nodename.split(".")[0]:
        raise SamplerMechanicsError("physical GPU host is not the allocated allowed host")
    inventory = record.get("nvidia_smi_inventory")
    if not isinstance(inventory, list) or len(inventory) != 1:
        raise SamplerMechanicsError("physical GPU inventory must contain exactly one device")
    if quarantine["known_failing_UUID"] in str(inventory[0]):
        raise SamplerMechanicsError("known failing physical GPU was allocated")
    if record.get("jax", {}).get("backend") != "gpu":
        raise SamplerMechanicsError("physical GPU record does not report JAX GPU")
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "host": host,
        "nvidia_smi_inventory": inventory,
        "slurm": record.get("slurm", {}),
    }


def build_scan_truth(
    fine_white: np.ndarray,
    controller: Mapping[str, object],
    base: Mapping[str, object],
) -> dict[str, np.ndarray | float | int]:
    import jax
    import jax.numpy as jnp
    from pmwd import Configuration, SimpleLCDM, boltzmann, linear_modes, lpt, scatter
    from pmwd.nbody import nbody_init, nbody_step

    grid = base["grid"]
    cosmology = base["cosmology"]
    integrator = controller["truth_integrator"]
    n = int(grid["truth_N"])
    conf = Configuration(
        ptcl_spacing=float(grid["truth_cell_size_cMpc_h"]),
        ptcl_grid_shape=(n,) * 3,
        mesh_shape=int(integrator["mesh_to_particle_ratio"]),
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
        lpt_order=int(integrator["lpt_order"]),
        a_start=float(integrator["a_start"]),
        a_stop=float(integrator["a_stop"]),
        a_lpt_maxstep=float(integrator["a_lpt_maxstep"]),
        a_nbody_maxstep=float(integrator["a_nbody_maxstep"]),
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
        particles, observables = nbody_init(conf.a_nbody[0], particles, observables, cosmo, conf)

        def step(carry, scale_factors):
            return nbody_step(
                scale_factors[0],
                scale_factors[1],
                carry[0],
                carry[1],
                cosmo,
                conf,
            ), None

        pairs = jnp.stack((conf.a_nbody[:-1], conf.a_nbody[1:]), axis=1)
        (particles, _observables), _ = jax.lax.scan(step, (particles, observables), pairs)
        density = scatter(particles, conf)
        momentum = scatter(particles, conf, val=particles.vel * 100.0)
        return density, momentum

    density_j, momentum_j = forward(jnp.asarray(fine_white, dtype=jnp.float64))
    density = np.asarray(density_j, dtype=np.float64)
    momentum = np.asarray(momentum_j, dtype=np.float64)
    if density.shape != (n, n, n) or momentum.shape != (n, n, n, 3):
        raise SamplerMechanicsError("scan truth shape mismatch")
    if not np.all(np.isfinite(density)) or not np.all(np.isfinite(momentum)):
        raise SamplerMechanicsError("scan truth is nonfinite")
    if np.any(density < 0.0) or abs(float(density.mean()) - 1.0) > 2e-12:
        raise SamplerMechanicsError("scan truth is not conservative and nonnegative")
    velocity = np.divide(
        momentum,
        density[..., None],
        out=np.zeros_like(momentum),
        where=density[..., None] > 1e-10,
    )
    coarse_n = int(grid["inference_N"])
    coarse_mass = phasec_v5.block_sum(density, coarse_n)
    coarse_momentum = phasec_v5.block_sum(momentum, coarse_n)
    coarse_velocity = np.divide(
        coarse_momentum,
        coarse_mass[..., None],
        out=np.zeros_like(coarse_momentum),
        where=coarse_mass[..., None] > 1e-10,
    )
    ratio = n // coarse_n
    return {
        "fine_density": density,
        "fine_velocity": velocity,
        "coarse_delta": coarse_mass / ratio**3 - 1.0,
        "coarse_velocity": coarse_velocity,
        "density_min": float(density.min()),
        "density_max": float(density.max()),
        "empty_velocity_cell_count": int(np.count_nonzero(density <= 1e-10)),
        "a_nbody_step_count": int(conf.a_nbody_num),
    }


def build_standardized_model(
    base: Mapping[str, object],
    response6: np.ndarray,
    mock: Mapping[str, np.ndarray],
    design: Mapping[str, np.ndarray],
):
    import jax.numpy as jnp

    physical_nlp, physical_count_lambda, _physical_initial, metadata = (
        phasec_v5.build_inference_model(base, response6, mock, design)
    )
    field_size = int(metadata["field_size"])
    alpha_mean = jnp.asarray(metadata["alpha_mean"])
    logbias_mean = jnp.asarray(metadata["logbias_mean"])
    logfog_mean = jnp.asarray(metadata["logfog_mean"])
    alpha_sigma = float(base["inference_model"]["alpha_log_sigma"])
    bias_sigma = float(base["inference_model"]["bias_log_sigma"])
    fog_sigma = float(base["inference_model"]["FoG_log_sigma"])

    def standard_to_physical(vector):
        white = vector[:field_size]
        offset = field_size
        alpha = alpha_mean + alpha_sigma * vector[offset : offset + 6]
        offset += 6
        logbias = logbias_mean + bias_sigma * vector[offset : offset + 6]
        offset += 6
        logfog = logfog_mean + fog_sigma * vector[offset : offset + 6]
        offset += 6
        selection_unit = vector[offset : offset + 2]
        offset += 2
        q_unit = vector[offset : offset + 4]
        return jnp.concatenate((white, alpha, logbias, logfog, selection_unit, q_unit))

    def negative_log_posterior(vector):
        return physical_nlp(standard_to_physical(vector))

    def count_lambda(vector, response_scale=1.0):
        return physical_count_lambda(standard_to_physical(vector), response_scale)

    initial = np.zeros(field_size + 24, dtype=np.float64)
    if initial.size != 32792:
        raise SamplerMechanicsError("standardized initial vector dimension mismatch")
    metadata = dict(metadata)
    metadata["standard_to_physical"] = standard_to_physical
    metadata["parameterization"] = "all coordinates standard normal before likelihood"
    return negative_log_posterior, count_lambda, initial, metadata


def run_identity_hmc(
    negative_log_posterior: Callable,
    initial: np.ndarray,
    controller: Mapping[str, object],
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
    import blackjax
    import jax
    import jax.numpy as jnp
    from blackjax.adaptation.step_size import dual_averaging_adaptation
    from scipy.optimize import minimize

    map_contract = controller["mechanics"]["MAP"]
    sampler = controller["mechanics"]["sampler"]
    value_and_grad = jax.jit(jax.value_and_grad(negative_log_posterior))

    def scipy_value_gradient(vector):
        value, gradient = value_and_grad(jnp.asarray(vector, dtype=jnp.float64))
        return float(value), np.asarray(gradient, dtype=np.float64)

    optimization = minimize(
        scipy_value_gradient,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": int(map_contract["maximum_iterations"]),
            "maxls": int(map_contract["maximum_line_search_steps"]),
            "ftol": float(map_contract["objective_relative_tolerance"]),
            "gtol": 0.0,
        },
    )
    map_position = np.asarray(optimization.x, dtype=np.float64)
    map_value_j, map_gradient_j = value_and_grad(jnp.asarray(map_position))
    map_value = float(map_value_j)
    map_gradient = np.asarray(map_gradient_j, dtype=np.float64)
    if not math.isfinite(map_value) or not np.all(np.isfinite(map_gradient)):
        raise SamplerMechanicsError("MAP value or gradient is nonfinite")
    map_gradient_rms = float(np.linalg.norm(map_gradient) / math.sqrt(map_gradient.size))

    logdensity = lambda vector: -negative_log_posterior(vector)
    identity_inverse_mass = jnp.ones(initial.size, dtype=jnp.float64)
    hmc_kernel = blackjax.hmc.build_kernel(
        divergence_threshold=float(sampler["divergence_energy_threshold"])
    )
    da_init, da_update, da_final = dual_averaging_adaptation(
        float(sampler["target_acceptance_rate"])
    )
    integration_steps = int(sampler["integration_steps"])
    warmup_steps = int(sampler["warmup_steps"])
    draw_count = int(sampler["posterior_draws_per_chain"])

    @jax.jit
    def warmup_chain(key, start):
        state = blackjax.hmc.init(start, logdensity)
        adaptation_state = da_init(float(sampler["initial_step_size"]))
        keys = jax.random.split(key, warmup_steps)

        def one_step(carry, transition_key):
            current_state, current_adaptation = carry
            step_size = jnp.exp(current_adaptation.log_step_size)
            next_state, info = hmc_kernel(
                transition_key,
                current_state,
                logdensity,
                step_size,
                identity_inverse_mass,
                integration_steps,
            )
            next_adaptation = da_update(current_adaptation, info.acceptance_rate)
            record = (info.acceptance_rate, info.is_divergent, info.energy)
            return (next_state, next_adaptation), record

        (state, adaptation_state), records = jax.lax.scan(
            one_step, (state, adaptation_state), keys
        )
        return state, da_final(adaptation_state), records

    @jax.jit
    def sample_chain(key, state, step_size):
        keys = jax.random.split(key, draw_count)

        def one_step(current_state, transition_key):
            next_state, info = hmc_kernel(
                transition_key,
                current_state,
                logdensity,
                step_size,
                identity_inverse_mass,
                integration_steps,
            )
            record = (
                next_state.position,
                next_state.logdensity,
                info.acceptance_rate,
                info.is_divergent,
                info.energy,
            )
            return next_state, record

        return jax.lax.scan(one_step, state, keys)

    chain_count = int(sampler["chain_count"])
    jitter_rng = phasec_v5.tagged_rng(seed, int(controller["rng_tags"]["chain_initialization"]))
    draws = []
    step_sizes = []
    warmup_acceptance = []
    warmup_divergences = []
    warmup_energies_finite = []
    acceptance = []
    divergences = []
    energies = []
    logdensities = []
    for chain in range(chain_count):
        start = map_position + float(sampler["chain_initial_jitter_std"]) * jitter_rng.standard_normal(initial.size)
        warmup_key = jax.random.PRNGKey(np.uint32(seed + 1009 * (chain + 1)))
        state, step_size, warmup_records = warmup_chain(warmup_key, jnp.asarray(start))
        warmup_accept, warmup_divergent, warmup_energy = warmup_records
        sample_key = jax.random.PRNGKey(np.uint32(seed + 9173 * (chain + 1)))
        _last, records = sample_chain(sample_key, state, step_size)
        positions, logdensity_values, accept_values, divergent_values, energy_values = records
        positions_np = np.asarray(positions, dtype=np.float64)
        if not np.all(np.isfinite(positions_np)):
            raise SamplerMechanicsError(f"chain {chain} contains nonfinite positions")
        draws.append(positions_np)
        step_sizes.append(float(step_size))
        warmup_acceptance.append(float(np.mean(np.asarray(warmup_accept))))
        warmup_divergences.append(int(np.count_nonzero(np.asarray(warmup_divergent))))
        warmup_energies_finite.append(bool(np.all(np.isfinite(np.asarray(warmup_energy)))))
        acceptance.append(np.asarray(accept_values, dtype=np.float64))
        divergences.append(np.asarray(divergent_values, dtype=bool))
        energies.append(np.asarray(energy_values, dtype=np.float64))
        logdensities.append(np.asarray(logdensity_values, dtype=np.float64))

    return np.asarray(draws, dtype=np.float64), {
        "MAP_value": map_value,
        "MAP_gradient_norm": float(np.linalg.norm(map_gradient)),
        "MAP_gradient_RMS": map_gradient_rms,
        "MAP_iterations": int(optimization.nit),
        "MAP_status": int(optimization.status),
        "MAP_message": str(optimization.message),
        "step_size": np.asarray(step_sizes),
        "warmup_mean_acceptance": np.asarray(warmup_acceptance),
        "warmup_divergence_count": np.asarray(warmup_divergences),
        "warmup_energies_finite": np.asarray(warmup_energies_finite),
        "acceptance_rate": np.asarray(acceptance),
        "is_divergent": np.asarray(divergences),
        "energy": np.asarray(energies),
        "logdensity": np.asarray(logdensities),
        "identity_inverse_mass_dimension": initial.size,
    }


def derived_intensity_audit(
    draws: np.ndarray,
    count_lambda: Callable,
    maximum_allowed: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    import jax
    import jax.numpy as jnp

    def one_draw(vector):
        intensity, _delta, _velocity = count_lambda(vector, response_scale=1.0)
        finite = jnp.isfinite(intensity)
        return (
            jnp.all(finite),
            jnp.all(intensity >= 0.0),
            jnp.max(jnp.where(finite, intensity, 0.0)),
        )

    @jax.jit
    def one_chain(vectors):
        return jax.lax.map(one_draw, vectors)

    finite_rows = []
    nonnegative_rows = []
    maxima_rows = []
    for chain in range(draws.shape[0]):
        finite, nonnegative, maxima = one_chain(jnp.asarray(draws[chain], dtype=jnp.float64))
        finite_rows.append(np.asarray(finite, dtype=bool))
        nonnegative_rows.append(np.asarray(nonnegative, dtype=bool))
        maxima_rows.append(np.asarray(maxima, dtype=np.float64))
    finite_array = np.asarray(finite_rows)
    nonnegative_array = np.asarray(nonnegative_rows)
    maxima_array = np.asarray(maxima_rows)
    all_finite = bool(np.all(finite_array))
    all_nonnegative = bool(np.all(nonnegative_array))
    maximum = float(np.max(maxima_array)) if np.all(np.isfinite(maxima_array)) else None
    below_limit = maximum is not None and maximum <= maximum_allowed
    summary = {
        "evaluated_retained_draw_count": int(draws.shape[0] * draws.shape[1]),
        "expected_retained_draw_count": 2048,
        "all_finite": all_finite,
        "all_nonnegative": all_nonnegative,
        "maximum_count_intensity": maximum,
        "maximum_allowed": float(maximum_allowed),
        "below_maximum": bool(below_limit),
        "clipping_used": False,
        "pass": bool(all_finite and all_nonnegative and below_limit),
    }
    return summary, {
        "derived_intensity_all_finite_by_draw": finite_array,
        "derived_intensity_all_nonnegative_by_draw": nonnegative_array,
        "derived_intensity_maximum_by_draw": maxima_array,
    }


def convergence_projections(
    draws: np.ndarray,
    metadata: Mapping[str, object],
    base: Mapping[str, object],
    controller: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    field_size = int(metadata["field_size"])
    nuisance = draws[:, :, field_size:]
    field_indices = np.asarray(controller["mechanics"]["field_probe_indices"], dtype=int)
    field_probes = draws[:, :, field_indices]
    roi_names, roi_weights, _roi_effective = phasec_v5.build_roi_weights(base)
    transfer = np.asarray(metadata["transfer"])
    roi_sums = roi_weights.sum(axis=(1, 2, 3))
    roi_projection = np.empty((draws.shape[0], draws.shape[1], len(roi_names)), dtype=np.float64)
    chunk_size = int(controller["mechanics"]["host_projection_chunk_draws"])
    n = int(base["grid"]["inference_N"])
    for chain in range(draws.shape[0]):
        for start in range(0, draws.shape[1], chunk_size):
            stop = min(start + chunk_size, draws.shape[1])
            white = draws[chain, start:stop, :field_size].reshape((-1, n, n, n))
            modes = np.fft.fftn(white, axes=(1, 2, 3), norm="ortho") * transfer[None, ...]
            delta = np.fft.ifftn(modes, axes=(1, 2, 3), norm="ortho").real
            roi_projection[chain, start:stop] = np.tensordot(
                delta, roi_weights, axes=((1, 2, 3), (1, 2, 3))
            ) / roi_sums[None, :]
    logdensity = np.asarray(metadata["sampler_logdensity"], dtype=np.float64)[..., None]
    projections = np.concatenate((nuisance, field_probes, roi_projection, logdensity), axis=2)
    names = (
        [
            *[f"alpha_unit_{index}" for index in range(6)],
            *[f"logbias_unit_{index}" for index in range(6)],
            *[f"logFoG_unit_{index}" for index in range(6)],
            "selection_unit_radial",
            "selection_unit_angular",
            *[f"velocity_q_unit_{index}" for index in range(4)],
        ]
        + [f"white_coordinate_{index}" for index in field_indices]
        + [f"density_ROI_{name}" for name in roi_names]
        + ["logdensity"]
    )
    diagnostics = phasec_v5.chain_diagnostics(projections, names)
    return diagnostics, {
        "nuisance_unit_samples": nuisance.astype(np.float32),
        "field_probe_indices": field_indices,
        "field_probe_samples": field_probes.astype(np.float32),
        "roi_names": np.asarray(roi_names),
        "roi_density_projection_samples": roi_projection.astype(np.float32),
        "convergence_projection_samples": projections.astype(np.float32),
    }


def run_task(
    program_path: str | Path,
    output_root: str | Path,
    task_index: int,
    implementation_commit: str,
    device_record_path: str | Path,
) -> None:
    controller, program_sha, base = load_program(program_path)
    if task_index not in (0, 1):
        raise SamplerMechanicsError("mechanics task index must be 0 or 1")
    if len(implementation_commit) != 40:
        raise SamplerMechanicsError("implementation commit must be a full Git hash")
    assignment = controller["assignments"][task_index]
    seed = int(assignment["seed"])
    arm = str(assignment["arm"])
    output = Path(output_root) / task_name(assignment)
    staging = output.parent / f".{output.name}.staging"
    if output.exists() or staging.exists():
        raise SamplerMechanicsError("mechanics output or staging already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    import jax

    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise SamplerMechanicsError("mechanics task requires one allocated Slurm GPU")
    device_record = load_device_record(device_record_path, controller)
    response6, response4 = phasec_v5._load_selection(base)
    nbar, bias = phasec_v5._published_prior_arrays(base)
    args = phasec_v5.fixed.frozen_args(base["input_bindings"]["CF4_catalog"]["path"])
    design = linear.prepare_catalog(args)
    fine_white, _coarse_white, nesting = phasec_v5.nested_white_fields(
        seed,
        int(base["grid"]["inference_N"]),
        int(base["grid"]["truth_N"]),
        int(controller["rng_tags"]["high_k_white"]),
    )
    truth = build_scan_truth(fine_white, controller, base)
    truth_count_intensity, stress_meta = phasec_v5.truth_intensity(
        arm, truth, response6, response4, nbar, bias, base, seed
    )
    mock = phasec_v5.generate_mock_data(arm, seed, truth_count_intensity, truth, design, base)
    negative_log_posterior, count_lambda, initial, model_meta = build_standardized_model(
        base, response6, mock, design
    )
    draws, sampler_arrays = run_identity_hmc(
        negative_log_posterior, initial, controller, seed
    )
    derived_summary, derived_arrays = derived_intensity_audit(
        draws, count_lambda, float(controller["gates"]["derived_count_intensity_max"])
    )
    model_meta["sampler_logdensity"] = sampler_arrays["logdensity"]
    convergence, projection_arrays = convergence_projections(
        draws, model_meta, base, controller
    )
    gates = controller["gates"]
    divergence_fraction = float(np.mean(sampler_arrays["is_divergent"]))
    checks = {
        "MAP_gradient_RMS": float(sampler_arrays["MAP_gradient_RMS"])
        <= float(gates["MAP_gradient_RMS_max"]),
        "all_draws_finite": bool(np.all(np.isfinite(draws))),
        "all_sampling_energies_finite": bool(np.all(np.isfinite(sampler_arrays["energy"]))),
        "all_warmup_energies_finite": bool(np.all(sampler_arrays["warmup_energies_finite"])),
        "derived_intensity_gate": bool(derived_summary["pass"]),
        "Rhat": float(convergence["max_Rhat"])
        <= float(gates["rank_normalized_split_Rhat_max"]),
        "bulk_ESS": float(convergence["min_bulk_ESS"]) >= float(gates["bulk_ESS_min"]),
        "tail_ESS": float(convergence["min_tail_ESS"]) >= float(gates["tail_ESS_min"]),
        "divergence_fraction": divergence_fraction <= float(gates["divergence_fraction_max"]),
        "identity_inverse_mass": int(sampler_arrays["identity_inverse_mass_dimension"])
        == int(controller["mechanics"]["latent_dimension"]),
    }
    pilot_pass = bool(all(checks.values()))
    result = {
        "schema": TASK_SCHEMA,
        "status": "PASS_SAMPLER_MECHANICS_TASK" if pilot_pass else "NO_GO_SAMPLER_MECHANICS_TASK",
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
            "jax_version": jax.__version__,
            "float64": bool(jax.config.x64_enabled),
            "physical_GPU_record": device_record,
        },
        "truth": {
            "nested_white": nesting,
            "a_nbody_maxstep": float(controller["truth_integrator"]["a_nbody_maxstep"]),
            "a_nbody_step_count": int(truth["a_nbody_step_count"]),
            "density_min": float(truth["density_min"]),
            "density_max": float(truth["density_max"]),
            "stress": stress_meta,
        },
        "mock": {
            "counts_train_total": int(np.sum(mock["counts_train"])),
            "counts_holdout_total": int(np.sum(mock["counts_holdout"])),
            "CF4_geometry_row_count": int(np.asarray(design["pos"]).shape[0]),
            "actual_2Mpp_counts_read": False,
            "actual_CF4_velocity_datum_used": False,
            "validation_seed_read": False,
        },
        "MAP": {
            "value": float(sampler_arrays["MAP_value"]),
            "gradient_norm": float(sampler_arrays["MAP_gradient_norm"]),
            "gradient_RMS": float(sampler_arrays["MAP_gradient_RMS"]),
            "iterations": int(sampler_arrays["MAP_iterations"]),
            "status": int(sampler_arrays["MAP_status"]),
            "message": str(sampler_arrays["MAP_message"]),
        },
        "sampler": {
            "latent_dimension": int(draws.shape[-1]),
            "chain_count": int(draws.shape[0]),
            "draws_per_chain": int(draws.shape[1]),
            "integration_steps": int(controller["mechanics"]["sampler"]["integration_steps"]),
            "inverse_mass_matrix": "identity",
            "adaptation": "dual-averaging step-size only",
            "step_size": np.asarray(sampler_arrays["step_size"]).tolist(),
            "warmup_mean_acceptance": np.asarray(sampler_arrays["warmup_mean_acceptance"]).tolist(),
            "warmup_divergence_count": np.asarray(sampler_arrays["warmup_divergence_count"]).astype(int).tolist(),
            "sampling_mean_acceptance": float(np.mean(sampler_arrays["acceptance_rate"])),
            "sampling_divergence_fraction": divergence_fraction,
            "convergence": convergence,
        },
        "derived_count_intensity": derived_summary,
        "checks": checks,
        "pilot_pass": pilot_pass,
        "decision": {
            "remaining_mock_sampler_indices_allowed": False,
            "actual_observational_posterior_allowed": False,
            "validation_or_Phase_D_allowed": False,
        },
        "semantics": {
            "mock_only": True,
            "full_field_draws_not_stored": True,
            "posterior_predictive_RNG_called": False,
            "actual_present_day_density_or_velocity_posterior_created": False,
            "target_0p3_cMpc_h_reached": False,
        },
    }
    staging.mkdir(mode=0o700)
    np.savez_compressed(
        staging / "diagnostics.npz",
        **projection_arrays,
        **derived_arrays,
        sampler_acceptance_rate=np.asarray(sampler_arrays["acceptance_rate"]),
        sampler_is_divergent=np.asarray(sampler_arrays["is_divergent"]),
        sampler_energy=np.asarray(sampler_arrays["energy"]),
        sampler_logdensity=np.asarray(sampler_arrays["logdensity"]),
    )
    (staging / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = artifact_manifest(staging, "ouruniv-cf4-phasec-sampler-mechanics-task-manifest-v1")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete = {
        "schema": "ouruniv-cf4-phasec-sampler-mechanics-task-complete-v1",
        "result_sha256": sha256_file(staging / "result.json"),
        "manifest_sha256": sha256_file(staging / "manifest.json"),
        "pilot_pass": pilot_pass,
    }
    (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
    os.replace(staging, output)


def validate_task(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != TASK_FILES:
        raise SamplerMechanicsError("mechanics task artifact file set mismatch")
    result = json.loads((root / "result.json").read_text())
    if result.get("schema") != TASK_SCHEMA or not isinstance(result.get("pilot_pass"), bool):
        raise SamplerMechanicsError("mechanics task result schema mismatch")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("result_sha256") != sha256_file(root / "result.json"):
        raise SamplerMechanicsError("mechanics task result hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise SamplerMechanicsError("mechanics task manifest hash mismatch")
    if complete.get("pilot_pass") != result["pilot_pass"]:
        raise SamplerMechanicsError("mechanics task decision marker mismatch")
    validate_manifest(root, "ouruniv-cf4-phasec-sampler-mechanics-task-manifest-v1")
    return result


def aggregate(
    program_path: str | Path,
    output_root: str | Path,
    aggregate_output: str | Path,
    implementation_commit: str,
) -> None:
    controller, program_sha, _base = load_program(program_path)
    if len(implementation_commit) != 40:
        raise SamplerMechanicsError("implementation commit must be a full Git hash")
    output = Path(aggregate_output)
    staging = output.parent / f".{output.name}.staging"
    if output.exists() or staging.exists():
        raise SamplerMechanicsError("mechanics aggregate output already exists")
    outcomes = []
    for assignment in controller["assignments"]:
        task_dir = Path(output_root) / task_name(assignment)
        try:
            result = validate_task(task_dir)
            if result.get("assignment") != assignment:
                raise SamplerMechanicsError("mechanics task assignment mismatch")
            outcomes.append(
                {
                    "assignment": assignment,
                    "artifact_status": "VALID",
                    "pilot_pass": bool(result["pilot_pass"]),
                    "result_sha256": sha256_file(task_dir / "result.json"),
                    "MAP": result["MAP"],
                    "sampler": result["sampler"],
                    "derived_count_intensity": result["derived_count_intensity"],
                    "checks": result["checks"],
                    "physical_GPU_record": result["environment"]["physical_GPU_record"],
                }
            )
        except Exception as exc:
            outcomes.append(
                {
                    "assignment": assignment,
                    "artifact_status": "MISSING_OR_INVALID",
                    "pilot_pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    aggregate_pass = bool(len(outcomes) == 2 and all(row["pilot_pass"] for row in outcomes))
    result = {
        "schema": AGGREGATE_SCHEMA,
        "status": (
            "PASS_BOTH_SAMPLER_MECHANICS_TASKS_RELEASE_REMAINING_MOCK_INDICES"
            if aggregate_pass
            else "NO_GO_SAMPLER_MECHANICS_STOP_BEFORE_REMAINING_MOCK_INDICES"
        ),
        "program": {"path": str(Path(program_path).resolve()), "sha256": program_sha},
        "implementation_commit": implementation_commit,
        "task_count": len(outcomes),
        "valid_artifact_count": sum(row["artifact_status"] == "VALID" for row in outcomes),
        "passing_task_count": sum(row["pilot_pass"] for row in outcomes),
        "both_pilot_tasks_pass": aggregate_pass,
        "outcomes": outcomes,
        "decision": {
            "remaining_mock_sampler_indices_allowed": (
                [1, 2, 3, 4, 5, 7] if aggregate_pass else []
            ),
            "actual_observational_posterior_allowed": False,
            "validation_or_Phase_D_allowed": False,
        },
        "scope_firewall": controller["scope_firewall"],
    }
    staging.mkdir(mode=0o700)
    (staging / "aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = artifact_manifest(staging, "ouruniv-cf4-phasec-sampler-mechanics-aggregate-manifest-v1")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete = {
        "schema": "ouruniv-cf4-phasec-sampler-mechanics-aggregate-complete-v1",
        "aggregate_sha256": sha256_file(staging / "aggregate.json"),
        "manifest_sha256": sha256_file(staging / "manifest.json"),
        "both_pilot_tasks_pass": aggregate_pass,
    }
    (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
    os.replace(staging, output)


def validate_aggregate(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != AGGREGATE_FILES:
        raise SamplerMechanicsError("mechanics aggregate artifact set mismatch")
    result = json.loads((root / "aggregate.json").read_text())
    if result.get("schema") != AGGREGATE_SCHEMA or result.get("task_count") != 2:
        raise SamplerMechanicsError("mechanics aggregate schema mismatch")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("aggregate_sha256") != sha256_file(root / "aggregate.json"):
        raise SamplerMechanicsError("mechanics aggregate hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise SamplerMechanicsError("mechanics aggregate manifest hash mismatch")
    if complete.get("both_pilot_tasks_pass") != result.get("both_pilot_tasks_pass"):
        raise SamplerMechanicsError("mechanics aggregate decision mismatch")
    validate_manifest(root, "ouruniv-cf4-phasec-sampler-mechanics-aggregate-manifest-v1")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--program", required=True)
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--task-index", type=int, required=True)
    run_parser.add_argument("--implementation-commit", required=True)
    run_parser.add_argument("--device-record", required=True)
    validate_parser = subparsers.add_parser("validate-task")
    validate_parser.add_argument("--directory", required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--program", required=True)
    aggregate_parser.add_argument("--output-root", required=True)
    aggregate_parser.add_argument("--aggregate-output", required=True)
    aggregate_parser.add_argument("--implementation-commit", required=True)
    validate_aggregate_parser = subparsers.add_parser("validate-aggregate")
    validate_aggregate_parser.add_argument("--directory", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run_task(
            args.program,
            args.output_root,
            args.task_index,
            args.implementation_commit,
            args.device_record,
        )
    elif args.command == "validate-task":
        validate_task(args.directory)
    elif args.command == "aggregate":
        aggregate(args.program, args.output_root, args.aggregate_output, args.implementation_commit)
    elif args.command == "validate-aggregate":
        validate_aggregate(args.directory)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
