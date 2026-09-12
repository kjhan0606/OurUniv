#!/usr/bin/env python3
"""Mock-only sampler-mechanics V5 with conservative initialization and warm-up diagnostics."""

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

import cf4_phasec_sampler_mechanics_pilot as v1


SCHEMA = "ouruniv-cf4-phasec-sampler-mechanics-pilot-v5"
TASK_SCHEMA = "ouruniv-cf4-phasec-sampler-mechanics-v5-task-v1"
AGGREGATE_SCHEMA = "ouruniv-cf4-phasec-sampler-mechanics-v5-aggregate-v1"
TASK_FILES = {"diagnostics.npz", "result.json", "manifest.json", "COMPLETE"}
AGGREGATE_FILES = {"aggregate.json", "manifest.json", "COMPLETE"}
INITIAL_STEP_SIZE = 0.0005


class SamplerMechanicsV5Error(ValueError):
    """The frozen L=32 sampler-mechanics contract was violated."""


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
    source = Path(path)
    payload = source.read_bytes()
    try:
        controller = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SamplerMechanicsV5Error("cannot parse v5 sampler program") from exc
    if controller.get("schema") != SCHEMA:
        raise SamplerMechanicsV5Error("v5 sampler schema mismatch")
    auth = controller.get("authorization", {})
    for key in (
        "replacement_Phase_C_mock_sampler_mechanics_v5",
        "Slurm_GPU_array_indices_0_and_6",
        "Slurm_CPU_aggregate",
        "GPFS_read_bound_mock_inputs",
        "GPFS_write_new_outputs_only",
    ):
        if auth.get(key) is not True:
            raise SamplerMechanicsV5Error(f"missing v5 authorization: {key}")
    for key in (
        "actual_observational_field_inference",
        "actual_2Mpp_count_read",
        "actual_CF4_velocity_datum_used",
        "validation_seed_access",
        "Phase_D_or_later",
    ):
        if auth.get(key) is not False:
            raise SamplerMechanicsV5Error(f"forbidden v5 scope enabled: {key}")
    if controller.get("assignments") != expected_assignments():
        raise SamplerMechanicsV5Error("v5 sampler assignments changed")
    expected_lineage = {
        "base_program",
        "replacement_program",
        "generator_gate_PASS_record",
        "syn06_GPU_sweep_record",
        "previous_sampler_program",
        "previous_sampler_implementation",
        "acf_audit_record",
        "V5_implementation",
        "fixed_implementation",
        "linear_implementation",
        "sampler_mechanics_implementation",
        "failed_v2_program",
        "failed_v2_implementation",
        "failed_v2_aggregate",
        "failed_v2_task_00_result",
        "failed_v2_task_01_result",
        "failed_v3_program",
        "failed_v3_implementation",
        "failed_v3_aggregate",
        "failed_v3_task_00_result",
        "failed_v3_task_01_result",
        "failed_v4_program",
        "failed_v4_implementation",
        "failed_v4_aggregate",
        "failed_v4_task_00_result",
        "failed_v4_task_01_result",
    }
    lineage = controller.get("lineage", {})
    if set(lineage) != expected_lineage:
        raise SamplerMechanicsV5Error("v5 lineage set changed")
    for name, binding in lineage.items():
        source_path = Path(str(binding.get("path", "")))
        expected_hash = str(binding.get("sha256", ""))
        if expected_hash.startswith("TO_BE_FILLED"):
            raise SamplerMechanicsV5Error("v5 implementation is not frozen")
        if not source_path.is_file() or sha256_file(source_path) != expected_hash:
            raise SamplerMechanicsV5Error(f"v5 lineage mismatch: {name}")

    mechanics = controller.get("mechanics", {})
    if mechanics.get("latent_dimension") != 32792:
        raise SamplerMechanicsV5Error("v5 latent dimension changed")
    expected_map = {
        "optimizer": "SciPy L-BFGS-B with exact JAX gradient",
        "maximum_iterations": 1536,
        "maximum_line_search_steps": 80,
        "objective_relative_tolerance": 1e-12,
        "projected_gradient_infinity_tolerance": 0.25,
        "require_optimizer_success": True,
        "finite_value_and_gradient_required": True,
    }
    if mechanics.get("MAP") != expected_map:
        raise SamplerMechanicsV5Error("v5 MAP mechanics changed")
    expected_sampler = {
        "algorithm": "BlackJAX static Euclidean HMC",
        "chain_count": 4,
        "warmup_steps": 1024,
        "posterior_draws_per_chain": 2048,
        "integration_steps": 32,
        "initial_step_size": INITIAL_STEP_SIZE,
        "target_acceptance_rate": 0.9,
        "chain_initial_jitter_std": 0.05,
        "divergence_energy_threshold": 1000.0,
        "inverse_mass_matrix": "identity",
        "adaptation": "dual-averaging raw proposal with fixed maximum used step",
        "maximum_step_size": 0.04,
    }
    if mechanics.get("sampler") != expected_sampler:
        raise SamplerMechanicsV5Error("v5 sampler mechanics changed")
    if mechanics.get("field_probe_indices") != [0, 1, 31, 32, 1024, 4096, 16384, 32767]:
        raise SamplerMechanicsV5Error("v5 field probes changed")
    expected_gates = {
        "MAP_gradient_RMS_max": 0.25,
        "MAP_optimizer_success_required": True,
        "derived_count_intensity_max": 1000000.0,
        "rank_normalized_split_Rhat_max": 1.05,
        "bulk_ESS_min": 100.0,
        "tail_ESS_min": 100.0,
        "warmup_divergence_fraction_max": 0.01,
        "divergence_fraction_max": 0.01,
        "all_draws_and_energies_finite": True,
        "all_derived_count_intensities_finite_and_nonnegative": True,
        "warmup_energy_trace_must_be_stored": True,
        "clipping_allowed": False,
    }
    if controller.get("gates") != expected_gates:
        raise SamplerMechanicsV5Error("v5 sampler gates changed")
    if controller.get("truth_integrator") != {
        "lpt_order": 2,
        "a_start": 0.015625,
        "a_stop": 1.0,
        "a_lpt_maxstep": 0.0078125,
        "a_nbody_maxstep": 0.00390625,
        "mesh_to_particle_ratio": 1,
        "float_dtype": "float64",
    }:
        raise SamplerMechanicsV5Error("v5 truth integrator changed")
    quarantine = controller.get("hardware_quarantine", {})
    if (
        quarantine.get("excluded_node") != "syn06"
        or quarantine.get("known_failing_UUID") != "GPU-906578dd-9007-fdbd-3c6a-a0c5821e24d6"
        or quarantine.get("allowed_partition") != "a40"
        or quarantine.get("expected_nodes") != ["syn05", "syn07"]
    ):
        raise SamplerMechanicsV5Error("v5 hardware quarantine changed")

    base_path = Path(controller["lineage"]["base_program"]["path"])
    base = json.loads(base_path.read_text())
    if base.get("schema") != v1.phasec_v5.SCHEMA:
        raise SamplerMechanicsV5Error("base mock schema mismatch")
    for binding in base.get("input_bindings", {}).values():
        input_path = Path(str(binding["path"]))
        if not input_path.is_file() or sha256_file(input_path) != binding["sha256"]:
            raise SamplerMechanicsV5Error(f"base mock input mismatch: {input_path}")
    expected_base_assignments = [
        {"index": index, "seed": 2026083000 + index, "arm": "ABCD"[index // 2]}
        for index in range(8)
    ]
    if base.get("mock_assignments") != expected_base_assignments:
        raise SamplerMechanicsV5Error("base mock assignments changed")
    for key in (
        "actual_observational_field_inference",
        "actual_2Mpp_count_read",
        "actual_CF4_velocity_datum_used",
        "validation_seed_access",
        "Phase_D_or_later",
    ):
        if base.get("authorization", {}).get(key) is not False:
            raise SamplerMechanicsV5Error(f"base mock forbidden scope enabled: {key}")
    return controller, hashlib.sha256(payload).hexdigest(), copy.deepcopy(base)


def task_name(assignment: Mapping[str, object]) -> str:
    return (
        f"mechanics_v5_{int(assignment['task_index']):02d}_mock_{int(assignment['mock_index']):02d}"
        f"_seed_{int(assignment['seed'])}_arm_{assignment['arm']}"
    )


def artifact_manifest(directory: Path, schema: str) -> dict[str, object]:
    files = []
    for path in sorted(directory.iterdir()):
        if path.name in {"manifest.json", "COMPLETE"}:
            continue
        files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": schema, "files": files}


def validate_manifest(directory: Path, schema: str) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != schema:
        raise SamplerMechanicsV5Error("v5 manifest schema mismatch")
    rows = manifest.get("files")
    expected = {path.name for path in directory.iterdir() if path.name not in {"manifest.json", "COMPLETE"}}
    if not isinstance(rows, list) or {row.get("name") for row in rows} != expected:
        raise SamplerMechanicsV5Error("v5 manifest file set mismatch")
    for row in rows:
        path = directory / str(row["name"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise SamplerMechanicsV5Error(f"v5 artifact hash mismatch: {path.name}")


def load_device_record(path: str | Path, controller: Mapping[str, object]) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise SamplerMechanicsV5Error("v5 device record is missing")
    record = json.loads(source.read_text())
    if record.get("schema") != "ouruniv-cf4-phasec-same-gpu-device-record-v1":
        raise SamplerMechanicsV5Error("v5 device record schema mismatch")
    quarantine = controller["hardware_quarantine"]
    host = str(record.get("host", ""))
    if host not in quarantine["expected_nodes"] or host != os.uname().nodename.split(".")[0]:
        raise SamplerMechanicsV5Error("v5 device host is not allowed")
    inventory = record.get("nvidia_smi_inventory")
    if not isinstance(inventory, list) or len(inventory) != 1:
        raise SamplerMechanicsV5Error("v5 GPU inventory must contain one device")
    if quarantine["known_failing_UUID"] in str(inventory[0]):
        raise SamplerMechanicsV5Error("v5 known failing GPU allocated")
    if record.get("jax", {}).get("backend") != "gpu":
        raise SamplerMechanicsV5Error("v5 device record does not report GPU backend")
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "host": host,
        "nvidia_smi_inventory": inventory,
        "slurm": record.get("slurm", {}),
    }


def run_identity_hmc(
    negative_log_posterior: Callable,
    initial: np.ndarray,
    controller: Mapping[str, object],
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int | bool]]:
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
            "gtol": float(map_contract["projected_gradient_infinity_tolerance"]),
        },
    )
    map_position = np.asarray(optimization.x, dtype=np.float64)
    map_value_j, map_gradient_j = value_and_grad(jnp.asarray(map_position))
    map_value = float(map_value_j)
    map_gradient = np.asarray(map_gradient_j, dtype=np.float64)
    if not math.isfinite(map_value) or not np.all(np.isfinite(map_gradient)):
        raise SamplerMechanicsV5Error("v5 MAP value or gradient is nonfinite")
    map_gradient_rms = float(np.linalg.norm(map_gradient) / math.sqrt(map_gradient.size))
    if not np.all(np.isfinite(map_position)):
        raise SamplerMechanicsV5Error("v5 MAP position is nonfinite")

    logdensity = lambda vector: -negative_log_posterior(vector)
    warmup_initialization_values = []
    warmup_initialization_gradient_rms = []
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
    maximum_step_size = jnp.asarray(sampler["maximum_step_size"], dtype=jnp.float64)

    @jax.jit
    def warmup_chain(key, start):
        state = blackjax.hmc.init(start, logdensity)
        adaptation_state = da_init(float(sampler["initial_step_size"]))
        keys = jax.random.split(key, warmup_steps)

        def one_step(carry, transition_key):
            current_state, current_adaptation, step_index = carry
            adapted_raw_step_size = jnp.exp(current_adaptation.log_step_size)
            # The first recorded raw proposal is frozen to the configured
            # initial value.  Every HMC transition uses min(raw, cap); raw is
            # retained so cap activity cannot be hidden by the applied trace.
            raw_step_size = jnp.where(
                step_index == 0,
                jnp.asarray(sampler["initial_step_size"], dtype=jnp.float64),
                adapted_raw_step_size,
            )
            used_step_size = jnp.minimum(raw_step_size, maximum_step_size)
            cap_applied = raw_step_size > maximum_step_size
            next_state, info = hmc_kernel(
                transition_key,
                current_state,
                logdensity,
                used_step_size,
                identity_inverse_mass,
                integration_steps,
            )
            next_adaptation = da_update(current_adaptation, info.acceptance_rate)
            record = (
                raw_step_size,
                used_step_size,
                cap_applied,
                info.acceptance_rate,
                info.is_divergent,
                info.energy,
            )
            return (next_state, next_adaptation, step_index + 1), record

        (state, adaptation_state, _step_index), records = jax.lax.scan(
            one_step, (state, adaptation_state, jnp.asarray(0, dtype=jnp.int32)), keys
        )
        final_raw_step_size = da_final(adaptation_state)
        final_used_step_size = jnp.minimum(final_raw_step_size, maximum_step_size)
        final_cap_applied = final_raw_step_size > maximum_step_size
        return state, final_raw_step_size, final_used_step_size, final_cap_applied, records

    @jax.jit
    def sample_chain(key, state, used_step_size):
        keys = jax.random.split(key, draw_count)

        def one_step(current_state, transition_key):
            next_state, info = hmc_kernel(
                transition_key,
                current_state,
                logdensity,
                used_step_size,
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
    jitter_rng = v1.phasec_v5.tagged_rng(seed, int(controller["rng_tags"]["chain_initialization"]))
    draws = []
    sampling_raw_step_sizes = []
    sampling_used_step_sizes = []
    sampling_cap_applied = []
    warmup_acceptance = []
    warmup_acceptance_traces = []
    warmup_raw_step_size_traces = []
    warmup_used_step_size_traces = []
    warmup_cap_applied_traces = []
    warmup_divergences = []
    warmup_divergence_flags = []
    warmup_energies = []
    acceptance = []
    divergences = []
    energies = []
    logdensities = []
    for chain in range(chain_count):
        start = map_position + float(sampler["chain_initial_jitter_std"]) * jitter_rng.standard_normal(initial.size)
        start_value, start_gradient = value_and_grad(jnp.asarray(start, dtype=jnp.float64))
        start_value = float(start_value)
        start_gradient = np.asarray(start_gradient, dtype=np.float64)
        if not math.isfinite(start_value) or not np.all(np.isfinite(start_gradient)):
            raise SamplerMechanicsV5Error(f"v5 chain {chain} initial value or gradient is nonfinite")
        warmup_initialization_values.append(start_value)
        warmup_initialization_gradient_rms.append(
            float(np.linalg.norm(start_gradient) / math.sqrt(start_gradient.size))
        )
        warmup_key = jax.random.PRNGKey(np.uint32(seed + 1009 * (chain + 1)))
        state, final_raw_step_size, final_used_step_size, final_cap_applied, warmup_records = warmup_chain(
            warmup_key, jnp.asarray(start)
        )
        (
            warmup_raw_step_size,
            warmup_used_step_size,
            warmup_cap_applied,
            warmup_accept,
            warmup_divergent,
            warmup_energy,
        ) = warmup_records
        sample_key = jax.random.PRNGKey(np.uint32(seed + 9173 * (chain + 1)))
        _last, records = sample_chain(sample_key, state, final_used_step_size)
        positions, logdensity_values, accept_values, divergent_values, energy_values = records
        positions_np = np.asarray(positions, dtype=np.float64)
        if not np.all(np.isfinite(positions_np)):
            raise SamplerMechanicsV5Error(f"v5 chain {chain} contains nonfinite positions")
        draws.append(positions_np)
        sampling_raw_step_sizes.append(float(final_raw_step_size))
        sampling_used_step_sizes.append(float(final_used_step_size))
        sampling_cap_applied.append(bool(final_cap_applied))
        warmup_accept_np = np.asarray(warmup_accept, dtype=np.float64)
        warmup_raw_step_size_np = np.asarray(warmup_raw_step_size, dtype=np.float64)
        warmup_used_step_size_np = np.asarray(warmup_used_step_size, dtype=np.float64)
        warmup_cap_applied_np = np.asarray(warmup_cap_applied, dtype=bool)
        warmup_divergent_np = np.asarray(warmup_divergent, dtype=bool)
        warmup_energy_np = np.asarray(warmup_energy, dtype=np.float64)
        warmup_acceptance.append(float(np.mean(warmup_accept_np)))
        warmup_acceptance_traces.append(warmup_accept_np)
        warmup_raw_step_size_traces.append(warmup_raw_step_size_np)
        warmup_used_step_size_traces.append(warmup_used_step_size_np)
        warmup_cap_applied_traces.append(warmup_cap_applied_np)
        warmup_divergences.append(int(np.count_nonzero(warmup_divergent_np)))
        warmup_divergence_flags.append(warmup_divergent_np)
        warmup_energies.append(warmup_energy_np)
        acceptance.append(np.asarray(accept_values, dtype=np.float64))
        divergences.append(np.asarray(divergent_values, dtype=bool))
        energies.append(np.asarray(energy_values, dtype=np.float64))
        logdensities.append(np.asarray(logdensity_values, dtype=np.float64))

    warmup_energies_np = np.asarray(warmup_energies, dtype=np.float64)
    warmup_acceptance_traces_np = np.asarray(warmup_acceptance_traces, dtype=np.float64)
    warmup_raw_step_size_traces_np = np.asarray(warmup_raw_step_size_traces, dtype=np.float64)
    warmup_used_step_size_traces_np = np.asarray(warmup_used_step_size_traces, dtype=np.float64)
    warmup_cap_applied_traces_np = np.asarray(warmup_cap_applied_traces, dtype=bool)
    warmup_divergence_flags_np = np.asarray(warmup_divergence_flags, dtype=bool)
    block_size = 128
    if warmup_steps % block_size != 0:
        raise SamplerMechanicsV5Error("v5 warmup steps must be divisible by diagnostic block size")
    block_count = warmup_steps // block_size
    warmup_energy_finite_by_block = np.all(
        np.isfinite(warmup_energies_np).reshape(chain_count, block_count, block_size), axis=2
    )
    warmup_divergence_fraction_by_block = np.mean(
        warmup_divergence_flags_np.reshape(chain_count, block_count, block_size), axis=2
    )
    return np.asarray(draws, dtype=np.float64), {
        "MAP_value": map_value,
        "MAP_gradient_norm": float(np.linalg.norm(map_gradient)),
        "MAP_gradient_RMS": map_gradient_rms,
        "MAP_iterations": int(optimization.nit),
        "MAP_status": int(optimization.status),
        "MAP_success": bool(optimization.success),
        "MAP_message": str(optimization.message),
        "sampling_raw_step_size": np.asarray(sampling_raw_step_sizes, dtype=np.float64),
        "sampling_used_step_size": np.asarray(sampling_used_step_sizes, dtype=np.float64),
        "sampling_cap_applied": np.asarray(sampling_cap_applied, dtype=bool),
        "warmup_mean_acceptance": np.asarray(warmup_acceptance),
        "warmup_acceptance_rate": warmup_acceptance_traces_np,
        "warmup_raw_step_size": warmup_raw_step_size_traces_np,
        "warmup_used_step_size": warmup_used_step_size_traces_np,
        "warmup_cap_applied": warmup_cap_applied_traces_np,
        "warmup_cap_applied_count": np.count_nonzero(warmup_cap_applied_traces_np, axis=1).astype(int),
        "warmup_divergence_count": np.asarray(warmup_divergences),
        "warmup_is_divergent": warmup_divergence_flags_np,
        "warmup_energy": warmup_energies_np,
        "warmup_energies_finite": np.all(np.isfinite(warmup_energies_np), axis=1),
        "warmup_energy_finite_by_block": warmup_energy_finite_by_block,
        "warmup_divergence_fraction_by_block": warmup_divergence_fraction_by_block,
        "warmup_initialization_value": np.asarray(warmup_initialization_values),
        "warmup_initialization_gradient_RMS": np.asarray(warmup_initialization_gradient_rms),
        "acceptance_rate": np.asarray(acceptance),
        "is_divergent": np.asarray(divergences),
        "energy": np.asarray(energies),
        "logdensity": np.asarray(logdensities),
        "identity_inverse_mass_dimension": initial.size,
    }


def run_task(
    program_path: str | Path,
    output_root: str | Path,
    task_index: int,
    implementation_commit: str,
    device_record_path: str | Path,
) -> None:
    controller, program_sha, base = load_program(program_path)
    if task_index not in (0, 1) or len(implementation_commit) != 40:
        raise SamplerMechanicsV5Error("invalid v5 task index or commit")
    assignment = controller["assignments"][task_index]
    output = Path(output_root) / task_name(assignment)
    staging = output.parent / f".{output.name}.staging"
    if output.exists() or staging.exists():
        raise SamplerMechanicsV5Error("v5 output or staging already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    import jax

    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise SamplerMechanicsV5Error("v5 task requires one Slurm GPU")
    device_record = load_device_record(device_record_path, controller)
    response6, response4 = v1.phasec_v5._load_selection(base)
    nbar, bias = v1.phasec_v5._published_prior_arrays(base)
    args = v1.phasec_v5.fixed.frozen_args(base["input_bindings"]["CF4_catalog"]["path"])
    design = v1.linear.prepare_catalog(args)
    fine_white, _coarse_white, nesting = v1.phasec_v5.nested_white_fields(
        int(assignment["seed"]),
        int(base["grid"]["inference_N"]),
        int(base["grid"]["truth_N"]),
        int(controller["rng_tags"]["high_k_white"]),
    )
    truth = v1.build_scan_truth(fine_white, controller, base)
    truth_intensity, stress_meta = v1.phasec_v5.truth_intensity(
        str(assignment["arm"]), truth, response6, response4, nbar, bias, base, int(assignment["seed"])
    )
    mock = v1.phasec_v5.generate_mock_data(
        str(assignment["arm"]), int(assignment["seed"]), truth_intensity, truth, design, base
    )
    negative_log_posterior, count_lambda, initial, model_meta = v1.build_standardized_model(
        base, response6, mock, design
    )
    draws, sampler_arrays = run_identity_hmc(
        negative_log_posterior, initial, controller, int(assignment["seed"])
    )
    derived_summary, derived_arrays = v1.derived_intensity_audit(
        draws, count_lambda, float(controller["gates"]["derived_count_intensity_max"])
    )
    derived_summary["expected_retained_draw_count"] = int(draws.shape[0] * draws.shape[1])
    model_meta["sampler_logdensity"] = sampler_arrays["logdensity"]
    convergence, projection_arrays = v1.convergence_projections(
        draws, model_meta, base, controller
    )
    gates = controller["gates"]
    sampler_config = controller["mechanics"]["sampler"]
    sampling_divergence_fraction = float(np.mean(sampler_arrays["is_divergent"]))
    warmup_divergence_fraction = float(
        np.sum(sampler_arrays["warmup_divergence_count"])
        / (draws.shape[0] * int(sampler_config["warmup_steps"]))
    )
    expected_warmup_shape = (
        draws.shape[0],
        int(controller["mechanics"]["sampler"]["warmup_steps"]),
    )
    warmup_trace_stored = bool(
        np.asarray(sampler_arrays["warmup_energy"]).shape
        == expected_warmup_shape
        and np.asarray(sampler_arrays["warmup_is_divergent"]).shape
        == expected_warmup_shape
        and np.asarray(sampler_arrays["warmup_acceptance_rate"]).shape
        == expected_warmup_shape
        and np.asarray(sampler_arrays["warmup_raw_step_size"]).shape
        == expected_warmup_shape
        and np.asarray(sampler_arrays["warmup_used_step_size"]).shape
        == expected_warmup_shape
        and np.asarray(sampler_arrays["warmup_cap_applied"]).shape
        == expected_warmup_shape
    )
    maximum_step_size = float(sampler_config["maximum_step_size"])
    warmup_raw_step_size_np = np.asarray(sampler_arrays["warmup_raw_step_size"], dtype=np.float64)
    warmup_used_step_size_np = np.asarray(sampler_arrays["warmup_used_step_size"], dtype=np.float64)
    warmup_cap_applied_np = np.asarray(sampler_arrays["warmup_cap_applied"], dtype=bool)
    warmup_raw_step_size_finite_positive = bool(
        np.all(np.isfinite(warmup_raw_step_size_np)) and np.all(warmup_raw_step_size_np > 0.0)
    )
    warmup_used_step_size_finite_positive = bool(
        np.all(np.isfinite(warmup_used_step_size_np)) and np.all(warmup_used_step_size_np > 0.0)
    )
    warmup_step_size_initial_match = bool(
        warmup_raw_step_size_finite_positive
        and warmup_used_step_size_finite_positive
        and warmup_raw_step_size_np.shape == expected_warmup_shape
        and warmup_used_step_size_np.shape == expected_warmup_shape
        and np.array_equal(
            warmup_raw_step_size_np[:, 0],
            np.full(draws.shape[0], float(sampler_config["initial_step_size"]), dtype=np.float64),
        )
        and np.array_equal(
            warmup_used_step_size_np[:, 0],
            np.full(draws.shape[0], float(sampler_config["initial_step_size"]), dtype=np.float64),
        )
    )
    warmup_cap_contract = bool(
        warmup_raw_step_size_finite_positive
        and warmup_used_step_size_finite_positive
        and np.all(warmup_used_step_size_np <= maximum_step_size)
        and np.array_equal(warmup_used_step_size_np, np.minimum(warmup_raw_step_size_np, maximum_step_size))
        and np.array_equal(warmup_cap_applied_np, warmup_raw_step_size_np > maximum_step_size)
    )
    sampling_raw_step_size_np = np.asarray(sampler_arrays["sampling_raw_step_size"], dtype=np.float64)
    sampling_used_step_size_np = np.asarray(sampler_arrays["sampling_used_step_size"], dtype=np.float64)
    sampling_cap_applied_np = np.asarray(sampler_arrays["sampling_cap_applied"], dtype=bool)
    sampling_step_size_contract = bool(
        sampling_raw_step_size_np.shape == (draws.shape[0],)
        and sampling_used_step_size_np.shape == (draws.shape[0],)
        and sampling_cap_applied_np.shape == (draws.shape[0],)
        and np.all(np.isfinite(sampling_raw_step_size_np))
        and np.all(sampling_raw_step_size_np > 0.0)
        and np.all(np.isfinite(sampling_used_step_size_np))
        and np.all(sampling_used_step_size_np > 0.0)
        and np.all(sampling_used_step_size_np <= maximum_step_size)
        and np.array_equal(sampling_used_step_size_np, np.minimum(sampling_raw_step_size_np, maximum_step_size))
        and np.array_equal(sampling_cap_applied_np, sampling_raw_step_size_np > maximum_step_size)
    )
    checks = {
        "MAP_optimizer_success": bool(sampler_arrays["MAP_success"])
        if gates["MAP_optimizer_success_required"]
        else True,
        "MAP_gradient_RMS": float(sampler_arrays["MAP_gradient_RMS"])
        <= float(gates["MAP_gradient_RMS_max"]),
        "all_draws_finite": bool(np.all(np.isfinite(draws))),
        "all_sampling_energies_finite": bool(np.all(np.isfinite(sampler_arrays["energy"]))),
        "all_warmup_energies_finite": bool(np.all(sampler_arrays["warmup_energies_finite"])),
        "warmup_energy_trace_stored": warmup_trace_stored,
        "warmup_raw_used_step_size_contract": bool(
            warmup_step_size_initial_match and warmup_cap_contract and warmup_trace_stored
        ),
        "sampling_raw_used_step_size_contract": sampling_step_size_contract,
        "warmup_divergence_fraction": warmup_divergence_fraction
        <= float(gates["warmup_divergence_fraction_max"]),
        "derived_draw_count": derived_summary["evaluated_retained_draw_count"]
        == derived_summary["expected_retained_draw_count"],
        "derived_intensity_gate": bool(derived_summary["pass"]),
        "Rhat": float(convergence["max_Rhat"])
        <= float(gates["rank_normalized_split_Rhat_max"]),
        "bulk_ESS": float(convergence["min_bulk_ESS"]) >= float(gates["bulk_ESS_min"]),
        "tail_ESS": float(convergence["min_tail_ESS"]) >= float(gates["tail_ESS_min"]),
        "divergence_fraction": sampling_divergence_fraction
        <= float(gates["divergence_fraction_max"]),
        "identity_inverse_mass": int(sampler_arrays["identity_inverse_mass_dimension"])
        == int(controller["mechanics"]["latent_dimension"]),
    }
    pilot_pass = bool(all(checks.values()))
    result = {
        "schema": TASK_SCHEMA,
        "status": "PASS_SAMPLER_MECHANICS_V5_TASK" if pilot_pass else "NO_GO_SAMPLER_MECHANICS_V5_TASK",
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
            "success": bool(sampler_arrays["MAP_success"]),
            "message": str(sampler_arrays["MAP_message"]),
        },
        "sampler": {
            "latent_dimension": int(draws.shape[-1]),
            "chain_count": int(draws.shape[0]),
            "draws_per_chain": int(draws.shape[1]),
            "integration_steps": int(controller["mechanics"]["sampler"]["integration_steps"]),
            "inverse_mass_matrix": "identity",
            "adaptation": "dual-averaging raw proposal with fixed maximum used step",
            "maximum_step_size": maximum_step_size,
            "sampling_raw_step_size": sampling_raw_step_size_np.tolist(),
            "sampling_used_step_size": sampling_used_step_size_np.tolist(),
            "sampling_cap_applied": sampling_cap_applied_np.tolist(),
            "sampling_cap_applied_count": int(np.count_nonzero(sampling_cap_applied_np)),
            "sampling_raw_used_step_size_contract": sampling_step_size_contract,
            "warmup_mean_acceptance": np.asarray(sampler_arrays["warmup_mean_acceptance"]).tolist(),
            "warmup_initialization_value": np.asarray(sampler_arrays["warmup_initialization_value"]).tolist(),
            "warmup_initialization_gradient_RMS": np.asarray(
                sampler_arrays["warmup_initialization_gradient_RMS"]
            ).tolist(),
            "warmup_divergence_count": np.asarray(sampler_arrays["warmup_divergence_count"]).astype(int).tolist(),
            "warmup_divergence_fraction": warmup_divergence_fraction,
            "warmup_divergence_fraction_by_block": np.asarray(
                sampler_arrays["warmup_divergence_fraction_by_block"]
            ).tolist(),
            "warmup_energy_finite_by_block": np.asarray(
                sampler_arrays["warmup_energy_finite_by_block"]
            ).tolist(),
            "warmup_raw_step_size_first": warmup_raw_step_size_np[:, 0].tolist(),
            "warmup_raw_step_size_last": warmup_raw_step_size_np[:, -1].tolist(),
            "warmup_used_step_size_first": warmup_used_step_size_np[:, 0].tolist(),
            "warmup_used_step_size_last": warmup_used_step_size_np[:, -1].tolist(),
            "warmup_cap_applied_count": np.count_nonzero(
                warmup_cap_applied_np, axis=1
            ).astype(int).tolist(),
            "warmup_raw_used_step_size_contract": bool(
                warmup_step_size_initial_match and warmup_cap_contract and warmup_trace_stored
            ),
            "warmup_energy_trace_stored": warmup_trace_stored,
            "warmup_energy_min": (
                float(np.min(sampler_arrays["warmup_energy"]))
                if np.all(np.isfinite(sampler_arrays["warmup_energy"]))
                else None
            ),
            "warmup_energy_max": (
                float(np.max(sampler_arrays["warmup_energy"]))
                if np.all(np.isfinite(sampler_arrays["warmup_energy"]))
                else None
            ),
            "sampling_mean_acceptance": float(np.mean(sampler_arrays["acceptance_rate"])),
            "sampling_divergence_fraction": sampling_divergence_fraction,
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
        sampler_warmup_acceptance_rate=np.asarray(sampler_arrays["warmup_acceptance_rate"]),
        sampler_warmup_is_divergent=np.asarray(sampler_arrays["warmup_is_divergent"]),
        sampler_warmup_divergence_count=np.asarray(sampler_arrays["warmup_divergence_count"]),
        sampler_warmup_energy=np.asarray(sampler_arrays["warmup_energy"]),
        sampler_warmup_raw_step_size=np.asarray(sampler_arrays["warmup_raw_step_size"]),
        sampler_warmup_used_step_size=np.asarray(sampler_arrays["warmup_used_step_size"]),
        sampler_warmup_cap_applied=np.asarray(sampler_arrays["warmup_cap_applied"]),
        sampler_sampling_raw_step_size=np.asarray(sampler_arrays["sampling_raw_step_size"]),
        sampler_sampling_used_step_size=np.asarray(sampler_arrays["sampling_used_step_size"]),
        sampler_sampling_cap_applied=np.asarray(sampler_arrays["sampling_cap_applied"]),
        sampler_warmup_energy_finite_by_block=np.asarray(
            sampler_arrays["warmup_energy_finite_by_block"]
        ),
        sampler_warmup_divergence_fraction_by_block=np.asarray(
            sampler_arrays["warmup_divergence_fraction_by_block"]
        ),
        sampler_warmup_initialization_value=np.asarray(sampler_arrays["warmup_initialization_value"]),
        sampler_warmup_initialization_gradient_RMS=np.asarray(
            sampler_arrays["warmup_initialization_gradient_RMS"]
        ),
    )
    (staging / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (staging / "manifest.json").write_text(
        json.dumps(artifact_manifest(staging, "ouruniv-cf4-phasec-sampler-mechanics-v5-task-manifest-v1"), indent=2, sort_keys=True)
        + "\n"
    )
    complete = {
        "schema": "ouruniv-cf4-phasec-sampler-mechanics-v5-task-complete-v1",
        "result_sha256": sha256_file(staging / "result.json"),
        "manifest_sha256": sha256_file(staging / "manifest.json"),
        "pilot_pass": pilot_pass,
    }
    (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
    os.replace(staging, output)


def validate_task(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != TASK_FILES:
        raise SamplerMechanicsV5Error("v5 task artifact set mismatch")
    result = json.loads((root / "result.json").read_text())
    if result.get("schema") != TASK_SCHEMA or not isinstance(result.get("pilot_pass"), bool):
        raise SamplerMechanicsV5Error("v5 task result schema mismatch")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("result_sha256") != sha256_file(root / "result.json"):
        raise SamplerMechanicsV5Error("v5 task result hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise SamplerMechanicsV5Error("v5 task manifest hash mismatch")
    if complete.get("pilot_pass") != result["pilot_pass"]:
        raise SamplerMechanicsV5Error("v5 task decision mismatch")
    validate_manifest(root, "ouruniv-cf4-phasec-sampler-mechanics-v5-task-manifest-v1")
    expected_warmup_shape = (4, 1024)
    expected_warmup_block_shape = (4, 8)
    expected_sampling_shape = (4, 2048)
    with np.load(root / "diagnostics.npz", allow_pickle=False) as archive:
        required = {
            "sampler_is_divergent",
            "sampler_energy",
            "sampler_warmup_energy",
            "sampler_warmup_is_divergent",
            "sampler_warmup_acceptance_rate",
            "sampler_warmup_raw_step_size",
            "sampler_warmup_used_step_size",
            "sampler_warmup_cap_applied",
            "sampler_sampling_raw_step_size",
            "sampler_sampling_used_step_size",
            "sampler_sampling_cap_applied",
            "sampler_warmup_divergence_count",
            "sampler_warmup_energy_finite_by_block",
            "sampler_warmup_divergence_fraction_by_block",
            "sampler_warmup_initialization_value",
            "sampler_warmup_initialization_gradient_RMS",
        }
        if not required.issubset(archive.files):
            raise SamplerMechanicsV5Error("v5 warmup traces are missing")
        for name in (
            "sampler_warmup_energy",
            "sampler_warmup_is_divergent",
            "sampler_warmup_acceptance_rate",
            "sampler_warmup_raw_step_size",
            "sampler_warmup_used_step_size",
            "sampler_warmup_cap_applied",
        ):
            if np.asarray(archive[name]).shape != expected_warmup_shape:
                raise SamplerMechanicsV5Error("v5 warmup trace shape mismatch")
        if np.asarray(archive["sampler_warmup_divergence_count"]).shape != (4,):
            raise SamplerMechanicsV5Error("v5 warmup count shape mismatch")
        for name in (
            "sampler_warmup_energy_finite_by_block",
            "sampler_warmup_divergence_fraction_by_block",
        ):
            if np.asarray(archive[name]).shape != expected_warmup_block_shape:
                raise SamplerMechanicsV5Error("v5 warmup block trace shape mismatch")
        for name in (
            "sampler_warmup_initialization_value",
            "sampler_warmup_initialization_gradient_RMS",
        ):
            if np.asarray(archive[name]).shape != (4,):
                raise SamplerMechanicsV5Error("v5 warmup initialization trace shape mismatch")
        warmup_is_divergent = np.asarray(archive["sampler_warmup_is_divergent"], dtype=bool)
        warmup_energy = np.asarray(archive["sampler_warmup_energy"], dtype=np.float64)
        warmup_raw_step_size = np.asarray(archive["sampler_warmup_raw_step_size"], dtype=np.float64)
        warmup_used_step_size = np.asarray(archive["sampler_warmup_used_step_size"], dtype=np.float64)
        warmup_cap_applied = np.asarray(archive["sampler_warmup_cap_applied"], dtype=bool)
        sampling_raw_step_size = np.asarray(archive["sampler_sampling_raw_step_size"], dtype=np.float64)
        sampling_used_step_size = np.asarray(archive["sampler_sampling_used_step_size"], dtype=np.float64)
        sampling_cap_applied = np.asarray(archive["sampler_sampling_cap_applied"], dtype=bool)
        warmup_divergence_count = np.count_nonzero(warmup_is_divergent, axis=1).astype(int)
        warmup_divergence_fraction = float(np.sum(warmup_divergence_count) / warmup_is_divergent.size)
        warmup_divergence_fraction_by_block = np.mean(
            warmup_is_divergent.reshape(4, 8, 128), axis=2
        )
        warmup_energy_finite_by_block = np.all(
            np.isfinite(warmup_energy).reshape(4, 8, 128), axis=2
        )
        warmup_energy_finite = bool(np.all(np.isfinite(warmup_energy)))
        maximum_step_size = 0.04
        warmup_raw_step_size_finite_positive = bool(
            np.all(np.isfinite(warmup_raw_step_size)) and np.all(warmup_raw_step_size > 0.0)
        )
        warmup_used_step_size_finite_positive = bool(
            np.all(np.isfinite(warmup_used_step_size)) and np.all(warmup_used_step_size > 0.0)
        )
        warmup_step_size_initial_match = bool(
            warmup_raw_step_size_finite_positive
            and warmup_used_step_size_finite_positive
            and np.array_equal(
                warmup_raw_step_size[:, 0],
                np.full(expected_warmup_shape[0], INITIAL_STEP_SIZE, dtype=np.float64),
            )
            and np.array_equal(
                warmup_used_step_size[:, 0],
                np.full(expected_warmup_shape[0], INITIAL_STEP_SIZE, dtype=np.float64),
            )
        )
        warmup_step_size_contract = bool(
            warmup_raw_step_size_finite_positive
            and warmup_used_step_size_finite_positive
            and np.all(warmup_used_step_size <= maximum_step_size)
            and np.array_equal(warmup_used_step_size, np.minimum(warmup_raw_step_size, maximum_step_size))
            and np.array_equal(warmup_cap_applied, warmup_raw_step_size > maximum_step_size)
        )
        sampling_step_size_contract = bool(
            sampling_raw_step_size.shape == (4,)
            and sampling_used_step_size.shape == (4,)
            and sampling_cap_applied.shape == (4,)
            and np.all(np.isfinite(sampling_raw_step_size))
            and np.all(sampling_raw_step_size > 0.0)
            and np.all(np.isfinite(sampling_used_step_size))
            and np.all(sampling_used_step_size > 0.0)
            and np.all(sampling_used_step_size <= maximum_step_size)
            and np.array_equal(sampling_used_step_size, np.minimum(sampling_raw_step_size, maximum_step_size))
            and np.array_equal(sampling_cap_applied, sampling_raw_step_size > maximum_step_size)
        )
        if not (warmup_raw_step_size_finite_positive and warmup_used_step_size_finite_positive):
            raise SamplerMechanicsV5Error("v5 warmup raw/used step sizes must be finite and positive")
        if not (
            sampling_raw_step_size.shape == (4,)
            and sampling_used_step_size.shape == (4,)
            and sampling_cap_applied.shape == (4,)
            and np.all(np.isfinite(sampling_raw_step_size))
            and np.all(sampling_raw_step_size > 0.0)
            and np.all(np.isfinite(sampling_used_step_size))
            and np.all(sampling_used_step_size > 0.0)
        ):
            raise SamplerMechanicsV5Error("v5 sampling raw/used step sizes must be finite and positive")
        sampling_is_divergent = np.asarray(archive["sampler_is_divergent"], dtype=bool)
        sampling_energy = np.asarray(archive["sampler_energy"], dtype=np.float64)
        if sampling_is_divergent.shape != expected_sampling_shape or sampling_energy.shape != expected_sampling_shape:
            raise SamplerMechanicsV5Error("v5 sampling trace shape mismatch")
        sampling_divergence_fraction = float(np.mean(sampling_is_divergent))
        sampling_energy_finite = bool(np.all(np.isfinite(sampling_energy)))
        initialization_value = np.asarray(
            archive["sampler_warmup_initialization_value"], dtype=np.float64
        )
        initialization_gradient_rms = np.asarray(
            archive["sampler_warmup_initialization_gradient_RMS"], dtype=np.float64
        )
        sampler = result.get("sampler", {})
        checks = result.get("checks", {})
        if not isinstance(sampler, dict) or not isinstance(checks, dict):
            raise SamplerMechanicsV5Error("v5 task sampler/check records are missing")
        if not np.array_equal(
            np.asarray(sampler.get("warmup_divergence_count"), dtype=int),
            warmup_divergence_count,
        ):
            raise SamplerMechanicsV5Error("v5 warmup divergence count does not match NPZ")
        if float(sampler.get("warmup_divergence_fraction")) != warmup_divergence_fraction:
            raise SamplerMechanicsV5Error("v5 warmup divergence fraction does not match NPZ")
        if not np.array_equal(
            np.asarray(sampler.get("warmup_divergence_fraction_by_block"), dtype=np.float64),
            warmup_divergence_fraction_by_block,
        ):
            raise SamplerMechanicsV5Error("v5 warmup block divergence trace does not match NPZ")
        if not np.array_equal(
            np.asarray(sampler.get("warmup_energy_finite_by_block"), dtype=bool),
            warmup_energy_finite_by_block,
        ):
            raise SamplerMechanicsV5Error("v5 warmup block energy trace does not match NPZ")
        if not np.array_equal(
            np.asarray(sampler.get("warmup_initialization_value"), dtype=np.float64),
            initialization_value,
        ) or not np.array_equal(
            np.asarray(sampler.get("warmup_initialization_gradient_RMS"), dtype=np.float64),
            initialization_gradient_rms,
        ):
            raise SamplerMechanicsV5Error("v5 initialization trace does not match NPZ")
        if not warmup_step_size_initial_match:
            raise SamplerMechanicsV5Error("v5 warmup first-used step size does not match frozen initial step")
        if float(sampler.get("maximum_step_size")) != maximum_step_size:
            raise SamplerMechanicsV5Error("v5 maximum step size does not match frozen contract")
        for key, expected in (
            ("warmup_raw_step_size_first", warmup_raw_step_size[:, 0]),
            ("warmup_raw_step_size_last", warmup_raw_step_size[:, -1]),
            ("warmup_used_step_size_first", warmup_used_step_size[:, 0]),
            ("warmup_used_step_size_last", warmup_used_step_size[:, -1]),
            ("sampling_raw_step_size", sampling_raw_step_size),
            ("sampling_used_step_size", sampling_used_step_size),
            ("sampling_cap_applied", sampling_cap_applied),
        ):
            if not np.array_equal(np.asarray(sampler.get(key)), expected):
                raise SamplerMechanicsV5Error(f"v5 {key} does not match NPZ")
        warmup_cap_count = np.count_nonzero(warmup_cap_applied, axis=1).astype(int)
        sampling_cap_count = int(np.count_nonzero(sampling_cap_applied))
        if not np.array_equal(
            np.asarray(sampler.get("warmup_cap_applied_count"), dtype=int), warmup_cap_count
        ) or int(sampler.get("sampling_cap_applied_count")) != sampling_cap_count:
            raise SamplerMechanicsV5Error("v5 cap-applied counts do not match NPZ masks")
        warmup_contract_gate = bool(
            warmup_step_size_initial_match and warmup_step_size_contract
        )
        if sampler.get("warmup_raw_used_step_size_contract") is not warmup_contract_gate:
            raise SamplerMechanicsV5Error("v5 warmup raw/used step-size gate does not match NPZ")
        if checks.get("warmup_raw_used_step_size_contract") is not warmup_contract_gate:
            raise SamplerMechanicsV5Error("v5 warmup raw/used step-size check does not match NPZ")
        if sampler.get("sampling_raw_used_step_size_contract") is not sampling_step_size_contract:
            raise SamplerMechanicsV5Error("v5 sampling raw/used step-size gate does not match NPZ")
        if checks.get("sampling_raw_used_step_size_contract") is not sampling_step_size_contract:
            raise SamplerMechanicsV5Error("v5 sampling raw/used step-size check does not match NPZ")
        if checks.get("all_warmup_energies_finite") is not warmup_energy_finite:
            raise SamplerMechanicsV5Error("v5 warmup energy finiteness gate does not match NPZ")
        if checks.get("warmup_divergence_fraction") is not (
            warmup_divergence_fraction <= 0.01
        ):
            raise SamplerMechanicsV5Error("v5 warmup divergence gate does not match NPZ")
        if float(sampler.get("sampling_divergence_fraction")) != sampling_divergence_fraction:
            raise SamplerMechanicsV5Error("v5 sampling divergence fraction does not match NPZ")
        if checks.get("divergence_fraction") is not (
            sampling_divergence_fraction <= 0.01
        ):
            raise SamplerMechanicsV5Error("v5 sampling divergence gate does not match NPZ")
        if checks.get("all_sampling_energies_finite") is not sampling_energy_finite:
            raise SamplerMechanicsV5Error("v5 sampling energy finiteness gate does not match NPZ")
        if sampler.get("warmup_energy_trace_stored") is not True or checks.get(
            "warmup_energy_trace_stored"
        ) is not True:
            raise SamplerMechanicsV5Error("v5 warmup trace storage gate is not true")
        if checks.get("warmup_energy_trace_stored") is not True:
            raise SamplerMechanicsV5Error("v5 warmup trace storage check is inconsistent")
        if bool(checks) and bool(result.get("pilot_pass")) != bool(all(checks.values())):
            raise SamplerMechanicsV5Error("v5 pilot pass does not match task checks")
    if result["checks"].get("warmup_energy_trace_stored") is not True:
        raise SamplerMechanicsV5Error("v5 warmup energy trace storage gate failed")
    return result


def aggregate(
    program_path: str | Path,
    output_root: str | Path,
    aggregate_output: str | Path,
    implementation_commit: str,
) -> None:
    controller, program_sha, _base = load_program(program_path)
    if len(implementation_commit) != 40:
        raise SamplerMechanicsV5Error("v5 aggregate commit must be full hash")
    implementation_sha = sha256_file(__file__)
    output = Path(aggregate_output)
    staging = output.parent / f".{output.name}.staging"
    if output.exists() or staging.exists():
        raise SamplerMechanicsV5Error("v5 aggregate output already exists")
    outcomes = []
    for assignment in controller["assignments"]:
        task_dir = Path(output_root) / task_name(assignment)
        try:
            result = validate_task(task_dir)
            if result.get("assignment") != assignment:
                raise SamplerMechanicsV5Error("v5 task assignment mismatch")
            if result.get("program", {}).get("sha256") != program_sha:
                raise SamplerMechanicsV5Error("v5 task program hash mismatch")
            implementation = result.get("implementation", {})
            if implementation.get("path") != str(Path(__file__).resolve()):
                raise SamplerMechanicsV5Error("v5 task implementation path mismatch")
            if implementation.get("sha256") != implementation_sha:
                raise SamplerMechanicsV5Error("v5 task implementation hash mismatch")
            if implementation.get("commit") != implementation_commit:
                raise SamplerMechanicsV5Error("v5 task implementation commit mismatch")
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
        "status": "PASS_BOTH_SAMPLER_MECHANICS_V5_TASKS" if aggregate_pass else "NO_GO_SAMPLER_MECHANICS_V5_STOP",
        "program": {"path": str(Path(program_path).resolve()), "sha256": program_sha},
        "implementation_commit": implementation_commit,
        "task_count": len(outcomes),
        "valid_artifact_count": sum(row["artifact_status"] == "VALID" for row in outcomes),
        "passing_task_count": sum(row["pilot_pass"] for row in outcomes),
        "both_pilot_tasks_pass": aggregate_pass,
        "outcomes": outcomes,
        "decision": {
            "remaining_mock_sampler_indices_allowed": [1, 2, 3, 4, 5, 7] if aggregate_pass else [],
            "actual_observational_posterior_allowed": False,
            "validation_or_Phase_D_allowed": False,
        },
        "scope_firewall": controller["scope_firewall"],
    }
    staging.mkdir(mode=0o700)
    (staging / "aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (staging / "manifest.json").write_text(
        json.dumps(artifact_manifest(staging, "ouruniv-cf4-phasec-sampler-mechanics-v5-aggregate-manifest-v1"), indent=2, sort_keys=True)
        + "\n"
    )
    complete = {
        "schema": "ouruniv-cf4-phasec-sampler-mechanics-v5-aggregate-complete-v1",
        "aggregate_sha256": sha256_file(staging / "aggregate.json"),
        "manifest_sha256": sha256_file(staging / "manifest.json"),
        "both_pilot_tasks_pass": aggregate_pass,
    }
    (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
    os.replace(staging, output)


def validate_aggregate(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != AGGREGATE_FILES:
        raise SamplerMechanicsV5Error("v5 aggregate artifact set mismatch")
    result = json.loads((root / "aggregate.json").read_text())
    if result.get("schema") != AGGREGATE_SCHEMA or result.get("task_count") != 2:
        raise SamplerMechanicsV5Error("v5 aggregate schema mismatch")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("aggregate_sha256") != sha256_file(root / "aggregate.json"):
        raise SamplerMechanicsV5Error("v5 aggregate hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise SamplerMechanicsV5Error("v5 aggregate manifest hash mismatch")
    if complete.get("both_pilot_tasks_pass") != result.get("both_pilot_tasks_pass"):
        raise SamplerMechanicsV5Error("v5 aggregate decision mismatch")
    validate_manifest(root, "ouruniv-cf4-phasec-sampler-mechanics-v5-aggregate-manifest-v1")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--program", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--task-index", type=int, required=True)
    run.add_argument("--implementation-commit", required=True)
    run.add_argument("--device-record", required=True)
    check = sub.add_parser("validate-task")
    check.add_argument("--directory", required=True)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--program", required=True)
    agg.add_argument("--output-root", required=True)
    agg.add_argument("--aggregate-output", required=True)
    agg.add_argument("--implementation-commit", required=True)
    agg_check = sub.add_parser("validate-aggregate")
    agg_check.add_argument("--directory", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        run_task(args.program, args.output_root, args.task_index, args.implementation_commit, args.device_record)
    elif args.command == "validate-task":
        validate_task(args.directory)
    elif args.command == "aggregate":
        aggregate(args.program, args.output_root, args.aggregate_output, args.implementation_commit)
    else:
        validate_aggregate(args.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
