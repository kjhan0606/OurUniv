#!/usr/bin/env python3
"""All-seed PMWD generator gate for the replacement mock-only Phase C."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

import cf4_datum_bearing_z0_phasec_pilot as phasec_v5


SCHEMA = "ouruniv-cf4-datum-bearing-z0-phasec-redesign-v1"
TASK_SCHEMA = "ouruniv-cf4-phasec-redesign-generator-task-v1"
AGGREGATE_SCHEMA = "ouruniv-cf4-phasec-redesign-generator-aggregate-v1"
TASK_FILES = {"result.json", "manifest.json", "COMPLETE"}
AGGREGATE_FILES = {"aggregate.json", "manifest.json", "COMPLETE"}


class GeneratorGateError(ValueError):
    """The frozen replacement Phase-C generator contract was violated."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_assignments() -> list[dict[str, object]]:
    return [
        {"index": index, "seed": 2026083000 + index, "arm": "ABCD"[index // 2]}
        for index in range(8)
    ]


def load_program(path: str | Path) -> tuple[dict[str, object], str]:
    payload = Path(path).read_bytes()
    try:
        program = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise GeneratorGateError("cannot parse replacement Phase-C program") from exc
    if program.get("schema") != SCHEMA:
        raise GeneratorGateError("replacement Phase-C schema mismatch")
    authorization = program.get("authorization", {})
    for key in (
        "replacement_Phase_C_mock_design_implementation_and_execution",
        "Slurm_GPU_generator_array",
        "Slurm_CPU_generator_aggregate",
        "GPFS_read_bound_inputs",
        "GPFS_write_new_outputs_only",
    ):
        if authorization.get(key) is not True:
            raise GeneratorGateError(f"missing replacement authorization: {key}")
    for key in (
        "actual_observational_field_inference",
        "actual_2Mpp_count_read",
        "actual_CF4_velocity_datum_used",
        "validation_seed_access",
        "Phase_D_or_later",
        "IC_PM_HOP_RAMSES",
    ):
        if authorization.get(key) is not False:
            raise GeneratorGateError(f"forbidden replacement authorization enabled: {key}")
    if program.get("mock_assignments") != expected_assignments():
        raise GeneratorGateError("the exact eight-seed assignment changed")
    if program.get("grid") != {
        "box_size_cMpc_h": 384.0,
        "inference_N": 32,
        "inference_cell_size_cMpc_h": 12.0,
        "truth_N": 64,
        "truth_cell_size_cMpc_h": 6.0,
    }:
        raise GeneratorGateError("replacement grid contract changed")
    candidates = program.get("generator_gate", {}).get("candidates")
    if candidates != [
        {
            "name": "convergence_1_over_128",
            "a_nbody_maxstep": 0.0078125,
            "role": "coarser time-step convergence member; never selected for later truth",
        },
        {
            "name": "production_1_over_256",
            "a_nbody_maxstep": 0.00390625,
            "role": "only truth integration allowed in the replacement sampler pilot if the all-seed gate passes",
        },
    ]:
        raise GeneratorGateError("generator candidate ladder changed")
    for name, binding in program.get("lineage", {}).items():
        source = Path(str(binding.get("path", "")))
        expected_hash = str(binding.get("sha256", ""))
        if not source.is_file():
            raise GeneratorGateError(f"missing lineage binding: {name}")
        if expected_hash.startswith("TO_BE_FILLED"):
            raise GeneratorGateError("generator implementation binding is not frozen")
        if sha256_file(source) != expected_hash:
            raise GeneratorGateError(f"lineage SHA256 mismatch: {name}")
    return program, hashlib.sha256(payload).hexdigest()


def task_name(index: int, seed: int) -> str:
    return f"seed_{index:02d}_{seed}"


def artifact_manifest(directory: Path, schema: str) -> dict[str, object]:
    files = []
    for path in sorted(directory.iterdir()):
        if path.name in {"manifest.json", "COMPLETE"}:
            continue
        files.append(
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return {"schema": schema, "files": files}


def finite_range(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    return float(finite.min()), float(finite.max())


def build_candidate(
    fine_white: np.ndarray,
    program: Mapping[str, object],
    candidate: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    import jax
    import jax.numpy as jnp
    from pmwd import Configuration, SimpleLCDM, boltzmann, linear_modes, lpt, nbody, scatter

    grid = program["grid"]
    generator = program["generator_gate"]
    cosmology = program["cosmology"]
    n = int(grid["truth_N"])
    coarse_n = int(grid["inference_N"])
    conf = Configuration(
        ptcl_spacing=float(grid["truth_cell_size_cMpc_h"]),
        ptcl_grid_shape=(n,) * 3,
        mesh_shape=int(generator["mesh_to_particle_ratio"]),
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
        lpt_order=int(generator["lpt_order"]),
        a_start=float(generator["a_start"]),
        a_stop=float(generator["a_stop"]),
        a_lpt_maxstep=float(generator["a_lpt_maxstep"]),
        a_nbody_maxstep=float(candidate["a_nbody_maxstep"]),
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

    try:
        density_j, momentum_j = forward(jnp.asarray(fine_white, dtype=jnp.float64))
        density = np.asarray(density_j, dtype=np.float64)
        momentum = np.asarray(momentum_j, dtype=np.float64)
    except Exception as exc:  # preserve a durable all-seed gate outcome
        return {
            "name": str(candidate["name"]),
            "a_nbody_maxstep": float(candidate["a_nbody_maxstep"]),
            "a_nbody_step_count": int(conf.a_nbody_num),
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "pass": False,
        }, None

    expected_density_shape = (n, n, n)
    expected_momentum_shape = (n, n, n, 3)
    shape_pass = density.shape == expected_density_shape and momentum.shape == expected_momentum_shape
    density_finite = bool(np.all(np.isfinite(density)))
    momentum_finite = bool(np.all(np.isfinite(momentum)))
    density_min, density_max = finite_range(density)
    momentum_min, momentum_max = finite_range(momentum)
    density_mean = float(np.mean(density)) if density_finite else None
    mean_error = abs(density_mean - 1.0) if density_mean is not None else None
    density_nonnegative = density_min is not None and density_min >= 0.0
    gates = program["generator_gate"]["per_candidate_gates"]
    base_pass = bool(
        shape_pass
        and density_finite
        and momentum_finite
        and density_nonnegative
        and mean_error is not None
        and mean_error <= float(gates["density_mean_absolute_error_max"])
    )
    summary: dict[str, object] = {
        "name": str(candidate["name"]),
        "a_nbody_maxstep": float(candidate["a_nbody_maxstep"]),
        "a_nbody_actual_step": float(conf.a_nbody_step),
        "a_nbody_step_count": int(conf.a_nbody_num),
        "density_shape": list(density.shape),
        "momentum_shape": list(momentum.shape),
        "density_all_finite": density_finite,
        "momentum_all_finite": momentum_finite,
        "density_nonnegative": bool(density_nonnegative),
        "density_mean": density_mean,
        "density_mean_absolute_error": mean_error,
        "density_min": density_min,
        "density_max": density_max,
        "momentum_min": momentum_min,
        "momentum_max": momentum_max,
        "pass": False,
    }
    if not base_pass:
        return summary, None

    velocity = np.divide(
        momentum,
        density[..., None],
        out=np.zeros_like(momentum),
        where=density[..., None] > 1.0e-10,
    )
    coarse_mass = phasec_v5.block_sum(density, coarse_n)
    coarse_momentum = phasec_v5.block_sum(momentum, coarse_n)
    coarse_velocity = np.divide(
        coarse_momentum,
        coarse_mass[..., None],
        out=np.zeros_like(coarse_momentum),
        where=coarse_mass[..., None] > 1.0e-10,
    )
    ratio = n // coarse_n
    coarse_delta = coarse_mass / ratio**3 - 1.0
    coarse_finite = bool(
        np.all(np.isfinite(velocity))
        and np.all(np.isfinite(coarse_delta))
        and np.all(np.isfinite(coarse_velocity))
    )
    summary["fine_velocity_all_finite"] = bool(np.all(np.isfinite(velocity)))
    summary["coarse_density_and_velocity_all_finite"] = coarse_finite
    summary["empty_fine_velocity_cell_count"] = int(np.count_nonzero(density <= 1.0e-10))
    summary["pass"] = coarse_finite
    if not coarse_finite:
        return summary, None
    return summary, {"coarse_delta": coarse_delta, "coarse_velocity": coarse_velocity}


def correlation_and_relative_l2(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float]:
    candidate_flat = np.asarray(candidate, dtype=np.float64).reshape(-1)
    reference_flat = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate_centre = candidate_flat - candidate_flat.mean()
    reference_centre = reference_flat - reference_flat.mean()
    denominator = math.sqrt(
        float(np.dot(candidate_centre, candidate_centre))
        * float(np.dot(reference_centre, reference_centre))
    )
    correlation = float(np.dot(candidate_centre, reference_centre) / denominator)
    relative_l2 = float(
        np.linalg.norm(candidate_flat - reference_flat)
        / max(np.linalg.norm(reference_flat), np.finfo(float).tiny)
    )
    return correlation, relative_l2


def run_task(
    program_path: str | Path,
    output_root: str | Path,
    task_index: int,
    implementation_commit: str,
) -> None:
    program, program_sha = load_program(program_path)
    if task_index < 0 or task_index >= 8:
        raise GeneratorGateError("generator task index outside 0-7")
    if len(implementation_commit) != 40:
        raise GeneratorGateError("implementation commit must be a full Git hash")
    assignment = program["mock_assignments"][task_index]
    seed = int(assignment["seed"])
    output = Path(output_root) / task_name(task_index, seed)
    staging = output.parent / f".{output.name}.staging"
    if output.exists() or staging.exists():
        raise GeneratorGateError("generator task output or staging already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    import jax

    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise GeneratorGateError("generator gate requires an allocated Slurm GPU")
    fine_white, _coarse_white, nesting = phasec_v5.nested_white_fields(
        seed,
        int(program["grid"]["inference_N"]),
        int(program["grid"]["truth_N"]),
        int(program["rng_tags"]["high_k_white"]),
    )
    candidate_summaries = []
    candidate_arrays = []
    for candidate in program["generator_gate"]["candidates"]:
        summary, arrays = build_candidate(fine_white, program, candidate)
        candidate_summaries.append(summary)
        candidate_arrays.append(arrays)

    comparison: dict[str, object] = {"available": False, "pass": False}
    if all(arrays is not None for arrays in candidate_arrays):
        convergence_arrays, production_arrays = candidate_arrays
        density_corr, density_l2 = correlation_and_relative_l2(
            convergence_arrays["coarse_delta"], production_arrays["coarse_delta"]
        )
        velocity_corr, velocity_l2 = correlation_and_relative_l2(
            convergence_arrays["coarse_velocity"], production_arrays["coarse_velocity"]
        )
        gates = program["generator_gate"]["time_convergence_gates_on_N32_fields"]
        comparison = {
            "available": True,
            "density_cross_correlation": density_corr,
            "density_relative_L2": density_l2,
            "velocity_cross_correlation": velocity_corr,
            "velocity_relative_L2": velocity_l2,
            "gates": gates,
            "pass": bool(
                density_corr >= float(gates["density_cross_correlation_min"])
                and density_l2 <= float(gates["density_relative_L2_max"])
                and velocity_corr >= float(gates["velocity_cross_correlation_min"])
                and velocity_l2 <= float(gates["velocity_relative_L2_max"])
            ),
        }
    task_pass = bool(all(row["pass"] for row in candidate_summaries) and comparison["pass"])
    result = {
        "schema": TASK_SCHEMA,
        "status": "PASS_REPLACEMENT_PHASE_C_GENERATOR_SEED" if task_pass else "FAIL_REPLACEMENT_PHASE_C_GENERATOR_SEED",
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
            "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
        },
        "nested_white": nesting,
        "candidates": candidate_summaries,
        "time_convergence": comparison,
        "pass": task_pass,
        "scope": {
            "mock_seed_only": True,
            "count_or_velocity_mock_generated": False,
            "optimizer_or_sampler_run": False,
            "actual_observational_datum_used": False,
            "validation_seed_used": False,
            "posterior_created": False,
            "target_0p3_cMpc_h_reached": False,
        },
    }
    staging.mkdir(mode=0o700)
    (staging / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = artifact_manifest(staging, "ouruniv-cf4-redesign-generator-task-manifest-v1")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete = {
        "schema": "ouruniv-cf4-redesign-generator-task-complete-v1",
        "result_sha256": sha256_file(staging / "result.json"),
        "manifest_sha256": sha256_file(staging / "manifest.json"),
        "pass": task_pass,
    }
    (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
    os.replace(staging, output)


def validate_task(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != TASK_FILES:
        raise GeneratorGateError("generator task artifact file set mismatch")
    result = json.loads((root / "result.json").read_text())
    if result.get("schema") != TASK_SCHEMA or not isinstance(result.get("pass"), bool):
        raise GeneratorGateError("generator task result schema mismatch")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("result_sha256") != sha256_file(root / "result.json"):
        raise GeneratorGateError("generator task result hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise GeneratorGateError("generator task manifest hash mismatch")
    if complete.get("pass") is not result["pass"]:
        raise GeneratorGateError("generator task pass marker mismatch")
    return result


def aggregate(
    program_path: str | Path,
    output_root: str | Path,
    aggregate_output: str | Path,
    implementation_commit: str,
) -> None:
    program, program_sha = load_program(program_path)
    output = Path(aggregate_output)
    staging = output.parent / f".{output.name}.staging"
    if output.exists() or staging.exists():
        raise GeneratorGateError("generator aggregate output or staging already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for assignment in program["mock_assignments"]:
        directory = Path(output_root) / task_name(int(assignment["index"]), int(assignment["seed"]))
        try:
            result = validate_task(directory)
            if result.get("assignment") != assignment:
                raise GeneratorGateError("generator task assignment mismatch")
            outcomes.append(
                {
                    "assignment": assignment,
                    "artifact_status": "VALID",
                    "pass": bool(result["pass"]),
                    "result_sha256": sha256_file(directory / "result.json"),
                    "candidates": result["candidates"],
                    "time_convergence": result["time_convergence"],
                }
            )
        except Exception as exc:
            outcomes.append(
                {
                    "assignment": assignment,
                    "artifact_status": "MISSING_OR_INVALID",
                    "pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    aggregate_pass = bool(len(outcomes) == 8 and all(row["pass"] for row in outcomes))
    result = {
        "schema": AGGREGATE_SCHEMA,
        "status": "PASS_REPLACEMENT_PHASE_C_ALL_SEED_GENERATOR_GATE" if aggregate_pass else "FAIL_REPLACEMENT_PHASE_C_ALL_SEED_GENERATOR_GATE_STOP_BEFORE_SAMPLER",
        "program": {"path": str(Path(program_path).resolve()), "sha256": program_sha},
        "implementation_commit": implementation_commit,
        "task_count": len(outcomes),
        "valid_artifact_count": int(sum(row["artifact_status"] == "VALID" for row in outcomes)),
        "passing_seed_count": int(sum(row["pass"] for row in outcomes)),
        "all_seed_gate_pass": aggregate_pass,
        "outcomes": outcomes,
        "decision": {
            "replacement_sampler_mechanics_pilot_allowed": aggregate_pass,
            "actual_observational_posterior_allowed": False,
            "validation_or_Phase_D_allowed": False,
        },
        "scope": {
            "actual_observational_datum_used": False,
            "posterior_created": False,
            "target_0p3_cMpc_h_reached": False,
        },
    }
    staging.mkdir(mode=0o700)
    (staging / "aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = artifact_manifest(staging, "ouruniv-cf4-redesign-generator-aggregate-manifest-v1")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete = {
        "schema": "ouruniv-cf4-redesign-generator-aggregate-complete-v1",
        "aggregate_sha256": sha256_file(staging / "aggregate.json"),
        "manifest_sha256": sha256_file(staging / "manifest.json"),
        "pass": aggregate_pass,
    }
    (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
    os.replace(staging, output)


def validate_aggregate(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != AGGREGATE_FILES:
        raise GeneratorGateError("generator aggregate artifact file set mismatch")
    result = json.loads((root / "aggregate.json").read_text())
    if result.get("schema") != AGGREGATE_SCHEMA or result.get("task_count") != 8:
        raise GeneratorGateError("generator aggregate result schema mismatch")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("aggregate_sha256") != sha256_file(root / "aggregate.json"):
        raise GeneratorGateError("generator aggregate result hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise GeneratorGateError("generator aggregate manifest hash mismatch")
    if complete.get("pass") is not result["all_seed_gate_pass"]:
        raise GeneratorGateError("generator aggregate pass marker mismatch")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--program", required=True)
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--task-index", required=True, type=int)
    run_parser.add_argument("--implementation-commit", required=True)
    task_parser = subparsers.add_parser("validate-task")
    task_parser.add_argument("--directory", required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--program", required=True)
    aggregate_parser.add_argument("--output-root", required=True)
    aggregate_parser.add_argument("--aggregate-output", required=True)
    aggregate_parser.add_argument("--implementation-commit", required=True)
    validate_parser = subparsers.add_parser("validate-aggregate")
    validate_parser.add_argument("--directory", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        run_task(args.program, args.output_root, args.task_index, args.implementation_commit)
    elif args.command == "validate-task":
        validate_task(args.directory)
    elif args.command == "aggregate":
        aggregate(args.program, args.output_root, args.aggregate_output, args.implementation_commit)
    elif args.command == "validate-aggregate":
        validate_aggregate(args.directory)


if __name__ == "__main__":
    main()
