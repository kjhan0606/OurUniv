#!/usr/bin/env python3
"""Step-localized PMWD scan-loop diagnostic for two frozen mock seeds."""

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
import cf4_phasec_redesign_generator_gate as generator_v1


SCHEMA = "ouruniv-cf4-phasec-pmwd-scan-diagnostic-v1"
TASK_SCHEMA = "ouruniv-cf4-phasec-pmwd-scan-diagnostic-task-v1"
AGGREGATE_SCHEMA = "ouruniv-cf4-phasec-pmwd-scan-diagnostic-aggregate-v1"
TASK_FILES = {"result.json", "manifest.json", "COMPLETE"}
AGGREGATE_FILES = {"aggregate.json", "manifest.json", "COMPLETE"}
COMPONENT_NAMES = ("displacement", "velocity", "acceleration")


class ScanDiagnosticError(ValueError):
    """The frozen scan diagnostic contract was violated."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_program(path: str | Path) -> tuple[dict[str, object], str]:
    payload = Path(path).read_bytes()
    try:
        program = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ScanDiagnosticError("cannot parse PMWD scan diagnostic") from exc
    if program.get("schema") != SCHEMA:
        raise ScanDiagnosticError("PMWD scan diagnostic schema mismatch")
    authorization = program.get("authorization", {})
    for key in ("Slurm_GPU_diagnostic_array", "Slurm_CPU_aggregate", "GPFS_write_new_outputs_only"):
        if authorization.get(key) is not True:
            raise ScanDiagnosticError(f"missing scan diagnostic authorization: {key}")
    for key in ("sampler", "actual_observational_data", "validation_seed", "Phase_D_or_later"):
        if authorization.get(key) is not False:
            raise ScanDiagnosticError(f"forbidden scan diagnostic scope enabled: {key}")
    expected_assignments = [
        {
            "index": 0,
            "seed": 2026083002,
            "prior_monolithic_outcomes": {
                "1_over_64": "finite",
                "1_over_128": "nonfinite",
                "1_over_256": "nonfinite",
            },
        },
        {
            "index": 1,
            "seed": 2026083007,
            "prior_monolithic_outcomes": {
                "1_over_64": "nonfinite",
                "1_over_128": "finite",
                "1_over_256": "finite",
            },
        },
    ]
    if program.get("assignments") != expected_assignments:
        raise ScanDiagnosticError("diagnostic seed assignment changed")
    if program.get("integrator", {}).get("a_nbody_maxsteps") != [1 / 64, 1 / 128, 1 / 256]:
        raise ScanDiagnosticError("diagnostic integration ladder changed")
    if program.get("high_k_white_tag") != 843927701:
        raise ScanDiagnosticError("diagnostic high-k RNG tag changed")
    for name, binding in program.get("lineage", {}).items():
        source = Path(str(binding.get("path", "")))
        expected_hash = str(binding.get("sha256", ""))
        if not source.is_file():
            raise ScanDiagnosticError(f"missing lineage file: {name}")
        if expected_hash.startswith("TO_BE_FILLED"):
            raise ScanDiagnosticError("diagnostic implementation binding is not frozen")
        if sha256_file(source) != expected_hash:
            raise ScanDiagnosticError(f"lineage hash mismatch: {name}")
    return program, hashlib.sha256(payload).hexdigest()


def artifact_manifest(directory: Path, schema: str) -> dict[str, object]:
    rows = []
    for path in sorted(directory.iterdir()):
        if path.name in {"manifest.json", "COMPLETE"}:
            continue
        rows.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": schema, "files": rows}


def task_name(index: int, seed: int) -> str:
    return f"seed_{index:02d}_{seed}"


def finite_range(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    return float(finite.min()), float(finite.max())


def particle_metric_rows(metrics: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for component, values in zip(COMPONENT_NAMES, np.asarray(metrics), strict=True):
        rows.append(
            {
                "component": component,
                "all_finite": bool(values[0]),
                "nonfinite_count": int(values[1]),
                "finite_max_abs": float(values[2]),
            }
        )
    return rows


def build_scan_candidate(
    fine_white: np.ndarray,
    program: Mapping[str, object],
    a_nbody_maxstep: float,
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    import jax
    import jax.numpy as jnp
    from pmwd import Configuration, SimpleLCDM, boltzmann, linear_modes, lpt, scatter
    from pmwd.nbody import nbody_init, nbody_step

    grid = program["grid"]
    integrator = program["integrator"]
    cosmology = program["cosmology"]
    n = int(grid["truth_N"])
    coarse_n = int(grid["inference_N"])
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
        a_nbody_maxstep=float(a_nbody_maxstep),
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

    def metrics3(particles):
        rows = []
        for values in (particles.disp, particles.vel, particles.acc):
            finite = jnp.isfinite(values)
            rows.append(
                jnp.stack(
                    (
                        jnp.all(finite).astype(jnp.float64),
                        jnp.count_nonzero(~finite).astype(jnp.float64),
                        jnp.max(jnp.where(finite, jnp.abs(values), 0.0)),
                    )
                )
            )
        return jnp.stack(rows)

    def metrics_lpt(particles):
        zero_acceleration = particles.replace(acc=jnp.zeros_like(particles.disp))
        return metrics3(zero_acceleration)

    @jax.jit
    def forward(white):
        modes = linear_modes(white, cosmo, conf)
        particles, observables = lpt(modes, cosmo, conf)
        lpt_metrics = metrics_lpt(particles)
        particles, observables = nbody_init(conf.a_nbody[0], particles, observables, cosmo, conf)
        init_metrics = metrics3(particles)

        def scan_step(carry, scale_factors):
            current_particles, current_observables = carry
            next_particles, next_observables = nbody_step(
                scale_factors[0],
                scale_factors[1],
                current_particles,
                current_observables,
                cosmo,
                conf,
            )
            return (next_particles, next_observables), metrics3(next_particles)

        scale_factor_pairs = jnp.stack((conf.a_nbody[:-1], conf.a_nbody[1:]), axis=1)
        (particles, observables), step_metrics = jax.lax.scan(
            scan_step,
            (particles, observables),
            scale_factor_pairs,
        )
        del observables
        density = scatter(particles, conf)
        momentum = scatter(particles, conf, val=particles.vel * 100.0)
        return density, momentum, lpt_metrics, init_metrics, step_metrics

    try:
        density_j, momentum_j, lpt_j, init_j, history_j = forward(
            jnp.asarray(fine_white, dtype=jnp.float64)
        )
        density = np.asarray(density_j, dtype=np.float64)
        momentum = np.asarray(momentum_j, dtype=np.float64)
        lpt_metrics = np.asarray(lpt_j, dtype=np.float64)
        init_metrics = np.asarray(init_j, dtype=np.float64)
        step_metrics = np.asarray(history_j, dtype=np.float64)
    except Exception as exc:
        return {
            "a_nbody_maxstep": float(a_nbody_maxstep),
            "a_nbody_step_count": int(conf.a_nbody_num),
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "pass": False,
        }, None

    step_rows = []
    first_nonfinite = None
    for step_index, metrics in enumerate(step_metrics):
        components = particle_metric_rows(metrics)
        for component in components:
            if not component["all_finite"] and first_nonfinite is None:
                first_nonfinite = {
                    "step_index_one_based": step_index + 1,
                    "a_end": float(conf.a_nbody[step_index + 1]),
                    "component": component["component"],
                    "nonfinite_count": component["nonfinite_count"],
                }
        step_rows.append(
            {
                "step_index_one_based": step_index + 1,
                "a_end": float(conf.a_nbody[step_index + 1]),
                "components": components,
            }
        )
    density_finite = bool(np.all(np.isfinite(density)))
    momentum_finite = bool(np.all(np.isfinite(momentum)))
    density_min, density_max = finite_range(density)
    momentum_min, momentum_max = finite_range(momentum)
    density_mean = float(density.mean()) if density_finite else None
    mean_error = abs(density_mean - 1.0) if density_mean is not None else None
    final_pass = bool(
        all(row["all_finite"] for row in particle_metric_rows(lpt_metrics))
        and all(row["all_finite"] for row in particle_metric_rows(init_metrics))
        and first_nonfinite is None
        and density_finite
        and momentum_finite
        and density_min is not None
        and density_min >= 0.0
        and mean_error is not None
        and mean_error <= 2e-12
    )
    summary: dict[str, object] = {
        "a_nbody_maxstep": float(a_nbody_maxstep),
        "a_nbody_actual_step": float(conf.a_nbody_step),
        "a_nbody_step_count": int(conf.a_nbody_num),
        "LPT_components": particle_metric_rows(lpt_metrics),
        "nbody_init_components": particle_metric_rows(init_metrics),
        "first_nonfinite": first_nonfinite,
        "step_history": step_rows,
        "density_all_finite": density_finite,
        "momentum_all_finite": momentum_finite,
        "density_mean": density_mean,
        "density_mean_absolute_error": mean_error,
        "density_min": density_min,
        "density_max": density_max,
        "momentum_min": momentum_min,
        "momentum_max": momentum_max,
        "pass": final_pass,
    }
    if not final_pass:
        return summary, None
    coarse_mass = phasec_v5.block_sum(density, coarse_n)
    coarse_momentum = phasec_v5.block_sum(momentum, coarse_n)
    coarse_velocity = np.divide(
        coarse_momentum,
        coarse_mass[..., None],
        out=np.zeros_like(coarse_momentum),
        where=coarse_mass[..., None] > 1e-10,
    )
    ratio = n // coarse_n
    return summary, {
        "coarse_delta": coarse_mass / ratio**3 - 1.0,
        "coarse_velocity": coarse_velocity,
    }


def run_task(
    program_path: str | Path,
    output_root: str | Path,
    task_index: int,
    implementation_commit: str,
) -> None:
    program, program_sha = load_program(program_path)
    if task_index < 0 or task_index >= 2:
        raise ScanDiagnosticError("scan diagnostic task index outside 0-1")
    if len(implementation_commit) != 40:
        raise ScanDiagnosticError("implementation commit must be a full Git hash")
    assignment = program["assignments"][task_index]
    seed = int(assignment["seed"])
    output = Path(output_root) / task_name(task_index, seed)
    staging = output.parent / f".{output.name}.staging"
    if output.exists() or staging.exists():
        raise ScanDiagnosticError("scan diagnostic output or staging already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    import jax

    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise ScanDiagnosticError("scan diagnostic requires an allocated Slurm GPU")
    fine_white, _coarse_white, nesting = phasec_v5.nested_white_fields(
        seed,
        int(program["grid"]["inference_N"]),
        int(program["grid"]["truth_N"]),
        int(program["high_k_white_tag"]),
    )
    summaries = []
    arrays_by_step: dict[float, dict[str, np.ndarray] | None] = {}
    for maxstep in program["integrator"]["a_nbody_maxsteps"]:
        summary, arrays = build_scan_candidate(fine_white, program, float(maxstep))
        summaries.append(summary)
        arrays_by_step[float(maxstep)] = arrays
    comparison: dict[str, object] = {"available": False, "pass": False}
    convergence_arrays = arrays_by_step[1 / 128]
    production_arrays = arrays_by_step[1 / 256]
    if convergence_arrays is not None and production_arrays is not None:
        density_corr, density_l2 = generator_v1.correlation_and_relative_l2(
            convergence_arrays["coarse_delta"], production_arrays["coarse_delta"]
        )
        velocity_corr, velocity_l2 = generator_v1.correlation_and_relative_l2(
            convergence_arrays["coarse_velocity"], production_arrays["coarse_velocity"]
        )
        gates = program["diagnostics"]["comparison_gates"]
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
    task_pass = bool(all(row["pass"] for row in summaries) and comparison["pass"])
    result = {
        "schema": TASK_SCHEMA,
        "status": "PASS_PMWD_SCAN_DIAGNOSTIC_SEED" if task_pass else "FAIL_PMWD_SCAN_DIAGNOSTIC_SEED",
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
        "scan_candidates": summaries,
        "time_convergence": comparison,
        "pass": task_pass,
        "scope_firewall": program["scope_firewall"],
    }
    staging.mkdir(mode=0o700)
    (staging / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = artifact_manifest(staging, "ouruniv-cf4-pmwd-scan-task-manifest-v1")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete = {
        "schema": "ouruniv-cf4-pmwd-scan-task-complete-v1",
        "result_sha256": sha256_file(staging / "result.json"),
        "manifest_sha256": sha256_file(staging / "manifest.json"),
        "pass": task_pass,
    }
    (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
    os.replace(staging, output)


def validate_task(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != TASK_FILES:
        raise ScanDiagnosticError("scan task artifact file set mismatch")
    result = json.loads((root / "result.json").read_text())
    if result.get("schema") != TASK_SCHEMA or not isinstance(result.get("pass"), bool):
        raise ScanDiagnosticError("scan task result schema mismatch")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("result_sha256") != sha256_file(root / "result.json"):
        raise ScanDiagnosticError("scan task result hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise ScanDiagnosticError("scan task manifest hash mismatch")
    if complete.get("pass") != result["pass"]:
        raise ScanDiagnosticError("scan task pass marker mismatch")
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
        raise ScanDiagnosticError("scan aggregate output or staging already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for assignment in program["assignments"]:
        directory = Path(output_root) / task_name(int(assignment["index"]), int(assignment["seed"]))
        try:
            result = validate_task(directory)
            if result.get("assignment") != assignment:
                raise ScanDiagnosticError("scan task assignment mismatch")
            outcomes.append(
                {
                    "assignment": assignment,
                    "artifact_status": "VALID",
                    "pass": bool(result["pass"]),
                    "result_sha256": sha256_file(directory / "result.json"),
                    "scan_candidates": result["scan_candidates"],
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
    aggregate_pass = bool(len(outcomes) == 2 and all(row["pass"] for row in outcomes))
    result = {
        "schema": AGGREGATE_SCHEMA,
        "status": "PASS_PMWD_SCAN_LOOP_REPAIR_DIAGNOSTIC" if aggregate_pass else "FAIL_PMWD_SCAN_LOOP_REPAIR_DIAGNOSTIC",
        "program": {"path": str(Path(program_path).resolve()), "sha256": program_sha},
        "implementation_commit": implementation_commit,
        "task_count": len(outcomes),
        "valid_artifact_count": int(sum(row["artifact_status"] == "VALID" for row in outcomes)),
        "all_diagnostic_gates_pass": aggregate_pass,
        "outcomes": outcomes,
        "decision": {
            "scan_loop_all_eight_generator_gate_allowed": aggregate_pass,
            "sampler_allowed": False,
            "actual_observational_posterior_allowed": False,
        },
        "scope_firewall": program["scope_firewall"],
    }
    staging.mkdir(mode=0o700)
    (staging / "aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = artifact_manifest(staging, "ouruniv-cf4-pmwd-scan-aggregate-manifest-v1")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete = {
        "schema": "ouruniv-cf4-pmwd-scan-aggregate-complete-v1",
        "aggregate_sha256": sha256_file(staging / "aggregate.json"),
        "manifest_sha256": sha256_file(staging / "manifest.json"),
        "pass": aggregate_pass,
    }
    (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
    os.replace(staging, output)


def validate_aggregate(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != AGGREGATE_FILES:
        raise ScanDiagnosticError("scan aggregate artifact file set mismatch")
    result = json.loads((root / "aggregate.json").read_text())
    if result.get("schema") != AGGREGATE_SCHEMA or result.get("task_count") != 2:
        raise ScanDiagnosticError("scan aggregate result schema mismatch")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("aggregate_sha256") != sha256_file(root / "aggregate.json"):
        raise ScanDiagnosticError("scan aggregate result hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise ScanDiagnosticError("scan aggregate manifest hash mismatch")
    if complete.get("pass") != result["all_diagnostic_gates_pass"]:
        raise ScanDiagnosticError("scan aggregate pass marker mismatch")
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
