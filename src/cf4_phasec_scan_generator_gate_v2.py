#!/usr/bin/env python3
"""All-eight-seed PMWD generator gate using the admitted scan-loop repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

import cf4_datum_bearing_z0_phasec_pilot as phasec_v5
import cf4_phasec_pmwd_scan_diagnostic as scan_diagnostic
import cf4_phasec_redesign_generator_gate as generator_v1


SCHEMA = "ouruniv-cf4-phasec-scan-generator-gate-v2"
TASK_SCHEMA = "ouruniv-cf4-phasec-scan-generator-task-v2"
AGGREGATE_SCHEMA = "ouruniv-cf4-phasec-scan-generator-aggregate-v2"
TASK_FILES = {"result.json", "manifest.json", "COMPLETE"}
AGGREGATE_FILES = {"aggregate.json", "manifest.json", "COMPLETE"}


class ScanGeneratorError(ValueError):
    """The frozen all-eight scan generator contract was violated."""


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
        raise ScanGeneratorError("cannot parse scan generator program") from exc
    if program.get("schema") != SCHEMA:
        raise ScanGeneratorError("scan generator schema mismatch")
    authorization = program.get("authorization", {})
    for key in ("Slurm_GPU_array", "Slurm_CPU_aggregate", "GPFS_write_new_outputs_only"):
        if authorization.get(key) is not True:
            raise ScanGeneratorError(f"missing scan generator authorization: {key}")
    for key in ("sampler", "actual_observational_data", "validation_seed", "Phase_D_or_later"):
        if authorization.get(key) is not False:
            raise ScanGeneratorError(f"forbidden scan generator scope enabled: {key}")
    if program.get("assignments") != expected_assignments():
        raise ScanGeneratorError("all-eight assignment changed")
    integrator = program.get("integrator", {})
    if integrator.get("a_nbody_maxsteps") != [1 / 128, 1 / 256]:
        raise ScanGeneratorError("scan generator integration ladder changed")
    if integrator.get("production_a_nbody_maxstep") != 1 / 256:
        raise ScanGeneratorError("scan generator production integration changed")
    if program.get("high_k_white_tag") != 843927701:
        raise ScanGeneratorError("scan generator high-k RNG tag changed")
    for name, binding in program.get("lineage", {}).items():
        source = Path(str(binding.get("path", "")))
        expected_hash = str(binding.get("sha256", ""))
        if not source.is_file():
            raise ScanGeneratorError(f"missing scan generator lineage: {name}")
        if expected_hash.startswith("TO_BE_FILLED"):
            raise ScanGeneratorError("scan generator implementation is not frozen")
        if sha256_file(source) != expected_hash:
            raise ScanGeneratorError(f"scan generator lineage hash mismatch: {name}")
    return program, hashlib.sha256(payload).hexdigest()


def task_name(index: int, seed: int) -> str:
    return f"seed_{index:02d}_{seed}"


def artifact_manifest(directory: Path, schema: str) -> dict[str, object]:
    rows = []
    for path in sorted(directory.iterdir()):
        if path.name in {"manifest.json", "COMPLETE"}:
            continue
        rows.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": schema, "files": rows}


def run_task(
    program_path: str | Path,
    output_root: str | Path,
    task_index: int,
    implementation_commit: str,
) -> None:
    program, program_sha = load_program(program_path)
    if task_index < 0 or task_index >= 8:
        raise ScanGeneratorError("scan generator task index outside 0-7")
    if len(implementation_commit) != 40:
        raise ScanGeneratorError("implementation commit must be a full Git hash")
    assignment = program["assignments"][task_index]
    seed = int(assignment["seed"])
    output = Path(output_root) / task_name(task_index, seed)
    staging = output.parent / f".{output.name}.staging"
    if output.exists() or staging.exists():
        raise ScanGeneratorError("scan generator output or staging already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    import jax

    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise ScanGeneratorError("scan generator requires an allocated Slurm GPU")
    fine_white, _coarse_white, nesting = phasec_v5.nested_white_fields(
        seed,
        int(program["grid"]["inference_N"]),
        int(program["grid"]["truth_N"]),
        int(program["high_k_white_tag"]),
    )
    summaries = []
    arrays_by_step = {}
    for maxstep in program["integrator"]["a_nbody_maxsteps"]:
        summary, arrays = scan_diagnostic.build_scan_candidate(fine_white, program, float(maxstep))
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
        gates = program["gates"]
        comparison = {
            "available": True,
            "density_cross_correlation": density_corr,
            "density_relative_L2": density_l2,
            "velocity_cross_correlation": velocity_corr,
            "velocity_relative_L2": velocity_l2,
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
        "status": "PASS_SCAN_GENERATOR_SEED" if task_pass else "FAIL_SCAN_GENERATOR_SEED",
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
    manifest = artifact_manifest(staging, "ouruniv-cf4-scan-generator-task-manifest-v2")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete = {
        "schema": "ouruniv-cf4-scan-generator-task-complete-v2",
        "result_sha256": sha256_file(staging / "result.json"),
        "manifest_sha256": sha256_file(staging / "manifest.json"),
        "pass": task_pass,
    }
    (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
    os.replace(staging, output)


def validate_task(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != TASK_FILES:
        raise ScanGeneratorError("scan generator task artifact file set mismatch")
    result = json.loads((root / "result.json").read_text())
    if result.get("schema") != TASK_SCHEMA or not isinstance(result.get("pass"), bool):
        raise ScanGeneratorError("scan generator task result schema mismatch")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("result_sha256") != sha256_file(root / "result.json"):
        raise ScanGeneratorError("scan generator task result hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise ScanGeneratorError("scan generator task manifest hash mismatch")
    if complete.get("pass") != result["pass"]:
        raise ScanGeneratorError("scan generator task pass marker mismatch")
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
        raise ScanGeneratorError("scan generator aggregate output or staging already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for assignment in program["assignments"]:
        directory = Path(output_root) / task_name(int(assignment["index"]), int(assignment["seed"]))
        try:
            result = validate_task(directory)
            if result.get("assignment") != assignment:
                raise ScanGeneratorError("scan generator assignment mismatch")
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
    aggregate_pass = bool(len(outcomes) == 8 and all(row["pass"] for row in outcomes))
    result = {
        "schema": AGGREGATE_SCHEMA,
        "status": "PASS_ALL_EIGHT_SCAN_GENERATOR_GATE" if aggregate_pass else "FAIL_ALL_EIGHT_SCAN_GENERATOR_GATE_STOP_BEFORE_SAMPLER",
        "program": {"path": str(Path(program_path).resolve()), "sha256": program_sha},
        "implementation_commit": implementation_commit,
        "task_count": len(outcomes),
        "valid_artifact_count": int(sum(row["artifact_status"] == "VALID" for row in outcomes)),
        "passing_seed_count": int(sum(row["pass"] for row in outcomes)),
        "all_seed_gate_pass": aggregate_pass,
        "outcomes": outcomes,
        "decision": {
            "sampler_mechanics_pilot_indices_0_and_6_allowed": aggregate_pass,
            "actual_observational_posterior_allowed": False,
            "validation_or_Phase_D_allowed": False,
        },
        "scope_firewall": program["scope_firewall"],
    }
    staging.mkdir(mode=0o700)
    (staging / "aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = artifact_manifest(staging, "ouruniv-cf4-scan-generator-aggregate-manifest-v2")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete = {
        "schema": "ouruniv-cf4-scan-generator-aggregate-complete-v2",
        "aggregate_sha256": sha256_file(staging / "aggregate.json"),
        "manifest_sha256": sha256_file(staging / "manifest.json"),
        "pass": aggregate_pass,
    }
    (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
    os.replace(staging, output)


def validate_aggregate(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != AGGREGATE_FILES:
        raise ScanGeneratorError("scan generator aggregate artifact file set mismatch")
    result = json.loads((root / "aggregate.json").read_text())
    if result.get("schema") != AGGREGATE_SCHEMA or result.get("task_count") != 8:
        raise ScanGeneratorError("scan generator aggregate result schema mismatch")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("aggregate_sha256") != sha256_file(root / "aggregate.json"):
        raise ScanGeneratorError("scan generator aggregate result hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise ScanGeneratorError("scan generator aggregate manifest hash mismatch")
    if complete.get("pass") != result["all_seed_gate_pass"]:
        raise ScanGeneratorError("scan generator aggregate pass marker mismatch")
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
