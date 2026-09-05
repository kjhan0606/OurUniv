#!/usr/bin/env python3
"""Aggregate exact scan-gate repeats made serially on one Slurm GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

import cf4_phasec_scan_generator_gate_v2 as scan_gate


SCHEMA = "ouruniv-cf4-phasec-same-gpu-repeatability-diagnostic-v1"
DEVICE_SCHEMA = "ouruniv-cf4-phasec-same-gpu-device-record-v1"
AGGREGATE_SCHEMA = "ouruniv-cf4-phasec-same-gpu-repeatability-aggregate-v1"
AGGREGATE_FILES = {"aggregate.json", "manifest.json", "COMPLETE"}


class RepeatabilityDiagnosticError(ValueError):
    """The frozen same-GPU repeatability diagnostic contract was violated."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_assignments() -> list[dict[str, object]]:
    return [
        {"index": 1, "seed": 2026083001, "role": "scan_gate_failure"},
        {"index": 6, "seed": 2026083006, "role": "scan_gate_failure"},
        {"index": 2, "seed": 2026083002, "role": "scan_gate_pass_control"},
        {"index": 7, "seed": 2026083007, "role": "scan_gate_pass_control"},
    ]


def load_program(path: str | Path) -> tuple[dict[str, object], str]:
    payload = Path(path).read_bytes()
    try:
        program = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RepeatabilityDiagnosticError("cannot parse repeatability program") from exc
    if program.get("schema") != SCHEMA:
        raise RepeatabilityDiagnosticError("repeatability program schema mismatch")
    authorization = program.get("authorization", {})
    for key in (
        "Slurm_single_GPU_serial_exact_repeats",
        "GPFS_read_bound_mock_inputs",
        "GPFS_write_new_outputs_only",
    ):
        if authorization.get(key) is not True:
            raise RepeatabilityDiagnosticError(f"missing repeatability authorization: {key}")
    for key in (
        "sampler",
        "actual_observational_data",
        "validation_seed",
        "Phase_D_or_later",
    ):
        if authorization.get(key) is not False:
            raise RepeatabilityDiagnosticError(f"forbidden repeatability scope enabled: {key}")
    if program.get("assignments") != expected_assignments():
        raise RepeatabilityDiagnosticError("repeatability assignment changed")
    if program.get("repetitions") != [0, 1]:
        raise RepeatabilityDiagnosticError("repeatability count changed")
    execution = program.get("execution", {})
    if execution.get("fresh_process_per_seed_repeat") is not True:
        raise RepeatabilityDiagnosticError("fresh-process isolation disabled")
    if execution.get("concurrent_numerical_processes") != 1:
        raise RepeatabilityDiagnosticError("serial execution contract changed")
    for name, binding in program.get("lineage", {}).items():
        source = Path(str(binding.get("path", "")))
        expected_hash = str(binding.get("sha256", ""))
        if not source.is_file():
            raise RepeatabilityDiagnosticError(f"missing repeatability lineage: {name}")
        if expected_hash.startswith("TO_BE_FILLED"):
            raise RepeatabilityDiagnosticError("repeatability implementation is not frozen")
        if sha256_file(source) != expected_hash:
            raise RepeatabilityDiagnosticError(f"repeatability lineage hash mismatch: {name}")
    return program, hashlib.sha256(payload).hexdigest()


def artifact_manifest(directory: Path, schema: str) -> dict[str, object]:
    rows = []
    for path in sorted(directory.iterdir()):
        if path.name in {"manifest.json", "COMPLETE"}:
            continue
        rows.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": schema, "files": rows}


def capture_device(program_path: str | Path, output_root: str | Path) -> None:
    program, program_sha = load_program(program_path)
    root = Path(output_root)
    if not root.is_dir():
        raise RepeatabilityDiagnosticError("output root must be created by the Slurm runner")
    output = root / "device.json"
    if output.exists():
        raise RepeatabilityDiagnosticError("device record already exists")

    import jax
    import jaxlib

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RepeatabilityDiagnosticError("diagnostic requires exactly one visible JAX GPU")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    record = {
        "schema": DEVICE_SCHEMA,
        "program_sha256": program_sha,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", ""),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS", ""),
            "node_list": os.environ.get("SLURM_JOB_NODELIST", ""),
            "submit_host": os.environ.get("SLURM_SUBMIT_HOST", ""),
        },
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi_inventory": inventory,
        "jax": {
            "version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "device_kinds": [device.device_kind for device in jax.devices()],
            "enable_x64": bool(jax.config.x64_enabled),
        },
        "execution_contract": program["execution"],
    }
    with output.open("x") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def science_signature(result: Mapping[str, object]) -> dict[str, object]:
    return {
        "assignment": result.get("assignment"),
        "nested_white": result.get("nested_white"),
        "scan_candidates": result.get("scan_candidates"),
        "time_convergence": result.get("time_convergence"),
        "pass": result.get("pass"),
    }


def summarize_seed_repeats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(rows) != 2:
        raise RepeatabilityDiagnosticError("each seed requires exactly two repeats")
    valid = all(row.get("artifact_status") == "VALID" for row in rows)
    signatures = [row.get("science_signature") for row in rows]
    exact_science_repeat = bool(valid and signatures[0] == signatures[1])
    result_hash_repeat = bool(
        valid and rows[0].get("result_sha256") == rows[1].get("result_sha256")
    )
    all_pass = bool(valid and all(row.get("pass") is True for row in rows))
    if not exact_science_repeat:
        classification = "NONREPRODUCIBLE_SAME_GPU"
    elif all_pass:
        classification = "REPEATABLE_ALL_PASS"
    else:
        classification = "REPEATABLE_FAILURE"
    return {
        "valid_artifacts": valid,
        "exact_science_repeat": exact_science_repeat,
        "exact_result_hash_repeat": result_hash_repeat,
        "all_repeats_pass": all_pass,
        "classification": classification,
    }


def aggregate(
    program_path: str | Path,
    output_root: str | Path,
    aggregate_output: str | Path,
    implementation_commit: str,
) -> None:
    program, program_sha = load_program(program_path)
    if len(implementation_commit) != 40:
        raise RepeatabilityDiagnosticError("implementation commit must be a full Git hash")
    root = Path(output_root)
    device_path = root / "device.json"
    if not device_path.is_file():
        raise RepeatabilityDiagnosticError("missing same-GPU device record")
    device = json.loads(device_path.read_text())
    if device.get("schema") != DEVICE_SCHEMA or device.get("program_sha256") != program_sha:
        raise RepeatabilityDiagnosticError("same-GPU device record mismatch")

    output = Path(aggregate_output)
    staging = output.parent / f".{output.name}.staging"
    if output.exists() or staging.exists():
        raise RepeatabilityDiagnosticError("repeatability aggregate output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    outcomes = []
    by_seed: dict[int, list[dict[str, object]]] = {}
    for repeat in program["repetitions"]:
        for assignment in program["assignments"]:
            index = int(assignment["index"])
            seed = int(assignment["seed"])
            task_dir = root / f"repeat_{repeat}" / scan_gate.task_name(index, seed)
            try:
                result = scan_gate.validate_task(task_dir)
                expected_assignment = {
                    "index": index,
                    "seed": seed,
                    "arm": "ABCD"[index // 2],
                }
                if result.get("assignment") != expected_assignment:
                    raise RepeatabilityDiagnosticError("repeated task assignment mismatch")
                row = {
                    "repeat": repeat,
                    "assignment": assignment,
                    "artifact_status": "VALID",
                    "pass": bool(result["pass"]),
                    "result_sha256": sha256_file(task_dir / "result.json"),
                    "science_signature": science_signature(result),
                }
            except Exception as exc:
                row = {
                    "repeat": repeat,
                    "assignment": assignment,
                    "artifact_status": "MISSING_OR_INVALID",
                    "pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            outcomes.append(row)
            by_seed.setdefault(seed, []).append(row)

    seed_summaries = []
    for assignment in program["assignments"]:
        seed = int(assignment["seed"])
        summary = summarize_seed_repeats(by_seed[seed])
        seed_summaries.append({"assignment": assignment, **summary})
    valid_count = sum(row["artifact_status"] == "VALID" for row in outcomes)
    same_gpu_repeatability_pass = bool(
        valid_count == len(program["assignments"]) * len(program["repetitions"])
        and all(row["exact_science_repeat"] for row in seed_summaries)
    )
    all_repeats_pass = bool(all(row["all_repeats_pass"] for row in seed_summaries))
    if not same_gpu_repeatability_pass:
        status = "FAIL_NONREPRODUCIBLE_ON_ONE_GPU_STOP_BEFORE_SAMPLER"
    elif not all_repeats_pass:
        status = "PASS_REPEATABILITY_WITH_STABLE_NUMERICAL_FAILURE_STOP_BEFORE_SAMPLER"
    else:
        status = "PASS_REPEATABILITY_AND_ALL_RUNS_FINITE_REAUDIT_GENERATOR_BEFORE_SAMPLER"
    result = {
        "schema": AGGREGATE_SCHEMA,
        "status": status,
        "program": {"path": str(Path(program_path).resolve()), "sha256": program_sha},
        "implementation_commit": implementation_commit,
        "device": device,
        "valid_artifact_count": valid_count,
        "expected_artifact_count": len(program["assignments"]) * len(program["repetitions"]),
        "same_gpu_repeatability_pass": same_gpu_repeatability_pass,
        "all_repeats_pass": all_repeats_pass,
        "outcomes": outcomes,
        "seed_summaries": seed_summaries,
        "decision": {
            "sampler_allowed": False,
            "actual_observational_posterior_allowed": False,
            "validation_or_Phase_D_allowed": False,
            "next_step": (
                "audit the first differing LPT or PM stage"
                if not same_gpu_repeatability_pass
                else "isolate the stable numerical failure"
                if not all_repeats_pass
                else "repeat the all-eight generator gate on the bound physical GPU"
            ),
        },
        "scope_firewall": program["scope_firewall"],
    }
    staging.mkdir(mode=0o700)
    (staging / "aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = artifact_manifest(staging, "ouruniv-cf4-phasec-same-gpu-repeatability-manifest-v1")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete = {
        "schema": "ouruniv-cf4-phasec-same-gpu-repeatability-complete-v1",
        "aggregate_sha256": sha256_file(staging / "aggregate.json"),
        "manifest_sha256": sha256_file(staging / "manifest.json"),
        "same_gpu_repeatability_pass": same_gpu_repeatability_pass,
        "all_repeats_pass": all_repeats_pass,
    }
    (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
    os.replace(staging, output)


def validate_aggregate(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != AGGREGATE_FILES:
        raise RepeatabilityDiagnosticError("repeatability aggregate artifact set mismatch")
    result = json.loads((root / "aggregate.json").read_text())
    if result.get("schema") != AGGREGATE_SCHEMA:
        raise RepeatabilityDiagnosticError("repeatability aggregate schema mismatch")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("aggregate_sha256") != sha256_file(root / "aggregate.json"):
        raise RepeatabilityDiagnosticError("repeatability aggregate hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise RepeatabilityDiagnosticError("repeatability manifest hash mismatch")
    if complete.get("same_gpu_repeatability_pass") != result.get("same_gpu_repeatability_pass"):
        raise RepeatabilityDiagnosticError("repeatability decision marker mismatch")
    if complete.get("all_repeats_pass") != result.get("all_repeats_pass"):
        raise RepeatabilityDiagnosticError("all-pass decision marker mismatch")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture-device")
    capture.add_argument("--program", required=True)
    capture.add_argument("--output-root", required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--program", required=True)
    aggregate_parser.add_argument("--output-root", required=True)
    aggregate_parser.add_argument("--aggregate-output", required=True)
    aggregate_parser.add_argument("--implementation-commit", required=True)
    validate = subparsers.add_parser("validate-aggregate")
    validate.add_argument("--directory", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "capture-device":
        capture_device(args.program, args.output_root)
    elif args.command == "aggregate":
        aggregate(args.program, args.output_root, args.aggregate_output, args.implementation_commit)
    elif args.command == "validate-aggregate":
        validate_aggregate(args.directory)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
