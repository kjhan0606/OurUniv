#!/usr/bin/env python3
"""Independent offline ACF audit of the completed mock sampler-mechanics pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCHEMA = "ouruniv-cf4-phasec-sampler-mechanics-acf-audit-v1"
RESULT_SCHEMA = "ouruniv-cf4-phasec-sampler-mechanics-task-v1"
AGGREGATE_SCHEMA = "ouruniv-cf4-phasec-sampler-mechanics-aggregate-v1"
TASK_NAMES = {
    "task_0": "mechanics_00_mock_00_seed_2026083000_arm_A",
    "task_1": "mechanics_01_mock_06_seed_2026083006_arm_D",
}
NUISANCE_NAMES = (
    [f"alpha_unit_{i}" for i in range(6)]
    + [f"logbias_unit_{i}" for i in range(6)]
    + [f"logFoG_unit_{i}" for i in range(6)]
    + ["selection_unit_radial", "selection_unit_angular"]
    + [f"velocity_q_unit_{i}" for i in range(4)]
)


class ACFAuditError(ValueError):
    """The frozen offline ACF audit contract was violated."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_program(path: str | Path) -> tuple[dict[str, object], str]:
    source = Path(path)
    payload = source.read_bytes()
    try:
        program = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ACFAuditError("cannot parse ACF audit program") from exc
    if program.get("schema") != SCHEMA:
        raise ACFAuditError("ACF audit schema mismatch")
    authorization = program.get("authorization", {})
    if authorization.get("offline_existing_draws_only") is not True:
        raise ACFAuditError("offline existing-draw authorization is absent")
    for key in (
        "new_GPU_calculation",
        "actual_observational_field_inference",
        "actual_2Mpp_count_read",
        "actual_CF4_velocity_datum_used",
        "validation_or_Phase_D",
    ):
        if authorization.get(key) is not False:
            raise ACFAuditError(f"forbidden ACF scope enabled: {key}")
    if program.get("task_keys") != ["task_0", "task_1"]:
        raise ACFAuditError("ACF task set changed")
    if program.get("analysis", {}).get("maximum_lag") != 64:
        raise ACFAuditError("ACF maximum lag changed")
    for name, binding in program.get("lineage", {}).items():
        path = Path(str(binding.get("path", "")))
        if not path.is_file() or sha256_file(path) != binding.get("sha256"):
            raise ACFAuditError(f"ACF lineage mismatch: {name}")
    return program, hashlib.sha256(payload).hexdigest()


def autocorrelation(values: np.ndarray, maximum_lag: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < maximum_lag + 2:
        raise ACFAuditError("ACF input has insufficient draws")
    if not np.all(np.isfinite(values)):
        raise ACFAuditError("ACF input is nonfinite")
    centered = values - values.mean()
    n = centered.size
    variance = float(np.dot(centered, centered) / n)
    if variance <= 0.0:
        return np.ones(maximum_lag + 1, dtype=np.float64)
    size = 1 << (2 * n - 1).bit_length()
    transformed = np.fft.rfft(centered, n=size)
    covariance = np.fft.irfft(transformed * np.conjugate(transformed), n=size)[: n + 0]
    covariance = covariance / np.arange(n, 0, -1)
    return covariance[: maximum_lag + 1] / covariance[0]


def integrated_ess(chains: np.ndarray) -> float:
    """Geyer initial-positive-sequence ESS, independently implemented."""

    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2:
        raise ACFAuditError("ESS chain shape mismatch")
    m, n = chains.shape
    chain_variance = np.var(chains, axis=1, ddof=1)
    within = float(np.mean(chain_variance))
    between = float(n * np.var(chains.mean(axis=1), ddof=1))
    variance_plus = (n - 1.0) / n * within + between / n
    if variance_plus <= 0.0:
        return float(m * n)
    autocovariance = []
    for chain in chains:
        centered = chain - chain.mean()
        size = 1 << (2 * n - 1).bit_length()
        transformed = np.fft.rfft(centered, n=size)
        acov = np.fft.irfft(transformed * np.conjugate(transformed), n=size)[:n]
        autocovariance.append(acov / np.arange(n, 0, -1))
    autocovariance = np.asarray(autocovariance)
    rho = np.ones(n, dtype=np.float64)
    for lag in range(1, n):
        rho[lag] = 1.0 - (within - float(np.mean(autocovariance[:, lag]))) / variance_plus
    pair_sums = []
    for lag in range(1, n - 1, 2):
        pair = float(rho[lag] + rho[lag + 1])
        if pair < 0.0:
            break
        pair_sums.append(pair)
    for index in range(1, len(pair_sums)):
        pair_sums[index] = min(pair_sums[index], pair_sums[index - 1])
    tau = max(1.0, -1.0 + 2.0 * (1.0 + sum(pair_sums)))
    return float(min(m * n, m * n / tau))


def projection_names(field_indices: np.ndarray, roi_names: np.ndarray) -> list[str]:
    names = list(NUISANCE_NAMES)
    names.extend(f"white_coordinate_{int(index)}" for index in field_indices)
    names.extend(f"density_ROI_{str(name)}" for name in roi_names)
    names.append("logdensity")
    return names


def audit_task(task_root: Path, maximum_lag: int) -> dict[str, object]:
    result_path = task_root / "result.json"
    diagnostics_path = task_root / "diagnostics.npz"
    if not result_path.is_file() or not diagnostics_path.is_file():
        raise ACFAuditError(f"missing task inputs: {task_root}")
    result = json.loads(result_path.read_text())
    if result.get("schema") != RESULT_SCHEMA or result.get("pilot_pass") is not False:
        raise ACFAuditError("ACF input task is not the completed NO-GO artifact")
    with np.load(diagnostics_path, allow_pickle=False) as archive:
        required = {
            "convergence_projection_samples",
            "field_probe_indices",
            "field_probe_samples",
            "nuisance_unit_samples",
            "roi_density_projection_samples",
            "roi_names",
            "sampler_energy",
            "sampler_is_divergent",
        }
        if not required.issubset(archive.files):
            raise ACFAuditError("ACF diagnostics input set is incomplete")
        samples = np.asarray(archive["convergence_projection_samples"], dtype=np.float64)
        field_indices = np.asarray(archive["field_probe_indices"], dtype=np.int64)
        roi_names = np.asarray(archive["roi_names"]).astype(str)
        energies = np.asarray(archive["sampler_energy"], dtype=np.float64)
        divergences = np.asarray(archive["sampler_is_divergent"], dtype=bool)
    if samples.shape[:2] != (4, 512) or samples.shape[2] != 39:
        raise ACFAuditError("ACF projection shape is not 4x512x39")
    if field_indices.tolist() != [0, 1, 31, 32, 1024, 4096, 16384, 32767]:
        raise ACFAuditError("ACF field probes changed")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(energies)):
        raise ACFAuditError("sampling diagnostics are nonfinite")
    names = projection_names(field_indices, roi_names)
    if len(names) != samples.shape[2]:
        raise ACFAuditError("ACF projection labels do not match samples")

    rows = []
    for index, name in enumerate(names):
        chains = samples[:, :, index]
        acfs = np.asarray([autocorrelation(chain, maximum_lag) for chain in chains])
        rows.append(
            {
                "name": name,
                "lag1_by_chain": acfs[:, 1].tolist(),
                "lag1_mean": float(np.mean(acfs[:, 1])),
                "lag1_std": float(np.std(acfs[:, 1])),
                "acf_lags_0_to_maximum": acfs.mean(axis=0).tolist(),
                "integrated_ESS": integrated_ess(chains),
                "chain_means": chains.mean(axis=1).tolist(),
                "chain_std": chains.std(axis=1, ddof=1).tolist(),
            }
        )

    step_sizes = np.asarray(result["sampler"]["step_size"], dtype=np.float64)
    integration_steps = int(result["sampler"]["integration_steps"])
    trajectory_lengths = step_sizes * integration_steps
    predicted_rho = np.cos(trajectory_lengths)
    predicted_ess = (4 * 512) * (1.0 - predicted_rho) / (1.0 + predicted_rho)
    white_rows = rows[24:32]
    observed_rho = float(np.mean([row["lag1_mean"] for row in white_rows]))
    observed_ess = float(np.mean([row["integrated_ESS"] for row in white_rows]))
    mean_predicted_rho = float(np.mean(predicted_rho))
    mean_predicted_ess = float(np.mean(predicted_ess))
    difference = abs(observed_rho - mean_predicted_rho)
    if difference <= 0.05 and observed_ess < 100.0 and mean_predicted_ess < 100.0:
        classification = "CONSISTENT_WITH_SHORT_TRAJECTORY"
    elif observed_rho > mean_predicted_rho + 0.05 or observed_ess < 0.75 * mean_predicted_ess:
        classification = "POSSIBLE_ADDITIONAL_POSTERIOR_GEOMETRY_OR_INITIALIZATION_EFFECT"
    else:
        classification = "SHORT_TRAJECTORY_NOT_ISOLATED_BY_ACF"
    return {
        "task": result["assignment"],
        "input_hashes": {
            "result_json": sha256_file(result_path),
            "diagnostics_npz": sha256_file(diagnostics_path),
        },
        "sampler": {
            "step_size": step_sizes.tolist(),
            "integration_steps": integration_steps,
            "trajectory_length": trajectory_lengths.tolist(),
            "predicted_harmonic_lag1": predicted_rho.tolist(),
            "predicted_harmonic_ESS": predicted_ess.tolist(),
        },
        "sampling_diagnostics": {
            "sampling_energy_all_finite": bool(np.all(np.isfinite(energies))),
            "warmup_energy_trace_available": False,
            "sampling_divergence_fraction": float(np.mean(divergences)),
        },
        "white_probe_summary": {
            "observed_lag1_mean": observed_rho,
            "predicted_lag1_mean": mean_predicted_rho,
            "absolute_lag1_difference": float(difference),
            "observed_integrated_ESS_mean": observed_ess,
            "predicted_harmonic_ESS_mean": mean_predicted_ess,
            "observed_integrated_ESS_min": float(min(row["integrated_ESS"] for row in white_rows)),
            "classification": classification,
        },
        "projections": rows,
    }


def audit(
    program_path: str | Path,
    output: str | Path,
    implementation_commit: str,
) -> None:
    program, program_sha = load_program(program_path)
    if len(implementation_commit) != 40:
        raise ACFAuditError("implementation commit must be a full Git hash")
    output_path = Path(output)
    if output_path.exists():
        raise ACFAuditError("ACF audit output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for key in program["task_keys"]:
        task_root = Path(program["inputs"][key]["task_root"])
        expected_result_hash = program["inputs"][key]["result_sha256"]
        if sha256_file(task_root / "result.json") != expected_result_hash:
            raise ACFAuditError(f"task result hash changed: {key}")
        if sha256_file(task_root / "diagnostics.npz") != program["inputs"][key]["diagnostics_sha256"]:
            raise ACFAuditError(f"task diagnostics hash changed: {key}")
        outcomes.append(audit_task(task_root, int(program["analysis"]["maximum_lag"])))
    aggregate_path = Path(program["inputs"]["aggregate"]["path"])
    if sha256_file(aggregate_path) != program["inputs"]["aggregate"]["sha256"]:
        raise ACFAuditError("aggregate input hash changed")
    aggregate = json.loads(aggregate_path.read_text())
    if aggregate.get("schema") != AGGREGATE_SCHEMA or aggregate.get("both_pilot_tasks_pass") is not False:
        raise ACFAuditError("aggregate input is not the expected sampler NO-GO")
    all_short = all(
        row["white_probe_summary"]["classification"] == "CONSISTENT_WITH_SHORT_TRAJECTORY"
        for row in outcomes
    )
    result = {
        "schema": SCHEMA,
        "status": "PASS_OFFLINE_ACF_AUDIT",
        "program": {
            "path": str(Path(program_path).resolve()),
            "sha256": program_sha,
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(__file__),
            "commit": implementation_commit,
        },
        "task_count": len(outcomes),
        "outcomes": outcomes,
        "interpretation": {
            "all_white_probes_consistent_with_short_trajectory": all_short,
            "new_GPU_run_required_by_this_audit": False,
            "threshold_relaxation_allowed": False,
            "remaining_mock_indices_released": False,
            "actual_observational_posterior_allowed": False,
            "validation_or_Phase_D_allowed": False,
        },
        "scope_firewall": program["scope_firewall"],
    }
    staging = output_path.parent / f".{output_path.name}.staging"
    if staging.exists():
        raise ACFAuditError("ACF audit staging output already exists")
    staging.mkdir(mode=0o700)
    (staging / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    manifest = {
        "schema": "ouruniv-cf4-phasec-sampler-mechanics-acf-audit-manifest-v1",
        "files": [{"name": "audit.json", "bytes": (staging / "audit.json").stat().st_size, "sha256": sha256_file(staging / "audit.json")}],
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete = {
        "schema": "ouruniv-cf4-phasec-sampler-mechanics-acf-audit-complete-v1",
        "audit_sha256": sha256_file(staging / "audit.json"),
        "manifest_sha256": sha256_file(staging / "manifest.json"),
    }
    (staging / "COMPLETE").write_text(json.dumps(complete, sort_keys=True) + "\n")
    os.replace(staging, output_path)


def validate(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    expected = {"audit.json", "manifest.json", "COMPLETE"}
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected:
        raise ACFAuditError("ACF audit artifact set mismatch")
    result = json.loads((root / "audit.json").read_text())
    if result.get("schema") != SCHEMA:
        raise ACFAuditError("ACF audit result schema mismatch")
    complete = json.loads((root / "COMPLETE").read_text())
    if complete.get("audit_sha256") != sha256_file(root / "audit.json"):
        raise ACFAuditError("ACF audit result hash mismatch")
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise ACFAuditError("ACF audit manifest hash mismatch")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--program", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--implementation-commit", required=True)
    check = sub.add_parser("validate")
    check.add_argument("--directory", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        audit(args.program, args.output, args.implementation_commit)
    else:
        validate(args.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
