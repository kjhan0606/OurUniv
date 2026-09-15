#!/usr/bin/env python3
"""Run and validate the non-scientific CF4 frontier calibration smoke.

This driver exercises manifest binding and the development metric interface on
deterministic synthetic arrays.  It does not create CF4 selection/noise truth
mocks, evaluate coverage, or authorize an observational frontier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cf4_kf_bin_manifest import (  # noqa: E402
    canonical_json_bytes,
    validate_manifest_envelope,
)
from cf4_mock_calibration import (  # noqa: E402
    NOT_EVALUATED,
    compute_development_smoke_metrics,
    development_upstream_gate_schema,
)


CONFIG_SCHEMA = "ouruniv-cf4-kf-calibration-smoke-execution-v1"
RESULT_SCHEMA = "ouruniv-cf4-kf-calibration-smoke-result-v1"
ARTIFACT_SCHEMA = "ouruniv-cf4-kf-calibration-smoke-artifact-manifest-v1"
COMPLETE_SCHEMA = "ouruniv-cf4-kf-calibration-smoke-complete-v1"
EXPECTED_FILES = {"result.json", "manifest.json", "COMPLETE"}


class SmokeError(ValueError):
    """The implementation smoke violates its frozen execution contract."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeError(f"cannot parse JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise SmokeError(f"JSON root must be an object: {path}")
    return value, payload


def _write_once(path: Path, value: object) -> None:
    payload = canonical_json_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _require_sha(value: object, label: str, length: int = 64) -> str:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise SmokeError(f"{label} must be lowercase {length}-hex")
    return value


def _validate_config(config: dict[str, object]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise SmokeError("unexpected smoke config schema")
    if config.get("status") != "USER_APPROVED_SINGLE_IMPLEMENTATION_SMOKE":
        raise SmokeError("smoke config is not user-approved and active")
    authority = config.get("authorization")
    if not isinstance(authority, dict):
        raise SmokeError("authorization record is absent")
    required_true = (
        "final_manifest_materialization_authorized",
        "single_Slurm_implementation_smoke_authorized",
        "GPFS_declared_read_write_authorized",
    )
    required_false = (
        "development_64_mock_science_execution_authorized",
        "untouched_256_mock_validation_authorized",
        "KF_EXPAND_authorized",
        "science_inference_or_frontier_claim_authorized",
        "IC_PM_HOP_RAMSES_authorized",
        "network_access_authorized",
        "retry_authorized",
    )
    if any(authority.get(key) is not True for key in required_true):
        raise SmokeError("required smoke authority is missing")
    if any(authority.get(key) is not False for key in required_false):
        raise SmokeError("smoke config contains forbidden authority")


def _source_bindings(config: dict[str, object]) -> dict[str, dict[str, object]]:
    bindings = config.get("source_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise SmokeError("source bindings are absent")
    checked: dict[str, dict[str, object]] = {}
    for label, record in bindings.items():
        if not isinstance(label, str) or not isinstance(record, dict):
            raise SmokeError("source binding is malformed")
        path_value = record.get("path")
        expected = _require_sha(record.get("sha256"), f"{label} SHA256")
        if not isinstance(path_value, str):
            raise SmokeError(f"{label} path is absent")
        path = ROOT / path_value
        payload = path.read_bytes()
        observed = _sha256(payload)
        if observed != expected:
            raise SmokeError(f"{label} source SHA256 mismatch")
        checked[label] = {
            "path": path_value,
            "sha256": observed,
            "bytes": len(payload),
        }
    return checked


def _load_bound_manifest(
    config: dict[str, object], manifest_path: Path
) -> tuple[dict[str, object], bytes]:
    value, payload = _load_object(manifest_path)
    body = validate_manifest_envelope(value)
    binding = config.get("bin_manifest")
    if not isinstance(binding, dict):
        raise SmokeError("bin-manifest binding is absent")
    bound_path = binding.get("path")
    if not isinstance(bound_path, str) or (ROOT / bound_path).resolve() != manifest_path.resolve():
        raise SmokeError("bin-manifest path mismatch")
    if _sha256(payload) != _require_sha(binding.get("file_sha256"), "manifest file SHA256"):
        raise SmokeError("bin-manifest file SHA256 mismatch")
    if value.get("manifest_body_sha256") != _require_sha(
        binding.get("body_sha256"), "manifest body SHA256"
    ):
        raise SmokeError("bin-manifest body SHA256 mismatch")
    return body, payload


def _geometry_mask(body: dict[str, object], domain_id: str) -> np.ndarray:
    roi_id = domain_id.removesuffix("_delta")
    rows = body.get("ROI_support_records")
    if not isinstance(rows, list):
        raise SmokeError("manifest ROI support records are absent")
    matches = [row for row in rows if isinstance(row, dict) and row.get("ROI_id") == roi_id]
    if len(matches) != 1:
        raise SmokeError("smoke domain does not identify exactly one manifest ROI")
    records = matches[0].get("all_native_bin_geometry_records")
    if not isinstance(records, list) or len(records) != 38:
        raise SmokeError("manifest ROI geometry records are incomplete")
    mask = np.asarray([record.get("geometry_supported") for record in records])
    if mask.dtype != np.dtype(bool) or mask.shape != (38,):
        raise SmokeError("manifest ROI geometry mask is not exact boolean length 38")
    return mask


def run_smoke(
    config_path: Path,
    manifest_path: Path,
    output_directory: Path,
    implementation_commit: str,
) -> dict[str, object]:
    config, config_bytes = _load_object(config_path)
    _validate_config(config)
    commit = _require_sha(implementation_commit, "implementation commit", length=40)
    if not output_directory.is_dir() or any(output_directory.iterdir()):
        raise SmokeError("staging output directory must exist and be empty")
    body, manifest_bytes = _load_bound_manifest(config, manifest_path)
    sources = _source_bindings(config)
    contract = config.get("smoke_contract")
    if not isinstance(contract, dict):
        raise SmokeError("smoke contract is absent")
    mock_count = int(contract.get("mock_count", 0))
    draw_count = int(contract.get("posterior_draw_count", 0))
    modes_per_bin = int(contract.get("synthetic_modes_per_native_bin", 0))
    seed = int(contract.get("implementation_seed", -1))
    domain_id = contract.get("domain_id")
    if (
        mock_count != 4
        or draw_count != 8
        or modes_per_bin != 4
        or seed != 913
        or domain_id != "Local_Group_delta"
    ):
        raise SmokeError("smoke dimensions, seed, or domain changed")

    declared_bins = np.arange(38, dtype=np.int64)
    mode_bins = np.repeat(declared_bins, modes_per_bin)
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(mock_count, mode_bins.size))
    centered_draw_noise = rng.normal(
        size=(mock_count, draw_count, mode_bins.size)
    )
    centered_draw_noise -= centered_draw_noise.mean(axis=1, keepdims=True)
    posterior = truth[:, None, :] + 0.2 * centered_draw_noise
    upstream = development_upstream_gate_schema(
        np.zeros(38, dtype=bool), np.zeros(38, dtype=bool)
    )
    metrics = compute_development_smoke_metrics(
        truth,
        posterior,
        np.ones(mode_bins.size, dtype=np.float64),
        mode_bins,
        declared_bins,
        _geometry_mask(body, str(domain_id)),
        upstream,
        domain_id=str(domain_id),
        bin_manifest_body_sha256=str(config["bin_manifest"]["body_sha256"]),
    )
    if (
        metrics.get("CF4_selection_noise_truth_mock_provenance_validated") is not False
        or metrics.get("development_science_metric_allowed") is not False
        or metrics.get("strict_frontier_or_science_claim_allowed") is not False
        or any(metrics.get("strict_gate_before_geometry", []))
        or any(metrics.get("strict_gate_intersection_with_geometry", []))
    ):
        raise SmokeError("implementation smoke escaped its fail-closed state")

    result = {
        "schema": RESULT_SCHEMA,
        "status": "SMOKE_PASS",
        "mode": "implementation_smoke",
        "implementation_commit": commit,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_bytes),
        "bin_manifest_path": str(manifest_path),
        "bin_manifest_file_sha256": _sha256(manifest_bytes),
        "bin_manifest_body_sha256": config["bin_manifest"]["body_sha256"],
        "source_bindings": sources,
        "metrics": metrics,
        "science_disposition": "NO_SCIENCE_CLAIM_IMPLEMENTATION_SMOKE_ONLY",
        "development_64_mock_science_execution_performed": False,
        "untouched_256_mock_validation_performed": False,
        "KF_EXPAND_authorized": False,
    }
    _write_once(output_directory / "result.json", result)
    result_bytes = (output_directory / "result.json").read_bytes()
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "status": "SMOKE_PASS",
        "implementation_commit": commit,
        "payloads": {
            "result.json": {
                "sha256": _sha256(result_bytes),
                "bytes": len(result_bytes),
            }
        },
    }
    _write_once(output_directory / "manifest.json", artifact)
    artifact_bytes = (output_directory / "manifest.json").read_bytes()
    complete = {
        "schema": COMPLETE_SCHEMA,
        "status": "SMOKE_PASS",
        "implementation_commit": commit,
        "manifest_sha256": _sha256(artifact_bytes),
    }
    _write_once(output_directory / "COMPLETE", complete)
    return result


def validate_smoke(directory: Path, config_path: Path, manifest_path: Path) -> dict[str, object]:
    if not directory.is_dir() or {item.name for item in directory.iterdir()} != EXPECTED_FILES:
        raise SmokeError("smoke artifact file set is not exact")
    config, config_bytes = _load_object(config_path)
    _validate_config(config)
    sources = _source_bindings(config)
    _, manifest_bytes = _load_bound_manifest(config, manifest_path)
    result, result_bytes = _load_object(directory / "result.json")
    artifact, artifact_bytes = _load_object(directory / "manifest.json")
    complete, complete_bytes = _load_object(directory / "COMPLETE")
    for path, value, payload in (
        (directory / "result.json", result, result_bytes),
        (directory / "manifest.json", artifact, artifact_bytes),
        (directory / "COMPLETE", complete, complete_bytes),
    ):
        if payload != canonical_json_bytes(value):
            raise SmokeError(f"artifact is not canonical JSON: {path.name}")
    if result.get("schema") != RESULT_SCHEMA or result.get("status") != "SMOKE_PASS":
        raise SmokeError("result is not SMOKE_PASS v1")
    if result.get("config_sha256") != _sha256(config_bytes):
        raise SmokeError("result/config binding mismatch")
    if result.get("bin_manifest_file_sha256") != _sha256(manifest_bytes):
        raise SmokeError("result/bin-manifest binding mismatch")
    if result.get("source_bindings") != sources:
        raise SmokeError("result/source binding mismatch")
    if (
        result.get("science_disposition")
        != "NO_SCIENCE_CLAIM_IMPLEMENTATION_SMOKE_ONLY"
        or result.get("development_64_mock_science_execution_performed") is not False
        or result.get("untouched_256_mock_validation_performed") is not False
        or result.get("KF_EXPAND_authorized") is not False
    ):
        raise SmokeError("result contains a forbidden science state")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise SmokeError("result metrics are absent")
    for status in ("coverage68_status", "coverage95_status", "heldout_improvement_status"):
        if metrics.get(status) != NOT_EVALUATED:
            raise SmokeError("smoke contains evaluated validation-only metric")
    strict_before = metrics.get("strict_gate_before_geometry")
    strict_intersection = metrics.get("strict_gate_intersection_with_geometry")
    if (
        not isinstance(strict_before, list)
        or len(strict_before) != 38
        or any(type(item) is not bool for item in strict_before)
        or not isinstance(strict_intersection, list)
        or len(strict_intersection) != 38
        or any(type(item) is not bool for item in strict_intersection)
    ):
        raise SmokeError("smoke strict-gate vectors are malformed")
    if (
        metrics.get("CF4_selection_noise_truth_mock_provenance_validated") is not False
        or metrics.get("development_science_metric_allowed") is not False
        or metrics.get("strict_frontier_or_science_claim_allowed") is not False
        or any(strict_before)
        or any(strict_intersection)
    ):
        raise SmokeError("smoke result is not fail-closed")
    if artifact != {
        "schema": ARTIFACT_SCHEMA,
        "status": "SMOKE_PASS",
        "implementation_commit": result.get("implementation_commit"),
        "payloads": {
            "result.json": {
                "sha256": _sha256(result_bytes),
                "bytes": len(result_bytes),
            }
        },
    }:
        raise SmokeError("artifact manifest/result binding mismatch")
    if complete != {
        "schema": COMPLETE_SCHEMA,
        "status": "SMOKE_PASS",
        "implementation_commit": result.get("implementation_commit"),
        "manifest_sha256": _sha256(artifact_bytes),
    }:
        raise SmokeError("COMPLETE/artifact-manifest binding mismatch")
    return {
        "status": "PASS",
        "directory": str(directory),
        "result_sha256": _sha256(result_bytes),
        "bin_manifest_body_sha256": result["bin_manifest_body_sha256"],
        "science_disposition": result["science_disposition"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--output-directory", required=True, type=Path)
    run.add_argument("--implementation-commit", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--directory", required=True, type=Path)
    validate.add_argument("--config", required=True, type=Path)
    validate.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            report = run_smoke(
                args.config,
                args.manifest,
                args.output_directory,
                args.implementation_commit,
            )
            summary = {
                "status": report["status"],
                "science_disposition": report["science_disposition"],
            }
        else:
            summary = validate_smoke(args.directory, args.config, args.manifest)
    except (OSError, KeyError, TypeError, ValueError, SmokeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
