"""v6 fail-closed authorization boundary; no grant or execution is shipped."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from typing import Any

import cf4_aggregate_evidence_smc_execution as base_execution
import cf4_aggregate_evidence_smc_shared_annealing_v6 as shared_v6


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DESIGN = ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_design_v6.json"
CANONICAL_PROGRAM = ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_program_v6.json"
CANONICAL_GRANT = ROOT / "config/cf4_aggregate_evidence_smc_execution_grant_v6.json"
EXTERNAL_RELEASE = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_execution_authorization_v6_release.json")
EXTERNAL_MANIFEST = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_execution_authorization_v6_manifest.json")
RECEIPT_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_receipts")
PILOT_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_disposable_pilot")
DATA_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6")
STATE_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_run")
DESIGN_SHA256 = "cdbb6d0299a1012f402422ad9001a7cb58907db2175c14e9590e018325af7772"
SHARED_DESIGN_SHA256 = "5d17ad724e780b4c5f5d0f2de1b497a9207719f338627ca92441e4acbf7adb18"
SHARED_SOURCE_SHA256 = "b6f676caca512af9bd88be12f54b08844bd3d3f6e335ac1e5cc9d4265482c060"
BASE_PROGRAM_SHA256 = "74cd10fdff0171daff6984ebc8db13cfd82d6dc495891ff585b81ac9eb0129c5"

AUTHORIZATION = {
    "v6_design_and_implementation_authorized": True,
    "grant_creation_authorized": False,
    "external_release_or_manifest_creation_authorized": False,
    "pilot_execution_authorized": False,
    "production_execution_authorized": False,
    "cache_population_authorized": False,
    "downstream_execution_authorized": False,
    "automatic_retry_retune_scale_up_or_follow_on_authorized": False,
}
COMPLETE_STATUSES = {
    "complete_pass_production_smc", "complete_scientific_fail_production_smc",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PermissionError(f"{label} is absent")
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PermissionError(f"{label} is not valid JSON") from error
    if not isinstance(result, dict):
        raise PermissionError(f"{label} is not an object")
    return result


def _v6_path(value: str, expected: Path, label: str) -> None:
    path = Path(value)
    if "aggregate_evidence_smc_v5" in str(path).lower() or path.resolve() != expected.resolve():
        raise PermissionError(f"{label} is not the canonical v6-only path")


def _expected_hard_pins() -> dict[str, dict[str, str]]:
    return {
        "shared_annealing_design": {
            "path": "config/cf4_aggregate_evidence_smc_shared_annealing_v6_design.json",
            "sha256": SHARED_DESIGN_SHA256,
        },
        "shared_annealing_source": {
            "path": "src/cf4_aggregate_evidence_smc_shared_annealing_v6.py",
            "sha256": SHARED_SOURCE_SHA256,
        },
        "base_production_program": {
            "path": "config/cf4_aggregate_evidence_smc_production_program.json",
            "sha256": BASE_PROGRAM_SHA256,
            "source_commit": "6630b6b04ab02e513d47f1667617384894eb349f",
            "capability_commit": "22587e47232782feb4c08768d8f64d853d76e62b",
        },
    }


def _expected_fixed_science() -> dict[str, Any]:
    return {
        "master_seeds": [2026082301, 2026082302, 2026082303, 2026082304],
        "particles_per_replicate": 2048,
        "parent_seed_range_inclusive": [3193, 3448], "parent_count": 256,
        "target_CESS_fraction": 0.8, "resampling_ESS_fraction": 0.5,
        "MH_sweeps_per_stage": 4, "maximum_temperature_stages": 256,
        "worker_processes": 8, "threads_per_worker": 1,
        "replicates_sequential": True,
    }


def _expected_storage() -> dict[str, Any]:
    return {
        "grant": str(CANONICAL_GRANT.relative_to(ROOT)), "release": str(EXTERNAL_RELEASE),
        "manifest": str(EXTERNAL_MANIFEST), "receipt_root": str(RECEIPT_ROOT),
        "pilot_root": str(PILOT_ROOT), "data_root": str(DATA_ROOT),
        "state_root": str(STATE_ROOT), "restart_or_checkpoint_import": False,
    }


def validate_authorization_program(program: dict[str, Any]) -> None:
    if (
        program.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-program-v6"
        or program.get("status") != "frozen_versioned_one_shot_program_execution_unauthorized"
        or program.get("authorization") != AUTHORIZATION
        or program.get("hard_pins") != _expected_hard_pins()
        or program.get("fixed_science") != _expected_fixed_science()
        or program.get("storage") != _expected_storage()
    ):
        raise PermissionError("v6 authorization program contract changed")
    if program.get("authorization_design") != {
        "path": str(CANONICAL_DESIGN.relative_to(ROOT)), "sha256": DESIGN_SHA256,
    }:
        raise PermissionError("v6 authorization design pin changed")
    future = program.get("future_shared_pilot_and_production_contract", {})
    if future.get("pilot_master_seed") != 2026082301 or future.get("pilot_N") != 2048 \
            or future.get("pilot_all_256_parents") is not True \
            or future.get("pilot_cache_posterior_and_scientific_result_reuse") is not False \
            or future.get("pilot_close_and_dispose_before_production") is not True \
            or future.get("all_four_masters_including_pilot_seed_rerun_under_immutable_schedule") is not True \
            or future.get("stage_parity_mismatch_is_architecture_failure") is not True \
            or future.get("unreachable_without_future_grant_release_and_manifest") is not True:
        raise PermissionError("v6 future shared-annealing contract changed")
    interface = program.get("future_grant_interface", {})
    if interface != {
        "canonical_path": str(CANONICAL_GRANT.relative_to(ROOT)),
        "schema": "ouruniv-cf4-aggregate-evidence-smc-execution-grant-v6",
        "current_grant_present": False, "runtime_override_allowed": False,
        "must_bind_program_design_shared_source_and_v6_paths": True,
    }:
        raise PermissionError("v6 future grant interface changed")
    for row in _expected_hard_pins().values():
        if sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise PermissionError(f"v6 hard-pinned input changed: {row['path']}")
    if sha256_file(CANONICAL_DESIGN) != DESIGN_SHA256:
        raise PermissionError("v6 authorization design hash changed")
    shared_v6.validate_frozen_v6_parameters()


def load_canonical_authorization_program() -> dict[str, Any]:
    program = _load_object(CANONICAL_PROGRAM, "canonical v6 authorization program")
    validate_authorization_program(program)
    return program


def require_execution_authorization(program: dict[str, Any]) -> dict[str, Any]:
    """Read-only canonical gate; absence is fatal before every reservation."""
    validate_authorization_program(program)
    if CANONICAL_GRANT.exists():
        # A future grant must be paired with external lineage evidence.  This
        # implementation deliberately cannot turn that evidence into execution.
        if not EXTERNAL_RELEASE.is_file() or not EXTERNAL_MANIFEST.is_file():
            raise PermissionError("v6 grant lacks paired external release and manifest")
        raise PermissionError("v6 future grant is unreachable in this implementation")
    raise PermissionError("v6 sealed one-shot grant is absent; execution remains unauthorized")


def read_only_v6_postcheck(data_directory: Path) -> dict[str, Any]:
    """Future runner completion contract; this helper never writes artifacts."""
    root = Path(data_directory).resolve()
    if "aggregate_evidence_smc_v5" in str(root).lower() or root != DATA_ROOT.resolve():
        raise PermissionError("v6 postcheck rejects noncanonical or v5 data paths")
    result_path, manifest_path = root / "result.json", root / "manifest.json"
    if not result_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("v6 result or manifest is absent")
    if manifest_path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise PermissionError("v6 manifest must be read-only before COMPLETE")
    checked = base_execution.validate_published_bundle(root)
    if checked.get("status") not in COMPLETE_STATUSES or checked.get("valid_scientific_complete") is not True:
        raise RuntimeError("v6 postcheck is not an allowed scientific completion")
    return checked


def run_authorized_v6(program_path: Path) -> None:
    """Sole public entry. It always fails closed in this shipped change."""
    if Path(program_path).resolve() != CANONICAL_PROGRAM.resolve():
        raise PermissionError("authorized v6 accepts only the canonical program path")
    require_execution_authorization(load_canonical_authorization_program())
    raise AssertionError("unreachable v6 execution path")
