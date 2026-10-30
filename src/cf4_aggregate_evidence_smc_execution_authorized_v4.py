"""Versioned fail-closed authorization boundary for aggregate-evidence SMC.

This module cannot authorize the production run by itself.  Its sole public
entry accepts the frozen canonical authorization program and additionally
requires a future, separately committed and audited one-shot grant at one
fixed path.  No grant is shipped with this implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

import cf4_aggregate_evidence_smc_execution as base_execution


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PROGRAM = (
    ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_program_v4.json"
)
PROGRAM_SHA256 = "a4fcefa5d5bc163563df1bf5041c91415c3a4d6d8bdf20e710790e5e4e7f0b23"
CANONICAL_DESIGN = (
    ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_design_v4.json"
)
DESIGN_SHA256 = "48d9d173bbc5c4758e345ac1cffd3d61901663fb9feb0a1274ffb2b3336fbd69"
GRANT_RELATIVE_PATH = "config/cf4_aggregate_evidence_smc_execution_grant_v4.json"
CANONICAL_GRANT = ROOT / GRANT_RELATIVE_PATH
AUTHORIZATION_RESULT_RELATIVE_PATH = (
    "config/cf4_aggregate_evidence_smc_execution_authorization_implementation_"
    "result_record_v4.json"
)
EXTERNAL_RELEASE = Path(
    "/gpfs/kjhan/CF4/recon/linear_cr/"
    "aggregate_evidence_smc_execution_authorization_v4_release.json"
)
AUTHORIZATION_DESIGN_BASE_COMMIT = "d3213fa8fa2effe82dc6874911d21132dc088b4b"
RUNNER_IMPLEMENTATION_COMMIT = "f4e282cb1fe1e80a1184ead23d1fe5892a0c7c5e"
RUNNER_RESULT_PATH = (
    ROOT / "config/cf4_aggregate_evidence_smc_runner_implementation_result_record.json"
)
RUNNER_RESULT_SHA256 = "d96708a9f6b4998237aba4b4078918e2b483b7bb7cbf370bb1866da692d9f92a"
BASE_PROGRAM = ROOT / "config/cf4_aggregate_evidence_smc_production_program.json"
BASE_PROGRAM_SHA256 = "74cd10fdff0171daff6984ebc8db13cfd82d6dc495891ff585b81ac9eb0129c5"
DATA_DIRECTORY = Path(
    "/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v4"
)
STATE_DIRECTORY = Path(
    "/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v4_run"
)

AUTHORIZATION_KEYS = {
    "versioned_one_shot_authorization_design_and_implementation_authorized",
    "production_SMC_execution_authorized",
    "oracle_cache_population_authorized",
    "conditional_field_bank_authorized",
    "candidate_generation_authorized",
    "parent_or_seed_selection_authorized",
    "PM_authorized",
    "HOP_authorized",
    "RAMSES_authorized",
    "downstream_execution_authorized",
    "automatic_retry_authorized",
    "automatic_retune_authorized",
    "automatic_scale_up_authorized",
    "automatic_follow_on_authorized",
}

AUTHORIZATION_V2_PATHS = (
    "config/cf4_aggregate_evidence_smc_execution_authorization_design_v4.json",
    "config/cf4_aggregate_evidence_smc_execution_authorization_program_v4.json",
    "src/cf4_aggregate_evidence_smc_execution_authorized_v4.py",
    "tests/test_cf4_aggregate_evidence_smc_execution_authorized_v4.py",
    "scripts/run_cf4_aggregate_evidence_smc_authorized_v4_lageunha.sh",
    "scripts/launch_cf4_aggregate_evidence_smc_authorized_v4_lageunha.sh",
    "tests/test_cf4_aggregate_evidence_smc_authorized_v4_runner.py",
)

GRANT_KEYS = {
    "schema",
    "status",
    "one_shot",
    "grant_parent_commit",
    "authorization_program_sha256",
    "authorization_design_base_commit",
    "runner_implementation_commit",
    "runner_result_record_sha256",
    "authorization_v4_implementation_commit",
    "authorization_v4_implementation_result_record",
    "authorization_v4_files",
    "data_directory",
    "state_directory",
    "authorization",
    "precommit_audit_verdict",
}

EXPECTED_RUNNER_FILES = (
    (
        "config/cf4_aggregate_evidence_smc_production_program.json",
        "74cd10fdff0171daff6984ebc8db13cfd82d6dc495891ff585b81ac9eb0129c5",
        "36df434fb9e5b94dffa50af520361df67e1462c9",
        "100644",
    ),
    (
        "scripts/launch_cf4_aggregate_evidence_smc_production_lageunha.sh",
        "330000c2c74092b529541518e743fde751a1b91394412c2376448c5c250e3473",
        "688d1dd10ae10f9b2e071961057bdabadcd7a38e",
        "100755",
    ),
    (
        "scripts/run_cf4_aggregate_evidence_smc_production_lageunha.sh",
        "e0e3c359ca92ed556a2cfe240121263407e664bbc709cc93670a941eee889431",
        "7bc72c27cd5001dbca483534f535f0120010aa22",
        "100755",
    ),
    (
        "scripts/status_cf4_aggregate_evidence_smc_production.sh",
        "5a122ee557783d838ceea2beba70c66d56bd7c97fa0a388c428922da34baab23",
        "dc2af65ff9b6329dfd8292264ef29d9dc9d4a16a",
        "100755",
    ),
    (
        "src/cf4_aggregate_evidence_smc_execution.py",
        "64f68b8bd0ae69c02509fdd940f80a393439c7c94362c5e279b8edc7b1a08533",
        "683f6d8da86b72db86ecb1b95ac23e51100867f9",
        "100644",
    ),
    (
        "tests/test_cf4_aggregate_evidence_smc_execution.py",
        "8d15adb282e0c8e16138ed9559a34e619cb7de2fc3ad57ec250b7c00fe339fbc",
        "00cd38427804032c1a8eaa650f66a28429be517b",
        "100644",
    ),
    (
        "tests/test_cf4_aggregate_evidence_smc_production_runner.py",
        "b68fe387c82e64dd6de7ade594949a9ae2fb7e1a8dbb377fe630e90d89e14e38",
        "1564c205142ba65e438da5dfcc8ad9990fdcdcec",
        "100644",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_oid(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _git_mode(path: Path) -> str:
    mode = Path(path).stat().st_mode
    return "100755" if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) else "100644"


def _resolved(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeError(f"{label} is not a full lowercase SHA256")
    return text


def _require_commit(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeError(f"{label} is not a full lowercase Git commit")
    return text


def _expected_authorization() -> dict[str, bool]:
    result = {key: False for key in AUTHORIZATION_KEYS}
    result["versioned_one_shot_authorization_design_and_implementation_authorized"] = True
    return result


def _validate_design() -> None:
    if sha256_file(CANONICAL_DESIGN) != DESIGN_SHA256:
        raise RuntimeError("authorization design hash mismatch")
    design = json.loads(CANONICAL_DESIGN.read_text())
    if design.get("schema") != (
        "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-design-v4"
    ) or design.get("status") != "frozen_design_only_execution_unauthorized" \
            or design.get("authorization") != _expected_authorization() \
            or design.get("future_grant_interface", {}).get(
                "present_in_this_change"
            ) is not False:
        raise RuntimeError("authorization design contract changed")


def validate_authorization_program(
    program: dict[str, Any], *, verify_file_hashes: bool
) -> None:
    if program.get("schema") != (
        "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-program-v4"
    ) or program.get("status") != (
        "frozen_versioned_one_shot_program_execution_unauthorized"
    ) or program.get("authorization_design_base_commit") != (
        AUTHORIZATION_DESIGN_BASE_COMMIT
    ) or program.get("runner_implementation_commit") != RUNNER_IMPLEMENTATION_COMMIT:
        raise RuntimeError("authorization program identity or lineage changed")
    if program.get("authorization") != _expected_authorization():
        raise RuntimeError("authorization program matrix is not exactly fail closed")
    design = program.get("authorization_design", {})
    result_record = program.get("runner_implementation_result_record", {})
    base_program_record = program.get("base_production_program", {})
    if design != {
        "path": str(CANONICAL_DESIGN.relative_to(ROOT)),
        "sha256": DESIGN_SHA256,
    } or result_record != {
        "path": str(RUNNER_RESULT_PATH.relative_to(ROOT)),
        "sha256": RUNNER_RESULT_SHA256,
    } or base_program_record != {
        "path": str(BASE_PROGRAM.relative_to(ROOT)),
        "sha256": BASE_PROGRAM_SHA256,
    }:
        raise RuntimeError("authorization predecessor hardpin changed")
    expected_rows = [
        {
            "path": path,
            "sha256": digest,
            "git_blob_oid": blob,
            "git_mode": mode,
        }
        for path, digest, blob, mode in EXPECTED_RUNNER_FILES
    ]
    if program.get("audited_runner_files") != expected_rows:
        raise RuntimeError("authorization runner file hardpins changed")
    storage = program.get("storage", {})
    if storage != {
        "data_directory": str(DATA_DIRECTORY),
        "state_directory": str(STATE_DIRECTORY),
        "exclusive_reservation": True,
        "restart_or_checkpoint_import": False,
    }:
        raise RuntimeError("authorization storage contract changed")
    grant = program.get("future_grant_interface", {})
    if grant != {
        "canonical_path": GRANT_RELATIVE_PATH,
        "schema": "ouruniv-cf4-aggregate-evidence-smc-execution-grant-v4",
        "required_status": "sealed_one_shot_execution_authorization",
        "current_grant_present": False,
        "runtime_grant_path_override_allowed": False,
        "single_separately_audited_commit_required": True,
        "worktree_HEAD_local_tracking_remote_identity_required": True,
        "authorization_commit_field_forbidden": True,
        "postcommit_audit_verdict_field_forbidden": True,
        "grant_parent_commit_source": (
            "The commit containing the immutable authorization-v4 "
            "implementation result record."
        ),
        "authorization_v4_implementation_result_record_path": (
            AUTHORIZATION_RESULT_RELATIVE_PATH
        ),
        "prospective_values_resolved_from_grant_parent_not_self_pinned": True,
        "exact_commit_chain": (
            "grant_parent^ == authorization_v4_implementation_commit and "
            "grant_HEAD^ == grant_parent_commit"
        ),
        "result_record_and_grant_commits_each_exact_one_file_addition": True,
        "seven_files_verified_at_exact_implementation_commit": True,
    }:
        raise RuntimeError("future one-shot grant interface changed")
    if program.get("external_pre_execution_release") != {
        "canonical_path": str(EXTERNAL_RELEASE),
        "schema": (
            "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-"
            "release-v4"
        ),
        "required_status": "complete_pass_external_postcommit_lineage_audit",
        "required_verdict": "LINEAGE GO",
        "current_release_present": False,
        "committed_in_grant_commit": False,
        "runtime_path_override_allowed": False,
        "runner_SHA256_snapshot_and_postflight_revalidation_required": True,
    }:
        raise RuntimeError("external pre-execution lineage release changed")
    if sha256_file(BASE_PROGRAM) != BASE_PROGRAM_SHA256:
        raise RuntimeError("base production program hash mismatch")
    base_program = json.loads(BASE_PROGRAM.read_text())
    frozen = program.get("frozen_input_lineage", {})
    if frozen.get("pinned_local_files") != base_program.get("pinned_local_files") \
            or frozen.get("external_inputs") != base_program.get("external_inputs") \
            or frozen.get("parent_seed_range_inclusive") != [3193, 3448] \
            or frozen.get("parent_count") != 256:
        raise RuntimeError("frozen production input lineage changed")
    if verify_file_hashes:
        _validate_design()
        if sha256_file(RUNNER_RESULT_PATH) != RUNNER_RESULT_SHA256:
            raise RuntimeError("runner result record hash mismatch")
        for path, digest, blob, mode in EXPECTED_RUNNER_FILES:
            resolved = ROOT / path
            if not resolved.is_file() or sha256_file(resolved) != digest \
                    or _git_blob_oid(resolved) != blob \
                    or _git_mode(resolved) != mode:
                raise RuntimeError(f"audited runner file changed: {path}")
        verified_base = base_execution.load_canonical_program(
            verify_file_hashes=True
        )
        if verified_base != base_program:
            raise RuntimeError("base production program reload changed")


def load_canonical_authorization_program(
    *, verify_file_hashes: bool
) -> dict[str, Any]:
    if sha256_file(CANONICAL_PROGRAM) != PROGRAM_SHA256:
        raise RuntimeError("canonical authorization program hash mismatch")
    program = json.loads(CANONICAL_PROGRAM.read_text())
    validate_authorization_program(
        program, verify_file_hashes=verify_file_hashes
    )
    return program


def _git_output(*arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError) as error:
        raise RuntimeError("future one-shot grant Git seal check failed") from error


def _validate_result_record(grant: dict[str, Any], *, parent: str) -> str:
    binding = grant.get("authorization_v4_implementation_result_record")
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"} \
            or binding.get("path") != AUTHORIZATION_RESULT_RELATIVE_PATH:
        raise RuntimeError("authorization-v4 implementation result binding changed")
    expected_sha = _require_sha256(
        binding.get("sha256"), "authorization-v4 implementation result record"
    )
    tracked_record = _git_output(
        "show", f"{parent}:{AUTHORIZATION_RESULT_RELATIVE_PATH}"
    )
    result_changes = _git_output(
        "diff-tree", "--no-commit-id", "--name-status", "-r", parent
    ).decode().splitlines()
    if result_changes != [f"A\t{AUTHORIZATION_RESULT_RELATIVE_PATH}"]:
        raise RuntimeError(
            "grant parent is not the exact implementation result-record commit"
        )
    record_path = ROOT / AUTHORIZATION_RESULT_RELATIVE_PATH
    if not record_path.is_file() or record_path.read_bytes() != tracked_record \
            or hashlib.sha256(tracked_record).hexdigest() != expected_sha:
        raise RuntimeError("authorization-v4 implementation result record changed")
    record = json.loads(tracked_record)
    implementation_commit = _require_commit(
        grant.get("authorization_v4_implementation_commit"),
        "authorization-v4 implementation commit",
    )
    implementation_parent = _git_output(
        "rev-parse", f"{parent}^"
    ).decode().strip()
    if implementation_parent != implementation_commit:
        raise RuntimeError(
            "grant parent is not the direct child of the authorization-v4 "
            "implementation commit"
        )
    if record.get("schema") != (
        "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-"
        "implementation-result-record-v1"
    ) or record.get("status") != (
        "complete_pass_postcommit_authorization_v4_implementation"
    ) or record.get("commit_lineage", {}).get("git_commit") != (
        implementation_commit
    ):
        raise RuntimeError("authorization-v4 implementation result lineage changed")
    rows = record.get("committed_authorization_files")
    if grant.get("authorization_v4_files") != rows \
            or not isinstance(rows, list) or len(rows) != 7 \
            or [row.get("path") for row in rows] != list(AUTHORIZATION_V2_PATHS):
        raise RuntimeError("authorization-v4 seven-file binding changed")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "path", "sha256", "git_blob_oid", "git_mode"
        }:
            raise RuntimeError("authorization-v4 file record schema changed")
        relative = row["path"]
        digest = _require_sha256(row.get("sha256"), relative)
        blob = str(row.get("git_blob_oid"))
        mode = str(row.get("git_mode"))
        path = ROOT / relative
        tracked = _git_output("show", f"{implementation_commit}:{relative}")
        tree_fields = _git_output(
            "ls-tree", implementation_commit, "--", relative
        ).decode().split()
        if len(blob) != 40 \
                or any(character not in "0123456789abcdef" for character in blob) \
                or mode not in {"100644", "100755"} \
                or len(tree_fields) != 4 \
                or tree_fields[0] != mode or tree_fields[1] != "blob" \
                or tree_fields[2] != blob or tree_fields[3] != relative \
                or not path.is_file() or path.read_bytes() != tracked \
                or sha256_file(path) != digest \
                or _git_blob_oid(path) != blob or _git_mode(path) != mode:
            raise RuntimeError(f"authorization-v4 file changed: {relative}")
    return expected_sha


def _validate_git_seal(
    grant: dict[str, Any], grant_path: Path
) -> tuple[str, str, str]:
    refs = (
        "HEAD",
        "refs/heads/agent/freeze-zoom-pipeline",
        "refs/remotes/origin/agent/freeze-zoom-pipeline",
    )
    resolved = [_git_output("rev-parse", ref).decode().strip() for ref in refs]
    head = resolved[0]
    remote_fields = _git_output(
        "ls-remote", "--heads", "origin",
        "refs/heads/agent/freeze-zoom-pipeline",
    ).decode().split()
    remote = remote_fields[0] if remote_fields else ""
    if any(value != head for value in (*resolved, remote)):
        raise RuntimeError("future one-shot grant is not sealed at identical refs")
    parent = _git_output("rev-parse", "HEAD^").decode().strip()
    if parent != _require_commit(grant.get("grant_parent_commit"), "grant parent"):
        raise RuntimeError("future one-shot grant parent changed")
    try:
        relative = str(grant_path.relative_to(ROOT))
    except ValueError as error:
        raise RuntimeError("future one-shot grant is outside repository") from error
    if relative != GRANT_RELATIVE_PATH:
        raise RuntimeError("future one-shot grant relative path changed")
    tracked = _git_output("show", f"HEAD:{relative}")
    if tracked != grant_path.read_bytes():
        raise RuntimeError("future one-shot grant tracked blob or worktree changed")
    changes = _git_output(
        "diff-tree", "--no-commit-id", "--name-status", "-r", "HEAD"
    ).decode().splitlines()
    if changes != [f"A\t{GRANT_RELATIVE_PATH}"]:
        raise RuntimeError("grant commit is not the exact one-file addition")
    result_sha = _validate_result_record(grant, parent=parent)
    return head, parent, result_sha


def _validate_external_lineage_release(
    grant: dict[str, Any], *, head: str, parent: str, result_sha: str
) -> None:
    try:
        EXTERNAL_RELEASE.resolve().relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise PermissionError(
            "external LINEAGE GO release must not be self-committed"
        )
    if not EXTERNAL_RELEASE.is_file():
        raise PermissionError(
            "external postcommit LINEAGE GO release is absent; production SMC "
            "execution remains unauthorized"
        )
    release = json.loads(EXTERNAL_RELEASE.read_text())
    expected_keys = {
        "schema", "status", "verdict", "grant_commit", "grant_parent_commit",
        "grant_sha256", "authorization_v4_implementation_result_record_sha256",
    }
    if set(release) != expected_keys \
            or release.get("schema") != (
                "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-"
                "release-v4"
            ) \
            or release.get("status") != (
                "complete_pass_external_postcommit_lineage_audit"
            ) \
            or release.get("verdict") != "LINEAGE GO" \
            or release.get("grant_commit") != head \
            or release.get("grant_parent_commit") != parent \
            or release.get("grant_sha256") != sha256_file(CANONICAL_GRANT) \
            or release.get(
                "authorization_v4_implementation_result_record_sha256"
            ) != result_sha:
        raise PermissionError("external postcommit LINEAGE GO release is wrong")


def validate_future_grant(
    grant_path: Path, program: dict[str, Any]
) -> dict[str, Any]:
    validate_authorization_program(program, verify_file_hashes=False)
    path = Path(grant_path).resolve()
    if path != CANONICAL_GRANT.resolve():
        raise PermissionError("future one-shot grant path is not canonical")
    if not path.is_file():
        raise PermissionError(
            "future sealed one-shot execution grant is absent; production SMC "
            "execution remains unauthorized"
        )
    grant = json.loads(path.read_text())
    expected_grant_authorization = {
        "production_SMC_execution_authorized": True,
        "oracle_cache_population_authorized": True,
        "conditional_field_bank_authorized": False,
        "candidate_generation_authorized": False,
        "parent_or_seed_selection_authorized": False,
        "PM_authorized": False,
        "HOP_authorized": False,
        "RAMSES_authorized": False,
        "downstream_execution_authorized": False,
        "automatic_retry_authorized": False,
        "automatic_retune_authorized": False,
        "automatic_scale_up_authorized": False,
        "automatic_follow_on_authorized": False,
    }
    if set(grant) != GRANT_KEYS or grant.get("schema") != (
            "ouruniv-cf4-aggregate-evidence-smc-execution-grant-v4"
    ) or grant.get("status") != "sealed_one_shot_execution_authorization" \
            or grant.get("one_shot") is not True \
            or grant.get("authorization_program_sha256") != PROGRAM_SHA256 \
            or grant.get("authorization_design_base_commit") != (
                AUTHORIZATION_DESIGN_BASE_COMMIT
            ) \
            or grant.get("runner_implementation_commit") != (
                RUNNER_IMPLEMENTATION_COMMIT
            ) \
            or grant.get("runner_result_record_sha256") != RUNNER_RESULT_SHA256 \
            or grant.get("data_directory") != str(DATA_DIRECTORY) \
            or grant.get("state_directory") != str(STATE_DIRECTORY) \
            or grant.get("authorization") != expected_grant_authorization \
            or grant.get("precommit_audit_verdict") != "EXECUTION GO":
        raise PermissionError("future one-shot execution grant is wrong or unsealed")
    head, parent, result_sha = _validate_git_seal(grant, path)
    _validate_external_lineage_release(
        grant, head=head, parent=parent, result_sha=result_sha
    )
    return grant


def require_execution_authorization(program: dict[str, Any]) -> dict[str, Any]:
    """Require the fixed future grant; no caller-selected path is accepted."""
    return validate_future_grant(CANONICAL_GRANT, program)


def run_authorized_v4(program_path: Path) -> dict[str, Any]:
    """Sole public entry; it cannot receive a grant path or runtime override."""
    path = Path(program_path).resolve()
    if path != CANONICAL_PROGRAM.resolve():
        raise PermissionError("authorized v4 accepts only the canonical program path")
    program = load_canonical_authorization_program(verify_file_hashes=False)
    require_execution_authorization(program)
    program = load_canonical_authorization_program(verify_file_hashes=True)
    base_program = base_execution.load_canonical_program(verify_file_hashes=True)
    return base_execution._execute_into_reserved_directory(
        base_program,
        DATA_DIRECTORY,
        validation_runner=base_execution.run_validation,
        evaluator_factory=base_execution._actual_evaluator_factory(base_program),
        control_runner=base_execution.run_sealed_regression_control,
        capability_core=base_execution._run_fixed_capability_core,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_authorized_v4(arguments.program)
    print(
        f"[aggregate-evidence-smc-authorized-v1] status={result['status']} "
        f"outcome={result['outcome_kind']}"
    )


if __name__ == "__main__":
    main()
