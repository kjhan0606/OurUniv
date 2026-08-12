#!/usr/bin/env python
"""Seal the terminal V72 stage-A or stage-B outcome."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor
from hong2021_v72_sqt import (
    DECISION_SCHEMA,
    PROGRAM_FREEZE_COMMIT,
    PROGRAM_SHA256,
    authorize_parent_evidence,
    load_program,
    strict_json,
    validate_preflight,
)
from hong2021_v72_metadata_recovery import RECORD_SCHEMA as RECOVERY_RECORD_SCHEMA


SCHEMA = "hong2021-v72-terminal-sealed-result-v1"


def validate_stage_decision(
    program: dict[str, Any],
    path: Path,
    stage: str,
    preflight_sha: str,
    repo: Path,
    commit: str,
    stage_A_sha: str | None = None,
) -> tuple[dict[str, Any], str]:
    expected_path = Path(program["output_roots"][f"stage_{stage}"]) / "decision.json"
    if path.resolve() != expected_path.resolve():
        raise ValueError(f"V72 stage-{stage} decision path differs")
    decision = strict_json(path)
    passed = decision.get("stage_pass")
    if (
        decision.get("schema") != DECISION_SCHEMA
        or decision.get("status") != "complete_single_V72_SQT_stage_gate"
        or decision.get("stage") != stage
        or decision.get("program_sha256") != PROGRAM_SHA256
        or decision.get("preflight_sha256") != preflight_sha
        or not isinstance(passed, bool)
        or decision.get("candidate_selected") is not passed
        or canonical_digest(decision) != decision.get("decision_digest_sha256")
        or decision.get("single_use_stage") is not True
        or decision.get("diagnostic_arms_used_for_selection") is not False
        or decision.get("unequal_sample_global_maximum_used") is not False
        or decision.get(
            "candidate_strata_seed_member_sampler_metric_or_threshold_tuned"
        )
        is not False
        or decision.get("training_gradient_or_optimizer_performed") is not False
        or decision.get("stage_A_accessed") is not True
        or decision.get("stage_B_accessed") is not (stage == "B")
        or decision.get("independent_EAGLE_accessed") is not False
        or decision.get("independent_gate_locked") is not True
        or not _is_ancestor(repo, str(decision.get("gate_code_commit")), commit)
    ):
        raise ValueError(f"V72 stage-{stage} decision integrity differs")
    if stage == "A":
        expected_branch = (
            (
                "V72_SQT_passes_fresh_stage_A",
                "run_one_locked_stage_B_without_candidate_changes",
            )
            if passed else (
                "V72_SQT_is_not_fresh_screen_sufficient",
                "seal_and_stop_without_stage_B_or_EAGLE_access_or_posthoc_candidate_changes",
            )
        )
        if decision.get("stage_A_decision") is not None:
            raise ValueError("V72 stage A points to a stage-A parent")
    else:
        expected_branch = (
            (
                "V72_SQT_is_fresh_two_stage_sufficient",
                "seal_and_await_new_explicit_user_approval_before_independent_EAGLE",
            )
            if passed else (
                "V72_SQT_is_not_fresh_confirmation_sufficient",
                "seal_and_stop_before_EAGLE_without_repeating_either_stage",
            )
        )
        if decision.get("stage_A_decision_sha256") != stage_A_sha:
            raise ValueError("V72 stage B does not bind stage A")
    if (decision.get("classification"), decision.get("next")) != expected_branch:
        raise ValueError(f"V72 stage-{stage} decision branch differs")
    return decision, sha256_file(path)


def seal(
    program_path: Path,
    repo: Path,
    preflight_path: Path,
    preflight_sha: str,
    stage_A_path: Path,
    stage_B_path: Path | None,
    metadata_recovery_path: Path | None = None,
) -> dict[str, Any]:
    repo = repo.resolve()
    program = load_program(program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if not clean or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit):
        raise RuntimeError("V72 sealing requires a clean frozen worktree")
    evidence = authorize_parent_evidence(program, repo, commit)
    validate_preflight(preflight_path.resolve(), preflight_sha, repo, commit)
    expected_recovery = Path(program["output_roots"]["sequence"]) / (
        "stage_A_metadata_recovery.json"
    )
    recovery_sha: str | None = None
    if metadata_recovery_path is not None:
        if metadata_recovery_path.resolve() != expected_recovery.resolve():
            raise ValueError("V72 seal metadata recovery path differs")
        recovery = strict_json(metadata_recovery_path.resolve())
        recovery_sha = sha256_file(metadata_recovery_path.resolve())
        if (
            recovery.get("schema") != RECOVERY_RECORD_SCHEMA
            or recovery.get("status")
            != "complete_metadata_only_recovery_evaluation_may_resume"
            or recovery.get("v72_program_sha256") != PROGRAM_SHA256
            or recovery.get("all_nine_dataset_manifests_unchanged") is not True
            or recovery.get("sampling_repeated") is not False
            or recovery.get("stage_B_accessed") is not False
            or recovery.get("independent_EAGLE_accessed") is not False
            or canonical_digest(recovery) != recovery.get("decision_digest_sha256")
            or not _is_ancestor(repo, str(recovery.get("recovery_code_commit")), commit)
        ):
            raise ValueError("V72 metadata recovery record differs")
    first, first_sha = validate_stage_decision(
        program, stage_A_path.resolve(), "A", preflight_sha, repo, commit
    )
    second: dict[str, Any] | None = None
    second_sha: str | None = None
    if bool(first["stage_pass"]):
        if stage_B_path is None:
            raise ValueError("V72 passed stage A requires stage B before sealing")
        second, second_sha = validate_stage_decision(
            program, stage_B_path.resolve(), "B", preflight_sha,
            repo, commit, first_sha,
        )
        final_pass = bool(second["stage_pass"])
        classification = str(second["classification"])
        next_step = str(second["next"])
        status = (
            "sealed_two_stage_pass_waiting_explicit_EAGLE_approval"
            if final_pass
            else "sealed_stage_B_failure_independent_gate_locked"
        )
    else:
        if stage_B_path is not None or Path(program["output_roots"]["stage_B"]).exists():
            raise ValueError("V72 rejected stage A forbids stage B")
        final_pass = False
        classification = str(first["classification"])
        next_step = str(first["next"])
        status = "sealed_stage_A_failure_stage_B_unopened_independent_gate_locked"
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "program_freeze_commit": PROGRAM_FREEZE_COMMIT,
        "sealing_code_commit": commit,
        "worktree_clean": clean,
        "preflight": str(preflight_path.resolve()),
        "preflight_sha256": preflight_sha,
        "metadata_recovery": (
            str(metadata_recovery_path.resolve())
            if metadata_recovery_path is not None else None
        ),
        "metadata_recovery_sha256": recovery_sha,
        "v71_terminal_seal": evidence["v71_terminal_seal"],
        "v71_terminal_seal_sha256": evidence["v71_terminal_seal_sha256"],
        "stage_A_decision": str(stage_A_path.resolve()),
        "stage_A_decision_sha256": first_sha,
        "stage_A_pass": bool(first["stage_pass"]),
        "stage_B_decision": str(stage_B_path.resolve()) if stage_B_path else None,
        "stage_B_decision_sha256": second_sha,
        "stage_B_accessed": second is not None,
        "stage_B_pass": bool(second["stage_pass"]) if second is not None else None,
        "two_stage_pass": final_pass,
        "classification": classification,
        "next": next_step,
        "stage_A_single_use_consumed": True,
        "stage_B_single_use_consumed": second is not None,
        "candidate_strata_seed_member_sampler_metric_or_threshold_tuned": False,
        "training_gradient_or_optimizer_performed": False,
        "unequal_sample_global_maximum_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_EAGLE_accessed": False,
        "independent_gate_locked": True,
        "explicit_user_approval_required_before_EAGLE": final_pass,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--stage-A-decision", type=Path, required=True)
    parser.add_argument("--stage-B-decision", type=Path)
    parser.add_argument("--metadata-recovery", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    program = load_program(args.program.resolve(), args.repo.resolve())
    if args.out.resolve() != Path(program["output_roots"]["terminal_seal"]).resolve():
        raise ValueError("V72 terminal seal output path differs")
    if args.out.exists():
        raise FileExistsError("V72 refuses an existing terminal seal")
    result = seal(
        args.program, args.repo, args.preflight, args.preflight_sha256,
        args.stage_A_decision, args.stage_B_decision, args.metadata_recovery,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
