#!/usr/bin/env python
"""Seal the terminal single-use V71 Path-B development result."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor
from hong2021_v71_development_gate import SCHEMA as DEVELOPMENT_SCHEMA
from hong2021_v71_ecc import (
    PROGRAM_FREEZE_COMMIT,
    PROGRAM_SHA256,
    authorize_parent_evidence,
    load_program,
    strict_json,
    validate_preflight,
)


SCHEMA = "hong2021-v71-terminal-sealed-result-v1"


def validate_development(
    program: dict[str, Any],
    path: Path,
    preflight_sha: str,
    repo: Path,
    commit: str,
) -> tuple[dict[str, Any], str]:
    expected = Path(program["output_roots"]["development"]) / "development_decision.json"
    if path.resolve() != expected.resolve():
        raise ValueError("V71 seal development path differs")
    decision = strict_json(path)
    passed = decision.get("development_pass")
    if (
        decision.get("schema") != DEVELOPMENT_SCHEMA
        or decision.get("status")
        != "complete_single_locked_V71_ECC_three_domain_development_gate"
        or decision.get("program_sha256") != PROGRAM_SHA256
        or decision.get("preflight_sha256") != preflight_sha
        or not isinstance(passed, bool)
        or canonical_digest(decision) != decision.get("decision_digest_sha256")
        or decision.get("single_locked_development_attempt") is not True
        or decision.get("diagnostic_control_used_for_selection") is not False
        or decision.get("fresh_train_only_V71_screen_available") is not False
        or decision.get("V70_gate_used_as_V71_selection_evidence") is not False
        or decision.get("training_or_gradient_performed_by_development") is not False
        or decision.get(
            "checkpoint_sampler_seed_member_count_metric_threshold_or_gate_tuned"
        )
        is not False
        or decision.get("independent_EAGLE_accessed") is not False
        or decision.get("independent_gate_locked") is not True
        or not _is_ancestor(repo, str(decision.get("gate_code_commit")), commit)
    ):
        raise ValueError("V71 seal development integrity differs")
    expected_branch = (
        (
            "V71_tail_preserving_ECC_is_development_sufficient",
            "seal_V71_and_await_new_explicit_user_approval_before_independent_EAGLE_access",
        )
        if passed
        else (
            "V71_tail_preserving_ECC_is_not_development_sufficient",
            "seal_the_failure_and_stop_before_independent_EAGLE_without_repeating_development_or_tuning_from_its_results",
        )
    )
    if (decision.get("classification"), decision.get("next")) != expected_branch:
        raise ValueError("V71 seal development branch differs")
    return decision, sha256_file(path)


def seal(
    program_path: Path,
    repo: Path,
    preflight_path: Path,
    preflight_sha: str,
    development_path: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    program = load_program(program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if not clean or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit):
        raise RuntimeError("V71 sealing requires a clean frozen worktree")
    evidence = authorize_parent_evidence(program, repo, commit)
    validate_preflight(preflight_path.resolve(), preflight_sha, repo, commit)
    development, development_sha = validate_development(
        program, development_path.resolve(), preflight_sha, repo, commit
    )
    passed = bool(development["development_pass"])
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "sealed_development_pass_waiting_new_explicit_EAGLE_approval"
            if passed
            else "sealed_development_failure_independent_gate_locked"
        ),
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "program_freeze_commit": PROGRAM_FREEZE_COMMIT,
        "sealing_code_commit": commit,
        "worktree_clean": clean,
        "preflight": str(preflight_path.resolve()),
        "preflight_sha256": preflight_sha,
        "v70_train_gate_sha256": evidence["v70_train_gate_sha256"],
        "v70_terminal_seal_sha256": evidence["v70_terminal_seal_sha256"],
        "fresh_train_only_V71_screen_available": False,
        "V70_gate_used_as_V71_selection_evidence": False,
        "development_decision": str(development_path.resolve()),
        "development_decision_sha256": development_sha,
        "development_accessed": True,
        "development_pass": passed,
        "classification": development["classification"],
        "next": development["next"],
        "single_development_attempt_consumed": True,
        "posthoc_training_sampling_metric_threshold_or_gate_tuning": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_EAGLE_accessed": False,
        "independent_gate_locked": True,
        "explicit_user_approval_required_before_EAGLE": passed,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--development-decision", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    program = load_program(args.program.resolve(), args.repo.resolve())
    if args.out.resolve() != Path(program["output_roots"]["terminal_seal"]).resolve():
        raise ValueError("V71 terminal seal output path differs")
    if args.out.exists():
        raise FileExistsError("V71 refuses an existing sealed result")
    result = seal(
        args.program, args.repo, args.preflight, args.preflight_sha256,
        args.development_decision,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
