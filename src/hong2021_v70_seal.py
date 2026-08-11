#!/usr/bin/env python
"""Seal the terminal V70 train-gate or locked-development outcome."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor
from hong2021_v70_development_gate import SCHEMA as DEVELOPMENT_SCHEMA
from hong2021_v70_development_sample import PROGRAM_SHA256, load_program
from hong2021_v70_train_gate import SCHEMA as TRAIN_GATE_SCHEMA


SCHEMA = "hong2021-v70-terminal-sealed-result-v1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def validate_train_gate(
    program: dict[str, Any], path: Path, repo: Path, commit: str
) -> tuple[dict[str, Any], str]:
    parent = program["parent_programs"]
    if path.resolve() != Path(parent["required_train_gate_decision"]).resolve():
        raise ValueError("V70 seal train-gate path differs")
    decision = _json(path)
    if (
        decision.get("schema") != TRAIN_GATE_SCHEMA
        or decision.get("status") != parent["required_train_gate_status"]
        or decision.get("program_sha256")
        != parent["v70_train_gate_program_sha256"]
        or canonical_digest(decision) != decision.get("decision_digest_sha256")
        or decision.get("validation_accessed") is not False
        or decision.get("development_accessed") is not False
        or decision.get("historical_EAGLE_accessed") is not False
        or decision.get("independent_EAGLE_accessed") is not False
        or decision.get("independent_gate_locked") is not True
        or not _is_ancestor(repo, str(decision.get("code_commit")), commit)
    ):
        raise ValueError("V70 seal train-gate integrity differs")
    selected = decision.get("candidate_selected")
    if selected is not decision.get("train_mechanism_pass") or not isinstance(
        selected, bool
    ):
        raise ValueError("V70 seal train-gate selection differs")
    expected = (
        (
            parent["required_classification"],
            parent["required_next"],
        )
        if selected
        else (
            "query_aligned_latent_spatial_score_does_not_learn_cross_domain_joint_structure",
            "stop_before_development_without_posthoc_training_sampling_or_gate_tuning",
        )
    )
    if (decision.get("classification"), decision.get("next")) != expected:
        raise ValueError("V70 seal train-gate branch differs")
    return decision, sha256_file(path)


def validate_development(
    program: dict[str, Any], path: Path, train_gate_sha: str,
    repo: Path, commit: str,
) -> tuple[dict[str, Any], str]:
    if path.resolve() != Path(program["output"]).resolve() / "development_decision.json":
        raise ValueError("V70 seal development path differs")
    decision = _json(path)
    passed = decision.get("development_pass")
    if (
        decision.get("schema") != DEVELOPMENT_SCHEMA
        or decision.get("status")
        != "complete_single_locked_three_domain_development_gate"
        or decision.get("program_sha256") != PROGRAM_SHA256
        or decision.get("train_mechanism_gate_sha256") != train_gate_sha
        or decision.get("train_mechanism_pass") is not True
        or not isinstance(passed, bool)
        or canonical_digest(decision) != decision.get("decision_digest_sha256")
        or decision.get("single_locked_development_attempt") is not True
        or decision.get("diagnostic_control_used_for_selection") is not False
        or decision.get("training_or_gradient_performed_by_development") is not False
        or decision.get(
            "checkpoint_sampler_seed_member_count_threshold_or_gate_tuned"
        )
        is not False
        or decision.get("independent_EAGLE_accessed") is not False
        or decision.get("independent_gate_locked") is not True
        or not _is_ancestor(repo, str(decision.get("gate_code_commit")), commit)
    ):
        raise ValueError("V70 seal development integrity differs")
    expected = (
        (
            "V70_is_development_sufficient",
            "seal_V70_and_await_explicit_user_approval_before_independent_EAGLE_access",
        )
        if passed
        else (
            "V70_joint_spatial_model_is_not_development_sufficient",
            "seal_the_failure_and_stop_before_independent_EAGLE_without_sampler_threshold_or_model_tuning",
        )
    )
    if (decision.get("classification"), decision.get("next")) != expected:
        raise ValueError("V70 seal development branch differs")
    return decision, sha256_file(path)


def seal(
    program_path: Path,
    repo: Path,
    train_gate_path: Path,
    development_path: Path | None,
) -> dict[str, Any]:
    repo = repo.resolve()
    program = load_program(program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V70 sealing requires a clean worktree")
    train, train_sha = validate_train_gate(
        program, train_gate_path.resolve(), repo, commit
    )
    selected = bool(train["candidate_selected"])
    development: dict[str, Any] | None = None
    development_sha: str | None = None
    if selected:
        if development_path is None:
            raise ValueError("V70 selected train gate requires development decision")
        development, development_sha = validate_development(
            program, development_path.resolve(), train_sha, repo, commit
        )
        terminal_pass = bool(development["development_pass"])
        classification = str(development["classification"])
        next_step = str(development["next"])
        status = (
            "sealed_development_pass_waiting_explicit_EAGLE_approval"
            if terminal_pass
            else "sealed_development_failure_independent_gate_locked"
        )
    else:
        if development_path is not None:
            raise ValueError("V70 rejected train gate forbids development decision")
        terminal_pass = False
        classification = str(train["classification"])
        next_step = str(train["next"])
        status = "sealed_train_gate_rejection_development_not_accessed"
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "sealing_code_commit": commit,
        "worktree_clean": clean,
        "train_gate": str(train_gate_path.resolve()),
        "train_gate_sha256": train_sha,
        "train_mechanism_pass": selected,
        "development_decision": (
            str(development_path.resolve()) if development_path is not None else None
        ),
        "development_decision_sha256": development_sha,
        "development_accessed": selected,
        "development_pass": terminal_pass if selected else None,
        "classification": classification,
        "next": next_step,
        "posthoc_training_sampling_threshold_or_gate_tuning": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_EAGLE_accessed": False,
        "independent_gate_locked": True,
        "explicit_user_approval_required_before_EAGLE": bool(
            selected and terminal_pass
        ),
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--development-decision", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("V70 refuses an existing sealed result")
    result = seal(
        args.program, args.repo, args.train_gate, args.development_decision
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
