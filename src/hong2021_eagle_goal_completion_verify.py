#!/usr/bin/env python3
"""Verify the immutable evidence behind the Hong/EAGLE terminal audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "hong2021-eagle-goal-completion-audit-v1"
STATUS = "complete_negative_scientific_outcome_no_independent_gate_authorized"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_pointer(payload: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer!r}")
    value = payload
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(token)]
        else:
            value = value[token]
    return value


def verify(audit_path: Path, repo: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text())
    if audit.get("schema") != SCHEMA or audit.get("status") != STATUS:
        raise RuntimeError("unexpected completion-audit schema or status")

    interpretation = audit.get("interpretation", {})
    if interpretation.get("model_that_passes_EAGLE_claimed") is not False:
        raise RuntimeError("audit must not claim a model passed EAGLE")
    if interpretation.get("historical_EAGLE_failure_solved_claimed") is not False:
        raise RuntimeError("audit must not claim the historical EAGLE failure was solved")
    if interpretation.get("workflow_objective_completed") is not True:
        raise RuntimeError("workflow completion was not asserted")

    repo = repo.resolve()
    verified: list[dict[str, Any]] = []
    for row in audit.get("evidence", []):
        evidence_path = (repo / row["path"]).resolve()
        try:
            evidence_path.relative_to(repo)
        except ValueError as exc:
            raise RuntimeError(f"evidence escapes repository: {row['path']}") from exc
        actual_sha256 = sha256_file(evidence_path)
        if actual_sha256 != row["sha256"]:
            raise RuntimeError(
                f"evidence hash mismatch for {row['path']}: "
                f"expected {row['sha256']}, got {actual_sha256}"
            )
        payload = json.loads(evidence_path.read_text())
        for pointer, expected in row.get("assertions", {}).items():
            actual = json_pointer(payload, pointer)
            if actual != expected:
                raise RuntimeError(
                    f"evidence assertion failed for {row['path']} {pointer}: "
                    f"expected {expected!r}, got {actual!r}"
                )
        verified.append(
            {
                "path": row["path"],
                "sha256": actual_sha256,
                "assertions_verified": len(row.get("assertions", {})),
            }
        )

    outcome = audit.get("locked_sequence_outcome", {})
    required_false = (
        "development_candidate_passed",
        "Astrid_independent_gate_opened",
        "historical_EAGLE_reopened_after_V13",
        "grid_HOP_authorized",
        "RAMSES_promotion_authorized",
    )
    for key in required_false:
        if outcome.get(key) is not False:
            raise RuntimeError(f"locked-sequence outcome {key} must be false")
    if outcome.get("terminal_action") != (
        "stop_Hong_ML_path_and_return_to_CF4_constrained_realization_pipeline"
    ):
        raise RuntimeError("unexpected terminal action")

    return {
        "schema": "hong2021-eagle-goal-completion-verification-v1",
        "status": "verified",
        "audit_sha256": sha256_file(audit_path),
        "evidence_files_verified": len(verified),
        "assertions_verified": sum(row["assertions_verified"] for row in verified),
        "evidence": verified,
        "terminal_action": outcome["terminal_action"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("config/hong2021_eagle_goal_completion_audit.json"),
    )
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(verify(args.audit, args.repo), indent=2))


if __name__ == "__main__":
    main()
