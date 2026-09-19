import json
from pathlib import Path

import pytest

from hong2021_eagle_goal_completion_verify import verify


REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "config/hong2021_eagle_goal_completion_audit.json"


def test_completion_audit_verifies_all_immutable_evidence() -> None:
    report = verify(AUDIT, REPO)
    assert report["status"] == "verified"
    assert report["evidence_files_verified"] == 10
    assert report["assertions_verified"] == 34
    assert report["terminal_action"] == (
        "stop_Hong_ML_path_and_return_to_CF4_constrained_realization_pipeline"
    )


def test_completion_audit_rejects_changed_evidence_hash(tmp_path: Path) -> None:
    audit = json.loads(AUDIT.read_text())
    audit["evidence"][0]["sha256"] = "0" * 64
    modified = tmp_path / "modified_audit.json"
    modified.write_text(json.dumps(audit))

    with pytest.raises(RuntimeError, match="evidence hash mismatch"):
        verify(modified, REPO)
