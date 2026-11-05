import json
import subprocess
from pathlib import Path

import pytest

from hong2021_v14_freeze import EXACT_ONE_SHOT_COMMAND, SCHEMA, sha256, verify_seal


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_committed_seal_verifies_and_detects_external_artifact_change(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "test")
    protocol = repo / "src" / "protocol.py"
    protocol.parent.mkdir()
    protocol.write_text("frozen = True\n")
    git(repo, "add", "src/protocol.py")
    git(repo, "commit", "-m", "freeze code")
    code_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()

    external = tmp_path / "selected.pt"
    provenance = tmp_path / "decision.json"
    external.write_bytes(b"selected-model")
    provenance.write_text('{"development_pass": true}\n')
    astrid = tmp_path / "Astrid"
    seal = repo / "config" / "seal.json"
    seal.parent.mkdir()
    record = {
        "schema": SCHEMA,
        "code_commit_before_seal_record": code_commit,
        "development_gate": {"passed": True},
        "artifacts": {
            "edm": {
                "path": str(external), "bytes": external.stat().st_size,
                "sha256": sha256(external),
            }
        },
        "provenance": {
            "decision": {
                "path": str(provenance), "bytes": provenance.stat().st_size,
                "sha256": sha256(provenance),
            }
        },
        "tracked_protocol_files": [{
            "path": "src/protocol.py", "bytes": protocol.stat().st_size,
            "sha256": sha256(protocol),
        }],
        "astrid_preopen": {"root": str(astrid)},
        "one_shot": {"command": EXACT_ONE_SHOT_COMMAND},
    }
    seal.write_text(json.dumps(record, indent=2) + "\n")
    git(repo, "add", "config/seal.json")
    git(repo, "commit", "-m", "commit exact seal")

    verified = verify_seal(
        seal, repo=repo, require_committed=True, require_unopened=True
    )
    assert verified["schema"] == SCHEMA
    external.write_bytes(b"changed-model")
    with pytest.raises(RuntimeError, match="hash mismatch|changed or vanished"):
        verify_seal(seal, repo=repo, require_committed=True)


def test_unopened_check_rejects_any_astrid_file(tmp_path):
    # The full committed-record behavior is covered above; this asserts that a
    # pre-existing independent file is visible to the same filesystem scan.
    from hong2021_v14_freeze import astrid_files

    root = tmp_path / "Astrid"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "snapshot.hdf5").write_bytes(b"do not open")
    assert astrid_files(root) == [str((root / "raw" / "snapshot.hdf5").resolve())]
