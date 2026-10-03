import json
import subprocess
from pathlib import Path

import pytest

from hong2021_v14_freeze import sha256
from hong2021_v17_freeze import EXACT_ONE_SHOT_COMMAND, SCHEMA, verify_seal


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_v17_committed_seal_requires_direct_single_file_commit(tmp_path) -> None:
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
    external.write_bytes(b"v17-model")
    provenance.write_text('{"development_pass": true}\n')
    seal = repo / "config" / "seal.json"
    seal.parent.mkdir()
    record = {
        "schema": SCHEMA,
        "code_commit_before_seal_record": code_commit,
        "development_gate": {"passed": True},
        "model_invariants": {
            "decoder_upsampling": "nearest",
            "denoising_loss": {
                "fft_norm": "ortho",
                "mode_counts": [146, 3596, 25296, 233105],
            },
        },
        "artifacts": {"edm": {
            "path": str(external), "bytes": external.stat().st_size,
            "sha256": sha256(external),
        }},
        "provenance": {"decision": {
            "path": str(provenance), "bytes": provenance.stat().st_size,
            "sha256": sha256(provenance),
        }},
        "tracked_protocol_files": [{
            "path": "src/protocol.py", "bytes": protocol.stat().st_size,
            "sha256": sha256(protocol),
        }],
        "astrid_preopen": {"root": str(tmp_path / "Astrid")},
        "one_shot": {"command": EXACT_ONE_SHOT_COMMAND},
    }
    seal.write_text(json.dumps(record, indent=2) + "\n")
    git(repo, "add", "config/seal.json")
    git(repo, "commit", "-m", "commit V17 seal")
    assert verify_seal(seal, repo=repo, require_unopened=True)["schema"] == SCHEMA
    provenance.write_text('{"development_pass": false}\n')
    with pytest.raises(RuntimeError, match="hash mismatch|changed or vanished"):
        verify_seal(seal, repo=repo)


def test_v17_seal_rejects_wrong_loss_invariant(tmp_path) -> None:
    record = {
        "schema": SCHEMA,
        "development_gate": {"passed": True},
        "model_invariants": {
            "decoder_upsampling": "nearest",
            "denoising_loss": {"fft_norm": "backward", "mode_counts": []},
        },
    }
    path = tmp_path / "seal.json"
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="loss invariant"):
        verify_seal(path, repo=tmp_path, require_committed=False)
