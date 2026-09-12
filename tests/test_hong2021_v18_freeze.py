import json
import subprocess
from pathlib import Path

import pytest

from hong2021_v14_edm import V15_E3_SCHEMA
from hong2021_v14_freeze import sha256
from hong2021_v18_edm import FROZEN_REGISTRY_SHA256
from hong2021_v18_freeze import EXACT_ONE_SHOT_COMMAND, SCHEMA, verify_seal
from hong2021_v18_init import EXPECTED_MODE_COUNTS_BY_GRID, SCHEMA as INIT_SCHEMA


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _initialization() -> dict:
    return {
        "registry_sha256": FROZEN_REGISTRY_SHA256,
        "schema": INIT_SCHEMA,
        "expected_mode_counts_by_grid": {
            str(grid): list(counts)
            for grid, counts in EXPECTED_MODE_COUNTS_BY_GRID.items()
        },
        "inference_initialization_mapping": {
            "astrid_grid": 80,
            "astrid_mode_counts": list(EXPECTED_MODE_COUNTS_BY_GRID[80]),
            "astrid_data_used_in_mapping": False,
            "inference_data_remeasurement_forbidden": True,
            "free_parameters_added": 0,
        },
    }


def test_v18_committed_seal_requires_direct_single_file_commit(tmp_path) -> None:
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
    external.write_bytes(b"v18-model")
    provenance.write_text('{"development_pass": true}\n')
    seal = repo / "config" / "seal.json"
    seal.parent.mkdir()
    record = {
        "schema": SCHEMA,
        "code_commit_before_seal_record": code_commit,
        "development_gate": {"passed": True},
        "model_invariants": {
            "edm_schema": V15_E3_SCHEMA,
            "decoder_upsampling": "nearest",
        },
        "initialization": _initialization(),
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
    git(repo, "commit", "-m", "commit V18 seal")
    assert verify_seal(seal, repo=repo, require_unopened=True)["schema"] == SCHEMA
    provenance.write_text('{"development_pass": false}\n')
    with pytest.raises(RuntimeError, match="hash mismatch|changed or vanished"):
        verify_seal(seal, repo=repo)


def test_v18_seal_rejects_missing_80_grid_mapping(tmp_path) -> None:
    initialization = _initialization()
    del initialization["expected_mode_counts_by_grid"]["80"]
    record = {
        "schema": SCHEMA,
        "development_gate": {"passed": True},
        "model_invariants": {
            "edm_schema": V15_E3_SCHEMA,
            "decoder_upsampling": "nearest",
        },
        "initialization": initialization,
    }
    path = tmp_path / "seal.json"
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="both frozen Fourier grids"):
        verify_seal(path, repo=tmp_path, require_committed=False)
