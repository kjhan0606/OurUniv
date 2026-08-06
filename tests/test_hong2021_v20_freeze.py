from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hong2021_astrid_seal import verify_astrid_seal
from hong2021_v14_edm import V20_E8_SCHEMA
from hong2021_v14_freeze import sha256
from hong2021_v20_edm import FROZEN_REGISTRY_SHA256, P_MEAN, P_STD
from hong2021_v20_freeze import EXACT_ONE_SHOT_COMMAND, SCHEMA, verify_seal


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _record(tmp_path: Path, repo: Path, code_commit: str) -> dict:
    protocol = repo / "src" / "protocol.py"
    external = tmp_path / "selected.pt"
    provenance = tmp_path / "decision.json"
    external.write_bytes(b"v20-model")
    provenance.write_text('{"development_pass": true}\n')
    return {
        "schema": SCHEMA,
        "code_commit_before_seal_record": code_commit,
        "development_gate": {"passed": True},
        "model_invariants": {
            "edm_schema": V20_E8_SCHEMA,
            "decoder_upsampling": "nearest",
            "edm_p_mean": P_MEAN,
            "edm_p_std": P_STD,
            "edm_p_mean_mode": "sigma_data_fraction",
        },
        "initialization": {
            "registry_sha256": FROZEN_REGISTRY_SHA256,
            "expected_mode_counts_by_grid": {
                "64": [146, 3596, 25296, 233105],
                "80": [250, 6872, 49928, 454949],
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
        "astrid_preopen": {
            "root": str(tmp_path / "Astrid"),
            "path_accessed_during_v20_development_or_seal_creation": False,
        },
        "one_shot": {"command": EXACT_ONE_SHOT_COMMAND},
    }


def test_v20_seal_is_direct_single_file_commit_and_dispatchable(tmp_path) -> None:
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
    seal = repo / "config" / "seal.json"
    seal.parent.mkdir()
    seal.write_text(json.dumps(_record(tmp_path, repo, code_commit), indent=2) + "\n")
    git(repo, "add", "config/seal.json")
    git(repo, "commit", "-m", "commit V20 seal")
    assert verify_seal(seal, repo=repo, require_unopened=True)["schema"] == SCHEMA
    assert verify_astrid_seal(
        seal, repo=repo, require_committed=True, require_unopened=True
    )["schema"] == SCHEMA


def test_v20_seal_rejects_relative_noise_mutation(tmp_path) -> None:
    record = {
        "schema": SCHEMA,
        "development_gate": {"passed": True},
        "model_invariants": {
            "edm_schema": V20_E8_SCHEMA,
            "edm_p_mean": P_MEAN,
            "edm_p_std": P_STD,
            "edm_p_mean_mode": "absolute",
        },
        "initialization": {
            "registry_sha256": FROZEN_REGISTRY_SHA256,
            "expected_mode_counts_by_grid": {
                "64": [146, 3596, 25296, 233105],
                "80": [250, 6872, 49928, 454949],
            },
        },
    }
    path = tmp_path / "seal.json"
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="relative p_mean mode"):
        verify_seal(path, repo=tmp_path, require_committed=False)
