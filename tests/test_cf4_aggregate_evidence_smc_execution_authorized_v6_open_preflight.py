import copy
from pathlib import Path

import pytest

import cf4_aggregate_evidence_smc_execution_authorized_v6_open_preflight as preflight


def test_valid_pilot_preflight_is_read_only_and_hostname_is_ascii_normalized():
    before = {root: root.exists() for root in preflight.RUNTIME_ROOTS}
    value = preflight.run_preflight_v6_open(preflight.PROGRAM, "pilot", "LagEunha.cluster")
    assert value["status"] == "complete_preflight_only_v6_open"
    assert value["hostname"] == "lageunha"
    assert {root: root.exists() for root in preflight.RUNTIME_ROOTS} == before == {root: False for root in preflight.RUNTIME_ROOTS}


def test_absent_grant_and_rebound_pair_fail_closed(monkeypatch):
    monkeypatch.setattr(preflight, "GRANT", preflight.ROOT / "config/missing_v6_open_grant.json")
    with pytest.raises(PermissionError, match="absent"):
        preflight._check_pair()
    monkeypatch.undo()
    original = preflight._read_json

    def rebound(path, label):
        value = original(path, label)
        if label == "grant":
            value = copy.deepcopy(value)
            value["external_release"]["release_id"] = "a" * 64
        return value

    monkeypatch.setattr(preflight, "_read_json", rebound)
    with pytest.raises(PermissionError, match="binding"):
        preflight.run_preflight_v6_open(preflight.PROGRAM, "pilot", "LAGEUNHA")


def test_short_hash_mode_stage_and_path_failures(monkeypatch, tmp_path):
    with pytest.raises(PermissionError, match="full SHA-256"):
        preflight._full_hash("abc", "test")
    original_mode = preflight._require_mode

    def wrong_release_mode(path, expected, label):
        if path == preflight.RELEASE:
            raise PermissionError("release mode is not 0444")
        return original_mode(path, expected, label)

    monkeypatch.setattr(preflight, "_require_mode", wrong_release_mode)
    with pytest.raises(PermissionError, match="mode"):
        preflight.run_preflight_v6_open(preflight.PROGRAM, "pilot", "lageunha")
    monkeypatch.undo()
    with pytest.raises(PermissionError, match="only pilot"):
        preflight.run_preflight_v6_open(preflight.PROGRAM, "production", "lageunha")
    with pytest.raises(PermissionError, match="canonical"):
        preflight.run_preflight_v6_open(tmp_path / "wrong_v6_open_program.json", "pilot", "lageunha")
    with pytest.raises(PermissionError, match="host gate"):
        preflight.run_preflight_v6_open(preflight.PROGRAM, "pilot", "syn101")


def test_pins_are_v6_open_only_and_runtime_authorization_is_false():
    design, program, lift = preflight._check_config_hashes()
    preflight._check_program_pins(design, program, lift)
    assert all(preflight.AUTH[key] is False for key in ("pilot_execution_authorized", "production_stage_authorized", "receipt_creation_authorized", "cache_population_authorized", "downstream_execution_authorized", "automatic_follow_on_authorized"))
    assert design["fixed_science"]["worker_processes"] == 8
    assert design["fixed_science"]["threads_per_worker"] == 1
    assert not any("v4" in str(value).lower() or "v5" in str(value).lower() for value in program["paths"].values())
