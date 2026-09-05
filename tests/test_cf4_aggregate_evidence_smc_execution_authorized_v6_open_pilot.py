import copy
from pathlib import Path

import pytest

import cf4_aggregate_evidence_smc_execution_authorized_v6_open_pilot as pilot


def test_absent_future_authorization_refuses_before_all_runtime_writes():
    before = {path: path.exists() for path in (pilot.RECEIPTS, pilot.PILOT, pilot.DATA_FORBIDDEN, pilot.STATE_FORBIDDEN)}
    with pytest.raises(PermissionError, match="future pilot-execution authorization record is absent"):
        pilot.run_authorized_v6_open_pilot(pilot.PROGRAM, "LagEunha.cluster")
    assert {path: path.exists() for path in before} == before == {path: False for path in before}


def test_hash_pair_path_and_host_fail_closed(monkeypatch, tmp_path):
    with pytest.raises(PermissionError, match="canonical"):
        pilot.run_authorized_v6_open_pilot(tmp_path / "wrong_v6_open_program.json", "lageunha")
    with pytest.raises(PermissionError, match="host gate"):
        pilot.run_authorized_v6_open_pilot(pilot.PROGRAM, "syn101")
    original_json = pilot._json

    def rebound(path, label):
        value = original_json(path, label)
        if label == "open grant":
            value = copy.deepcopy(value)
            value["external_release"]["release_id"] = "a" * 64
        return value

    monkeypatch.setattr(pilot, "_json", rebound)
    with pytest.raises(PermissionError, match="identity"):
        pilot.run_authorized_v6_open_pilot(pilot.PROGRAM, "lageunha")
    monkeypatch.undo()
    with pytest.raises(PermissionError, match="not canonical"):
        pilot._canonical("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v5", pilot.PILOT, "pilot")


def test_one_shot_lifecycle_is_gated_before_atomic_receipt_mutation():
    program = pilot.load_program()
    with pytest.raises(PermissionError, match="future pilot-execution authorization record is absent"):
        pilot.create_pilot_receipt(program, "a" * 64)
    source = pilot.Path(pilot.__file__).read_text()
    assert "os.link(OPEN_RELEASE, receipt / \"release.anchor\")" in source
    assert "os.O_WRONLY | os.O_CREAT | os.O_EXCL" in source
    assert "schedule_manifest.json" in source and "os.chmod(target, 0o444)" in source
    assert not pilot.RECEIPTS.exists() and not pilot.PILOT.exists()


def test_lifecycle_and_terminal_status_mapping_are_scientific_or_invalid_only():
    design = __import__("json").loads(pilot.DESIGN.read_text())
    assert design["pilot_lifecycle"]["pilot_output_root_only"] is True
    assert design["pilot_lifecycle"]["pilot_close_and_dispose_required_before_any_production"] is True
    assert pilot.map_terminal_status("complete_pass_production_smc")["outcome_kind"] == "scientific"
    assert pilot.map_terminal_status("complete_scientific_fail_production_smc")["failure_class"] == "scientific"
    assert pilot.map_terminal_status("missing")=={"status":"FAILED","outcome_kind":"invalid","failure_class":"invalid_provenance_or_execution"}
    assert pilot.AUTH["production_stage_authorized"] is False and pilot.AUTH["cache_population_authorized"] is False
