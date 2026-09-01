import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_twompp_completeness_outlier_audit_program_v1.json"
SOURCE = ROOT / "src/cf4_twompp_completeness_outlier_audit_v1.py"
RUNNER = ROOT / "scripts/run_cf4_twompp_completeness_outlier_audit_v1.sbatch"
V4_PROGRAM = ROOT / "config/cf4_twompp_disjoint_tracer_pilot_program_v4.json"
V4_SOURCE = ROOT / "src/cf4_twompp_disjoint_tracer_pilot_v4.py"
V4_RECORD = ROOT / "config/cf4_twompp_disjoint_tracer_pilot_v4_result_record_v1.json"
FAILURE_RECORD = (
    ROOT / "config/cf4_twompp_completeness_outlier_audit_v1_failure_result_record.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("test_twompp_outlier_v1", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_isolated_invocation_succeeds() -> None:
    completed = subprocess.run(
        ["/home/kjhan/miniconda3/bin/python3.13", "-I", "-P", str(SOURCE), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "run-audit" in completed.stdout
    assert "validate-audit" in completed.stdout


def test_program_binds_parent_and_implementation() -> None:
    program = json.loads(PROGRAM.read_text())
    assert program["parent_v4_program"]["sha256"] == hashlib.sha256(
        V4_PROGRAM.read_bytes()
    ).hexdigest()
    assert program["parent_v4_implementation"]["sha256"] == hashlib.sha256(
        V4_SOURCE.read_bytes()
    ).hexdigest()
    assert program["parent_v4_result_record"]["sha256"] == hashlib.sha256(
        V4_RECORD.read_bytes()
    ).hexdigest()
    assert program["implementation"]["sha256"] == hashlib.sha256(
        SOURCE.read_bytes()
    ).hexdigest()
    authorization = program["authorization"]
    assert authorization["technical_audit"]
    for key in (
        "field_inference",
        "likelihood_datum_consumption",
        "joint_information_budget",
        "resolution_increase",
        "IC_PM_HOP_RAMSES",
        "automatic_follow_on",
    ):
        assert authorization[key] is False


def test_gates_are_frozen_before_execution() -> None:
    program = json.loads(PROGRAM.read_text())
    gates = program["frozen_gates"]
    assert gates["zero_exposure_max_count_inclusive"] == 0
    assert gates["large_absolute_difference_strictly_greater_than"] == 0.05
    assert gates["large_difference_fraction_max_inclusive"] == 0.01
    assert gates["usable_tracer_count_min_inclusive"] == 20000
    assert gates["all_six_usable_populations_must_be_nonempty"]
    assert program["implementation_contract"]["no_threshold_adaptation_after_result"]


def test_loader_reuses_exact_passing_v4_parent() -> None:
    module = _load_module()
    program, effective, parent, digest = module.load_program(PROGRAM)
    assert digest == hashlib.sha256(PROGRAM.read_bytes()).hexdigest()
    assert program["status"] == "AUTHORIZED_TECHNICAL_OUTLIER_ZERO_EXPOSURE_AUDIT_ONLY"
    assert effective["angular_completeness_gate"]["coordinate_convention_source"]
    assert parent["status"] == module.v4.STATUS_PASS
    assert parent["failed_gates"] == []


def test_population_counts_are_an_exact_six_way_partition() -> None:
    module = _load_module()
    selected = np.ones(6, dtype=bool)
    apparent = np.array([0, 0, 0, 1, 1, 1])
    absolute = np.array([0, 1, 2, 0, 1, 2])
    counts = module._population_counts(
        selected, apparent, absolute, [-25.0, -24.0, -23.0, -21.0]
    )
    assert len(counts) == 6
    assert sum(counts.values()) == 6
    assert set(counts.values()) == {1}


def test_failed_metrics_and_outliers_are_preserved_before_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    failed_result = {
        "schema": module.RESULT_SCHEMA,
        "status": module.STATUS_FAIL,
        "gates": {"zero_exposure": False},
        "failed_gates": ["zero_exposure"],
        "field_inference_executed": False,
    }
    rows = [
        {
            "recno": 1,
            "RA_deg": 0.0,
            "DEC_deg": 0.0,
            "Ksmag": 10.0,
            "apparent_bin": 0,
            "absolute_bin": 0,
            "assigned_map": "11_5",
            "pixel_RING": 0,
            "map_exposure": 0.0,
            "catalog_mark": 1.0,
            "absolute_difference": 1.0,
            "zero_exposure": True,
            "nearest_central_or_neighbor_difference": 1.0,
            "neighbor_reconciled_at_threshold": False,
        }
    ]
    monkeypatch.setattr(
        module, "collect_audit", lambda *args, **kwargs: (failed_result, rows)
    )
    output = tmp_path / "audit"
    with pytest.raises(module.base.PilotError, match="diagnostics preserved"):
        module.publish_audit(PROGRAM, output, "0" * 40)
    staging = tmp_path / ".audit.staging"
    assert not output.exists()
    assert {path.name for path in staging.iterdir()} == {
        "result.json",
        "outliers.csv",
        "FAILED",
    }
    assert "recno" in (staging / "outliers.csv").read_text()


def test_runner_pins_resources_and_controller_boundary() -> None:
    text = RUNNER.read_text()
    assert f"program_sha={hashlib.sha256(PROGRAM.read_bytes()).hexdigest()}" in text
    assert f"source_sha={hashlib.sha256(SOURCE.read_bytes()).hexdigest()}" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --mem=768M" in text
    assert "#SBATCH --time=00:15:00" in text
    assert '"$SUBMISSION_CONTROLLER" == syntax' in text
    assert 'host_name" != syntax' in text
    assert 'host_name" != syn101' in text
    assert "scripts/tripwire/**" in text
    assert "run-audit" in text and "validate-audit" in text
    assert "renameat2" not in text
    assert "pgrep" not in text
    assert "--requeue" not in text


def test_failure_record_binds_preserved_metrics_and_stops_joint_execution() -> None:
    record = json.loads(FAILURE_RECORD.read_text())
    assert record["execution"]["Slurm_job_id"] == 329539
    assert record["execution"]["failure_mode"] == (
        "intentional_fail_closed_scientific_input_gate"
    )
    for key in ("result", "outliers", "FAILED", "stdout", "stderr"):
        binding = record["preserved_failure_artifacts"][key]
        raw = Path(binding["path"]).read_bytes()
        assert len(raw) == binding["bytes"]
        assert hashlib.sha256(raw).hexdigest() == binding["sha256"]
    gate = record["gate_result"]
    assert gate["zero_exposure_count"] == 1
    assert gate["large_difference_count"] == 319
    assert gate["large_difference_fraction"] <= gate["large_difference_fraction_limit"]
    assert gate["usable_after_frozen_outlier_exclusion_count"] == 36635
    assert all(count > 0 for count in gate["usable_six_population_counts"].values())
    decision = record["scientific_decision"]
    assert decision["unfiltered_36954_tracer_count_likelihood"] == "NO_GO"
    assert decision["joint_information_budget_execution"] == "STOPPED_NOT_RUN"
    assert decision["density_field_inference"] == "NOT_RUN"
    assert record["recommended_redesign"]["approval_required"] is True
