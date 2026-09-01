import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
V1_PROGRAM = ROOT / "config/cf4_twompp_disjoint_tracer_pilot_program_v1.json"
V3_PROGRAM = ROOT / "config/cf4_twompp_disjoint_tracer_pilot_program_v3.json"
V3_SOURCE = ROOT / "src/cf4_twompp_disjoint_tracer_pilot_v3.py"
V3_FAILURE = (
    ROOT / "config/cf4_twompp_disjoint_tracer_pilot_v3_failure_audit_v1.json"
)
PROGRAM = ROOT / "config/cf4_twompp_disjoint_tracer_pilot_program_v4.json"
SOURCE = ROOT / "src/cf4_twompp_disjoint_tracer_pilot_v4.py"
RUNNER = ROOT / "scripts/run_cf4_twompp_disjoint_tracer_pilot_v4.sbatch"


def _load_v4():
    spec = importlib.util.spec_from_file_location("test_twompp_pilot_v4", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v4_exact_isolated_invocation_succeeds() -> None:
    completed = subprocess.run(
        ["/home/kjhan/miniconda3/bin/python3.13", "-I", "-P", str(SOURCE), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "run-pilot" in completed.stdout
    assert "validate-pilot" in completed.stdout


def test_v4_binds_v3_failure_evidence_and_source() -> None:
    program = json.loads(PROGRAM.read_text())
    assert program["frozen_v1_program"]["sha256"] == hashlib.sha256(
        V1_PROGRAM.read_bytes()
    ).hexdigest()
    assert program["failed_v3_program"]["sha256"] == hashlib.sha256(
        V3_PROGRAM.read_bytes()
    ).hexdigest()
    assert program["failed_v3_implementation"]["sha256"] == hashlib.sha256(
        V3_SOURCE.read_bytes()
    ).hexdigest()
    assert program["v3_failure_audit"]["sha256"] == hashlib.sha256(
        V3_FAILURE.read_bytes()
    ).hexdigest()
    assert program["implementation"]["sha256"] == hashlib.sha256(
        SOURCE.read_bytes()
    ).hexdigest()


def test_v4_changes_only_the_declared_coordinate_convention() -> None:
    module = _load_v4()
    merged, digest = module.load_program(PROGRAM)
    frozen = json.loads(V1_PROGRAM.read_text())
    assert digest == hashlib.sha256(PROGRAM.read_bytes()).hexdigest()
    for key in (
        "inputs",
        "cosmology",
        "no_double_counting",
        "tracer_design",
        "catalog_gate",
        "carrick_reference_gate",
        "future_information_gate_not_executed_by_pilot",
    ):
        assert merged[key] == frozen[key]
    old_gate = frozen["angular_completeness_gate"]
    new_gate = merged["angular_completeness_gate"]
    for key in (
        "HEALPix_NSIDE",
        "ordering",
        "pixel_values_range_inclusive",
        "median_absolute_difference_max_inclusive",
        "p95_absolute_difference_max_inclusive",
    ):
        assert new_gate[key] == old_gate[key]
    assert "equatorial ICRS RA/DEC" in new_gate["comparison"]
    correction = merged["coordinate_correction_v4"]
    assert correction["numerical_threshold_changed"] is False
    assert correction["CF4_exclusion_policy_changed"] is False
    assert correction["selection_or_population_changed"] is False
    assert correction["field_inference_added"] is False
    assert correction["automatic_follow_on_after_v4"] is False


def test_equatorial_directions_apply_no_frame_rotation() -> None:
    module = _load_v4()
    longitude, latitude = module.equatorial_directions(
        np.array([0.0, 90.0, 360.0]), np.array([-30.0, 0.0, 45.0])
    )
    assert np.allclose(longitude, [0.0, np.pi / 2.0, 0.0])
    assert np.allclose(latitude, [-np.pi / 6.0, 0.0, np.pi / 4.0])
    with pytest.raises(module.base.PilotError):
        module.equatorial_directions(np.array([0.0]), np.array([91.0]))


def test_failed_gate_metrics_are_preserved_before_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_v4()
    failed_result = {
        "schema": module.RESULT_SCHEMA,
        "status": module.STATUS_FAIL,
        "gates": {"example_gate": False},
        "failed_gates": ["example_gate"],
        "field_inference_executed": False,
    }
    monkeypatch.setattr(module, "collect_audit", lambda *args, **kwargs: failed_result)
    output = tmp_path / "pilot"
    with pytest.raises(module.base.PilotError, match="diagnostics preserved"):
        module.publish_pilot(PROGRAM, output, "0" * 40)
    staging = tmp_path / ".pilot.staging"
    assert not output.exists()
    assert {path.name for path in staging.iterdir()} == {"result.json", "FAILED"}
    saved = json.loads((staging / "result.json").read_text())
    assert saved["failed_gates"] == ["example_gate"]


def test_v3_failure_audit_makes_no_unearned_science_claim() -> None:
    audit = json.loads(V3_FAILURE.read_text())
    v3 = audit["preserved_attempts"][2]
    assert v3["Slurm_job_id"] == 329500
    assert v3["technical_gate_reached"]
    assert not v3["field_inference_reached"]
    assert not v3["exact_agreement_metrics_serialized"]
    assert audit["coordinate_diagnosis"]["verdict"] == (
        "ANGULAR_COMPLETENESS_COORDINATE_CONVENTION_MISMATCH"
    )
    assert all(value is False for value in audit["claims"].values())


def test_v4_runner_pins_resources_and_controller_boundary() -> None:
    text = RUNNER.read_text()
    assert f"program_sha={hashlib.sha256(PROGRAM.read_bytes()).hexdigest()}" in text
    assert f"source_sha={hashlib.sha256(SOURCE.read_bytes()).hexdigest()}" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --mem=1024M" in text
    assert "#SBATCH --time=00:15:00" in text
    assert '"$SUBMISSION_CONTROLLER" == syntax' in text
    assert 'host_name" != syntax' in text
    assert 'host_name" != syn101' in text
    assert "scripts/tripwire/**" in text
    assert "run-pilot" in text and "validate-pilot" in text
    assert "renameat2" not in text
    assert "pgrep" not in text
    assert "--requeue" not in text
