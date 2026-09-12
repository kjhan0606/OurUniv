import csv
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_twompp_metadata_consistent_subset_program_v1.json"
SOURCE = ROOT / "src/cf4_twompp_metadata_consistent_subset_v1.py"
RUNNER = ROOT / "scripts/run_cf4_twompp_metadata_consistent_subset_v1.sbatch"
PARENT_PROGRAM = ROOT / "config/cf4_twompp_completeness_outlier_audit_program_v1.json"
PARENT_SOURCE = ROOT / "src/cf4_twompp_completeness_outlier_audit_v1.py"
PARENT_RECORD = (
    ROOT / "config/cf4_twompp_completeness_outlier_audit_v1_failure_result_record.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("test_twompp_subset_v1", SOURCE)
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
    assert "run-validation" in completed.stdout
    assert "validate-subset" in completed.stdout


def test_program_binds_failure_lineage_and_implementation() -> None:
    program = json.loads(PROGRAM.read_text())
    assert program["parent_outlier_program"]["sha256"] == hashlib.sha256(
        PARENT_PROGRAM.read_bytes()
    ).hexdigest()
    assert program["parent_outlier_implementation"]["sha256"] == hashlib.sha256(
        PARENT_SOURCE.read_bytes()
    ).hexdigest()
    assert program["parent_failure_result_record"]["sha256"] == hashlib.sha256(
        PARENT_RECORD.read_bytes()
    ).hexdigest()
    assert program["implementation"]["sha256"] == hashlib.sha256(
        SOURCE.read_bytes()
    ).hexdigest()
    authorization = program["authorization"]
    assert authorization["subset_validation"]
    for key in (
        "field_inference",
        "likelihood_datum_consumption",
        "joint_information_budget",
        "resolution_increase",
        "IC_PM_HOP_RAMSES",
        "automatic_follow_on",
    ):
        assert authorization[key] is False


def test_frozen_subset_contract_has_no_adaptive_selection() -> None:
    frozen = json.loads(PROGRAM.read_text())["frozen_subset"]
    assert frozen["excluded_recno_count_exact"] == 319
    assert frozen["large_absolute_difference_strictly_greater_than"] == 0.05
    assert frozen["retained_tracer_count_exact"] == 36635
    assert frozen["retained_zero_exposure_count_exact"] == 0
    assert frozen["retained_large_difference_count_exact"] == 0
    assert sum(frozen["retained_six_population_counts"].values()) == 36635
    assert frozen["density_outcome_access_allowed_for_selection"] is False
    assert frozen["threshold_adaptation_allowed"] is False
    assert frozen["reintroduction_of_any_excluded_recno_allowed"] is False


def test_loader_requires_exact_single_parent_failure() -> None:
    module = _load_module()
    program, _, parent, outliers, digest = module.load_program(PROGRAM)
    assert digest == hashlib.sha256(PROGRAM.read_bytes()).hexdigest()
    assert program["status"] == "AUTHORIZED_FREEZE_AND_VALIDATE_36635_TRACER_SUBSET_ONLY"
    assert parent["failed_gates"] == ["eligible_zero_exposure_count_zero"]
    assert hashlib.sha256(outliers.read_bytes()).hexdigest() == program["frozen_outliers"][
        "sha256"
    ]


def test_exclusion_reader_is_unique_sorted_and_rule_bound(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "outliers.csv"
    path.write_text(
        "recno,absolute_difference,zero_exposure,assigned_map\n"
        "9,0.06,False,12_5\n"
        "2,0.00,True,12_5\n"
    )
    recnos, rows = module.read_frozen_exclusions(path, 2)
    assert recnos == [2, 9]
    assert set(rows) == {2, 9}
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "recno,absolute_difference,zero_exposure,assigned_map\n"
        "2,0.06,False,12_5\n"
        "2,0.07,False,12_5\n"
    )
    with pytest.raises(module.base.PilotError, match="not unique"):
        module.read_frozen_exclusions(duplicate, 2)


def test_canonical_exclusion_manifest_is_sorted_and_parent_bound() -> None:
    module = _load_module()
    raw = module.canonical_exclusion_csv_bytes([2, 9], "a" * 64)
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    assert [int(row["recno"]) for row in rows] == [2, 9]
    assert {row["parent_outliers_sha256"] for row in rows} == {"a" * 64}
    assert len({row["reason"] for row in rows}) == 1


def test_failed_validation_preserves_result_and_manifest_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    result = {
        "schema": module.RESULT_SCHEMA,
        "status": module.STATUS_FAIL,
        "gates": {"positive_exposure": False},
        "failed_gates": ["positive_exposure"],
        "field_inference_executed": False,
    }
    exclusions = b"recno,reason,parent_outliers_sha256\n1,x,y\n"
    monkeypatch.setattr(
        module, "collect_validation", lambda *args, **kwargs: (result, exclusions)
    )
    output = tmp_path / "validation"
    with pytest.raises(module.base.PilotError, match="diagnostics preserved"):
        module.publish_validation(PROGRAM, output, "0" * 40)
    stage = tmp_path / ".validation.staging"
    assert not output.exists()
    assert {path.name for path in stage.iterdir()} == {
        "result.json",
        "excluded_recnos.csv",
        "FAILED",
    }


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
    assert "run-validation" in text and "validate-subset" in text
    assert "renameat2" not in text
    assert "pgrep" not in text
    assert "--requeue" not in text
