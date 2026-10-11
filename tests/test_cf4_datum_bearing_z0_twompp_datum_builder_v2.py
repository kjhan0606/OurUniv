import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_datum_bearing_z0_twompp_datum_builder_program_v2.json"
FAILURE = ROOT / "config/cf4_datum_bearing_z0_twompp_datum_builder_v1_failure_result_record.json"
SOURCE = ROOT / "src/cf4_datum_bearing_z0_twompp_datum_builder_v2.py"
RUNNER = ROOT / "scripts/run_cf4_datum_bearing_z0_twompp_datum_builder_v2.sbatch"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_cf4_z0_datum_builder_v2", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_failure_record_is_pre_datum_and_scientifically_inconclusive():
    failure = json.loads(FAILURE.read_text())
    assert failure["status"] == "FAILED_BEFORE_CATALOG_DESERIALIZATION_WRONG_FROZEN_LOADER_BINDING"
    assert failure["execution"]["Slurm_job_id"] == 329827
    assert failure["failure"]["actual_VizieR_header_names"] == ["_RA", "_DE"]
    assert failure["failure"]["science_input_read_as_datum"] is False
    assert failure["failure"]["gate_evaluated"] is False
    assert failure["failure"]["public_output_created"] is False
    assert failure["scientific_interpretation"]["Phase_A_science_result"] == "NOT_EVALUATED"


def test_v2_program_binds_failure_and_changes_only_loader_lineage():
    program = json.loads(PROGRAM.read_text())
    assert program["schema"] == "ouruniv-cf4-datum-bearing-z0-twompp-datum-builder-program-v2"
    for binding in program["bindings"].values():
        path = Path(binding["path"])
        if not path.is_absolute():
            path = ROOT / path
        assert path.stat().st_size == binding["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    correction = program["implementation_correction"]
    assert correction["failed_job_id"] == 329827
    assert correction["replacement_loader"].endswith("_v3.py")
    for key in (
        "input_changed",
        "row_selection_changed",
        "split_changed",
        "selection_changed",
        "count_assignment_changed",
        "gate_changed",
        "resource_changed",
        "science_policy_changed",
    ):
        assert correction[key] is False
    authorization = program["authorization"]
    assert authorization["Phase_A_datum_builder"] is True
    assert authorization["field_inference"] is False
    assert authorization["mock_seed_access"] is False
    assert authorization["Phase_B_or_later"] is False


def test_wrapper_exposes_isolated_cli_and_corrected_program_schema():
    completed = subprocess.run(
        ["/home/kjhan/miniconda3/bin/python3.13", "-I", "-P", str(SOURCE), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "run-datum-builder" in completed.stdout
    module = _load_module()
    loaded, digest = module.v1.load_program(PROGRAM)
    assert loaded["schema"] == module.v1.PROGRAM_SCHEMA
    assert digest == hashlib.sha256(PROGRAM.read_bytes()).hexdigest()


def test_wrapper_returns_v3_patched_base_with_actual_vizier_header(tmp_path):
    module = _load_module()
    tracer_path = ROOT / "src/cf4_twompp_disjoint_tracer_pilot_v3.py"
    facade = module.v1._load_module(tracer_path, "_test_corrected_tracer_facade")
    catalog = tmp_path / "catalog.csv"
    fields = [
        "recno",
        "Ksmag",
        "Vcmb",
        "c11_5",
        "c12_5",
        "Cln",
        "Ref",
        "_RA",
        "_DE",
    ]
    with catalog.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "recno": 1,
                "Ksmag": 10.5,
                "Vcmb": 1000.0,
                "c11_5": 1.0,
                "c12_5": 1.0,
                "Cln": 0,
                "Ref": "real",
                "_RA": 12.0,
                "_DE": -3.0,
            }
        )
    loaded = facade.load_catalog(catalog)
    assert loaded["RA"].tolist() == [12.0]
    assert loaded["DEC"].tolist() == [-3.0]


def test_v2_runner_pins_resources_lineage_and_no_follow_on():
    text = RUNNER.read_text()
    assert f"program_sha={hashlib.sha256(PROGRAM.read_bytes()).hexdigest()}" in text
    assert f"source_sha={hashlib.sha256(SOURCE.read_bytes()).hexdigest()}" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --mem=1024M" in text
    assert "#SBATCH --time=00:20:00" in text
    assert '"$SUBMISSION_CONTROLLER" == syntax' in text
    assert 'host_name" != syntax' in text
    assert 'host_name" != syn101' in text
    assert "scripts/tripwire/**" in text
    assert "run-datum-builder" in text and "validate-datum" in text
    assert "renameat2" not in text
    assert "pgrep" not in text
    assert "srun" not in text
    assert "sbatch" not in text
