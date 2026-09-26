import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1_PROGRAM = ROOT / "config/cf4_twompp_disjoint_tracer_pilot_program_v1.json"
V2_PROGRAM = ROOT / "config/cf4_twompp_disjoint_tracer_pilot_program_v2.json"
V2_SOURCE = ROOT / "src/cf4_twompp_disjoint_tracer_pilot_v2.py"
PROGRAM = ROOT / "config/cf4_twompp_disjoint_tracer_pilot_program_v3.json"
SOURCE = ROOT / "src/cf4_twompp_disjoint_tracer_pilot_v3.py"
RUNNER = ROOT / "scripts/run_cf4_twompp_disjoint_tracer_pilot_v3.sbatch"


def _load_v3():
    spec = importlib.util.spec_from_file_location("test_twompp_pilot_v3", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v3_isolated_invocation_succeeds_before_submission() -> None:
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


def test_v3_binds_both_preserved_failures_and_exact_source() -> None:
    program = json.loads(PROGRAM.read_text())
    assert program["frozen_v1_program"]["sha256"] == hashlib.sha256(
        V1_PROGRAM.read_bytes()
    ).hexdigest()
    assert program["failed_v2_program"]["sha256"] == hashlib.sha256(
        V2_PROGRAM.read_bytes()
    ).hexdigest()
    assert program["failed_v2_implementation"]["sha256"] == hashlib.sha256(
        V2_SOURCE.read_bytes()
    ).hexdigest()
    assert program["implementation"]["sha256"] == hashlib.sha256(
        SOURCE.read_bytes()
    ).hexdigest()
    correction = program["implementation_correction_v3"]
    assert correction["failed_v1_Slurm_job_id"] == 329498
    assert correction["failed_v2_Slurm_job_id"] == 329499
    assert correction["failure_before_program_load"]
    assert correction["pre_submission_exact_isolated_invocation_required"]
    for key in (
        "input_changed",
        "exclusion_policy_changed",
        "selection_or_population_changed",
        "threshold_changed",
        "random_seed_changed",
        "scientific_design_changed",
        "automatic_retry_after_v3_failure",
    ):
        assert correction[key] is False


def test_v3_overlay_and_catalog_loader_preserve_v1_science(tmp_path: Path) -> None:
    module = _load_v3()
    merged, digest = module.load_program(PROGRAM)
    frozen = json.loads(V1_PROGRAM.read_text())
    assert digest == hashlib.sha256(PROGRAM.read_bytes()).hexdigest()
    for key in (
        "inputs",
        "cosmology",
        "no_double_counting",
        "tracer_design",
        "catalog_gate",
        "angular_completeness_gate",
        "carrick_reference_gate",
        "future_information_gate_not_executed_by_pilot",
    ):
        assert merged[key] == frozen[key]
    path = tmp_path / "catalog.csv"
    path.write_text(
        "recno,Ksmag,Vcmb,c11_5,c12_5,Cln,Ref,_RA,_DE\n"
        "9,10.2,900,0.9,0.8,0,real,22.0,3.0\n"
    )
    catalog = module.load_catalog(path)
    assert catalog["recno"].tolist() == [9]
    assert catalog["RA"].tolist() == [22.0]
    assert catalog["DEC"].tolist() == [3.0]


def test_v3_runner_pins_hashes_and_has_no_automatic_follow_on() -> None:
    text = RUNNER.read_text()
    assert f"program_sha={hashlib.sha256(PROGRAM.read_bytes()).hexdigest()}" in text
    assert f"source_sha={hashlib.sha256(SOURCE.read_bytes()).hexdigest()}" in text
    assert "#SBATCH --mem=1024M" in text
    assert "#SBATCH --time=00:15:00" in text
    assert '"$SUBMISSION_CONTROLLER" == syntax' in text
    assert 'host_name" != syntax' in text
    assert 'host_name" != syn101' in text
    assert "scripts/tripwire/**" in text
    assert "renameat2" not in text
    assert "pgrep" not in text
    assert "--requeue" not in text
