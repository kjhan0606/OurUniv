import hashlib
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

v2 = importlib.import_module("cf4_twompp_disjoint_tracer_pilot_v2")


V1_PROGRAM = ROOT / "config/cf4_twompp_disjoint_tracer_pilot_program_v1.json"
PROGRAM = ROOT / "config/cf4_twompp_disjoint_tracer_pilot_program_v2.json"
SOURCE = ROOT / "src/cf4_twompp_disjoint_tracer_pilot_v2.py"
RUNNER = ROOT / "scripts/run_cf4_twompp_disjoint_tracer_pilot_v2.sbatch"


def _program() -> dict:
    return json.loads(PROGRAM.read_text())


def test_v2_binds_failed_v1_and_changes_only_the_header_contract() -> None:
    program = _program()
    assert program["frozen_v1_program"]["sha256"] == hashlib.sha256(
        V1_PROGRAM.read_bytes()
    ).hexdigest()
    assert program["implementation"]["sha256"] == hashlib.sha256(
        SOURCE.read_bytes()
    ).hexdigest()
    correction = program["implementation_correction_v2"]
    assert correction["failed_v1_Slurm_job_id"] == 329498
    assert correction["failure_before_catalog_deserialization"]
    for key in (
        "input_changed",
        "exclusion_policy_changed",
        "selection_or_population_changed",
        "threshold_changed",
        "random_seed_changed",
        "scientific_design_changed",
        "automatic_retry_after_v2_failure",
    ):
        assert correction[key] is False


def test_v2_loader_accepts_exact_vizier_RA_DE_header(tmp_path: Path) -> None:
    path = tmp_path / "catalog.csv"
    path.write_text(
        "recno,Ksmag,Vcmb,c11_5,c12_5,Cln,Ref,_RA,_DE\n"
        "7,10.5,1200,0.95,,0,real,12.5,-4.0\n"
    )
    result = v2.load_catalog(path)
    assert result["recno"].tolist() == [7]
    assert result["RA"].tolist() == [12.5]
    assert result["DEC"].tolist() == [-4.0]
    assert result["c12_5"].shape == (1,)


def test_v2_overlay_preserves_the_frozen_science_design() -> None:
    merged, digest = v2.load_program(PROGRAM)
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
    assert merged["execution"]["output"].endswith("twompp_disjoint_tracer_v2/pilot")


def test_v2_runner_pins_hashes_and_preserves_controller_boundary() -> None:
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
