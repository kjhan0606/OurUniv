import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_twompp_selection_quadrature_diagnostic_program_v1.json"
SOURCE = ROOT / "src/cf4_twompp_selection_quadrature_diagnostic_v1.py"
RUNNER = ROOT / "scripts/run_cf4_twompp_selection_quadrature_diagnostic_v1.sbatch"
FAILURE = ROOT / "config/cf4_datum_bearing_z0_twompp_datum_builder_v2_failure_result_record.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_cf4_selection_quadrature_v1", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_program_binds_failed_gate_and_is_diagnostic_only():
    program = json.loads(PROGRAM.read_text())
    assert program["status"] == "AUTHORIZED_GLOBAL_ORDER4_ORDER6_SELECTION_DIAGNOSTIC_ONLY"
    for binding in program["bindings"].values():
        path = Path(binding["path"])
        if not path.is_absolute():
            path = ROOT / path
        assert path.stat().st_size == binding["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    authorization = program["authorization"]
    assert authorization["quadrature_diagnostic"] is True
    for key in (
        "datum_publication",
        "field_inference",
        "mock_seed_access",
        "Phase_B_or_later",
        "selection_order_promotion",
        "automatic_follow_on",
    ):
        assert authorization[key] is False
    assert program["design"]["diagnostic_orders_per_axis"] == [4, 6]
    assert program["design"]["count_occupancy_used_to_modify_exposure"] is False
    assert program["design"]["epsilon_floor_allowed"] is False
    assert program["design"]["promotion_or_repair_decision_in_this_job"] is False


def test_failure_record_requires_quadrature_diagnostic_not_epsilon_floor():
    failure = json.loads(FAILURE.read_text())
    gate = failure["gate_result"]
    assert gate["failed_gates"] == ["no_positive_count_in_nonpositive_exposure"]
    assert gate["positive_count_nonpositive_exposure_population_voxel_count"] == 30
    science = failure["scientific_interpretation"]
    assert science["epsilon_floor_or_occupied_voxel_override_allowed"] is False
    assert science["outcome_adaptive_selection_repair_allowed"] is False
    assert science["field_inference_allowed"] is False
    assert science["Phase_B_allowed"] is False


def test_isolated_cli_and_program_load_succeed():
    completed = subprocess.run(
        ["/home/kjhan/miniconda3/bin/python3.13", "-I", "-P", str(SOURCE), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "run-diagnostic" in completed.stdout
    assert "validate-diagnostic" in completed.stdout
    module = _load_module()
    loaded, digest = module.load_program(PROGRAM)
    assert loaded["schema"] == module.PROGRAM_SCHEMA
    assert digest == hashlib.sha256(PROGRAM.read_bytes()).hexdigest()


def test_exposure_summary_counts_population_voxels_and_galaxies_separately():
    module = _load_module()
    counts = np.zeros((2, 2, 2, 2), dtype=np.int64)
    exposure = np.ones_like(counts, dtype=np.float64)
    counts[0, 0, 0, 0] = 3
    counts[1, 1, 1, 1] = 2
    exposure[0, 0, 0, 0] = 0.0
    summary = module.exposure_summary(counts, exposure)
    assert summary["positive_count_nonpositive_exposure_population_voxel_count"] == 1
    assert summary["galaxy_count_in_nonpositive_exposure_population_voxels"] == 3
    assert summary["failed_population_voxel_count_by_population"] == [1, 0]
    assert summary["failed_galaxy_count_by_population"] == [3, 0]


def test_comparison_reports_exact_zero_for_identical_exposure():
    module = _load_module()
    values = np.ones((2, 2, 2, 2), dtype=np.float64)
    result = module.comparison(values, values.copy())
    assert result["candidate_minus_reference_maximum_absolute"] == 0.0
    assert result["relative_L1_by_population"] == [0.0, 0.0]
    assert result["relative_support_change_by_population"] == [0.0, 0.0]
    assert result["positive_support_disagreement_fraction_by_population"] == [0.0, 0.0]


def test_source_uses_global_gauss_weights_and_has_no_random_or_promotion_path():
    text = SOURCE.read_text()
    assert "np.polynomial.legendre.leggauss(order)" in text
    assert "weights[ix] * weights[iy] * weights[iz] / 8.0" in text
    assert "np.random" not in text
    assert "default_rng" not in text
    assert '"datum_published": False' in text
    assert '"selection_order_promotion_allowed_by_this_diagnostic": False' in text


def test_runner_pins_controller_resources_and_no_follow_on():
    text = RUNNER.read_text()
    assert f"program_sha={hashlib.sha256(PROGRAM.read_bytes()).hexdigest()}" in text
    assert f"source_sha={hashlib.sha256(SOURCE.read_bytes()).hexdigest()}" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --mem=1024M" in text
    assert "#SBATCH --time=00:30:00" in text
    assert '"$SUBMISSION_CONTROLLER" == syntax' in text
    assert 'host_name" != syntax' in text
    assert 'host_name" != syn101' in text
    assert "scripts/tripwire/**" in text
    assert "run-diagnostic" in text and "validate-diagnostic" in text
    assert "renameat2" not in text
    assert "pgrep" not in text
    assert "srun" not in text
    assert "sbatch" not in text
