import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_datum_bearing_z0_twompp_datum_publisher_program_v3.json"
SOURCE = ROOT / "src/cf4_datum_bearing_z0_twompp_datum_publisher_v3.py"
RUNNER = ROOT / "scripts/run_cf4_datum_bearing_z0_twompp_datum_publisher_v3.sbatch"
QUADRATURE_RECORD = ROOT / "config/cf4_twompp_selection_quadrature_diagnostic_v1_result_record.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_cf4_z0_datum_publisher_v3", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quadrature_result_recommends_order6_without_final_convergence_claim():
    record = json.loads(QUADRATURE_RECORD.read_text())
    assert record["status"] == "COMPLETE_ORDER6_RECOMMENDED_FOR_PHASE_A_DATUM_ONLY"
    assert record["measured_result"]["order2"][
        "positive_count_nonpositive_exposure_population_voxels"
    ] == 30
    assert record["measured_result"]["order4"][
        "positive_count_nonpositive_exposure_population_voxels"
    ] == 0
    assert record["measured_result"]["order6"][
        "positive_count_nonpositive_exposure_population_voxels"
    ] == 0
    evaluation = record["driver_evaluation"]
    assert evaluation["epsilon_floor_used"] is False
    assert evaluation["occupied_voxel_override_used"] is False
    assert evaluation["recommended_Phase_A_exposure"] == "raw_selection_order6"
    assert evaluation["claim_of_final_selection_convergence"] is False
    assert evaluation["actual_field_inference_allowed"] is False
    assert evaluation["Phase_B_allowed"] is False


def test_program_binds_sources_and_authorizes_publication_only():
    program = json.loads(PROGRAM.read_text())
    assert program["status"] == "AUTHORIZED_PHASE_A_ORDER6_DATUM_PUBLICATION_ONLY"
    for binding in program["bindings"].values():
        path = Path(binding["path"])
        if not path.is_absolute():
            path = ROOT / path
        assert path.stat().st_size == binding["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    authorization = program["authorization"]
    assert authorization["Phase_A_datum_publication"] is True
    for key in (
        "field_inference",
        "mock_seed_access",
        "Phase_B_or_later",
        "resolution_increase",
        "IC_PM_HOP_RAMSES",
        "automatic_follow_on",
    ):
        assert authorization[key] is False
    datum = program["datum"]
    assert datum["selection_quadrature_order_per_axis"] == 6
    assert datum["selection_subpoints_per_voxel"] == 216
    assert datum["selection_normalized_to_observed_totals"] is False
    assert datum["epsilon_floor_or_occupied_override_used"] is False
    assert datum["final_selection_convergence_claim"] is False


def test_isolated_cli_and_program_load_succeed():
    completed = subprocess.run(
        ["/home/kjhan/miniconda3/bin/python3.13", "-I", "-P", str(SOURCE), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "run-publication" in completed.stdout
    assert "validate-datum" in completed.stdout
    module = _load_module()
    loaded, digest = module.load_program(PROGRAM)
    assert loaded["schema"] == module.PROGRAM_SCHEMA
    assert digest == hashlib.sha256(PROGRAM.read_bytes()).hexdigest()


def test_row_manifest_reconstruction_is_exact(tmp_path):
    module = _load_module()
    path = tmp_path / "rows.csv"
    fields = [
        "recno",
        "population_index",
        "voxel_flat_index",
        "split",
        "redshift_space_radius_cMpc_h",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "recno": 2,
                "population_index": 0,
                "voxel_flat_index": 0,
                "split": "train",
                "redshift_space_radius_cMpc_h": 10.0,
            }
        )
        writer.writerow(
            {
                "recno": 9,
                "population_index": 1,
                "voxel_flat_index": 7,
                "split": "holdout",
                "redshift_space_radius_cMpc_h": 20.0,
            }
        )
    all_counts, train, holdout, summary = module.reconstruct_manifest_counts(
        path, grid=2, expected_rows=2
    )
    assert all_counts.shape == (6, 2, 2, 2)
    assert all_counts.sum() == 2
    assert np.array_equal(all_counts, train + holdout)
    assert train[0].ravel()[0] == 1
    assert holdout[1].ravel()[7] == 1
    assert summary == {
        "row_count": 2,
        "unique_recno_count": 2,
        "train_row_count": 1,
        "holdout_row_count": 1,
    }


def test_source_has_no_exposure_floor_randomness_or_follow_on():
    text = SOURCE.read_text()
    assert "np.maximum(exposure" not in text
    assert "epsilon" not in text.lower()
    assert "np.random" not in text
    assert "default_rng" not in text
    assert '"field_inference_executed": False' in text
    assert '"mock_seed_accessed": False' in text
    assert '"Phase_B_allowed_by_this_result": False' in text


def test_runner_pins_memory_lineage_and_no_follow_on():
    text = RUNNER.read_text()
    assert f"program_sha={hashlib.sha256(PROGRAM.read_bytes()).hexdigest()}" in text
    assert f"source_sha={hashlib.sha256(SOURCE.read_bytes()).hexdigest()}" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --mem=512M" in text
    assert "#SBATCH --time=00:10:00" in text
    assert '"$SUBMISSION_CONTROLLER" == syntax' in text
    assert 'host_name" != syntax' in text
    assert 'host_name" != syn101' in text
    assert "scripts/tripwire/**" in text
    assert "run-publication" in text and "validate-datum" in text
    assert "renameat2" not in text
    assert "pgrep" not in text
    assert "srun" not in text
    assert "sbatch" not in text
