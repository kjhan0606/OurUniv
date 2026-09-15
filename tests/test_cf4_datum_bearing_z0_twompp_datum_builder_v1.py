import csv
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
REDESIGN = ROOT / "config/cf4_datum_bearing_z0_density_likelihood_redesign_v2.json"
PROGRAM = ROOT / "config/cf4_datum_bearing_z0_twompp_datum_builder_program_v1.json"
SOURCE = ROOT / "src/cf4_datum_bearing_z0_twompp_datum_builder_v1.py"
RUNNER = ROOT / "scripts/run_cf4_datum_bearing_z0_twompp_datum_builder_v1.sbatch"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_cf4_z0_datum_builder_v1", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_corrected_redesign_is_phase_a_only_and_rejects_audit_overreach():
    design = json.loads(REDESIGN.read_text())
    assert design["status"] == "CORRECTED_CONDITIONAL_GO_PHASE_A_AUTHORIZED_ONLY"
    authorization = design["authorization"]
    assert authorization["Phase_A_implementation"] is True
    assert authorization["Phase_A_single_Slurm_CPU_validation"] is True
    for key in ("field_inference", "mock_seed_consumption", "Phase_B_or_later"):
        assert authorization[key] is False
    corrections = design["adopted_corrections"]
    assert "spatial mean of eta" in corrections["density_monopole_gauge"]
    assert "plane-parallel" in corrections["full_sky_RSD"]
    assert corrections["RSD_deposition"].startswith("NGP applies only to observed")
    assert "prior_dominated" in corrections["low_exposure_label"]
    rejected = design["rejected_or_corrected_audit_statements"]
    assert "not adopted" in rejected["invented_numerical_thresholds"]
    assert "not a project result" in rejected["observational_k_estimate"]


def test_program_binds_every_input_and_implementation():
    program = json.loads(PROGRAM.read_text())
    assert program["status"] == "USER_AUTHORIZED_PHASE_A_ACTUAL_DATUM_BUILDER_ONLY"
    for binding in program["bindings"].values():
        path = Path(binding["path"])
        if not path.is_absolute():
            path = ROOT / path
        assert path.stat().st_size == binding["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    authorization = program["authorization"]
    assert authorization["Phase_A_datum_builder"] is True
    for key in (
        "field_inference",
        "mock_seed_access",
        "Phase_B_or_later",
        "resolution_increase",
        "IC_PM_HOP_RAMSES",
        "automatic_follow_on",
    ):
        assert authorization[key] is False
    assert sum(program["frozen_subset"]["population_counts_exact"]) == 36635
    assert program["crossmatch_policy"]["excluded_unique_targets_exact"] == 17007


def test_isolated_cli_and_program_load_succeed():
    completed = subprocess.run(
        ["/home/kjhan/miniconda3/bin/python3.13", "-I", "-P", str(SOURCE), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "run-datum-builder" in completed.stdout
    assert "validate-datum" in completed.stdout
    module = _load_module()
    loaded, digest = module.load_program(PROGRAM)
    assert loaded["schema"] == module.PROGRAM_SCHEMA
    assert digest == hashlib.sha256(PROGRAM.read_bytes()).hexdigest()


def test_hash_holdout_is_deterministic_and_uses_exact_integer_fraction_rule():
    module = _load_module()
    recnos = np.arange(1, 2001, dtype=np.int64)
    first = module.hash_holdout_mask(recnos, "frozen", 1, 5)
    second = module.hash_holdout_mask(recnos, "frozen", 1, 5)
    different = module.hash_holdout_mask(recnos, "different", 1, 5)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert 0.15 < np.mean(first) < 0.25
    with pytest.raises(module.DatumError, match="invalid holdout"):
        module.hash_holdout_mask(recnos, "", 1, 5)


def test_integer_count_grids_are_exact_and_exhaustive():
    module = _load_module()
    population = np.array([0, 0, 1, 1, 1])
    voxel = np.array([0, 0, 1, 1, 7])
    holdout = np.array([False, True, False, True, False])
    all_counts, train, held = module.integer_count_grids(
        population, voxel, holdout, population_count=2, grid=2
    )
    assert all_counts.shape == (2, 2, 2, 2)
    assert all_counts.dtype == np.int64
    assert np.array_equal(all_counts, train + held)
    assert all_counts[0].ravel()[0] == 2
    assert all_counts[1].ravel()[1] == 2
    assert all_counts[1].ravel()[7] == 1
    assert all_counts.sum() == 5


def test_row_manifest_is_sorted_and_split_explicit():
    module = _load_module()
    raw = module._row_manifest_bytes(
        np.array([9, 2]),
        np.array([1, 0]),
        np.array([5, 3]),
        np.array([True, False]),
        np.array([10.0, 20.0]),
    )
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    assert [int(row["recno"]) for row in rows] == [2, 9]
    assert [row["split"] for row in rows] == ["train", "holdout"]
    assert list(rows[0]) == [
        "recno",
        "population_index",
        "voxel_flat_index",
        "split",
        "redshift_space_radius_cMpc_h",
    ]


def test_failed_gate_preserves_staging_without_publication(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "load_program",
        lambda path: ({"schema": module.PROGRAM_SCHEMA}, "a" * 64),
    )
    arrays = {
        "counts_all": np.zeros((6, 2, 2, 2), dtype=np.int64),
        "counts_train": np.zeros((6, 2, 2, 2), dtype=np.int64),
        "counts_holdout": np.zeros((6, 2, 2, 2), dtype=np.int64),
        "raw_selection_exposure": np.zeros((6, 2, 2, 2), dtype=np.float64),
    }
    failed = {
        "schema": module.RESULT_SCHEMA,
        "status": module.STATUS_FAIL,
        "failed_gates": ["example"],
    }
    monkeypatch.setattr(
        module, "collect_datum", lambda *args: (failed, arrays, b"recno\n")
    )
    output = tmp_path / "datum"
    with pytest.raises(module.DatumError, match="diagnostics preserved"):
        module.publish_datum(PROGRAM, output, "0" * 40)
    assert not output.exists()
    stage = tmp_path / ".datum.staging"
    assert {path.name for path in stage.iterdir()} == {
        "datum.npz",
        "row_manifest.csv",
        "result.json",
        "FAILED",
    }


def test_source_does_not_normalize_selection_or_access_mock_randomness():
    text = SOURCE.read_text()
    assert "population_counts * selection" not in text
    assert "expected_counts" not in text
    assert "np.random" not in text
    assert "default_rng" not in text
    assert "field_inference_executed\": False" in text
    assert "mock_seed_accessed\": False" in text


def test_runner_pins_controller_resources_lineage_and_no_follow_on():
    text = RUNNER.read_text()
    program_sha = hashlib.sha256(PROGRAM.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert f"program_sha={program_sha}" in text
    assert f"source_sha={source_sha}" in text
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
