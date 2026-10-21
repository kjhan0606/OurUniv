import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cf4_twompp_disjoint_tracer_pilot as pilot


PROGRAM_PATH = ROOT / "config/cf4_twompp_disjoint_tracer_pilot_program_v1.json"
SOURCE_PATH = ROOT / "src/cf4_twompp_disjoint_tracer_pilot.py"
RUNNER_PATH = ROOT / "scripts/run_cf4_twompp_disjoint_tracer_pilot_v1.sbatch"


def _program() -> dict:
    return json.loads(PROGRAM_PATH.read_text())


def test_program_binds_source_and_closes_science_firewalls() -> None:
    program = _program()
    assert program["implementation"]["sha256"] == hashlib.sha256(
        SOURCE_PATH.read_bytes()
    ).hexdigest()
    authorization = program["authorization"]
    assert authorization["technical_pilot"]
    for key in (
        "field_inference",
        "likelihood_datum_consumed_by_field_inference",
        "joint_information_production",
        "resolution_increase",
        "new_truth_seed",
        "untouched_256_validation",
        "ML_training",
        "IC_PM_HOP_RAMSES",
        "automatic_follow_on",
    ):
        assert authorization[key] is False


def test_no_double_counting_and_resolution_contract_are_conservative() -> None:
    program = _program()
    contract = program["no_double_counting"]
    assert contract["expected_unique_2Mpp_targets_excluded"] == 17007
    assert set(contract["excluded_crossmatch_classes"]) == {
        "secure_joint_mark",
        "coordinate_redshift_conflict",
        "nonreciprocal_collision",
        "extended_review_candidate",
    }
    design = program["tracer_design"]
    assert design["grid_N"] == 32
    assert design["box_size_cMpc_h"] / design["grid_N"] == 12.0
    assert design["radial_max_cMpc_h"] <= design["box_size_cMpc_h"] / 2 - 12.0
    assert len(design["absolute_K_edges"]) - 1 == 3
    assert design["population_count"] == 6
    assert program["selected_tracer_route"]["observational_resolution_claim_cMpc_h"] is None


def test_crossmatch_exclusion_uses_every_non_unmatched_target(tmp_path: Path) -> None:
    path = tmp_path / "mapping.csv"
    path.write_text(
        "cf4_recno,twompp_recno,match_class\n"
        "1,101,secure_joint_mark\n"
        "2,102,coordinate_redshift_conflict\n"
        "3,102,nonreciprocal_collision\n"
        "4,,unmatched\n"
    )
    targets, counts = pilot.read_crossmatch_exclusions(path, 2)
    assert targets == {101, 102}
    assert counts == {
        "secure_joint_mark": 1,
        "coordinate_redshift_conflict": 1,
        "nonreciprocal_collision": 1,
        "unmatched": 1,
    }


def test_catalog_classification_is_an_exact_fail_closed_partition() -> None:
    count = 8
    catalog = {
        "recno": np.arange(1, count + 1),
        "Ksmag": np.array([10.0, 10.0, 10.0, 13.0, 10.0, 10.0, 10.0, 10.0]),
        "Vcmb": np.array([1000.0, 1000.0, -1.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]),
        "c12_5": np.ones(count),
        "Cln": np.array([0, 1, 0, 0, 0, 0, 0, 0]),
        "Ref": np.array(["zoa", "real", "real", "real", "real", "real", "real", "real"]),
        "RA": np.zeros(count),
        "DEC": np.zeros(count),
    }
    distance = np.array([10.0, 10.0, np.nan, 10.0, 10.0, 181.0, 10.0, 10.0])
    absolute = np.array([-22.0, -22.0, np.nan, -22.0, -22.0, -22.0, -20.0, -22.0])
    design = _program()["tracer_design"]
    eligible, reasons, apparent, absolute_bin = pilot.classify_disjoint_tracer(
        catalog, {5}, distance, absolute, design
    )
    assert eligible.tolist() == [False, False, False, False, False, False, False, True]
    assert reasons == {
        "eligible_disjoint_tracer": 1,
        "excluded_ZoA_imputation": 1,
        "excluded_absolute_magnitude": 1,
        "excluded_any_CF4_match_candidate": 1,
        "excluded_apparent_magnitude_or_mask": 1,
        "excluded_cloned_redshift": 1,
        "excluded_invalid_observed_fields": 1,
        "excluded_radial_support": 1,
    }
    assert apparent.shape == absolute_bin.shape == (count,)


def test_voxel_summary_reports_six_populations_without_wraparound() -> None:
    design = _program()["tracer_design"]
    unit = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
         [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    )
    distance = np.full(6, 100.0)
    eligible = np.ones(6, dtype=bool)
    apparent = np.array([0, 0, 0, 1, 1, 1])
    absolute = np.array([0, 1, 2, 0, 1, 2])
    summary = pilot.voxel_summary(
        unit, distance, eligible, apparent, absolute, design
    )
    assert summary["eligible_row_count"] == 6
    assert summary["occupied_voxel_count"] == 6
    assert set(summary["six_population_counts"].values()) == {1}


def test_runner_pins_hashes_resources_and_controller_boundary() -> None:
    text = RUNNER_PATH.read_text()
    assert f"program_sha={hashlib.sha256(PROGRAM_PATH.read_bytes()).hexdigest()}" in text
    assert f"source_sha={hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()}" in text
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
