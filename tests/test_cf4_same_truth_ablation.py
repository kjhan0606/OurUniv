import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cf4_same_truth_ablation as ablation  # noqa: E402


PROGRAM = ROOT / "config/cf4_same_truth_likelihood_ablation_program_v1.json"
SOURCE = ROOT / "src/cf4_same_truth_ablation.py"


def _args():
    return SimpleNamespace(
        edm_floor=0.04343,
        error_scale=0.9,
        sigma_nl=0.0,
        box_size=384.0,
        bulk_prior=150.0,
        h0_prior=3.0,
    )


def _catalog():
    count = 8
    direction = np.zeros((count, 3))
    direction[:, 0] = 1.0
    return {
        "H0": np.array(74.6),
        "v3k": np.linspace(2000.0, 9000.0, count),
        "dist": np.linspace(30.0, 100.0, count),
        "e_dm": np.linspace(0.05, 0.2, count),
        "nhat": direction,
    }


def test_direct_arm_uses_frozen_unbiased_distance_and_moment_variance():
    catalog = _catalog()
    raw = np.array([1, 3, 6])
    hold = np.array([False, True, False])
    result = ablation.direct_lognormal_design(catalog, raw, hold, _args())
    sigma_ln = catalog["e_dm"][raw] * np.log(10.0) / 5.0
    corrected = catalog["dist"][raw] * np.exp(-0.5 * sigma_ln**2)
    expected_variance = (
        0.9 * 74.6 * corrected * np.sqrt(np.expm1(sigma_ln**2))
    ) ** 2
    np.testing.assert_allclose(result["B"][:, 3], -corrected)
    np.testing.assert_allclose(result["vobs"], catalog["v3k"][raw] - 74.6 * corrected)
    np.testing.assert_allclose(result["variance"], expected_variance)
    np.testing.assert_array_equal(result["holdout"], hold)


def test_A_and_C_share_one_nontruth_noise_stream_and_B_uses_stored_distance_noise():
    count = 22136
    direction = np.zeros((count, 3))
    direction[:, 0] = 1.0
    fields = {
        "mock_cz": np.linspace(2000.0, 10000.0, count),
        "mock_observed_distance": np.linspace(30.0, 100.0, count),
        "mock_distance_error_mag": np.full(count, 0.1),
        "mock_direction": direction,
        "mock_true_distance": np.linspace(29.0, 99.0, count),
        "mock_true_position": np.zeros((count, 3)),
    }
    raw = np.array([0, 5, 17, 101])
    bgc = {
        "raw_idx": raw,
        "holdout": np.array([False, True, False, True]),
        "variance": np.array([100.0, 121.0, 144.0, 169.0]),
        "q_std": np.array([150.0, 150.0, 150.0, 3.0]),
        "pos": np.ones((4, 3)),
        "B": np.column_stack((direction[raw], -np.arange(1.0, 5.0))),
    }
    first, first_noise = ablation.paired_arm_designs(fields, bgc, _args(), 77)
    second, second_noise = ablation.paired_arm_designs(fields, bgc, _args(), 77)
    np.testing.assert_array_equal(first_noise, second_noise)
    np.testing.assert_array_equal(first["a"]["variance"], first["c"]["variance"])
    assert bool(first["a"]["exact_gaussian"])
    assert not bool(first["b"]["exact_gaussian"])
    assert bool(first["c"]["exact_gaussian"])


def test_preregistered_lowest_bin_tree_is_deterministic_and_fail_first():
    assert ablation.classify_lowest_bin(
        {"a": False, "b": True, "c": True, "d": True}, True
    ) == "IDEAL_TRUE_POSITION_BASELINE_INSUFFICIENT"
    assert ablation.classify_lowest_bin(
        {"a": True, "b": True, "c": True, "d": False}, False
    ) == "BGC_DATUM_OR_GENERATIVE_LIKELIHOOD_MISMATCH"
    assert ablation.classify_lowest_bin(
        {"a": True, "b": False, "c": True, "d": False}, False
    ) == "MULTIPLE_DIRECT_DISTANCE_AND_BGC_DATUM_LOSSES"
    assert ablation.classify_lowest_bin(
        {"a": True, "b": True, "c": True, "d": True}, False
    ) == "POPULATION_GENERATOR_FIDELITY_REMAINS_FAILED"


def test_program_binds_sources_and_forbids_new_truth_and_validation():
    program = json.loads(PROGRAM.read_text())
    assert tuple(program["arms"]) == ("a", "b", "c", "d")
    assert program["arms"]["d"]["execution"] == "read_only_completed_artifact"
    assert program["development"]["new_truth_seed_count"] == 0
    assert program["shared_pairing_contract"]["population_selection_removed_or_independently_tested"] is False
    authorization = program["authorization"]
    assert authorization["new_truth_generation"] is False
    assert authorization["population_generator_retuning"] is False
    assert authorization["untouched_256_mock_validation"] is False
    assert authorization["frontier_promotion"] is False
    for collection in ("repository_bindings", "source_bindings"):
        for record in program[collection].values():
            assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record["sha256"]


def test_source_reuses_stored_truth_and_does_not_generate_a_truth_seed_field():
    source = SOURCE.read_text()
    assert 'truth_white=fields["truth_white"]' in source
    assert 'truth_nuisance_q=fields["truth_nuisance_q"]' in source
    assert 'np.random.default_rng(int(seeds["truth"]))' not in source
    assert "2026083064" not in source
    assert "2026083320" not in source
    assert "mock_true_radial_velocity" in source
    assert "population_selection_removed_or_validated\": False" in source


def test_memory_requests_have_at_least_twenty_percent_headroom():
    execution = json.loads(PROGRAM.read_text())["execution"]
    assert execution["member_requested_memory_MiB"] >= 1.2 * execution["member_expected_peak_memory_MiB"]
    assert execution["aggregate_requested_memory_MiB"] >= 1.2 * execution["aggregate_expected_peak_memory_MiB"]
