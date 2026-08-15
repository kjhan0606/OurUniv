import hashlib
import json
from pathlib import Path

import numpy as np

from cf4_lg_mode_release_reference import (
    bootstrap_simultaneous_calibration,
    higher_tail_conformal_p,
    mahalanobis_distance,
    parse_shell_edges,
    profile_gaussian_nuisance,
    radial_residual_metrics,
    released_shell_geometry,
    released_shell_metrics,
    summary_coordinates,
    two_group_max_ks_permutation,
)


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/p2_lg_v8_cf4_mode_release_reference.json"


def test_profile_gaussian_nuisance_matches_direct_augmented_quadratic():
    observed = np.array([2.0, -1.0, 0.5])
    prediction = np.array([0.2, -0.4, 0.1])
    variance = np.array([1.0, 4.0, 2.0])
    design = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    prior = np.array([2.0, 3.0])
    result = profile_gaussian_nuisance(
        observed, prediction, variance, design, prior
    )
    residual = observed - prediction - design @ result["qhat"]
    expected = np.sum(residual**2 / variance) + np.sum(
        (result["qhat"] / prior) ** 2
    )
    assert np.isclose(result["marginal_deviance"], expected)

    raw = observed - prediction
    covariance = np.diag(variance) + design @ np.diag(prior**2) @ design.T
    direct_marginal = raw @ np.linalg.solve(covariance, raw)
    assert np.isclose(result["marginal_deviance"], direct_marginal)


def test_radial_residual_metrics_include_final_right_edge_only_once():
    residual = np.array([1.0, -1.0, 2.0, 0.0])
    cz = np.array([1.0, 2.0, 3.0, 4.0])
    bias, rms, count = radial_residual_metrics(residual, cz, [1.0, 3.0, 4.0])
    np.testing.assert_array_equal(count, [2, 2])
    np.testing.assert_allclose(bias, [0.0, 1.0])
    np.testing.assert_allclose(rms, [1.0, np.sqrt(2.0)])


def test_higher_tail_conformal_excludes_target_but_adds_finite_sample_one():
    assert higher_tail_conformal_p(4.0, np.array([1.0, 2.0, 3.0])) == 0.25
    assert higher_tail_conformal_p(2.0, np.array([1.0, 2.0, 3.0])) == 0.75


def test_released_shell_metrics_use_hermitian_weighted_energy():
    n = 8
    box = 8.0
    edges = np.array([0.0, 2.0, 3.0, np.inf])
    geometry = released_shell_geometry(n, box, 2, edges)
    field = np.random.default_rng(4).normal(size=(n, n, n))
    zero = np.zeros_like(np.fft.rfftn(field, norm="ortho"))
    metrics = released_shell_metrics(field, zero, zero, geometry)
    np.testing.assert_allclose(metrics["Eres"], metrics["Pwhite"])
    np.testing.assert_allclose(
        metrics["delta_E_parent3429"], metrics["Pwhite"]
    )
    assert np.all(geometry["weight_sums"] > 0)


def test_released_shells_exactly_partition_mask_and_obey_parseval_sum():
    n = 8
    edges = parse_shell_edges([0.0, 2.0, 3.0, None])
    geometry = released_shell_geometry(n, 8.0, 2, edges)
    covered = np.logical_or.reduce(geometry["masks"])
    np.testing.assert_array_equal(covered, geometry["released_mask"])
    assert sum(np.count_nonzero(mask) for mask in geometry["masks"]) == np.count_nonzero(
        geometry["released_mask"]
    )

    field = np.random.default_rng(5).normal(size=(n, n, n))
    zero = np.zeros_like(np.fft.rfftn(field, norm="ortho"))
    metrics = released_shell_metrics(field, zero, zero, geometry)
    weighted_shell_sum = np.sum(metrics["Pwhite"] * geometry["weight_sums"])
    field_fft = np.fft.rfftn(field, norm="ortho")
    weights = np.broadcast_to(geometry["weights"], field_fft.shape)
    direct = np.sum(
        weights[geometry["released_mask"]]
        * np.abs(field_fft[geometry["released_mask"]]) ** 2
    )
    assert np.isclose(weighted_shell_sum, direct)


def test_exceedance_summary_uses_fraction_for_255_vs_256_banks():
    calibration = np.arange(255.0)[:, None]
    proposal = np.linspace(0.0, 254.0, 256)[:, None]
    q99 = np.quantile(calibration, 0.99, axis=0, method="linear")
    cal_summary = summary_coordinates(calibration, q99)
    prop_summary = summary_coordinates(proposal, q99)
    assert 0.0 <= cal_summary[-1] <= 1.0
    assert 0.0 <= prop_summary[-1] <= 1.0
    assert abs(cal_summary[-1] - prop_summary[-1]) < 1.0 / 255.0


def test_four_dimensional_mahalanobis_uses_joint_covariance():
    covariance = np.array([
        [1.0, 0.9, 0.0, 0.0],
        [0.9, 1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 3.0],
    ])
    values = np.array([[1.0, -1.0, 0.0, 0.0]])
    distance = mahalanobis_distance(values, np.zeros(4), covariance)
    direct = np.sqrt(values[0] @ np.linalg.solve(covariance, values[0]))
    assert np.isclose(distance[0], direct)


def test_two_group_max_ks_permutation_is_deterministic_and_detects_shift():
    rng = np.random.default_rng(27)
    matrix = np.vstack((
        rng.normal(0.0, 1.0, size=(20, 3)),
        rng.normal(2.5, 1.0, size=(20, 3)),
    ))
    first = two_group_max_ks_permutation(
        matrix, first_group_size=20, iterations=999, seed=31, chunk_size=73
    )
    second = two_group_max_ks_permutation(
        matrix, first_group_size=20, iterations=999, seed=31, chunk_size=128
    )
    assert np.isclose(first["observed_statistic"], second["observed_statistic"])
    assert first["permutation_pvalue"] == second["permutation_pvalue"]
    assert first["permutation_pvalue"] < 0.01


def test_bootstrap_calibration_is_deterministic_and_joint_across_families():
    rng = np.random.default_rng(8)
    families = {
        "a": rng.normal(size=(31, 2)),
        "b": np.abs(rng.normal(size=(31, 3))),
    }
    first = bootstrap_simultaneous_calibration(
        families, iterations=200, seed=91, chunk_size=37
    )
    second = bootstrap_simultaneous_calibration(
        families, iterations=200, seed=91, chunk_size=64
    )
    assert np.isclose(
        first["simultaneous_studentized_max_critical"],
        second["simultaneous_studentized_max_critical"],
    )
    for name in families:
        np.testing.assert_allclose(
            first["families"][name]["bootstrap_studentization_scale"],
            second["families"][name]["bootstrap_studentization_scale"],
        )


def test_reference_program_pins_code_and_keeps_projection_firewall_closed():
    program = json.loads(PROGRAM.read_text())
    implementation = REPO / program["implementation"]["path"]
    assert hashlib.sha256(implementation.read_bytes()).hexdigest() == program[
        "implementation"
    ]["sha256"]
    assert program["information_firewall"]["reference_data_only"] is True
    assert (
        program["information_firewall"]["V8_projection_CF4_metrics_opened"]
        is False
    )
    assert program["decision"]["fresh_V9_authorized_now"] is False
    assert program["decision"]["RAMSES_authorized"] is False
    groups = program["reference"]["mean_groups"]
    assert [(row["seed_range_inclusive"], row["count"]) for row in groups] == [
        ([3193, 3328], 136),
        ([3329, 3448], 120),
    ]
    assert program["reference"]["chain_homogeneity"]["coordinate_count"] == 42
    assert program["reference"]["canonical_mean_policy"].startswith(
        "Use the seed3429"
    )
