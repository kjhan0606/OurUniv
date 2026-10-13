import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts/cf4_r2_ray_selection_training_impact.py")
SPEC = importlib.util.spec_from_file_location(
    "cf4_r2_ray_selection_training_impact", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pairwise_proxy_gate_passes_only_with_common_positive_support():
    n = MODULE.N
    keys = np.array([0, 1, n ** 3], dtype=np.int64)
    counts = np.array([4, 9, 7], dtype=np.int64)
    old = np.array([1., 2., 3.])
    low = np.array([2., 4., 3.])
    high = np.array([2., 4., 3.])

    rows = MODULE.summarize_exposure_triplet(keys, counts, old, low, high)

    assert rows[0]["resolution_gate_pass"]
    assert rows[1]["resolution_gate_pass"]
    assert rows[0]["nside2048_vs_nside1024_common_support"][
        "full_signed_delta_log_likelihood_nats"] == 0.


def test_occupied_zero_exposure_is_not_floored_or_given_finite_delta():
    n = MODULE.N
    keys = np.array([0, 1, n ** 3], dtype=np.int64)
    counts = np.array([4, 9, 7], dtype=np.int64)
    old = np.array([1., 0., 3.])
    low = np.array([2., 0., 3.])
    high = np.array([2., 5., 3.])

    rows = MODULE.summarize_exposure_triplet(keys, counts, old, low, high)
    population0 = rows[0]
    comparison = population0["nside2048_vs_nside1024_common_support"]

    assert population0["support"]["nside1024_zero_galaxies"] == 9
    assert population0["support"][
        "nside1024_zero_nside2048_positive_galaxies"] == 9
    assert comparison["full_signed_delta_finite"] is False
    assert comparison["full_signed_delta_log_likelihood_nats"] is None
    assert population0["resolution_gate_pass"] is False


def test_zero_exposure_classification_distinguishes_domain_and_ray_miss():
    ray = MODULE.ray
    assert MODULE._classify_zero_exposure(1., 4., 0) == "outside_selection_domain"
    assert MODULE._classify_zero_exposure(4., 6., 0) == (
        "positive_volume_domain_intersection_missed_by_finite_rays")
    assert MODULE._classify_zero_exposure(6., 8., 2) == (
        "rays_intersect_domain_but_population_exposure_is_zero")
    assert ray.R_INNER == 5.


def test_training_geometry_summary_aggregates_population_rows_by_cell():
    n = MODULE.N
    keys = np.array([0, 1, n ** 3], dtype=np.int64)
    counts = np.array([2, 4, 3], dtype=np.int64)
    rmin = np.array([6., 6., 6.])
    rmax = np.array([7., 7., 7.])
    geometry1024 = np.array([1., .999, 1.])
    geometry2048 = np.array([.98, .999, .98])

    result = MODULE._summarize_training_interior_geometry(
        keys, counts, rmin, rmax, geometry1024, geometry2048)

    assert result["unique_training_cells"] == 2
    assert result["fully_interior_training_galaxies"] == 9
    assert result["nside2048"]["unique_cells_abs_deviation_gt_1e-2"] == 1
    assert result["nside2048"][
        "training_galaxies_in_cells_abs_deviation_gt_1e-2"] == 5
