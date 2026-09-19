from pathlib import Path

import numpy as np

import hong2021_v82a_consumed_autopsy as audit


def test_uniform_rank_shape_is_exact_reference() -> None:
    result = audit.rank_shape(np.full(17, 100, dtype=np.int64))
    assert result["total_variation_from_uniform"] == 0.0
    assert np.isclose(result["mean_rank_minus_8"], 0.0)
    assert np.isclose(result["rank_variance_over_uniform"], 1.0)
    assert np.isclose(result["edge_fraction_over_uniform"], 1.0)
    assert np.isclose(result["central_fraction_over_uniform"], 1.0)
    assert np.isclose(result["outer4_minus_central3_excess"], 0.0)


def test_rank_shape_separates_u_shape_bias_and_slope() -> None:
    underdispersed = np.ones(17, dtype=np.int64)
    underdispersed[[0, 16]] = 100
    result = audit.rank_shape(underdispersed)
    assert result["rank_variance_over_uniform"] > 1.0
    assert result["edge_fraction_over_uniform"] > 1.0
    assert abs(result["mean_rank_minus_8"]) < 1e-12
    sloped = np.arange(1, 18, dtype=np.int64)
    assert audit.rank_shape(sloped)["mean_rank_minus_8"] > 0.0
    assert audit.rank_shape(sloped)["left_minus_right_fraction"] < 0.0


def test_equal_count_strata_are_deterministic_even_with_ties() -> None:
    field = np.zeros((4, 4, 4), dtype=np.float64)
    first = audit.equal_count_strata(field)
    second = audit.equal_count_strata(field)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.bincount(first.ravel()), [16, 16, 16, 16])


def test_rank_and_coverage_uses_strict_rank_and_four_strata() -> None:
    truth = np.zeros((4, 4, 4), dtype=np.float64)
    residual = np.arange(16, dtype=np.float64)[:, None, None, None]
    residual = np.broadcast_to(residual, (16, 4, 4, 4))
    strata = audit.equal_count_strata(np.arange(64).reshape(4, 4, 4))
    result = audit.rank_and_coverage(residual, truth, strata)
    assert result["shape"]["histogram"][0] == 64
    assert len(result["conditional_mean_quartiles"]) == 4
    assert all(row["histogram"][0] == 16 for row in result["conditional_mean_quartiles"])


def test_evaluator_rank_crosscheck_is_exactly_float32() -> None:
    mean = np.full((2, 2, 2), np.float32(0.1), dtype=np.float32)
    truth = mean.copy()
    sample = np.broadcast_to(mean, (16, 2, 2, 2)).copy()
    result = audit.evaluator_rank_histogram(sample, truth, mean)
    assert result.dtype == np.dtype("int64")
    assert result[0] == 8
    assert result.sum() == 8


def test_phase_cosine_identifies_same_and_opposite_fields() -> None:
    rng = np.random.default_rng(8)
    field = rng.normal(size=(1, 64, 64, 64))
    binner = audit.SpectralBinner(64, audit.VOXEL_MPC_H)
    transformed = binner.transform(field)
    same = audit.phase_cosine(binner, transformed, transformed)
    opposite = audit.phase_cosine(binner, transformed, -transformed)
    assert all(np.isclose(value, 1.0) for value in same.values())
    assert all(np.isclose(value, -1.0) for value in opposite.values())


def test_source_is_read_only_and_contains_no_model_fit_or_sampling_import() -> None:
    source = Path(audit.__file__).read_text()
    assert "torch" not in source
    assert "hong2021_v80_sample" not in source
    assert "create_dataset" not in source
    assert "h5py.File(paths[\"candidate\"], \"r\")" in source
    assert "h5py.File(paths[\"control\"], \"r\")" in source
    assert "def evaluator_rank_histogram(" in source
