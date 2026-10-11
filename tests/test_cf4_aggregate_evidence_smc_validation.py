import numpy as np

from cf4_aggregate_evidence_smc_validation import (
    dense_discrete_log_z,
    run_dense_discrete_validation,
    run_null_likelihood_validation,
)


def test_dense_discrete_likelihood_is_seed_frozen_and_multimodal():
    first = dense_discrete_log_z()
    second = dense_discrete_log_z()
    np.testing.assert_array_equal(first, second)
    assert first.shape == (64, 8)
    maximum = np.max(first, axis=1, keepdims=True)
    aggregate = maximum[:, 0] + np.log(
        np.mean(np.exp(first - maximum), axis=1)
    )
    local_maximum = (
        (aggregate > np.roll(aggregate, 1))
        & (aggregate > np.roll(aggregate, -1))
    )
    assert np.count_nonzero(local_maximum) >= 2


def test_null_likelihood_validation_passes_frozen_contract():
    result = run_null_likelihood_validation()
    assert result["pass"] is True
    assert result["beta_history"] == [[0.0, 1.0]] * 4
    assert result["log_normalizer_absolute_error"] <= 1e-12
    assert result["uniform_parent_max_error"] <= 1e-12


def test_dense_discrete_validation_matches_exact_enumeration():
    result = run_dense_discrete_validation()
    assert result["pass"] is True
    assert result["geometry_states"] == 64
    assert result["synthetic_parents"] == 8
    assert result["multimodal"] is True
    assert sum(result["replicate_resampling_events"]) >= 1
    assert min(result["replicate_genealogical_ESS"]) < 2048.0
    assert result["pooled_log_normalizer_absolute_error"] <= 0.05
    assert result["pooled_parent_probability_L1"] <= 0.05
