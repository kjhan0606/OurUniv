from __future__ import annotations

import builtins

import numpy as np
import pytest

import cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_read_only_analysis as analysis


def _fixture() -> dict[str, object]:
    cache_keys = np.array(
        [
            [-2, 0, 0, 1, 0, 0],
            [-1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0],
        ],
        dtype=np.int16,
    )
    cache_log_z = np.zeros((4, analysis.N_PARENTS), dtype=np.float64)
    cache_log_z[0, 0] = 3.0
    cache_log_z[1, 1] = 3.0
    cache_log_z[2, 2] = 3.0
    cache_log_z[3, 3] = 3.0
    replicate_keys = [
        cache_keys[[0, 0, 1]],
        cache_keys[[1, 2]],
        cache_keys[[2, 3]],
        cache_keys[[0, 3]],
    ]
    replicate_weights = [
        np.array([0.2, 0.3, 0.5], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([0.25, 0.75], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
    ]
    row_max = np.max(cache_log_z, axis=1, keepdims=True)
    parent_given_key = np.exp(cache_log_z - row_max)
    parent_given_key /= np.sum(parent_given_key, axis=1, keepdims=True)
    cache_log_z_bar = row_max[:, 0] + np.log(
        np.mean(np.exp(cache_log_z - row_max), axis=1)
    )
    key_mass = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.25, 0.75],
            [0.5, 0.0, 0.0, 0.5],
        ],
        dtype=np.float64,
    )
    stored_p_rep = key_mass @ parent_given_key
    log_i_bar = np.array([-1.0, -1.1, -0.9, -1.2], dtype=np.float64)
    pooling = np.exp(log_i_bar - np.max(log_i_bar))
    pooling /= np.sum(pooling)
    stored_p_pool = pooling @ stored_p_rep
    return {
        "replicate_keys": replicate_keys,
        "replicate_weights": replicate_weights,
        "cache_keys": cache_keys,
        "cache_log_z": cache_log_z,
        "cache_log_z_bar": cache_log_z_bar,
        "replicate_log_z_bar": [
            cache_log_z_bar[[0, 0, 1]],
            cache_log_z_bar[[1, 2]],
            cache_log_z_bar[[2, 3]],
            cache_log_z_bar[[0, 3]],
        ],
        "stored_p_rep": stored_p_rep,
        "log_i_bar": log_i_bar,
        "stored_p_pool": stored_p_pool,
        "parent_seed": np.arange(3193, 3449, dtype=np.int32),
    }


def test_public_driver_refuses_before_any_file_read(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def forbidden_open(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("file read occurred")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    monkeypatch.setattr(np, "load", forbidden_open)
    with pytest.raises(PermissionError, match="not authorized"):
        analysis.run_canonical_parent_key_overlap_read_only_analysis()
    assert calls == 0


def test_lineage_constants_and_pair_order_are_frozen() -> None:
    assert analysis.DESIGN_COMMIT == "cf5f8a7911686d7c18c9ed48a793b1115b508b1a"
    assert analysis.DESIGN_SHA256 == "cb1762024b93552a6bbeb7f3aae851bd38da1e0e3226a006d6f886a341a1a062"
    assert analysis.POSTMORTEM_COMMIT == "809045cad0dbe8fe4e81d2514409a7aa02fc0fa9"
    assert analysis.POSTMORTEM_SHA256 == "c009b2466aefa8f6224254cbb4e761b408c39c671e2e9c1002432620d195fa24"
    assert analysis.PAIR_ORDER == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def test_valid_analysis_factorization_overlap_and_immutable_results() -> None:
    result = analysis._analyze_arrays(**_fixture())
    assert tuple(result["pair_order"]) == analysis.PAIR_ORDER
    assert result["pair_L1_matrix"].shape == (4, 4)
    assert result["factorization_max_abs_residual"] <= analysis.TOLERANCE
    assert result["pooling_max_abs_residual"] <= analysis.TOLERANCE
    assert result["cache_logmeanexp_max_abs_residual"] <= analysis.TOLERANCE
    assert np.all(
        result["replicate_cache_log_Z_bar_max_abs_residual"] <= analysis.TOLERANCE
    )
    assert np.allclose(np.sum(result["evidence_pooling_weights"]), 1.0)
    assert np.all(result["replicate_unique_key_count"] == np.array([2, 2, 2, 2]))
    for pair in result["pairs"]:
        assert 0.0 <= pair["key_Jaccard"] <= 1.0
        assert 0.0 <= pair["weighted_key_overlap"] <= 1.0
        assert pair["coordinate_marginal_TV"].shape == (6,)
        assert set(pair["top_k_cumulative_fraction"]) == set(analysis.TOP_K)
    with pytest.raises(ValueError):
        result["pair_L1_matrix"][0, 1] = 99.0
    with pytest.raises(TypeError):
        result["new"] = 1


def test_key_and_parent_ranking_ties_are_deterministic() -> None:
    keys = np.array(
        [[1, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        dtype=np.int16,
    )
    key_order = analysis._rank_keys(keys, np.array([0.5, 0.5, 0.7], dtype=np.float64))
    assert key_order.tolist() == [2, 1, 0]
    seeds = np.arange(3193, 3449, dtype=np.int32)
    contributions = np.zeros(256, dtype=np.float64)
    contributions[[0, 1, 2]] = [0.5, 0.7, 0.7]
    parent_order = analysis._rank_parents(seeds, contributions)
    assert parent_order[:3].tolist() == [1, 2, 0]


def test_coordinate_marginal_uses_sorted_union_and_missing_zero() -> None:
    tv = analysis._categorical_marginal_tv(
        np.array([-1, 0], dtype=np.int16),
        np.array([0.25, 0.75], dtype=np.float64),
        np.array([0, 1], dtype=np.int16),
        np.array([0.5, 0.5], dtype=np.float64),
    )
    assert tv == pytest.approx(0.5, abs=0.0)


@pytest.mark.parametrize("field", ["dtype", "shape", "finite", "normalization"])
def test_malformed_replicate_inputs_are_rejected(field: str) -> None:
    fixture = _fixture()
    if field == "dtype":
        fixture["replicate_keys"][0] = fixture["replicate_keys"][0].astype(np.int32)
    elif field == "shape":
        fixture["replicate_keys"][0] = fixture["replicate_keys"][0][:, :5]
    elif field == "finite":
        fixture["replicate_weights"][0][0] = np.nan
    else:
        fixture["replicate_weights"][0] *= 0.5
    with pytest.raises(analysis.AnalysisContractError):
        analysis._analyze_arrays(**fixture)


def test_cache_order_duplicate_factorization_and_pool_mutations_are_rejected() -> None:
    fixture = _fixture()
    fixture["cache_keys"] = fixture["cache_keys"][[1, 0, 2, 3]]
    fixture["cache_log_z"] = fixture["cache_log_z"][[1, 0, 2, 3]]
    with pytest.raises(analysis.AnalysisContractError, match="sorted"):
        analysis._analyze_arrays(**fixture)

    fixture = _fixture()
    fixture["cache_keys"][1] = fixture["cache_keys"][0]
    with pytest.raises(analysis.AnalysisContractError):
        analysis._analyze_arrays(**fixture)

    fixture = _fixture()
    fixture["stored_p_rep"][0, 0] += 1.0e-6
    fixture["stored_p_rep"][0, 1] -= 1.0e-6
    with pytest.raises(analysis.AnalysisContractError, match="factorization"):
        analysis._analyze_arrays(**fixture)

    fixture = _fixture()
    fixture["stored_p_pool"][0] += 1.0e-6
    fixture["stored_p_pool"][1] -= 1.0e-6
    with pytest.raises(analysis.AnalysisContractError, match="pool"):
        analysis._analyze_arrays(**fixture)


def test_cache_logmeanexp_and_replicate_log_z_bar_mutations_are_rejected() -> None:
    fixture = _fixture()
    fixture["cache_log_z_bar"][0] += 1.0e-6
    with pytest.raises(analysis.AnalysisContractError, match="logmeanexp"):
        analysis._analyze_arrays(**fixture)

    fixture = _fixture()
    fixture["replicate_log_z_bar"][0][0] += 1.0e-6
    with pytest.raises(analysis.AnalysisContractError, match="cache mismatch"):
        analysis._analyze_arrays(**fixture)
