"""Execution-free parent/key overlap diagnostics for the sealed grammar v2 run."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


DESIGN_COMMIT = "cf5f8a7911686d7c18c9ed48a793b1115b508b1a"
DESIGN_SHA256 = "cb1762024b93552a6bbeb7f3aae851bd38da1e0e3226a006d6f886a341a1a062"
POSTMORTEM_COMMIT = "809045cad0dbe8fe4e81d2514409a7aa02fc0fa9"
POSTMORTEM_SHA256 = "c009b2466aefa8f6224254cbb4e761b408c39c671e2e9c1002432620d195fa24"
PAIR_ORDER = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
TOP_K = (1, 5, 10, 20, 50)
N_REPLICATES = 4
N_PARENTS = 256
TOLERANCE = 1.0e-12


class AnalysisContractError(ValueError):
    """Raised when a synthetic input violates the frozen analysis contract."""


def run_canonical_parent_key_overlap_read_only_analysis() -> None:
    """Refuse before any canonical path or sealed artifact can be read."""

    raise PermissionError(
        "canonical parent/key analysis execution and artifact reads are not authorized"
    )


def _require_array(
    value: np.ndarray,
    *,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int | None, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(dtype):
        raise AnalysisContractError(f"{name} dtype mismatch")
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        raise AnalysisContractError(f"{name} shape mismatch")
    if not np.all(np.isfinite(array)):
        raise AnalysisContractError(f"{name} is not finite")
    return array


def _require_probability(value: np.ndarray, *, name: str) -> np.ndarray:
    array = _require_array(
        value, name=name, dtype=np.float64, shape=(value.shape[0],)
    )
    if np.any(array < 0.0) or not np.isclose(
        np.sum(array, dtype=np.float64), 1.0, rtol=0.0, atol=TOLERANCE
    ):
        raise AnalysisContractError(f"{name} is not a normalized probability")
    return array


def _lexicographic_order(keys: np.ndarray) -> np.ndarray:
    return np.lexsort(tuple(keys[:, column] for column in range(5, -1, -1)))


def _rank_keys(keys: np.ndarray, scores: np.ndarray) -> np.ndarray:
    keys = _require_array(keys, name="rank keys", dtype=np.int16, shape=(None, 6))
    scores = _require_array(
        scores, name="rank key scores", dtype=np.float64, shape=(keys.shape[0],)
    )
    return np.lexsort(
        (
            keys[:, 5],
            keys[:, 4],
            keys[:, 3],
            keys[:, 2],
            keys[:, 1],
            keys[:, 0],
            -scores,
        )
    )


def _rank_parents(parent_seed: np.ndarray, contributions: np.ndarray) -> np.ndarray:
    parent_seed = _require_array(
        parent_seed, name="parent seed", dtype=np.int32, shape=(N_PARENTS,)
    )
    contributions = _require_array(
        contributions,
        name="parent contributions",
        dtype=np.float64,
        shape=(N_PARENTS,),
    )
    return np.lexsort((parent_seed, -contributions))


def _categorical_marginal_tv(
    values_a: np.ndarray,
    masses_a: np.ndarray,
    values_b: np.ndarray,
    masses_b: np.ndarray,
) -> float:
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    masses_a = np.asarray(masses_a, dtype=np.float64)
    masses_b = np.asarray(masses_b, dtype=np.float64)
    if values_a.ndim != 1 or values_b.ndim != 1:
        raise AnalysisContractError("categorical values must be one dimensional")
    if masses_a.shape != values_a.shape or masses_b.shape != values_b.shape:
        raise AnalysisContractError("categorical mass shape mismatch")
    if not np.all(np.isfinite(masses_a)) or not np.all(np.isfinite(masses_b)):
        raise AnalysisContractError("categorical mass is not finite")
    support = np.union1d(values_a, values_b)
    aligned_a = np.zeros(support.shape, dtype=np.float64)
    aligned_b = np.zeros(support.shape, dtype=np.float64)
    np.add.at(aligned_a, np.searchsorted(support, values_a), masses_a)
    np.add.at(aligned_b, np.searchsorted(support, values_b), masses_b)
    return float(0.5 * np.sum(np.abs(aligned_a - aligned_b), dtype=np.float64))


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        frozen = np.array(value, copy=True)
        frozen.setflags(write=False)
        return frozen
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _analyze_arrays(
    *,
    replicate_keys: Sequence[np.ndarray],
    replicate_weights: Sequence[np.ndarray],
    cache_keys: np.ndarray,
    cache_log_z: np.ndarray,
    cache_log_z_bar: np.ndarray,
    replicate_log_z_bar: Sequence[np.ndarray],
    stored_p_rep: np.ndarray,
    log_i_bar: np.ndarray,
    stored_p_pool: np.ndarray,
    parent_seed: np.ndarray,
) -> Mapping[str, Any]:
    """Analyze already-loaded synthetic arrays without filesystem or RNG access."""

    if (
        len(replicate_keys) != N_REPLICATES
        or len(replicate_weights) != N_REPLICATES
        or len(replicate_log_z_bar) != N_REPLICATES
    ):
        raise AnalysisContractError("exactly four replicates are required")
    cache_keys = _require_array(
        cache_keys, name="cache keys", dtype=np.int16, shape=(None, 6)
    )
    cache_log_z = _require_array(
        cache_log_z,
        name="cache log_Z",
        dtype=np.float64,
        shape=(cache_keys.shape[0], N_PARENTS),
    )
    if cache_keys.shape[0] == 0:
        raise AnalysisContractError("cache is empty")
    cache_log_z_bar = _require_array(
        cache_log_z_bar,
        name="cache log_Z_bar",
        dtype=np.float64,
        shape=(cache_keys.shape[0],),
    )
    if not np.array_equal(_lexicographic_order(cache_keys), np.arange(cache_keys.shape[0])):
        raise AnalysisContractError("cache keys are not lexicographically sorted")
    if np.any(np.all(cache_keys[1:] == cache_keys[:-1], axis=1)):
        raise AnalysisContractError("cache keys are not unique")
    parent_seed = _require_array(
        parent_seed, name="parent seed", dtype=np.int32, shape=(N_PARENTS,)
    )
    if np.unique(parent_seed).size != N_PARENTS:
        raise AnalysisContractError("parent seeds are not unique")
    stored_p_rep = _require_array(
        stored_p_rep,
        name="stored P_rep",
        dtype=np.float64,
        shape=(N_REPLICATES, N_PARENTS),
    )
    if np.any(stored_p_rep < 0.0) or not np.allclose(
        np.sum(stored_p_rep, axis=1), 1.0, rtol=0.0, atol=TOLERANCE
    ):
        raise AnalysisContractError("stored P_rep is not normalized")
    log_i_bar = _require_array(
        log_i_bar, name="log_I_bar", dtype=np.float64, shape=(N_REPLICATES,)
    )
    stored_p_pool = _require_probability(stored_p_pool, name="stored P_pool")
    if stored_p_pool.shape != (N_PARENTS,):
        raise AnalysisContractError("stored P_pool shape mismatch")

    row_max = np.max(cache_log_z, axis=1, keepdims=True)
    reconstructed_cache_log_z_bar = row_max[:, 0] + np.log(
        np.mean(np.exp(cache_log_z - row_max), axis=1)
    )
    cache_logmeanexp_residual = float(
        np.max(np.abs(reconstructed_cache_log_z_bar - cache_log_z_bar))
    )
    if cache_logmeanexp_residual > TOLERANCE:
        raise AnalysisContractError("cache logmeanexp mismatch")
    parent_given_key = np.exp(cache_log_z - row_max)
    parent_given_key /= np.sum(parent_given_key, axis=1, keepdims=True)
    key_index = {tuple(row.tolist()): index for index, row in enumerate(cache_keys)}
    key_mass = np.zeros((N_REPLICATES, cache_keys.shape[0]), dtype=np.float64)
    replicate_cache_log_z_bar_residual = np.zeros(N_REPLICATES, dtype=np.float64)

    for replicate, (keys_value, weights_value, log_z_bar_value) in enumerate(
        zip(replicate_keys, replicate_weights, replicate_log_z_bar, strict=True)
    ):
        keys = _require_array(
            keys_value,
            name=f"replicate {replicate} keys",
            dtype=np.int16,
            shape=(None, 6),
        )
        weights = _require_probability(
            weights_value, name=f"replicate {replicate} weights"
        )
        if weights.shape != (keys.shape[0],):
            raise AnalysisContractError("replicate key/weight length mismatch")
        try:
            indices = np.fromiter(
                (key_index[tuple(row.tolist())] for row in keys),
                dtype=np.int64,
                count=keys.shape[0],
            )
        except KeyError as error:
            raise AnalysisContractError("replicate key is absent from cache") from error
        log_z_bar = _require_array(
            log_z_bar_value,
            name=f"replicate {replicate} log_Z_bar",
            dtype=np.float64,
            shape=(keys.shape[0],),
        )
        replicate_cache_log_z_bar_residual[replicate] = np.max(
            np.abs(log_z_bar - cache_log_z_bar[indices]), initial=0.0
        )
        if replicate_cache_log_z_bar_residual[replicate] > TOLERANCE:
            raise AnalysisContractError("replicate log_Z_bar cache mismatch")
        np.add.at(key_mass[replicate], indices, weights)

    reconstructed_p_rep = key_mass @ parent_given_key
    factorization_residual = float(
        np.max(np.abs(reconstructed_p_rep - stored_p_rep))
    )
    if factorization_residual > TOLERANCE:
        raise AnalysisContractError("key to parent factorization mismatch")
    shifted_log_i = log_i_bar - np.max(log_i_bar)
    pooling_weights = np.exp(shifted_log_i)
    pooling_weights /= np.sum(pooling_weights)
    reconstructed_p_pool = pooling_weights @ stored_p_rep
    pooling_residual = float(np.max(np.abs(reconstructed_p_pool - stored_p_pool)))
    if pooling_residual > TOLERANCE:
        raise AnalysisContractError("evidence weighted pool mismatch")

    pair_l1_matrix = np.zeros((N_REPLICATES, N_REPLICATES), dtype=np.float64)
    pair_outputs: list[dict[str, Any]] = []
    for replicate_a, replicate_b in PAIR_ORDER:
        signed = stored_p_rep[replicate_a] - stored_p_rep[replicate_b]
        contribution = np.abs(signed)
        total = float(np.sum(contribution, dtype=np.float64))
        pair_l1_matrix[replicate_a, replicate_b] = total
        pair_l1_matrix[replicate_b, replicate_a] = total
        fraction = contribution / total if total > 0.0 else np.zeros_like(contribution)
        parent_order = _rank_parents(parent_seed, contribution)
        cumulative = np.cumsum(contribution[parent_order], dtype=np.float64)
        key_score = np.abs(key_mass[replicate_a] - key_mass[replicate_b])
        key_order = _rank_keys(cache_keys, key_score)
        support_a = key_mass[replicate_a] > 0.0
        support_b = key_mass[replicate_b] > 0.0
        intersection = int(np.count_nonzero(support_a & support_b))
        union = int(np.count_nonzero(support_a | support_b))
        jaccard = float(intersection / union) if union else 1.0
        overlap = float(
            np.sum(np.minimum(key_mass[replicate_a], key_mass[replicate_b]))
        )
        marginal_tv = np.array(
            [
                _categorical_marginal_tv(
                    cache_keys[:, coordinate],
                    key_mass[replicate_a],
                    cache_keys[:, coordinate],
                    key_mass[replicate_b],
                )
                for coordinate in range(6)
            ],
            dtype=np.float64,
        )
        if not (0.0 <= overlap <= 1.0 and 0.0 <= jaccard <= 1.0):
            raise AnalysisContractError("overlap metric is outside [0,1]")
        pair_outputs.append(
            {
                "pair": (replicate_a, replicate_b),
                "L1": total,
                "signed_parent_difference": signed,
                "parent_contribution": contribution,
                "parent_fractional_contribution": fraction,
                "parent_rank_order": parent_order,
                "parent_seed_ranked": parent_seed[parent_order],
                "top_k_cumulative_fraction": {
                    top_k: float(cumulative[min(top_k, N_PARENTS) - 1] / total)
                    if total > 0.0
                    else 0.0
                    for top_k in TOP_K
                },
                "key_score": key_score,
                "key_rank_order": key_order,
                "keys_ranked": cache_keys[key_order],
                "key_intersection": intersection,
                "key_union": union,
                "key_Jaccard": jaccard,
                "weighted_key_overlap": overlap,
                "coordinate_marginal_TV": marginal_tv,
            }
        )

    pair_values = np.array([row["L1"] for row in pair_outputs], dtype=np.float64)
    maximum_pair_index = int(np.argmax(pair_values))
    replicate_to_pool_l1 = np.sum(
        np.abs(stored_p_rep - stored_p_pool[None, :]), axis=1
    )
    result = {
        "pair_order": PAIR_ORDER,
        "pair_L1_matrix": pair_l1_matrix,
        "pairs": pair_outputs,
        "maximum_pair": PAIR_ORDER[maximum_pair_index],
        "replicate_unique_key_count": np.count_nonzero(key_mass > 0.0, axis=1),
        "key_mass": key_mass,
        "parent_given_key": parent_given_key,
        "reconstructed_cache_log_Z_bar": reconstructed_cache_log_z_bar,
        "cache_logmeanexp_max_abs_residual": cache_logmeanexp_residual,
        "replicate_cache_log_Z_bar_max_abs_residual": replicate_cache_log_z_bar_residual,
        "reconstructed_P_rep": reconstructed_p_rep,
        "factorization_max_abs_residual": factorization_residual,
        "evidence_pooling_weights": pooling_weights,
        "reconstructed_P_pool": reconstructed_p_pool,
        "pooling_max_abs_residual": pooling_residual,
        "replicate_to_pool_L1": replicate_to_pool_l1,
    }
    return _freeze(result)
