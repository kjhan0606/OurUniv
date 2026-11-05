from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_residual_evaluate import CENTERED_SCHEMAS
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import (
    DESIGN_AUDIT_SHA256,
    DOMAIN_ORDER,
    DONOR_COUNTS,
    ENSEMBLE_SCHEMA,
    REGISTRY_SHA256,
    DonorLibrary,
    pool_local_condition,
    select_donors,
    source_balanced_fit,
    source_quota,
)


REPO = Path(__file__).parents[1]


def test_v28_registry_design_hash_and_firewall():
    registry_path = REPO / "config/hong2021_v28_development_program.json"
    assert sha256_file(registry_path) == REGISTRY_SHA256
    registry = json.loads(registry_path.read_text())
    assert registry["design_audit"]["sha256"] == DESIGN_AUDIT_SHA256
    assert registry["donor_library"]["total_objects"] == 1043
    assert registry["approval_firewall"]["Astrid_attempts_remaining"] == 1
    assert ENSEMBLE_SCHEMA in CENTERED_SCHEMAS


def test_pool_local_condition_is_exact_block_average():
    condition = np.zeros((4, 64, 64, 64), dtype=np.float32)
    condition[0] = np.arange(64, dtype=np.float32)[:, None, None]
    pooled = pool_local_condition(condition)
    assert pooled.shape == (3, 8, 8, 8)
    assert np.allclose(pooled[0, :, 0, 0], np.arange(8) * 8 + 3.5)


def test_source_balanced_fit_weights_sources_not_objects():
    rows = {
        domain: np.full((DONOR_COUNTS[domain], 1), value, dtype=np.float32)
        for domain, value in zip(DOMAIN_ORDER, (0.0, 3.0, 6.0), strict=True)
    }
    fit = source_balanced_fit(rows)
    assert np.allclose(fit["mean"], [3.0])
    assert np.allclose(fit["std"], [np.sqrt(6.0)])


def test_source_quota_is_exactly_balanced_over_48_queries():
    totals = {domain: 0 for domain in DOMAIN_ORDER}
    for position in range(48):
        quota = source_quota(position)
        assert sorted(quota.values()) == [5, 5, 6]
        assert sum(quota.values()) == 16
        for domain in DOMAIN_ORDER:
            totals[domain] += quota[domain]
    assert totals == {domain: 256 for domain in DOMAIN_ORDER}


def test_matching_finds_orientation_and_unique_source_quota():
    rng = np.random.default_rng(18)
    query = rng.normal(size=(3, 8, 8, 8)).astype(np.float32)
    transform = 17
    permutation, reflections = CUBE_ISOMETRIES[transform]
    donor = apply_cube_isometry(query, permutation, reflections)
    local = {}
    global_rows = {}
    for domain in DOMAIN_ORDER:
        rows = np.full(
            (DONOR_COUNTS[domain], 3, 8, 8, 8), 20.0, dtype=np.float32
        )
        rows[:8] = donor
        local[domain] = rows
        global_rows[domain] = np.zeros((DONOR_COUNTS[domain], 8), dtype=np.float32)
    library = DonorLibrary(
        local=local,
        global_features=global_rows,
        local_fit={"mean": [0.0] * 3, "std": [1.0] * 3},
        global_fit={"mean": [0.0] * 8, "std": [1.0] * 8},
        data_paths={},
        cache_paths={},
    )
    selected = select_donors(
        query, np.zeros(8, dtype=np.float32), library, global_query_position=0
    )
    assert len(selected) == 16
    assert {domain: sum(row["source"] == domain for row in selected) for domain in DOMAIN_ORDER} == source_quota(0)
    assert len({(row["source"], row["donor_index"]) for row in selected}) == 16
    assert max(row["local_distance"] for row in selected) < 1.0e-12
