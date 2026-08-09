import hashlib
from pathlib import Path

import numpy as np

from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v32_context_audit import (
    FEATURE_NAMES,
    PROGRAM_SHA256,
    block_context_rows,
    fit_source_balanced_ridge,
    native_multiplicity,
    parity_subsample,
    ridge_metrics,
    source_balanced_standardization,
)


REPO = Path(__file__).resolve().parents[1]


def test_v32_program_hash_and_firewall():
    path = REPO / "config/hong2021_v32_context_audit_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text().lower()
    assert '"posthoc_ak": false' in text
    assert '"astrid_access": "forbidden"' in text
    assert '"historical_eagle_access": "forbidden"' in text


def test_native_multiplicity_counts_galaxies_not_only_cells():
    count = np.asarray([0, 1, 2, 3, 0], dtype=np.float64)
    result = native_multiplicity(count)
    assert result == {
        "cells": 5,
        "occupied_cells": 3,
        "multi_galaxy_cells": 2,
        "galaxies": 6,
        "galaxies_in_multi_galaxy_cells": 5,
    }


def test_block_velocity_dispersion_lower_bound_and_shapes():
    count = np.zeros((64, 64, 64), dtype=np.float64)
    velocity = np.zeros_like(count)
    count[0, 0, 0] = 1
    count[1, 0, 0] = 1
    velocity[0, 0, 0] = 0
    velocity[1, 0, 0] = 10
    backbone = np.zeros_like(count)
    truth = np.ones_like(count) * 0.2
    feature, target, diagnostic = block_context_rows(
        count, velocity, backbone, truth, 4
    )
    assert feature.shape == (16, 16, 16, len(FEATURE_NAMES))
    assert target.shape == (16, 16, 16)
    assert np.isclose(feature[0, 0, 0, 1], 5.0)
    assert np.isclose(feature[0, 0, 0, 5], 5.0)
    assert np.allclose(target, 0.2)
    assert diagnostic["velocity_dispersion_nonzero_fraction"] == 1 / 16**3


def test_parity_subsample_uses_one_eighth_of_factor4_grid():
    value = np.arange(16**3).reshape(16, 16, 16)
    selected = parity_subsample(value, 7)
    assert selected.shape == (8, 8, 8)
    offsets = {
        tuple(slice_.start for slice_ in (
            slice(index % 2, None, 2),
            slice((index // 2) % 2, None, 2),
            slice((index // 4) % 2, None, 2),
        ))
        for index in range(8)
    }
    assert len(offsets) == 8


def test_source_balanced_ridge_recovers_common_linear_relation():
    rows = {}
    targets = {}
    for source, n, offset in zip(DOMAIN_ORDER, (20, 200, 50), (-1.0, 0.0, 1.0), strict=True):
        x = np.linspace(-2.0, 2.0, n) + offset
        feature = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
        feature[:, 0] = x
        rows[source] = feature
        targets[source] = 1.5 + 2.0 * x
    mean, std = source_balanced_standardization(rows)
    coefficient = fit_source_balanced_ridge(
        rows, targets, (0,), mean, std, ridge_lambda=0.0
    )
    for source in DOMAIN_ORDER:
        result = ridge_metrics(
            rows[source], targets[source], (0,), mean, std, coefficient
        )
        assert result["rmse"] < 1.0e-10
        assert np.isclose(result["pearson_prediction_target"], 1.0)
