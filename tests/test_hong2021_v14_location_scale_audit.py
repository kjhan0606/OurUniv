from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from hong2021_residual_v8_context import FEATURE_NAMES
from hong2021_v14_baseline_audit import BAND_EDGES_H_MPC, SCHEMA as CACHE_SCHEMA
from hong2021_v14_location_scale_audit import (
    balanced_moments,
    fit_source_balanced_ridge,
    load_cache,
    ridge_predict,
)


def test_three_source_balanced_moments_ignore_source_sizes() -> None:
    sources = [
        np.zeros((20, 2)),
        np.ones((3, 2)),
        np.full((7, 2), 2.0),
    ]
    mean, std = balanced_moments(sources)
    np.testing.assert_allclose(mean, 1.0)
    np.testing.assert_allclose(std, np.sqrt(2.0 / 3.0))


def test_multioutput_ridge_recovers_shared_linear_relation() -> None:
    features = [
        np.arange(12, dtype=np.float64)[:, None],
        np.arange(4, dtype=np.float64)[:, None] + 20.0,
        np.arange(7, dtype=np.float64)[:, None] - 15.0,
    ]
    targets = [
        np.column_stack((0.25 + 0.4 * value[:, 0], -0.5 + 2.0 * value[:, 0]))
        for value in features
    ]
    mean, std, coefficients = fit_source_balanced_ridge(features, targets, 0.0)
    actual = ridge_predict(np.array([[3.5]]), mean, std, coefficients)
    np.testing.assert_allclose(actual, [[1.65, 6.5]], atol=1.0e-12, rtol=0)


def _write_cache(path: Path, domain: str) -> None:
    rng = np.random.default_rng(17)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "observable_context_features", data=rng.normal(size=(5, len(FEATURE_NAMES)))
        )
        handle.create_dataset("residual_dc", data=rng.normal(size=5))
        handle.create_dataset("residual_band_rms", data=np.exp(rng.normal(size=(5, 4))))
        handle.attrs["schema"] = CACHE_SCHEMA
        handle.attrs["domain"] = domain
        handle.attrs["feature_uses_target"] = False
        handle.attrs["feature_names"] = json.dumps(list(FEATURE_NAMES))
        handle.attrs["band_edges_h_mpc"] = json.dumps(BAND_EDGES_H_MPC)


def test_cache_loader_checks_declared_domain_and_firewall(tmp_path: Path) -> None:
    cache = tmp_path / "cache.h5"
    _write_cache(cache, "TNG100")
    features, targets = load_cache(cache, "TNG100")
    assert features.shape == (5, 8)
    assert targets.shape == (5, 5)
    with pytest.raises(ValueError, match="does not match"):
        load_cache(cache, "CAMELS-SIMBA")
    forbidden = tmp_path / "Astrid_cache.h5"
    _write_cache(forbidden, "TNG100")
    with pytest.raises(ValueError, match="forbidden"):
        load_cache(forbidden, "TNG100")
