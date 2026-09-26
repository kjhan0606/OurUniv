from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hong2021_dc_correction import (
    ENSEMBLE_SCHEMA,
    SCHEMA,
    apply,
    balanced_moments,
    fit_ridge,
    predict,
)
from hong2021_residual_v12_gaussianized import SCHEMA as V12_ENSEMBLE_SCHEMA


def test_source_balanced_moments_ignore_unequal_sample_counts() -> None:
    tng = np.zeros((10, 2), dtype=np.float64)
    simba = np.full((2, 2), 2.0, dtype=np.float64)
    mean, std = balanced_moments(tng, simba)
    np.testing.assert_allclose(mean, 1.0)
    np.testing.assert_allclose(std, 1.0)


def test_weighted_ridge_recovers_linear_dc_with_unequal_sources() -> None:
    tng_x = np.arange(10, dtype=np.float64)[:, None]
    simba_x = np.arange(3, dtype=np.float64)[:, None] + 20.0
    tng_y = 0.25 + 0.4 * tng_x[:, 0]
    simba_y = 0.25 + 0.4 * simba_x[:, 0]
    beta = fit_ridge(tng_x, tng_y, simba_x, simba_y, 0.0)
    np.testing.assert_allclose(beta, [0.25, 0.4], atol=1.0e-12, rtol=0)


def _write_apply_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    rng = np.random.default_rng(91)
    data_path = root / "data.h5"
    cache_path = root / "cache.h5"
    ensemble_path = root / "v12.h5"
    model_path = root / "model.json"
    observable = rng.normal(size=(3, 2, 4, 4, 4)).astype(np.float32)
    observable[:, 0] = np.abs(observable[:, 0])
    truth = rng.normal(size=(3, 1, 4, 4, 4)).astype(np.float32)
    mean = rng.normal(size=(2, 1, 4, 4, 4)).astype(np.float32)
    residual = rng.normal(size=(2, 3, 1, 4, 4, 4)).astype(np.float32)
    residual -= residual.mean(axis=(-3, -2, -1), keepdims=True)
    samples = mean[:, None] + residual
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("input", data=observable)
        handle.create_dataset("target", data=truth)
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["input_preprocessing"] = json.dumps({"mode": "faithful"})
    with h5py.File(ensemble_path, "w") as handle:
        handle.create_dataset("sample", data=samples, chunks=(1, 1, 1, 4, 4, 4))
        handle.create_dataset("conditional_mean", data=mean)
        handle.create_dataset("truth", data=truth[[2, 0]])
        handle.create_dataset("source_index", data=np.array([2, 0]))
        handle.attrs["schema"] = V12_ENSEMBLE_SCHEMA
        handle.attrs["diagnostic_k_h_mpc"] = 0.3
    model = {
        "schema": SCHEMA,
        "feature_mean": [0.0] * 8,
        "feature_std": [1.0] * 8,
        # A constant correction makes the invariants exact and independent of
        # the observable fixture values.
        "coefficients": [0.125] + [0.0] * 8,
    }
    model_path.write_text(json.dumps(model))
    return data_path, cache_path, ensemble_path, model_path


def test_apply_shifts_mean_and_every_member_without_changing_residual(
    tmp_path: Path,
) -> None:
    data, cache, ensemble, model = _write_apply_fixture(tmp_path)
    output = tmp_path / "v13.h5"
    with h5py.File(ensemble, "r") as before:
        old_mean = before["conditional_mean"][:]
        old_sample = before["sample"][:]
        old_truth = before["truth"][:]
    apply(
        argparse.Namespace(
            ensemble=str(ensemble),
            data=str(data),
            preprocessing_cache=str(cache),
            model=str(model),
            out=str(output),
        )
    )
    with h5py.File(output, "r") as after:
        assert after.attrs["schema"] == ENSEMBLE_SCHEMA
        assert not bool(after.attrs["dc_prediction_uses_target"])
        new_mean = after["conditional_mean"][:]
        new_sample = after["sample"][:]
        np.testing.assert_allclose(new_mean, old_mean + 0.125)
        np.testing.assert_allclose(new_sample, old_sample + 0.125)
        np.testing.assert_array_equal(after["truth"][:], old_truth)
        np.testing.assert_allclose(
            new_sample - new_mean[:, None],
            old_sample - old_mean[:, None],
            atol=2.0e-7,
            rtol=0,
        )


def test_predict_uses_only_serialized_features() -> None:
    model = {
        "feature_mean": [1.0, 2.0],
        "feature_std": [2.0, 4.0],
        "coefficients": [0.5, 2.0, -3.0],
    }
    actual = predict(np.array([[3.0, 6.0]]), model)
    np.testing.assert_allclose(actual, [-0.5])
