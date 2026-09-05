from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from hong2021_v14_edm import (
    V20_E8_SCHEMA,
    V20_GAUSSIANIZED_CACHE_SCHEMA,
    V14ResidualDataset,
    decoder_upsampling_for_schema,
)
from hong2021_v20_gaussianize import exact_zero_dc_projection, transform_cube
from hong2021_v20_gaussianize import (
    CACHE_SCHEMA,
    SOURCE_CACHE_SCHEMA,
    TRANSFORM_SCHEMA,
    prepare_inference_cache,
    sha256_file,
)


def test_v20_exact_zero_dc_projection_is_deterministic_and_bounded() -> None:
    generator = np.random.default_rng(200081)
    value = generator.normal(size=(64, 64, 64)).astype(np.float32)
    value = (value - value.mean(dtype=np.float64)).astype(np.float32)
    first, first_audit = exact_zero_dc_projection(value)
    second, second_audit = exact_zero_dc_projection(value)
    assert np.array_equal(first, second)
    assert first_audit == second_audit
    assert first_audit["index"] == int(np.argmin(np.abs(value.reshape(-1))))
    assert first_audit["passes"] in (0, 1, 2)
    assert abs(first_audit["adjustment"]) <= 0.03
    assert first_audit["final_ortho_dc"] <= 1.0e-9


def test_v20_transform_cube_stores_latent_dc_but_emits_zero_dc() -> None:
    transform = {
        "z_knots": [-5.0, 0.0, 5.0],
        "residual_value_knots": [-2.0, 0.0, 2.0],
    }
    source = np.linspace(-1.5, 1.75, 64**3, dtype=np.float32).reshape((64,) * 3)
    value, latent_dc, audit = transform_cube(source, transform)
    assert latent_dc != 0.0
    assert value.dtype == np.float32
    assert audit["final_ortho_dc"] <= 1.0e-9
    assert abs(float(value.sum(dtype=np.float64))) / 512.0 <= 1.0e-9


def test_v20_projection_and_inference_cache_support_frozen_80_grid(tmp_path) -> None:
    source = tmp_path / "astrid_standardized.h5"
    transform_path = tmp_path / "gaussianization.json"
    output = tmp_path / "astrid_gaussianized.h5"
    transform_path.write_text(json.dumps({
        "schema": TRANSFORM_SCHEMA,
        "z_knots": [-5.0, 0.0, 5.0],
        "residual_value_knots": [-2.0, 0.0, 2.0],
    }))
    value = np.linspace(-1.0, 1.0, 80**3, dtype=np.float32).reshape((1, 1, 80, 80, 80))
    with h5py.File(source, "w") as handle:
        handle.attrs.update({
            "schema": SOURCE_CACHE_SCHEMA,
            "complete": True,
            "scale_prediction_uses_target": False,
            "voxel_mpc_h": 0.3125,
            "input_preprocessing": json.dumps({"mode": "test"}),
        })
        handle.create_dataset("standardized_residual", data=value)
        handle.create_dataset("conditional_mean", data=np.zeros_like(value))
        handle.create_dataset("predicted_residual_dc", data=np.zeros(1, dtype=np.float32))
        handle.create_dataset("predicted_band_scales", data=np.ones((1, 4), dtype=np.float32))
    report = prepare_inference_cache(source, output, transform_path)
    assert report["grid"] == 80
    assert report["objects"] == 1
    assert report["maximum_absolute_ortho_dc"] <= 1.0e-9
    assert report["sha256"] == sha256_file(output)
    with h5py.File(output, "r") as handle:
        assert str(handle.attrs["schema"]) == CACHE_SCHEMA
        assert bool(handle.attrs["complete"])
        assert handle["standardized_residual"].shape == (1, 1, 80, 80, 80)
        ortho_dc = abs(
            float(handle["standardized_residual"][0, 0].sum(dtype=np.float64))
        ) / np.sqrt(80**3)
        assert ortho_dc <= 1.0e-9


def test_v14_dataset_accepts_only_registered_v20_cache_schema(tmp_path) -> None:
    data = tmp_path / "data.h5"
    cache = tmp_path / "cache.h5"
    with h5py.File(data, "w") as handle:
        handle.create_dataset("input", shape=(1, 4, 4, 4, 4), dtype="f4")
        handle.create_dataset("target", shape=(1, 1, 4, 4, 4), dtype="f4")
    with h5py.File(cache, "w") as handle:
        handle.attrs.update({
            "schema": V20_GAUSSIANIZED_CACHE_SCHEMA,
            "complete": True,
            "scale_prediction_uses_target": False,
            "input_preprocessing": json.dumps({"mode": "test"}),
            "voxel_mpc_h": 0.3125,
            "standardized_residual_rms": 1.0,
        })
        handle.create_dataset("conditional_mean", shape=(1, 1, 4, 4, 4), dtype="f4")
        handle.create_dataset("standardized_residual", shape=(1, 1, 4, 4, 4), dtype="f4")
        handle.create_dataset("predicted_residual_dc", shape=(1,), dtype="f4")
        handle.create_dataset("predicted_band_scales", data=np.ones((1, 4), dtype=np.float32))
    dataset = V14ResidualDataset(data, cache, False)
    assert dataset.cache_schema == V20_GAUSSIANIZED_CACHE_SCHEMA
    assert decoder_upsampling_for_schema(V20_E8_SCHEMA) == "nearest"


def test_v20_registry_binds_final_fable_addenda_and_artifacts() -> None:
    registry = json.loads(Path("config/hong2021_v20_development_program.json").read_text())
    experiment = registry["e8_gaussianized_marginal_retrain"]
    assert registry["independent_audit"]["record_sha256"] == (
        "ba42a2262163c182d2dc8fdcf899851a7350345de157e232fdf1fe717fc31201"
    )
    assert experiment["dc_projection"]["maximum_postprojection_ortho_dc"] == 1e-9
    assert experiment["dc_projection"]["measured_all_six_caches"]["pass_2_count"] == 0
    assert experiment["gaussianization"]["validation_or_test_data_used"] is False
