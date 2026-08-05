from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from hong2021_residual_diffusion import ConditionalResidualUNet
from hong2021_residual_v8_context import FEATURE_NAMES
from hong2021_v14_baseline_audit import SCHEMA as BASELINE_SCHEMA
from hong2021_v14_location_scale import SCHEMA as LOCATION_SCHEMA
from hong2021_v14_mean_correction import SCHEMA as CORRECTION_SCHEMA
from hong2021_v14_residual_cache import (
    CORRECTED_SCHEMA,
    STANDARDIZED_SCHEMA,
    prepare_corrected,
    prepare_standardized,
)


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(44)
    data = root / "data.h5"
    baseline = root / "baseline.h5"
    checkpoint = root / "correction.pt"
    observable = rng.normal(size=(2, 2, 8, 8, 8)).astype(np.float32)
    observable[:, 0] = np.abs(observable[:, 0])
    truth = rng.normal(size=(2, 1, 8, 8, 8)).astype(np.float32)
    mean = rng.normal(size=(2, 1, 8, 8, 8)).astype(np.float32)
    residual = truth - mean
    centered = residual - residual.mean(axis=(-3, -2, -1), keepdims=True)
    with h5py.File(data, "w") as handle:
        handle.create_dataset("input", data=observable)
        handle.create_dataset("target", data=truth)
        handle.attrs["voxel_mpc_h"] = 0.3125
    with h5py.File(baseline, "w") as handle:
        handle.create_dataset("conditional_mean", data=mean)
        handle.create_dataset("centered_residual", data=centered)
        handle.attrs["schema"] = BASELINE_SCHEMA
        handle.attrs["domain"] = "TNG100"
        handle.attrs["feature_uses_target"] = False
        handle.attrs["input_preprocessing"] = json.dumps({"mode": "faithful"})
    model = ConditionalResidualUNet(base_channels=4)
    torch.nn.init.zeros_(model.output.weight)
    torch.nn.init.zeros_(model.output.bias)
    torch.save(
        {
            "schema": CORRECTION_SCHEMA,
            "base_channels": 4,
            "step": 1,
            "ema_model": model.state_dict(),
        },
        checkpoint,
    )
    return data, baseline, checkpoint


def test_corrected_and_standardized_cache_end_to_end(tmp_path: Path) -> None:
    data, baseline, checkpoint = _fixture(tmp_path)
    corrected = tmp_path / "corrected.h5"
    report = prepare_corrected(
        argparse.Namespace(
            domain="TNG100",
            data=str(data),
            baseline_cache=str(baseline),
            correction_checkpoint=str(checkpoint),
            out=str(corrected),
            batch=2,
            workers=0,
            device="cpu",
        )
    )
    assert report["samples"] == 2
    with h5py.File(corrected, "r") as handle:
        assert handle.attrs["schema"] == CORRECTED_SCHEMA
        assert bool(handle.attrs["complete"])
        np.testing.assert_allclose(
            handle["centered_residual"][:].mean(axis=(-3, -2, -1)),
            0.0,
            atol=2.0e-8,
        )

    location = tmp_path / "location.json"
    location.write_text(
        json.dumps(
            {
                "schema": LOCATION_SCHEMA,
                "feature_names": list(FEATURE_NAMES),
                "feature_mean": [0.0] * len(FEATURE_NAMES),
                "feature_std": [1.0] * len(FEATURE_NAMES),
                "location": {"coefficients": [0.0] * (len(FEATURE_NAMES) + 1)},
                "bands": [
                    {"kind": "constant_log_rms", "log_scale": float(np.log(value))}
                    for value in (0.1, 0.2, 0.3, 0.4)
                ],
            }
        )
    )
    standardized = tmp_path / "standardized.h5"
    standardized_report = prepare_standardized(
        argparse.Namespace(
            corrected_cache=str(corrected),
            location_scale_model=str(location),
            out=str(standardized),
            chunk=1,
        )
    )
    assert standardized_report["standardized_residual_rms"] > 0
    with h5py.File(standardized, "r") as handle:
        assert handle.attrs["schema"] == STANDARDIZED_SCHEMA
        assert bool(handle.attrs["complete"])
        np.testing.assert_allclose(
            handle["predicted_band_scales"][:],
            np.array([[0.1, 0.2, 0.3, 0.4]] * 2),
            rtol=1.0e-6,
        )
