from __future__ import annotations

import json

import numpy as np
import torch

from hong2021_v15_diagnostics import _band_power_sums, field_trend


def test_band_power_sums_assigns_each_non_dc_mode_once() -> None:
    value = torch.randn(2, 8, 8, 8)
    masks = torch.zeros(4, 8, 8, 8)
    masks[0, :4] = 1
    masks[1, 4:] = 1
    masks[:, 0, 0, 0] = 0
    measured = _band_power_sums(value, masks).sum(dim=-1)
    spectrum = torch.fft.fftn(value, dim=(-3, -2, -1))
    expected = (spectrum.real.square() + spectrum.imag.square()).sum((-3, -2, -1))
    expected -= spectrum[:, 0, 0, 0].real.square()
    assert torch.allclose(measured, expected, rtol=1e-5, atol=1e-3)


def test_field_trend_extracts_frozen_statistics(tmp_path) -> None:
    metric = {
        "candidates": {
            "edm": {
                "checkpoint_step": 5000,
                "fourier_log_density": {
                    key: {
                        band: float(index + 1)
                        for index, band in enumerate(
                            ("0.3-1_h_mpc", "1-3_h_mpc", "3-6_h_mpc", "6-10.0531_h_mpc")
                        )
                    }
                    for key in (
                        "generated_total_power_over_truth",
                        "generated_residual_power_over_truth_residual",
                        "ensemble_mean_cross_correlation_with_truth",
                    )
                },
                "residual_calibration": {"generated_over_truth_rms": 1.02},
                "environment": {
                    group: {
                        "local_peak_count_rho_gt_10": {"mean": value},
                        "local_peak_count_rho_gt_100": {"mean": value},
                        "local_void_count_rho_lt_0.1": {"mean": value},
                    }
                    for group, value in (("truth", 2.0), ("generated", 3.0))
                },
            }
        }
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metric))
    rows = field_trend([path])
    assert rows[0]["step"] == 5000
    assert rows[0]["generated_over_truth_rms"] == 1.02
    assert np.isclose(
        rows[0]["environment_generated_over_truth"]["local_peak_count_rho_gt_100"],
        1.5,
    )
    assert len(rows[0]["metrics_sha256"]) == 64
