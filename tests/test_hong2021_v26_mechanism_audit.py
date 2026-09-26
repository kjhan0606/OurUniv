from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts/hong2021_v26_mechanism_audit.py"
SPEC = importlib.util.spec_from_file_location("hong2021_v26_mechanism_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_tensor_moments_reports_fixed_tail_and_shape_independent_moments() -> None:
    accumulator = MODULE.TensorMoments()
    accumulator.update(torch.tensor([[-6.0, -1.0], [1.0, 6.0]]))
    report = accumulator.report()
    assert report["count"] == 4
    assert report["mean"] == pytest.approx(0.0)
    assert report["standard_deviation"] == pytest.approx((18.5) ** 0.5)
    assert report["skewness"] == pytest.approx(0.0)
    assert report["absolute_tail_counts"]["5.0"] == 2
    assert report["absolute_tail_fractions"]["6.0"] == pytest.approx(0.5)


def _history_row(step: int, balanced: float, gradient: float) -> dict:
    values = [float(step) / 100_000 + index for index in range(6)]
    return {
        "step": step,
        "balanced_validation_nll": balanced,
        "train_scale_nll_coarse_to_fine": [value - 0.1 for value in values],
        "fixed_validation": {
            domain: {"scale_nll_coarse_to_fine": values}
            for domain in ("TNG100", "SIMBA", "Swift-EAGLE")
        },
        "gradient_diagnostic": {"mean_norm_before_fixed_clip": gradient},
    }


def test_history_audit_exposes_dimension_weighting_and_nonfinite_gradients() -> None:
    history = [
        _history_row(500, 1.0, 1.0),
        _history_row(10_000, 0.8, float("inf")),
        _history_row(20_000, 0.7, 0.9),
        _history_row(25_000, 0.6, 0.8),
        _history_row(30_000, 0.5, 0.7),
    ]
    report = MODULE.analyze_history(history)
    assert report["finest_scale_objective_fraction"] == pytest.approx(0.875003337873)
    assert report["nonfinite_mean_gradient_intervals"] == [10_000]
    row = report["scale_resolved"]["TNG100"][0]
    assert row["change_10000_to_30000"] == pytest.approx(0.2)
    assert row["worsened_10000_to_30000"] is True


def _domain_mechanism_row(truth_std: float, truth_maximum: float) -> dict:
    return {
        "roundtrip": {
            "maximum_absolute_latent_error": 0.01,
            "rms_latent_error": 5.0e-4,
            "maximum_absolute_logdet_cancellation_coarse_to_fine": [
                1.0e-5,
                1.0e-4,
                1.0e-3,
                1.0e-2,
                0.1,
                1.0,
            ],
        },
        "stored_ensemble_replay": {"maximum_absolute_y_difference": 1.0e-7},
        "truth_latent": {
            "absolute_maximum": truth_maximum,
            "absolute_tail_fractions": {"5.0": 1.0e-6},
        },
        "generated_latent": {
            "absolute_maximum": truth_maximum + 2.0,
            "absolute_tail_fractions": {"5.0": 4.0e-6},
        },
        "truth_base_z_coarse_to_fine": [
            {"standard_deviation": truth_std},
            {"standard_deviation": 1.2},
        ],
        "generated_base_z_coarse_to_fine": [
            {"standard_deviation": 1.0},
            {"standard_deviation": 1.0},
        ],
    }


def test_mechanism_summary_does_not_mistake_small_roundtrip_residual_for_cause() -> None:
    domains = {
        "TNG100": _domain_mechanism_row(1.6, 5.0),
        "SIMBA": _domain_mechanism_row(1.3, 5.0),
        "Swift": _domain_mechanism_row(1.2, 5.0),
    }
    optimization = {
        "scale_resolved": {
            name: [
                {"coarse_to_fine_index": index, "worsened_10000_to_30000": index < 2}
                for index in range(6)
            ]
            for name in ("TNG100", "SIMBA", "Swift-EAGLE")
        },
        "finest_scale_objective_fraction": 0.875,
        "nonfinite_mean_gradient_intervals": [4500, 6500],
    }
    report = MODULE._mechanism_summary({"30000": domains}, optimization)
    assert report["trained_flow_roundtrip_stable"] is True
    assert report["coarse_base_underfit_all_domains"] is True
    assert report["classification"].startswith("coarse_scale_conditional_underfit")
