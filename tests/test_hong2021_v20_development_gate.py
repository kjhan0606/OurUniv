from __future__ import annotations

import h5py
import numpy as np
import pytest

import hong2021_v20_development_gate as gate


def _domain(*, q3: float, maximum: float, q4: float, q5: bool = True):
    checks = {name: q5 for name in gate.Q5_CHECKS}
    return {
        "field_gate": {"checks": checks},
        "mechanism_Q3_Q4": {
            "delta_q99_999_dex": q3,
            "generated_max_above_truth_max_dex": maximum,
            "generated_over_truth_mean_delta_squared": q4,
        },
    }


def test_v20_mechanism_passes_are_all_domain_and_selection_free() -> None:
    registry = {
        "e8_gaussianized_marginal_retrain": {"mechanism_diagnostics": {
            "Q3": {
                "per_domain_absolute_delta_q99_999_log10rho_max_dex": 0.10,
                "per_domain_generated_max_log10rho_above_truth_max_dex": 0.30,
            },
            "Q4": {"per_domain_generated_over_truth_mean_delta_squared_max": 1.5},
        }}
    }
    final = {"domains": {
        "tng": _domain(q3=0.10, maximum=0.30, q4=1.5),
        "simba_dev": _domain(q3=-0.10, maximum=0.20, q4=1.2),
        "swift_dev": _domain(q3=0.0, maximum=0.0, q4=1.0),
    }}
    assert gate._mechanism_passes(registry, final) == (True, True, True)
    final["domains"]["swift_dev"] = _domain(q3=0.11, maximum=0.0, q4=1.0)
    assert gate._mechanism_passes(registry, final) == (False, True, True)


def test_v20_marginal_diagnostics_uses_unique_truth_and_all_generated(tmp_path) -> None:
    path = tmp_path / "ensemble.h5"
    truth = np.array([0.0, 0.1], dtype=np.float32).reshape(2, 1, 1, 1, 1)
    sample = np.array([[0.0, 0.0], [0.1, 0.2]], dtype=np.float32).reshape(
        2, 2, 1, 1, 1, 1
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("truth", data=truth)
        handle.create_dataset("sample", data=sample)
    result = gate.marginal_diagnostics(path)
    truth_delta = np.power(10.0, 4.5 * truth.astype(np.float64)) - 1.0
    generated_delta = np.power(10.0, 4.5 * sample.astype(np.float64)) - 1.0
    assert result["truth_mean_delta_squared"] == float(np.mean(truth_delta**2))
    assert result["generated_mean_delta_squared"] == float(
        np.mean(generated_delta**2)
    )
    assert result["generated_max_log10rho"] == pytest.approx(0.9)
