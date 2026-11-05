from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from hong2021_v16_development_gate import (
    _load_expected_indices,
    _next_after_failure,
    _validate_source_indices,
)
from hong2021_v15_development_gate import sha256_file


def _candidate(tmp_path, generated: float, deterministic: float, *, swift_peak: bool):
    metrics = {
        "candidates": {
            "edm": {
                "two_point_cosmic_mean": {
                    "generated_vs_truth_ks": {
                        "by_scale": {"0-1_mpc_h": {"mean": generated}}
                    },
                    "deterministic_vs_truth_ks": {
                        "by_scale": {"0-1_mpc_h": {"mean": deterministic}}
                    },
                }
            }
        }
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics))
    names = (
        "high_k_total_power_within_10_percent",
        "residual_rms_within_10_percent",
        "rank_histogram_tv_at_most_0.05",
        "finite_ensemble_coverage_within_0.03",
        "density_pdf_improves_deterministic",
        "two_point_improves_deterministic_all_scales",
        "all_selected_peak_void_statistics_improve_deterministic",
        "exact_dc_projection",
    )
    swift_checks = {name: True for name in names}
    swift_checks["all_selected_peak_void_statistics_improve_deterministic"] = swift_peak
    return {
        "domains": {
            "tng": {"metrics": str(path), "field_gate": {"pass": generated < deterministic}},
            "simba_dev": {"field_gate": {"pass": True}},
            "swift_dev": {
                "field_gate": {"pass": swift_peak, "checks": swift_checks}
            },
        }
    }


def test_v16_failure_branch_prioritizes_persistent_tng_sub_mpc_error(tmp_path) -> None:
    row = _candidate(tmp_path, 0.20, 0.19, swift_peak=False)
    assert _next_after_failure(row) == (
        "stop_unopened_then_predeclare_band_balanced_denoising_loss"
    )


def test_v16_failure_branch_selects_conditional_scale_only_for_swift_peak(tmp_path) -> None:
    row = _candidate(tmp_path, 0.18, 0.19, swift_peak=False)
    assert _next_after_failure(row) == (
        "stop_unopened_then_predeclare_all_band_target_free_conditional_scale"
    )


def test_v16_gate_rejects_changed_development_subset(tmp_path) -> None:
    ensemble = tmp_path / "ensemble.h5"
    with h5py.File(ensemble, "w") as handle:
        handle.create_dataset("source_index", data=np.arange(16, dtype=np.int64))
    _validate_source_indices(ensemble, list(range(16)))
    changed = list(range(15)) + [16]
    with pytest.raises(ValueError, match="frozen V16 subset"):
        _validate_source_indices(ensemble, changed)


def test_v16_gate_rejects_changed_subset_reference_file(tmp_path, monkeypatch) -> None:
    config = tmp_path / "config"
    config.mkdir()
    subset = config / "subset.json"
    subset.write_text(json.dumps({"indices": list(range(16))}) + "\n")
    monkeypatch.setattr(
        "hong2021_v16_development_gate.FROZEN_DEVELOPMENT_OBJECT_SHA256",
        {"config/subset.json": sha256_file(subset)},
    )
    assert _load_expected_indices("config/subset.json", tmp_path) == list(range(16))
    subset.write_text(json.dumps({"indices": list(range(1, 17))}) + "\n")
    with pytest.raises(ValueError, match="subset file differs from its frozen hash"):
        _load_expected_indices("config/subset.json", tmp_path)
