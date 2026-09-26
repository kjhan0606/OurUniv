from __future__ import annotations

import h5py
import json
from pathlib import Path

import pytest

import hong2021_v19_development_gate as gate
from hong2021_v14_edm import V19_E7_SCHEMA
from hong2021_v19_edm import FROZEN_REGISTRY_SHA256, P_MEAN, P_STD


CHECK_NAMES = (
    "high_k_total_power_within_10_percent",
    "residual_rms_within_10_percent",
    "rank_histogram_tv_at_most_0.05",
    "finite_ensemble_coverage_within_0.03",
    "density_pdf_improves_deterministic",
    "two_point_improves_deterministic_all_scales",
    "all_selected_peak_void_statistics_improve_deterministic",
    "exact_dc_projection",
)


def _domain(*, passed: bool, q2: bool = True):
    checks = {name: True for name in CHECK_NAMES}
    if not passed:
        checks["two_point_improves_deterministic_all_scales"] = False
    return {
        "field_gate": {"pass": passed, "checks": checks},
        "mechanism_Q2": {"pass": q2},
        "two_point_by_scale": {
            "0-1_mpc_h": {"improves_deterministic": passed},
            "1-3_mpc_h": {"improves_deterministic": True},
            "3-10_mpc_h": {"improves_deterministic": True},
        },
    }


def test_v19_expected_failure_routes_to_symmetric_two_point_audit(monkeypatch) -> None:
    final = {"domains": {
        "tng": _domain(passed=False),
        "simba_dev": _domain(passed=True),
        "swift_dev": _domain(passed=True),
    }}
    monkeypatch.setattr(gate, "_previously_passing_regressions", lambda *args: [])
    result = gate.classify_failure(
        registry={}, final=final, q1_pass=True, repo=Path(".")
    )
    assert result["Q2_all_domains"] is True
    assert result["TNG_only_0_1_two_point_signature"] is True
    assert result["next"] == "stop_unopened_and_audit_symmetric_2pcf_plus_nonlinear_stage1_band0_predictability"


def test_v19_tng_signature_does_not_require_other_domains_to_pass(monkeypatch) -> None:
    final = {"domains": {
        "tng": _domain(passed=False),
        "simba_dev": _domain(passed=False),
        "swift_dev": _domain(passed=False),
    }}
    monkeypatch.setattr(gate, "_previously_passing_regressions", lambda *args: [])
    result = gate.classify_failure(
        registry={}, final=final, q1_pass=True, repo=Path(".")
    )
    assert result["TNG_only_0_1_two_point_signature"] is True


def test_v19_q1_failure_routes_to_architecture_audit(monkeypatch) -> None:
    final = {"domains": {name: _domain(passed=False) for name in (
        "tng", "simba_dev", "swift_dev"
    )}}
    monkeypatch.setattr(gate, "_previously_passing_regressions", lambda *args: [])
    monkeypatch.setattr(gate, "_tng_fails_only_zero_to_one", lambda *args: False)
    result = gate.classify_failure(
        registry={}, final=final, q1_pass=False, repo=Path(".")
    )
    assert result["class"] == "sigma_coverage_falsified_audit_architecture_representation"


def test_v19_regression_forbids_noise_interpolation(monkeypatch) -> None:
    final = {"domains": {name: _domain(passed=False) for name in (
        "tng", "simba_dev", "swift_dev"
    )}}
    monkeypatch.setattr(gate, "_previously_passing_regressions", lambda *args: ["tng:pdf"])
    monkeypatch.setattr(gate, "_tng_fails_only_zero_to_one", lambda *args: False)
    result = gate.classify_failure(
        registry={}, final=final, q1_pass=True, repo=Path(".")
    )
    assert result["next"] == "stop_unopened_discard_v19_distribution_without_interpolation"


def test_v19_ensemble_rejects_wrong_development_grid(tmp_path, monkeypatch) -> None:
    path = tmp_path / "wrong.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sample", shape=(16, 16, 1, 80, 80, 80), dtype="f4")
        handle.create_dataset("conditional_mean", shape=(16, 1, 80, 80, 80), dtype="f4")
        handle.create_dataset("truth", shape=(16, 1, 80, 80, 80), dtype="f4")
        handle.create_dataset("source_index", data=list(range(16)))
    monkeypatch.setattr(gate, "_validate_ensemble", lambda *args, **kwargs: None)
    registry = json.loads(Path("config/hong2021_v19_development_program.json").read_text())
    with pytest.raises(ValueError, match=r"64\^3"):
        gate._validate_e7_ensemble(
            path, registry=registry, domain="tng", step=5000,
            checkpoint_path=tmp_path / "checkpoint.pt", checkpoint_sha="0" * 64,
            expected_indices=list(range(16)), gate_commit="a" * 40,
        )


def test_v19_ensemble_requires_frozen_noise_and_registry_metadata(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "missing.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sample", shape=(16, 16, 1, 64, 64, 64), dtype="f4")
        handle.create_dataset("conditional_mean", shape=(16, 1, 64, 64, 64), dtype="f4")
        handle.create_dataset("truth", shape=(16, 1, 64, 64, 64), dtype="f4")
        handle.create_dataset("source_index", data=list(range(16)))
    monkeypatch.setattr(gate, "_validate_ensemble", lambda *args, **kwargs: None)
    registry = json.loads(Path("config/hong2021_v19_development_program.json").read_text())
    with pytest.raises(ValueError, match="metadata differs"):
        gate._validate_e7_ensemble(
            path, registry=registry, domain="tng", step=5000,
            checkpoint_path=tmp_path / "checkpoint.pt", checkpoint_sha="0" * 64,
            expected_indices=list(range(16)), gate_commit="a" * 40,
        )
