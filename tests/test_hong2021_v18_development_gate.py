from __future__ import annotations

import h5py
import json
from pathlib import Path
import pytest

import hong2021_v18_development_gate as gate
from hong2021_v18_development_gate import (
    _e3_candidate,
    _failure_classification,
    _only_swift_peak_void,
    _validate_checkpoint,
)
from hong2021_v15_development_gate import canonical_digest
from hong2021_v18_init import sha256_file


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


def _domain(*, gate_pass: bool, p1a: bool, p1b: bool, ks: float = 0.25):
    checks = {name: True for name in CHECK_NAMES}
    if not gate_pass:
        checks["two_point_improves_deterministic_all_scales"] = False
    return {
        "field_gate": {"pass": gate_pass, "checks": checks},
        "report_only_residual_power_over_truth": {
            "P1a_delta_at_least_0.15": p1a,
            "P1b_ratio_at_least_0.75": p1b,
        },
        "two_point_0_1_mpc_h": {"generated": ks},
    }


def test_v18_failure_classification_is_fixed_by_p1a_then_p1b() -> None:
    row = {"domains": {name: _domain(gate_pass=False, p1a=False, p1b=False) for name in ("tng", "simba_dev", "swift_dev")}}
    assert _failure_classification(row)["class"] == "P1a_failed_discard_initialization_hypothesis"
    row = {"domains": {name: _domain(gate_pass=False, p1a=True, p1b=False) for name in ("tng", "simba_dev", "swift_dev")}}
    assert _failure_classification(row)["class"] == "P1a_passed_P1b_failed_audit_score_sigma_coverage"
    row = {"domains": {name: _domain(gate_pass=False, p1a=True, p1b=True, ks=0.18) for name in ("tng", "simba_dev", "swift_dev")}}
    result = _failure_classification(row)
    assert result["class"] == "P1a_P1b_passed_audit_remaining_conditional_structure"
    assert result["pooled_ks_adjudication_signature_triggered"] is True


def test_v18_conditional_scale_branch_requires_only_swift_peak_failure() -> None:
    domains = {
        "tng": _domain(gate_pass=True, p1a=True, p1b=True),
        "simba_dev": _domain(gate_pass=True, p1a=True, p1b=True),
        "swift_dev": _domain(gate_pass=False, p1a=True, p1b=True),
    }
    swift = domains["swift_dev"]["field_gate"]["checks"]
    swift["two_point_improves_deterministic_all_scales"] = True
    swift["all_selected_peak_void_statistics_improve_deterministic"] = False
    assert _only_swift_peak_void({"domains": domains}) is True
    domains["tng"]["field_gate"]["pass"] = False
    assert _only_swift_peak_void({"domains": domains}) is False


def test_v18_development_gate_rejects_inference_grid_ensemble(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "wrong_grid.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sample", shape=(16, 16, 1, 80, 80, 80), dtype="f4")
        handle.create_dataset("conditional_mean", shape=(16, 1, 80, 80, 80), dtype="f4")
        handle.create_dataset("truth", shape=(16, 1, 80, 80, 80), dtype="f4")
        handle.create_dataset("source_index", data=list(range(16)))
    monkeypatch.setattr(gate, "_validate_ensemble", lambda *args, **kwargs: None)
    registry = json.loads(
        Path("config/hong2021_v18_development_program.json").read_text()
    )
    with pytest.raises(ValueError, match=r"64\^3"):
        gate._validate_e6_ensemble(
            path,
            registry=registry,
            registry_path=tmp_path / "registry.json",
            domain="tng",
            step=5000,
            checkpoint_path=tmp_path / "checkpoint.pt",
            expected_indices=list(range(16)),
        )


def test_v18_development_gate_rejects_missing_initialization_metadata(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "missing_init.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sample", shape=(16, 16, 1, 64, 64, 64), dtype="f4")
        handle.create_dataset("conditional_mean", shape=(16, 1, 64, 64, 64), dtype="f4")
        handle.create_dataset("truth", shape=(16, 1, 64, 64, 64), dtype="f4")
        handle.create_dataset("source_index", data=list(range(16)))
    monkeypatch.setattr(gate, "_validate_ensemble", lambda *args, **kwargs: None)
    registry = json.loads(
        Path("config/hong2021_v18_development_program.json").read_text()
    )
    with pytest.raises(ValueError, match="initialization metadata"):
        gate._validate_e6_ensemble(
            path,
            registry=registry,
            registry_path=tmp_path / "registry.json",
            domain="tng",
            step=5000,
            checkpoint_path=tmp_path / "checkpoint.pt",
            expected_indices=list(range(16)),
        )


def test_v18_checkpoint_sha_mismatch_is_rejected_before_loading(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-the-frozen-checkpoint")
    registry = {
        "parent_evidence": {
            "e3_checkpoints": [{
                "step": 5000, "path": str(checkpoint), "sha256": "0" * 64,
            }]
        }
    }
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        _validate_checkpoint(registry, 5000)


def test_v18_e3_decision_digest_mutation_is_rejected(tmp_path) -> None:
    path = tmp_path / "e3.json"
    decision = {"candidates": [{"step": 5000}], "development_pass": False}
    decision["decision_digest_sha256"] = canonical_digest(decision)
    path.write_text(json.dumps(decision))
    registry = {"parent_evidence": {
        "e3_decision": str(path),
        "e3_decision_sha256": sha256_file(path),
        "e3_decision_digest_sha256": canonical_digest(decision),
    }}
    assert _e3_candidate(registry, 5000)["step"] == 5000
    decision["candidates"][0]["step"] = 10000
    path.write_text(json.dumps(decision))
    with pytest.raises(ValueError, match="decision changed"):
        _e3_candidate(registry, 5000)
