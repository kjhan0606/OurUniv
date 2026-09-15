from __future__ import annotations

import json
import subprocess

import h5py
import numpy as np
import pytest

import hong2021_v20_development_gate as gate
from hong2021_v14_edm import V20_E8_SCHEMA, resolve_edm_p_mean


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


def test_v20_gate_accepts_the_mode_emitted_by_training(tmp_path, monkeypatch) -> None:
    _, produced_mode = resolve_edm_p_mean(
        0.9999915369331587, fixed_p_mean=0.0, sigma_data_fraction=0.6
    )
    assert produced_mode == "log_sigma_data_fraction"
    registry = {
        "e8_gaussianized_marginal_retrain": {
            "initialization_and_normalization": {
                "sigma_data": 0.9999915369331587,
            },
        },
    }
    training = tmp_path / "training"
    training.mkdir()
    (training / "run.json").write_text(json.dumps({
        "status": "complete",
        "schema": V20_E8_SCHEMA,
        "experiment_registry_sha256": gate.FROZEN_REGISTRY_SHA256,
        "edm_p_mean": gate.P_MEAN,
        "edm_p_std": gate.P_STD,
        "edm_p_mean_mode": produced_mode,
        "sigma_data": 0.9999915369331587,
    }))
    monkeypatch.setattr(gate, "load_frozen_registry", lambda path, repo: registry)

    class ReachedPastRunMetadata(RuntimeError):
        pass

    def reached(_registry):
        raise ReachedPastRunMetadata

    monkeypatch.setattr(gate, "_remeasure_variance", reached)
    with pytest.raises(ReachedPastRunMetadata):
        gate.evaluate(
            root=tmp_path / "candidates", training=training,
            registry_path=tmp_path / "registry.json", repo=tmp_path,
            gate_code_commit="a" * 40,
        )


def test_v20_sampling_commit_may_be_an_ancestor_of_gate_commit(monkeypatch) -> None:
    class Result:
        returncode = 0

    seen = []

    def fake_run(command, **kwargs):
        seen.append(command)
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    sampling = "a" * 40
    gate_commit = "b" * 40
    assert gate._sampling_commit_is_ancestor(sampling, gate_commit)
    assert seen[-1][-2:] == [sampling, gate_commit]
