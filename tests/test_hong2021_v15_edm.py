from __future__ import annotations

import argparse
import json

import pytest

from hong2021_v14_edm import V15_E2_SCHEMA, V15_E3_SCHEMA
from hong2021_v15_edm import frozen_training_namespace


def _arguments(tmp_path, experiment: str) -> argparse.Namespace:
    registry = {
        "schema": "hong2021-v15-predeclared-development-program-v1",
        "e2_noise_distribution": {
            "steps": 10000,
            "candidate_steps": [5000, 10000],
            "tail_exponent": 0.5,
            "tail_maximum": 10.0,
            "training_seed": 144021,
            "validation_seeds": {"TNG100": 99173, "SIMBA": 99174, "Swift": 99175},
        },
        "e3_tail_weight": {
            "steps": 10000,
            "candidate_steps": [5000, 10000],
            "tail_exponent": 0.25,
            "tail_maximum": 10.0,
            "training_seed": 144021,
            "validation_seeds": {"TNG100": 99173, "SIMBA": 99174, "Swift": 99175},
        },
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))
    values = {
        "experiment": experiment,
        "registry": registry_path,
        "repo": tmp_path,
        "out": str(tmp_path / "out"),
        "workers": 1,
        "device": "cpu",
    }
    for prefix in ("tng_train", "tng_validation", "simba_train", "simba_validation", "swift_train", "swift_validation"):
        values[f"{prefix}_data"] = f"{prefix}_data.h5"
        values[f"{prefix}_cache"] = f"{prefix}_cache.h5"
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    ("experiment", "schema", "tail"),
    (("e2", V15_E2_SCHEMA, 0.5), ("e3", V15_E3_SCHEMA, 0.25)),
)
def test_frozen_namespace_exposes_only_predeclared_change(
    tmp_path, monkeypatch, experiment, schema, tail
) -> None:
    monkeypatch.setattr("hong2021_v15_edm.git_state", lambda repo: ("a" * 40, True))
    actual = frozen_training_namespace(_arguments(tmp_path, experiment))
    assert actual.steps == 10000
    assert actual.candidate_steps == "5000,10000"
    assert actual.edm_p_mean_sigma_data_fraction == 0.6
    assert actual.edm_p_std == 1.2
    assert actual.tail_exponent == tail
    assert actual.run_schema == schema
    assert actual.code_commit_at_launch == "a" * 40


def test_frozen_namespace_rejects_dirty_worktree(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("hong2021_v15_edm.git_state", lambda repo: ("a" * 40, False))
    with pytest.raises(RuntimeError, match="clean committed"):
        frozen_training_namespace(_arguments(tmp_path, "e2"))
