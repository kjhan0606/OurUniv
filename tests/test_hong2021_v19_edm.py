from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

import hong2021_v19_edm as v19
from hong2021_v14_edm import V19_E7_SCHEMA, decoder_upsampling_for_schema


def test_v19_registry_derives_the_frozen_noise_distribution() -> None:
    registry = json.loads(Path("config/hong2021_v19_development_program.json").read_text())
    experiment = registry["e7_band_anchored_noise_retrain"]
    v19._validate_exact_experiment(experiment)
    noise = experiment["training_noise"]
    p_mean, p_std, components = v19.derive_band_anchored_noise(
        noise["source_balanced_per_band_mode_variance"]
    )
    assert p_mean == v19.P_MEAN
    assert p_std == v19.P_STD
    assert components == noise["component_log_medians"]
    assert decoder_upsampling_for_schema(V19_E7_SCHEMA) == "nearest"


@pytest.mark.parametrize("mutation", ("p_mean", "p_std", "seed", "steps"))
def test_v19_exact_experiment_rejects_scientific_mutation(mutation: str) -> None:
    registry = json.loads(Path("config/hong2021_v19_development_program.json").read_text())
    experiment = copy.deepcopy(registry["e7_band_anchored_noise_retrain"])
    if mutation == "p_mean":
        experiment["training_noise"]["p_mean"] += 0.01
    elif mutation == "p_std":
        experiment["training_noise"]["p_std"] += 0.01
    elif mutation == "seed":
        experiment["sampling_seeds"]["TNG100"] += 1
    else:
        experiment["steps"] += 1
    with pytest.raises(ValueError):
        v19._validate_exact_experiment(experiment)


def test_v19_namespace_changes_only_registered_noise_distribution(
    tmp_path, monkeypatch
) -> None:
    registry = json.loads(Path("config/hong2021_v19_development_program.json").read_text())
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("stub")
    monkeypatch.setattr(v19, "load_frozen_registry", lambda path, repo: registry)
    monkeypatch.setattr(v19, "git_state", lambda repo: ("a" * 40, True))
    args = argparse.Namespace(
        repo=tmp_path, registry=registry_path, out=tmp_path / "out", device="cpu"
    )
    actual = v19.frozen_training_namespace(args)
    assert actual.run_schema == V19_E7_SCHEMA
    assert actual.edm_p_mean == v19.P_MEAN
    assert actual.edm_p_std == v19.P_STD
    assert actual.edm_p_mean_sigma_data_fraction is None
    assert actual.steps == 10000
    assert actual.candidate_steps == "5000,10000"
    assert actual.tail_exponent == 0.25
    assert actual.seed == 144021
    assert actual.validation_seeds == [99173, 99174, 99175]
    assert actual.code_commit_at_launch == "a" * 40


def test_v19_namespace_rejects_existing_output(tmp_path, monkeypatch) -> None:
    registry = json.loads(Path("config/hong2021_v19_development_program.json").read_text())
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("stub")
    output = tmp_path / "out"
    output.mkdir()
    monkeypatch.setattr(v19, "load_frozen_registry", lambda path, repo: registry)
    monkeypatch.setattr(v19, "git_state", lambda repo: ("a" * 40, True))
    args = argparse.Namespace(
        repo=tmp_path, registry=registry_path, out=output, device="cpu"
    )
    with pytest.raises(RuntimeError, match="pre-existing training output"):
        v19.frozen_training_namespace(args)
