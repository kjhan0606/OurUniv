import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import hong2021_v79_complete_gate as gate


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/hong2021_v79_complete_candidate_agnostic_gate_program.json"


def test_program_is_byte_bound_and_preserves_implementation_only_firewall() -> None:
    assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == gate.PROGRAM_SHA256
    program = json.loads(PROGRAM.read_text())
    assert program["schema"] == gate.PROGRAM_SCHEMA
    assert program["status"] == gate.PROGRAM_STATUS
    authorization = program["implementation_only_authorization"]
    assert authorization["implement_and_unit_test_candidate_agnostic_gate"] is True
    assert authorization["read_any_selected_validation_input_or_target"] is False
    assert authorization["construct_train_sample_or_evaluate_candidate"] is False
    assert authorization["run_V79_gate"] is False


def test_frozen_selection_has_exact_disjoint_32_query_contract() -> None:
    program = json.loads(PROGRAM.read_text())
    selection = gate.frozen_indices(program)
    assert set(selection) == set(gate.DOMAIN_ORDER)
    assert all(len(indices) == 32 and len(set(indices)) == 32 for indices in selection.values())
    assert selection["TNG100"][:4] == [10, 11, 13, 20]
    assert selection["SIMBA"][-4:] == [53, 54, 57, 58]
    assert selection["Swift"][-4:] == [125, 128, 132, 141]


def test_exact_keys_rejects_unbound_extras() -> None:
    gate._exact_keys({"path": "x", "sha256": "y"}, {"path", "sha256"}, "row")
    with pytest.raises(ValueError, match="keys differ"):
        gate._exact_keys(
            {"path": "x", "sha256": "y", "unbound": "z"},
            {"path", "sha256"},
            "row",
        )


def test_bitwise_equality_rejects_signed_zero_and_dtype_changes() -> None:
    positive = np.asarray([0.0, 1.0], dtype=np.float32)
    negative = np.asarray([-0.0, 1.0], dtype=np.float32)
    assert not gate._bitwise_equal(positive, negative)
    assert not gate._bitwise_equal(positive, positive.astype(np.float64))
    assert gate._bitwise_equal(positive, positive.copy())


def test_environment_requires_strict_improvement_for_every_frozen_field() -> None:
    environment = {name: {} for name in ("truth", "deterministic", "generated")}
    for field in gate.ENVIRONMENT_FIELDS:
        environment["truth"][field] = {"mean": 10.0}
        environment["deterministic"][field] = {"mean": 12.0}
        environment["generated"][field] = {"mean": 11.0}
    assert gate.environment_improves({"environment": environment})
    environment["generated"][gate.ENVIRONMENT_FIELDS[0]]["mean"] = 12.0
    assert not gate.environment_improves({"environment": environment})


def _small_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    queries = 2
    spatial = (2, 2, 2)
    sample_shape = (queries, gate.MEMBERS, 1, *spatial)
    field_shape = (queries, 1, *spatial)
    monkeypatch.setattr(gate, "QUERIES", queries)
    monkeypatch.setattr(gate, "SAMPLE_SHAPE", sample_shape)
    monkeypatch.setattr(gate, "FIELD_SHAPE", field_shape)
    monkeypatch.setattr(gate, "SOURCE_SHAPE", (queries,))
    paths = (tmp_path / "candidate.h5", tmp_path / "control.h5")
    latent_digest = np.arange(queries * gate.MEMBERS * 32, dtype=np.uint8).reshape(
        queries, gate.MEMBERS, 32
    )
    pairing_digest = hashlib.sha256(latent_digest.tobytes()).hexdigest()
    truth = np.linspace(-0.1, 0.1, np.prod(field_shape), dtype=np.float32).reshape(field_shape)
    mean = np.zeros(field_shape, dtype=np.float32)
    for offset, path in enumerate(paths):
        with h5py.File(path, "w") as handle:
            sample = np.broadcast_to(mean[:, None], sample_shape).copy()
            sample += np.float32(offset * 0.01)
            sample -= sample.mean(axis=(-3, -2, -1), keepdims=True)
            handle.create_dataset("sample", data=sample)
            handle.create_dataset("truth", data=truth)
            handle.create_dataset("conditional_mean", data=mean)
            handle.create_dataset("source_index", data=np.asarray([3, 5]))
            handle.create_dataset("initial_latent_sha256", data=latent_digest)
            handle.attrs["sampler"] = "frozen"
            handle.attrs["innovation_pairing_digest"] = pairing_digest
    return paths


def test_ensemble_pair_checks_shapes_equality_pairing_and_dc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, control = _small_pair(tmp_path, monkeypatch)
    with h5py.File(candidate, "r") as handle:
        pairing_digest = hashlib.sha256(
            handle["initial_latent_sha256"][:].tobytes()
        ).hexdigest()
    result = gate.validate_ensemble_pair(
        candidate,
        control,
        [3, 5],
        {"sampler": "frozen"},
        {"sampler": "frozen"},
        {
            "rule": "same frozen innovation",
            "innovation_pairing_digest": pairing_digest,
            "candidate_control_pairing_proven": True,
        },
    )
    assert result["residual_DC_pass"]
    assert result["truth_conditional_mean_source_index_and_optional_input_bitwise_equal"]
    assert result["maximum_absolute_residual_DC"] == 0.0


def test_ensemble_pair_rejects_truth_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, control = _small_pair(tmp_path, monkeypatch)
    with h5py.File(candidate, "r") as handle:
        pairing_digest = hashlib.sha256(
            handle["initial_latent_sha256"][:].tobytes()
        ).hexdigest()
    with h5py.File(control, "r+") as handle:
        handle["truth"][0, 0, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="truth differs"):
        gate.validate_ensemble_pair(
            candidate,
            control,
            [3, 5],
            {"sampler": "frozen"},
            {"sampler": "frozen"},
            {
                "rule": "same frozen innovation",
                "innovation_pairing_digest": pairing_digest,
                "candidate_control_pairing_proven": True,
            },
        )


def test_energy_score_is_querywise_and_uses_all_16_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "QUERIES", 2)
    path = tmp_path / "ensemble.h5"
    sample = np.zeros((2, 16, 1, 2, 2, 2), dtype=np.float32)
    sample[0, :, 0, 0, 0, 0] = np.arange(16, dtype=np.float32) / 100
    sample[1, :, 0, 0, 0, 0] = np.arange(16, dtype=np.float32) / 200
    truth = np.zeros((2, 1, 2, 2, 2), dtype=np.float32)
    truth[:, 0, 0, 0, 0] = [0.05, 0.02]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sample", data=sample)
        handle.create_dataset("truth", data=truth)
    result = gate.cube_maximum_energy_score(path)
    expected = [
        gate.scalar_energy_score(4.5 * sample[index, :, 0].max(axis=(-3, -2, -1)), 4.5 * truth[index, 0].max())
        for index in range(2)
    ]
    assert result["truth_cubes"] == 2
    assert np.allclose(result["per_query_unbiased_energy_score"], expected)
    assert np.isclose(result["mean_unbiased_energy_score"], np.mean(expected))


def test_rank_coverage_block_uses_six_p_values_and_bonferroni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "QUERIES", 2)
    monkeypatch.setattr(gate, "RELABELINGS", 99)
    rng = np.random.default_rng(79001)
    table = gate.v75.build_table_from_fields(rng.normal(size=(2, 17, 64)), 79002)
    monkeypatch.setattr(gate, "ensemble_label_table", lambda path, seed: table)
    domains = {
        domain: {
            "candidate_ensemble": Path(f"/{domain}.h5"),
            "candidate_ensemble_sha256": str(index + 1) * 64,
        }
        for index, domain in enumerate(gate.DOMAIN_ORDER)
    }
    block, result = gate.rank_coverage_observation(domains)
    assert len(result["six_individual_p_values"]) == 6
    assert block == min(1.0, 6.0 * min(result["six_individual_p_values"]))
    assert all(row["random_relabelings"] == 99 for row in result["domains"].values())


def test_source_requires_explicit_authorization_and_has_no_runner() -> None:
    source = (REPO / "src/hong2021_v79_complete_gate.py").read_text()
    assert "explicit_user_approval_for_candidate_design_and_single_use_execution" in source
    assert 'or authorization.get("V79_single_use_gate_execution") is not True' in source
    assert '"prior_gate_disclosure": False' in source
    assert '"retry_authorized": False' in source
    assert '"V72_stage_B_accessed": False' in source
    assert not (REPO / "scripts/hong2021_v79_complete_gate_lageunha.sh").exists()
