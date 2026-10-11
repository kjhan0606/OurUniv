import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import cf4_aggregate_evidence_smc_capability as capability
from cf4_aggregate_evidence_oracle import PRODUCTION_PARENT_SEEDS
from cf4_aggregate_evidence_smc_capability import (
    CAPABILITY_DESIGN,
    TwoPhaseArtifactFirewall,
    _one_sided_tied_ks,
    _permutation_pvalue,
    classify_failure,
    evaluate_cf4_gates,
    evaluate_pre_cf4_gates,
    OracleEvaluationFailure,
    run_production_capability,
)


def test_capability_design_pins_exact_method_and_authorizes_no_execution():
    design = json.loads(CAPABILITY_DESIGN.read_text())
    assert design["fixed_smc"] == {
        "replicate_count": 4,
        "replicates_sequential": True,
        "particles_per_replicate": 2048,
        "bit_generator": "PCG64DXSM",
        "master_seeds": [2026082301, 2026082302, 2026082303, 2026082304],
        "target_CESS_fraction": 0.8,
        "resample_when_ESS_strictly_below_fraction": 0.5,
        "resampling": "systematic",
        "MH_sweeps_per_positive_temperature_stage": 4,
        "move_names": ["q_local", "axis_local", "joint_local", "prior_independence"],
        "move_probabilities": [0.4, 0.3, 0.2, 0.1],
        "q_scales": [0.25, 0.6, 1.5],
        "q_scale_probabilities": [0.5, 0.3, 0.2],
        "vMF_kappa": [100.0, 10.0, 1.0],
        "vMF_kappa_probabilities": [0.5, 0.3, 0.2],
        "maximum_positive_temperature_stages": 256,
        "runtime_override_allowed": False,
        "particle_pooling_across_replicates": False,
        "integral_averaging_across_replicates": True,
    }
    assert design["parallel_oracle"]["parent_blocks"] == [
        [3193 + 32 * index, 3224 + 32 * index] for index in range(8)
    ]
    assert design["sealed_oracle_control"]["inside_rows"] == [
        0, 68, 136, 204, 272, 341, 409, 477,
        545, 613, 682, 750, 818, 886, 954, 1023,
    ]
    assert design["sealed_oracle_control"]["outside_rows"] == [
        0, 9, 18, 27, 36, 45, 54, 63,
    ]
    assert design["authorization"][
        "capability_implementation_and_unit_tests_authorized"
    ] is True
    assert not any(
        value for key, value in design["authorization"].items()
        if key != "capability_implementation_and_unit_tests_authorized"
    )


def test_public_entry_is_program_only_and_refuses_before_private_core(monkeypatch):
    calls = []
    assert set(inspect.signature(run_production_capability).parameters) == {
        "program_path"
    }
    assert set(inspect.signature(
        capability._run_fixed_capability_core
    ).parameters) == {"oracle", "output_directory"}
    assert set(inspect.signature(
        TwoPhaseArtifactFirewall.open_calibration
    ).parameters) == {"self"}
    assert set(inspect.signature(
        TwoPhaseArtifactFirewall.publish_cf4_gates
    ).parameters) == {"self"}
    monkeypatch.setattr(
        capability, "_run_fixed_capability_core", lambda *args: calls.append(args)
    )
    with np.testing.assert_raises_regex(PermissionError, "canonical program"):
        run_production_capability(Path("not-canonical.json"))
    with np.testing.assert_raises_regex(PermissionError, "not authorized"):
        run_production_capability(capability.CANONICAL_PROGRAM)
    assert calls == []


def _fake_replicate(master_seed):
    keys = np.zeros((2048, 6), dtype=np.int16)
    keys[:, :3] = 288
    keys[:, 3] = 3
    empty_move = {
        "proposal_count": {
            "q_local": 512, "axis_local": 512,
            "joint_local": 512, "prior_independence": 512,
        },
        "acceptance_count": {
            "q_local": 256, "axis_local": 256,
            "joint_local": 256, "prior_independence": 256,
        },
        "q_scale_proposal_count": np.asarray([512, 307, 205]),
        "q_scale_acceptance_count": np.asarray([256, 153, 103]),
        "axis_scale_proposal_count": np.asarray([512, 307, 205]),
        "axis_scale_acceptance_count": np.asarray([256, 153, 103]),
    }
    return SimpleNamespace(
        master_seed=master_seed,
        midpoint_mpc_h=np.zeros((2048, 3), dtype=np.float64),
        axis=np.tile([1.0, 0.0, 0.0], (2048, 1)),
        keys=keys,
        weights=np.full(2048, 1.0 / 2048.0),
        log_z_bar=np.zeros(2048),
        ancestor_labels=np.arange(2048),
        beta_history=np.asarray([0.0, 1.0]),
        conditional_ess_history=np.asarray([2048.0]),
        particle_ess_history=np.asarray([2048.0, 2048.0]),
        log_normalizer_increment=np.asarray([0.0]),
        resampling_ancestors=[],
        move_history=[[dict(empty_move) for _ in range(4)]],
        log_normalizer=0.0,
        genealogical_ess=2048.0,
    )


class _FakeTerminalOracle:
    def __init__(self):
        self.registered = []
        self.sealed = False

    def register_terminal_history(self, master_seed, keys):
        assert not self.sealed
        self.registered.append((master_seed, keys.copy()))

    def seal_terminal_histories(self):
        assert [row[0] for row in self.registered] == [
            2026082301, 2026082302, 2026082303, 2026082304
        ]
        self.sealed = True

    def terminal_parent_log_z(self, master_seed, keys):
        assert self.sealed
        return np.zeros((2048, 256), dtype=np.float64)


def test_two_phase_firewall_excludes_cf4_and_opens_loader_only_after_seal(
    tmp_path, monkeypatch
):
    firewall = TwoPhaseArtifactFirewall(tmp_path / "output")
    loader_calls = []

    original_loader = capability._load_pinned_calibration

    def counted_loader():
        loader_calls.append(True)
        return original_loader()

    monkeypatch.setattr(capability, "_load_pinned_calibration", counted_loader)

    with np.testing.assert_raises_regex(PermissionError, "closed before"):
        firewall.open_calibration()
    assert loader_calls == []
    oracle = _FakeTerminalOracle()
    replicates = tuple(
        _fake_replicate(seed)
        for seed in (2026082301, 2026082302, 2026082303, 2026082304)
    )
    firewall.register_terminal_histories(replicates, oracle)
    assert loader_calls == []
    assert not list((tmp_path / "output").iterdir())
    frozen = firewall.seal_terminal_phase(oracle)
    assert frozen.path.is_file() and len(frozen.sha256) == 64
    with np.load(frozen.path, allow_pickle=False) as item:
        assert tuple(item.files) == (
            "master_seed", "parent_seed", "log_I_bar", "P_rep", "P_pool"
        )
        assert not any("cf4" in key.lower() for key in item.files)
        assert item["log_I_bar"].shape == (4,)
        assert item["P_rep"].shape == (4, 256)
        assert item["P_pool"].shape == (256,)
        np.testing.assert_allclose(item["P_pool"], 1.0 / 256.0)
    assert all(path.is_file() for path in sorted((tmp_path / "output").glob("replicate_*.npz")))
    calibration = firewall.open_calibration()
    assert calibration.reference_q99 == 19851.02909664355
    assert calibration.source_sha256 == capability.CALIBRATION_SHA256
    np.testing.assert_array_equal(calibration.parent_seed, PRODUCTION_PARENT_SEEDS)
    terminal, genealogy, temperature = firewall._reload_phase_one_artifacts()
    np.testing.assert_array_equal(terminal["log_I_bar"], np.zeros(4))
    np.testing.assert_array_equal(genealogy, np.full(4, 2048.0))
    assert all(temperature.values())
    assert loader_calls == [True]


def test_batch_registration_rejects_malformed_matrix_before_any_oracle_call(tmp_path):
    cases = []
    bad_weight = [_fake_replicate(seed) for seed in capability.PRODUCTION_REPLICATE_MASTER_SEEDS]
    bad_weight[-1].weights[0] += 0.1
    cases.append(bad_weight)
    bad_key = [_fake_replicate(seed) for seed in capability.PRODUCTION_REPLICATE_MASTER_SEEDS]
    bad_key[-1].keys[0, 0] += 1
    cases.append(bad_key)
    bad_move = [_fake_replicate(seed) for seed in capability.PRODUCTION_REPLICATE_MASTER_SEEDS]
    bad_move[-1].move_history[0][0]["proposal_count"]["q_local"] -= 1
    cases.append(bad_move)
    bad_resampling = [
        _fake_replicate(seed)
        for seed in capability.PRODUCTION_REPLICATE_MASTER_SEEDS
    ]
    bad_resampling[-1].resampling_ancestors.append(
        np.arange(2048, dtype=np.int64)
    )
    cases.append(bad_resampling)
    for index, replicates in enumerate(cases):
        oracle = _FakeTerminalOracle()
        firewall = TwoPhaseArtifactFirewall(tmp_path / f"malformed_{index}")
        with np.testing.assert_raises(RuntimeError):
            firewall.register_terminal_histories(replicates, oracle)
        assert oracle.registered == []


def test_terminal_parent_logmeanexp_must_match_stored_aggregate(tmp_path):
    class InconsistentTerminalOracle(_FakeTerminalOracle):
        def terminal_parent_log_z(self, master_seed, keys):
            value = super().terminal_parent_log_z(master_seed, keys)
            value[:, 0] = 1.0
            return value

    oracle = InconsistentTerminalOracle()
    firewall = TwoPhaseArtifactFirewall(tmp_path / "inconsistent_parent")
    firewall.register_terminal_histories(tuple(
        _fake_replicate(seed)
        for seed in capability.PRODUCTION_REPLICATE_MASTER_SEEDS
    ), oracle)
    with np.testing.assert_raises_regex(
        RuntimeError, "aggregate evidence is inconsistent"
    ):
        firewall.seal_terminal_phase(oracle)
    assert not list((tmp_path / "inconsistent_parent").iterdir())


def _sealed_and_open_firewall(path):
    firewall = TwoPhaseArtifactFirewall(path)
    oracle = _FakeTerminalOracle()
    firewall.register_terminal_histories(tuple(
        _fake_replicate(seed)
        for seed in capability.PRODUCTION_REPLICATE_MASTER_SEEDS
    ), oracle)
    frozen = firewall.seal_terminal_phase(oracle)
    firewall.open_calibration()
    return firewall, frozen


def test_publish_rejects_sealed_terminal_mutation(tmp_path):
    firewall, frozen = _sealed_and_open_firewall(tmp_path / "mutated")
    with frozen.path.open("ab") as stream:
        stream.write(b"mutation")
    with np.testing.assert_raises_regex(RuntimeError, "terminal parent artifact changed"):
        firewall.publish_cf4_gates()


def test_publish_rejects_bound_calibration_substitution(tmp_path):
    firewall, _ = _sealed_and_open_firewall(tmp_path / "substituted")
    firewall._calibration = object()
    with np.testing.assert_raises_regex(RuntimeError, "object was substituted"):
        firewall.publish_cf4_gates()


def test_publish_reloads_and_rejects_bound_calibration_mutation(tmp_path):
    firewall, _ = _sealed_and_open_firewall(tmp_path / "calibration_mutated")
    firewall._calibration.deviance.flags.writeable = True
    firewall._calibration.deviance[0] += 1.0
    firewall._calibration.deviance.flags.writeable = False
    with np.testing.assert_raises_regex(RuntimeError, "content changed"):
        firewall.publish_cf4_gates()


def test_oracle_runtime_message_cannot_spoof_valid_architecture_stop(
    tmp_path, monkeypatch
):
    class SpoofingOracle:
        def evaluate(self, midpoint, axis):
            raise RuntimeError("temperature schedule stagnated")

    def synthetic_core(master_seed, oracle):
        return oracle.evaluate(
            np.zeros((1, 3), dtype=np.float64),
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        )

    monkeypatch.setattr(capability, "run_smc_replicate", synthetic_core)
    with np.testing.assert_raises_regex(
        OracleEvaluationFailure,
        "oracle evaluate failed: temperature schedule stagnated",
    ):
        capability._run_fixed_capability_core(
            SpoofingOracle(), tmp_path / "oracle_spoof"
        )
    assert capability.classify_smc_runtime_failure(
        OracleEvaluationFailure("oracle evaluate failed")
    )[0] == "invalid_failed"
    assert not (tmp_path / "oracle_spoof/capability_result.json").exists()


def test_pre_cf4_gate_boundaries_are_inclusive_and_use_all_six_pairs():
    uniform = np.full(256, 1.0 / 256.0)
    result = evaluate_pre_cf4_gates(
        np.asarray([-0.1, -0.05, 0.05, 0.1]),
        np.tile(uniform, (4, 1)),
        uniform,
        np.full(4, 128.0),
    )
    assert result["metrics"]["replicate_log_I_bar_range"] == 0.2
    assert len(result["metrics"]["six_pairwise_P_rep_L1"]) == 6
    assert all(result["gates"].values())

    se_amplitude = 0.1 * math.sqrt(3.0)
    se_boundary = evaluate_pre_cf4_gates(
        np.asarray([-se_amplitude, -se_amplitude, se_amplitude, se_amplitude]),
        np.tile(uniform, (4, 1)),
        uniform,
        np.full(4, 128.0),
    )
    assert se_boundary["metrics"]["replicate_log_I_bar_sample_SE"] == 0.1
    assert se_boundary["gates"]["replicate_log_I_bar_sample_SE"] is True

    base = np.zeros(256)
    base[:10] = 0.1
    shifted = base.copy()
    shifted[0] = 0.0
    shifted[10] = 0.1
    pair_boundary = evaluate_pre_cf4_gates(
        np.zeros(4), np.stack((base, shifted, base, shifted)),
        uniform, np.full(4, 128.0),
    )
    assert pair_boundary["metrics"]["maximum_pairwise_P_rep_L1"] == 0.2
    assert pair_boundary["gates"]["replicate_parent_probability_L1"] is True

    ten = np.zeros(256)
    ten[:10] = 0.1
    thirty_two = np.zeros(256)
    thirty_two[:32] = 1.0 / 32.0
    max_boundary = evaluate_pre_cf4_gates(
        np.zeros(4), np.tile(ten, (4, 1)), ten, np.full(4, 128.0)
    )
    ess_boundary = evaluate_pre_cf4_gates(
        np.zeros(4), np.tile(thirty_two, (4, 1)), thirty_two,
        np.full(4, 128.0),
    )
    assert max_boundary["metrics"]["maximum_P_pool"] == 0.1
    assert max_boundary["gates"]["maximum_pooled_parent_probability"] is True
    assert ess_boundary["metrics"]["pooled_parent_ESS"] == 32.0
    assert ess_boundary["gates"]["pooled_parent_ESS"] is True


def test_cf4_ties_quantile_boundary_and_fixed_permutation_are_exact():
    tied_values = np.asarray([0.0, 0.0, 1.0, 1.0])
    tied_weights = np.asarray([0.1, 0.1, 0.4, 0.4])
    assert abs(_one_sided_tied_ks(tied_values, tied_weights) - 0.3) < 1e-15

    values = np.zeros(256)
    values[-1] = 2.0
    weights = np.zeros(256)
    weights[0] = 0.95
    weights[-1] = 0.05
    result = evaluate_cf4_gates(values, weights, 1.0, 0.0)
    assert result["metrics"]["weighted_CF4_Q99_exceedance_mass"] == 0.05
    assert result["metrics"]["weighted_CF4_Q90"] == 0.0
    assert result["gates"]["weighted_CF4_Q99_exceedance_mass"] is True
    assert result["gates"]["weighted_CF4_Q90"] is True
    assert result["metrics"]["permutations"] == 100000
    assert result["metrics"]["permutation_pvalue"] == (
        result["metrics"]["permutation_exceedance_count"] + 1
    ) / 100001

    deterministic_values = np.repeat(np.arange(8, dtype=np.float64), 32)
    deterministic_weights = np.arange(1, 257, dtype=np.float64)
    deterministic_weights /= deterministic_weights.sum()
    first = _permutation_pvalue(deterministic_values, deterministic_weights)
    second = _permutation_pvalue(deterministic_values, deterministic_weights)
    assert first == second


def test_failure_classification_uses_frozen_priority():
    gates = {gate: True for gate, _ in capability.GATE_FAILURE_PRIORITY}
    gates["weighted_CF4_Q90"] = False
    gates["genealogical_ESS"] = False
    gates["replicate_log_I_bar_range"] = False
    assert classify_failure(gates) == "replicate_log_I_bar_range"
    assert classify_failure({}) == "invalid_lineage_or_authorization"
    complete = {
        gate: True for gate, _ in capability.GATE_FAILURE_PRIORITY
    }
    assert classify_failure(complete) is None
    assert capability.classify_lifecycle_status(complete) == "complete_pass"
    architecture_stop = dict(complete, temperature_stagnation_absent=False)
    assert classify_failure(architecture_stop) == "SMC_temperature_stagnation"
    assert capability.classify_lifecycle_status(
        architecture_stop
    ) == "complete_scientific_fail"
    invalid = dict(complete, lineage_and_authorization=False)
    assert capability.classify_lifecycle_status(invalid) == "invalid_failed"
    assert capability.classify_smc_runtime_failure(
        RuntimeError("temperature schedule stagnated")
    ) == ("complete_scientific_fail", "SMC_temperature_stagnation")
    assert capability.classify_smc_runtime_failure(
        RuntimeError("SMC replicate did not reach beta=1")
    ) == ("complete_scientific_fail", "SMC_maximum_temperature_stages")
    assert capability.classify_smc_runtime_failure(
        RuntimeError("unexpected corruption")
    )[0] == "invalid_failed"
    with np.testing.assert_raises_regex(ValueError, "unknown frozen gate"):
        classify_failure({"retuned_gate": False})
