import ast
from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import cf4_aggregate_evidence_smc_v6_open_shared_schedule_production as production


@pytest.fixture(scope="module")
def contract():
    return production.load_frozen_contract()


def _empty_move():
    return {
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


def _replicate(seed, log_i=0.0, beta=production.SHARED_BETA):
    keys = np.zeros((2048, 6), dtype=np.int16)
    keys[:, :3] = 288
    keys[:, 3] = 3
    move = _empty_move()
    return SimpleNamespace(
        master_seed=seed,
        midpoint_mpc_h=np.zeros((2048, 3), dtype=np.float64),
        axis=np.tile([1.0, 0.0, 0.0], (2048, 1)),
        keys=keys,
        weights=np.full(2048, 1.0 / 2048.0, dtype=np.float64),
        log_z_bar=np.zeros(2048, dtype=np.float64),
        ancestor_labels=np.arange(2048, dtype=np.int64),
        beta_history=np.asarray(beta, dtype=np.float64),
        conditional_ess_history=np.full(5, 2048.0, dtype=np.float64),
        particle_ess_history=np.full(6, 2048.0, dtype=np.float64),
        log_normalizer_increment=np.full(5, log_i / 5.0, dtype=np.float64),
        resampling_ancestors=[],
        move_history=[[dict(move) for _ in range(4)] for _ in range(5)],
        log_normalizer=float(log_i),
        genealogical_ess=2048.0,
    )


def _probability(index, peak=0.02):
    value = np.full(256, (1.0 - peak) / 255.0, dtype=np.float64)
    value[index] = peak
    return value


def _product(seed, index=0, log_i=0.0, contract=None, **flags):
    if contract is None:
        contract = production.load_frozen_contract()
    probability = _probability(index)
    parent_log_z = np.tile(np.log(256.0 * probability), (2048, 1))
    provenance = production.FreshReplicateProvenance(
        master_seed=seed,
        fresh_token=f"token-{seed}",
        evaluator_namespace=f"v6-production-evaluator-{seed}",
        cache_namespace=contract.cache_namespace,
        pilot_state_reused=False,
        v5_state_reused=False,
        pilot_cache_reused=False,
        pilot_particles_reused=False,
        pilot_rng_state_reused=False,
        evaluator_closed=True,
        evaluator_close_count=1,
    )
    provenance = replace(provenance, **flags)
    return production.FreshReplicateProduct(
        replicate=_replicate(seed, log_i),
        parent_log_z=parent_log_z,
        parent_seeds=np.asarray(production.PARENT_SEEDS, dtype=np.int64),
        provenance=provenance,
        _seal=production._PRODUCT_SEAL,
    )


class _Lease:
    def __init__(
        self, seed, contract, index=0, *, close_ok=True, terminal_ok=True,
        oracle=None, events=None,
    ):
        self.oracle = oracle if oracle is not None else object()
        probability = _probability(index)
        self._terminal_parent_log_z = np.tile(
            np.log(256.0 * probability), (2048, 1)
        )
        self.parent_seeds = np.asarray(production.PARENT_SEEDS, dtype=np.int64)
        self.provenance = production.FreshReplicateProvenance(
            master_seed=seed,
            fresh_token=f"lease-token-{seed}",
            evaluator_namespace=f"lease-evaluator-{seed}",
            cache_namespace=contract.cache_namespace,
            pilot_state_reused=False,
            v5_state_reused=False,
            pilot_cache_reused=False,
            pilot_particles_reused=False,
            pilot_rng_state_reused=False,
            evaluator_closed=False,
            evaluator_close_count=0,
        )
        self.closed = False
        self.close_count = 0
        self.close_ok = close_ok
        self.terminal_ok = terminal_ok
        self.events = events if events is not None else []
        self.terminal_calls = []

    def terminal_parent_log_z(self, master_seed, keys):
        self.events.append("terminal")
        keys = np.asarray(keys)
        self.terminal_calls.append((master_seed, keys.copy()))
        if not self.terminal_ok:
            raise RuntimeError("terminal accessor failed")
        if master_seed != self.provenance.master_seed:
            raise RuntimeError("wrong terminal master seed")
        expected = _replicate(master_seed).keys
        if keys.dtype != np.int16 or not np.array_equal(keys, expected):
            raise RuntimeError("wrong terminal keys")
        return self._terminal_parent_log_z

    def close(self):
        self.events.append("close")
        self.close_count += 1
        if not self.close_ok:
            raise RuntimeError("close failed")
        self.closed = True


def _products(contract, log_i=(0.0, 0.0, 0.0, 0.0)):
    return tuple(
        _product(seed, index, log_i[index], contract)
        for index, seed in enumerate(contract.master_seeds)
    )


def test_load_contract_pins_design_erratum_git_and_all_inputs(contract):
    assert contract.design_sha256 == production.DESIGN_SHA256
    assert contract.erratum_sha256 == production.ERRATUM_SHA256
    assert contract.schedule_sha256 == production.SCHEDULE_SHA256
    assert contract.master_seeds == production.MASTER_SEEDS
    assert contract.particles == 2048 and len(contract.parent_seeds) == 256
    assert production._is_ancestor(production.DESIGN_COMMIT, production._head_sha())
    assert production._is_ancestor(production.ERRATUM_COMMIT, production._head_sha())


def test_forged_contract_is_rejected_by_every_scientific_boundary(contract):
    forged = replace(contract, cache_namespace=contract.cache_namespace + "-forged")
    with pytest.raises(production.ArchitectureFailure, match="identity changed"):
        production.validate_shared_schedule(production.SHARED_BETA, forged)
    with pytest.raises(production.ArchitectureFailure, match="identity changed"):
        production.validate_replicate_product(
            _product(2026082301, contract=forged), 2026082301, forged
        )
    with pytest.raises(production.ArchitectureFailure, match="identity changed"):
        production.build_terminal_summary(_products(contract), forged)
    exact_values_but_unsealed = production.FrozenProductionContract(
        design_sha256=contract.design_sha256,
        erratum_sha256=contract.erratum_sha256,
        schedule_sha256=contract.schedule_sha256,
        beta=contract.beta,
        master_seeds=contract.master_seeds,
        particles=contract.particles,
        parent_seeds=contract.parent_seeds,
        cache_namespace=contract.cache_namespace,
        _seal=object(),
    )
    with pytest.raises(production.ArchitectureFailure, match="identity changed"):
        production.validate_shared_schedule(
            production.SHARED_BETA, exact_values_but_unsealed
        )


@pytest.mark.parametrize("target", ["design", "erratum"])
def test_design_and_erratum_mutation_are_rejected(monkeypatch, tmp_path, target):
    source = production.DESIGN if target == "design" else production.ERRATUM
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    monkeypatch.setattr(production, target.upper(), changed)
    with pytest.raises(production.ArchitectureFailure, match="hash mismatch"):
        production.load_frozen_contract()


def test_schedule_manifest_hash_and_exact_schedule_validation(contract):
    value = production.validate_shared_schedule(production.SHARED_BETA, contract)
    assert value.dtype == np.float64 and not value.flags.writeable
    design = production._json(production.DESIGN)
    left = Path(design["hard_pins"]["receipt_schedule_manifest"]["path"])
    right = Path(design["hard_pins"]["pilot_schedule_manifest"]["path"])
    assert left.read_bytes() == right.read_bytes()
    assert hashlib.sha256(left.read_bytes()).hexdigest() == production.MANIFEST_SHA256
    for invalid in (
        [0.0, 0.2, 1.0],
        [0.0, np.nan, 1.0],
        [0.0, 0.5, 0.4, 1.0],
    ):
        with pytest.raises(production.ArchitectureFailure, match="stage_parity"):
            production.validate_shared_schedule(invalid, contract)


def test_fixed_core_never_calls_adaptive_selector_and_has_exact_histories(
    monkeypatch, contract,
):
    monkeypatch.setattr(
        production.base, "select_temperature_increment",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("adaptive")),
    )
    monkeypatch.setattr(
        production.base, "initialize_particles",
        lambda seed, count: (
            np.zeros((count, 3), dtype=np.float64),
            np.tile([1.0, 0.0, 0.0], (count, 1)),
        ),
    )
    monkeypatch.setattr(production.base, "particle_ess", lambda weights: 2048.0)
    monkeypatch.setattr(
        production.base, "conditional_ess",
        lambda weights, log_z, delta: 2048.0,
    )
    monkeypatch.setattr(
        production.base, "update_weights_and_normalizer",
        lambda weights, log_z, delta: (weights.copy(), float(delta)),
    )
    monkeypatch.setattr(
        production.base, "mh_rejuvenation_sweep",
        lambda midpoint, axis, keys, log_z, beta, oracle, seed, stage, sweep:
        (midpoint, axis, keys, log_z, _empty_move()),
    )
    class Oracle:
        def evaluate(self, midpoint, axis):
            return np.zeros((2048, 6), dtype=np.int16), np.zeros(2048)
    replicate = production._run_fixed_schedule_replicate_core(
        2026082301, Oracle(), contract
    )
    assert replicate.beta_history.shape == (6,)
    assert replicate.conditional_ess_history.shape == (5,)
    assert replicate.particle_ess_history.shape == (6,)
    assert replicate.log_normalizer_increment.shape == (5,)
    assert len(replicate.move_history) == 5
    assert all(len(stage) == 4 for stage in replicate.move_history)


def test_factory_seed_order_fixed_core_count_isolation_and_exact_close(
    monkeypatch, contract,
):
    class Factory:
        def __init__(self):
            self.calls = []
            self.leases = []
            self.events = []
        def __call__(self, seed, frozen):
            self.calls.append((seed, frozen))
            lease = _Lease(
                seed, frozen, len(self.calls) - 1, events=self.events
            )
            self.leases.append(lease)
            return lease
    factory = Factory()
    core_calls = []
    monkeypatch.setattr(
        production, "_run_fixed_schedule_replicate_core",
        lambda seed, oracle, frozen: (
            core_calls.append((seed, oracle, frozen))
            or factory.events.append("core")
            or _replicate(seed)
        ),
    )
    products = production._run_four_fresh_replicates(factory, contract)
    assert [row[0] for row in factory.calls] == list(production.MASTER_SEEDS)
    assert len(products) == 4 and all(row[1] is contract for row in factory.calls)
    assert [row[0] for row in core_calls] == list(production.MASTER_SEEDS)
    assert len({id(row[1]) for row in core_calls}) == 4
    assert all(lease.closed and lease.close_count == 1 for lease in factory.leases)
    assert all(product.provenance.evaluator_close_count == 1 for product in products)
    assert factory.events == ["core", "terminal", "close"] * 4
    for lease, seed in zip(factory.leases, production.MASTER_SEEDS):
        assert len(lease.terminal_calls) == 1
        observed_seed, observed_keys = lease.terminal_calls[0]
        assert observed_seed == seed
        np.testing.assert_array_equal(observed_keys, _replicate(seed).keys)


def test_duplicate_product_token_namespace_and_reuse_flags_are_rejected(contract):
    products = list(_products(contract))
    with pytest.raises(production.ArchitectureFailure):
        production.build_terminal_summary([products[0]] * 4, contract)
    products[1] = replace(
        products[1], provenance=replace(
            products[1].provenance,
            fresh_token=products[0].provenance.fresh_token,
        )
    )
    with pytest.raises(production.ArchitectureFailure, match="independent"):
        production.build_terminal_summary(products, contract)
    for flag in (
        "pilot_state_reused", "v5_state_reused", "pilot_cache_reused",
        "pilot_particles_reused", "pilot_rng_state_reused",
    ):
        bad = _product(2026082301, contract=contract, **{flag: True})
        with pytest.raises(production.ArchitectureFailure, match="reuse"):
            production.validate_replicate_product(bad, 2026082301, contract)


def test_factory_exception_closes_current_lease_exactly_once(monkeypatch, contract):
    class Factory:
        def __init__(self):
            self.calls = 0
            self.leases = []
        def __call__(self, seed, frozen):
            self.calls += 1
            lease = _Lease(seed, frozen, self.calls - 1)
            self.leases.append(lease)
            return lease
    factory = Factory()
    monkeypatch.setattr(
        production, "_run_fixed_schedule_replicate_core",
        lambda *args: (_ for _ in ()).throw(RuntimeError("core failed")),
    )
    with pytest.raises(RuntimeError, match="core failed"):
        production._run_four_fresh_replicates(factory, contract)
    assert len(factory.leases) == 1
    assert factory.leases[0].closed and factory.leases[0].close_count == 1


def test_duplicate_lease_or_oracle_and_bad_close_are_rejected(monkeypatch, contract):
    monkeypatch.setattr(
        production, "_run_fixed_schedule_replicate_core",
        lambda seed, oracle, frozen: _replicate(seed),
    )
    shared_oracle = object()
    class DuplicateOracleFactory:
        def __call__(self, seed, frozen):
            return _Lease(seed, frozen, oracle=shared_oracle)
    with pytest.raises(production.ArchitectureFailure, match="isolation"):
        production._run_four_fresh_replicates(DuplicateOracleFactory(), contract)

    class BadCloseFactory:
        def __call__(self, seed, frozen):
            return _Lease(seed, frozen, close_ok=False)
    with pytest.raises(production.ArchitectureFailure, match="close failed"):
        production._run_four_fresh_replicates(BadCloseFactory(), contract)


def test_terminal_accessor_exception_closes_exactly_once(monkeypatch, contract):
    monkeypatch.setattr(
        production, "_run_fixed_schedule_replicate_core",
        lambda seed, oracle, frozen: _replicate(seed),
    )
    class Factory:
        def __init__(self):
            self.lease = None
        def __call__(self, seed, frozen):
            self.lease = _Lease(seed, frozen, terminal_ok=False)
            return self.lease
    factory = Factory()
    with pytest.raises(RuntimeError, match="terminal accessor failed"):
        production._run_four_fresh_replicates(factory, contract)
    assert factory.lease.closed and factory.lease.close_count == 1
    assert factory.lease.events == ["terminal", "close"]


def test_terminal_accessor_rejects_wrong_seed_keys_and_static_array_lease(
    monkeypatch, contract,
):
    lease = _Lease(2026082301, contract)
    keys = _replicate(2026082301).keys.copy()
    with pytest.raises(RuntimeError, match="master seed"):
        lease.terminal_parent_log_z(2026082302, keys)
    keys[0, 0] += 1
    with pytest.raises(RuntimeError, match="terminal keys"):
        lease.terminal_parent_log_z(2026082301, keys)

    monkeypatch.setattr(
        production, "_run_fixed_schedule_replicate_core",
        lambda seed, oracle, frozen: _replicate(seed),
    )
    class StaticLease(_Lease):
        terminal_parent_log_z = np.zeros((2048, 256), dtype=np.float64)
    class StaticFactory:
        def __init__(self):
            self.lease = None
        def __call__(self, seed, frozen):
            self.lease = StaticLease(seed, frozen)
            return self.lease
    factory = StaticFactory()
    with pytest.raises(production.ArchitectureFailure, match="invalid lease"):
        production._run_four_fresh_replicates(factory, contract)
    assert factory.lease.closed and factory.lease.close_count == 1


def test_replicate_dtype_shape_finite_normalization_and_beta_are_enforced(contract):
    product = _product(2026082301, contract=contract)
    for field, value in (
        ("weights", product.replicate.weights.astype(np.float32)),
        ("axis", np.full((2048, 3), np.nan)),
        ("weights", np.full(2048, 2.0 / 2048.0)),
    ):
        broken = SimpleNamespace(**vars(product.replicate)); setattr(broken, field, value)
        with pytest.raises(RuntimeError):
            production.validate_replicate_product(
                replace(product, replicate=broken), 2026082301, contract
            )
    broken = SimpleNamespace(**vars(product.replicate))
    broken.beta_history = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    with pytest.raises(production.ArchitectureFailure, match="stage_parity"):
        production.validate_replicate_product(
            replace(product, replicate=broken), 2026082301, contract
        )


def test_parent_logz_shape_nonfinite_and_aggregate_mismatch_are_rejected(contract):
    product = _product(2026082301, contract=contract)
    with pytest.raises(production.ArchitectureFailure, match="invalid product"):
        production.validate_replicate_product(
            replace(product, _seal=object()), 2026082301, contract
        )
    bad_values = (
        np.zeros((2048, 255), dtype=np.float64),
        np.full((2048, 256), np.nan),
        product.parent_log_z + 1.0,
    )
    for value in bad_values:
        with pytest.raises(production.ArchitectureFailure, match="parent-logZ"):
            production.validate_replicate_product(
                replace(product, parent_log_z=value), 2026082301, contract
            )


@pytest.mark.parametrize(
    "parent_seeds",
    [
        np.asarray(production.PARENT_SEEDS[::-1], dtype=np.int64),
        np.asarray((3193,) * 256, dtype=np.int64),
        np.asarray(production.PARENT_SEEDS, dtype=np.int32),
    ],
)
def test_parent_seed_order_dtype_and_uniqueness_are_enforced(contract, parent_seeds):
    product = _product(2026082301, contract=contract)
    with pytest.raises(production.ArchitectureFailure, match="parent seed order"):
        production.validate_replicate_product(
            replace(product, parent_seeds=parent_seeds), 2026082301, contract
        )
    reversed_columns = replace(
        product,
        parent_log_z=product.parent_log_z[:, ::-1].copy(),
        parent_seeds=np.asarray(production.PARENT_SEEDS[::-1], dtype=np.int64),
    )
    with pytest.raises(production.ArchitectureFailure, match="parent seed order"):
        production.validate_replicate_product(
            reversed_columns, 2026082301, contract
        )


def test_validated_parent_seed_copy_is_exact_and_read_only(contract):
    product = _product(2026082301, contract=contract)
    validated = production.validate_replicate_product(
        product, 2026082301, contract
    )
    np.testing.assert_array_equal(
        validated.parent_seeds,
        np.asarray(production.PARENT_SEEDS, dtype=np.int64),
    )
    assert not validated.parent_seeds.flags.writeable
    product.parent_seeds[:] = 0
    assert validated.parent_seeds[0] == 3193


def test_evidence_weighted_pool_is_base_exact_and_not_arithmetic_mean(contract):
    products = _products(contract, log_i=(0.0, 1.0, 2.0, 3.0))
    summary = production.build_terminal_summary(products, contract)
    expected, expected_log_i = production.base.pool_parent_probabilities(
        summary.log_I_bar, summary.P_rep
    )
    np.testing.assert_array_equal(summary.P_pool, expected)
    assert summary.pooled_log_I_bar == expected_log_i
    assert not np.allclose(
        summary.P_pool, summary.P_rep_arithmetic_mean_diagnostic_only,
        rtol=0.0, atol=1e-8,
    )


@pytest.mark.parametrize("offset, accepted", [(5e-13, True), (2e-12, False)])
def test_erratum_pooling_crosscheck_uses_exact_1e_minus_12_tolerance(
    monkeypatch, contract, offset, accepted,
):
    original = production.base.pool_parent_probabilities

    def shifted(log_i, p_rep):
        pool, pooled_log_i = original(log_i, p_rep)
        pool = pool.copy()
        pool[0] += offset
        pool[1] -= offset
        return pool, pooled_log_i + offset

    monkeypatch.setattr(production.base, "pool_parent_probabilities", shifted)
    if accepted:
        production.build_terminal_summary(_products(contract), contract)
    else:
        with pytest.raises(production.ArchitectureFailure, match="erratum"):
            production.build_terminal_summary(_products(contract), contract)


def test_pre_cf4_diagnostics_uses_evidence_weighted_pool(monkeypatch, contract):
    summary = production.build_terminal_summary(
        _products(contract, log_i=(0.0, 1.0, 2.0, 3.0)), contract
    )
    observed = []

    def paired(log_i, p_rep, p_pool, *, calibration_draws):
        del log_i, p_rep, calibration_draws
        observed.append(np.asarray(p_pool).copy())
        return {
            "log_I_bar_range_pass": True,
            "log_I_bar_sample_SE_pass": True,
            "L1_diagnostic_threshold_pass": True,
            "null_calibration": {
                "coherent_pass": True, "q99": 0.0, "q999": 0.0,
                "draws": 20000, "seed": 2026081801,
                "tail_probability": 1.0,
            },
        }

    monkeypatch.setattr(production.shared, "paired_incoherence_diagnostics", paired)
    diagnostics = production.evaluate_pre_cf4_diagnostics(summary, contract)
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], summary.P_pool)
    assert not np.array_equal(
        observed[0], summary.P_rep_arithmetic_mean_diagnostic_only
    )
    assert set(diagnostics.gates) == production.PRE_CF4_GATE_KEYS
    assert not set(diagnostics.gates).intersection(production.VALIDITY_GATE_KEYS)
    assert diagnostics.metrics["L1_null_exceedance_count"] == 20000


def test_l1_20000_draw_golden_and_channel_mapping():
    calibration = production.shared.calibrate_max_l1_null(
        np.full(256, 1.0 / 256.0), 0.5
    )
    assert calibration.q99 == 0.451171875
    assert calibration.q999 == 0.462890625
    assert calibration.tail_probability == 1.0 / 20001.0
    assert production._null_exceedance_count({
        "draws": calibration.draws,
        "seed": calibration.seed,
        "tail_probability": calibration.tail_probability,
    }) == 0
    assert (0 + 1) / 20001 == calibration.tail_probability
    mapped = production._map_shared_diagnostic_channels([
        "replicate_log_I_bar_range", "replicate_parent_probability_L1",
    ])
    assert mapped == (
        "replicate_log_I_bar_range",
        "replicate_parent_probability_L1_null_tail",
    )
    assert "replicate_parent_probability_L1" not in mapped


def test_paired_incoherence_precedence_and_inclusive_thresholds(contract):
    gates = {name: True for name in production.PRE_CF4_GATE_KEYS}
    gates["replicate_log_I_bar_range"] = False
    gates["replicate_parent_probability_L1_null_tail"] = False
    gates["replicate_log_I_bar_sample_SE"] = False
    failed = tuple(name for name, passed in gates.items() if not passed)
    assert production._classify_primary_failure(
        failed, gates, contract
    ) == "paired_incoherence"
    log_i = np.asarray([0.0, 0.2, 0.1, 0.1])
    assert np.max(log_i) - np.min(log_i) == 0.2
    assert production.capability.evaluate_pre_cf4_gates(
        log_i, np.full((4, 256), 1 / 256), np.full(256, 1 / 256),
        np.full(4, 128.0),
    )["gates"]["replicate_log_I_bar_range"] is True


def test_invalid_failure_precedes_paired_incoherence(contract):
    gates = {name: True for name in production.PRE_CF4_GATE_KEYS}
    gates["lineage_and_authorization"] = False
    gates["replicate_log_I_bar_range"] = False
    gates["replicate_parent_probability_L1_null_tail"] = False
    failed = tuple(name for name, passed in gates.items() if not passed)
    assert production._classify_primary_failure(
        failed, gates, contract
    ) == "invalid_lineage_or_authorization"


def _pre(gates, contract):
    failed = tuple(name for name, passed in gates.items() if not passed)
    return production.PreCF4Diagnostics(
        metrics=MappingProxyType({}), gates=MappingProxyType(gates),
        failed_channels=failed,
        primary_failure=production._classify_primary_failure(
            failed, gates, contract
        ),
    )


def _validity():
    return {name: True for name in production.VALIDITY_GATE_KEYS}


def test_complete_gate_keysets_priority_and_outcomes(contract):
    pre_gates = {name: True for name in production.PRE_CF4_GATE_KEYS}
    cf4 = {name: True for name in production.CF4_GATE_KEYS}
    decision = production.classify_complete_gate_set(
        _validity(), _pre(pre_gates, contract), cf4, contract
    )
    assert decision.outcome_kind == "pass" and decision.primary_failure is None
    science = dict(pre_gates); science["genealogical_ESS"] = False
    decision = production.classify_complete_gate_set(
        _validity(), _pre(science, contract), cf4, contract
    )
    assert decision.outcome_kind == "scientific_fail"
    invalid = _validity(); invalid["lineage_and_authorization"] = False
    decision = production.classify_complete_gate_set(
        invalid, _pre(pre_gates, contract), cf4, contract
    )
    assert decision.outcome_kind == "invalid"
    for broken in ({}, {**cf4, "unknown": True}):
        with pytest.raises(production.ArchitectureFailure, match="keyset"):
            production.classify_complete_gate_set(
                _validity(), _pre(pre_gates, contract), broken, contract
            )
    wrong = dict(pre_gates); wrong["unknown"] = True
    with pytest.raises(production.ArchitectureFailure, match="keyset"):
        production.classify_complete_gate_set(
            _validity(), _pre(wrong, contract), cf4, contract
        )
    for broken in ({}, {**_validity(), "unknown": True}):
        with pytest.raises(production.ArchitectureFailure, match="validity gate"):
            production.classify_complete_gate_set(
                broken, _pre(pre_gates, contract), cf4, contract
            )
    for gate, failure in production.GATE_FAILURE_PRIORITY:
        all_gates = {name: True for name in production.PRE_CF4_GATE_KEYS | production.CF4_GATE_KEYS}
        all_gates[gate] = False
        assert production._classify_primary_failure((gate,), all_gates, contract) == failure


def test_complete_gate_values_must_be_exact_booleans(contract):
    pre_gates = {name: True for name in production.PRE_CF4_GATE_KEYS}
    cf4 = {name: True for name in production.CF4_GATE_KEYS}
    for value in (1, np.bool_(True), None):
        broken = dict(pre_gates)
        broken["replicate_log_I_bar_range"] = value
        with pytest.raises(production.ArchitectureFailure, match="keyset"):
            production.classify_complete_gate_set(
                _validity(), _pre(broken, contract), cf4, contract
            )
    for value in (1, np.bool_(True), None):
        broken = _validity()
        broken["lineage_and_authorization"] = value
        with pytest.raises(production.ArchitectureFailure, match="validity gate"):
            production.classify_complete_gate_set(
                broken, _pre(pre_gates, contract), cf4, contract
            )


def test_forged_terminal_summary_and_primary_failure_are_rejected(contract):
    summary = production.build_terminal_summary(_products(contract), contract)
    arithmetic = summary.P_rep_arithmetic_mean_diagnostic_only.copy()
    arithmetic[0] += 1e-4
    with pytest.raises(production.ArchitectureFailure, match="pooling contract"):
        production.evaluate_pre_cf4_diagnostics(
            replace(summary, P_rep_arithmetic_mean_diagnostic_only=arithmetic),
            contract,
        )
    gates = {name: True for name in production.PRE_CF4_GATE_KEYS}
    forged_pre = production.PreCF4Diagnostics(
        metrics=MappingProxyType({}),
        gates=MappingProxyType(gates),
        failed_channels=(),
        primary_failure="replicate_log_I_bar_range",
    )
    with pytest.raises(production.ArchitectureFailure, match="primary failure"):
        production.classify_complete_gate_set(
            _validity(), forged_pre,
            {name: True for name in production.CF4_GATE_KEYS}, contract,
        )


def test_arrays_are_read_only_and_caller_inputs_are_isolated(contract):
    products = _products(contract)
    summary = production.build_terminal_summary(products, contract)
    for value in (
        summary.master_seed, summary.beta_history, summary.log_I_bar,
        summary.P_rep, summary.P_pool,
        summary.P_rep_arithmetic_mean_diagnostic_only,
        summary.genealogical_ESS,
    ):
        assert not value.flags.writeable
    original = summary.P_rep.copy()
    products[0].parent_log_z[:] = 0.0
    np.testing.assert_array_equal(summary.P_rep, original)
    with pytest.raises(ValueError):
        summary.P_pool[0] = 1.0


def test_public_refusal_is_first_action_and_pure_apis_create_no_roots(
    monkeypatch, contract,
):
    calls = []
    factory = lambda *args: calls.append(args)
    roots = [Path(value) for value in production._json(production.DESIGN)["artifact_isolation"].values() if isinstance(value, str) and value.startswith("/gpfs/")]
    before = [path.exists() for path in roots]
    monkeypatch.setattr(production, "load_frozen_contract", lambda: calls.append("load"))
    with pytest.raises(PermissionError, match="unauthorized"):
        production.run_production_capability(factory, Path("output"))
    assert calls == [] and before == [path.exists() for path in roots]


def test_module_has_no_cli_writer_launcher_or_cache_population_surface():
    source = Path(production.__file__).read_text()
    tree = ast.parse(source)
    assert "__main__" not in source and "argparse" not in source
    banned_modules = {"subprocess", "shutil"}
    banned_calls = {
        "save", "savez", "savez_compressed", "chmod", "link", "unlink",
        "rename", "replace", "mkdir", "makedirs", "remove", "rmdir",
        "system", "popen",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {alias.name.split(".")[0] for alias in node.names}
            assert not names.intersection(banned_modules)
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                node.func.id if isinstance(node.func, ast.Name) else ""
            )
            if name == "replace" and isinstance(node.func, ast.Name):
                continue
            assert name not in banned_calls
    assert set(inspect.signature(production.run_production_capability).parameters) == {"args", "kwargs"}
