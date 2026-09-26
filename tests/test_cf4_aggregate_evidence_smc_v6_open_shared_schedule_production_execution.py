from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cf4_aggregate_evidence_smc_v6_open_shared_schedule_production as capability
import cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution as execution


def _move():
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


def _replicate(seed, oracle, contract=None):
    del contract
    midpoint = np.zeros((2048, 3), dtype=np.float64)
    axis = np.tile([1.0, 0.0, 0.0], (2048, 1))
    keys, log_z_bar = oracle.evaluate(midpoint, axis)
    return SimpleNamespace(
        master_seed=seed,
        midpoint_mpc_h=midpoint,
        axis=axis,
        keys=keys,
        weights=np.full(2048, 1.0 / 2048.0, dtype=np.float64),
        log_z_bar=log_z_bar,
        ancestor_labels=np.arange(2048, dtype=np.int64),
        beta_history=np.asarray(capability.SHARED_BETA, dtype=np.float64),
        conditional_ess_history=np.full(5, 2048.0, dtype=np.float64),
        particle_ess_history=np.full(6, 2048.0, dtype=np.float64),
        log_normalizer_increment=np.zeros(5, dtype=np.float64),
        resampling_ancestors=[],
        move_history=[[_move() for _ in range(4)] for _ in range(5)],
        log_normalizer=0.0,
        genealogical_ess=2048.0,
    )


class _Evaluator:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def __call__(self, keys):
        self.calls += 1
        return keys, np.zeros((len(keys), 256), dtype=np.float64)

    def close(self):
        if self.closed:
            raise RuntimeError("double close")
        self.closed = True


def _cf4(summary):
    return execution._canonical_cf4_gate_provider(summary)


def _control():
    return {
        "control_rows": 24,
        "control_evaluator_discarded": True,
        "control_cache_discarded": True,
        "production_cache_empty": True,
        "sealed_summary": {
            "schema": "ouruniv-cf4-sealed-oracle-production-control-summary-v1",
            "status": "complete_pass_exact_24_row_control",
            "selection_sha256": "6902ab3f1a6c7fb2e5d9416d49ee956380c83eacc5d6b1b4fd2b678f64a59198",
            "inside_source_rows": [0, 68, 136, 204, 272, 341, 409, 477, 545, 613, 682, 750, 818, 886, 954, 1023],
            "outside_source_rows": [0, 9, 18, 27, 36, 45, 54, 63],
            "inside_row_count": 16,
            "outside_row_count": 8,
            "global_unique_key_count": 24,
            "parent_seed_first": 3193,
            "parent_seed_last": 3448,
            "parent_count": 256,
            "inside_max_abs_difference": 0.0,
            "outside_max_abs_difference": 0.0,
            "control_cache_manifest_sha256": "0" * 64,
            "control_cache_reuse_authorized": False,
        },
    }


def _cache_directory(data_directory):
    value = data_directory.with_name(data_directory.name + "_cache")
    value.mkdir()
    return value


def test_program_pins_design_science_resources_and_all_authorization_closed():
    program = execution.load_canonical_program(verify_file_hashes=True)
    assert program["execution_design"]["commit"] == execution.EXECUTION_DESIGN_COMMIT
    assert program["fixed_science"]["beta"] == list(capability.SHARED_BETA)
    assert program["resource_contract"]["required_MemAvailable_GiB"] == 80
    assert program["resource_contract"]["Slurm_submission"] is False
    assert set(program["authorization"]) == execution.AUTHORIZATION_KEYS
    assert not any(program["authorization"].values())


def test_public_refusal_precedes_program_loader_factory_and_filesystem(monkeypatch):
    calls = []
    monkeypatch.setattr(execution, "load_canonical_program", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(PermissionError, match="unauthorized"):
        execution.run_production_execution(lambda: calls.append("factory"), Path("output"))
    assert calls == []


def test_program_mutation_and_open_authorization_are_rejected(monkeypatch, tmp_path):
    changed = tmp_path / "program.json"
    changed.write_bytes(execution.CANONICAL_PROGRAM.read_bytes() + b"\n")
    monkeypatch.setattr(execution, "CANONICAL_PROGRAM", changed)
    with pytest.raises(execution.ExecutionContractError, match="hash"):
        execution.load_canonical_program()

    value = json.loads(execution.ROOT.joinpath(
        "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_program.json"
    ).read_text())
    value["authorization"]["production_execution_authorized"] = True
    changed.write_text(json.dumps(value))
    monkeypatch.setattr(execution, "PROGRAM_SHA256", execution.sha256_file(changed))
    with pytest.raises(execution.ExecutionContractError, match="authorization"):
        execution.load_canonical_program(verify_file_hashes=False)


def test_fresh_factory_distinct_evaluators_shared_memo_and_exact_close(monkeypatch):
    contract = capability.load_frozen_contract()
    evaluators = []

    def builder():
        value = _Evaluator()
        evaluators.append(value)
        return value

    factory = execution.FreshExactLeaseFactory(builder, contract.cache_namespace)
    monkeypatch.setattr(
        capability, "_run_fixed_schedule_replicate_core", _replicate
    )
    products = capability._run_four_fresh_replicates(factory, contract)
    assert len(products) == 4 and len(evaluators) == 4
    assert sum(value.calls for value in evaluators) == 1
    assert all(value.closed for value in evaluators)
    assert all(lease.closed and lease.close_count == 1 for lease in factory.created_leases)
    assert len({id(lease.evaluator) for lease in factory.created_leases}) == 4
    assert all(product.parent_seeds[0] == 3193 for product in products)


def test_terminal_accessor_requires_final_keys_and_one_call():
    evaluator = _Evaluator()
    shared = {}
    oracle = execution._MemoizedAggregateOracle(evaluator, shared, 2026082301)
    replicate = _replicate(2026082301, oracle)
    first = oracle.terminal_parent_log_z(2026082301, replicate.keys.copy())
    assert first.shape == (2048, 256)
    with pytest.raises(execution.ExecutionContractError, match="call count"):
        oracle.terminal_parent_log_z(2026082301, replicate.keys.copy())
    wrong_seed = execution._MemoizedAggregateOracle(
        _Evaluator(), {}, 2026082301
    )
    wrong_replicate = _replicate(2026082301, wrong_seed)
    with pytest.raises(execution.ExecutionContractError, match="seed"):
        wrong_seed.terminal_parent_log_z(2026082302, wrong_replicate.keys.copy())
    other = execution._MemoizedAggregateOracle(_Evaluator(), {}, 2026082301)
    with pytest.raises(execution.ExecutionContractError, match="unevaluated"):
        other.terminal_parent_log_z(2026082301, replicate.keys.copy())


def test_private_fake_execution_publishes_read_only_exact_bundle(monkeypatch, tmp_path):
    program = execution.load_canonical_program(verify_file_hashes=True)
    contract = capability.load_frozen_contract()
    evaluators = []

    def builder():
        value = _Evaluator()
        evaluators.append(value)
        return value

    factory = execution.FreshExactLeaseFactory(builder, contract.cache_namespace)
    monkeypatch.setattr(
        capability, "_run_fixed_schedule_replicate_core", _replicate
    )
    cf4_opened_after_terminal_seal = []

    def post_terminal_cf4(summary):
        terminal = tmp_path / "terminal_parent_frozen.npz"
        replicates = [tmp_path / f"replicate_{index}.npz" for index in range(4)]
        assert terminal.is_file() and all(path.is_file() for path in replicates)
        assert all(path.stat().st_mode & 0o222 == 0 for path in [terminal, *replicates])
        assert not (tmp_path / "post_terminal_cf4_gates.json").exists()
        cf4_opened_after_terminal_seal.append(True)
        return _cf4(summary)

    result = execution._execute_reserved_synthetic_test_only(
        program,
        contract,
        tmp_path,
        lease_factory=factory,
        control_runner=_control,
        cf4_gate_provider=post_terminal_cf4,
        cache_directory=_cache_directory(tmp_path),
    )
    assert result["status"] == "complete_pass_production_smc"
    assert cf4_opened_after_terminal_seal == [True]
    checked = execution.validate_published_bundle(tmp_path)
    assert checked["valid_scientific_complete"] is True
    assert {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()} == {
        "sealed_oracle_control_summary.json",
        "replicate_0.npz", "replicate_1.npz", "replicate_2.npz",
        "replicate_3.npz", "terminal_parent_frozen.npz",
        "post_terminal_cf4_gates.npz", "post_terminal_cf4_gates.json",
        "result.json", "manifest.json",
    }
    assert all(
        path.stat().st_mode & 0o222 == 0
        for path in tmp_path.rglob("*") if path.is_file()
    )
    cache_directory = tmp_path.with_name(tmp_path.name + "_cache")
    assert {path.name for path in cache_directory.iterdir()} == {
        "shard_000000.npz", "manifest.json",
    }
    assert all(path.stat().st_mode & 0o222 == 0 for path in cache_directory.iterdir())


def test_postcheck_rejects_mutation_writable_and_unbound_file(monkeypatch, tmp_path):
    program = execution.load_canonical_program(verify_file_hashes=True)
    contract = capability.load_frozen_contract()
    factory = execution.FreshExactLeaseFactory(_Evaluator, contract.cache_namespace)
    monkeypatch.setattr(capability, "_run_fixed_schedule_replicate_core", _replicate)
    execution._execute_reserved_synthetic_test_only(
        program, contract, tmp_path, lease_factory=factory,
        control_runner=_control,
        cf4_gate_provider=_cf4,
        cache_directory=_cache_directory(tmp_path),
    )
    result = tmp_path / "result.json"
    result.chmod(0o644)
    with pytest.raises(execution.ExecutionContractError, match="writable"):
        execution.validate_published_bundle(tmp_path)
    result.chmod(0o444)
    extra = tmp_path / "extra.json"
    extra.write_text("{}")
    with pytest.raises(execution.ExecutionContractError, match="unbound"):
        execution.validate_published_bundle(tmp_path)
    extra.unlink()
    cache_directory = tmp_path.with_name(tmp_path.name + "_cache")
    nested_cache = cache_directory / "unbound"
    nested_cache.mkdir()
    (nested_cache / "hidden.bin").write_bytes(b"unbound")
    with pytest.raises(execution.ExecutionContractError, match="cache directory"):
        execution.validate_published_bundle(tmp_path)
    (nested_cache / "hidden.bin").unlink()
    nested_cache.rmdir()
    empty_cache = cache_directory / "empty"
    empty_cache.mkdir()
    with pytest.raises(execution.ExecutionContractError, match="cache directory"):
        execution.validate_published_bundle(tmp_path)
    empty_cache.rmdir()
    leftover_control = tmp_path / ".sealed_control_runtime"
    leftover_control.mkdir()
    with pytest.raises(execution.ExecutionContractError, match="unbound"):
        execution.validate_published_bundle(tmp_path)
    leftover_control.rmdir()
    control_path = tmp_path / "sealed_oracle_control_summary.json"
    external_control = tmp_path.parent / f"{tmp_path.name}_control_target.json"
    external_control.write_bytes(control_path.read_bytes())
    external_control.chmod(0o444)
    control_path.unlink()
    control_path.symlink_to(external_control)
    with pytest.raises(execution.ExecutionContractError, match="disk"):
        execution.validate_published_bundle(tmp_path)
    control_path.unlink()
    control_path.write_bytes(external_control.read_bytes())
    control_path.chmod(0o444)
    external_control.chmod(0o644)
    external_control.unlink()
    cf4_path = tmp_path / "post_terminal_cf4_gates.json"
    manifest_path = tmp_path / "manifest.json"
    cf4_value = json.loads(cf4_path.read_text())
    cf4_value["gates"]["weighted_CF4_Q90"] = not cf4_value["gates"][
        "weighted_CF4_Q90"
    ]
    cf4_path.chmod(0o644)
    cf4_path.write_text(json.dumps(cf4_value, indent=2, sort_keys=True) + "\n")
    cf4_path.chmod(0o444)
    manifest_value = json.loads(manifest_path.read_text())
    row = next(
        item for item in manifest_value["artifacts"]
        if item["path"] == str(cf4_path.resolve())
    )
    row["sha256"] = execution.sha256_file(cf4_path)
    row["byte_count"] = cf4_path.stat().st_size
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest_value, indent=2, sort_keys=True) + "\n")
    manifest_path.chmod(0o444)
    with pytest.raises(execution.ExecutionContractError, match="CF4 decision"):
        execution.validate_published_bundle(tmp_path)
    replicate_paths = [tmp_path / f"replicate_{index}.npz" for index in range(4)]
    terminal_path = tmp_path / "terminal_parent_frozen.npz"
    execution._validated_validity_gates(replicate_paths, terminal_path)
    with np.load(replicate_paths[0], allow_pickle=False) as item:
        arrays = {name: item[name].copy() for name in item.files}
    arrays["beta_history"][1] += 1e-4
    replicate_paths[0].chmod(0o644)
    np.savez(replicate_paths[0], **arrays)
    replicate_paths[0].chmod(0o444)
    with pytest.raises(execution.ExecutionContractError, match="history contract"):
        execution._validated_validity_gates(replicate_paths, terminal_path)


def test_scientific_gate_failure_is_valid_complete(monkeypatch, tmp_path):
    program = execution.load_canonical_program(verify_file_hashes=True)
    contract = capability.load_frozen_contract()
    factory = execution.FreshExactLeaseFactory(_Evaluator, contract.cache_namespace)
    monkeypatch.setattr(capability, "_run_fixed_schedule_replicate_core", _replicate)

    def ranged_replicate(seed, oracle, contract=None):
        value = _replicate(seed, oracle, contract)
        if seed == capability.MASTER_SEEDS[-1]:
            value.log_normalizer_increment[0] = 0.3
            value.log_normalizer = 0.3
        return value

    monkeypatch.setattr(capability, "_run_fixed_schedule_replicate_core", ranged_replicate)
    result = execution._execute_reserved_synthetic_test_only(
        program, contract, tmp_path, lease_factory=factory,
        control_runner=_control,
        cf4_gate_provider=_cf4,
        cache_directory=_cache_directory(tmp_path),
    )
    assert result["status"] == "complete_scientific_fail_production_smc"
    assert result["outcome_kind"] == "scientific_fail"
    assert result["failure_class"] == "replicate_log_I_bar_range"
    assert execution.validate_published_bundle(tmp_path)[
        "valid_scientific_complete"
    ] is True


def test_postcheck_rejects_parent_evidence_contrast_with_same_logmean(
    monkeypatch, tmp_path
):
    program = execution.load_canonical_program(verify_file_hashes=True)
    contract = capability.load_frozen_contract()
    factory = execution.FreshExactLeaseFactory(_Evaluator, contract.cache_namespace)
    monkeypatch.setattr(capability, "_run_fixed_schedule_replicate_core", _replicate)
    cache_directory = _cache_directory(tmp_path)
    execution._execute_reserved_synthetic_test_only(
        program, contract, tmp_path, lease_factory=factory,
        control_runner=_control, cf4_gate_provider=_cf4,
        cache_directory=cache_directory,
    )
    shard = cache_directory / "shard_000000.npz"
    with np.load(shard, allow_pickle=False) as item:
        keys = item["keys"].copy()
        log_z_bar = item["log_Z_bar"].copy()
    contrast = np.linspace(-2.0, 2.0, 256, dtype=np.float64)
    contrast -= float(execution.logmeanexp_parent(contrast[None, :])[0])
    forged_log_z = log_z_bar[:, None] + contrast[None, :]
    assert np.allclose(
        execution.logmeanexp_parent(forged_log_z), log_z_bar,
        rtol=0.0, atol=1e-12,
    )
    shard.chmod(0o644)
    np.savez(shard, keys=keys, log_Z=forged_log_z, log_Z_bar=log_z_bar)
    shard.chmod(0o444)
    shard_record = execution._artifact_record(
        shard, lineage=execution.PROGRAM_SHA256
    )
    cache_manifest_path = cache_directory / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    cache_manifest["shards"] = [shard_record]
    cache_manifest_path.chmod(0o644)
    cache_manifest_path.write_text(
        json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n"
    )
    cache_manifest_path.chmod(0o444)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    replacements = {
        str(shard.resolve()): shard_record,
        str(cache_manifest_path.resolve()): execution._artifact_record(
            cache_manifest_path, lineage=execution.PROGRAM_SHA256
        ),
    }
    manifest["artifacts"] = [
        replacements.get(row["path"], row) for row in manifest["artifacts"]
    ]
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_path.chmod(0o444)
    with pytest.raises(execution.ExecutionContractError, match="parent posterior"):
        execution.validate_published_bundle(tmp_path)


def test_private_control_failure_precedes_factory(monkeypatch, tmp_path):
    calls = []
    class Factory:
        def __call__(self, *args):
            calls.append(args)
    with pytest.raises(execution.ExecutionContractError, match="control"):
        execution._execute_reserved_synthetic_test_only(
            execution.load_canonical_program(), capability.load_frozen_contract(),
            tmp_path,
            lease_factory=Factory(),
            control_runner=lambda: {"control_rows": 23},
            cf4_gate_provider=lambda summary: {},
            cache_directory=_cache_directory(tmp_path),
        )
    assert calls == []


def test_private_executor_rejects_forged_program_before_artifacts(tmp_path):
    program = execution.load_canonical_program()
    forged = dict(program)
    forged["purpose"] = "forged"
    cache = _cache_directory(tmp_path)
    with pytest.raises(execution.ExecutionContractError, match="differs"):
        execution._execute_reserved_synthetic_test_only(
            forged, capability.load_frozen_contract(), tmp_path,
            lease_factory=lambda *args: None,
            control_runner=lambda: {},
            cf4_gate_provider=lambda summary: {},
            cache_directory=cache,
        )
    assert list(tmp_path.iterdir()) == [] and list(cache.iterdir()) == []


def test_canonical_private_entry_has_no_science_callback_override(monkeypatch, tmp_path):
    program = execution.load_canonical_program()
    contract = capability.load_frozen_contract()
    arrays = tmp_path / "arrays.npz"
    arrays.write_bytes(b"sealed")
    returned = _Evaluator()

    def fake_control(factory, regression_arrays, root):
        assert callable(factory) and regression_arrays == arrays
        root.mkdir()
        (root / "production_cache").mkdir()
        summary = root / "sealed_oracle_control_summary.json"
        summary.write_text(json.dumps(_control()["sealed_summary"]))
        control = SimpleNamespace(
            control_evaluator_discarded=True,
            control_cache_discarded=True,
            covariance_cache_identity_distinct=True,
            production_covariance_cached_key_count=0,
            production_covariance_evaluation_batches=0,
            production_cache_empty=True,
            summary_path=str(summary),
        )
        return control, returned, SimpleNamespace(shard_count=0)

    captured = {}

    def fake_execute(*args, **kwargs):
        captured.update(kwargs)
        control = kwargs["control_runner"]()
        assert control["sealed_summary"]["global_unique_key_count"] == 24
        return {"status": "synthetic-wiring-only"}

    monkeypatch.setattr(execution, "_canonical_regression_arrays", lambda: arrays)
    monkeypatch.setattr(execution, "run_sealed_regression_control", fake_control)
    monkeypatch.setattr(execution, "_actual_evaluator_builder", lambda value: _Evaluator)
    monkeypatch.setattr(execution, "_execute_reserved_synthetic_test_only", fake_execute)
    (tmp_path / "data").mkdir()
    (tmp_path / "cache").mkdir()
    result = execution._execute_reserved_canonical_private(
        program, contract, tmp_path / "data", tmp_path / "cache"
    )
    assert result["status"] == "synthetic-wiring-only"
    assert captured["cf4_gate_provider"] is execution._canonical_cf4_gate_provider
    assert returned.closed is True
    assert not (tmp_path / "data" / ".sealed_control_runtime").exists()
