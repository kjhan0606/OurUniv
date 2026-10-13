import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np

import cf4_aggregate_evidence_smc_execution as execution
from cf4_aggregate_evidence_parallel_oracle import (
    AppendOnlyEvidenceCache,
    RegressionControlResult,
)


def _program():
    return json.loads(execution.CANONICAL_PROGRAM.read_text())


class _Covariance:
    def __init__(self):
        self.evaluated_covariance_keys = 0
        self.evaluation_batches = 0


class _Evaluator:
    def __init__(self):
        self.covariance_cache = _Covariance()
        self.close_count = 0

    def __call__(self, keys):
        values = [tuple(int(item) for item in key) for key in keys]
        self.covariance_cache.evaluated_covariance_keys += len(values)
        self.covariance_cache.evaluation_batches += 1
        return values, np.zeros((len(values), 256), dtype=np.float64)

    def close(self):
        self.close_count += 1


def _control_runner(created):
    def run(factory, regression_arrays, namespace_root):
        assert Path(regression_arrays).name == "arrays.npz"
        root = Path(namespace_root)
        root.mkdir(parents=False, exist_ok=False)
        control = factory()
        created.append(control._evaluator)
        control.close()
        summary_path = root / "sealed_oracle_control_summary.json"
        execution._atomic_json(summary_path, {
            "schema": "ouruniv-cf4-sealed-oracle-production-control-summary-v1",
            "status": "complete_pass_exact_24_row_control",
            "selection_sha256": "1" * 64,
        })
        production = factory()
        created.append(production._evaluator)
        cache = AppendOnlyEvidenceCache(root / "production_cache")
        result = RegressionControlResult(
            0.0,
            0.0,
            "1" * 64,
            str(summary_path.resolve()),
            execution.sha256_file(summary_path),
            True,
            True,
            True,
            0,
            0,
            True,
        )
        return result, production, cache

    return run


def _atomic_npz(path, **arrays):
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rewrite_json(path, value):
    path.unlink()
    execution._atomic_json(path, value)


def _rewrite_npz(path, mutate):
    with np.load(path, allow_pickle=False) as item:
        arrays = {name: item[name].copy() for name in item.files}
    mutate(arrays)
    path.unlink()
    _atomic_npz(path, **arrays)


def _smc_records(data):
    root = data / "smc"
    paths = [
        *(root / f"replicate_{index}.npz" for index in range(4)),
        root / "terminal_parent_frozen.npz",
        root / "post_terminal_cf4_gates.npz",
        root / "post_terminal_cf4_gates.json",
    ]
    return [execution._recorded_artifact(path) for path in paths]


def _terminal_capability(status="complete_pass", failure=None):
    def core(oracle, output_directory):
        oracle.evaluate(
            np.zeros((1, 3), dtype=np.float64),
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        )
        root = Path(output_directory)
        root.mkdir(parents=False, exist_ok=False)
        for index, seed in enumerate((2026082301, 2026082302, 2026082303, 2026082304)):
            midpoint = np.zeros((2048, 3), dtype=np.float64)
            axis = np.tile(
                np.asarray([1.0, 0.0, 0.0], dtype=np.float64), (2048, 1)
            )
            keys = np.zeros((2048, 6), dtype=np.int16)
            keys[:, 3] = 3
            move = np.zeros((1, 4, 4), dtype=np.int64)
            scale = np.zeros((1, 4, 3), dtype=np.int64)
            _atomic_npz(
                root / f"replicate_{index}.npz",
                master_seed=np.asarray(seed, dtype=np.int64),
                midpoint_mpc_h=midpoint,
                axis=axis,
                keys=keys,
                weights=np.full(2048, 1.0 / 2048.0),
                log_Z_bar=np.zeros(2048, dtype=np.float64),
                ancestor_labels=np.arange(2048, dtype=np.int64),
                beta_history=np.asarray([0.0, 1.0]),
                conditional_ESS_history=np.asarray([2048.0]),
                particle_ESS_history=np.asarray([2048.0, 2048.0]),
                log_normalizer_increment=np.asarray([0.0]),
                log_I_bar=np.asarray(0.0, dtype=np.float64),
                genealogical_ESS=np.asarray(2048.0, dtype=np.float64),
                resampling_ancestors=np.empty((0, 2048), dtype=np.int64),
                move_proposal_count=move,
                move_acceptance_count=move,
                q_scale_proposal_count=scale,
                q_scale_acceptance_count=scale,
                axis_scale_proposal_count=scale,
                axis_scale_acceptance_count=scale,
            )
        _atomic_npz(
            root / "terminal_parent_frozen.npz",
            master_seed=np.asarray(
                [2026082301, 2026082302, 2026082303, 2026082304],
                dtype=np.int64,
            ),
            parent_seed=np.arange(3193, 3449, dtype=np.int32),
            log_I_bar=np.zeros(4),
            P_rep=np.full((4, 256), 1.0 / 256.0),
            P_pool=np.full(256, 1.0 / 256.0),
        )
        _atomic_npz(
            root / "post_terminal_cf4_gates.npz",
            parent_seed=np.arange(3193, 3449, dtype=np.int32),
            deviance=np.arange(256, dtype=np.float64),
            P_pool=np.full(256, 1.0 / 256.0),
        )
        terminal_path = root / "terminal_parent_frozen.npz"
        post_terminal_path = root / "post_terminal_cf4_gates.npz"
        summary = {
            "schema": "ouruniv-cf4-aggregate-evidence-post-terminal-cf4-gates-v1",
            "status": status,
            "failure_class": failure,
            "gates": {
                name: failure != failure_class
                for name, failure_class in execution.GATE_FAILURE_PRIORITY
            },
            "decision": {
                key: False for key in execution.CAPABILITY_DECISION_KEYS
            },
            "terminal_parent_frozen": str(terminal_path.resolve()),
            "terminal_parent_frozen_sha256": execution.sha256_file(terminal_path),
            "post_terminal_arrays": str(post_terminal_path.resolve()),
            "post_terminal_arrays_sha256": execution.sha256_file(post_terminal_path),
        }
        execution._atomic_json(root / "post_terminal_cf4_gates.json", summary)
        return summary

    return core


def _architecture_capability(oracle, output_directory):
    oracle.evaluate(
        np.zeros((1, 3), dtype=np.float64),
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
    )
    root = Path(output_directory)
    root.mkdir(parents=False, exist_ok=False)
    summary = {
        "schema": "ouruniv-cf4-aggregate-evidence-smc-capability-result-v1",
        "status": "complete_scientific_fail",
        "failure_class": "SMC_temperature_stagnation",
        "valid_scientific_architecture_stop": True,
        "CF4_calibration_opened": False,
        "automatic_retry_retune_or_scale_up_authorized": False,
        "decision": {
            key: False for key in execution.CAPABILITY_DECISION_KEYS
        },
    }
    execution._atomic_json(root / "capability_result.json", summary)
    return summary


def _run_fake(tmp_path, capability_core):
    data = tmp_path / "data"
    data.mkdir()
    created = []
    result = execution._execute_into_reserved_directory(
        _program(),
        data,
        validation_runner=lambda: {
            "schema": "ouruniv-cf4-aggregate-evidence-smc-synthetic-validation-v1",
            "status": "complete_pass",
            "all_pass": True,
        },
        evaluator_factory=_Evaluator,
        control_runner=_control_runner(created),
        capability_core=capability_core,
    )
    return data, created, result


def test_program_exactly_pins_audited_lineage_and_keeps_execution_closed():
    program = _program()
    execution.validate_program_document(program, verify_file_hashes=False)
    assert program["source_commit"] == "6630b6b04ab02e513d47f1667617384894eb349f"
    assert program["capability_commit"] == "22587e47232782feb4c08768d8f64d853d76e62b"
    assert program["capability_result_record"]["sha256"] == (
        "0eba7e9ae0bf09613b53cf636dcda8cef615be58dea3c645abca1b8d8786fc30"
    )
    assert len(program["audited_capability_files"]) == 5
    assert len(program["pinned_local_files"]) == 16
    for row in program["pinned_local_files"]:
        assert len(row["sha256"]) == 64
        path = execution._resolved_input_path(row["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    authorization = program["authorization"]
    allowed = "production_program_and_runner_design_and_implementation_authorized"
    assert authorization[allowed] is True
    assert not any(value for key, value in authorization.items() if key != allowed)
    assert program["storage"]["data_directory"] == str(execution.DATA_DIRECTORY)
    assert program["storage"]["state_directory"] == str(execution.STATE_DIRECTORY)


def test_public_entry_refuses_wrong_path_hash_and_false_authorization_before_work(
    tmp_path, monkeypatch
):
    calls = []
    assert set(inspect.signature(execution.run_production).parameters) == {
        "program_path"
    }
    monkeypatch.setattr(
        execution, "_execute_authorized_program", lambda program: calls.append(program)
    )
    with np.testing.assert_raises_regex(PermissionError, "canonical program"):
        execution.run_production(tmp_path / "wrong.json")
    with np.testing.assert_raises_regex(PermissionError, "not authorized"):
        execution.run_production(execution.CANONICAL_PROGRAM)
    monkeypatch.setattr(execution, "PROGRAM_SHA256", "0" * 64)
    with np.testing.assert_raises_regex(RuntimeError, "program hash mismatch"):
        execution.run_production(execution.CANONICAL_PROGRAM)
    assert calls == []
    assert not (tmp_path / "data").exists()


def test_fake_terminal_pass_seals_cache_artifacts_and_closes_both_pools(tmp_path):
    data, created, result = _run_fake(tmp_path, _terminal_capability())
    assert result["status"] == "complete_pass_production_smc"
    assert result["outcome_kind"] == "terminal"
    assert [item.close_count for item in created] == [1, 1]
    manifest = json.loads((data / "manifest.json").read_text())
    assert len(manifest["production_cache_shards"]) == 1
    assert len(manifest["SMC_artifacts"]) == 7
    assert manifest["all_NPZ_arrays_finite"] is True
    assert manifest["production_covariance_cached_key_count_at_open"] == 0
    assert manifest["production_covariance_evaluation_batches_at_open"] == 0
    assert not list(data.rglob(".*.tmp"))
    assert execution.validate_published_bundle(data) == {
        "status": "complete_pass_production_smc",
        "outcome_kind": "terminal",
        "failure_class": None,
        "valid_scientific_complete": True,
    }


def test_fake_stable_scientific_gate_failure_is_complete(tmp_path):
    data, created, result = _run_fake(
        tmp_path,
        _terminal_capability("complete_scientific_fail", "genealogical_ESS"),
    )
    assert result["status"] == "complete_scientific_fail_production_smc"
    assert result["failure_class"] == "genealogical_ESS"
    assert result["outcome_kind"] == "terminal"
    assert [item.close_count for item in created] == [1, 1]
    assert execution.validate_published_bundle(data)["valid_scientific_complete"] is True


def test_fake_architecture_stop_is_complete_without_terminal_artifacts(tmp_path):
    data, created, result = _run_fake(tmp_path, _architecture_capability)
    assert result["status"] == "complete_scientific_fail_production_smc"
    assert result["failure_class"] == "SMC_temperature_stagnation"
    assert result["outcome_kind"] == "architecture_stop"
    assert [item.close_count for item in created] == [1, 1]
    manifest = json.loads((data / "manifest.json").read_text())
    assert [Path(row["path"]).name for row in manifest["SMC_artifacts"]] == [
        "capability_result.json"
    ]
    assert execution.validate_published_bundle(data)["valid_scientific_complete"] is True


def test_invalid_core_failure_closes_pools_and_publishes_no_terminal_result(tmp_path):
    data = tmp_path / "invalid"
    data.mkdir()
    created = []

    def invalid_core(oracle, output_directory):
        Path(output_directory).mkdir()
        raise RuntimeError("oracle lineage or nonfinite failure")

    with np.testing.assert_raises_regex(RuntimeError, "oracle lineage"):
        execution._execute_into_reserved_directory(
            _program(),
            data,
            validation_runner=lambda: {
                "schema": "ouruniv-cf4-aggregate-evidence-smc-synthetic-validation-v1",
                "status": "complete_pass", "all_pass": True
            },
            evaluator_factory=_Evaluator,
            control_runner=_control_runner(created),
            capability_core=invalid_core,
        )
    assert [item.close_count for item in created] == [1, 1]
    assert not (data / "result.json").exists()
    assert not (data / "manifest.json").exists()


def test_postcheck_rejects_atomic_artifact_mutation(tmp_path):
    data, _, _ = _run_fake(tmp_path, _terminal_capability())
    with (data / "smc/terminal_parent_frozen.npz").open("ab") as stream:
        stream.write(b"mutation")
    with np.testing.assert_raises_regex(RuntimeError, "artifact hash changed"):
        execution.validate_published_bundle(data)


def test_postcheck_rejects_unmanifested_cache_shard(tmp_path):
    data, _, _ = _run_fake(tmp_path, _terminal_capability())
    shard = data / "oracle/production_cache/shard_999999.npz"
    _atomic_npz(
        shard,
        keys=np.asarray([[0, 0, 0, 3, 0, 0]], dtype=np.int16),
        log_Z=np.zeros((1, 256), dtype=np.float64),
        log_Z_bar=np.zeros(1, dtype=np.float64),
    )
    with np.testing.assert_raises_regex(RuntimeError, "shard set changed"):
        execution.validate_published_bundle(data)


def test_decision_and_capability_summary_require_exact_schema_keys_and_failures():
    exact_decision = {
        key: False for key in execution.CAPABILITY_DECISION_KEYS
    }
    execution._require_downstream_closed(
        {"decision": exact_decision}, execution.CAPABILITY_DECISION_KEYS
    )
    for changed in (
        {key: value for key, value in exact_decision.items() if key != "HOP_authorized"},
        {**exact_decision, "invented_authorization": False},
        {**exact_decision, "HOP_authorized": True},
    ):
        with np.testing.assert_raises_regex(RuntimeError, "forbidden downstream"):
            execution._require_downstream_closed(
                {"decision": changed}, execution.CAPABILITY_DECISION_KEYS
            )
    exact_result_decision = {
        key: False for key in execution.RESULT_DECISION_KEYS
    }
    for changed in (
        {
            key: value for key, value in exact_result_decision.items()
            if key != "candidate_generation_authorized"
        },
        {**exact_result_decision, "invented_authorization": False},
        {**exact_result_decision, "production_SMC_execution_authorized": True},
    ):
        with np.testing.assert_raises_regex(RuntimeError, "forbidden downstream"):
            execution._require_downstream_closed(
                {"decision": changed}, execution.RESULT_DECISION_KEYS
            )

    valid = {
        "schema": "ouruniv-cf4-aggregate-evidence-post-terminal-cf4-gates-v1",
        "status": "complete_pass",
        "failure_class": None,
        "gates": {
            name: True for name, _ in execution.GATE_FAILURE_PRIORITY
        },
    }
    assert execution._validate_capability_summary(valid) == (
        "complete_pass_production_smc", "terminal"
    )
    for changed in (
        {**valid, "schema": "wrong"},
        {**valid, "gates": {**valid["gates"], "extra": True}},
        {
            **valid,
            "status": "complete_scientific_fail",
            "failure_class": "invented_failure",
        },
    ):
        with np.testing.assert_raises(RuntimeError):
            execution._validate_capability_summary(changed)


def test_published_bundle_rejects_schema_lineage_binding_and_gate_mutations(tmp_path):
    mutations = (
        "manifest_schema", "program_path", "missing_lineage", "extra_lineage",
        "result_schema", "decision", "artifact_binding", "capability_gates",
    )
    for name in mutations:
        case = tmp_path / name
        case.mkdir()
        data, _, _ = _run_fake(case, _terminal_capability())
        result_path = data / "result.json"
        manifest_path = data / "manifest.json"
        result = json.loads(result_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if name == "manifest_schema":
            manifest["schema"] = "wrong"
        elif name == "program_path":
            manifest["program"] = str(data / "invented_program.json")
        elif name == "missing_lineage":
            manifest["local_lineage_sha256"].pop(
                "src/cf4_aggregate_evidence_smc.py"
            )
        elif name == "extra_lineage":
            manifest["external_lineage_sha256"]["invented"] = "0" * 64
        elif name == "result_schema":
            result["schema"] = "wrong"
        elif name == "decision":
            del result["decision"]["HOP_authorized"]
        elif name == "artifact_binding":
            result["synthetic_validation_sha256"] = "0" * 64
        else:
            result["capability_gates"]["genealogical_ESS"] = False
        if name not in {
            "manifest_schema", "program_path", "missing_lineage", "extra_lineage"
        }:
            _rewrite_json(result_path, result)
            manifest["result"] = execution._json_artifact(result_path)
        _rewrite_json(manifest_path, manifest)
        with np.testing.assert_raises(RuntimeError):
            execution.validate_published_bundle(data)


def test_published_bundle_rehashes_pinned_local_and_external_inputs(
    tmp_path, monkeypatch
):
    data, _, _ = _run_fake(tmp_path, _terminal_capability())
    program = _program()
    original_sha256_file = execution.sha256_file
    targets = (
        execution._resolved_input_path(program["pinned_local_files"][0]["path"]),
        execution._resolved_input_path(
            program["external_inputs"]["response_atlas_manifest"]["path"]
        ),
    )
    expected_messages = ("local hash mismatch", "external hash mismatch")
    for target, expected_message in zip(targets, expected_messages):
        def forged_sha256_file(path, *, _target=target):
            candidate = Path(path).resolve()
            if candidate == _target.resolve():
                return "0" * 64
            return original_sha256_file(path)

        monkeypatch.setattr(execution, "sha256_file", forged_sha256_file)
        with np.testing.assert_raises_regex(RuntimeError, expected_message):
            execution.validate_published_bundle(data)
        monkeypatch.setattr(execution, "sha256_file", original_sha256_file)


def test_terminal_capability_must_bind_both_published_npz_artifacts(tmp_path):
    for field in (
        "terminal_parent_frozen",
        "terminal_parent_frozen_sha256",
        "post_terminal_arrays",
        "post_terminal_arrays_sha256",
    ):
        for mutation in ("missing", "mismatch"):
            case = tmp_path / f"{field}_{mutation}"
            case.mkdir()
            data, _, _ = _run_fake(case, _terminal_capability())
            manifest_path = data / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            capability_path = data / "smc/post_terminal_cf4_gates.json"
            capability = json.loads(capability_path.read_text())
            if mutation == "missing":
                capability.pop(field)
            else:
                capability[field] = "0" * 64 if field.endswith("sha256") else str(
                    (data / "wrong.npz").resolve()
                )
            _rewrite_json(capability_path, capability)
            manifest["SMC_artifacts"][6] = execution._json_artifact(
                capability_path
            )
            _rewrite_json(manifest_path, manifest)
            with np.testing.assert_raises_regex(
                RuntimeError, "terminal capability artifact binding changed"
            ):
                execution.validate_published_bundle(data)


def test_architecture_stop_requires_exact_closed_state_flags(tmp_path):
    for field in (
        "CF4_calibration_opened",
        "automatic_retry_retune_or_scale_up_authorized",
    ):
        for mutation in ("missing", "true"):
            case = tmp_path / f"{field}_{mutation}"
            case.mkdir()
            data, _, _ = _run_fake(case, _architecture_capability)
            manifest_path = data / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            capability_path = data / "smc/capability_result.json"
            capability = json.loads(capability_path.read_text())
            if mutation == "missing":
                capability.pop(field)
            else:
                capability[field] = True
            _rewrite_json(capability_path, capability)
            manifest["SMC_artifacts"][0] = execution._json_artifact(
                capability_path
            )
            _rewrite_json(manifest_path, manifest)
            with np.testing.assert_raises_regex(
                RuntimeError, "architecture-stop closed-state flags changed"
            ):
                execution.validate_published_bundle(data)


def test_manifest_must_bind_sealed_control_selection_sha(tmp_path):
    for mutation in ("missing", "mismatch"):
        case = tmp_path / mutation
        case.mkdir()
        data, _, _ = _run_fake(case, _terminal_capability())
        manifest_path = data / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if mutation == "missing":
            manifest.pop("control_selection_sha256")
        else:
            manifest["control_selection_sha256"] = "2" * 64
        _rewrite_json(manifest_path, manifest)
        with np.testing.assert_raises_regex(
            RuntimeError, "control selection"
        ):
            execution.validate_published_bundle(data)


def _two_shard_cache(path):
    cache = AppendOnlyEvidenceCache(path)
    for x in (0, 1):
        keys = np.asarray([[x, 0, 0, 3, 0, 0]], dtype=np.int16)
        cache.append(keys, np.zeros((1, 256), dtype=np.float64))
    manifest_path, _ = cache.seal()
    value = json.loads(manifest_path.read_text())
    records = [
        execution._recorded_artifact(Path(row["path"]))
        for row in value["shards"]
    ]
    return manifest_path, value, records


def test_cache_validator_rejects_schema_order_rowcount_duplicate_and_logmean(tmp_path):
    for failure in (
        "schema", "order", "rowcount", "total", "duplicate", "logmean"
    ):
        directory = tmp_path / failure
        manifest_path, manifest, records = _two_shard_cache(directory)
        if failure == "schema":
            manifest["schema"] = "wrong"
        elif failure == "order":
            manifest["shards"] = list(reversed(manifest["shards"]))
            records = list(reversed(records))
        elif failure == "rowcount":
            manifest["shards"][0]["row_count"] = 2
        elif failure == "total":
            manifest["total_row_count"] = 3
        else:
            shard = Path(manifest["shards"][1 if failure == "duplicate" else 0]["path"])

            def mutate(arrays):
                if failure == "duplicate":
                    arrays["keys"][0, 0] = 0
                else:
                    arrays["log_Z_bar"] += 1.0

            _rewrite_npz(shard, mutate)
            index = 1 if failure == "duplicate" else 0
            manifest["shards"][index]["sha256"] = execution.sha256_file(shard)
            records[index] = execution._recorded_artifact(shard)
        _rewrite_json(manifest_path, manifest)
        with np.testing.assert_raises(RuntimeError):
            execution._validate_cache_bundle(manifest_path, records)


def test_smc_validator_rejects_seed_dtype_normalization_beta_and_history(tmp_path):
    cases = {
        "master": (0, lambda arrays: arrays.__setitem__(
            "master_seed", np.asarray(1, dtype=np.int64)
        )),
        "resampling_dtype": (0, lambda arrays: arrays.__setitem__(
            "resampling_ancestors", arrays["resampling_ancestors"].astype(np.float64)
        )),
        "resampling_rows": (0, lambda arrays: arrays.__setitem__(
            "particle_ESS_history", np.asarray([2048.0, 1000.0])
        )),
        "weights": (0, lambda arrays: arrays["weights"].__setitem__(0, 0.5)),
        "beta": (0, lambda arrays: arrays.__setitem__(
            "beta_history", np.asarray([0.0, 0.5])
        )),
        "history": (0, lambda arrays: arrays.__setitem__(
            "conditional_ESS_history", np.empty(0, dtype=np.float64)
        )),
        "parent_seed": (4, lambda arrays: arrays["parent_seed"].__setitem__(0, 1)),
        "parent_probability": (4, lambda arrays: arrays["P_pool"].__setitem__(0, 0.5)),
    }
    for name, (record_index, mutate) in cases.items():
        case = tmp_path / name
        case.mkdir()
        data, _, _ = _run_fake(case, _terminal_capability())
        records = _smc_records(data)
        path = Path(records[record_index]["path"])
        _rewrite_npz(path, mutate)
        records[record_index] = execution._recorded_artifact(path)
        with np.testing.assert_raises(RuntimeError):
            execution._validate_smc_artifact_records(records, "terminal")
