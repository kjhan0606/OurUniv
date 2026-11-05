import copy
import inspect
import hashlib
from pathlib import Path
import signal
import subprocess

import pytest

import cf4_aggregate_evidence_smc_execution_authorized_v6_open_pilot_execution as execution


def _record():
    return {
        "schema": "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-pilot-execution-v6-open-v2",
        "status": "sealed_one_shot_pilot_execution_authorization", "authorized_at": "2026-08-20T12:30:00+09:00",
        "authorization_id": "1" * 64, "one_shot": True, "host": "lageunha", "commit_chain": {},
        "authorization": copy.deepcopy(execution.EXPECTED_AUTH), "hard_pins": {},
        "paths": {"receipt_root": str(execution.sealed.RECEIPTS), "pilot_root": str(execution.sealed.PILOT), "data_root_forbidden": str(execution.sealed.DATA_FORBIDDEN), "state_root_forbidden": str(execution.sealed.STATE_FORBIDDEN)},
        "fixed_science": execution._fixed_science(),
        "runtime_precondition": {"receipt_root_absent": True, "pilot_root_absent": True, "data_root_absent": True, "state_root_absent": True},
        "lifecycle": {"release_anchor": "release.anchor", "snapshot": "snapshot.json", "running_marker": "RUNNING", "complete_marker": "COMPLETE", "failed_marker": "FAILED", "schedule_manifest": "schedule_manifest.json", "pilot_output_only": True, "preserve_failed_receipt": True, "automatic_retry": False},
    }


def _patch_record_loader(monkeypatch, record):
    monkeypatch.setattr(execution, "_json", lambda path, label: copy.deepcopy(record) if path == execution.RECORD else {})
    monkeypatch.setattr(execution, "_verify_commit_chain", lambda value: None)
    monkeypatch.setattr(execution, "_all_pinned_files", lambda value: {"x": (execution.SOURCE, "0" * 64)})
    monkeypatch.setattr(execution, "_verify_runtime_absence", lambda: None)
    monkeypatch.setattr(execution, "_validate_implementation_result", lambda value: None)
    monkeypatch.setattr(execution.sealed, "load_program", lambda: {})
    monkeypatch.setattr(execution.sealed, "verify_pinned_provenance", lambda program: None)
    monkeypatch.setattr(execution.shared, "validate_frozen_v6_parameters", lambda: None)
    monkeypatch.setattr(execution.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(execution, "_verify_local_resources", lambda: None)
    monkeypatch.setattr(execution, "_require_canonical_json_mode", lambda path, value, mode, label: None)


def _patch_revalidation(monkeypatch, record, pinned):
    monkeypatch.setattr(execution, "_json", lambda path, label: copy.deepcopy(record))
    monkeypatch.setattr(execution, "_verify_commit_chain", lambda value: None)
    monkeypatch.setattr(execution, "_all_pinned_files", lambda value: pinned)
    monkeypatch.setattr(execution, "_validate_implementation_result", lambda value: None)
    monkeypatch.setattr(execution, "_verify_local_resources", lambda: None)


def test_record_exact_authorization_and_ephemeral_cache_contract(monkeypatch):
    record = _record(); _patch_record_loader(monkeypatch, record)
    loaded, _ = execution.load_pilot_execution_record()
    assert loaded["authorization"] == execution.EXPECTED_AUTH
    assert loaded["authorization"]["ephemeral_in_memory_pilot_cache_authorized"] is True
    assert loaded["authorization"]["persistent_cache_population_authorized"] is False
    record["authorization"]["persistent_cache_population_authorized"] = True
    _patch_record_loader(monkeypatch, record)
    with pytest.raises(PermissionError, match="authorization record changed"):
        execution.load_pilot_execution_record()


def test_public_host_gate_cannot_be_overridden(monkeypatch):
    assert list(inspect.signature(execution.run_authorized_disposable_pilot).parameters) == []
    assert list(inspect.signature(execution.preflight_only).parameters) == []
    monkeypatch.setattr(execution.socket, "gethostname", lambda: "syntax")
    with pytest.raises(PermissionError, match="host gate"):
        execution.run_authorized_disposable_pilot()


class _FakePilot:
    def __init__(self, beta=(0.0, 0.5, 1.0)):
        self.beta_history = beta; self.master_seed = 2026082301; self.particle_count = 2048
        self.parent_seeds = tuple(range(3193, 3449)); self.closed = False; self.cache_disposed = False
    def close(self):
        self.closed = True; self.cache_disposed = True


def test_fake_pilot_success_failure_and_exception_always_dispose():
    pilot = _FakePilot(); schedule = execution._run_disposable(lambda: pilot)
    assert schedule.beta == (0.0, 0.5, 1.0) and pilot.closed and pilot.cache_disposed
    bad = _FakePilot(beta=(0.0, 0.0, 1.0))
    with pytest.raises(Exception): execution._run_disposable(lambda: bad)
    assert bad.closed and bad.cache_disposed
    class Exploding(_FakePilot):
        @property
        def beta_history(self): raise RuntimeError("pilot failed")
        @beta_history.setter
        def beta_history(self, value): self._beta = value
    boom = Exploding()
    with pytest.raises(RuntimeError, match="pilot failed"): execution._run_disposable(lambda: boom)
    assert boom.closed and boom.cache_disposed


def _temporary_runtime(monkeypatch, tmp_path):
    receipt, pilot, data, state = (tmp_path / name for name in ("receipts", "pilot", "data", "state"))
    release, record = tmp_path / "release.json", tmp_path / "record.json"
    release.write_text("release"); record.write_text("record")
    monkeypatch.setattr(execution.sealed, "RECEIPTS", receipt); monkeypatch.setattr(execution.sealed, "PILOT", pilot)
    monkeypatch.setattr(execution.sealed, "DATA_FORBIDDEN", data); monkeypatch.setattr(execution.sealed, "STATE_FORBIDDEN", state)
    monkeypatch.setattr(execution.sealed, "OPEN_RELEASE", release); monkeypatch.setattr(execution.sealed, "OPEN_RELEASE_SHA", execution.sealed.sha256_file(release))
    monkeypatch.setattr(execution, "RECORD", record); monkeypatch.setattr(execution, "_snapshot", lambda record_sha, pinned: {"record": record_sha})
    return receipt, pilot, data, state, release, record


def test_receipt_one_shot_anchor_and_marker_lifecycle(monkeypatch, tmp_path):
    receipt_root, _, _, _, _, record_path = _temporary_runtime(monkeypatch, tmp_path)
    record = {"authorization_id": "a" * 64}; record_sha = execution.sealed.sha256_file(record_path)
    receipt, snapshot = execution._create_receipt(record, record_sha, {})
    assert (receipt / "RUNNING").is_file() and (receipt / "release.anchor").stat().st_ino == execution.sealed.OPEN_RELEASE.stat().st_ino
    _patch_revalidation(monkeypatch, record, {})
    execution._revalidate(receipt, snapshot, record, record_sha, {})
    execution._publish_marker(receipt, "COMPLETE", {"status": "complete"})
    assert [name for name in execution.MARKERS if (receipt / name).exists()] == ["COMPLETE"]
    with pytest.raises(FileExistsError): execution._create_receipt(record, record_sha, {})
    assert receipt_root.exists()


def test_receipt_failure_keeps_exact_failed_marker(monkeypatch, tmp_path):
    _temporary_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(execution, "_snapshot", lambda record_sha, pinned: (_ for _ in ()).throw(RuntimeError("snapshot failed")))
    with pytest.raises(RuntimeError, match="snapshot failed"):
        execution._create_receipt({"authorization_id": "b" * 64}, execution.sealed.sha256_file(execution.RECORD), {})
    receipt = execution.sealed.RECEIPTS / ("b" * 64) / "pilot"
    assert [name for name in execution.MARKERS if (receipt / name).exists()] == ["FAILED"]


def test_signal_during_running_publish_transitions_to_failed(monkeypatch, tmp_path):
    _temporary_runtime(monkeypatch, tmp_path)
    original = execution._write_exclusive_json
    def interrupt_after_write(path, value, mode=0o444):
        original(path, value, mode)
        if path.name == "RUNNING":
            raise execution.PilotInterrupted("injected signal")
    monkeypatch.setattr(execution, "_write_exclusive_json", interrupt_after_write)
    with pytest.raises(execution.PilotInterrupted, match="injected signal"):
        execution._create_receipt({"authorization_id": "f" * 64}, execution.sealed.sha256_file(execution.RECORD), {})
    receipt = execution.sealed.RECEIPTS / ("f" * 64) / "pilot"
    assert [name for name in execution.MARKERS if (receipt / name).exists()] == ["FAILED"]


def test_receipt_reservation_failure_before_directory_is_zero_write(monkeypatch, tmp_path):
    receipt_root, _, _, _, _, _ = _temporary_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(execution.os, "mkdir", lambda path, mode: (_ for _ in ()).throw(OSError("no reservation")))
    with pytest.raises(OSError, match="no reservation"):
        execution._create_receipt({"authorization_id": "0" * 64}, "0" * 64, {})
    assert not receipt_root.exists()


def test_postflight_record_or_anchor_mutation_is_rejected(monkeypatch, tmp_path):
    _, _, _, _, release, record_path = _temporary_runtime(monkeypatch, tmp_path)
    record = {"authorization_id": "c" * 64}; record_sha = execution.sealed.sha256_file(record_path)
    receipt, snapshot = execution._create_receipt(record, record_sha, {})
    _patch_revalidation(monkeypatch, record, {})
    record_path.write_text("changed")
    with pytest.raises(PermissionError, match="provenance"): execution._revalidate(receipt, snapshot, record, record_sha, {})
    record_path.write_text("record"); release.unlink(); release.write_text("release")
    with pytest.raises(PermissionError, match="provenance"): execution._revalidate(receipt, snapshot, record, execution.sealed.sha256_file(record_path), {})


def test_revalidate_rechecks_commit_chain_and_all_pins(monkeypatch, tmp_path):
    _temporary_runtime(monkeypatch, tmp_path)
    record = {"authorization_id": "d" * 64}; record_sha = execution.sealed.sha256_file(execution.RECORD)
    receipt, snapshot = execution._create_receipt(record, record_sha, {})
    calls = []
    monkeypatch.setattr(execution, "_json", lambda path, label: copy.deepcopy(record))
    monkeypatch.setattr(execution, "_verify_commit_chain", lambda value: calls.append("chain"))
    monkeypatch.setattr(execution, "_all_pinned_files", lambda value: calls.append("pins") or {})
    monkeypatch.setattr(execution, "_validate_implementation_result", lambda value: calls.append("result"))
    monkeypatch.setattr(execution, "_verify_local_resources", lambda: calls.append("resources"))
    execution._revalidate(receipt, snapshot, record, record_sha, {})
    assert calls == ["resources", "chain", "pins", "result"]


def test_direct_public_preflight_rejects_environment_bypass(monkeypatch):
    monkeypatch.setattr(execution.socket, "gethostname", lambda: "LagEunha.cluster")
    monkeypatch.setattr(execution, "_json", lambda path, label: _record())
    monkeypatch.setattr(execution, "_require_canonical_json_mode", lambda path, value, mode, label: None)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "PYTHONNOUSERSITE", "MALLOC_ARENA_MAX"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(PermissionError, match="environment mismatch"):
        execution.preflight_only()


def test_signal_handlers_are_installed_before_receipt_creation(monkeypatch):
    monkeypatch.setattr(execution.socket, "gethostname", lambda: "LagEunha")
    monkeypatch.setattr(execution, "load_pilot_execution_record", lambda: ({"authorization_id": "e" * 64}, {}))
    monkeypatch.setattr(execution.sealed, "sha256_file", lambda path: "0" * 64)
    installed = {}
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_DFL)
    monkeypatch.setattr(signal, "signal", lambda sig, handler: installed.__setitem__(sig, handler))
    def fail_after_check(record, record_sha, pinned):
        assert installed[signal.SIGINT] is execution._signal_failure
        assert installed[signal.SIGTERM] is execution._signal_failure
        raise RuntimeError("receipt stop")
    monkeypatch.setattr(execution, "_create_receipt", fail_after_check)
    with pytest.raises(RuntimeError, match="receipt stop"):
        execution.run_authorized_disposable_pilot()


def test_science_dependencies_and_no_private_production_path():
    source = Path(execution.__file__).read_text()
    assert set(execution.SCIENCE_PINS) == {"smc_source", "oracle_source", "parallel_oracle_source", "shared_annealing_source", "phase_cache_source", "projection_source"}
    assert "validate_frozen_v6_parameters" in source and "WORKER_PROCESSES != 8" in source
    assert "run_production" not in source and "_execute_into_reserved_directory" not in source
    assert "hostname:" not in source


def test_untracked_scope_rejects_shadow_modules_and_allows_tripwire(monkeypatch, tmp_path):
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    tripwire = tmp_path / "scripts/tripwire/owned"; tripwire.parent.mkdir(parents=True); tripwire.write_text("user\n")
    monkeypatch.setattr(execution, "ROOT", tmp_path)
    execution._verify_untracked_scope()
    (tmp_path / "cf4_aggregate_evidence_smc.py").write_text("raise RuntimeError('shadow')\n")
    with pytest.raises(PermissionError, match="unauthorized worktree entry"):
        execution._verify_untracked_scope()
    (tmp_path / "cf4_aggregate_evidence_smc.py").unlink()
    shadow = tmp_path / "src/sitecustomize.py"; shadow.parent.mkdir(exist_ok=True); shadow.write_text("raise RuntimeError('shadow')\n")
    with pytest.raises(PermissionError, match="unauthorized worktree entry"):
        execution._verify_untracked_scope()


def test_import_origin_must_be_canonical(monkeypatch):
    monkeypatch.setattr(execution.smc_module, "__file__", "/tmp/shadow/cf4_aggregate_evidence_smc.py")
    with pytest.raises(PermissionError, match="noncanonical Python import origin"):
        execution._verify_import_origins()


def _git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _commit(repo, message):
    subprocess.run(["git", "-c", "user.name=pilot-test", "-c", "user.email=pilot@example.invalid", "commit", "-m", message], cwd=repo, check=True, capture_output=True)
    return _git(repo, "rev-parse", "HEAD")


def test_actual_temporary_git_chain_and_remote_ref_mutation(monkeypatch, tmp_path):
    repo, remote = tmp_path / "repo", tmp_path / "remote.git"
    repo.mkdir(); subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", str(remote)], cwd=repo, check=True)
    (repo / "README").write_text("sealed\n"); subprocess.run(["git", "add", "README"], cwd=repo, check=True)
    sealed_commit = _commit(repo, "sealed")

    paths = {
        "SOURCE": repo / "src/wrapper.py", "RUNNER": repo / "scripts/run.sh",
        "LAUNCHER": repo / "scripts/launch.sh", "STATUS_SCRIPT": repo / "scripts/status.sh",
        "TEST_SOURCE": repo / "tests/test_source.py", "TEST_RUNNER": repo / "tests/test_runner.py",
    }
    for name, path in paths.items():
        path.parent.mkdir(exist_ok=True); path.write_text(f"{name}\n")
        path.chmod(0o755 if path.suffix == ".sh" else 0o644)
    subprocess.run(["git", "add", "src", "scripts", "tests"], cwd=repo, check=True)
    implementation_commit = _commit(repo, "implementation")
    result = repo / "config/result.json"; result.parent.mkdir(); result.write_text("{}\n")
    subprocess.run(["git", "add", str(result.relative_to(repo))], cwd=repo, check=True)
    result_commit = _commit(repo, "result")
    record_path = repo / "config/record.json"; record_path.write_text("{}\n")
    subprocess.run(["git", "add", str(record_path.relative_to(repo))], cwd=repo, check=True)
    head = _commit(repo, "authorization")
    subprocess.run(["git", "push", "-u", "origin", "main"], cwd=repo, check=True, capture_output=True)

    monkeypatch.setattr(execution, "ROOT", repo); monkeypatch.setattr(execution, "BRANCH", "main")
    monkeypatch.setattr(execution, "SEALED_BOUNDARY_COMMIT", sealed_commit)
    monkeypatch.setattr(execution, "RECORD", record_path); monkeypatch.setattr(execution, "IMPLEMENTATION_RESULT", result)
    for name, path in paths.items(): monkeypatch.setattr(execution, name, path)
    record = {"commit_chain": {"implementation_result_commit": result_commit, "implementation_commit": implementation_commit, "sealed_boundary_commit": sealed_commit, "branch": "main", "remote_ref": "origin/main"}}
    execution._verify_commit_chain(record)
    subprocess.run(["git", "update-ref", "refs/heads/main", result_commit], cwd=remote, check=True)
    with pytest.raises(PermissionError, match="lineage mismatch"):
        execution._verify_commit_chain(record)


def test_implementation_result_semantic_tamper_and_mode_change(monkeypatch, tmp_path):
    paths = [tmp_path / "wrapper.py", tmp_path / "run.sh", tmp_path / "launch.sh", tmp_path / "status.sh", tmp_path / "test_source.py", tmp_path / "test_runner.py"]
    for index, path in enumerate(paths):
        path.write_text(f"file-{index}\n"); path.chmod(0o755 if path.suffix == ".sh" else 0o644)
    names = ("SOURCE", "RUNNER", "LAUNCHER", "STATUS_SCRIPT", "TEST_SOURCE", "TEST_RUNNER")
    for name, path in zip(names, paths): monkeypatch.setattr(execution, name, path)
    monkeypatch.setattr(execution, "ROOT", tmp_path)
    result_path = tmp_path / "result.json"; monkeypatch.setattr(execution, "IMPLEMENTATION_RESULT", result_path)
    commit = "1" * 40
    rows, pins, blobs = [], {}, {}
    for path in paths:
        relative = str(path.relative_to(tmp_path)); digest = hashlib.sha256(path.read_bytes()).hexdigest()
        pin_name = execution._implementation_pin_name(path); mode = "0755" if path.suffix == ".sh" else "0644"
        pins[pin_name] = {"path": relative, "sha256": digest, "mode": mode}
        blob = hashlib.sha1(path.read_bytes()).hexdigest(); blobs[f"{commit}:{relative}"] = blob
        rows.append({"path": relative, "sha256": digest, "git_blob": blob, "mode": mode})
    value = {
        "schema": "ouruniv-cf4-v6-open-pilot-execution-implementation-result-v1",
        "status": "complete_pass_postcommit_pilot_execution_implementation",
        "commit_lineage": {"implementation_commit": commit, "parent_commit": execution.SEALED_BOUNDARY_COMMIT, "branch": execution.BRANCH, "remote_ref": f"origin/{execution.BRANCH}"},
        "implementation_files": rows,
        "tests": {"command": execution.FOCUSED_TEST_COMMAND, "passed": 20, "environment": {"host": "test", "python": "test", "numpy": "test", "scipy": "test", "pytest": "test"}},
        "runtime_state": {"receipt_present": False, "pilot_present": False, "data_present": False, "state_present": False, "execution_occurred": False},
        "authorization": {"implementation_accepted": True, "pilot_execution_authorized": False, "production_execution_authorized": False},
    }
    result_path.write_bytes(execution._canonical_json(value)); result_path.chmod(0o644)
    monkeypatch.setattr(execution, "_git", lambda *args: blobs[args[1]])
    record = {"commit_chain": {"implementation_commit": commit}, "hard_pins": pins}
    execution._validate_implementation_result(record)
    value["runtime_state"]["execution_occurred"] = True; result_path.write_bytes(execution._canonical_json(value))
    with pytest.raises(PermissionError, match="runtime or authorization"):
        execution._validate_implementation_result(record)

    target = paths[0]; target.chmod(0o666)
    pin = {"hard_pins": {"target": {"path": str(target.relative_to(tmp_path)), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "mode": "0644"}}}
    with pytest.raises(PermissionError, match="mode mismatch"):
        execution._require_pin(pin, "target", target, mode=0o644)
