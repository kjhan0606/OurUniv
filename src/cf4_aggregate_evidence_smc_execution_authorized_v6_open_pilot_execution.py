"""One-shot, additive v6-open disposable-pilot executor.

The sealed pilot boundary is never modified or bypassed.  This module accepts
only a separately committed authorization record whose direct ancestry seals
this implementation and its implementation-result record.
"""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import stat
import subprocess
from pathlib import Path
from typing import Any, Callable

import cf4_aggregate_evidence_oracle as oracle_module
import cf4_aggregate_evidence_parallel_oracle as parallel_oracle_module
import cf4_aggregate_evidence_smc as smc_module
from cf4_aggregate_evidence_oracle import AggregateEvidenceControllerOracle
from cf4_aggregate_evidence_parallel_oracle import ParallelExactAtlasEvaluator, WORKER_PROCESSES
from cf4_aggregate_evidence_smc import run_smc_replicate
import cf4_aggregate_evidence_smc_execution_authorized_v6_open_pilot as sealed
import cf4_aggregate_evidence_smc_shared_annealing_v6 as shared

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_pilot_execution_result_record_v6_open.json"
IMPLEMENTATION_RESULT = ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_pilot_execution_implementation_result_record_v6_open.json"
BASE_PROGRAM = ROOT / "config/cf4_aggregate_evidence_smc_production_program.json"
RUNNER = ROOT / "scripts/run_cf4_aggregate_evidence_smc_authorized_v6_open_pilot_execution_lageunha.sh"
LAUNCHER = ROOT / "scripts/launch_cf4_aggregate_evidence_smc_authorized_v6_open_pilot_execution_lageunha.sh"
STATUS_SCRIPT = ROOT / "scripts/status_cf4_aggregate_evidence_smc_authorized_v6_open_pilot_execution.sh"
SOURCE = Path(__file__).resolve()
TEST_SOURCE = ROOT / "tests/test_cf4_aggregate_evidence_smc_execution_authorized_v6_open_pilot_execution.py"
TEST_RUNNER = ROOT / "tests/test_cf4_aggregate_evidence_smc_authorized_v6_open_pilot_execution_runner.py"
SEALED_BOUNDARY_COMMIT = "01677bf9c2c0f243d20c1d2899e20baa55d2075d"
BRANCH = "agent/freeze-zoom-pipeline"
BASE_PROGRAM_SHA = "74cd10fdff0171daff6984ebc8db13cfd82d6dc495891ff585b81ac9eb0129c5"
SCIENCE_PINS = {
    "smc_source": (ROOT / "src/cf4_aggregate_evidence_smc.py", "392c75d823fb055e7b592299aa90540b1176d4cfd3c11442b446e42fd11f8337"),
    "oracle_source": (ROOT / "src/cf4_aggregate_evidence_oracle.py", "3da4bd598f381e8a6fccd1dc2ae179cdd01e14d80ad0ad6c25dd1b3a93631d7f"),
    "parallel_oracle_source": (ROOT / "src/cf4_aggregate_evidence_parallel_oracle.py", "1d4d587a2e54c676f71c5c68fcd0744db21ec3006501f4185b3854f7c3b19e3c"),
    "shared_annealing_source": (ROOT / "src/cf4_aggregate_evidence_smc_shared_annealing_v6.py", "b6f676caca512af9bd88be12f54b08844bd3d3f6e335ac1e5cc9d4265482c060"),
    "phase_cache_source": (ROOT / "src/cf4_peak_evidence_phase_cache.py", "6359497a141aa0814c0dd663353ae6623f01e3676c0c2fdfea5a1b272e9d7106"),
    "projection_source": (ROOT / "src/cf4_projection_contract.py", "14ff16637980cf2c7565189b3e07bd899afe163ba71c6b1d123e70eb71a6f63f"),
}
EXPECTED_AUTH = {
    "implementation_authorized": True,
    "pilot_stage_authorized": True,
    "receipt_creation_authorized": True,
    "pilot_execution_authorized_now": True,
    "ephemeral_in_memory_pilot_cache_authorized": True,
    "persistent_cache_population_authorized": False,
    "pilot_cache_reuse_authorized": False,
    "production_stage_authorized": False,
    "downstream_execution_authorized": False,
    "automatic_follow_on_authorized": False,
}
RECORD_KEYS = {
    "schema", "status", "authorized_at", "authorization_id", "one_shot", "host",
    "commit_chain", "authorization", "hard_pins", "paths", "fixed_science",
    "runtime_precondition", "lifecycle",
}
RESULT_KEYS = {"schema", "status", "commit_lineage", "implementation_files", "tests", "runtime_state", "authorization"}
FOCUSED_TEST_COMMAND = [
    "env", "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=src", "pytest", "-q",
    "tests/test_cf4_aggregate_evidence_smc_execution_authorized_v6_open_pilot_execution.py",
    "tests/test_cf4_aggregate_evidence_smc_authorized_v6_open_pilot_execution_runner.py",
]
MARKERS = ("RUNNING", "COMPLETE", "FAILED")


def _json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PermissionError(f"{label} is absent")
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(f"{label} is invalid") from error
    if not isinstance(value, dict):
        raise PermissionError(f"{label} is invalid")
    return value


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _require_canonical_json_mode(path: Path, value: dict[str, Any], mode: int, label: str) -> None:
    if stat.S_IMODE(path.stat().st_mode) != mode or path.read_bytes() != _canonical_json(value):
        raise PermissionError(f"{label} mode or encoding changed")


def _full_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PermissionError(f"{label} is not a full SHA-256")
    return value


def _git_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise PermissionError(f"{label} is not a full Git SHA")
    return value


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _verify_commit_chain(record: dict[str, Any]) -> None:
    chain = record.get("commit_chain")
    if not isinstance(chain, dict) or set(chain) != {
        "implementation_result_commit", "implementation_commit",
        "sealed_boundary_commit", "branch", "remote_ref",
    }:
        raise PermissionError("pilot authorization commit chain changed")
    for name in ("implementation_result_commit", "implementation_commit", "sealed_boundary_commit"):
        _git_sha(chain.get(name), name)
    head = _git("rev-parse", "HEAD")
    tracking = _git("rev-parse", "@{upstream}")
    remote = _git("ls-remote", "origin", f"refs/heads/{BRANCH}").split()[0]
    commits = [head, _git("rev-parse", "HEAD^"), _git("rev-parse", "HEAD^^"), _git("rev-parse", "HEAD^^^")]
    expected = [head, chain["implementation_result_commit"], chain["implementation_commit"], SEALED_BOUNDARY_COMMIT]
    if head != tracking or head != remote or commits != expected or chain["sealed_boundary_commit"] != SEALED_BOUNDARY_COMMIT or chain["branch"] != BRANCH or chain["remote_ref"] != f"origin/{BRANCH}":
        raise PermissionError("pilot authorization commit lineage mismatch")
    if _git("diff", "--name-status", "HEAD^", "HEAD").splitlines() != [f"A\t{RECORD.relative_to(ROOT)}"]:
        raise PermissionError("authorization commit scope changed")
    if _git("diff", "--name-status", "HEAD^^", "HEAD^").splitlines() != [f"A\t{IMPLEMENTATION_RESULT.relative_to(ROOT)}"]:
        raise PermissionError("implementation-result commit scope changed")
    expected_implementation = sorted(
        f"A\t{path.relative_to(ROOT)}"
        for path in (SOURCE, RUNNER, LAUNCHER, STATUS_SCRIPT, TEST_SOURCE, TEST_RUNNER)
    )
    if sorted(_git("diff", "--name-status", "HEAD^^^", "HEAD^^").splitlines()) != expected_implementation:
        raise PermissionError("pilot implementation commit scope changed")
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "config", "src", "scripts", "tests"],
        cwd=ROOT,
        check=False,
    )
    if clean.returncode != 0:
        raise PermissionError("tracked pilot science paths are dirty")
    _verify_untracked_scope()


def _verify_untracked_scope() -> None:
    output = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=ROOT,
    )
    for raw in output.split(b"\0"):
        if not raw:
            continue
        try:
            entry = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PermissionError("worktree status is not UTF-8") from error
        if entry.startswith("?? scripts/tripwire/"):
            continue
        raise PermissionError(f"unauthorized worktree entry: {entry}")


def _pin_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _require_pin(record: dict[str, Any], name: str, path: Path, digest: str | None = None, mode: int | None = None) -> str:
    pin = record["hard_pins"].get(name)
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256", "mode"} or pin["path"] != _pin_path(path):
        raise PermissionError(f"{name} hard pin mismatch")
    expected = _full_sha(pin["sha256"], f"{name} SHA") if digest is None else digest
    if pin["sha256"] != expected or sealed.sha256_file(path) != expected:
        raise PermissionError(f"{name} bytes changed")
    expected_mode = stat.S_IMODE(path.stat().st_mode) if mode is None else mode
    if pin["mode"] != f"{expected_mode:04o}" or stat.S_IMODE(path.stat().st_mode) != expected_mode:
        raise PermissionError(f"{name} mode mismatch")
    return expected


def _fixed_science() -> dict[str, Any]:
    return {
        "pilot_master_seed": 2026082301, "particles_per_replicate": 2048,
        "parent_seed_range_inclusive": [3193, 3448], "parent_count": 256,
        "worker_processes": 8, "threads_per_worker": 1, "replicates_sequential": True,
        "target_CESS_fraction": 0.8, "resampling_ESS_fraction": 0.5,
        "MH_sweeps_per_stage": 4, "maximum_temperature_stages": 256,
    }


def _all_pinned_files(record: dict[str, Any]) -> dict[str, tuple[Path, str]]:
    fixed = {
        "sealed_pilot_design": (sealed.DESIGN, sealed.DESIGN_SHA, 0o644),
        "sealed_pilot_program": (sealed.PROGRAM, sealed.PROGRAM_SHA, 0o644),
        "sealed_pilot_source": (Path(sealed.__file__), "df000fc7b228a83cf1834a4b1da0012e16684df50b648ce45b3f5438234dad24", 0o644),
        "open_grant": (sealed.OPEN_GRANT, sealed.OPEN_GRANT_SHA, 0o644),
        "open_release": (sealed.OPEN_RELEASE, sealed.OPEN_RELEASE_SHA, 0o444),
        "open_manifest": (sealed.OPEN_MANIFEST, sealed.OPEN_MANIFEST_SHA, 0o444),
        "closed_grant": (sealed.CLOSED_GRANT, sealed.CLOSED_GRANT_SHA, 0o644),
        "closed_release": (sealed.CLOSED_RELEASE, sealed.CLOSED_RELEASE_SHA, 0o444),
        "closed_manifest": (sealed.CLOSED_MANIFEST, sealed.CLOSED_MANIFEST_SHA, 0o444),
        "preflight_implementation_record": (sealed.PREFLIGHT_IMPLEMENTATION, sealed.PREFLIGHT_IMPLEMENTATION_SHA, 0o644),
        "preflight_result_record": (sealed.PREFLIGHT_RESULT, sealed.PREFLIGHT_RESULT_SHA, 0o644),
        "base_program": (BASE_PROGRAM, BASE_PROGRAM_SHA, 0o644),
        "implementation_result": (IMPLEMENTATION_RESULT, None, 0o644),
        "execution_wrapper_source": (SOURCE, None, 0o644),
        "runner": (RUNNER, None, 0o755), "launcher": (LAUNCHER, None, 0o755),
        "status_script": (STATUS_SCRIPT, None, 0o755),
        "test_source": (TEST_SOURCE, None, 0o644), "test_runner": (TEST_RUNNER, None, 0o644),
    }
    for name, (path, digest) in SCIENCE_PINS.items():
        fixed[name] = (path, digest, 0o644)
    if set(record.get("hard_pins", {})) != set(fixed):
        raise PermissionError("pilot execution hard-pin keyset changed")
    output: dict[str, tuple[Path, str]] = {}
    for name, (path, digest, mode) in fixed.items():
        output[name] = (path, _require_pin(record, name, path, digest, mode))
    return output


def _validate_implementation_result(record: dict[str, Any]) -> None:
    result = _json(IMPLEMENTATION_RESULT, "pilot implementation result record")
    _require_canonical_json_mode(IMPLEMENTATION_RESULT, result, 0o644, "pilot implementation result")
    chain = record["commit_chain"]
    if set(result) != RESULT_KEYS or result.get("schema") != "ouruniv-cf4-v6-open-pilot-execution-implementation-result-v1" or result.get("status") != "complete_pass_postcommit_pilot_execution_implementation":
        raise PermissionError("pilot implementation result schema changed")
    if result.get("commit_lineage") != {"implementation_commit": chain["implementation_commit"], "parent_commit": SEALED_BOUNDARY_COMMIT, "branch": BRANCH, "remote_ref": f"origin/{BRANCH}"}:
        raise PermissionError("pilot implementation result lineage changed")
    paths = (SOURCE, RUNNER, LAUNCHER, STATUS_SCRIPT, TEST_SOURCE, TEST_RUNNER)
    rows = result.get("implementation_files")
    if not isinstance(rows, list) or len(rows) != len(paths):
        raise PermissionError("pilot implementation result file rows changed")
    by_path = {row.get("path"): row for row in rows if isinstance(row, dict)}
    for path in paths:
        relative = str(path.relative_to(ROOT))
        row = by_path.get(relative)
        pin = record["hard_pins"][_implementation_pin_name(path)]
        blob = _git("rev-parse", f"{chain['implementation_commit']}:{relative}")
        if row != {"path": relative, "sha256": pin["sha256"], "git_blob": blob, "mode": pin["mode"]}:
            raise PermissionError("pilot implementation result row changed")
    tests = result.get("tests")
    environment = tests.get("environment") if isinstance(tests, dict) else None
    if (
        not isinstance(tests, dict)
        or set(tests) != {"command", "passed", "environment"}
        or tests.get("command") != FOCUSED_TEST_COMMAND
        or tests.get("passed") != 20
        or not isinstance(environment, dict)
        or set(environment) != {"host", "python", "numpy", "scipy", "pytest"}
        or not all(isinstance(value, str) and value for value in environment.values())
    ):
        raise PermissionError("pilot implementation test provenance changed")
    if result.get("runtime_state") != {"receipt_present": False, "pilot_present": False, "data_present": False, "state_present": False, "execution_occurred": False} or result.get("authorization") != {"implementation_accepted": True, "pilot_execution_authorized": False, "production_execution_authorized": False}:
        raise PermissionError("pilot implementation result runtime or authorization changed")


def _implementation_pin_name(path: Path) -> str:
    return {
        SOURCE: "execution_wrapper_source", RUNNER: "runner", LAUNCHER: "launcher",
        STATUS_SCRIPT: "status_script", TEST_SOURCE: "test_source", TEST_RUNNER: "test_runner",
    }[path]


def _verify_runtime_absence() -> None:
    if any(path.exists() for path in (sealed.RECEIPTS, sealed.PILOT, sealed.DATA_FORBIDDEN, sealed.STATE_FORBIDDEN)):
        raise PermissionError("pilot runtime namespace is already present")


def _verify_local_resources() -> None:
    required = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1", "PYTHONPATH": str(ROOT / "src"), "MALLOC_ARENA_MAX": "2"}
    if any(os.environ.get(name) != value for name, value in required.items()):
        raise PermissionError("pilot thread or process environment mismatch")
    if (os.cpu_count() or 0) < 8:
        raise PermissionError("pilot requires eight worker CPUs")
    try:
        available_kib = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    except (OSError, StopIteration, ValueError) as error:
        raise PermissionError("pilot memory availability is unknown") from error
    if available_kib < 67_108_864:
        raise PermissionError("pilot requires at least 64 GiB available RAM")


def _verify_import_origins() -> None:
    expected = {
        oracle_module: SCIENCE_PINS["oracle_source"][0],
        parallel_oracle_module: SCIENCE_PINS["parallel_oracle_source"][0],
        smc_module: SCIENCE_PINS["smc_source"][0],
        shared: SCIENCE_PINS["shared_annealing_source"][0],
        sealed: ROOT / "src/cf4_aggregate_evidence_smc_execution_authorized_v6_open_pilot.py",
    }
    for module, path in expected.items():
        origin = getattr(module, "__file__", None)
        if origin is None or Path(origin).resolve() != path.resolve():
            raise PermissionError(f"noncanonical Python import origin for {module.__name__}")


def load_pilot_execution_record() -> tuple[dict[str, Any], dict[str, tuple[Path, str]]]:
    record = _json(RECORD, "pilot execution authorization record")
    _require_canonical_json_mode(RECORD, record, 0o644, "pilot execution authorization record")
    if set(record) != RECORD_KEYS or record.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-pilot-execution-v6-open-v2" or record.get("status") != "sealed_one_shot_pilot_execution_authorization" or record.get("one_shot") is not True or record.get("host") != "lageunha" or record.get("authorization") != EXPECTED_AUTH:
        raise PermissionError("pilot execution authorization record changed")
    _full_sha(record.get("authorization_id"), "authorization id")
    if record.get("authorized_at") != "2026-08-20T12:30:00+09:00" or record.get("fixed_science") != _fixed_science() or record.get("runtime_precondition") != {"receipt_root_absent": True, "pilot_root_absent": True, "data_root_absent": True, "state_root_absent": True}:
        raise PermissionError("pilot execution scientific or runtime contract changed")
    paths = record.get("paths")
    expected_paths = {
        "receipt_root": str(sealed.RECEIPTS), "pilot_root": str(sealed.PILOT),
        "data_root_forbidden": str(sealed.DATA_FORBIDDEN), "state_root_forbidden": str(sealed.STATE_FORBIDDEN),
    }
    if paths != expected_paths:
        raise PermissionError("pilot execution paths changed")
    if record.get("lifecycle") != {"release_anchor": "release.anchor", "snapshot": "snapshot.json", "running_marker": "RUNNING", "complete_marker": "COMPLETE", "failed_marker": "FAILED", "schedule_manifest": "schedule_manifest.json", "pilot_output_only": True, "preserve_failed_receipt": True, "automatic_retry": False}:
        raise PermissionError("pilot execution lifecycle changed")
    shared.validate_frozen_v6_parameters()
    if WORKER_PROCESSES != 8:
        raise PermissionError("parallel pilot worker count changed")
    _verify_local_resources()
    _verify_import_origins()
    sealed.verify_pinned_provenance(sealed.load_program())
    _verify_commit_chain(record)
    pinned = _all_pinned_files(record)
    _validate_implementation_result(record)
    _verify_runtime_absence()
    return record, pinned


def _snapshot(record_sha: str, pinned: dict[str, tuple[Path, str]]) -> dict[str, Any]:
    info = sealed.OPEN_RELEASE.stat()
    return {
        "schema": "ouruniv-cf4-v6-open-pilot-execution-snapshot-v2",
        "record_sha256": record_sha,
        "git_head": _git("rev-parse", "HEAD"),
        "pinned_sha256": {name: sealed.sha256_file(path) for name, (path, _) in sorted(pinned.items())},
        "release": {"sha256": sealed.sha256_file(sealed.OPEN_RELEASE), "dev": info.st_dev, "ino": info.st_ino, "size": info.st_size, "nlink": info.st_nlink},
    }


def _write_exclusive_json(path: Path, value: dict[str, Any], mode: int = 0o444) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json(value))
    os.chmod(path, mode)


def _publish_marker(receipt: Path, name: str, value: dict[str, Any]) -> None:
    if name not in MARKERS:
        raise ValueError("invalid lifecycle marker")
    for other in MARKERS:
        if other != "RUNNING" and (receipt / other).exists():
            raise PermissionError("pilot lifecycle marker already terminal")
    running = receipt / "RUNNING"
    if name != "RUNNING" and running.exists():
        running.unlink()
    _write_exclusive_json(receipt / name, value)


def _create_receipt(record: dict[str, Any], record_sha: str, pinned: dict[str, tuple[Path, str]]) -> tuple[Path, dict[str, Any]]:
    receipt = sealed.RECEIPTS / record["authorization_id"] / "pilot"
    try:
        signals = {signal.SIGINT, signal.SIGTERM}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals) if hasattr(signal, "pthread_sigmask") else None
        try:
            os.mkdir(sealed.RECEIPTS, 0o700)
            os.mkdir(receipt.parent, 0o700)
            os.mkdir(receipt, 0o700)
            _publish_marker(receipt, "RUNNING", {"status": "reserving_disposable_pilot"})
        finally:
            if previous_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        os.link(sealed.OPEN_RELEASE, receipt / "release.anchor")
        value = _snapshot(record_sha, pinned)
        _write_exclusive_json(receipt / "snapshot.json", value)
        return receipt, value
    except BaseException as error:
        if receipt.is_dir():
            try:
                _publish_marker(receipt, "FAILED", {"status": "failed_invalid_pilot_receipt", "failure_class": type(error).__name__, "message": str(error)})
            except BaseException:
                pass
        raise


def _revalidate(
    receipt: Path,
    expected: dict[str, Any],
    record: dict[str, Any],
    record_sha: str,
    pinned: dict[str, tuple[Path, str]],
) -> None:
    _verify_local_resources()
    current_record = _json(RECORD, "pilot execution authorization record")
    if current_record != record:
        raise PermissionError("pilot execution authorization record changed")
    _verify_commit_chain(record)
    current_pins = _all_pinned_files(record)
    _validate_implementation_result(record)
    if current_pins != pinned:
        raise PermissionError("pilot execution pinned-file set changed")
    anchor = receipt / "release.anchor"
    anchor_stat, source_stat = anchor.stat(), sealed.OPEN_RELEASE.stat()
    if not anchor.is_file() or (anchor_stat.st_dev, anchor_stat.st_ino, anchor_stat.st_size, anchor_stat.st_nlink) != (source_stat.st_dev, source_stat.st_ino, source_stat.st_size, source_stat.st_nlink) or sealed.sha256_file(anchor) != sealed.OPEN_RELEASE_SHA or sealed.sha256_file(RECORD) != record_sha or _snapshot(record_sha, pinned) != expected:
        raise PermissionError("pilot execution receipt provenance changed")
    if sealed.DATA_FORBIDDEN.exists() or sealed.STATE_FORBIDDEN.exists():
        raise PermissionError("forbidden production namespace appeared")


class _DisposablePilot:
    def __init__(self, evaluator: ParallelExactAtlasEvaluator, replicate: Any):
        self._evaluator = evaluator
        self.beta_history = replicate.beta_history
        self.master_seed = replicate.master_seed
        self.particle_count = len(replicate.weights)
        self.parent_seeds = tuple(range(3193, 3449))
        self.closed = False
        self.cache_disposed = False

    def close(self) -> None:
        if not self.closed:
            self._evaluator.close()
            self.closed = True
            self.cache_disposed = True


def _make_disposable_pilot() -> _DisposablePilot:
    base = _json(BASE_PROGRAM, "base program")
    external = base["external_inputs"]
    evaluator = ParallelExactAtlasEvaluator(Path(external["response_atlas_manifest"]["path"]), external["response_atlas_manifest"]["sha256"], Path(external["density_filter"]["path"]), external["density_filter"]["sha256"], ROOT / external["physical_model"]["path"], external["physical_model"]["sha256"])
    try:
        oracle = AggregateEvidenceControllerOracle(evaluator)
        replicate = run_smc_replicate(2026082301, oracle, particle_count=2048, target_cess_fraction=0.8, resampling_ess_fraction=0.5, maximum_temperature_stages=256, sweeps_per_stage=4)
        return _DisposablePilot(evaluator, replicate)
    except BaseException:
        evaluator.close()
        raise


def _run_disposable(factory: Callable[[], Any]) -> shared.SharedBetaSchedule:
    pilot = factory()
    try:
        return shared.freeze_shared_beta_schedule(pilot.beta_history, pilot_master_seed=pilot.master_seed, pilot_particle_count=pilot.particle_count, pilot_parent_seeds=pilot.parent_seeds)
    finally:
        pilot.close()
        if not bool(getattr(pilot, "closed", False)) or not bool(getattr(pilot, "cache_disposed", False)):
            raise shared.ArchitectureFailure("pilot was not closed and disposed")


class PilotInterrupted(RuntimeError):
    pass


def _signal_failure(signum: int, _frame: Any) -> None:
    raise PilotInterrupted(f"pilot interrupted by signal {signum}")


def run_authorized_disposable_pilot() -> dict[str, Any]:
    if sealed.ascii_lower_short_hostname(socket.gethostname()) != "lageunha":
        raise PermissionError("Lageunha host gate failed")
    record, pinned = load_pilot_execution_record()
    record_sha = sealed.sha256_file(RECORD)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, _signal_failure)
    try:
        receipt, before = _create_receipt(record, record_sha, pinned)
        _revalidate(receipt, before, record, record_sha, pinned)
        schedule = _run_disposable(_make_disposable_pilot)
        _revalidate(receipt, before, record, record_sha, pinned)
        output = sealed.PILOT / record["authorization_id"]
        os.mkdir(sealed.PILOT, 0o700)
        os.mkdir(output, 0o700)
        manifest = {
            "schema": "ouruniv-cf4-v6-open-pilot-schedule-manifest-v2",
            "status": "complete_disposable_pilot_schedule_only",
            "authorization_id": record["authorization_id"],
            "schedule": {"beta": list(schedule.beta), "sha256": schedule.schedule_sha256, "pilot_master_seed": schedule.pilot_master_seed, "particles": schedule.pilot_particle_count, "parent_count": len(schedule.pilot_parent_seeds)},
            "ephemeral_cache_disposed": True, "persistent_cache_created": False,
            "production_authorized": False, "downstream_authorized": False,
        }
        first, second = receipt / "schedule_manifest.json", output / "schedule_manifest.json"
        _write_exclusive_json(first, manifest)
        _write_exclusive_json(second, manifest)
        _revalidate(receipt, before, record, record_sha, pinned)
        if sealed.sha256_file(first) != sealed.sha256_file(second) or stat.S_IMODE(first.stat().st_mode) != 0o444 or stat.S_IMODE(second.stat().st_mode) != 0o444:
            raise PermissionError("pilot schedule manifests differ")
        _publish_marker(receipt, "COMPLETE", {"status": "complete_disposable_pilot_schedule_only", "schedule_sha256": schedule.schedule_sha256, "manifest_sha256": sealed.sha256_file(first)})
        if sum((receipt / name).exists() for name in MARKERS) != 1:
            raise PermissionError("pilot lifecycle marker count invalid")
        return manifest
    except BaseException as error:
        try:
            if "receipt" in locals() and not (receipt / "COMPLETE").exists() and not (receipt / "FAILED").exists():
                _publish_marker(receipt, "FAILED", {"status": "failed_invalid_pilot_execution", "failure_class": type(error).__name__, "message": str(error)})
        finally:
            raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def preflight_only() -> dict[str, Any]:
    if sealed.ascii_lower_short_hostname(socket.gethostname()) != "lageunha":
        raise PermissionError("Lageunha host gate failed")
    record, pinned = load_pilot_execution_record()
    return {"status": "complete_pilot_execution_preflight", "authorization_id": record["authorization_id"], "pinned_count": len(pinned)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    options = parser.parse_args()
    if options.preflight == options.run:
        raise SystemExit("exactly one of --preflight or --run is required")
    print(preflight_only() if options.preflight else run_authorized_disposable_pilot())
