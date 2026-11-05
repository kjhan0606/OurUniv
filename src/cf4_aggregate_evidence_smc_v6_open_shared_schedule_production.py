"""Artifact-free v6-open shared-schedule production capability.

This module contains pure validation, fixed-schedule SMC, terminal pooling, and
gate-classification logic only.  It deliberately has no executable entry point
and cannot create a cache, reserve a namespace, or publish an artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Literal, Mapping, NoReturn, Protocol, Sequence
import zlib

import numpy as np

import cf4_aggregate_evidence_smc as base
import cf4_aggregate_evidence_smc_capability as capability
import cf4_aggregate_evidence_smc_shared_annealing_v6 as shared
from cf4_aggregate_evidence_oracle import logmeanexp_parent


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_design.json"
ERRATUM = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_design_erratum_v1.json"
DESIGN_COMMIT = "fefb430523e058097f1d00e2f7602b36067d79e0"
DESIGN_SHA256 = "aa1425834627e6f5aa7e5b542b9354319fad3f3eff61a21b137dca36e7c371cc"
ERRATUM_COMMIT = "bc674604b36ea90e91549bd9bdcdf1791db9e216"
ERRATUM_SHA256 = "eeb32687e9dce54b89dc3a10241bfcc054719ab41934e74a2de732a86bcc9282"
SCHEDULE_SHA256 = "c1b96d871e8c66d04e5028ed32773dc19656ad5b60149a047e025d75a69221b6"
MANIFEST_SHA256 = "ad0caacaf3635f7c560b02758e26e3035a3e68f42aed7ed1d79e6ffaecd37aea"
CACHE_NAMESPACE = "/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_cache"
MASTER_SEEDS = (2026082301, 2026082302, 2026082303, 2026082304)
PARENT_SEEDS = tuple(range(3193, 3449))
PARTICLES = 2048
SHARED_BETA = (
    0.0,
    0.1531397846993059,
    0.3181984675451238,
    0.5129471527439684,
    0.7369993853258423,
    1.0,
)
DESIGN_KEYS = {
    "artifact_isolation", "authorization", "design_lineage",
    "diagnostic_outputs", "failure_semantics", "fixed_science",
    "forbidden_actions", "frozen_date", "hard_pins",
    "information_firewall", "lifecycle", "next_step", "pilot_schedule",
    "production_architecture", "prospective_gates", "purpose", "schema",
    "status",
}
ERRATUM_KEYS = {
    "audit_requirement", "authorization", "next_step", "purpose", "schema",
    "scientific_correction", "status", "superseded_design",
    "unchanged_contract",
}
HARD_PIN_KEYS = {
    "pilot_result_record", "receipt_schedule_manifest",
    "pilot_schedule_manifest", "shared_annealing_design",
    "shared_annealing_implementation_record", "shared_annealing_source",
    "production_capability_design", "production_capability_source",
    "production_base_program", "smc_source", "oracle_source",
    "parallel_oracle_source", "response_atlas_manifest",
    "oracle_regression_postmortem", "v5_postmortem",
}
CF4_GATE_KEYS = {
    "weighted_CF4_Q99_exceedance_mass",
    "weighted_CF4_Q90",
    "weighted_CF4_one_sided_KS_permutation",
}
VALIDITY_GATE_KEYS = {
    "lineage_and_authorization",
    "finite_and_artifact_contract",
    "terminal_phase_complete_and_sealed",
    "pilot_and_v5_state_reuse_absent",
    "shared_schedule_stage_parity",
    "all_replicates_reach_beta_one",
    "temperature_stagnation_absent",
    "maximum_temperature_stages",
}
PRE_CF4_GATE_KEYS = {
    "replicate_log_I_bar_range",
    "replicate_log_I_bar_sample_SE",
    "replicate_parent_probability_L1_null_tail",
    "genealogical_ESS",
    "pooled_parent_ESS",
    "maximum_pooled_parent_probability",
}
_CONTRACT_SEAL = object()
_PRODUCT_SEAL = object()
GATE_FAILURE_PRIORITY = (
    ("lineage_and_authorization", "invalid_lineage_or_authorization"),
    ("finite_and_artifact_contract", "nonfinite_or_artifact_contract"),
    ("terminal_phase_complete_and_sealed", "incomplete_or_unsealed_terminal_phase"),
    ("pilot_and_v5_state_reuse_absent", "pilot_or_v5_state_reuse_detected"),
    ("shared_schedule_stage_parity", "shared_schedule_stage_parity_architecture_failure"),
    ("all_replicates_reach_beta_one", "SMC_terminal_beta_not_one"),
    ("temperature_stagnation_absent", "SMC_temperature_stagnation"),
    ("maximum_temperature_stages", "SMC_maximum_temperature_stages"),
    ("replicate_log_I_bar_range", "replicate_log_I_bar_range"),
    ("replicate_log_I_bar_sample_SE", "replicate_log_I_bar_sample_SE"),
    ("replicate_parent_probability_L1_null_tail", "replicate_parent_probability_L1_null_tail"),
    ("genealogical_ESS", "genealogical_ESS"),
    ("pooled_parent_ESS", "pooled_parent_ESS"),
    ("maximum_pooled_parent_probability", "maximum_pooled_parent_probability"),
    ("weighted_CF4_Q99_exceedance_mass", "weighted_CF4_Q99_exceedance_mass"),
    ("weighted_CF4_Q90", "weighted_CF4_Q90"),
    ("weighted_CF4_one_sided_KS_permutation", "weighted_CF4_one_sided_KS_permutation"),
)
INVALID_FAILURES = {
    "invalid_lineage_or_authorization", "nonfinite_or_artifact_contract",
    "incomplete_or_unsealed_terminal_phase", "pilot_or_v5_state_reuse_detected",
    "shared_schedule_stage_parity_architecture_failure", "lifecycle_error",
}
CHANNEL_MAPPING = {
    "replicate_parent_probability_L1":
    "replicate_parent_probability_L1_null_tail",
}


class ArchitectureFailure(RuntimeError):
    """A fail-closed contract violation, never a scientific result."""


@dataclass(frozen=True)
class FrozenProductionContract:
    design_sha256: str
    erratum_sha256: str
    schedule_sha256: str
    beta: tuple[float, ...]
    master_seeds: tuple[int, ...]
    particles: int
    parent_seeds: tuple[int, ...]
    cache_namespace: str
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class FreshReplicateProvenance:
    master_seed: int
    fresh_token: str
    evaluator_namespace: str
    cache_namespace: str
    pilot_state_reused: bool
    v5_state_reused: bool
    pilot_cache_reused: bool
    pilot_particles_reused: bool
    pilot_rng_state_reused: bool
    evaluator_closed: bool
    evaluator_close_count: int


@dataclass(frozen=True)
class FreshReplicateProduct:
    replicate: base.SMCReplicate
    parent_log_z: np.ndarray
    parent_seeds: np.ndarray
    provenance: FreshReplicateProvenance
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class ValidatedReplicate:
    replicate: base.SMCReplicate
    parent_log_z: np.ndarray
    parent_seeds: np.ndarray
    parent_probability: np.ndarray
    provenance: FreshReplicateProvenance


@dataclass(frozen=True)
class TerminalSummary:
    master_seed: np.ndarray
    beta_history: np.ndarray
    log_I_bar: np.ndarray
    P_rep: np.ndarray
    P_pool: np.ndarray
    P_rep_arithmetic_mean_diagnostic_only: np.ndarray
    pooled_log_I_bar: float
    genealogical_ESS: np.ndarray
    provenance: tuple[FreshReplicateProvenance, ...]


@dataclass(frozen=True)
class PreCF4Diagnostics:
    metrics: Mapping[str, object]
    gates: Mapping[str, bool]
    failed_channels: tuple[str, ...]
    primary_failure: str | None


@dataclass(frozen=True)
class ProductionDecision:
    all_gates: Mapping[str, bool]
    failed_channels: tuple[str, ...]
    primary_failure: str | None
    outcome_kind: Literal["pass", "scientific_fail", "invalid"]


class FreshReplicateFactory(Protocol):
    def __call__(
        self, master_seed: int, contract: FrozenProductionContract
    ) -> FreshReplicateLease: ...


class FreshReplicateLease(Protocol):
    oracle: Any
    parent_seeds: np.ndarray
    provenance: FreshReplicateProvenance
    closed: bool
    close_count: int

    def terminal_parent_log_z(
        self, master_seed: int, keys: np.ndarray
    ) -> np.ndarray: ...

    def close(self) -> None: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArchitectureFailure(f"invalid frozen JSON: {path}") from error
    if not isinstance(value, dict):
        raise ArchitectureFailure(f"frozen JSON is not an object: {path}")
    return value


def _git_directory() -> Path:
    candidate = ROOT / ".git"
    if candidate.is_dir():
        return candidate
    if candidate.is_file():
        text = candidate.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir: "):
            path = Path(text[8:])
            return path if path.is_absolute() else (ROOT / path).resolve()
    raise ArchitectureFailure("Git metadata is unavailable")


def _git_object(sha: str) -> tuple[str, bytes]:
    if len(sha) != 40:
        raise ArchitectureFailure("invalid Git object identifier")
    path = _git_directory() / "objects" / sha[:2] / sha[2:]
    try:
        raw = zlib.decompress(path.read_bytes())
        header, payload = raw.split(b"\0", 1)
        kind, size = header.decode("ascii").split()
    except (OSError, ValueError, zlib.error, UnicodeDecodeError) as error:
        raise ArchitectureFailure("required loose Git object is unavailable") from error
    if int(size) != len(payload):
        raise ArchitectureFailure("Git object length mismatch")
    return kind, payload


def _head_sha() -> str:
    head = (_git_directory() / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    reference = head[5:]
    loose = _git_directory() / reference
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    packed = _git_directory() / "packed-refs"
    for line in packed.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "^")):
            sha, name = line.split(" ", 1)
            if name == reference:
                return sha
    raise ArchitectureFailure("HEAD reference is unresolved")


def _commit_fields(commit: str) -> tuple[str, tuple[str, ...]]:
    kind, payload = _git_object(commit)
    if kind != "commit":
        raise ArchitectureFailure("lineage object is not a commit")
    tree = ""
    parents: list[str] = []
    for line in payload.split(b"\n"):
        if line.startswith(b"tree "):
            tree = line[5:].decode("ascii")
        elif line.startswith(b"parent "):
            parents.append(line[7:].decode("ascii"))
        elif not line:
            break
    if len(tree) != 40:
        raise ArchitectureFailure("commit tree is missing")
    return tree, tuple(parents)


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    pending = [descendant]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current == ancestor:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(_commit_fields(current)[1])
    return False


def _tree_entries(tree: str) -> dict[str, tuple[str, str]]:
    kind, payload = _git_object(tree)
    if kind != "tree":
        raise ArchitectureFailure("Git path component is not a tree")
    result: dict[str, tuple[str, str]] = {}
    offset = 0
    while offset < len(payload):
        space = payload.index(b" ", offset)
        nul = payload.index(b"\0", space)
        mode = payload[offset:space].decode("ascii")
        name = payload[space + 1:nul].decode("utf-8")
        sha = payload[nul + 1:nul + 21].hex()
        result[name] = (mode, sha)
        offset = nul + 21
    return result


def _blob_at_commit(commit: str, relative: str) -> tuple[str, bytes]:
    tree = _commit_fields(commit)[0]
    parts = Path(relative).parts
    mode = ""
    sha = tree
    for index, part in enumerate(parts):
        entries = _tree_entries(sha)
        if part not in entries:
            raise ArchitectureFailure(f"Git path is absent: {relative}")
        mode, sha = entries[part]
        if index + 1 < len(parts) and mode != "40000":
            raise ArchitectureFailure(f"Git path is not a directory: {relative}")
    kind, payload = _git_object(sha)
    if kind != "blob":
        raise ArchitectureFailure(f"Git path is not a blob: {relative}")
    return sha, payload


def _verify_git_lineage() -> None:
    head = _head_sha()
    for commit, path in (
        (DESIGN_COMMIT, DESIGN),
        (ERRATUM_COMMIT, ERRATUM),
    ):
        if not _is_ancestor(commit, head):
            raise ArchitectureFailure("frozen design commit is not an ancestor")
        blob, payload = _blob_at_commit(commit, str(path.relative_to(ROOT)))
        expected_blob = hashlib.sha1(
            b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
        ).hexdigest()
        if blob != expected_blob or payload != path.read_bytes():
            raise ArchitectureFailure("frozen design blob differs from worktree")


def _validate_pin(name: str, pin: object) -> None:
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256", "mode", "role"}:
        raise ArchitectureFailure(f"invalid hard pin: {name}")
    path = Path(pin["path"])
    path = path if path.is_absolute() else ROOT / path
    if (
        _sha256(path) != pin["sha256"]
        or f"{stat.S_IMODE(path.stat().st_mode):04o}" != pin["mode"]
        or not isinstance(pin["role"], str)
        or not pin["role"]
    ):
        raise ArchitectureFailure(f"hard pin changed: {name}")


def load_frozen_contract() -> FrozenProductionContract:
    """Read and verify the committed design, erratum, and all frozen inputs."""
    if _sha256(DESIGN) != DESIGN_SHA256 or _sha256(ERRATUM) != ERRATUM_SHA256:
        raise ArchitectureFailure("design or erratum hash mismatch")
    design = _json(DESIGN)
    erratum = _json(ERRATUM)
    if (
        set(design) != DESIGN_KEYS
        or design.get("schema") != "ouruniv-cf4-v6-open-shared-schedule-production-design-v1"
        or design.get("status") != "frozen_prospective_design_only_all_execution_unauthorized"
        or set(erratum) != ERRATUM_KEYS
        or erratum.get("schema") != "ouruniv-cf4-v6-open-shared-schedule-production-design-erratum-v1"
        or erratum.get("status") != "prospective_scientific_correction_only_all_execution_unauthorized"
    ):
        raise ArchitectureFailure("design or erratum schema changed")
    _verify_git_lineage()
    pins = design.get("hard_pins")
    if not isinstance(pins, dict) or set(pins) != HARD_PIN_KEYS:
        raise ArchitectureFailure("design hard-pin keyset changed")
    for name, pin in pins.items():
        _validate_pin(name, pin)
    receipt = Path(pins["receipt_schedule_manifest"]["path"])
    pilot = Path(pins["pilot_schedule_manifest"]["path"])
    if receipt.read_bytes() != pilot.read_bytes() or _sha256(receipt) != MANIFEST_SHA256:
        raise ArchitectureFailure("sealed schedule manifests differ")
    schedule = design.get("pilot_schedule", {})
    correction = erratum.get("scientific_correction", {})
    expected_correction = {
        "field": "production_architecture.P_pool_definition",
        "old_value_forbidden": "mean(P_rep,axis=0)",
        "new_value": "evidence_weighted_pool_parent_probabilities(log_I_bar,P_rep)",
        "formula": "a=exp(log_I_bar-max(log_I_bar)); P_pool=sum(a[:,None]*P_rep,axis=0)/sum(a)",
        "pooled_log_I_bar": "max(log_I_bar)+log(mean(exp(log_I_bar-max(log_I_bar))))",
        "dtype": "float64",
        "normalization_tolerance": 1e-12,
        "canonical_source_path": "src/cf4_aggregate_evidence_smc.py",
        "canonical_source_sha256": "392c75d823fb055e7b592299aa90540b1176d4cfd3c11442b446e42fd11f8337",
        "L1_null_source": "corrected_evidence_weighted_P_pool",
        "arithmetic_mean_P_rep_role": "optional_diagnostic_only_never_gate_input",
    }
    if (
        tuple(schedule.get("beta", ())) != SHARED_BETA
        or schedule.get("schedule_sha256") != SCHEDULE_SHA256
        or schedule.get("manifest_sha256") != MANIFEST_SHA256
        or correction != expected_correction
        or erratum.get("superseded_design") != {
            "commit": DESIGN_COMMIT,
            "path": str(DESIGN.relative_to(ROOT)),
            "sha256": DESIGN_SHA256,
        }
        or erratum.get("unchanged_contract") != "all_other_fields_of_the_pinned_design_remain_exactly_in_force"
    ):
        raise ArchitectureFailure("schedule or erratum contract changed")
    frozen = shared.freeze_shared_beta_schedule(
        SHARED_BETA,
        pilot_master_seed=MASTER_SEEDS[0],
        pilot_particle_count=PARTICLES,
        pilot_parent_seeds=PARENT_SEEDS,
    )
    if frozen.schedule_sha256 != SCHEDULE_SHA256:
        raise ArchitectureFailure("shared schedule digest mismatch")
    isolation = design.get("artifact_isolation", {})
    roots = [isolation.get(name) for name in (
        "data_root", "state_root", "receipt_root", "cache_root"
    )]
    if any(not isinstance(path, str) or Path(path).exists() for path in roots):
        raise ArchitectureFailure("prospective production namespace is not absent")
    return FrozenProductionContract(
        design_sha256=DESIGN_SHA256,
        erratum_sha256=ERRATUM_SHA256,
        schedule_sha256=SCHEDULE_SHA256,
        beta=SHARED_BETA,
        master_seeds=MASTER_SEEDS,
        particles=PARTICLES,
        parent_seeds=PARENT_SEEDS,
        cache_namespace=str(isolation["cache_root"]),
        _seal=_CONTRACT_SEAL,
    )


def _require_contract_identity(contract: FrozenProductionContract) -> None:
    if (
        not isinstance(contract, FrozenProductionContract)
        or contract._seal is not _CONTRACT_SEAL
        or contract.design_sha256 != DESIGN_SHA256
        or contract.erratum_sha256 != ERRATUM_SHA256
        or contract.schedule_sha256 != SCHEDULE_SHA256
        or contract.beta != SHARED_BETA
        or contract.master_seeds != MASTER_SEEDS
        or contract.particles != PARTICLES
        or contract.parent_seeds != PARENT_SEEDS
        or contract.cache_namespace != CACHE_NAMESPACE
    ):
        raise ArchitectureFailure("frozen production contract identity changed")


def validate_shared_schedule(
    beta: Sequence[float], contract: FrozenProductionContract
) -> np.ndarray:
    _require_contract_identity(contract)
    value = np.asarray(beta, dtype=np.float64).copy()
    expected = np.asarray(contract.beta, dtype=np.float64)
    if (
        value.shape != expected.shape
        or not np.all(np.isfinite(value))
        or not np.array_equal(value, expected)
        or value[0] != 0.0
        or value[-1] != 1.0
        or np.any(np.diff(value) <= 0.0)
    ):
        raise ArchitectureFailure("shared_schedule_stage_parity_architecture_failure")
    value.flags.writeable = False
    return value


def _run_fixed_schedule_replicate_core(
    master_seed: int, oracle: Any, contract: FrozenProductionContract
) -> base.SMCReplicate:
    _require_contract_identity(contract)
    if master_seed not in contract.master_seeds or contract.particles != PARTICLES:
        raise ArchitectureFailure("fixed production master or particle count changed")
    beta_schedule = validate_shared_schedule(contract.beta, contract)
    midpoint, axis = base.initialize_particles(master_seed, PARTICLES)
    keys, log_z_bar = oracle.evaluate(midpoint, axis)
    log_z_bar = np.asarray(log_z_bar, dtype=np.float64)
    if log_z_bar.shape != (PARTICLES,) or not np.all(np.isfinite(log_z_bar)):
        raise RuntimeError("initial aggregate evidence is nonfinite")
    weights = np.full(PARTICLES, 1.0 / PARTICLES, dtype=np.float64)
    ancestors = np.arange(PARTICLES, dtype=np.int64)
    beta_history = [0.0]
    cess_history: list[float] = []
    ess_history = [base.particle_ess(weights)]
    increments: list[float] = []
    resampling_ancestors: list[np.ndarray] = []
    move_history: list[list[dict[str, Any]]] = []
    log_normalizer = 0.0
    resampling_event = 0
    for stage, (left, right) in enumerate(zip(beta_schedule, beta_schedule[1:])):
        delta = float(right - left)
        cess_history.append(base.conditional_ess(weights, log_z_bar, delta))
        weights, increment = base.update_weights_and_normalizer(
            weights, log_z_bar, delta
        )
        increments.append(increment)
        log_normalizer += increment
        pre_resampling_ess = base.particle_ess(weights)
        if pre_resampling_ess < base.RESAMPLING_ESS_FRACTION * PARTICLES:
            rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(
                int(master_seed), spawn_key=(1, resampling_event)
            )))
            selected = base.systematic_resampling(weights, rng)
            resampling_ancestors.append(selected)
            midpoint, axis, keys = midpoint[selected], axis[selected], keys[selected]
            log_z_bar, ancestors = log_z_bar[selected], ancestors[selected]
            weights = np.full(PARTICLES, 1.0 / PARTICLES, dtype=np.float64)
            resampling_event += 1
        stage_moves = []
        for sweep in range(base.SWEEPS_PER_STAGE):
            midpoint, axis, keys, log_z_bar, move = base.mh_rejuvenation_sweep(
                midpoint, axis, keys, log_z_bar, float(right), oracle,
                master_seed, stage, sweep,
            )
            stage_moves.append(move)
        move_history.append(stage_moves)
        beta_history.append(float(right))
        ess_history.append(pre_resampling_ess)
    return base.SMCReplicate(
        master_seed=int(master_seed), midpoint_mpc_h=np.asarray(midpoint),
        axis=np.asarray(axis), keys=np.asarray(keys), weights=np.asarray(weights),
        log_z_bar=np.asarray(log_z_bar), ancestor_labels=np.asarray(ancestors),
        beta_history=np.asarray(beta_history, dtype=np.float64),
        conditional_ess_history=np.asarray(cess_history, dtype=np.float64),
        particle_ess_history=np.asarray(ess_history, dtype=np.float64),
        log_normalizer_increment=np.asarray(increments, dtype=np.float64),
        resampling_ancestors=resampling_ancestors, move_history=move_history,
        log_normalizer=float(log_normalizer),
    )


def _freeze_array(value: np.ndarray, dtype: np.dtype[Any] | type) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.flags.writeable = False
    return result


def _validate_provenance(
    provenance: FreshReplicateProvenance,
    expected_seed: int,
    contract: FrozenProductionContract,
) -> None:
    if (
        provenance.master_seed != expected_seed
        or not provenance.fresh_token
        or not provenance.evaluator_namespace
        or provenance.cache_namespace != contract.cache_namespace
        or provenance.pilot_state_reused
        or provenance.v5_state_reused
        or provenance.pilot_cache_reused
        or provenance.pilot_particles_reused
        or provenance.pilot_rng_state_reused
        or not provenance.evaluator_closed
        or provenance.evaluator_close_count != 1
    ):
        raise ArchitectureFailure("pilot_or_v5_state_reuse_detected")


def validate_replicate_product(
    product: FreshReplicateProduct,
    expected_seed: int,
    contract: FrozenProductionContract,
) -> ValidatedReplicate:
    _require_contract_identity(contract)
    if (
        not isinstance(product, FreshReplicateProduct)
        or product._seal is not _PRODUCT_SEAL
    ):
        raise ArchitectureFailure("factory returned an invalid product")
    _validate_provenance(product.provenance, expected_seed, contract)
    if product.replicate.master_seed != expected_seed:
        raise ArchitectureFailure("replicate master seed changed")
    arrays = capability._replicate_arrays(product.replicate)
    validate_shared_schedule(arrays["beta_history"], contract)
    parent_log_z = np.asarray(product.parent_log_z)
    parent_seeds = np.asarray(product.parent_seeds)
    if (
        parent_log_z.dtype != np.float64
        or parent_log_z.shape != (PARTICLES, len(PARENT_SEEDS))
        or not np.all(np.isfinite(parent_log_z))
    ):
        raise ArchitectureFailure("terminal parent-logZ contract changed")
    if (
        parent_seeds.dtype != np.int64
        or parent_seeds.shape != (len(PARENT_SEEDS),)
        or not np.array_equal(parent_seeds, np.asarray(PARENT_SEEDS, dtype=np.int64))
    ):
        raise ArchitectureFailure("terminal parent seed order changed")
    aggregate = logmeanexp_parent(parent_log_z)
    if not np.allclose(
        aggregate, arrays["log_Z_bar"], rtol=0.0, atol=1e-12
    ):
        raise ArchitectureFailure("terminal parent-logZ aggregate mismatch")
    probability = base.replicate_parent_probability(
        arrays["weights"], parent_log_z
    )
    return ValidatedReplicate(
        replicate=product.replicate,
        parent_log_z=_freeze_array(parent_log_z, np.float64),
        parent_seeds=_freeze_array(parent_seeds, np.int64),
        parent_probability=_freeze_array(probability, np.float64),
        provenance=product.provenance,
    )


def _run_four_fresh_replicates(
    factory: FreshReplicateFactory, contract: FrozenProductionContract
) -> tuple[FreshReplicateProduct, ...]:
    _require_contract_identity(contract)
    products: list[FreshReplicateProduct] = []
    lease_ids: set[int] = set()
    oracle_ids: set[int] = set()
    for seed in contract.master_seeds:
        lease = factory(seed, contract)
        required = (
            "oracle", "terminal_parent_log_z", "parent_seeds", "provenance",
            "closed", "close_count", "close",
        )
        invalid_surface = any(not hasattr(lease, name) for name in required) or any(
            not callable(getattr(lease, name, None))
            for name in ("terminal_parent_log_z", "close")
        )
        if invalid_surface:
            close = getattr(lease, "close", None)
            if (
                callable(close)
                and getattr(lease, "closed", None) is False
                and getattr(lease, "close_count", None) == 0
            ):
                try:
                    close()
                except BaseException as error:
                    raise ArchitectureFailure("invalid lease close failed") from error
            raise ArchitectureFailure("factory returned an invalid lease")
        replicate: base.SMCReplicate | None = None
        parent_log_z: np.ndarray | None = None
        parent_seeds: np.ndarray | None = None
        primary_error: BaseException | None = None
        try:
            if (
                id(lease) in lease_ids
                or id(lease.oracle) in oracle_ids
                or lease.closed
                or lease.close_count != 0
            ):
                raise ArchitectureFailure("fresh evaluator lease isolation failed")
            lease_ids.add(id(lease))
            oracle_ids.add(id(lease.oracle))
            if not isinstance(lease.provenance, FreshReplicateProvenance):
                raise ArchitectureFailure("lease provenance type changed")
            if (
                lease.provenance.evaluator_closed
                or lease.provenance.evaluator_close_count != 0
            ):
                raise ArchitectureFailure("lease provenance was not fresh")
            _validate_provenance(
                replace(
                    lease.provenance,
                    evaluator_closed=True,
                    evaluator_close_count=1,
                ),
                seed,
                contract,
            )
            replicate = _run_fixed_schedule_replicate_core(
                seed, lease.oracle, contract
            )
            terminal_arrays = capability._replicate_arrays(replicate)
            terminal_keys = np.asarray(
                terminal_arrays["keys"], dtype=np.int16
            ).copy()
            parent_log_z = np.asarray(lease.terminal_parent_log_z(
                seed, terminal_keys
            )).copy()
            parent_seeds = np.asarray(lease.parent_seeds).copy()
        except BaseException as error:
            primary_error = error
        if not lease.closed and lease.close_count == 0:
            try:
                lease.close()
            except BaseException as error:
                raise ArchitectureFailure("evaluator lease close failed") from error
        if not lease.closed or lease.close_count != 1:
            raise ArchitectureFailure("evaluator lease was not closed exactly once")
        if primary_error is not None:
            raise primary_error
        if replicate is None or parent_log_z is None or parent_seeds is None:
            raise ArchitectureFailure("evaluator lease produced no terminal product")
        provenance = replace(
            lease.provenance,
            evaluator_closed=True,
            evaluator_close_count=1,
        )
        product = FreshReplicateProduct(
            replicate=replicate,
            parent_log_z=parent_log_z,
            parent_seeds=parent_seeds,
            provenance=provenance,
            _seal=_PRODUCT_SEAL,
        )
        validate_replicate_product(product, seed, contract)
        products.append(product)
    if (
        len(products) != 4
        or len({id(value) for value in products}) != 4
        or len({id(value.replicate) for value in products}) != 4
        or len({value.provenance.fresh_token for value in products}) != 4
        or len({value.provenance.evaluator_namespace for value in products}) != 4
        or tuple(value.provenance.master_seed for value in products) != contract.master_seeds
    ):
        raise ArchitectureFailure("fresh replicate factory isolation failed")
    return tuple(products)


def build_terminal_summary(
    products: Sequence[FreshReplicateProduct],
    contract: FrozenProductionContract,
) -> TerminalSummary:
    _require_contract_identity(contract)
    if len(products) != 4:
        raise ArchitectureFailure("exactly four terminal products are required")
    validated = tuple(
        validate_replicate_product(product, seed, contract)
        for product, seed in zip(products, contract.master_seeds)
    )
    if (
        len({id(product) for product in products}) != 4
        or len({id(value.replicate) for value in validated}) != 4
        or len({value.provenance.fresh_token for value in validated}) != 4
        or len({value.provenance.evaluator_namespace for value in validated}) != 4
    ):
        raise ArchitectureFailure("terminal products are not independent")
    log_i = np.asarray(
        [value.replicate.log_normalizer for value in validated], dtype=np.float64
    )
    p_rep = np.stack([value.parent_probability for value in validated])
    p_pool, pooled_log_i = base.pool_parent_probabilities(log_i, p_rep)
    maximum = float(np.max(log_i))
    relative = np.exp(log_i - maximum)
    independent_pool = np.sum(relative[:, None] * p_rep, axis=0) / np.sum(relative)
    independent_log_i = maximum + math.log(float(np.mean(relative)))
    if (
        not np.allclose(p_pool, independent_pool, rtol=0.0, atol=1e-12)
        or not math.isclose(pooled_log_i, independent_log_i, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ArchitectureFailure("evidence-weighted pooling disagrees with erratum")
    beta = np.stack([
        validate_shared_schedule(value.replicate.beta_history, contract)
        for value in validated
    ])
    return TerminalSummary(
        master_seed=_freeze_array(contract.master_seeds, np.int64),
        beta_history=_freeze_array(beta, np.float64),
        log_I_bar=_freeze_array(log_i, np.float64),
        P_rep=_freeze_array(p_rep, np.float64),
        P_pool=_freeze_array(p_pool, np.float64),
        P_rep_arithmetic_mean_diagnostic_only=_freeze_array(
            np.mean(p_rep, axis=0), np.float64
        ),
        pooled_log_I_bar=float(pooled_log_i),
        genealogical_ESS=_freeze_array([
            value.replicate.genealogical_ess for value in validated
        ], np.float64),
        provenance=tuple(value.provenance for value in validated),
    )


def _map_shared_diagnostic_channels(
    channels: Sequence[str],
) -> tuple[str, ...]:
    result: list[str] = []
    for channel in channels:
        mapped = CHANNEL_MAPPING.get(channel, channel)
        if mapped not in result:
            result.append(mapped)
    if "replicate_parent_probability_L1" in result:
        raise ArchitectureFailure("unmapped shared diagnostic channel")
    return tuple(result)


def _classify_primary_failure(
    failed_channels: Sequence[str],
    all_gates: Mapping[str, bool],
    contract: FrozenProductionContract,
) -> str | None:
    _require_contract_identity(contract)
    failed = _map_shared_diagnostic_channels(failed_channels)
    known = (
        {gate for gate, _ in GATE_FAILURE_PRIORITY}
        | {failure for _, failure in GATE_FAILURE_PRIORITY}
        | {"paired_incoherence"}
    )
    unknown = set(failed).difference(known)
    if unknown:
        raise ArchitectureFailure(f"unknown failure channel: {sorted(unknown)}")
    for gate, failure in GATE_FAILURE_PRIORITY[:8]:
        if all_gates.get(gate) is False or failure in failed:
            return failure
    if {
        "replicate_log_I_bar_range",
        "replicate_parent_probability_L1_null_tail",
    }.issubset(failed):
        return "paired_incoherence"
    for gate, failure in GATE_FAILURE_PRIORITY[8:]:
        if all_gates.get(gate) is False or failure in failed:
            return failure
    if failed:
        raise ArchitectureFailure(f"unknown failure channel: {failed[0]}")
    return None


def _validate_terminal_summary(
    summary: TerminalSummary, contract: FrozenProductionContract
) -> None:
    _require_contract_identity(contract)
    if not isinstance(summary, TerminalSummary):
        raise ArchitectureFailure("terminal summary type changed")
    arrays = (
        (summary.master_seed, np.int64, (4,)),
        (summary.beta_history, np.float64, (4, len(SHARED_BETA))),
        (summary.log_I_bar, np.float64, (4,)),
        (summary.P_rep, np.float64, (4, len(PARENT_SEEDS))),
        (summary.P_pool, np.float64, (len(PARENT_SEEDS),)),
        (
            summary.P_rep_arithmetic_mean_diagnostic_only,
            np.float64,
            (len(PARENT_SEEDS),),
        ),
        (summary.genealogical_ESS, np.float64, (4,)),
    )
    for value, dtype, shape in arrays:
        array = np.asarray(value)
        if array.dtype != dtype or array.shape != shape or not np.all(np.isfinite(array)):
            raise ArchitectureFailure("terminal summary array contract changed")
    if not np.array_equal(summary.master_seed, np.asarray(contract.master_seeds)):
        raise ArchitectureFailure("terminal master order changed")
    for row in summary.beta_history:
        validate_shared_schedule(row, contract)
    if (
        np.any(summary.P_rep < 0.0)
        or np.any(summary.P_pool < 0.0)
        or not np.allclose(summary.P_rep.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        or not math.isclose(float(summary.P_pool.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ArchitectureFailure("terminal parent probability contract changed")
    expected_pool, expected_log_i = base.pool_parent_probabilities(
        summary.log_I_bar, summary.P_rep
    )
    if (
        not np.allclose(summary.P_pool, expected_pool, rtol=0.0, atol=1e-12)
        or not math.isclose(
            summary.pooled_log_I_bar, expected_log_i, rel_tol=0.0, abs_tol=1e-12
        )
        or not np.allclose(
            summary.P_rep_arithmetic_mean_diagnostic_only,
            np.mean(summary.P_rep, axis=0),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ArchitectureFailure("terminal evidence pooling contract changed")
    if len(summary.provenance) != 4:
        raise ArchitectureFailure("terminal provenance count changed")
    for provenance, seed in zip(summary.provenance, contract.master_seeds):
        _validate_provenance(provenance, seed, contract)
    if (
        len({value.fresh_token for value in summary.provenance}) != 4
        or len({value.evaluator_namespace for value in summary.provenance}) != 4
    ):
        raise ArchitectureFailure("terminal provenance isolation changed")


def _null_exceedance_count(null_calibration: Mapping[str, object]) -> int:
    try:
        draws = null_calibration["draws"]
        seed = null_calibration["seed"]
        tail = float(null_calibration["tail_probability"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArchitectureFailure("L1 null calibration contract changed") from error
    if (
        type(draws) is not int
        or draws != shared.NULL_CALIBRATION_DRAWS
        or type(seed) is not int
        or seed != shared.NULL_CALIBRATION_SEED
        or not np.isfinite(tail)
    ):
        raise ArchitectureFailure("L1 null calibration contract changed")
    count = int(round(tail * (draws + 1) - 1.0))
    reconstructed = (count + 1) / (draws + 1)
    if count < 0 or count > draws or not math.isclose(
        tail, reconstructed, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ArchitectureFailure("L1 null exceedance count is inconsistent")
    return count


def evaluate_pre_cf4_diagnostics(
    summary: TerminalSummary,
    contract: FrozenProductionContract,
) -> PreCF4Diagnostics:
    _validate_terminal_summary(summary, contract)
    base_gates = capability.evaluate_pre_cf4_gates(
        summary.log_I_bar, summary.P_rep, summary.P_pool,
        summary.genealogical_ESS,
    )
    paired = shared.paired_incoherence_diagnostics(
        summary.log_I_bar, summary.P_rep, summary.P_pool,
        calibration_draws=shared.NULL_CALIBRATION_DRAWS,
    )
    null_calibration = paired["null_calibration"]
    exceedance_count = _null_exceedance_count(null_calibration)
    null_gate = bool(null_calibration["coherent_pass"])
    gates = {
        "replicate_log_I_bar_range": bool(paired["log_I_bar_range_pass"]),
        "replicate_log_I_bar_sample_SE": bool(paired["log_I_bar_sample_SE_pass"]),
        "replicate_parent_probability_L1_null_tail": null_gate,
        "genealogical_ESS": bool(base_gates["gates"]["genealogical_ESS"]),
        "pooled_parent_ESS": bool(base_gates["gates"]["pooled_parent_ESS"]),
        "maximum_pooled_parent_probability": bool(
            base_gates["gates"]["maximum_pooled_parent_probability"]
        ),
    }
    if set(gates) != PRE_CF4_GATE_KEYS:
        raise ArchitectureFailure("pre-CF4 gate keyset changed")
    failed = _map_shared_diagnostic_channels([
        name for name, passed in gates.items() if not passed
    ])
    primary = _classify_primary_failure(failed, gates, contract)
    metrics = dict(base_gates["metrics"])
    metrics.update({
        "L1_null_q99": paired["null_calibration"]["q99"],
        "L1_null_q999": paired["null_calibration"]["q999"],
        "L1_null_draws": paired["null_calibration"]["draws"],
        "L1_null_seed": paired["null_calibration"]["seed"],
        "L1_null_tail_probability": paired["null_calibration"]["tail_probability"],
        "L1_null_exceedance_count": exceedance_count,
        "L1_old_0p2_diagnostic_only_pass": paired["L1_diagnostic_threshold_pass"],
        "P_pool_source": "evidence_weighted",
    })
    return PreCF4Diagnostics(
        metrics=MappingProxyType(metrics),
        gates=MappingProxyType(gates),
        failed_channels=failed,
        primary_failure=primary,
    )


def classify_complete_gate_set(
    validity_gates: Mapping[str, bool],
    pre_cf4: PreCF4Diagnostics,
    cf4_gates: Mapping[str, bool],
    contract: FrozenProductionContract,
) -> ProductionDecision:
    _require_contract_identity(contract)
    if set(validity_gates) != VALIDITY_GATE_KEYS or any(
        type(value) is not bool for value in validity_gates.values()
    ):
        raise ArchitectureFailure("validity gate keyset is incomplete or unknown")
    if set(pre_cf4.gates) != PRE_CF4_GATE_KEYS or any(
        type(value) is not bool for value in pre_cf4.gates.values()
    ):
        raise ArchitectureFailure("pre-CF4 gate keyset is incomplete or unknown")
    if set(cf4_gates) != CF4_GATE_KEYS or any(
        type(value) is not bool for value in cf4_gates.values()
    ):
        raise ArchitectureFailure("CF4 gate keyset is incomplete or unknown")
    expected_pre_failed = _map_shared_diagnostic_channels(
        [name for name, passed in pre_cf4.gates.items() if passed is False]
    )
    if tuple(pre_cf4.failed_channels) != expected_pre_failed:
        raise ArchitectureFailure("pre-CF4 failed channels disagree with gates")
    expected_pre_primary = _classify_primary_failure(
        expected_pre_failed, pre_cf4.gates, contract
    )
    if pre_cf4.primary_failure != expected_pre_primary:
        raise ArchitectureFailure("pre-CF4 primary failure disagrees with gates")
    all_gates = dict(validity_gates)
    all_gates.update(pre_cf4.gates)
    all_gates.update(cf4_gates)
    failed = _map_shared_diagnostic_channels(
        tuple(name for name, passed in validity_gates.items() if not passed)
        + tuple(pre_cf4.failed_channels)
        + tuple(name for name, passed in cf4_gates.items() if not passed)
    )
    primary = _classify_primary_failure(failed, all_gates, contract)
    if primary is None:
        outcome: Literal["pass", "scientific_fail", "invalid"] = "pass"
    elif primary in INVALID_FAILURES:
        outcome = "invalid"
    else:
        outcome = "scientific_fail"
    return ProductionDecision(
        all_gates=MappingProxyType(all_gates),
        failed_channels=failed,
        primary_failure=primary,
        outcome_kind=outcome,
    )


def run_production_capability(*args: Any, **kwargs: Any) -> NoReturn:
    """Always refuse: no executable capability is authorized in this module."""
    raise PermissionError("v6-open shared-schedule production execution is unauthorized")
