#!/usr/bin/env python3
"""Prospective one-shot SMC capability; production remains unauthorized."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from cf4_aggregate_evidence_oracle import (
    PRODUCTION_PARENT_SEEDS,
    PRODUCTION_REPLICATE_MASTER_SEEDS,
    canonical_axis,
    geometry_key,
    logmeanexp_parent,
    sha256_file,
)
from cf4_aggregate_evidence_smc import (
    AXIS_KAPPA,
    AXIS_KAPPA_PROBABILITIES,
    MAXIMUM_TEMPERATURE_STAGES,
    MOVE_PROBABILITIES,
    PARTICLE_COUNT,
    Q_SCALES,
    Q_SCALE_PROBABILITIES,
    RESAMPLING_ESS_FRACTION,
    SWEEPS_PER_STAGE,
    TARGET_CESS_FRACTION,
    genealogical_ess,
    pool_parent_probabilities,
    replicate_parent_probability,
    run_smc_replicate,
)


ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_DESIGN = (
    ROOT / "config/cf4_aggregate_evidence_smc_production_capability_design.json"
)
CAPABILITY_DESIGN_SHA256 = (
    "2a253e049da3c02ef3cff4fc68a72a1f70efde616670538274a01d733460c888"
)
CANONICAL_PROGRAM = (
    ROOT / "config/cf4_aggregate_evidence_smc_production_program.json"
)
PERMUTATION_SEED = 2026081901
PERMUTATIONS = 100000
TERMINAL_KEYS = ("master_seed", "parent_seed", "log_I_bar", "P_rep", "P_pool")
CALIBRATION_PATH = Path(
    "/gpfs/kjhan/CF4/recon/linear_cr/v8_cf4_mode_release_reference/calibration.json"
)
CALIBRATION_SHA256 = (
    "c9edb6d0a108746fe18fa75295ab73f53286a25f9aa2725d132a0560375cb988"
)
CALIBRATION_SCHEMA = "ouruniv-cf4-lg-v8-mode-release-reference-calibration-v1"
CALIBRATION_STATUS = "complete_reference_calibration_parent3429_pass"


def _sha256(path: Path) -> str:
    return sha256_file(Path(path))


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_compiled_constants() -> None:
    exact = bool(
        PARTICLE_COUNT == 2048
        and PRODUCTION_REPLICATE_MASTER_SEEDS
        == (2026082301, 2026082302, 2026082303, 2026082304)
        and len(PRODUCTION_PARENT_SEEDS) == 256
        and PRODUCTION_PARENT_SEEDS == tuple(range(3193, 3449))
        and TARGET_CESS_FRACTION == 0.8
        and RESAMPLING_ESS_FRACTION == 0.5
        and SWEEPS_PER_STAGE == 4
        and MAXIMUM_TEMPERATURE_STAGES == 256
        and np.array_equal(MOVE_PROBABILITIES, [0.4, 0.3, 0.2, 0.1])
        and np.array_equal(Q_SCALES, [0.25, 0.6, 1.5])
        and np.array_equal(Q_SCALE_PROBABILITIES, [0.5, 0.3, 0.2])
        and np.array_equal(AXIS_KAPPA, [100.0, 10.0, 1.0])
        and np.array_equal(AXIS_KAPPA_PROBABILITIES, [0.5, 0.3, 0.2])
    )
    if not exact:
        raise RuntimeError("compiled SMC constants differ from the frozen capability")


def run_production_capability(program_path: Path):
    """The only public entry; fail closed until a separately frozen program exists."""
    path = Path(program_path).resolve()
    if path != CANONICAL_PROGRAM.resolve():
        raise PermissionError("production SMC accepts only the canonical program path")
    if _sha256(CAPABILITY_DESIGN) != CAPABILITY_DESIGN_SHA256:
        raise RuntimeError("production capability design hash mismatch")
    design = json.loads(CAPABILITY_DESIGN.read_text())
    if design["authorization"].get("production_execution_authorized") is not True:
        raise PermissionError("production SMC execution is not authorized")
    if not path.is_file():
        raise PermissionError("a separately frozen canonical production program is absent")
    raise PermissionError("production capability has no authorized one-shot program")


def _move_counts(replicate, field: str, width: int) -> np.ndarray:
    rows = []
    for stage in replicate.move_history:
        if len(stage) != SWEEPS_PER_STAGE:
            raise RuntimeError("replicate move schedule changed")
        for sweep in stage:
            value = sweep[field]
            if isinstance(value, dict):
                value = [value[name] for name in (
                    "q_local", "axis_local", "joint_local", "prior_independence"
                )]
            raw = np.asarray(value)
            if raw.shape != (width,) or not np.issubdtype(raw.dtype, np.integer):
                raise RuntimeError("replicate move-count width changed")
            row = raw.astype(np.int64, copy=False)
            rows.append(row)
    return np.stack(rows).reshape(len(replicate.move_history), SWEEPS_PER_STAGE, width)


def _replicate_arrays(replicate) -> dict[str, np.ndarray]:
    midpoint = np.asarray(replicate.midpoint_mpc_h)
    axis = np.asarray(replicate.axis)
    keys = np.asarray(replicate.keys)
    weights = np.asarray(replicate.weights)
    log_z_bar = np.asarray(replicate.log_z_bar)
    ancestors = np.asarray(replicate.ancestor_labels)
    beta = np.asarray(replicate.beta_history)
    cess = np.asarray(replicate.conditional_ess_history)
    ess = np.asarray(replicate.particle_ess_history)
    increments = np.asarray(replicate.log_normalizer_increment)
    stage_count = len(beta) - 1 if beta.ndim == 1 else -1
    if (
        replicate.master_seed not in PRODUCTION_REPLICATE_MASTER_SEEDS
        or midpoint.dtype != np.float64 or midpoint.shape != (2048, 3)
        or axis.dtype != np.float64 or axis.shape != (2048, 3)
        or keys.dtype != np.int16 or keys.shape != (2048, 6)
        or weights.dtype != np.float64 or weights.shape != (2048,)
        or log_z_bar.dtype != np.float64 or log_z_bar.shape != (2048,)
        or ancestors.dtype != np.int64 or ancestors.shape != (2048,)
        or beta.dtype != np.float64 or beta.ndim != 1
        or not 1 <= stage_count <= 256
        or cess.dtype != np.float64 or cess.shape != (stage_count,)
        or ess.dtype != np.float64 or ess.shape != (stage_count + 1,)
        or increments.dtype != np.float64 or increments.shape != (stage_count,)
    ):
        raise RuntimeError("terminal replicate dtype or shape contract failed")
    finite = all(np.all(np.isfinite(value)) for value in (
        midpoint, axis, weights, log_z_bar, beta, cess, ess, increments
    ))
    if (
        not finite
        or np.any(weights < 0.0)
        or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12)
        or np.any(ancestors < 0) or np.any(ancestors >= PARTICLE_COUNT)
        or beta[0] != 0.0 or beta[-1] != 1.0
        or np.any(np.diff(beta) <= 0.0)
        or np.any(beta < 0.0) or np.any(beta > 1.0)
        or np.any(cess <= 0.0) or np.any(cess > PARTICLE_COUNT * (1.0 + 1e-12))
        or np.any(ess <= 0.0) or np.any(ess > PARTICLE_COUNT * (1.0 + 1e-12))
        or not np.isclose(
            float(np.sum(increments)), float(replicate.log_normalizer),
            rtol=1e-12, atol=1e-12,
        )
    ):
        raise RuntimeError("terminal replicate normalization or history contract failed")
    norms = np.linalg.norm(axis, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=3e-15):
        raise RuntimeError("terminal axes are not unit vectors")
    canonical = np.stack([canonical_axis(value) for value in axis])
    if not np.allclose(axis, canonical, rtol=0.0, atol=3e-15):
        raise RuntimeError("terminal axes are not in canonical RP2 form")
    reconstructed = np.asarray([
        geometry_key(q, a) for q, a in zip(midpoint, axis)
    ], dtype=np.int16)
    if not np.array_equal(keys, reconstructed):
        raise RuntimeError("terminal keys do not reconstruct from midpoint and axis")
    recomputed_gess = genealogical_ess(ancestors, PARTICLE_COUNT)
    if not np.isclose(
        recomputed_gess, float(replicate.genealogical_ess),
        rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError("terminal genealogical ESS is inconsistent")
    if len(replicate.move_history) != stage_count:
        raise RuntimeError("terminal move-history stage count changed")
    resampling_rows = []
    for row in replicate.resampling_ancestors:
        value = np.asarray(row)
        if value.dtype != np.int64 or value.shape != (PARTICLE_COUNT,) \
                or np.any(value < 0) or np.any(value >= PARTICLE_COUNT):
            raise RuntimeError("resampling ancestor matrix contract failed")
        resampling_rows.append(value)
    resampling_matrix = (
        np.stack(resampling_rows)
        if resampling_rows
        else np.empty((0, PARTICLE_COUNT), dtype=np.int64)
    )
    expected_resampling_rows = int(np.count_nonzero(
        ess[1:] < RESAMPLING_ESS_FRACTION * PARTICLE_COUNT
    ))
    if len(resampling_rows) != expected_resampling_rows:
        raise RuntimeError(
            "resampling ancestor rows disagree with the strict ESS trigger"
        )
    arrays = {
        "master_seed": np.asarray(replicate.master_seed, dtype=np.int64),
        "midpoint_mpc_h": midpoint,
        "axis": axis,
        "keys": keys,
        "weights": weights,
        "log_Z_bar": log_z_bar,
        "ancestor_labels": ancestors,
        "beta_history": beta,
        "conditional_ESS_history": cess,
        "particle_ESS_history": ess,
        "log_normalizer_increment": increments,
        "log_I_bar": np.asarray(replicate.log_normalizer, dtype=np.float64),
        "genealogical_ESS": np.asarray(recomputed_gess, dtype=np.float64),
        "resampling_ancestors": resampling_matrix,
        "move_proposal_count": _move_counts(replicate, "proposal_count", 4),
        "move_acceptance_count": _move_counts(replicate, "acceptance_count", 4),
        "q_scale_proposal_count": _move_counts(
            replicate, "q_scale_proposal_count", 3
        ),
        "q_scale_acceptance_count": _move_counts(
            replicate, "q_scale_acceptance_count", 3
        ),
        "axis_scale_proposal_count": _move_counts(
            replicate, "axis_scale_proposal_count", 3
        ),
        "axis_scale_acceptance_count": _move_counts(
            replicate, "axis_scale_acceptance_count", 3
        ),
    }
    move_proposal = arrays["move_proposal_count"]
    move_acceptance = arrays["move_acceptance_count"]
    q_proposal = arrays["q_scale_proposal_count"]
    q_acceptance = arrays["q_scale_acceptance_count"]
    axis_proposal = arrays["axis_scale_proposal_count"]
    axis_acceptance = arrays["axis_scale_acceptance_count"]
    if (
        np.any(move_proposal < 0) or np.any(move_acceptance < 0)
        or np.any(q_proposal < 0) or np.any(q_acceptance < 0)
        or np.any(axis_proposal < 0) or np.any(axis_acceptance < 0)
        or np.any(move_acceptance > move_proposal)
        or np.any(q_acceptance > q_proposal)
        or np.any(axis_acceptance > axis_proposal)
        or not np.all(move_proposal.sum(axis=2) == PARTICLE_COUNT)
        or not np.array_equal(q_proposal.sum(axis=2), move_proposal[..., 0] + move_proposal[..., 2])
        or not np.array_equal(axis_proposal.sum(axis=2), move_proposal[..., 1] + move_proposal[..., 2])
        or not np.array_equal(q_acceptance.sum(axis=2), move_acceptance[..., 0] + move_acceptance[..., 2])
        or not np.array_equal(axis_acceptance.sum(axis=2), move_acceptance[..., 1] + move_acceptance[..., 2])
    ):
        raise RuntimeError("terminal move matrix totals or acceptance contract failed")
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise RuntimeError("replicate artifact contains nonfinite values")
    return arrays


@dataclass(frozen=True)
class FrozenTerminal:
    path: Path
    sha256: str
    master_seed: np.ndarray
    parent_seed: np.ndarray
    log_i_bar: np.ndarray
    p_rep: np.ndarray
    p_pool: np.ndarray
    genealogical_ess: np.ndarray
    replicate_artifact_sha256: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationData:
    parent_seed: np.ndarray
    deviance: np.ndarray
    reference_q99: float
    reference_q99p5: float
    source_path: Path
    source_sha256: str


def _load_pinned_calibration() -> CalibrationData:
    if _sha256(CALIBRATION_PATH) != CALIBRATION_SHA256:
        raise RuntimeError("canonical CF4 calibration hash mismatch")
    value = json.loads(CALIBRATION_PATH.read_text())
    if value.get("schema") != CALIBRATION_SCHEMA \
            or value.get("status") != CALIBRATION_STATUS \
            or value.get("reference_seed_count") != 256 \
            or value.get("calibration_seed_count") != 255 \
            or value.get("excluded_parent_seed") != 3429:
        raise RuntimeError("canonical CF4 calibration status contract failed")
    rows = value.get("rows", [])
    seeds = np.asarray([row.get("seed") for row in rows], dtype=np.int32)
    deviance = np.asarray(
        [row.get("marginal_deviance") for row in rows], dtype=np.float64
    )
    if seeds.shape != (256,) or not np.array_equal(
        seeds, np.asarray(PRODUCTION_PARENT_SEEDS, dtype=np.int32)
    ) or deviance.shape != (256,) or not np.all(np.isfinite(deviance)):
        raise RuntimeError("canonical CF4 calibration seed join failed")
    reference = deviance[seeds != 3429]
    thresholds = value.get("L3_reference_thresholds", {})
    q99 = float(thresholds.get("deviance_Q99", np.nan))
    q99p5 = float(thresholds.get("deviance_Q99p5", np.nan))
    if thresholds.get("quantile_method") != "linear" \
            or not np.isclose(
                q99, np.quantile(reference, 0.99, method="linear"),
                rtol=0.0, atol=1e-12,
            ) \
            or not np.isclose(
                q99p5, np.quantile(reference, 0.995, method="linear"),
                rtol=0.0, atol=1e-12,
            ):
        raise RuntimeError("canonical CF4 calibration thresholds changed")
    seeds.flags.writeable = False
    deviance.flags.writeable = False
    return CalibrationData(
        seeds, deviance, q99, q99p5,
        CALIBRATION_PATH.resolve(), CALIBRATION_SHA256,
    )


class TwoPhaseArtifactFirewall:
    """Prevent any calibration access before immutable terminal publication."""

    def __init__(self, output_directory: Path):
        self._lineage_validated = bool(
            _sha256(CAPABILITY_DESIGN) == CAPABILITY_DESIGN_SHA256
        )
        if not self._lineage_validated:
            raise RuntimeError("capability design lineage changed")
        _validate_compiled_constants()
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=False, exist_ok=False)
        self._replicates: list[Any] = []
        self._replicate_arrays: list[dict[str, np.ndarray]] = []
        self._replicate_paths: list[Path] = []
        self._replicate_hashes: list[str] = []
        self._frozen: FrozenTerminal | None = None
        self._calibration: CalibrationData | None = None
        self._calibration_identity: int | None = None
        self._calibration_opened = False
        self._cf4_published = False

    def register_terminal_histories(self, replicates, oracle) -> None:
        if self._frozen is not None or self._calibration_opened:
            raise RuntimeError("terminal phase is already sealed")
        if self._replicates:
            raise RuntimeError("terminal histories may be registered only once")
        values = tuple(replicates)
        if tuple(row.master_seed for row in values) != (
            PRODUCTION_REPLICATE_MASTER_SEEDS
        ):
            raise RuntimeError("all four completed replicates must be in frozen order")
        arrays = [_replicate_arrays(row) for row in values]
        for replicate in values:
            oracle.register_terminal_history(replicate.master_seed, replicate.keys)
        self._replicates.extend(values)
        self._replicate_arrays.extend(arrays)

    def seal_terminal_phase(self, oracle) -> FrozenTerminal:
        if self._frozen is not None:
            raise RuntimeError("terminal phase is already sealed")
        if tuple(row.master_seed for row in self._replicates) != (
            PRODUCTION_REPLICATE_MASTER_SEEDS
        ):
            raise RuntimeError("all four terminal histories are required before sealing")
        oracle.seal_terminal_histories()
        parent_vectors = []
        for replicate in self._replicates:
            parent_log_z = oracle.terminal_parent_log_z(
                replicate.master_seed, replicate.keys
            )
            if parent_log_z.shape != (2048, 256) \
                    or not np.all(np.isfinite(parent_log_z)):
                raise RuntimeError("terminal parent evidence contract failed")
            reconstructed_log_z_bar = logmeanexp_parent(parent_log_z)
            if np.max(np.abs(
                reconstructed_log_z_bar - replicate.log_z_bar
            )) > 1e-12:
                raise RuntimeError(
                    "terminal parent aggregate evidence is inconsistent"
                )
            parent_vectors.append(replicate_parent_probability(
                replicate.weights, parent_log_z
            ))
        log_i_bar = np.asarray(
            [row.log_normalizer for row in self._replicates], dtype=np.float64
        )
        p_rep = np.stack(parent_vectors).astype(np.float64)
        p_pool, _ = pool_parent_probabilities(log_i_bar, p_rep)
        for index, arrays in enumerate(self._replicate_arrays):
            replicate_path = self.output_directory / f"replicate_{index}.npz"
            _atomic_npz(replicate_path, arrays)
            self._replicate_paths.append(replicate_path)
            self._replicate_hashes.append(_sha256(replicate_path))
        if len(self._replicate_hashes) != 4 or any(
            _sha256(replicate_path) != digest
            for replicate_path, digest in zip(
                self._replicate_paths, self._replicate_hashes
            )
        ):
            raise RuntimeError("replicate artifact publication verification failed")
        path = self.output_directory / "terminal_parent_frozen.npz"
        arrays = {
            "master_seed": np.asarray(
                PRODUCTION_REPLICATE_MASTER_SEEDS, dtype=np.int64
            ),
            "parent_seed": np.asarray(PRODUCTION_PARENT_SEEDS, dtype=np.int32),
            "log_I_bar": log_i_bar,
            "P_rep": p_rep,
            "P_pool": np.asarray(p_pool, dtype=np.float64),
        }
        _atomic_npz(path, arrays)
        with np.load(path, allow_pickle=False) as item:
            if tuple(item.files) != TERMINAL_KEYS:
                raise RuntimeError("terminal parent file contains a forbidden field")
            for key, expected in arrays.items():
                if not np.array_equal(item[key], expected):
                    raise RuntimeError("terminal parent file failed atomic verification")
        frozen = FrozenTerminal(
            path=path,
            sha256=_sha256(path),
            master_seed=arrays["master_seed"],
            parent_seed=arrays["parent_seed"],
            log_i_bar=log_i_bar,
            p_rep=p_rep,
            p_pool=arrays["P_pool"],
            genealogical_ess=np.asarray(
                [row.genealogical_ess for row in self._replicates],
                dtype=np.float64,
            ),
            replicate_artifact_sha256=tuple(self._replicate_hashes),
        )
        self._frozen = frozen
        return frozen

    def _reload_phase_one_artifacts(
        self,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, bool]]:
        if self._frozen is None \
                or _sha256(self._frozen.path) != self._frozen.sha256:
            raise RuntimeError("sealed terminal parent artifact changed")
        if len(self._replicate_paths) != 4 or any(
            _sha256(path) != digest for path, digest in zip(
                self._replicate_paths, self._replicate_hashes
            )
        ):
            raise RuntimeError("sealed replicate artifact changed")
        with np.load(self._frozen.path, allow_pickle=False) as item:
            if tuple(item.files) != TERMINAL_KEYS:
                raise RuntimeError("sealed terminal parent key set changed")
            terminal = {key: item[key].copy() for key in TERMINAL_KEYS}
        terminal_contract = {
            "master_seed": (np.dtype("int64"), (4,)),
            "parent_seed": (np.dtype("int32"), (256,)),
            "log_I_bar": (np.dtype("float64"), (4,)),
            "P_rep": (np.dtype("float64"), (4, 256)),
            "P_pool": (np.dtype("float64"), (256,)),
        }
        for key, (dtype, shape) in terminal_contract.items():
            if terminal[key].dtype != dtype or terminal[key].shape != shape \
                    or not np.all(np.isfinite(terminal[key])):
                raise RuntimeError("sealed terminal parent array contract changed")
        if not np.array_equal(
            terminal["master_seed"], PRODUCTION_REPLICATE_MASTER_SEEDS
        ) or not np.array_equal(
            terminal["parent_seed"], PRODUCTION_PARENT_SEEDS
        ) or not np.allclose(
            terminal["P_rep"].sum(axis=1), 1.0, rtol=0.0, atol=1e-12
        ) or not np.isclose(
            terminal["P_pool"].sum(), 1.0, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("sealed terminal parent normalization changed")
        genealogy = []
        replicate_log_i = []
        reached_one = []
        within_stage_limit = []
        stagnation_absent = []
        for expected_seed, path in zip(
            PRODUCTION_REPLICATE_MASTER_SEEDS, self._replicate_paths
        ):
            with np.load(path, allow_pickle=False) as item:
                required = {
                    "master_seed", "ancestor_labels", "genealogical_ESS",
                    "beta_history", "log_normalizer_increment", "log_I_bar",
                    "weights", "keys", "midpoint_mpc_h", "axis",
                }
                if not required.issubset(item.files):
                    raise RuntimeError("sealed replicate artifact lost a required array")
                seed = int(item["master_seed"])
                ancestors = item["ancestor_labels"]
                stored_gess = float(item["genealogical_ESS"])
                beta = item["beta_history"]
                increments = item["log_normalizer_increment"]
                log_i = float(item["log_I_bar"])
                if seed != expected_seed \
                        or ancestors.dtype != np.int64 \
                        or ancestors.shape != (PARTICLE_COUNT,) \
                        or np.any(ancestors < 0) \
                        or np.any(ancestors >= PARTICLE_COUNT) \
                        or beta.dtype != np.float64 or beta.ndim != 1 \
                        or increments.dtype != np.float64 \
                        or increments.shape != (len(beta) - 1,) \
                        or not np.all(np.isfinite(beta)) \
                        or not np.all(np.isfinite(increments)) \
                        or not np.isclose(
                            increments.sum(), log_i, rtol=1e-12, atol=1e-12
                        ):
                    raise RuntimeError("sealed replicate history contract changed")
                actual_gess = genealogical_ess(ancestors, PARTICLE_COUNT)
                if not np.isclose(
                    actual_gess, stored_gess, rtol=0.0, atol=1e-12
                ):
                    raise RuntimeError("sealed replicate GESS changed")
                differences = np.diff(beta)
                genealogy.append(actual_gess)
                replicate_log_i.append(log_i)
                reached_one.append(bool(beta[0] == 0.0 and beta[-1] == 1.0))
                within_stage_limit.append(bool(1 <= len(differences) <= 256))
                stagnation_absent.append(bool(
                    np.all(differences > 0.0)
                    and not np.any(
                        ((1.0 - beta[:-1]) > 1e-10)
                        & (differences <= 1e-12)
                    )
                ))
        if not np.array_equal(
            np.asarray(replicate_log_i), terminal["log_I_bar"]
        ):
            raise RuntimeError("replicate and terminal log-I arrays disagree")
        return terminal, np.asarray(genealogy, dtype=np.float64), {
            "all_replicates_reach_beta_one": bool(all(reached_one)),
            "temperature_stagnation_absent": bool(all(stagnation_absent)),
            "maximum_temperature_stages": bool(all(within_stage_limit)),
        }

    def open_calibration(self) -> CalibrationData:
        if self._frozen is None:
            raise PermissionError("CF4 calibration is closed before terminal seal")
        if self._calibration_opened:
            raise RuntimeError("CF4 calibration may be opened only once")
        if _sha256(self._frozen.path) != self._frozen.sha256 or any(
            _sha256(path) != digest for path, digest in zip(
                self._replicate_paths, self._replicate_hashes
            )
        ):
            raise RuntimeError("phase-one artifact changed before CF4 opening")
        calibration = _load_pinned_calibration()
        self._calibration = calibration
        self._calibration_identity = id(calibration)
        self._calibration_opened = True
        return calibration

    def publish_cf4_gates(self) -> dict[str, Any]:
        if self._frozen is None or not self._calibration_opened:
            raise PermissionError("post-terminal CF4 publication is not open")
        if self._cf4_published:
            raise RuntimeError("post-terminal CF4 gates are already published")
        if self._calibration is None \
                or id(self._calibration) != self._calibration_identity:
            raise RuntimeError("bound CF4 calibration object was substituted")
        calibration = self._calibration
        if calibration.source_path != CALIBRATION_PATH.resolve() \
                or calibration.source_sha256 != CALIBRATION_SHA256 \
                or _sha256(CALIBRATION_PATH) != CALIBRATION_SHA256:
            raise RuntimeError("bound CF4 calibration lineage changed")
        reloaded_calibration = _load_pinned_calibration()
        if calibration.parent_seed.flags.writeable \
                or calibration.deviance.flags.writeable \
                or not np.array_equal(
                    calibration.parent_seed, reloaded_calibration.parent_seed
                ) \
                or not np.array_equal(
                    calibration.deviance, reloaded_calibration.deviance
                ) \
                or calibration.reference_q99 != reloaded_calibration.reference_q99 \
                or calibration.reference_q99p5 != reloaded_calibration.reference_q99p5 \
                or calibration.source_path != reloaded_calibration.source_path \
                or calibration.source_sha256 != reloaded_calibration.source_sha256:
            raise RuntimeError("bound CF4 calibration content changed")
        terminal, genealogy, temperature = self._reload_phase_one_artifacts()
        pre = evaluate_pre_cf4_gates(
            terminal["log_I_bar"],
            terminal["P_rep"],
            terminal["P_pool"],
            genealogy,
        )
        cf4 = evaluate_cf4_gates(
            calibration.deviance,
            terminal["P_pool"],
            calibration.reference_q99,
            calibration.reference_q99p5,
        )
        gates = {
            "lineage_and_authorization": bool(
                self._lineage_validated
                and calibration.source_sha256 == CALIBRATION_SHA256
            ),
            "finite_and_artifact_contract": bool(
                all(np.all(np.isfinite(value)) for value in terminal.values())
                and np.all(np.isfinite(genealogy))
            ),
            "terminal_phase_complete_and_sealed": bool(
                self._frozen is not None
                and len(self._replicate_paths) == 4
                and self._calibration_opened
            ),
            **temperature,
            **pre["gates"],
            **cf4["gates"],
        }
        failure = classify_failure(gates)
        lifecycle_status = classify_lifecycle_status(gates, failure)
        npz_path = self.output_directory / "post_terminal_cf4_gates.npz"
        _atomic_npz(npz_path, {
            "parent_seed": terminal["parent_seed"],
            "deviance": np.asarray(calibration.deviance, dtype=np.float64),
            "P_pool": terminal["P_pool"],
        })
        json_path = self.output_directory / "post_terminal_cf4_gates.json"
        summary = {
            "schema": "ouruniv-cf4-aggregate-evidence-post-terminal-cf4-gates-v1",
            "status": lifecycle_status,
            "terminal_parent_frozen": str(self._frozen.path.resolve()),
            "terminal_parent_frozen_sha256": self._frozen.sha256,
            "post_terminal_arrays": str(npz_path.resolve()),
            "post_terminal_arrays_sha256": _sha256(npz_path),
            "pre_CF4_metrics": pre["metrics"],
            "CF4_metrics": cf4["metrics"],
            "gates": gates,
            "failure_class": failure,
            "decision": {
                "production_SMC_execution_authorized": False,
                "conditional_field_bank_authorized": False,
                "parent_or_seed_selection_authorized": False,
                "PM_authorized": False,
                "HOP_authorized": False,
                "RAMSES_authorized": False,
                "downstream_execution_authorized": False,
                "automatic_follow_on": False,
            },
        }
        _atomic_json(json_path, summary)
        self._cf4_published = True
        return summary


def evaluate_pre_cf4_gates(
    log_i_bar: np.ndarray,
    p_rep: np.ndarray,
    p_pool: np.ndarray,
    genealogical_ess: np.ndarray,
) -> dict[str, Any]:
    log_i = np.asarray(log_i_bar, dtype=np.float64)
    replicate = np.asarray(p_rep, dtype=np.float64)
    pooled = np.asarray(p_pool, dtype=np.float64)
    genealogy = np.asarray(genealogical_ess, dtype=np.float64)
    if (
        log_i.shape != (4,)
        or replicate.shape != (4, 256)
        or pooled.shape != (256,)
        or genealogy.shape != (4,)
        or not all(np.all(np.isfinite(value)) for value in (
            log_i, replicate, pooled, genealogy
        ))
        or np.any(replicate < 0.0)
        or np.any(pooled < 0.0)
        or not np.allclose(replicate.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        or not np.isclose(pooled.sum(), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("pre-CF4 terminal gate inputs broke their contract")
    log_range = float(np.max(log_i) - np.min(log_i))
    log_se = float(np.std(log_i, ddof=1) / math.sqrt(4.0))
    pair_l1 = np.asarray([
        np.sum(np.abs(replicate[left] - replicate[right]))
        for left, right in itertools.combinations(range(4), 2)
    ])
    pooled_ess = float(1.0 / np.sum(pooled**2))
    maximum = float(np.max(pooled))
    return {
        "metrics": {
            "replicate_log_I_bar_range": log_range,
            "replicate_log_I_bar_sample_SE": log_se,
            "six_pairwise_P_rep_L1": pair_l1.tolist(),
            "maximum_pairwise_P_rep_L1": float(np.max(pair_l1)),
            "genealogical_ESS": genealogy.tolist(),
            "pooled_parent_ESS": pooled_ess,
            "maximum_P_pool": maximum,
        },
        "gates": {
            "replicate_log_I_bar_range": log_range <= 0.2,
            "replicate_log_I_bar_sample_SE": log_se <= 0.1,
            "replicate_parent_probability_L1": bool(np.all(pair_l1 <= 0.2)),
            "genealogical_ESS": bool(np.all(genealogy >= 128.0)),
            "pooled_parent_ESS": pooled_ess >= 32.0,
            "maximum_pooled_parent_probability": maximum <= 0.1,
        },
    }


def _stable_weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    cumulative[-1] = 1.0
    index = int(np.searchsorted(cumulative, probability, side="left"))
    return float(values[order[index]])


def _one_sided_tied_ks(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    ends = np.flatnonzero(np.r_[ordered_values[1:] != ordered_values[:-1], True])
    uniform = (ends + 1.0) / len(values)
    weighted = np.cumsum(ordered_weights, dtype=np.float64)[ends]
    return float(max(0.0, np.max(uniform - weighted)))


def _permutation_pvalue(values: np.ndarray, weights: np.ndarray) -> tuple[float, int]:
    observed = _one_sided_tied_ks(values, weights)
    rng = np.random.Generator(np.random.PCG64DXSM(PERMUTATION_SEED))
    exceedance = 0
    for _ in range(PERMUTATIONS):
        permuted = rng.permutation(weights)
        exceedance += _one_sided_tied_ks(values, permuted) >= observed
    return (exceedance + 1.0) / (PERMUTATIONS + 1.0), exceedance


def evaluate_cf4_gates(
    deviance: np.ndarray,
    p_pool: np.ndarray,
    reference_q99: float,
    reference_q99p5: float,
) -> dict[str, Any]:
    values = np.asarray(deviance, dtype=np.float64)
    weights = np.asarray(p_pool, dtype=np.float64)
    if (
        values.shape != (256,)
        or weights.shape != (256,)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12)
        or not np.all(np.isfinite([reference_q99, reference_q99p5]))
    ):
        raise ValueError("CF4 gate inputs broke their fixed contract")
    q99_mass = float(np.sum(weights[values > float(reference_q99)]))
    q90 = _stable_weighted_quantile(values, weights, 0.9)
    statistic = _one_sided_tied_ks(values, weights)
    pvalue, exceedance = _permutation_pvalue(values, weights)
    return {
        "metrics": {
            "weighted_CF4_Q99_exceedance_mass": q99_mass,
            "weighted_CF4_Q90": q90,
            "reference_CF4_Q99": float(reference_q99),
            "reference_CF4_Q99p5": float(reference_q99p5),
            "weighted_CF4_one_sided_KS": statistic,
            "permutation_seed": PERMUTATION_SEED,
            "permutations": PERMUTATIONS,
            "permutation_exceedance_count": exceedance,
            "permutation_pvalue": pvalue,
        },
        "gates": {
            "weighted_CF4_Q99_exceedance_mass": q99_mass <= 0.05,
            "weighted_CF4_Q90": q90 <= float(reference_q99p5),
            "weighted_CF4_one_sided_KS_permutation": pvalue >= 0.01,
        },
    }


GATE_FAILURE_PRIORITY = (
    ("lineage_and_authorization", "invalid_lineage_or_authorization"),
    ("finite_and_artifact_contract", "nonfinite_or_artifact_contract"),
    ("terminal_phase_complete_and_sealed", "incomplete_or_unsealed_terminal_phase"),
    ("all_replicates_reach_beta_one", "SMC_terminal_beta_not_one"),
    ("temperature_stagnation_absent", "SMC_temperature_stagnation"),
    ("maximum_temperature_stages", "SMC_maximum_temperature_stages"),
    ("replicate_log_I_bar_range", "replicate_log_I_bar_range"),
    ("replicate_log_I_bar_sample_SE", "replicate_log_I_bar_sample_SE"),
    ("replicate_parent_probability_L1", "replicate_parent_probability_L1"),
    ("genealogical_ESS", "genealogical_ESS"),
    ("pooled_parent_ESS", "pooled_parent_ESS"),
    ("maximum_pooled_parent_probability", "maximum_pooled_parent_probability"),
    ("weighted_CF4_Q99_exceedance_mass", "weighted_CF4_Q99_exceedance_mass"),
    ("weighted_CF4_Q90", "weighted_CF4_Q90"),
    ("weighted_CF4_one_sided_KS_permutation", "weighted_CF4_one_sided_KS_permutation"),
)


def classify_failure(gates: dict[str, bool]) -> str | None:
    known = {gate for gate, _ in GATE_FAILURE_PRIORITY}
    unknown = set(gates).difference(known)
    if unknown:
        raise ValueError(f"unknown frozen gate: {sorted(unknown)}")
    for gate, failure in GATE_FAILURE_PRIORITY:
        if gates.get(gate) is not True:
            return failure
    return None


def classify_lifecycle_status(
    gates: dict[str, bool], failure: str | None = None
) -> str:
    validity = (
        "lineage_and_authorization",
        "finite_and_artifact_contract",
        "terminal_phase_complete_and_sealed",
    )
    if not all(gates.get(name) is True for name in validity):
        return "invalid_failed"
    actual_failure = classify_failure(gates) if failure is None else failure
    return "complete_pass" if actual_failure is None else "complete_scientific_fail"


def classify_smc_runtime_failure(error: BaseException) -> tuple[str, str]:
    message = str(error)
    if message == "temperature schedule stagnated":
        return "complete_scientific_fail", "SMC_temperature_stagnation"
    if message == "SMC replicate did not reach beta=1":
        return "complete_scientific_fail", "SMC_maximum_temperature_stages"
    return "invalid_failed", "invalid_runtime_or_implementation_failure"


class OracleEvaluationFailure(RuntimeError):
    """Typed invalid failure at the SMC/oracle trust boundary."""


class _InvalidatingOracleBoundary:
    def __init__(self, oracle):
        self._oracle = oracle

    def _call(self, method: str, *args):
        try:
            return getattr(self._oracle, method)(*args)
        except RuntimeError as error:
            raise OracleEvaluationFailure(
                f"oracle {method} failed: {error}"
            ) from error

    def evaluate(self, midpoint_mpc_h, axis):
        return self._call("evaluate", midpoint_mpc_h, axis)

    def register_terminal_history(self, master_seed, keys):
        return self._call("register_terminal_history", master_seed, keys)

    def seal_terminal_histories(self):
        return self._call("seal_terminal_histories")

    def terminal_parent_log_z(self, master_seed, keys):
        return self._call("terminal_parent_log_z", master_seed, keys)


def _run_fixed_capability_core(oracle, output_directory: Path):
    """Private fixed core for future program wiring and synthetic tests only."""
    _validate_compiled_constants()
    firewall = TwoPhaseArtifactFirewall(output_directory)
    bounded_oracle = _InvalidatingOracleBoundary(oracle)
    try:
        replicates = tuple(
            run_smc_replicate(master_seed, bounded_oracle)
            for master_seed in PRODUCTION_REPLICATE_MASTER_SEEDS
        )
    except OracleEvaluationFailure:
        raise
    except RuntimeError as error:
        status, failure = classify_smc_runtime_failure(error)
        if status == "invalid_failed":
            raise
        summary = {
            "schema": "ouruniv-cf4-aggregate-evidence-smc-capability-result-v1",
            "status": status,
            "failure_class": failure,
            "valid_scientific_architecture_stop": True,
            "CF4_calibration_opened": False,
            "automatic_retry_retune_or_scale_up_authorized": False,
            "decision": {
                "production_SMC_execution_authorized": False,
                "conditional_field_bank_authorized": False,
                "parent_or_seed_selection_authorized": False,
                "PM_authorized": False,
                "HOP_authorized": False,
                "RAMSES_authorized": False,
                "downstream_execution_authorized": False,
                "automatic_follow_on": False,
            },
        }
        _atomic_json(firewall.output_directory / "capability_result.json", summary)
        return summary
    firewall.register_terminal_histories(replicates, bounded_oracle)
    firewall.seal_terminal_phase(bounded_oracle)
    firewall.open_calibration()
    return firewall.publish_cf4_gates()
