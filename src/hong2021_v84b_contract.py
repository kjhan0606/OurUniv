#!/usr/bin/env python
"""Shared prospective contract helpers for the V84B group-held-out model."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_v18_init import sha256_file


SCHEMA = "hong2021-v84b-group-held-out-spliced-tail-program-v1"
STATUS = "frozen_before_preflight_training_or_group_holdout_evaluation"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
V35_PROGRAM = Path("config/hong2021_v35_residual_spectrum_phase_program.json")
V35_PROGRAM_SHA256 = "161b5b9c7345c6777e39ebc342243ee12226b75d8e461c0db69f410ea2193e4a"
TNG_PARTITION_SEED = 840184
TNG_HOLDOUT_OBJECTS = 44
TNG_EMBARGO_MPC_H = 10.0
GROUP_SEEDS = {"SIMBA": 840185, "Swift": 840186}
GROUP_HOLDOUT_COUNTS = {"SIMBA": 2, "Swift": 3}


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo,
        check=False,
        capture_output=True,
    ).returncode == 0


def _periodic_distance(query: np.ndarray, reference: np.ndarray, box: float) -> np.ndarray:
    delta = np.abs(query[:, None] - reference[None])
    delta = np.minimum(delta, box - delta)
    return np.sqrt(np.square(delta).sum(axis=-1))


def group_partition(v35: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    tng_row = v35["development_domains"]["TNG100"]
    with h5py.File(tng_row["train_data"], "r") as handle:
        positions = np.asarray(handle["center_position_mpc_h"], dtype=np.float64)
    anchor = np.random.default_rng(TNG_PARTITION_SEED).uniform(0.0, 75.0, 3)
    anchor_distance = _periodic_distance(positions, anchor[None], 75.0)[:, 0]
    holdout = np.argsort(anchor_distance, kind="stable")[:TNG_HOLDOUT_OBJECTS]
    nearest_holdout = _periodic_distance(positions, positions[holdout], 75.0).min(axis=1)
    fit = np.flatnonzero(nearest_holdout >= TNG_EMBARGO_MPC_H)
    embargo = np.setdiff1d(
        np.arange(len(positions), dtype=np.int64), np.union1d(holdout, fit)
    )
    result["TNG100"] = {
        "fit": sorted(map(int, fit)),
        "holdout": sorted(map(int, holdout)),
        "embargo": sorted(map(int, embargo)),
        "anchor_mpc_h": anchor.tolist(),
        "minimum_holdout_fit_center_distance_mpc_h": float(
            _periodic_distance(positions[holdout], positions[fit], 75.0).min()
        ),
    }
    for domain in ("SIMBA", "Swift"):
        row = v35["development_domains"][domain]
        with h5py.File(row["train_data"], "r") as handle:
            groups = np.asarray(handle["realization"], dtype=np.int64)
        unique = np.unique(groups)
        permutation = np.random.default_rng(GROUP_SEEDS[domain]).permutation(unique)
        holdout_groups = np.sort(permutation[: GROUP_HOLDOUT_COUNTS[domain]])
        holdout = np.flatnonzero(np.isin(groups, holdout_groups))
        fit = np.flatnonzero(~np.isin(groups, holdout_groups))
        result[domain] = {
            "fit": sorted(map(int, fit)),
            "holdout": sorted(map(int, holdout)),
            "embargo": [],
            "fit_groups": sorted(map(int, np.unique(groups[fit]))),
            "holdout_groups": sorted(map(int, holdout_groups)),
        }
    return result


def partition_digest(partition: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(partition, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_program(
    program_path: Path,
    repo: Path,
    commit: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    repo = repo.resolve()
    program = strict_json(program_path.resolve())
    if program.get("schema") != SCHEMA or program.get("status") != STATUS:
        raise ValueError("V84B program schema or status differs")
    if sha256_file(repo / V35_PROGRAM) != V35_PROGRAM_SHA256:
        raise ValueError("V84B V35 definition differs")
    v35 = strict_json(repo / V35_PROGRAM)
    if tuple(v35.get("development_domains", {})) != DOMAIN_ORDER:
        raise ValueError("V84B domain order differs")
    for label, row in program["implementation"].items():
        if sha256_file(repo / row["path"]) != row["sha256"]:
            raise ValueError(f"V84B implementation differs: {label}")
    freeze_commit = program.get("freeze_commit")
    if not isinstance(freeze_commit, str) or not _is_ancestor(repo, freeze_commit, commit):
        raise ValueError("V84B code does not descend from its freeze commit")
    firewall = program.get("firewall", {})
    if (
        firewall.get("validation_payload_access") != "forbidden"
        or firewall.get("consumed_development_payload_access") != "forbidden"
        or firewall.get("Astrid_access") != "forbidden"
        or firewall.get("historical_EAGLE_access") != "forbidden"
        or firewall.get("independent_gate_locked") is not True
    ):
        raise ValueError("V84B firewall differs")
    partition = group_partition(v35)
    expected_counts = {
        domain: {
            key: len(partition[domain][key]) for key in ("fit", "holdout", "embargo")
        }
        for domain in DOMAIN_ORDER
    }
    if (
        program["partition"].get("sha256") != partition_digest(partition)
        or program["partition"].get("counts") != expected_counts
    ):
        raise ValueError("V84B group partition differs")
    return program, v35, partition


def validate_train_artifacts(program: dict[str, Any], v35: dict[str, Any]) -> None:
    """Validate immutable train inputs only; never touch validation paths."""
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        frozen = program["frozen_inputs"]["training_domains"][domain]
        for kind in ("data", "cache"):
            key = f"train_{kind}"
            digest = f"train_{kind}_sha256"
            path = Path(row[key]).resolve()
            if (
                str(path) != frozen[key]
                or row[digest] != frozen[digest]
                or sha256_file(path) != frozen[digest]
            ):
                raise ValueError(f"V84B {domain} train {kind} differs")


__all__ = [
    "DOMAIN_ORDER",
    "GROUP_HOLDOUT_COUNTS",
    "GROUP_SEEDS",
    "SCHEMA",
    "STATUS",
    "TNG_EMBARGO_MPC_H",
    "TNG_HOLDOUT_OBJECTS",
    "TNG_PARTITION_SEED",
    "V35_PROGRAM",
    "V35_PROGRAM_SHA256",
    "group_partition",
    "load_program",
    "partition_digest",
    "strict_json",
    "validate_train_artifacts",
]
