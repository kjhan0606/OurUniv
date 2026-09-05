#!/usr/bin/env python
"""Shared frozen contract helpers for the V83 marginal spline experiment."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from hong2021_v18_init import sha256_file


SCHEMA = "hong2021-v83-conditional-marginal-spline-flow-program-v1"
STATUS = "frozen_before_preflight_training_or_payload_evaluation"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
V35_PROGRAM = Path("config/hong2021_v35_residual_spectrum_phase_program.json")
V35_PROGRAM_SHA256 = "161b5b9c7345c6777e39ebc342243ee12226b75d8e461c0db69f410ea2193e4a"
PARTITION_SEED = 830083
HOLDOUT_FRACTION = 0.1


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def partition_indices(
    object_counts: dict[str, int],
) -> dict[str, dict[str, list[int]]]:
    result: dict[str, dict[str, list[int]]] = {}
    for offset, domain in enumerate(DOMAIN_ORDER):
        count = int(object_counts[domain])
        if count < 10:
            raise ValueError(f"V83 {domain} object count is too small")
        generator = np.random.default_rng(PARTITION_SEED + offset)
        permutation = generator.permutation(count)
        holdout_count = int(np.ceil(HOLDOUT_FRACTION * count))
        holdout = sorted(map(int, permutation[:holdout_count]))
        training = sorted(map(int, permutation[holdout_count:]))
        if set(holdout) & set(training) or sorted(holdout + training) != list(range(count)):
            raise RuntimeError(f"V83 {domain} partition is not exhaustive")
        result[domain] = {"fit": training, "holdout": holdout}
    return result


def partition_digest(partition: dict[str, dict[str, list[int]]]) -> str:
    payload = json.dumps(partition, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_program(
    program_path: Path,
    repo: Path,
    commit: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, list[int]]]]:
    repo = repo.resolve()
    program_path = program_path.resolve()
    program = strict_json(program_path)
    if program.get("schema") != SCHEMA or program.get("status") != STATUS:
        raise ValueError("V83 program schema or status differs")
    if sha256_file(repo / V35_PROGRAM) != V35_PROGRAM_SHA256:
        raise ValueError("V83 V35 data definition differs")
    v35 = strict_json(repo / V35_PROGRAM)
    if tuple(v35.get("development_domains", {})) != DOMAIN_ORDER:
        raise ValueError("V83 V35 domain order differs")
    frozen = program["frozen_inputs"]
    if (
        frozen.get("v35_program") != str(V35_PROGRAM)
        or frozen.get("v35_program_sha256") != V35_PROGRAM_SHA256
    ):
        raise ValueError("V83 V35 binding differs")
    for label, row in program["implementation"].items():
        path = repo / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"V83 implementation differs: {label}")
    freeze_commit = program.get("freeze_commit")
    if not isinstance(freeze_commit, str) or not _is_ancestor(repo, freeze_commit, commit):
        raise ValueError("V83 code does not descend from its freeze commit")
    if (
        program.get("firewall", {}).get("validation_payload_access") != "forbidden"
        or program.get("firewall", {}).get("Astrid_access") != "forbidden"
        or program.get("firewall", {}).get("historical_EAGLE_access") != "forbidden"
        or program.get("firewall", {}).get("independent_gate_locked") is not True
    ):
        raise ValueError("V83 firewall differs")
    counts = {
        domain: int(v35["development_domains"][domain]["train_objects"])
        for domain in DOMAIN_ORDER
    }
    partition = partition_indices(counts)
    if (
        program["partition"].get("seed") != PARTITION_SEED
        or program["partition"].get("holdout_fraction") != HOLDOUT_FRACTION
        or program["partition"].get("sha256") != partition_digest(partition)
        or program["partition"].get("counts")
        != {
            domain: {
                "fit": len(partition[domain]["fit"]),
                "holdout": len(partition[domain]["holdout"]),
            }
            for domain in DOMAIN_ORDER
        }
    ):
        raise ValueError("V83 fit/holdout partition differs")
    return program, v35, partition


def validate_train_artifacts(program: dict[str, Any], v35: dict[str, Any]) -> None:
    """Validate train inputs only; deliberately never touch validation paths."""
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        frozen = program["frozen_inputs"]["training_domains"][domain]
        for kind in ("data", "cache"):
            path_key = f"train_{kind}"
            digest_key = f"train_{kind}_sha256"
            path = Path(row[path_key]).resolve()
            if (
                str(path) != frozen[path_key]
                or row[digest_key] != frozen[digest_key]
                or sha256_file(path) != frozen[digest_key]
            ):
                raise ValueError(f"V83 {domain} train {kind} differs")


__all__ = [
    "DOMAIN_ORDER",
    "HOLDOUT_FRACTION",
    "PARTITION_SEED",
    "SCHEMA",
    "STATUS",
    "V35_PROGRAM",
    "V35_PROGRAM_SHA256",
    "load_program",
    "partition_digest",
    "partition_indices",
    "strict_json",
    "validate_train_artifacts",
]
