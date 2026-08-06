#!/usr/bin/env python
"""Train the frozen V22 long-horizon replication of V21."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hong2021_v14_edm import V22_E10_SCHEMA, train
from hong2021_v15_development_gate import canonical_digest
from hong2021_v18_init import sha256_file
from hong2021_v21_edm import (
    ARTIFACT_SHA256, load_frozen_program as load_v21_program,
    frozen_training_namespace as v21_training_namespace,
)


REGISTRY_SCHEMA = "hong2021-v22-long-horizon-development-program-v1"
REGISTRY_SHA256 = "2f2d5337ceecab413e647f54bcaa75e1502d76db3b24daafd22d1c1a2bd7cfbe"


def _resolve(value: str, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_frozen_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(path) != REGISTRY_SHA256:
        raise ValueError("V22 registry differs from its frozen hash")
    registry = json.loads(path.read_text())
    if registry.get("schema") != REGISTRY_SCHEMA or registry.get("status") != "frozen_before_implementation_or_execution":
        raise ValueError("V22 registry schema or status mismatch")
    parent = registry["parent_evidence"]
    v21_registry = _resolve(parent["v21_registry"], repo)
    v21_artifacts = _resolve(parent["v21_artifacts"], repo)
    if sha256_file(v21_registry) != parent["v21_registry_sha256"] or sha256_file(v21_artifacts) != parent["v21_artifacts_sha256"]:
        raise ValueError("V22 V21 parent hash mismatch")
    decision_path = Path(parent["v21_decision"])
    if sha256_file(decision_path) != parent["v21_decision_sha256"]:
        raise ValueError("V22 V21 decision file hash mismatch")
    decision = json.loads(decision_path.read_text())
    if canonical_digest(decision) != parent["v21_decision_digest_sha256"] or decision.get("development_pass") is not False:
        raise ValueError("V22 parent decision digest or failure state mismatch")
    if decision.get("Astrid_used") is not False or decision.get("EAGLE_RefL0100N1504_used") is not False:
        raise ValueError("V22 parent decision violated the independent-data firewall")
    audit_path = _resolve(parent["failure_audit"], repo)
    if sha256_file(audit_path) != parent["failure_audit_sha256"]:
        raise ValueError("V22 failure audit hash mismatch")
    change = registry["single_change"]
    if change != {
        "description": "Train the byte-identical V21 representation/model/protocol from scratch with CosineAnnealingLR over 30000 rather than 10000 optimizer steps.",
        "training_from_scratch": True, "continuation_checkpoint": None,
        "steps": 30000, "candidate_steps": [10000, 20000, 30000],
        "schedule": "CosineAnnealingLR over exactly 30000 optimizer steps",
        "minimum_learning_rate": 0.00002,
    }:
        raise ValueError("V22 differs from its single predeclared horizon change")
    _, artifacts, v20 = load_v21_program(v21_registry, v21_artifacts, repo)
    if sha256_file(v21_artifacts) != ARTIFACT_SHA256:
        raise ValueError("V22 inherited V21 artifacts differ")
    return registry, artifacts, v20, decision


def frozen_training_namespace(args: argparse.Namespace) -> argparse.Namespace:
    repo = args.repo.resolve()
    registry, _, _, _ = load_frozen_program(args.registry.resolve(), repo)
    parent = registry["parent_evidence"]
    base = v21_training_namespace(argparse.Namespace(
        repo=repo,
        registry=_resolve(parent["v21_registry"], repo),
        artifacts=_resolve(parent["v21_artifacts"], repo),
        out=args.out,
        device=args.device,
    ))
    base.steps = 30000
    base.candidate_steps = "10000,20000,30000"
    base.run_schema = V22_E10_SCHEMA
    base.experiment_registry = str(args.registry.resolve())
    base.experiment_registry_sha256 = REGISTRY_SHA256
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    train(frozen_training_namespace(parser.parse_args()))


if __name__ == "__main__":
    main()
