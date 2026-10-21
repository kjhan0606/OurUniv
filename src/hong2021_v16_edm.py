#!/usr/bin/env python
"""Launch only the predeclared V16-E4 trilinear-decoder experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hong2021_v14_edm import V16_E4_SCHEMA, train
from hong2021_v15_development_gate import canonical_digest
from hong2021_v15_edm import git_state, sha256_file


REGISTRY_SCHEMA = "hong2021-v16-predeclared-representation-redesign-program-v1"
FROZEN_REGISTRY_SHA256 = "f4935d277c2bbd9990f511eb7755010277713e3249abd165825501ef1c1ed6f4"


def _resolve_registry_reference(value: str, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_frozen_registry(path: Path, repo: Path) -> dict[str, Any]:
    registry_path = path.resolve()
    repo = repo.resolve()
    if sha256_file(registry_path) != FROZEN_REGISTRY_SHA256:
        raise ValueError("V16 registry differs from its pre-implementation frozen hash")
    registry = json.loads(registry_path.read_text())
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("invalid V16 development registry")
    if registry.get("status") != "frozen_after_v15_e3_development_failure_before_v16_e4":
        raise ValueError("V16 registry is not in its pre-E4 frozen state")
    parent = registry["parent_v15"]
    parent_registry = _resolve_registry_reference(parent["registry"], repo)
    if sha256_file(parent_registry) != parent["registry_sha256"]:
        raise ValueError("frozen V15 registry hash mismatch")
    decision_path = _resolve_registry_reference(parent["e3_decision"], repo)
    if sha256_file(decision_path) != parent["e3_decision_sha256"]:
        raise ValueError("frozen V15 E3 decision hash mismatch")
    decision = json.loads(decision_path.read_text())
    if canonical_digest(decision) != parent["e3_decision_digest_sha256"]:
        raise ValueError("frozen V15 E3 decision digest mismatch")
    if decision.get("development_pass") is not False:
        raise ValueError("V16 may run only after a sealed V15 E3 development failure")
    if decision.get("Astrid_used") is not False:
        raise ValueError("V15 E3 does not prove that Astrid stayed unopened")
    return registry


def frozen_training_namespace(args: argparse.Namespace) -> argparse.Namespace:
    registry_path = args.registry.resolve()
    repo = args.repo.resolve()
    registry = load_frozen_registry(registry_path, repo)
    experiment = registry["e4_trilinear_decoder"]
    exact = {
        "steps": 10000,
        "candidate_steps": [5000, 10000],
        "tail_exponent": 0.25,
        "tail_maximum": 10.0,
        "training_seed": 144021,
        "validation_seeds": {"TNG100": 99173, "SIMBA": 99174, "Swift": 99175},
    }
    for key, value in exact.items():
        if experiment.get(key) != value:
            raise ValueError(f"registry differs from the frozen V16 {key}")
    interpolation = experiment.get("interpolation", {})
    if interpolation != {
        "parent": "nearest",
        "candidate": "trilinear",
        "align_corners": False,
        "antialias": False,
        "decoder_transitions": 2,
    }:
        raise ValueError("registry differs from the frozen V16 interpolation")
    if experiment.get("edm_p_mean") != "log(0.6*sigma_data)" or experiment.get("edm_p_std") != 1.2:
        raise ValueError("registry differs from the frozen V16 noise distribution")
    sampler = experiment.get("sampler", {})
    if sampler != {
        "ensemble_members": 16,
        "steps": 40,
        "sigma_min": 0.002,
        "sigma_max": 40.0,
        "rho": 7.0,
    }:
        raise ValueError("registry differs from the frozen V16 sampler")
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V16 training requires a clean committed worktree")
    return argparse.Namespace(
        tng_train_data=args.tng_train_data,
        tng_train_cache=args.tng_train_cache,
        tng_validation_data=args.tng_validation_data,
        tng_validation_cache=args.tng_validation_cache,
        simba_train_data=args.simba_train_data,
        simba_train_cache=args.simba_train_cache,
        simba_validation_data=args.simba_validation_data,
        simba_validation_cache=args.simba_validation_cache,
        swift_train_data=args.swift_train_data,
        swift_train_cache=args.swift_train_cache,
        swift_validation_data=args.swift_validation_data,
        swift_validation_cache=args.swift_validation_cache,
        out=args.out,
        steps=10000,
        candidate_steps="5000,10000",
        batch=6,
        validation_batch=6,
        workers=args.workers,
        base_channels=32,
        lr=2.0e-4,
        min_lr=2.0e-5,
        weight_decay=1.0e-4,
        ema_decay=0.999,
        gradient_clip=1.0,
        edm_p_mean=-0.8,
        edm_p_mean_sigma_data_fraction=0.6,
        edm_p_std=1.2,
        tail_exponent=0.25,
        tail_maximum=10.0,
        validation_every=500,
        validation_seeds=[99173, 99174, 99175],
        seed=144021,
        device=args.device,
        smoke_limit=None,
        run_schema=V16_E4_SCHEMA,
        experiment_registry=str(registry_path),
        experiment_registry_sha256=sha256_file(registry_path),
        code_commit_at_launch=commit,
        worktree_clean_at_launch=clean,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    for prefix in (
        "tng-train", "tng-validation", "simba-train", "simba-validation",
        "swift-train", "swift-validation",
    ):
        parser.add_argument(f"--{prefix}-data", required=True)
        parser.add_argument(f"--{prefix}-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    train(frozen_training_namespace(build_parser().parse_args()))


if __name__ == "__main__":
    main()
