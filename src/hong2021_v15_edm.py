#!/usr/bin/env python
"""Launch only the two predeclared V15 EDM training variants."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from hong2021_v14_edm import V15_E2_SCHEMA, V15_E3_SCHEMA, train


REGISTRY_SCHEMA = "hong2021-v15-predeclared-development-program-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repo: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, not bool(dirty)


def frozen_training_namespace(args: argparse.Namespace) -> argparse.Namespace:
    registry_path = args.registry.resolve()
    registry = json.loads(registry_path.read_text())
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("invalid V15 development registry")
    experiment_key = {
        "e2": "e2_noise_distribution",
        "e3": "e3_tail_weight",
    }[args.experiment]
    experiment = registry[experiment_key]
    if experiment["steps"] != 10000 or experiment["candidate_steps"] != [5000, 10000]:
        raise ValueError("registry differs from the frozen V15 training budget")
    expected_tail = {"e2": 0.5, "e3": 0.25}[args.experiment]
    if experiment["tail_exponent"] != expected_tail or experiment["tail_maximum"] != 10.0:
        raise ValueError("registry differs from the frozen V15 tail objective")
    if experiment["training_seed"] != 144021:
        raise ValueError("registry differs from the frozen V15 training seed")
    validation_seeds = [
        experiment["validation_seeds"][name]
        for name in ("TNG100", "SIMBA", "Swift")
    ]
    if validation_seeds != [99173, 99174, 99175]:
        raise ValueError("registry differs from the frozen validation seeds")
    commit, clean = git_state(args.repo.resolve())
    if not clean:
        raise RuntimeError("V15 training requires a clean committed worktree")
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
        tail_exponent=expected_tail,
        tail_maximum=10.0,
        validation_every=500,
        validation_seeds=validation_seeds,
        seed=144021,
        device=args.device,
        smoke_limit=None,
        run_schema={"e2": V15_E2_SCHEMA, "e3": V15_E3_SCHEMA}[args.experiment],
        experiment_registry=str(registry_path),
        experiment_registry_sha256=sha256_file(registry_path),
        code_commit_at_launch=commit,
        worktree_clean_at_launch=clean,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("e2", "e3"), required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    for prefix in (
        "tng-train",
        "tng-validation",
        "simba-train",
        "simba-validation",
        "swift-train",
        "swift-validation",
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
