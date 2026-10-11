#!/usr/bin/env python
"""Integrity-bound three-domain development gate for V15 E2 and E3."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import h5py
import torch

from hong2021_v6_gate import field_gate
from hong2021_v14_edm import ENSEMBLE_SCHEMA, V15_E2_SCHEMA, V15_E3_SCHEMA


SCHEMA = "hong2021-v15-integrity-bound-three-domain-decision-v1"
DOMAINS = ("tng", "simba_dev", "swift_dev")
EXPECTED_SCHEMA = {"e2": V15_E2_SCHEMA, "e3": V15_E3_SCHEMA}
EXPECTED_TAIL = {"e2": 0.5, "e3": 0.25}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(report: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in report.items() if key != "decision_digest_sha256"}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def select_candidate_rows(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [row for row in rows if row["all_three_pass"]]
    return min(passing, key=lambda row: row["step"]) if passing else None


def _load_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return payload["candidates"]["edm"]


def _validate_checkpoint(
    path: Path,
    *,
    step: int,
    experiment: str,
    registry_sha256: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != EXPECTED_SCHEMA[experiment]:
        raise ValueError(f"unexpected {experiment.upper()} checkpoint schema")
    if int(checkpoint.get("step", -1)) != step:
        raise ValueError("checkpoint step differs from candidate step")
    if checkpoint.get("edm_p_mean_mode") != "log_sigma_data_fraction":
        raise ValueError("V15 checkpoint did not use relative noise placement")
    if float(checkpoint.get("edm_p_mean_sigma_data_fraction", -1)) != 0.6:
        raise ValueError("V15 checkpoint used the wrong sigma_data fraction")
    if float(checkpoint.get("edm_p_std", -1)) != 1.2:
        raise ValueError("V15 checkpoint used the wrong p_std")
    tail = checkpoint.get("tail_weight_fit", {})
    if float(tail.get("inverse_probability_exponent", -1)) != EXPECTED_TAIL[experiment]:
        raise ValueError("V15 checkpoint used the wrong tail exponent")
    if checkpoint.get("experiment_registry_sha256") != registry_sha256:
        raise ValueError("checkpoint registry digest differs from current registry")
    if checkpoint.get("worktree_clean_at_launch") is not True:
        raise ValueError("checkpoint was not launched from a clean worktree")
    commit = checkpoint.get("code_commit_at_launch")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("checkpoint has no valid launch commit")
    return checkpoint


def _validate_ensemble(
    path: Path,
    *,
    checkpoint: Path,
    checkpoint_schema: str,
    step: int,
    seed: int,
) -> None:
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("schema")) != ENSEMBLE_SCHEMA:
            raise ValueError("unexpected ensemble schema")
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError("incomplete ensemble")
        if Path(str(handle.attrs["checkpoint"])).resolve() != checkpoint.resolve():
            raise ValueError("ensemble used a different checkpoint")
        if str(handle.attrs.get("checkpoint_schema")) != checkpoint_schema:
            raise ValueError("ensemble checkpoint schema mismatch")
        if int(handle.attrs.get("checkpoint_step", -1)) != step:
            raise ValueError("ensemble checkpoint step mismatch")
        if int(handle.attrs.get("seed", -1)) != seed:
            raise ValueError("ensemble seed mismatch")
        if int(handle.attrs.get("sampling_steps", -1)) != 40:
            raise ValueError("ensemble sampling step mismatch")
        if float(handle.attrs.get("sigma_min", -1)) != 0.002:
            raise ValueError("ensemble sigma_min mismatch")
        if float(handle.attrs.get("sigma_max", -1)) != 40.0:
            raise ValueError("ensemble sigma_max mismatch")
        if float(handle.attrs.get("rho", -1)) != 7.0:
            raise ValueError("ensemble rho mismatch")
        if bool(handle.attrs.get("location_scale_uses_target", True)):
            raise ValueError("ensemble location-scale used a target")
        if handle["sample"].shape[:2] != (16, 16):
            raise ValueError("ensemble must contain 16 objects by 16 members")


def evaluate(
    *,
    root: Path,
    training: Path,
    registry_path: Path,
    experiment: str,
    gate_code_commit: str,
) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text())
    if registry.get("schema") != "hong2021-v15-predeclared-development-program-v1":
        raise ValueError("invalid V15 registry")
    registry_sha = sha256_file(registry_path)
    experiment_config = registry[
        {"e2": "e2_noise_distribution", "e3": "e3_tail_weight"}[experiment]
    ]
    steps = [int(value) for value in experiment_config["candidate_steps"]]
    if steps != [5000, 10000]:
        raise ValueError("candidate steps differ from the V15 registry")
    seed_map = {
        "tng": int(experiment_config["sampling_seeds"]["TNG100"]),
        "simba_dev": int(experiment_config["sampling_seeds"]["SIMBA"]),
        "swift_dev": int(experiment_config["sampling_seeds"]["Swift"]),
    }
    candidates = []
    for step in steps:
        checkpoint_path = training / "validation_checkpoints" / f"step_{step:06d}.pt"
        checkpoint = _validate_checkpoint(
            checkpoint_path,
            step=step,
            experiment=experiment,
            registry_sha256=registry_sha,
        )
        domains = {}
        for domain in DOMAINS:
            domain_root = root / f"step_{step:06d}" / domain
            ensemble_path = domain_root / "ensemble16_steps40.h5"
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            _validate_ensemble(
                ensemble_path,
                checkpoint=checkpoint_path,
                checkpoint_schema=EXPECTED_SCHEMA[experiment],
                step=step,
                seed=seed_map[domain],
            )
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble_path.resolve():
                raise ValueError("metrics refer to a different ensemble")
            domains[domain] = {
                "ensemble": str(ensemble_path.resolve()),
                "ensemble_sha256": sha256_file(ensemble_path),
                "metrics": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics),
            }
        candidates.append(
            {
                "step": step,
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "training_code_commit": checkpoint["code_commit_at_launch"],
                "domains": domains,
                "all_three_pass": all(
                    value["field_gate"]["pass"] for value in domains.values()
                ),
            }
        )
    selected = select_candidate_rows(candidates)
    report = {
        "schema": SCHEMA,
        "experiment": experiment,
        "registry": str(registry_path.resolve()),
        "registry_sha256": registry_sha,
        "gate_code_commit": gate_code_commit,
        "worktree_clean_at_gate": True,
        "EAGLE_RefL0100N1504_used": False,
        "Astrid_used": False,
        "predeclared_steps": steps,
        "selection_rule": "smallest predeclared step passing all unchanged V6 field checks independently in TNG100, SIMBA development, and Swift development",
        "candidates": candidates,
        "selected_step": None if selected is None else selected["step"],
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "development_pass": selected is not None,
        "next": (
            "freeze_exact_v15_hashes_before_astrid_one_shot"
            if selected is not None
            else ("run_predeclared_e3" if experiment == "e2" else "stop_without_opening_astrid")
        ),
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--experiment", choices=("e2", "e3"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    commit, clean = git_state(args.repo.resolve())
    if not clean:
        raise RuntimeError("V15 gate requires a clean committed worktree")
    report = evaluate(
        root=args.root,
        training=args.training,
        registry_path=args.registry,
        experiment=args.experiment,
        gate_code_commit=commit,
    )
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if canonical_digest(existing) != existing.get("decision_digest_sha256"):
            raise RuntimeError("existing V15 decision has an invalid digest")
        if existing != report:
            raise RuntimeError("existing V15 decision differs from recomputed decision")
        print(json.dumps(existing, indent=2))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
