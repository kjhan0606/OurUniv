#!/usr/bin/env python
"""Consumed-only V83 engineering gate using the frozen V79 observations."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np

import hong2021_v79_complete_gate as v79
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor
from hong2021_v83_contract import DOMAIN_ORDER, load_program
from hong2021_v83_sample import DOMAIN_KEYS, expected_attrs


SCHEMA = "hong2021-v83-consumed-development-engineering-gate-v1"
V79_PROGRAM = Path("config/hong2021_v79_complete_candidate_agnostic_gate_program.json")
BLOCK_ALPHA = 0.025
GLOBAL_ALPHA = 0.05


def prospective_pass(
    physical_p: float,
    rank_p: float,
    global_p: float,
    numerical_pass: bool,
) -> bool:
    return bool(
        physical_p > BLOCK_ALPHA
        and rank_p > BLOCK_ALPHA
        and global_p > GLOBAL_ALPHA
        and numerical_pass
    )


def _sampling_commit(path: Path, current_commit: str, repo: Path) -> str:
    import h5py

    with h5py.File(path, "r") as handle:
        value = handle.attrs.get("sampling_code_commit")
        if isinstance(value, bytes):
            value = value.decode()
        commit = str(value)
    if len(commit) != 40 or not _is_ancestor(repo, commit, current_commit):
        raise ValueError("V83 sampling commit is not frozen ancestry")
    return commit


def gate(
    program_path: Path,
    repo: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    train_gate_path: Path,
    train_gate_sha256: str,
    ensemble_root: Path,
    metrics_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    commit, clean = git_state(repo)
    if not clean or socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V83 development gate requires clean frozen Lageunha")
    if output_path.exists():
        raise FileExistsError("V83 development gate refuses an existing output")
    program, _, _ = load_program(program_path, repo, commit)
    if (
        output_path.resolve()
        != Path(program["output_roots"]["development_gate"]).resolve()
        or ensemble_root.resolve()
        != Path(program["output_roots"]["development"]).resolve()
        or metrics_root.resolve()
        != Path(program["output_roots"]["development_metrics"]).resolve()
    ):
        raise ValueError("V83 development output binding differs")
    if (
        sha256_file(checkpoint_path) != checkpoint_sha256
        or sha256_file(train_gate_path) != train_gate_sha256
    ):
        raise ValueError("V83 development authorization artifact differs")
    train_gate = json.loads(train_gate_path.read_text())
    if (
        train_gate.get("status") != "pass"
        or train_gate.get("train_holdout_mechanism_pass") is not True
        or train_gate.get("checkpoint_sha256") != checkpoint_sha256
        or train_gate.get("program_sha256") != sha256_file(program_path)
    ):
        raise ValueError("V83 train-only gate did not pass")
    _, references = v79.load_program((repo / V79_PROGRAM).resolve(), repo)
    domains: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Any] = {}
    numerical_pass = True
    for domain in DOMAIN_ORDER:
        domain_key = DOMAIN_KEYS[domain]
        candidate = ensemble_root / "candidate" / domain_key / "ensemble16.h5"
        control = ensemble_root / "control" / domain_key / "ensemble16.h5"
        candidate_metrics_path = (
            metrics_root / "candidate" / domain_key / "metrics.json"
        )
        control_metrics_path = metrics_root / "control" / domain_key / "metrics.json"
        candidate_sha = sha256_file(candidate)
        control_sha = sha256_file(control)
        candidate_metrics_sha = sha256_file(candidate_metrics_path)
        control_metrics_sha = sha256_file(control_metrics_path)
        selection = list(
            map(int, program["consumed_development"]["selection"][domain])
        )
        seed = int(program["consumed_development"]["seeds"][domain])
        binding = program["consumed_development"]["file_bindings"][domain]
        pairing_digest = program["consumed_development"]["pairing_sha256"][domain]
        sample_commit = _sampling_commit(candidate, commit, repo)
        if _sampling_commit(control, commit, repo) != sample_commit:
            raise ValueError(f"V83 {domain} candidate/control commit differs")
        candidate_attrs = expected_attrs(
            "candidate",
            seed,
            checkpoint_sha256,
            binding["data_sha256"],
            binding["cache_sha256"],
            sample_commit,
            pairing_digest,
        )
        control_attrs = expected_attrs(
            "control",
            seed,
            checkpoint_sha256,
            binding["data_sha256"],
            binding["cache_sha256"],
            sample_commit,
            pairing_digest,
        )
        pairing = {
            "rule": "same_PCG64_innovation_multiset_before_V72_SQT",
            "innovation_pairing_digest": pairing_digest,
            "candidate_control_pairing_proven": True,
        }
        numerical = v79.validate_ensemble_pair(
            candidate,
            control,
            selection,
            candidate_attrs,
            control_attrs,
            pairing,
        )
        numerical_pass = numerical_pass and bool(numerical["residual_DC_pass"])
        domains[domain] = {
            "candidate_ensemble": candidate,
            "candidate_ensemble_sha256": candidate_sha,
            "control_ensemble": control,
            "control_ensemble_sha256": control_sha,
            "candidate_metrics_path": candidate_metrics_path,
            "candidate_metrics_sha256": candidate_metrics_sha,
            "control_metrics_path": control_metrics_path,
            "control_metrics_sha256": control_metrics_sha,
            "candidate_metrics": v79.load_metrics(
                candidate_metrics_path, candidate, selection
            ),
            "control_metrics": v79.load_metrics(
                control_metrics_path, control, selection
            ),
            "numerical_and_pairing": numerical,
        }
        artifacts[domain] = {
            "candidate_ensemble": {
                "path": str(candidate),
                "sha256": candidate_sha,
            },
            "control_ensemble": {"path": str(control), "sha256": control_sha},
            "candidate_metrics": {
                "path": str(candidate_metrics_path),
                "sha256": candidate_metrics_sha,
            },
            "control_metrics": {
                "path": str(control_metrics_path),
                "sha256": control_metrics_sha,
            },
        }
    physical_p, physical = v79.physical_energy_observation(domains, references)
    rank_p, rank = v79.rank_coverage_observation(domains)
    global_p = float(
        v79.v78.global_p_value(np.asarray([physical_p]), np.asarray([rank_p]))[0]
    )
    passed = prospective_pass(physical_p, rank_p, global_p, numerical_pass)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass" if passed else "fail",
        "program": str(program_path.resolve()),
        "program_sha256": sha256_file(program_path),
        "code_commit": commit,
        "engineering_only": True,
        "same_previously_consumed_V80_selection_reused": True,
        "statistically_independent_validation_result": False,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "train_gate": str(train_gate_path.resolve()),
        "train_gate_sha256": train_gate_sha256,
        "artifacts": artifacts,
        "physical_energy_global_block": physical,
        "rank_coverage_global_block": rank,
        "physical_block_p": physical_p,
        "rank_block_p": rank_p,
        "complete_global_p": global_p,
        "thresholds": {
            "physical_block_alpha": BLOCK_ALPHA,
            "rank_block_alpha": BLOCK_ALPHA,
            "global_alpha": GLOBAL_ALPHA,
            "strict_greater_than": True,
        },
        "deterministic_numerical_and_pairing_pass": numerical_pass,
        "consumed_development_engineering_pass": passed,
        "may_be_reported_as_independent_validation_pass": False,
        "validation_payload_accessed": True,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
        "next": (
            "seal_single_V83_candidate_and_await_explicit_user_approval_before_independent_validation"
            if passed
            else "stop_V83_candidate_without_independent_validation"
        ),
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(partial, output_path)
    print(json.dumps(result, indent=2, allow_nan=False, default=str), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--train-gate-sha256", required=True)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gate(
        args.program,
        args.repo,
        args.checkpoint,
        args.checkpoint_sha256,
        args.train_gate,
        args.train_gate_sha256,
        args.ensemble_root,
        args.metrics_root,
        args.out,
    )


if __name__ == "__main__":
    main()
