#!/usr/bin/env python
"""Code-only V72 SQT preflight before either fresh stage is opened."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import torch

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v72_sqt import (
    PREFLIGHT_SCHEMA,
    PROGRAM_FREEZE_COMMIT,
    PROGRAM_SHA256,
    authorize_parent_evidence,
    conditioning_strata,
    load_program,
    spatial_quantile_transport,
    stage_selection,
    validate_frozen_gate_sources,
)


def run_preflight(program_path: Path, repo: Path, out: Path) -> dict:
    repo = repo.resolve()
    program = load_program(program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if not clean or socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V72 preflight requires a clean Lageunha worktree")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V72 preflight requires the Lageunha Ada GPU")
    evidence = authorize_parent_evidence(program, repo, commit)
    roots = program["output_roots"]
    if any(Path(roots[key]).exists() for key in ("stage_A", "stage_B", "terminal_seal")):
        raise FileExistsError("V72 preflight refuses existing single-use output")
    if out.exists():
        raise FileExistsError("V72 preflight refuses an existing preflight")
    validate_frozen_gate_sources(program, repo)
    first = stage_selection(program, "A")
    second = stage_selection(program, "B")

    generator = torch.Generator(device="cpu").manual_seed(720072)
    rank_source = torch.randn((16, 1, 64, 64, 64), generator=generator)
    marginal = torch.randn((16, 1, 64, 64, 64), generator=generator)
    score = torch.linspace(-1.0, 1.0, 64**3).reshape(1, 64, 64, 64)
    positions = conditioning_strata(score)
    coupled, diagnostics = spatial_quantile_transport(
        rank_source, marginal, positions
    )
    repeated, repeated_diagnostics = spatial_quantile_transport(
        rank_source, marginal, positions
    )
    if (
        not torch.equal(coupled, repeated)
        or diagnostics != repeated_diagnostics
        or diagnostics["pre_inverse_stratum_multiset_equal"] is not True
        or diagnostics["rank_disagreement_fraction_excluding_marginal_ties"]
        != 0.0
    ):
        raise RuntimeError("V72 synthetic SQT preflight differs")
    threshold = float(program["numerical_requirements"]["maximum_marginal_tied_voxel_fraction"])
    if float(diagnostics["marginal_tied_voxel_fraction"]) > threshold:
        raise RuntimeError("V72 synthetic SQT tie guard differs")

    result = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "complete_code_only_V72_preflight_stage_A_authorized",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "program_freeze_commit": PROGRAM_FREEZE_COMMIT,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "cuda_device": torch.cuda.get_device_name(0),
        "parent_evidence": evidence,
        "stage_A_selection": first,
        "stage_B_selection_digest_only": {
            domain: __import__("hashlib").sha256(
                json.dumps(second[domain], separators=(",", ":")).encode()
            ).hexdigest()
            for domain in second
        },
        "stage_A_stage_B_disjoint": all(
            not set(first[domain]) & set(second[domain]) for domain in first
        ),
        "synthetic_SQT": diagnostics,
        "synthetic_repeat_bitwise_equal": True,
        "fresh_files_hashed_for_integrity": True,
        "fresh_input_or_target_dataset_read": False,
        "stage_A_metric_read": False,
        "stage_B_metric_read": False,
        "stage_A_output_absent": True,
        "stage_B_output_absent": True,
        "training_gradient_optimizer_or_parameter_update": False,
        "preflight_pass": True,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(args.program, args.repo, args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
