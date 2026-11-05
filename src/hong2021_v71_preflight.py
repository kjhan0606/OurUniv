#!/usr/bin/env python
"""Code-only V71 Path-B preflight before development semantic access."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import torch

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v71_ecc import (
    PREFLIGHT_SCHEMA,
    PROGRAM_FREEZE_COMMIT,
    PROGRAM_SHA256,
    authorize_parent_evidence,
    ensemble_copula_couple,
    load_development_definition,
    load_program,
    validate_frozen_gate_sources,
)


def run_preflight(program_path: Path, repo: Path, out: Path) -> dict:
    repo = repo.resolve()
    program = load_program(program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V71 preflight requires a clean Lageunha worktree")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V71 preflight requires the Lageunha Ada GPU")
    evidence = authorize_parent_evidence(program, repo, commit)
    roots = program["output_roots"]
    development = Path(roots["development"]).resolve()
    seal = Path(roots["terminal_seal"]).resolve()
    if development.exists() or seal.exists() or out.exists():
        raise FileExistsError("V71 preflight refuses prior single-use output")
    validate_frozen_gate_sources(program, repo)
    v35 = load_development_definition(program, repo)

    generator = torch.Generator(device="cpu").manual_seed(710071)
    rank_source = torch.randn((16, 1, 5, 4, 3), generator=generator)
    marginal = torch.randn((16, 1, 5, 4, 3), generator=generator)
    marginal[3, 0, 0, 0, 0] = marginal[2, 0, 0, 0, 0]
    first, diagnostics = ensemble_copula_couple(rank_source, marginal)
    second, repeated = ensemble_copula_couple(rank_source, marginal)
    if (
        not torch.equal(first, second)
        or diagnostics != repeated
        or diagnostics["pre_inverse_sorted_latent_multiset_equal"] is not True
        or diagnostics[
            "candidate_rank_disagreement_fraction_excluding_control_ties"
        ]
        != 0.0
    ):
        raise RuntimeError("V71 synthetic ECC preflight differs")

    result = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "complete_code_only_path_B_preflight_development_authorized",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "program_freeze_commit": PROGRAM_FREEZE_COMMIT,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "parent_evidence": {
            "v70_train_gate_sha256": evidence["v70_train_gate_sha256"],
            "v70_terminal_seal_sha256": evidence["v70_terminal_seal_sha256"],
            "v70_candidate_selected": False,
            "v70_development_accessed": False,
        },
        "development_integrity": {
            domain: {
                "selection_sha256": program["immutable_development_selection"][
                    f"{domain}_selection_sha256"
                ],
                "validation_data_sha256": v35["development_domains"][domain][
                    "validation_data_sha256"
                ],
                "validation_cache_sha256": v35["development_domains"][domain][
                    "validation_cache_sha256"
                ],
            }
            for domain in ("TNG100", "SIMBA", "Swift")
        },
        "synthetic_ECC": diagnostics,
        "synthetic_ECC_repeat_bitwise_equal": True,
        "development_output_absent": True,
        "terminal_seal_absent": True,
        "development_payload_bytes_hashed_for_integrity": True,
        "development_payload_semantics_accessed": False,
        "development_truth_or_source_index_read": False,
        "training_gradient_optimizer_or_parameter_update": False,
        "fresh_train_only_V71_screen_available": False,
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
