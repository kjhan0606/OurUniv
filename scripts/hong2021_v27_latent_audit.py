#!/usr/bin/env python
"""Execute the frozen post-failure V27 latent audit on development data only."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import torch

from hong2021_v15_edm import git_state
from hong2021_v18_init import sha256_file
from hong2021_v26_mechanism_audit import analyze_history, audit_domain
from hong2021_v27 import (
    CANDIDATE_STEPS,
    MODEL_SCHEMA,
    REGISTRY_SHA256,
    _validate_checkpoint,
    build_model,
    load_frozen_program,
)
from hong2021_v27_latent_audit import (
    AUDIT_PROGRAM_SHA256,
    DOMAIN_ORDER,
    mechanism_summary,
)


SCHEMA = "hong2021-v27-frozen-trained-flow-latent-audit-v1"
DEFAULT_TRAINING = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/training/"
    "tng100_simba_swift_v27_e15_parent_aligned_haar_flow"
)
DEFAULT_DECISION = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/"
    "tng100_simba_swift_v27_e15_parent_aligned_haar_flow/development_decision.json"
)
DEFAULT_FAILURE_AUDIT = DEFAULT_DECISION.parent / "automatic_failure_audit.json"


def _verified_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} hash differs from the frozen audit program")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--failure-audit", type=Path, default=DEFAULT_FAILURE_AUDIT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if socket.gethostname().lower() != "lageunha":
        raise RuntimeError("V27 latent audit requires Lageunha")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V27 latent audit requires the Lageunha Ada GPU")
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V27 latent audit requires a clean committed worktree")
    output = args.out.resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite V27 latent audit: {output}")

    audit_program_path = args.audit_program.resolve()
    audit_program = _verified_json(
        audit_program_path, AUDIT_PROGRAM_SHA256, "V27 latent-audit program"
    )
    if (
        audit_program.get("schema")
        != "hong2021-v27-frozen-latent-audit-program-v1"
        or audit_program.get("status")
        != "frozen_before_v27_latent_audit_execution"
        or audit_program.get("firewall", {}).get("Astrid_accessed") is not False
        or audit_program.get("firewall", {}).get("historical_EAGLE_accessed")
        is not False
    ):
        raise ValueError("V27 latent-audit program or firewall differs")
    parent = audit_program["parent_evidence"]
    registry_path = (repo / parent["v27_registry"]).resolve()
    if (
        parent["v27_registry_sha256"] != REGISTRY_SHA256
        or sha256_file(registry_path) != REGISTRY_SHA256
    ):
        raise ValueError("V27 registry provenance differs")
    registry, artifacts, v20, _, haar = load_frozen_program(registry_path, repo)

    training_run = _verified_json(
        args.training / "run.json",
        parent["v27_training_run_sha256"],
        "V27 training run",
    )
    if (
        str(args.training.resolve()) != parent["v27_training"]
        or sha256_file(args.training / "history.json") != parent["v27_history_sha256"]
        or training_run.get("schema") != MODEL_SCHEMA
        or training_run.get("status") != "complete"
        or training_run.get("experiment_registry_sha256") != REGISTRY_SHA256
    ):
        raise ValueError("V27 completed-training provenance differs")
    decision = _verified_json(
        args.decision, parent["v27_decision_sha256"], "V27 development decision"
    )
    failure = _verified_json(
        args.failure_audit,
        parent["v27_failure_audit_sha256"],
        "V27 automatic failure audit",
    )
    if (
        str(args.decision.resolve()) != parent["v27_decision"]
        or decision.get("development_pass") is not False
        or decision.get("decision_digest_sha256")
        != parent["v27_decision_digest_sha256"]
        or decision.get("next")
        != "run_frozen_v27_latent_audit_then_test_train_only_empirical_joint_residual_control"
        or failure.get("decision_digest_sha256")
        != decision.get("decision_digest_sha256")
    ):
        raise ValueError("V27 failed-decision provenance differs")
    v26_audit_path = Path(parent["v26_mechanism_audit"])
    v26_audit = _verified_json(
        v26_audit_path,
        parent["v26_mechanism_audit_sha256"],
        "V26 mechanism audit",
    )
    if (
        v26_audit.get("schema") != "hong2021-v26-trained-flow-mechanism-audit-v2"
        or v26_audit.get("Astrid_accessed") is not False
        or v26_audit.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V26 comparison audit or firewall differs")

    optimization = analyze_history(
        json.loads((args.training / "history.json").read_text())
    )
    decision_candidates = {
        str(candidate["step"]): candidate for candidate in decision["candidates"]
    }
    device = torch.device(args.device)
    candidate_reports: dict[str, Any] = {}
    for step in CANDIDATE_STEPS:
        checkpoint_path = (
            args.training / "validation_checkpoints" / f"step_{step:06d}.pt"
        )
        checkpoint, checkpoint_sha = _validate_checkpoint(
            checkpoint_path, step=step, artifacts=artifacts
        )
        model = build_model(
            haar, checkpoint["observable_context_features"], device=device
        )
        model.load_state_dict(checkpoint["ema_model"])
        model.eval()
        domains = {}
        for domain in DOMAIN_ORDER:
            domains[domain] = audit_domain(
                domain=domain,
                step=step,
                model=model,
                checkpoint=checkpoint,
                registry=registry,
                artifacts=artifacts,
                v20=v20,
                decision_candidate=decision_candidates[str(step)],
                repo=repo,
                device=device,
            )
        candidate_reports[str(step)] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "domains": domains,
        }
        del model
        torch.cuda.empty_cache()

    compact = {step: row["domains"] for step, row in candidate_reports.items()}
    report = {
        "schema": SCHEMA,
        "status": "complete_development_only_post_failure_latent_audit",
        "purpose": audit_program["purpose"],
        "audit_program": str(audit_program_path),
        "audit_program_sha256": AUDIT_PROGRAM_SHA256,
        "registry": str(registry_path),
        "registry_sha256": REGISTRY_SHA256,
        "training": str(args.training.resolve()),
        "training_run_sha256": parent["v27_training_run_sha256"],
        "history_sha256": parent["v27_history_sha256"],
        "decision": str(args.decision.resolve()),
        "decision_sha256": parent["v27_decision_sha256"],
        "decision_digest_sha256": parent["v27_decision_digest_sha256"],
        "failure_audit": str(args.failure_audit.resolve()),
        "failure_audit_sha256": parent["v27_failure_audit_sha256"],
        "v26_mechanism_audit": str(v26_audit_path),
        "v26_mechanism_audit_sha256": parent["v26_mechanism_audit_sha256"],
        "execution_host": socket.gethostname(),
        "execution_gpu": torch.cuda.get_device_name(0),
        "audit_code_commit": commit,
        "worktree_clean_at_audit": clean,
        "optimization": optimization,
        "candidates": candidate_reports,
        "mechanism_summary": mechanism_summary(compact, optimization, v26_audit),
        "tuning_after_results": False,
        "training_horizon_extended": False,
        "thresholds_changed": False,
        "posthoc_clipping_or_Ak_applied": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(report["mechanism_summary"], indent=2), flush=True)
    print(f"[out] {output}", flush=True)


if __name__ == "__main__":
    main()
