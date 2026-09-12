#!/usr/bin/env python
"""Train and sample the frozen V25 proper unweighted-score experiment."""
from __future__ import annotations

import argparse
import json
import math
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hong2021_v14_edm import V25_E13_SCHEMA, train
from hong2021_v15_development_gate import canonical_digest
from hong2021_v18_init import sha256_file
from hong2021_v21_edm import ARTIFACT_SHA256, P_MEAN, P_STD
from hong2021_v22_edm import sample_frozen_conditional_affine
from hong2021_v24_edm import (
    PARAMETERS,
    frozen_training_namespace as v24_training_namespace,
    load_frozen_program as load_v24_program,
)


REGISTRY_SCHEMA = "hong2021-v25-proper-unweighted-score-development-program-v1"
REGISTRY_SHA256 = "aa4b00027e959a857275e6d2f5617a442238ef335f0207b9db8fafa9e6d350ea"
MODEL_SCHEMA = V25_E13_SCHEMA
FLOAT32_REPLAY_ATOL = float(np.finfo(np.float32).eps)


def _resolve(value: str, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_frozen_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    if sha256_file(path) != REGISTRY_SHA256:
        raise ValueError("V25 registry differs from its frozen hash")
    registry = json.loads(path.read_text())
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("status") != "frozen_before_implementation_or_execution"
    ):
        raise ValueError("V25 registry schema or status mismatch")
    baseline = registry["controlled_baseline"]
    v24_registry = _resolve(baseline["registry"], repo)
    if sha256_file(v24_registry) != baseline["registry_sha256"]:
        raise ValueError("V25 controlled V24 registry hash mismatch")
    decision_path = Path(baseline["decision"])
    if sha256_file(decision_path) != baseline["decision_sha256"]:
        raise ValueError("V25 controlled V24 decision hash mismatch")
    decision = json.loads(decision_path.read_text())
    if (
        canonical_digest(decision) != baseline["decision_digest_sha256"]
        or decision.get("development_pass") is not False
        or decision.get("Astrid_used") is not False
        or decision.get("EAGLE_RefL0100N1504_used") is not False
    ):
        raise ValueError("V25 controlled V24 decision is invalid")
    audit_spec = registry["mechanism_audit"]
    attestation_path = _resolve(audit_spec["attestation"], repo)
    if sha256_file(attestation_path) != audit_spec["attestation_sha256"]:
        raise ValueError("V25 mechanism attestation hash mismatch")
    attestation = json.loads(attestation_path.read_text())
    if (
        attestation.get("Astrid_accessed") is not False
        or attestation.get("historical_EAGLE_accessed") is not False
        or attestation.get("recommended_v25_single_change")
        != "Keep the complete V24 base-48 representation, data, training horizon, initialization, sampler, seeds, and gates, but replace 0.5 unweighted plus 0.5 target-tail-weighted EDM with the statistically proper 1.0 unweighted EDM score objective."
    ):
        raise ValueError("V25 mechanism attestation or firewall mismatch")
    for prefix in ("physical_tail_audit", "terminal_sampler_audit"):
        audit_path = Path(audit_spec[prefix])
        if sha256_file(audit_path) != audit_spec[f"{prefix}_sha256"]:
            raise ValueError(f"V25 {prefix} hash mismatch")
        audit = json.loads(audit_path.read_text())
        if (
            audit.get("audit_digest_sha256")
            != audit_spec[f"{prefix}_digest_sha256"]
            or audit.get("Astrid_accessed") is not False
            or audit.get("historical_EAGLE_accessed") is not False
        ):
            raise ValueError(f"V25 {prefix} digest or firewall mismatch")
    replay = attestation["terminal_sampler_replay"]
    if (
        replay["numerically_identical_within_one_float32_epsilon_fields"] != 208
        or replay["maximum_absolute_replay_difference_in_y"] > FLOAT32_REPLAY_ATOL
        or replay["failed_fields_with_terminal_centered_z_at_or_above_5"] != 0
    ):
        raise ValueError("V25 terminal replay interpretation mismatch")
    change = registry["single_change"]
    if (
        change.get("checkpoint_schema") != MODEL_SCHEMA
        or change.get("loss_coefficients_from")
        != {"unweighted": 0.5, "tail_weighted": 0.5}
        or change.get("loss_coefficients_to")
        != {"unweighted": 1.0, "tail_weighted": 0.0}
        or change.get("base_channels") != 48
        or change.get("parameters") != PARAMETERS
        or change.get("steps") != 30000
        or change.get("candidate_steps") != [10000, 20000, 30000]
    ):
        raise ValueError("V25 single objective change differs from registry")
    _, artifacts, v20, _ = load_v24_program(v24_registry, repo)
    if sha256_file(_resolve("config/hong2021_v21_derived_artifacts.json", repo)) != ARTIFACT_SHA256:
        raise ValueError("V25 inherited V21 artifacts differ")
    return registry, artifacts, v20, decision


def frozen_training_namespace(
    args: argparse.Namespace, *, require_preflight: bool = True
) -> argparse.Namespace:
    repo = args.repo.resolve()
    if socket.gethostname().lower() != "lageunha":
        raise RuntimeError("V25 training requires Lageunha")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V25 training requires the Lageunha Ada CUDA device")
    gpu = torch.cuda.get_device_name(0)
    if "ada" not in gpu.lower():
        raise RuntimeError(f"V25 training requires an Ada GPU, found {gpu}")
    registry, _, _, _ = load_frozen_program(args.registry.resolve(), repo)
    v24_registry = _resolve(registry["controlled_baseline"]["registry"], repo)
    base = v24_training_namespace(
        argparse.Namespace(
            repo=repo,
            registry=v24_registry,
            out=args.out,
            device=args.device,
        ),
        require_preflight=False,
    )
    base.run_schema = MODEL_SCHEMA
    base.experiment_registry = str(args.registry.resolve())
    base.experiment_registry_sha256 = REGISTRY_SHA256
    base.execution_host = socket.gethostname()
    base.execution_gpu = gpu
    if require_preflight:
        preflight_path = args.preflight.resolve()
        if not preflight_path.is_file():
            raise RuntimeError(f"V25 hard preflight is absent: {preflight_path}")
        preflight = json.loads(preflight_path.read_text())
        expected = {
            "schema": "hong2021-v25-hard-preflight-v1",
            "status": "pass",
            "code_commit": base.code_commit_at_launch,
            "registry_sha256": REGISTRY_SHA256,
            "host": socket.gethostname(),
            "gpu": gpu,
            "base_channels": 48,
            "parameters": PARAMETERS,
            "optimized_loss_coefficients": {
                "unweighted": 1.0,
                "tail_weighted": 0.0,
            },
        }
        for key, value in expected.items():
            if preflight.get(key) != value:
                raise RuntimeError(f"V25 hard preflight mismatch: {key}")
        if (
            preflight.get("Astrid_accessed") is not False
            or preflight.get("historical_EAGLE_accessed") is not False
        ):
            raise RuntimeError("V25 hard preflight violated the data firewall")
        base.hard_preflight = str(preflight_path)
        base.hard_preflight_sha256 = sha256_file(preflight_path)
    else:
        base.hard_preflight = None
        base.hard_preflight_sha256 = None
    return base


def _validate_checkpoint(
    path: Path, *, step: int, artifacts: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    digest = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != MODEL_SCHEMA
        or int(checkpoint.get("step", -1)) != step
    ):
        raise ValueError("V25 checkpoint schema or step mismatch")
    if (
        checkpoint.get("experiment_registry_sha256") != REGISTRY_SHA256
        or checkpoint.get("worktree_clean_at_launch") is not True
        or checkpoint.get("nondevelopment_data_used") is not False
    ):
        raise ValueError("V25 checkpoint provenance mismatch")
    if (
        checkpoint.get("base_channels") != 48
        or checkpoint.get("parameters") != PARAMETERS
        or int(checkpoint.get("steps", -1)) != 30000
        or checkpoint.get("candidate_steps") != [10000, 20000, 30000]
    ):
        raise ValueError("V25 checkpoint architecture or horizon mismatch")
    initialization = artifacts["initialization"]
    if float(checkpoint.get("sigma_data", math.nan)) != float(
        initialization["sigma_data"]
    ):
        raise ValueError("V25 checkpoint sigma_data mismatch")
    if (
        float(checkpoint.get("edm_p_mean", math.nan)) != P_MEAN
        or float(checkpoint.get("edm_p_std", math.nan)) != P_STD
    ):
        raise ValueError("V25 checkpoint noise constants mismatch")
    if checkpoint.get("denoising_loss") != {
        "coefficients": {"unweighted": 1.0, "tail_weighted": 0.0},
        "tail_weighted_role": "detached_diagnostic_only",
        "proper_score_for_unreweighted_conditional_distribution": True,
        "band_balanced": False,
    }:
        raise ValueError("V25 checkpoint is not the proper unweighted objective")
    if checkpoint.get("tail_weighted_validation_selection_role") != "none":
        raise ValueError("V25 tail-weighted diagnostic entered selection")
    preflight_path = Path(str(checkpoint.get("hard_preflight", "")))
    if (
        not preflight_path.is_file()
        or sha256_file(preflight_path) != checkpoint.get("hard_preflight_sha256")
    ):
        raise ValueError("V25 checkpoint hard-preflight seal mismatch")
    preflight = json.loads(preflight_path.read_text())
    if (
        preflight.get("schema") != "hong2021-v25-hard-preflight-v1"
        or preflight.get("status") != "pass"
        or preflight.get("code_commit") != checkpoint.get("code_commit_at_launch")
        or preflight.get("registry_sha256") != REGISTRY_SHA256
        or preflight.get("host") != checkpoint.get("execution_host")
        or preflight.get("gpu") != checkpoint.get("execution_gpu")
        or str(checkpoint.get("execution_host", "")).lower() != "lageunha"
        or "ada" not in str(checkpoint.get("execution_gpu", "")).lower()
    ):
        raise ValueError("V25 checkpoint execution environment is not sealed")
    return checkpoint, digest


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    sample_frozen_conditional_affine(
        args,
        program_loader=load_frozen_program,
        checkpoint_validator=_validate_checkpoint,
        registry_sha=REGISTRY_SHA256,
        registry_metadata_key="v25_registry_sha256",
        label="V25",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    training = sub.add_parser("train")
    sampling = sub.add_parser("sample")
    for item in (training, sampling):
        item.add_argument("--registry", type=Path, required=True)
        item.add_argument("--repo", type=Path, required=True)
        item.add_argument("--device", default="cuda")
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--preflight", type=Path, required=True)
    sampling.add_argument("--training-root", type=Path, required=True)
    sampling.add_argument(
        "--domain", choices=("TNG100", "SIMBA", "Swift"), required=True
    )
    sampling.add_argument(
        "--step", type=int, choices=(10000, 20000, 30000), required=True
    )
    sampling.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train(frozen_training_namespace(args)) if args.mode == "train" else sample(args)


if __name__ == "__main__":
    main()
