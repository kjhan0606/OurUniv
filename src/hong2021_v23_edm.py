#!/usr/bin/env python
"""Train and sample the frozen V23 conditional-mean-penalty experiment."""
from __future__ import annotations

import argparse
import json
import math
import socket
from pathlib import Path
from typing import Any

import torch

from hong2021_v14_edm import V23_E11_SCHEMA, train
from hong2021_v15_development_gate import canonical_digest
from hong2021_v18_init import sha256_file
from hong2021_v21_edm import ARTIFACT_SHA256, P_MEAN, P_STD
from hong2021_v22_edm import (
    frozen_training_namespace as v22_training_namespace,
    load_frozen_program as load_v22_program,
    sample_frozen_conditional_affine,
)


REGISTRY_SCHEMA = "hong2021-v23-conditional-mean-penalty-development-program-v1"
REGISTRY_SHA256 = "beb43cd57b71aaab4c4798100a009e3b36d300025d6295e08d22c0d71a15c797"
MODEL_SCHEMA = "hong2021-v23-conditional-mean-penalty-multiscale-edm-v1"


def _resolve(value: str, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_frozen_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(path) != REGISTRY_SHA256:
        raise ValueError("V23 registry differs from its frozen hash")
    registry = json.loads(path.read_text())
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("status") != "frozen_before_implementation_or_execution"
    ):
        raise ValueError("V23 registry schema or status mismatch")
    parent = registry["parent_evidence"]
    v22_registry = _resolve(parent["v22_registry"], repo)
    if sha256_file(v22_registry) != parent["v22_registry_sha256"]:
        raise ValueError("V23 V22 registry hash mismatch")
    decision_path = Path(parent["v22_decision"])
    if sha256_file(decision_path) != parent["v22_decision_sha256"]:
        raise ValueError("V23 V22 decision file hash mismatch")
    decision = json.loads(decision_path.read_text())
    if canonical_digest(decision) != parent["v22_decision_digest_sha256"]:
        raise ValueError("V23 V22 decision digest mismatch")
    if decision.get("development_pass") is not False:
        raise ValueError("V23 requires the frozen failed V22 decision")
    if (
        decision.get("Astrid_used") is not False
        or decision.get("EAGLE_RefL0100N1504_used") is not False
    ):
        raise ValueError("V23 parent decision violated the independent-data firewall")
    audit_path = Path(parent["v22_automatic_failure_audit"])
    if sha256_file(audit_path) != parent["v22_automatic_failure_audit_sha256"]:
        raise ValueError("V23 V22 automatic failure audit hash mismatch")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("Astrid_accessed") is not False
        or audit.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V23 V22 audit violated the independent-data firewall")
    if registry["single_change"]["model_schema"] != MODEL_SCHEMA:
        raise ValueError("V23 model schema differs from its registry")
    if registry["single_change"]["lambda_conditional_mean"] != 1.0:
        raise ValueError("V23 conditional coefficient differs from its registry")
    if registry["single_change"]["minimum_voxels_per_sample_bin"] != 64:
        raise ValueError("V23 conditional occupancy guard differs from its registry")
    _, artifacts, v20, v21_decision = load_v22_program(v22_registry, repo)
    if sha256_file(_resolve(parent["v21_artifacts"], repo)) != ARTIFACT_SHA256:
        raise ValueError("V23 inherited V21 artifacts differ")
    return registry, artifacts, v20, decision


def frozen_training_namespace(
    args: argparse.Namespace, *, require_preflight: bool = True
) -> argparse.Namespace:
    repo = args.repo.resolve()
    host = socket.gethostname()
    if host.lower() != "lageunha":
        raise RuntimeError(f"V23 training requires Lageunha, found {host}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V23 training requires the Lageunha Ada CUDA device")
    gpu = torch.cuda.get_device_name(0)
    if "ada" not in gpu.lower():
        raise RuntimeError(f"V23 training requires an Ada GPU, found {gpu}")
    registry, artifacts, _, _ = load_frozen_program(args.registry.resolve(), repo)
    base = v22_training_namespace(
        argparse.Namespace(
            repo=repo,
            registry=_resolve(registry["parent_evidence"]["v22_registry"], repo),
            out=args.out,
            device=args.device,
        )
    )
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    edges = profile.get("edges")
    if profile.get("bins") != 10 or not isinstance(edges, list) or len(edges) != 11:
        raise ValueError("V23 requires the frozen ten-bin V21 profile")
    base.run_schema = V23_E11_SCHEMA
    base.experiment_registry = str(args.registry.resolve())
    base.experiment_registry_sha256 = REGISTRY_SHA256
    base.conditional_mean_edges = edges
    base.lambda_conditional_mean = 1.0
    base.conditional_minimum_count = 64
    base.execution_host = host
    base.execution_gpu = gpu
    if require_preflight:
        preflight_path = args.preflight.resolve()
        if not preflight_path.is_file():
            raise RuntimeError(f"V23 hard preflight is absent: {preflight_path}")
        preflight = json.loads(preflight_path.read_text())
        expected = {
            "schema": "hong2021-v23-hard-preflight-v1",
            "status": "pass",
            "code_commit": base.code_commit_at_launch,
            "registry_sha256": REGISTRY_SHA256,
            "host": host,
            "gpu": gpu,
        }
        for key, value in expected.items():
            if preflight.get(key) != value:
                raise RuntimeError(f"V23 hard preflight mismatch: {key}")
        if preflight.get("Astrid_accessed") is not False or preflight.get(
            "historical_EAGLE_accessed"
        ) is not False:
            raise RuntimeError("V23 hard preflight violated the data firewall")
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
    if checkpoint.get("schema") != V23_E11_SCHEMA or int(
        checkpoint.get("step", -1)
    ) != step:
        raise ValueError("V23 checkpoint schema or step mismatch")
    if (
        checkpoint.get("experiment_registry_sha256") != REGISTRY_SHA256
        or checkpoint.get("worktree_clean_at_launch") is not True
    ):
        raise ValueError("V23 checkpoint provenance mismatch")
    initialization = artifacts["initialization"]
    if float(checkpoint.get("sigma_data", math.nan)) != float(
        initialization["sigma_data"]
    ):
        raise ValueError("V23 checkpoint sigma_data mismatch")
    if (
        float(checkpoint.get("edm_p_mean", math.nan)) != P_MEAN
        or float(checkpoint.get("edm_p_std", math.nan)) != P_STD
    ):
        raise ValueError("V23 checkpoint noise constants mismatch")
    if int(checkpoint.get("steps", -1)) != 30000 or checkpoint.get(
        "candidate_steps"
    ) != [10000, 20000, 30000]:
        raise ValueError("V23 checkpoint horizon differs from registry")
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    specification = checkpoint.get("denoising_loss", {})
    expected_specification = {
        "coefficients": {
            "unweighted": 0.5,
            "tail_weighted": 0.5,
            "conditional_mean": 1.0,
        },
        "conditional_mean_channel": 2,
        "conditional_mean_edges": profile["edges"],
        "conditional_minimum_count": 64,
        "conditional_bin_rule": "clamp(bucketize(m, edges, right=True)-1, 0, 9)",
        "band_balanced": False,
    }
    if specification != expected_specification:
        raise ValueError("V23 checkpoint loss specification mismatch")
    if checkpoint.get("conditional_validation_selection_role") != "none":
        raise ValueError("V23 checkpoint conditional diagnostic changed selection role")
    preflight_path = Path(str(checkpoint.get("hard_preflight", "")))
    if not preflight_path.is_file() or sha256_file(preflight_path) != checkpoint.get(
        "hard_preflight_sha256"
    ):
        raise ValueError("V23 checkpoint hard-preflight seal mismatch")
    preflight = json.loads(preflight_path.read_text())
    if (
        preflight.get("schema") != "hong2021-v23-hard-preflight-v1"
        or preflight.get("status") != "pass"
        or preflight.get("code_commit") != checkpoint.get("code_commit_at_launch")
        or preflight.get("registry_sha256") != REGISTRY_SHA256
        or preflight.get("host") != checkpoint.get("execution_host")
        or preflight.get("gpu") != checkpoint.get("execution_gpu")
        or str(checkpoint.get("execution_host", "")).lower() != "lageunha"
        or "ada" not in str(checkpoint.get("execution_gpu", "")).lower()
    ):
        raise ValueError("V23 checkpoint execution environment is not sealed")
    return checkpoint, digest


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    sample_frozen_conditional_affine(
        args,
        program_loader=load_frozen_program,
        checkpoint_validator=_validate_checkpoint,
        registry_sha=REGISTRY_SHA256,
        registry_metadata_key="v23_registry_sha256",
        label="V23",
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
