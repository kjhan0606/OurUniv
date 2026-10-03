#!/usr/bin/env python
"""Train and sample the frozen V24 base-48 capacity experiment."""
from __future__ import annotations

import argparse
import json
import math
import socket
from pathlib import Path
from typing import Any

import torch

from hong2021_v14_edm import V24_E12_SCHEMA, train
from hong2021_v15_development_gate import canonical_digest
from hong2021_v18_init import sha256_file
from hong2021_v21_edm import ARTIFACT_SHA256, P_MEAN, P_STD
from hong2021_v22_edm import (
    frozen_training_namespace as v22_training_namespace,
    load_frozen_program as load_v22_program,
    sample_frozen_conditional_affine,
)


REGISTRY_SCHEMA = "hong2021-v24-base48-capacity-development-program-v1"
REGISTRY_SHA256 = "b0c4503cd084feef0658bb78074d42aa87e8203fe63e3a43fcdf22195e2cc7d2"
MODEL_SCHEMA = "hong2021-v24-base48-capacity-multiscale-edm-v1"
PARAMETERS = 8133361


def _resolve(value: str, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_frozen_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(path) != REGISTRY_SHA256:
        raise ValueError("V24 registry differs from its frozen hash")
    registry = json.loads(path.read_text())
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("status") != "frozen_before_implementation_or_execution"
    ):
        raise ValueError("V24 registry schema or status mismatch")
    baseline = registry["controlled_baseline"]
    v22_registry = _resolve(baseline["registry"], repo)
    if sha256_file(v22_registry) != baseline["registry_sha256"]:
        raise ValueError("V24 V22 registry hash mismatch")
    decision_path = Path(baseline["decision"])
    if sha256_file(decision_path) != baseline["decision_sha256"]:
        raise ValueError("V24 V22 decision hash mismatch")
    decision = json.loads(decision_path.read_text())
    if (
        canonical_digest(decision) != baseline["decision_digest_sha256"]
        or decision.get("development_pass") is not False
        or decision.get("Astrid_used") is not False
        or decision.get("EAGLE_RefL0100N1504_used") is not False
    ):
        raise ValueError("V24 V22 baseline decision is invalid")
    negative = registry["intervening_negative_result"]
    for key, hash_key in (
        ("v23_registry", "v23_registry_sha256"),
        ("mechanism_audit_attestation", "mechanism_audit_attestation_sha256"),
    ):
        if sha256_file(_resolve(negative[key], repo)) != negative[hash_key]:
            raise ValueError(f"V24 negative-result evidence mismatch: {key}")
    v23_decision_path = Path(negative["v23_decision"])
    if sha256_file(v23_decision_path) != negative["v23_decision_sha256"]:
        raise ValueError("V24 V23 decision hash mismatch")
    v23_decision = json.loads(v23_decision_path.read_text())
    if (
        canonical_digest(v23_decision) != negative["v23_decision_digest_sha256"]
        or v23_decision.get("development_pass") is not False
        or v23_decision.get("Astrid_used") is not False
    ):
        raise ValueError("V24 V23 negative decision is invalid")
    mechanism_path = Path(negative["mechanism_audit"])
    if sha256_file(mechanism_path) != negative["mechanism_audit_sha256"]:
        raise ValueError("V24 mechanism audit hash mismatch")
    mechanism = json.loads(mechanism_path.read_text())
    if (
        mechanism.get("audit_digest_sha256")
        != negative["mechanism_audit_digest_sha256"]
        or mechanism.get("Astrid_accessed") is not False
        or mechanism.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V24 mechanism audit digest or firewall mismatch")
    change = registry["single_change"]
    if (
        change.get("model_schema") != MODEL_SCHEMA
        or change.get("base_channels_from") != 32
        or change.get("base_channels_to") != 48
        or change.get("expected_parameters") != PARAMETERS
        or change.get("steps") != 30000
        or change.get("candidate_steps") != [10000, 20000, 30000]
    ):
        raise ValueError("V24 single capacity change differs from registry")
    _, artifacts, v20, _ = load_v22_program(v22_registry, repo)
    if sha256_file(_resolve("config/hong2021_v21_derived_artifacts.json", repo)) != ARTIFACT_SHA256:
        raise ValueError("V24 inherited V21 artifacts differ")
    return registry, artifacts, v20, decision


def frozen_training_namespace(
    args: argparse.Namespace, *, require_preflight: bool = True
) -> argparse.Namespace:
    repo = args.repo.resolve()
    host = socket.gethostname()
    if host.lower() != "lageunha":
        raise RuntimeError(f"V24 training requires Lageunha, found {host}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V24 training requires the Lageunha Ada CUDA device")
    gpu = torch.cuda.get_device_name(0)
    if "ada" not in gpu.lower():
        raise RuntimeError(f"V24 training requires an Ada GPU, found {gpu}")
    registry, _, _, _ = load_frozen_program(args.registry.resolve(), repo)
    base = v22_training_namespace(
        argparse.Namespace(
            repo=repo,
            registry=_resolve(registry["controlled_baseline"]["registry"], repo),
            out=args.out,
            device=args.device,
        )
    )
    base.base_channels = 48
    base.run_schema = V24_E12_SCHEMA
    base.experiment_registry = str(args.registry.resolve())
    base.experiment_registry_sha256 = REGISTRY_SHA256
    base.execution_host = host
    base.execution_gpu = gpu
    if require_preflight:
        preflight_path = args.preflight.resolve()
        if not preflight_path.is_file():
            raise RuntimeError(f"V24 hard preflight is absent: {preflight_path}")
        preflight = json.loads(preflight_path.read_text())
        expected = {
            "schema": "hong2021-v24-hard-preflight-v1",
            "status": "pass",
            "code_commit": base.code_commit_at_launch,
            "registry_sha256": REGISTRY_SHA256,
            "host": host,
            "gpu": gpu,
            "base_channels": 48,
            "parameters": PARAMETERS,
        }
        for key, value in expected.items():
            if preflight.get(key) != value:
                raise RuntimeError(f"V24 hard preflight mismatch: {key}")
        if preflight.get("Astrid_accessed") is not False or preflight.get(
            "historical_EAGLE_accessed"
        ) is not False:
            raise RuntimeError("V24 hard preflight violated the data firewall")
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
    if checkpoint.get("schema") != V24_E12_SCHEMA or int(
        checkpoint.get("step", -1)
    ) != step:
        raise ValueError("V24 checkpoint schema or step mismatch")
    if (
        checkpoint.get("experiment_registry_sha256") != REGISTRY_SHA256
        or checkpoint.get("worktree_clean_at_launch") is not True
    ):
        raise ValueError("V24 checkpoint provenance mismatch")
    if checkpoint.get("base_channels") != 48 or checkpoint.get("parameters") != PARAMETERS:
        raise ValueError("V24 checkpoint capacity mismatch")
    initialization = artifacts["initialization"]
    if float(checkpoint.get("sigma_data", math.nan)) != float(
        initialization["sigma_data"]
    ):
        raise ValueError("V24 checkpoint sigma_data mismatch")
    if (
        float(checkpoint.get("edm_p_mean", math.nan)) != P_MEAN
        or float(checkpoint.get("edm_p_std", math.nan)) != P_STD
    ):
        raise ValueError("V24 checkpoint noise constants mismatch")
    if int(checkpoint.get("steps", -1)) != 30000 or checkpoint.get(
        "candidate_steps"
    ) != [10000, 20000, 30000]:
        raise ValueError("V24 checkpoint horizon differs from registry")
    if checkpoint.get("denoising_loss") != {
        "coefficients": {"unweighted": 0.5, "tail_weighted": 0.5},
        "band_balanced": False,
    }:
        raise ValueError("V24 checkpoint is not the restored V22 objective")
    preflight_path = Path(str(checkpoint.get("hard_preflight", "")))
    if not preflight_path.is_file() or sha256_file(preflight_path) != checkpoint.get(
        "hard_preflight_sha256"
    ):
        raise ValueError("V24 checkpoint hard-preflight seal mismatch")
    preflight = json.loads(preflight_path.read_text())
    if (
        preflight.get("schema") != "hong2021-v24-hard-preflight-v1"
        or preflight.get("status") != "pass"
        or preflight.get("code_commit") != checkpoint.get("code_commit_at_launch")
        or preflight.get("registry_sha256") != REGISTRY_SHA256
        or preflight.get("host") != checkpoint.get("execution_host")
        or preflight.get("gpu") != checkpoint.get("execution_gpu")
        or preflight.get("base_channels") != 48
        or preflight.get("parameters") != PARAMETERS
        or str(checkpoint.get("execution_host", "")).lower() != "lageunha"
        or "ada" not in str(checkpoint.get("execution_gpu", "")).lower()
    ):
        raise ValueError("V24 checkpoint execution environment is not sealed")
    return checkpoint, digest


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    sample_frozen_conditional_affine(
        args,
        program_loader=load_frozen_program,
        checkpoint_validator=_validate_checkpoint,
        registry_sha=REGISTRY_SHA256,
        registry_metadata_key="v24_registry_sha256",
        label="V24",
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
