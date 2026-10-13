#!/usr/bin/env python
"""Sample V54 through the proven V50 sampler with V54 integrity bindings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import torch

import hong2021_v50_sample as base
from hong2021_v15_development_gate import canonical_digest
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_KEYS, DOMAIN_ORDER
from hong2021_v50_network import LOWER_SUPPORT, UPPER_SUPPORT
from hong2021_v54_train import (
    CHECKPOINT_SCHEMA,
    LIKELIHOOD_FAMILY,
    PARAMETERS,
    PREFLIGHT_SCHEMA,
    PROGRAM_SHA256,
    REPORT_SCHEMA,
    SUPPORT_SHA256,
    TAIL_COEFFICIENT,
    load_program,
)
from hong2021_v54_train_gate import SCHEMA as TRAIN_GATE_SCHEMA


ENSEMBLE_SCHEMA = "hong2021-v54-physical-tail-brier-bounded-mixture-ensemble-v1"
METHOD = "train_only_physical_tail_Brier_bounded_query_local_mixture_empirical_rank_copula"
ARMS = ("bounded_query_local_mixture_copula", "rolled_parameter_control")


def _strict_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V54 sample {label} hash differs")
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _load_fit_factory(
    threshold_sha: str,
    train_gate_path: Path,
    train_gate_sha: str,
):
    def load_fit(
        checkpoint_path: Path,
        checkpoint_sha: str,
        report_path: Path,
        report_sha: str,
        cache_path: Path,
        cache_sha: str,
        preflight_path: Path,
        preflight_sha: str,
        commit: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if sha256_file(checkpoint_path) != checkpoint_sha:
            raise ValueError("V54 sample checkpoint hash differs")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        report = _strict_json(report_path, report_sha, "report")
        preflight = _strict_json(preflight_path, preflight_sha, "preflight")
        train_gate = _strict_json(train_gate_path, train_gate_sha, "train gate")
        if (
            checkpoint.get("schema") != CHECKPOINT_SCHEMA
            or checkpoint.get("program_sha256") != PROGRAM_SHA256
            or checkpoint.get("code_commit") != commit
            or checkpoint.get("step") != 12_000
            or checkpoint.get("parameters") != PARAMETERS
            or checkpoint.get("likelihood_family") != LIKELIHOOD_FAMILY
            or checkpoint.get("open_standardized_support") != [LOWER_SUPPORT, UPPER_SUPPORT]
            or checkpoint.get("support_selection_sha256") != SUPPORT_SHA256
            or checkpoint.get("conditioning_cache_sha256") != cache_sha
            or checkpoint.get("preflight_sha256") != preflight_sha
            or checkpoint.get("threshold_selection_sha256") != threshold_sha
            or checkpoint.get("tail_coefficient") != TAIL_COEFFICIENT
            or checkpoint.get("sample_clipping") is not False
            or checkpoint.get("component_scale_cap") is not False
            or report.get("schema") != REPORT_SCHEMA
            or report.get("checkpoint_sha256") != checkpoint_sha
            or report.get("threshold_selection_sha256") != threshold_sha
            or report.get("validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection") is not False
            or preflight.get("schema") != PREFLIGHT_SCHEMA
            or preflight.get("code_commit") != commit
            or train_gate.get("schema") != TRAIN_GATE_SCHEMA
            or train_gate.get("train_mechanism_pass") is not True
            or train_gate.get("checkpoint_sha256") != checkpoint_sha
            or train_gate.get("training_report_sha256") != report_sha
            or train_gate.get("threshold_selection_sha256") != threshold_sha
            or canonical_digest(train_gate) != train_gate.get("decision_digest_sha256")
            or train_gate.get("development_accessed") is not False
        ):
            raise ValueError("V54 sample fit or train-gate binding differs")
        return checkpoint, report

    return load_fit


def sample_all(
    program_path: Path,
    repo: Path,
    cache_path: Path,
    cache_sha: str,
    checkpoint_path: Path,
    checkpoint_sha: str,
    report_path: Path,
    report_sha: str,
    preflight_path: Path,
    preflight_sha: str,
    threshold_path: Path,
    threshold_sha: str,
    train_gate_path: Path,
    train_gate_sha: str,
    output_root: Path,
) -> None:
    names = (
        "PROGRAM_SHA256",
        "CHECKPOINT_SCHEMA",
        "REPORT_SCHEMA",
        "PREFLIGHT_SCHEMA",
        "LIKELIHOOD_FAMILY",
        "PARAMETERS",
        "ENSEMBLE_SCHEMA",
        "METHOD",
        "ARMS",
        "load_program",
        "load_fit",
    )
    saved = {name: getattr(base, name) for name in names}
    replacements = {
        "PROGRAM_SHA256": PROGRAM_SHA256,
        "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
        "REPORT_SCHEMA": REPORT_SCHEMA,
        "PREFLIGHT_SCHEMA": PREFLIGHT_SCHEMA,
        "LIKELIHOOD_FAMILY": LIKELIHOOD_FAMILY,
        "PARAMETERS": PARAMETERS,
        "ENSEMBLE_SCHEMA": ENSEMBLE_SCHEMA,
        "METHOD": METHOD,
        "ARMS": ARMS,
        "load_program": load_program,
        "load_fit": _load_fit_factory(threshold_sha, train_gate_path, train_gate_sha),
    }
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        base.sample_all(
            program_path,
            repo,
            cache_path,
            cache_sha,
            checkpoint_path,
            checkpoint_sha,
            report_path,
            report_sha,
            preflight_path,
            preflight_sha,
            output_root,
        )
    finally:
        for name, value in saved.items():
            setattr(base, name, value)
    for arm in ARMS:
        for domain in DOMAIN_ORDER:
            path = output_root / arm / "development_candidate" / DOMAIN_KEYS[domain] / "ensemble16.h5"
            with h5py.File(path, "r+") as handle:
                if handle.attrs.get("schema") != ENSEMBLE_SCHEMA or handle.attrs.get("v50_program_sha256") != PROGRAM_SHA256:
                    raise ValueError("V54 compatibility sampler metadata differs")
                del handle.attrs["v50_program_sha256"]
                handle.attrs.update({
                    "v54_program_sha256": PROGRAM_SHA256,
                    "threshold_selection": str(threshold_path.resolve()),
                    "threshold_selection_sha256": threshold_sha,
                    "train_mechanism_gate": str(train_gate_path.resolve()),
                    "train_mechanism_gate_sha256": train_gate_sha,
                    "train_mechanism_pass": True,
                    "tail_coefficient": TAIL_COEFFICIENT,
                    "structure_risk_unchanged_from_V50": True,
                })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("program", "repo", "cache", "checkpoint", "report", "preflight", "thresholds", "train-gate", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("cache", "checkpoint", "report", "preflight", "thresholds", "train-gate"):
        parser.add_argument(f"--{name}-sha256", required=True)
    args = parser.parse_args()
    sample_all(args.program, args.repo, args.cache, args.cache_sha256, args.checkpoint, args.checkpoint_sha256, args.report, args.report_sha256, args.preflight, args.preflight_sha256, args.thresholds, args.thresholds_sha256, args.train_gate, args.train_gate_sha256, args.out)


if __name__ == "__main__":
    main()
