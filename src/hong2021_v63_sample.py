#!/usr/bin/env python
"""Sample the train-gate-approved V63 model with frozen development ranks."""
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
from hong2021_v50_train import PARAMETERS, SUPPORT_SHA256
from hong2021_v56_train import LIKELIHOOD_FAMILY, load_program as load_v56_program
from hong2021_v63_preflight import PROGRAM_SHA256, _path, load_program
from hong2021_v63_train import (
    CHECKPOINT_SCHEMA,
    MOMENT_COEFFICIENT,
    PREFLIGHT_SCHEMA,
    QUADRATURE_ORDER,
    REPORT_SCHEMA,
    STEPS,
    _is_ancestor,
)
from hong2021_v63_train_gate import SCHEMA as TRAIN_GATE_SCHEMA
from hong2021_v63_train_gate import _load_fit as load_verified_fit


ENSEMBLE_SCHEMA = "hong2021-v63-conditional-log-physical-moment-ensemble-v1"
METHOD = "train_only_conditional_log_physical_moment_bounded_mixture_empirical_rank_copula"
ARMS = ("bounded_query_local_mixture_copula", "rolled_parameter_control")


def _strict_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V63 sample {label} hash differs")
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _base_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    program, v35, _, _, _, _, _ = load_program(path, repo)
    v56, _, inherited = load_v56_program(
        _path(repo, program["frozen_inputs"]["v56_program"]), repo
    )
    effective = dict(program)
    effective["inherited_inputs"] = v56["inherited_inputs"]
    return effective, v35, inherited


def _load_fit_factory(
    repo: Path,
    threshold_sha: str,
    grid_sha: str,
    train_gate_path: Path,
    train_gate_sha: str,
    boundaries: dict[str, float],
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
        if sha256_file(preflight_path) != preflight_sha:
            raise ValueError("V63 sample preflight hash differs")
        train_gate = _strict_json(train_gate_path, train_gate_sha, "train gate")
        if (
            train_gate.get("schema") != TRAIN_GATE_SCHEMA
            or train_gate.get("program_sha256") != PROGRAM_SHA256
            or train_gate.get("train_mechanism_pass") is not True
            or train_gate.get("checkpoint_sha256") != checkpoint_sha
            or train_gate.get("training_report_sha256") != report_sha
            or train_gate.get("preflight_sha256") != preflight_sha
            or train_gate.get("grid_sha256") != grid_sha
            or train_gate.get("development_accessed") is not False
            or train_gate.get("training_or_refit_performed_by_gate") is not False
            or train_gate.get("historical_EAGLE_accessed") is not False
            or train_gate.get("independent_gate_locked") is not True
            or canonical_digest(train_gate)
            != train_gate.get("decision_digest_sha256")
            or not _is_ancestor(repo, str(train_gate.get("code_commit")), commit)
        ):
            raise ValueError("V63 sample train-gate authorization differs")
        model, checkpoint = load_verified_fit(
            checkpoint_path,
            checkpoint_sha,
            report_path,
            report_sha,
            grid_sha,
            threshold_sha,
            preflight_sha,
            cache_sha,
            SUPPORT_SHA256,
            boundaries,
            repo,
            commit,
        )
        del model
        report = _strict_json(report_path, report_sha, "training report")
        if (
            checkpoint.get("schema") != CHECKPOINT_SCHEMA
            or checkpoint.get("step") != STEPS
            or checkpoint.get("parameters") != PARAMETERS
            or checkpoint.get("likelihood_family") != LIKELIHOOD_FAMILY
            or checkpoint.get("open_standardized_support")
            != [LOWER_SUPPORT, UPPER_SUPPORT]
            or report.get("schema") != REPORT_SCHEMA
            or report.get("checkpoint_sha256") != checkpoint_sha
        ):
            raise ValueError("V63 sample fit metadata differs")
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
    grid_path: Path,
    grid_sha: str,
    train_gate_path: Path,
    train_gate_sha: str,
    output_root: Path,
) -> None:
    repo = repo.resolve()
    program, _, _, _, _, _, _ = load_program(program_path, repo)
    frozen = program["frozen_inputs"]
    bindings = (
        (threshold_path.resolve(), _path(repo, frozen["v54_threshold_selection"]), threshold_sha, frozen["v54_threshold_selection_sha256"]),
        (grid_path.resolve(), _path(repo, frozen["v56_grid"]), grid_sha, frozen["v56_grid_sha256"]),
    )
    if any(a != b or c != d or sha256_file(a) != c for a, b, c, d in bindings):
        raise ValueError("V63 sample frozen threshold or grid differs")
    boundaries = {
        domain: float(program["sealed_q99_9_backbone_boundaries"][domain])
        for domain in DOMAIN_ORDER
    }
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
        "load_program": _base_program,
        "load_fit": _load_fit_factory(
            repo,
            threshold_sha,
            grid_sha,
            train_gate_path,
            train_gate_sha,
            boundaries,
        ),
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
            path = (
                output_root
                / arm
                / "development_candidate"
                / DOMAIN_KEYS[domain]
                / "ensemble16.h5"
            )
            with h5py.File(path, "r+") as handle:
                if (
                    handle.attrs.get("schema") != ENSEMBLE_SCHEMA
                    or handle.attrs.get("v50_program_sha256") != PROGRAM_SHA256
                ):
                    raise ValueError("V63 compatibility sampler metadata differs")
                del handle.attrs["v50_program_sha256"]
                handle.attrs.update(
                    {
                        "v63_program_sha256": PROGRAM_SHA256,
                        "threshold_selection": str(threshold_path.resolve()),
                        "threshold_selection_sha256": threshold_sha,
                        "grid": str(grid_path.resolve()),
                        "grid_sha256": grid_sha,
                        "train_mechanism_gate": str(train_gate_path.resolve()),
                        "train_mechanism_gate_sha256": train_gate_sha,
                        "train_mechanism_pass": True,
                        "moment_coefficient": MOMENT_COEFFICIENT,
                        "moment_quadrature_order": QUADRATURE_ORDER,
                        "conditional_moment_objective": True,
                        "structure_risk_unchanged_from_V50": True,
                        "development_sampling_authorized_by_train_gate": True,
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "program", "repo", "cache", "checkpoint", "report", "preflight",
        "thresholds", "grid", "train-gate", "out",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "cache", "checkpoint", "report", "preflight", "thresholds", "grid", "train-gate",
    ):
        parser.add_argument(f"--{name}-sha256", required=True)
    args = parser.parse_args()
    sample_all(
        args.program, args.repo, args.cache, args.cache_sha256,
        args.checkpoint, args.checkpoint_sha256, args.report, args.report_sha256,
        args.preflight, args.preflight_sha256, args.thresholds, args.thresholds_sha256,
        args.grid, args.grid_sha256, args.train_gate, args.train_gate_sha256, args.out,
    )


if __name__ == "__main__":
    main()
