#!/usr/bin/env python
"""Integrity-bound three-domain development gate for V17-E5."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import torch

from hong2021_v6_gate import field_gate
from hong2021_v14_edm import V17_E5_SCHEMA
from hong2021_v15_development_gate import (
    DOMAINS,
    _load_metrics,
    _validate_ensemble,
    canonical_digest,
    git_state,
    select_candidate_rows,
    sha256_file,
)
from hong2021_v17_edm import load_frozen_registry


SCHEMA = "hong2021-v17-integrity-bound-three-domain-decision-v1"
POWER_BANDS = ("0.3-1_h_mpc", "1-3_h_mpc", "3-6_h_mpc", "6-10.0531_h_mpc")
EXPECTED_MODE_COUNTS = [250, 6872, 49928, 454949]


def _validate_checkpoint(
    path: Path, *, step: int, registry_sha256: str
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != V17_E5_SCHEMA:
        raise ValueError("unexpected V17-E5 checkpoint schema")
    if int(checkpoint.get("step", -1)) != step:
        raise ValueError("checkpoint step differs from candidate step")
    if checkpoint.get("decoder_upsampling") != "nearest":
        raise ValueError("V17-E5 checkpoint did not return to nearest upsampling")
    if checkpoint.get("decoder_align_corners") is not None:
        raise ValueError("nearest V17 checkpoint has an align_corners setting")
    if checkpoint.get("edm_p_mean_mode") != "log_sigma_data_fraction":
        raise ValueError("V17-E5 checkpoint did not retain relative noise placement")
    if float(checkpoint.get("edm_p_mean_sigma_data_fraction", -1)) != 0.6:
        raise ValueError("V17-E5 checkpoint used the wrong sigma_data fraction")
    if float(checkpoint.get("edm_p_std", -1)) != 1.2:
        raise ValueError("V17-E5 checkpoint used the wrong p_std")
    tail = checkpoint.get("tail_weight_fit", {})
    if float(tail.get("inverse_probability_exponent", -1)) != 0.25:
        raise ValueError("V17-E5 checkpoint did not retain the E3 tail exponent")
    if float(tail.get("pre_normalization_maximum", -1)) != 10.0:
        raise ValueError("V17-E5 checkpoint did not retain the E3 tail maximum")
    loss = checkpoint.get("denoising_loss", {})
    if loss.get("coefficients") != {
        "unweighted": 1.0 / 3.0,
        "tail_weighted": 1.0 / 3.0,
        "band_balanced": 1.0 / 3.0,
    }:
        raise ValueError("V17-E5 checkpoint used the wrong loss coefficients")
    exact_loss = {
        "fft_transform": "full_complex_fftn",
        "fft_norm": "ortho",
        "fft_input_dtype": "float32",
        "fft_autocast_enabled": False,
        "dc_in_band_loss": False,
        "band_edges_h_mpc": [0.0, 1.0, 3.0, 6.0, "infinity"],
        "mode_counts": EXPECTED_MODE_COUNTS,
        "non_dc_modes": 511999,
        "grid": 80,
        "voxel_mpc_h": 0.3125,
        "parseval_relative_tolerance": 1.0e-4,
        "additional_rng_draws": 0,
    }
    for key, value in exact_loss.items():
        if loss.get(key) != value:
            raise ValueError(f"V17-E5 checkpoint loss metadata differs: {key}")
    if max(
        float(loss.get("parseval_self_check_relative_error", 1.0)),
        float(loss.get("mask_partition_self_check_relative_error", 1.0)),
    ) > 1.0e-4:
        raise ValueError("V17-E5 checkpoint failed the Fourier self-check")
    if checkpoint.get("experiment_registry_sha256") != registry_sha256:
        raise ValueError("checkpoint registry digest differs from current registry")
    if checkpoint.get("worktree_clean_at_launch") is not True:
        raise ValueError("checkpoint was not launched from a clean worktree")
    commit = checkpoint.get("code_commit_at_launch")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("checkpoint has no valid launch commit")
    return checkpoint


def _expected_indices(specification: list[int] | dict[str, str], repo: Path) -> list[int]:
    if isinstance(specification, list):
        indices = [int(value) for value in specification]
    else:
        path = repo / specification["path"]
        if sha256_file(path) != specification["sha256"]:
            raise ValueError("V17 development subset file differs from its frozen hash")
        indices = [int(value) for value in json.loads(path.read_text())["indices"]]
    if len(indices) != 16 or len(set(indices)) != 16 or min(indices) < 0:
        raise ValueError("V17 development subset must contain 16 unique indices")
    return indices


def _validate_source_indices(path: Path, expected: list[int]) -> None:
    with h5py.File(path, "r") as handle:
        actual = [int(value) for value in handle["source_index"][:]]
    if actual != expected:
        raise ValueError("ensemble source indices differ from the frozen V17 subset")


def _next_after_failure(final: dict[str, Any]) -> str:
    domains = final["domains"]
    swift_checks = domains["swift_dev"]["field_gate"]["checks"]
    only_swift_peak_void = (
        domains["tng"]["field_gate"]["pass"]
        and domains["simba_dev"]["field_gate"]["pass"]
        and not swift_checks["all_selected_peak_void_statistics_improve_deterministic"]
        and all(
            value for name, value in swift_checks.items()
            if name != "all_selected_peak_void_statistics_improve_deterministic"
        )
    )
    if only_swift_peak_void:
        return "stop_unopened_then_predeclare_all_band_target_free_conditional_scale"
    return "stop_unopened_and_obtain_new_independent_design_audit"


def evaluate(
    *, root: Path, training: Path, registry_path: Path, repo: Path,
    gate_code_commit: str,
) -> dict[str, Any]:
    repo = repo.resolve()
    registry = load_frozen_registry(registry_path, repo)
    registry_sha = sha256_file(registry_path)
    experiment = registry["e5_band_balanced_denoising"]
    steps = [int(value) for value in experiment["candidate_steps"]]
    if steps != [5000, 10000]:
        raise ValueError("candidate steps differ from the V17 registry")
    seed_map = {
        "tng": int(experiment["sampling_seeds"]["TNG100"]),
        "simba_dev": int(experiment["sampling_seeds"]["SIMBA"]),
        "swift_dev": int(experiment["sampling_seeds"]["Swift"]),
    }
    object_specifications = experiment["development_objects"]
    expected_indices = {
        "tng": _expected_indices(object_specifications["TNG100"], repo),
        "simba_dev": _expected_indices(object_specifications["SIMBA"], repo),
        "swift_dev": _expected_indices(object_specifications["Swift"], repo),
    }
    candidates = []
    for step in steps:
        checkpoint_path = training / "validation_checkpoints" / f"step_{step:06d}.pt"
        checkpoint = _validate_checkpoint(
            checkpoint_path, step=step, registry_sha256=registry_sha
        )
        domains = {}
        for domain in DOMAINS:
            domain_root = root / f"step_{step:06d}" / domain
            ensemble_path = domain_root / "ensemble16_steps40.h5"
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            _validate_ensemble(
                ensemble_path, checkpoint=checkpoint_path,
                checkpoint_schema=V17_E5_SCHEMA, step=step,
                seed=seed_map[domain],
            )
            _validate_source_indices(ensemble_path, expected_indices[domain])
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble_path.resolve():
                raise ValueError("metrics refer to a different ensemble")
            power = metrics["fourier_log_density"]["generated_total_power_over_truth"]
            domains[domain] = {
                "ensemble": str(ensemble_path.resolve()),
                "ensemble_sha256": sha256_file(ensemble_path),
                "metrics": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics),
                "report_only_total_power_over_truth": {
                    band: float(power[band]) for band in POWER_BANDS
                },
            }
        candidates.append({
            "step": step,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_code_commit": checkpoint["code_commit_at_launch"],
            "domains": domains,
            "all_three_pass": all(
                value["field_gate"]["pass"] for value in domains.values()
            ),
        })
    selected = select_candidate_rows(candidates)
    report = {
        "schema": SCHEMA,
        "experiment": "e5_band_balanced_denoising",
        "registry": str(registry_path.resolve()),
        "registry_sha256": registry_sha,
        "gate_code_commit": gate_code_commit,
        "worktree_clean_at_gate": True,
        "EAGLE_RefL0100N1504_used": False,
        "Astrid_used": False,
        "predeclared_steps": steps,
        "selection_rule": "smallest predeclared step passing all unchanged V6 field checks independently in TNG100, SIMBA development, and Swift development",
        "low_band_selection_role": "none; 0.3-1 and 1-3 h/Mpc are report-only",
        "candidates": candidates,
        "selected_step": None if selected is None else selected["step"],
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "development_pass": selected is not None,
        "next": (
            "freeze_exact_v17_hashes_before_astrid_one_shot"
            if selected is not None else _next_after_failure(candidates[-1])
        ),
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    commit, clean = git_state(args.repo.resolve())
    if not clean:
        raise RuntimeError("V17 gate requires a clean committed worktree")
    report = evaluate(
        root=args.root, training=args.training, registry_path=args.registry,
        repo=args.repo, gate_code_commit=commit,
    )
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if canonical_digest(existing) != existing.get("decision_digest_sha256"):
            raise RuntimeError("existing V17 decision has an invalid digest")
        if existing != report:
            raise RuntimeError("existing V17 decision differs from recomputed decision")
        print(json.dumps(existing, indent=2))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
