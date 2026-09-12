#!/usr/bin/env python
"""Sample the frozen V52 no-risk bounded likelihood with unchanged donor ranks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_development_gate import git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_KEYS, DOMAIN_ORDER
from hong2021_v31_copula import conditional_forward, load_model
from hong2021_v35_spectrum_phase import _backbone, _open_split
from hong2021_v37_query_alignment import _selection_arrays
from hong2021_v48_sample import _new_ensemble
from hong2021_v50_network import (
    LOWER_SUPPORT,
    UPPER_SUPPORT,
    LocalMixtureUNet,
    bounded_mixture_cdf,
    bounded_mixture_inverse,
    parameter_count,
)
from hong2021_v52_train import (
    CHECKPOINT_SCHEMA,
    LIKELIHOOD_FAMILY,
    PARAMETERS,
    PREFLIGHT_SCHEMA,
    PROGRAM_SHA256,
    REPORT_SCHEMA,
    SUPPORT_SHA256,
    _verified_json,
    load_cache,
    load_program,
    no_risk_condition_cube,
)


ENSEMBLE_SCHEMA = "hong2021-v52-matched-no-risk-bounded-mixture-ensemble-v1"
METHOD = "train_only_no_risk_bounded_query_local_logit_Gaussian_mixture_empirical_rank_copula"
ARMS = ("no_risk_query_local_mixture_copula", "rolled_parameter_control")
PARAMETER_ROLL = (16, 8, 4)


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
        raise ValueError("V52 checkpoint hash differs")
    report = _verified_json(report_path, report_sha, "report")
    preflight = _verified_json(preflight_path, preflight_sha, "preflight")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    exact_support = [LOWER_SUPPORT, UPPER_SUPPORT]
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("program_sha256") != PROGRAM_SHA256
        or checkpoint.get("code_commit") != commit
        or checkpoint.get("step") != 12_000
        or checkpoint.get("parameters") != PARAMETERS
        or checkpoint.get("likelihood_family") != LIKELIHOOD_FAMILY
        or checkpoint.get("open_standardized_support") != exact_support
        or checkpoint.get("support_selection_sha256") != SUPPORT_SHA256
        or checkpoint.get("conditioning_cache_sha256") != cache_sha
        or checkpoint.get("preflight_sha256") != preflight_sha
        or checkpoint.get("risk_channel_exact_standardized_zero") is not True
        or checkpoint.get("spatial_rank_transport") is not False
        or checkpoint.get("sample_clipping") is not False
        or checkpoint.get("component_scale_cap") is not False
    ):
        raise ValueError("V52 checkpoint metadata differs")
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("checkpoint_sha256") != checkpoint_sha
        or report.get("conditioning_cache_sha256") != cache_sha
        or report.get("preflight_sha256") != preflight_sha
        or report.get("likelihood_family") != LIKELIHOOD_FAMILY
        or report.get("support_selection_sha256") != SUPPORT_SHA256
        or report.get("open_standardized_support") != exact_support
        or report.get("risk_channel_exact_standardized_zero") is not True
        or report.get("sample_clipping") is not False
        or report.get("component_scale_cap") is not False
        or report.get("code_commit") != commit
        or report.get(
            "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection"
        )
        is not False
        or preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("code_commit") != commit
        or preflight.get("support_selection_sha256") != SUPPORT_SHA256
        or preflight.get("risk_channel_exact_standardized_zero") is not True
    ):
        raise ValueError("V52 report or preflight binding differs")
    if sha256_file(cache_path) != cache_sha:
        raise ValueError("V52 cache hash differs")
    return checkpoint, report


@torch.inference_mode()
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
    output_root: Path,
) -> None:
    program, v35, _ = load_program(program_path, repo.resolve())
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V52 sampling requires a clean worktree")
    if output_root.exists():
        raise FileExistsError("V52 refuses existing output root")
    checkpoint, _ = load_fit(
        checkpoint_path,
        checkpoint_sha,
        report_path,
        report_sha,
        cache_path,
        cache_sha,
        preflight_path,
        preflight_sha,
        commit,
    )
    prepared = load_cache(cache_path, cache_sha, commit)
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V52 sampling requires the Lageunha Ada GPU")
    device = torch.device("cuda")
    model = LocalMixtureUNet().to(device)
    if parameter_count(model) != PARAMETERS:
        raise RuntimeError("V52 sampling architecture differs")
    model.load_state_dict(checkpoint["ema_state_dict"])
    model.eval()
    copula = load_model(
        Path(program["inherited_inputs"]["conditional_copula_artifact"]),
        program["inherited_inputs"]["conditional_copula_artifact_sha256"],
    )
    selections = _selection_arrays(v35)
    train = {
        domain: _open_split(v35["development_domains"][domain], "train")
        for domain in DOMAIN_ORDER
    }
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    try:
        for domain in DOMAIN_ORDER:
            row = v35["development_domains"][domain]
            indices = np.asarray(selections[domain]["source_index"], dtype=np.int64)
            query_data, query_cache = _open_split(row, "validation")
            handles: dict[str, h5py.File] = {}
            datasets: dict[str, dict[str, h5py.Dataset]] = {}
            partials: dict[str, Path] = {}
            maxima = {arm: {"dc": 0.0, "cdf": 0.0} for arm in ARMS}
            bounded_extrema = {
                arm: {"minimum": float("inf"), "maximum": -float("inf")}
                for arm in ARMS
            }
            try:
                for arm in ARMS:
                    path = (
                        output_root
                        / arm
                        / "development_candidate"
                        / DOMAIN_KEYS[domain]
                        / "ensemble16.h5"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partials[arm] = path.with_suffix(path.suffix + ".partial")
                    handles[arm] = h5py.File(partials[arm], "w")
                    datasets[arm] = _new_ensemble(handles[arm])
                    for name, value in selections[domain].items():
                        handles[arm].create_dataset(name, data=value)
                for object_index, query_index in enumerate(indices):
                    condition, _, backbone = no_risk_condition_cube(
                        query_data,
                        query_cache,
                        prepared,
                        domain,
                        "validation",
                        int(query_index),
                    )
                    if np.count_nonzero(condition[5]) != 0:
                        raise RuntimeError("V52 sample query risk channel differs")
                    actual_parameter = model(
                        torch.from_numpy(condition[None]).to(device)
                    ).float()
                    parameter_by_arm = {
                        "no_risk_query_local_mixture_copula": actual_parameter,
                        "rolled_parameter_control": torch.roll(
                            actual_parameter,
                            shifts=PARAMETER_ROLL,
                            dims=(-3, -2, -1),
                        ),
                    }
                    ranks = []
                    digests = []
                    for member in range(16):
                        donor_source = DOMAIN_ORDER[
                            int(selections[domain]["donor_source"][object_index, member])
                        ]
                        donor_index = int(
                            selections[domain]["donor_index"][object_index, member]
                        )
                        isometry = int(
                            selections[domain]["donor_isometry"][object_index, member]
                        )
                        donor_data, donor_cache = train[donor_source]
                        donor_backbone = _backbone(donor_cache, donor_index)[None]
                        donor_truth = np.asarray(
                            donor_data["target"][donor_index], dtype=np.float32
                        )
                        rank = conditional_forward(
                            donor_truth - donor_backbone, donor_backbone, copula
                        )
                        axes, reflections = CUBE_ISOMETRIES[isometry]
                        rank = apply_cube_isometry(rank, axes, reflections)
                        ranks.append(rank)
                        digests.append(
                            np.frombuffer(
                                hashlib.sha256(
                                    np.sort(rank.reshape(-1)).tobytes()
                                ).digest(),
                                dtype=np.uint8,
                            )
                        )
                    rank_tensor = torch.from_numpy(np.stack(ranks)).to(device)
                    for arm in ARMS:
                        parameters = parameter_by_arm[arm].expand(
                            16, -1, -1, -1, -1
                        )
                        standardized = bounded_mixture_inverse(parameters, rank_tensor)
                        if not bool(
                            torch.all(
                                (standardized > LOWER_SUPPORT)
                                & (standardized < UPPER_SUPPORT)
                            ).cpu()
                        ):
                            raise RuntimeError("V52 sample left frozen support")
                        cdf_error = torch.amax(
                            torch.abs(
                                bounded_mixture_cdf(parameters, standardized)
                                - rank_tensor.clamp(1.0e-7, 1.0 - 1.0e-7)
                            ),
                            dim=(-4, -3, -2, -1),
                        )
                        standardized_numpy = standardized.cpu().numpy()
                        bounded_extrema[arm]["minimum"] = min(
                            bounded_extrema[arm]["minimum"],
                            float(standardized_numpy.min()),
                        )
                        bounded_extrema[arm]["maximum"] = max(
                            bounded_extrema[arm]["maximum"],
                            float(standardized_numpy.max()),
                        )
                        residual = standardized_numpy * target_std + target_mean
                        residual -= residual.mean(
                            axis=(-3, -2, -1), keepdims=True, dtype=np.float64
                        )
                        dc = np.abs(
                            residual.mean(axis=(-3, -2, -1), dtype=np.float64)
                        )
                        sample = backbone[None] + residual
                        if not np.isfinite(sample).all() or float(dc.max()) > 1.0e-7:
                            raise RuntimeError("V52 sampled field or DC differs")
                        datasets[arm]["sample"][object_index] = sample.astype(np.float32)
                        datasets[arm]["conditional_rank_multiset_sha256"][
                            object_index
                        ] = np.stack(digests)
                        datasets[arm]["maximum_inverse_CDF_error"][object_index] = (
                            cdf_error.cpu().numpy()
                        )
                        maxima[arm]["dc"] = max(maxima[arm]["dc"], float(dc.max()))
                        maxima[arm]["cdf"] = max(
                            maxima[arm]["cdf"], float(cdf_error.max().cpu())
                        )
                    amplitude = float(
                        prepared[f"{domain}/validation/object_amplitude"][int(query_index)]
                    )
                    truth = np.asarray(
                        query_data["target"][int(query_index)], dtype=np.float32
                    )
                    for arm in ARMS:
                        datasets[arm]["conditional_mean"][object_index] = backbone
                        datasets[arm]["truth"][object_index] = truth
                        datasets[arm]["object_amplitude_prediction"][object_index] = amplitude
                    print(f"[v52-sample] {domain} {object_index + 1}/16", flush=True)
                for arm in ARMS:
                    handles[arm].attrs.update(
                        {
                            "schema": ENSEMBLE_SCHEMA,
                            "method": METHOD,
                            "arm": arm,
                            "v52_program_sha256": PROGRAM_SHA256,
                            "checkpoint": str(checkpoint_path.resolve()),
                            "checkpoint_sha256": checkpoint_sha,
                            "training_report": str(report_path.resolve()),
                            "training_report_sha256": report_sha,
                            "conditioning_cache": str(cache_path.resolve()),
                            "conditioning_cache_sha256": cache_sha,
                            "preflight": str(preflight_path.resolve()),
                            "preflight_sha256": preflight_sha,
                            "support_selection_sha256": SUPPORT_SHA256,
                            "standardized_support_lower": LOWER_SUPPORT,
                            "standardized_support_upper": UPPER_SUPPORT,
                            "minimum_pre_DC_standardized_sample": bounded_extrema[arm][
                                "minimum"
                            ],
                            "maximum_pre_DC_standardized_sample": bounded_extrema[arm][
                                "maximum"
                            ],
                            "parent_selection": str(
                                Path(row["phase_object_selection"]).resolve()
                            ),
                            "parent_selection_sha256": row[
                                "phase_object_selection_sha256"
                            ],
                            "ensemble_members": 16,
                            "mixture_components": 5,
                            "likelihood_family": LIKELIHOOD_FAMILY,
                            "mixture_bisection_steps": 28,
                            "parameter_roll": json.dumps(PARAMETER_ROLL),
                            "structure_risk_channel_exact_standardized_zero": True,
                            "parameters_spatially_rolled": arm
                            == "rolled_parameter_control",
                            "conditional_rank_spatial_permutation": False,
                            "conditional_rank_multiset_preserved": True,
                            "global_residual_scale": 1.0,
                            "object_amplitude_post_calibration": False,
                            "maximum_absolute_residual_dc": maxima[arm]["dc"],
                            "maximum_inverse_CDF_error": maxima[arm]["cdf"],
                            "diagnostic_k_h_mpc": 1.0,
                            "validation_truth_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
                            "hard_density_or_residual_clipping": False,
                            "sample_clipping": False,
                            "component_scale_cap": False,
                            "donor_translation": False,
                            "donor_reselection": False,
                            "posthoc_Ak_used": False,
                            "worktree_clean_at_sampling": clean,
                            "sampling_code_commit": commit,
                            "Astrid_accessed": False,
                            "historical_EAGLE_accessed": False,
                            "independent_gate_locked": True,
                            "complete": True,
                        }
                    )
            finally:
                for handle in handles.values():
                    handle.close()
                query_data.close()
                query_cache.close()
            for arm in ARMS:
                os.replace(partials[arm], partials[arm].with_suffix(""))
    finally:
        for data, cache in train.values():
            data.close()
            cache.close()
        prepared.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sample_all(
        args.program,
        args.repo,
        args.cache,
        args.cache_sha256,
        args.checkpoint,
        args.checkpoint_sha256,
        args.report,
        args.report_sha256,
        args.preflight,
        args.preflight_sha256,
        args.out,
    )


if __name__ == "__main__":
    main()
