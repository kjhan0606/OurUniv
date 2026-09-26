#!/usr/bin/env python
"""Sample one frozen fresh V72 SQT stage after exact authorization."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

import h5py
import numpy as np
import torch

from hong2021_v15_development_gate import git_state
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v48_train import condition_cube
from hong2021_v50_network import (
    bounded_mixture_cdf,
    bounded_mixture_inverse,
    standard_normal_cdf,
)
from hong2021_v63_train import _is_ancestor
from hong2021_v70_latent_cache import _frozen_marginal
from hong2021_v70_train_gate import (
    _load_fit,
    heun_sample,
    load_program as load_v70_train_gate_program,
    project_residual_dc,
    sigma_schedule,
)
from hong2021_v72_sqt import (
    ARMS,
    CANDIDATE,
    CONTROL,
    DOMAIN_KEYS,
    DOMAIN_ORDER,
    ENSEMBLE_SCHEMA,
    METHOD,
    PROGRAM_FREEZE_COMMIT,
    PROGRAM_SHA256,
    RAW,
    authorize_parent_evidence,
    conditioning_strata,
    load_program,
    spatial_quantile_transport,
    stage_selection,
    validate_preflight,
    validate_stage_A_pass,
)


TRAIN_GATE_PROGRAM = Path("config/hong2021_v70_train_joint_structure_gate_program.json")


def _new_ensemble(handle: h5py.File) -> dict[str, h5py.Dataset]:
    return {
        "sample": handle.create_dataset(
            "sample", shape=(16, 16, 1, 64, 64, 64), dtype="f4",
            chunks=(1, 1, 1, 64, 64, 64), compression="lzf",
        ),
        "conditional_mean": handle.create_dataset(
            "conditional_mean", shape=(16, 1, 64, 64, 64), dtype="f4",
            compression="lzf",
        ),
        "truth": handle.create_dataset(
            "truth", shape=(16, 1, 64, 64, 64), dtype="f4", compression="lzf",
        ),
        "source_index": handle.create_dataset("source_index", shape=(16,), dtype="i8"),
        "object_amplitude_prediction": handle.create_dataset(
            "object_amplitude_prediction", shape=(16,), dtype="f4"
        ),
        "initial_latent_sha256": handle.create_dataset(
            "initial_latent_sha256", shape=(16, 16, 32), dtype="u1"
        ),
        "maximum_inverse_CDF_error": handle.create_dataset(
            "maximum_inverse_CDF_error", shape=(16, 16), dtype="f4"
        ),
        "pre_inverse_stratum_multiset_equal": handle.create_dataset(
            "pre_inverse_stratum_multiset_equal", shape=(16,), dtype="u1"
        ),
        "maximum_pre_inverse_stratum_multiset_error": handle.create_dataset(
            "maximum_pre_inverse_stratum_multiset_error", shape=(16,), dtype="f4"
        ),
        "marginal_tied_voxel_fraction": handle.create_dataset(
            "marginal_tied_voxel_fraction", shape=(16,), dtype="f4"
        ),
        "rank_disagreement_fraction_excluding_marginal_ties": handle.create_dataset(
            "rank_disagreement_fraction_excluding_marginal_ties",
            shape=(16,), dtype="f4",
        ),
        "maximum_physical_pre_DC_sorted_residual_difference": handle.create_dataset(
            "maximum_physical_pre_DC_sorted_residual_difference",
            shape=(16,), dtype="f4",
        ),
        "maximum_physical_post_DC_sorted_residual_difference": handle.create_dataset(
            "maximum_physical_post_DC_sorted_residual_difference",
            shape=(16,), dtype="f4",
        ),
    }


def _validate_fresh_rows(program: dict, v35: dict) -> None:
    frozen = program["frozen_inputs"]
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        for suffix, key in (("data", "validation_data"), ("cache", "validation_cache")):
            expected_path = Path(frozen[f"{domain}_validation_{suffix}"]).resolve()
            expected_sha = frozen[f"{domain}_validation_{suffix}_sha256"]
            if (
                Path(row[key]).resolve() != expected_path
                or row[f"{key}_sha256"] != expected_sha
            ):
                raise ValueError(f"V72 {domain} frozen validation row differs")


@torch.inference_mode()
def sample_stage(
    program_path: Path,
    repo: Path,
    preflight_path: Path,
    preflight_sha: str,
    stage: str,
    output_root: Path,
    stage_A_decision_path: Path | None = None,
    stage_A_decision_sha: str | None = None,
) -> None:
    repo = repo.resolve()
    program = load_program(program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V72 sampling requires clean frozen Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V72 sampling requires the Lageunha Ada GPU")
    evidence = authorize_parent_evidence(program, repo, commit)
    preflight = validate_preflight(preflight_path, preflight_sha, repo, commit)
    if stage == "B":
        if stage_A_decision_path is None or stage_A_decision_sha is None:
            raise ValueError("V72 stage B requires the exact stage-A pass")
        validate_stage_A_pass(
            program, stage_A_decision_path.resolve(), stage_A_decision_sha,
            preflight_sha, repo, commit,
        )
    elif stage != "A" or stage_A_decision_path is not None or stage_A_decision_sha is not None:
        raise ValueError("V72 stage authorization arguments differ")
    expected_output = Path(program["output_roots"][f"stage_{stage}"]).resolve()
    if output_root.resolve() != expected_output or output_root.exists():
        raise FileExistsError("V72 refuses a differing or existing stage output")
    if stage == "A" and Path(program["output_roots"]["stage_B"]).exists():
        raise FileExistsError("V72 stage B exists before stage A")

    train_program, v70, v35, _ = load_v70_train_gate_program(
        (repo / TRAIN_GATE_PROGRAM).resolve(), repo
    )
    _validate_fresh_rows(program, v35)
    device = torch.device("cuda")
    model, training_report, checkpoint_sha, report_sha = _load_fit(
        train_program, repo, commit, device
    )
    frozen = program["frozen_inputs"]
    if (
        checkpoint_sha != frozen["v70_checkpoint_sha256"]
        or report_sha != frozen["v70_training_report_sha256"]
    ):
        raise ValueError("V72 fixed V70 fit differs")
    marginal, inherited_v35, prepared = _frozen_marginal(v70, repo, commit, device)
    if inherited_v35["development_domains"] != v35["development_domains"]:
        raise ValueError("V72 frozen marginal source differs")
    selection = stage_selection(program, stage)
    sampling = program["fixed_sampling"]
    seed = int(sampling[f"stage_{stage}_noise_seed"])
    if (
        int(sampling["members_per_query"]) != 16
        or int(sampling["inference_batch"]) != 4
        or int(sampling["sigma_steps"]) != 40
        or float(sampling["sigma_minimum"]) != 0.002
        or float(sampling["sigma_maximum"]) != 40.0
        or float(sampling["rho"]) != 7.0
        or float(sampling["stochastic_churn"]) != 0.0
    ):
        raise ValueError("V72 frozen sampler differs")
    schedule = sigma_schedule(
        40, 0.002, 40.0, 7.0, device=device
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    maximum_dc = {arm: 0.0 for arm in ARMS}
    maximum_inverse = {arm: 0.0 for arm in ARMS}
    maximum_tie = 0.0
    maximum_pre_physical = 0.0
    maximum_post_physical = 0.0
    peak = 0
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for domain in DOMAIN_ORDER:
            row = v35["development_domains"][domain]
            indices = np.asarray(selection[domain], dtype=np.int64)
            data, cache = _open_split(row, "validation")
            handles: dict[str, h5py.File] = {}
            datasets: dict[str, dict[str, h5py.Dataset]] = {}
            partials: dict[str, Path] = {}
            try:
                for arm in ARMS:
                    path = (
                        output_root / arm / "fresh_candidate"
                        / DOMAIN_KEYS[domain] / "ensemble16.h5"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partials[arm] = path.with_suffix(path.suffix + ".partial")
                    handles[arm] = h5py.File(partials[arm], "w")
                    datasets[arm] = _new_ensemble(handles[arm])
                    datasets[arm]["source_index"][:] = indices
                for object_position, query_index in enumerate(indices):
                    condition, _, backbone = condition_cube(
                        data, cache, prepared, domain, "validation", int(query_index)
                    )
                    truth = np.asarray(data["target"][int(query_index)], dtype=np.float32)
                    condition_tensor = torch.from_numpy(condition[None]).to(device)
                    parameters = marginal(condition_tensor).float()
                    innovations: list[torch.Tensor] = []
                    raw_parts: list[torch.Tensor] = []
                    digests: list[np.ndarray] = []
                    for start in range(0, 16, 4):
                        innovation = torch.randn(
                            (4, 1, 64, 64, 64), device=device, generator=generator
                        )
                        for row_noise in innovation.detach().cpu().numpy():
                            digests.append(
                                np.frombuffer(
                                    hashlib.sha256(row_noise.tobytes()).digest(),
                                    dtype=np.uint8,
                                )
                            )
                        expanded = condition_tensor.expand(4, -1, -1, -1, -1)
                        raw_parts.append(heun_sample(model, expanded, innovation, schedule))
                        innovations.append(innovation)
                    innovation_all = torch.cat(innovations, dim=0)
                    raw_latent = torch.cat(raw_parts, dim=0)
                    score = torch.from_numpy(
                        (np.asarray(backbone[0], dtype=np.float32) + target_mean)[None]
                    ).to(device)
                    positions = conditioning_strata(score)
                    sqt_latent, diagnostics = spatial_quantile_transport(
                        raw_latent, innovation_all, positions
                    )
                    tie = float(diagnostics["marginal_tied_voxel_fraction"])
                    maximum_tie = max(maximum_tie, tie)
                    if tie > float(
                        program["numerical_requirements"]
                        ["maximum_marginal_tied_voxel_fraction"]
                    ):
                        raise RuntimeError("V72 marginal tie fraction exceeds freeze")
                    latents = {
                        CANDIDATE: sqt_latent,
                        RAW: raw_latent,
                        CONTROL: innovation_all,
                    }
                    parameter_batch = parameters.expand(16, -1, -1, -1, -1)
                    pre_dc: dict[str, np.ndarray] = {}
                    inverse_errors: dict[str, np.ndarray] = {}
                    for arm in ARMS:
                        uniform = standard_normal_cdf(latents[arm])
                        standardized = bounded_mixture_inverse(parameter_batch, uniform)
                        error = torch.amax(
                            torch.abs(
                                bounded_mixture_cdf(parameter_batch, standardized)
                                - uniform.clamp(1.0e-7, 1.0 - 1.0e-7)
                            ),
                            dim=(-4, -3, -2, -1),
                        )
                        inverse_errors[arm] = error.cpu().numpy()
                        maximum_inverse[arm] = max(
                            maximum_inverse[arm], float(error.max().cpu())
                        )
                        pre_dc[arm] = (
                            standardized.cpu().numpy().astype(np.float64)
                            * target_std + target_mean
                        )
                    pre_physical = float(
                        np.max(
                            np.abs(
                                np.sort(pre_dc[CANDIDATE], axis=None)
                                - np.sort(pre_dc[CONTROL], axis=None)
                            )
                        )
                    )
                    projected: dict[str, np.ndarray] = {}
                    for arm in ARMS:
                        projected[arm], dc = project_residual_dc(pre_dc[arm])
                        maximum_dc[arm] = max(maximum_dc[arm], dc)
                    post_physical = float(
                        np.max(
                            np.abs(
                                np.sort(projected[CANDIDATE], axis=None).astype(np.float64)
                                - np.sort(projected[CONTROL], axis=None).astype(np.float64)
                            )
                        )
                    )
                    maximum_pre_physical = max(maximum_pre_physical, pre_physical)
                    maximum_post_physical = max(maximum_post_physical, post_physical)
                    shared = {
                        "pre_inverse_stratum_multiset_equal": int(
                            bool(diagnostics["pre_inverse_stratum_multiset_equal"])
                        ),
                        "maximum_pre_inverse_stratum_multiset_error": float(
                            diagnostics["maximum_pre_inverse_stratum_multiset_error"]
                        ),
                        "marginal_tied_voxel_fraction": tie,
                        "rank_disagreement_fraction_excluding_marginal_ties": float(
                            diagnostics[
                                "rank_disagreement_fraction_excluding_marginal_ties"
                            ]
                        ),
                        "maximum_physical_pre_DC_sorted_residual_difference": pre_physical,
                        "maximum_physical_post_DC_sorted_residual_difference": post_physical,
                    }
                    for arm in ARMS:
                        sample = (backbone[None] + projected[arm]).astype(np.float32)
                        if sample.shape != (16, 1, 64, 64, 64) or not np.isfinite(sample).all():
                            raise RuntimeError("V72 generated field differs")
                        datasets[arm]["sample"][object_position] = sample
                        datasets[arm]["conditional_mean"][object_position] = backbone
                        datasets[arm]["truth"][object_position] = truth
                        datasets[arm]["object_amplitude_prediction"][object_position] = float(
                            prepared[f"{domain}/validation/object_amplitude"][int(query_index)]
                        )
                        datasets[arm]["initial_latent_sha256"][object_position] = np.stack(digests)
                        datasets[arm]["maximum_inverse_CDF_error"][object_position] = inverse_errors[arm]
                        for name, value in shared.items():
                            datasets[arm][name][object_position] = value
                    print(
                        f"[v72-stage-{stage}] {domain} {object_position + 1}/16 "
                        f"tie={tie:.3e} post={post_physical:.3e}",
                        flush=True,
                    )
                peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
                for arm in ARMS:
                    method = {
                        CANDIDATE: METHOD,
                        RAW: "raw_fixed_step30000_V70_latent_spatial_score_control",
                        CONTROL: "independent_voxel_V63_marginal_control",
                    }[arm]
                    handles[arm].attrs.update(
                        {
                            "schema": ENSEMBLE_SCHEMA,
                            "method": method,
                            "arm": arm,
                            "stage": stage,
                            "v72_program_sha256": PROGRAM_SHA256,
                            "preflight": str(preflight_path.resolve()),
                            "preflight_sha256": preflight_sha,
                            "stage_A_decision": (
                                str(stage_A_decision_path.resolve())
                                if stage_A_decision_path is not None else ""
                            ),
                            "stage_A_decision_sha256": stage_A_decision_sha or "",
                            "v71_terminal_seal_sha256": evidence["v71_terminal_seal_sha256"],
                            "checkpoint": str(Path(frozen["v70_checkpoint"]).resolve()),
                            "checkpoint_sha256": checkpoint_sha,
                            "training_report": str(Path(frozen["v70_training_report"]).resolve()),
                            "training_report_sha256": report_sha,
                            "training_initial_code_commit": training_report["initial_code_commit"],
                            "source_data": str(Path(row["validation_data"]).resolve()),
                            "source_data_sha256": row["validation_data_sha256"],
                            "source_cache": str(Path(row["validation_cache"]).resolve()),
                            "source_cache_sha256": row["validation_cache_sha256"],
                            "ensemble_members": 16,
                            "conditioning_strata": 16,
                            "voxels_per_stratum": 16384,
                            "noise_seed": seed,
                            "sampler_steps": 40,
                            "sigma_minimum": 0.002,
                            "sigma_maximum": 40.0,
                            "rho": 7.0,
                            "stochastic_churn": 0.0,
                            "diagnostic_k_h_mpc": 1.0,
                            "candidate_arm": arm == CANDIDATE,
                            "diagnostic_control_may_affect_selection": False,
                            "unequal_sample_global_maximum_used": False,
                            "sample_clipping": False,
                            "posthoc_Ak_used": False,
                            "gradient_computed": False,
                            "optimizer_constructed": False,
                            "optimizer_step_performed": False,
                            "maximum_absolute_residual_dc": maximum_dc[arm],
                            "maximum_inverse_CDF_error": maximum_inverse[arm],
                            "maximum_marginal_tied_voxel_fraction": maximum_tie,
                            "maximum_physical_pre_DC_sorted_residual_difference": maximum_pre_physical,
                            "maximum_physical_post_DC_sorted_residual_difference": maximum_post_physical,
                            "worktree_clean_at_sampling": clean,
                            "sampling_code_commit": commit,
                            "Astrid_accessed": False,
                            "historical_EAGLE_accessed": False,
                            "independent_EAGLE_accessed": False,
                            "independent_gate_locked": True,
                            "complete": True,
                        }
                    )
            finally:
                for handle in handles.values():
                    handle.close()
                data.close()
                cache.close()
            for arm in ARMS:
                os.replace(partials[arm], partials[arm].with_suffix(""))
    finally:
        prepared.close()
    numerical = program["numerical_requirements"]
    if (
        max(maximum_dc.values()) > float(numerical["maximum_absolute_residual_DC"])
        or max(maximum_inverse.values()) > float(numerical["maximum_inverse_CDF_error"])
        or peak >= int(numerical["peak_Ada_allocation_bytes"])
    ):
        raise RuntimeError("V72 sampling numerical tolerance differs")
    print(
        json.dumps(
            {
                "status": f"complete_single_V72_SQT_stage_{stage}_sampling",
                "preflight_sha256": preflight_sha,
                "stage_A_decision_sha256": stage_A_decision_sha,
                "maximum_absolute_residual_dc": maximum_dc,
                "maximum_inverse_CDF_error": maximum_inverse,
                "maximum_marginal_tied_voxel_fraction": maximum_tie,
                "maximum_physical_pre_DC_sorted_residual_difference": maximum_pre_physical,
                "maximum_physical_post_DC_sorted_residual_difference": maximum_post_physical,
                "peak_allocated_bytes": peak,
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--stage", choices=("A", "B"), required=True)
    parser.add_argument("--stage-A-decision", type=Path)
    parser.add_argument("--stage-A-decision-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sample_stage(
        args.program, args.repo, args.preflight, args.preflight_sha256,
        args.stage, args.out, args.stage_A_decision,
        args.stage_A_decision_sha256,
    )


if __name__ == "__main__":
    main()
