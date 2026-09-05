#!/usr/bin/env python
"""Run the single preflight-authorized V71 ECC development sample."""
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
from hong2021_v28_empirical import DOMAIN_KEYS, DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v37_query_alignment import _selection_arrays
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
from hong2021_v71_ecc import (
    ARMS,
    CANDIDATE,
    CONTROL,
    ENSEMBLE_SCHEMA,
    METHOD,
    PROGRAM_FREEZE_COMMIT,
    PROGRAM_SHA256,
    authorize_parent_evidence,
    ensemble_copula_couple,
    load_development_definition,
    load_program,
    resolve_path,
    strict_json,
    validate_preflight,
)


def _new_ensemble(handle: h5py.File) -> dict[str, h5py.Dataset]:
    datasets = {
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
        "object_amplitude_prediction": handle.create_dataset(
            "object_amplitude_prediction", shape=(16,), dtype="f4"
        ),
        "initial_latent_sha256": handle.create_dataset(
            "initial_latent_sha256", shape=(16, 16, 32), dtype="u1"
        ),
        "maximum_inverse_CDF_error": handle.create_dataset(
            "maximum_inverse_CDF_error", shape=(16, 16), dtype="f4"
        ),
        "pre_inverse_sorted_latent_multiset_equal": handle.create_dataset(
            "pre_inverse_sorted_latent_multiset_equal", shape=(16,), dtype="u1"
        ),
        "maximum_pre_inverse_sorted_latent_multiset_error": handle.create_dataset(
            "maximum_pre_inverse_sorted_latent_multiset_error",
            shape=(16,), dtype="f4",
        ),
        "maximum_pre_DC_sorted_residual_multiset_error": handle.create_dataset(
            "maximum_pre_DC_sorted_residual_multiset_error",
            shape=(16,), dtype="f4",
        ),
        "maximum_post_DC_sorted_residual_multiset_error": handle.create_dataset(
            "maximum_post_DC_sorted_residual_multiset_error",
            shape=(16,), dtype="f4",
        ),
        "control_tied_voxel_fraction": handle.create_dataset(
            "control_tied_voxel_fraction", shape=(16,), dtype="f4"
        ),
        "candidate_rank_disagreement_fraction_excluding_control_ties": (
            handle.create_dataset(
                "candidate_rank_disagreement_fraction_excluding_control_ties",
                shape=(16,), dtype="f4",
            )
        ),
    }
    return datasets


def _training_program(repo: Path) -> Path:
    result = strict_json(repo / "config/hong2021_v70_result_record.json")
    frozen = result["frozen_programs"]
    return resolve_path(repo, frozen["train_gate_program"])


@torch.inference_mode()
def sample_all(
    program_path: Path,
    repo: Path,
    preflight_path: Path,
    preflight_sha: str,
    output_root: Path,
) -> None:
    repo = repo.resolve()
    program = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V71 development sampling requires clean frozen Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V71 development sampling requires the Lageunha Ada GPU")
    authorize_parent_evidence(program, repo, commit)
    preflight = validate_preflight(
        preflight_path.resolve(), preflight_sha, repo, commit
    )
    expected_output = Path(program["output_roots"]["development"]).resolve()
    if output_root.resolve() != expected_output:
        raise ValueError("V71 development output path differs")
    if output_root.exists():
        raise FileExistsError("V71 refuses an existing development output")

    # Semantic development access begins only after every authorization above.
    v35 = load_development_definition(program, repo)
    train_program, v70, inherited_v35, _ = load_v70_train_gate_program(
        _training_program(repo), repo
    )
    if inherited_v35["development_domains"] != v35["development_domains"]:
        raise ValueError("V71 inherited development domain definition differs")
    device = torch.device("cuda")
    model, training_report, checkpoint_sha, report_sha = _load_fit(
        train_program, repo, commit, device
    )
    frozen = program["frozen_inputs"]
    if (
        checkpoint_sha != frozen["v70_checkpoint_sha256"]
        or report_sha != frozen["v70_training_report_sha256"]
    ):
        raise ValueError("V71 fixed V70 fit differs")
    marginal, marginal_v35, prepared = _frozen_marginal(v70, repo, commit, device)
    if marginal_v35["development_domains"] != v35["development_domains"]:
        raise ValueError("V71 frozen marginal source differs")

    selection = _selection_arrays(v35)
    sampling = program["fixed_sampling"]
    if (
        int(sampling["members_per_query"]) != 16
        or int(sampling["inference_batch"]) != 4
        or int(sampling["sigma_steps"]) != 40
        or float(sampling["sigma_minimum"]) != 0.002
        or float(sampling["sigma_maximum"]) != 40.0
        or float(sampling["rho"]) != 7.0
        or float(sampling["stochastic_churn"]) != 0.0
    ):
        raise ValueError("V71 frozen sampler differs")
    schedule = sigma_schedule(
        int(sampling["sigma_steps"]),
        float(sampling["sigma_minimum"]),
        float(sampling["sigma_maximum"]),
        float(sampling["rho"]),
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(int(sampling["noise_seed"]))
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    checkpoint_path = Path(frozen["v70_checkpoint"])
    report_path = Path(frozen["v70_training_report"])
    maximum_dc = {arm: 0.0 for arm in ARMS}
    maximum_inverse = {arm: 0.0 for arm in ARMS}
    maximum_post_dc = 0.0
    maximum_tie_fraction = 0.0
    peak = 0
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for domain in DOMAIN_ORDER:
            row = v35["development_domains"][domain]
            indices = np.asarray(selection[domain]["source_index"], dtype=np.int64)
            if indices.shape != (16,) or len(np.unique(indices)) != 16:
                raise ValueError(f"V71 {domain} development indices differ")
            data, cache = _open_split(row, "validation")
            handles: dict[str, h5py.File] = {}
            datasets: dict[str, dict[str, h5py.Dataset]] = {}
            partials: dict[str, Path] = {}
            try:
                for arm in ARMS:
                    path = (
                        output_root / arm / "development_candidate"
                        / DOMAIN_KEYS[domain] / "ensemble16.h5"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partials[arm] = path.with_suffix(path.suffix + ".partial")
                    handles[arm] = h5py.File(partials[arm], "w")
                    datasets[arm] = _new_ensemble(handles[arm])
                    for name, value in selection[domain].items():
                        handles[arm].create_dataset(name, data=value)

                for object_position, query_index in enumerate(indices):
                    condition, _, backbone = condition_cube(
                        data, cache, prepared, domain, "validation", int(query_index)
                    )
                    truth = np.asarray(data["target"][int(query_index)], dtype=np.float32)
                    condition_tensor = torch.from_numpy(condition[None]).to(device)
                    parameters = marginal(condition_tensor).float()
                    innovation_parts: list[torch.Tensor] = []
                    rank_parts: list[torch.Tensor] = []
                    digests: list[np.ndarray] = []
                    for start in range(0, 16, int(sampling["inference_batch"])):
                        count = min(int(sampling["inference_batch"]), 16 - start)
                        innovation = torch.randn(
                            (count, 1, 64, 64, 64),
                            device=device,
                            generator=generator,
                        )
                        for row_noise in innovation.detach().cpu().numpy():
                            digests.append(
                                np.frombuffer(
                                    hashlib.sha256(row_noise.tobytes()).digest(),
                                    dtype=np.uint8,
                                )
                            )
                        expanded = condition_tensor.expand(count, -1, -1, -1, -1)
                        rank_parts.append(
                            heun_sample(model, expanded, innovation, schedule)
                        )
                        innovation_parts.append(innovation)
                    innovation_all = torch.cat(innovation_parts, dim=0)
                    rank_source = torch.cat(rank_parts, dim=0)
                    ecc_latent, ecc_diagnostics = ensemble_copula_couple(
                        rank_source, innovation_all
                    )
                    tie_fraction = float(
                        ecc_diagnostics["control_tied_voxel_fraction"]
                    )
                    maximum_tie_fraction = max(maximum_tie_fraction, tie_fraction)
                    if tie_fraction > float(
                        program["code_only_preflight_and_numerical_requirements"]
                        ["required_during_single_sampling"]
                        ["maximum_control_tied_voxel_fraction"]
                    ):
                        raise RuntimeError("V71 control tie fraction exceeds freeze")

                    latents = {CANDIDATE: ecc_latent, CONTROL: innovation_all}
                    pre_dc: dict[str, np.ndarray] = {}
                    inverse_errors: dict[str, np.ndarray] = {}
                    parameter_batch = parameters.expand(16, -1, -1, -1, -1)
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
                    pre_dc_error = float(
                        np.max(
                            np.abs(
                                np.sort(pre_dc[CANDIDATE], axis=0)
                                - np.sort(pre_dc[CONTROL], axis=0)
                            )
                        )
                    )
                    if pre_dc_error != 0.0:
                        raise RuntimeError("V71 physical pre-DC multiset differs")
                    projected: dict[str, np.ndarray] = {}
                    for arm in ARMS:
                        projected[arm], dc = project_residual_dc(pre_dc[arm])
                        maximum_dc[arm] = max(maximum_dc[arm], dc)
                    post_dc_error = float(
                        np.max(
                            np.abs(
                                np.sort(projected[CANDIDATE], axis=0).astype(np.float64)
                                - np.sort(projected[CONTROL], axis=0).astype(np.float64)
                            )
                        )
                    )
                    maximum_post_dc = max(maximum_post_dc, post_dc_error)
                    shared = {
                        "pre_inverse_sorted_latent_multiset_equal": int(
                            bool(ecc_diagnostics[
                                "pre_inverse_sorted_latent_multiset_equal"
                            ])
                        ),
                        "maximum_pre_inverse_sorted_latent_multiset_error": float(
                            ecc_diagnostics[
                                "maximum_pre_inverse_sorted_latent_multiset_error"
                            ]
                        ),
                        "maximum_pre_DC_sorted_residual_multiset_error": pre_dc_error,
                        "maximum_post_DC_sorted_residual_multiset_error": post_dc_error,
                        "control_tied_voxel_fraction": tie_fraction,
                        "candidate_rank_disagreement_fraction_excluding_control_ties": float(
                            ecc_diagnostics[
                                "candidate_rank_disagreement_fraction_excluding_control_ties"
                            ]
                        ),
                    }
                    for arm in ARMS:
                        sample = (backbone[None] + projected[arm]).astype(np.float32)
                        if sample.shape != (16, 1, 64, 64, 64) or not np.isfinite(sample).all():
                            raise RuntimeError("V71 development generated field differs")
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
                    del rank_source, ecc_latent, innovation_all, innovation_parts, rank_parts
                    print(
                        f"[v71-development] {domain} {object_position + 1}/16 ",
                        f"tie={tie_fraction:.3e} post_dc={post_dc_error:.3e}",
                        flush=True,
                    )
                peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
                for arm in ARMS:
                    handles[arm].attrs.update(
                        {
                            "schema": ENSEMBLE_SCHEMA,
                            "method": METHOD if arm == CANDIDATE else "independent_voxel_V63_marginal_control",
                            "arm": arm,
                            "v71_development_program_sha256": PROGRAM_SHA256,
                            "v71_preflight": str(preflight_path.resolve()),
                            "v71_preflight_sha256": preflight_sha,
                            "v71_path_B_approved": True,
                            "fresh_train_only_V71_screen_available": False,
                            "v70_train_gate_candidate_selected": False,
                            "v70_train_gate_sha256": preflight["parent_evidence"]["v70_train_gate_sha256"],
                            "v70_terminal_seal_sha256": preflight["parent_evidence"]["v70_terminal_seal_sha256"],
                            "checkpoint": str(checkpoint_path.resolve()),
                            "checkpoint_sha256": checkpoint_sha,
                            "training_report": str(report_path.resolve()),
                            "training_report_sha256": report_sha,
                            "training_initial_code_commit": training_report["initial_code_commit"],
                            "parent_selection": str(Path(row["phase_object_selection"]).resolve()),
                            "parent_selection_sha256": row["phase_object_selection_sha256"],
                            "source_data": str(Path(row["validation_data"]).resolve()),
                            "source_data_sha256": row["validation_data_sha256"],
                            "source_cache": str(Path(row["validation_cache"]).resolve()),
                            "source_cache_sha256": row["validation_cache_sha256"],
                            "ensemble_members": 16,
                            "noise_seed": int(sampling["noise_seed"]),
                            "sampler_steps": 40,
                            "sigma_minimum": 0.002,
                            "sigma_maximum": 40.0,
                            "rho": 7.0,
                            "stochastic_churn": 0.0,
                            "diagnostic_k_h_mpc": 1.0,
                            "maximum_absolute_residual_dc": maximum_dc[arm],
                            "maximum_inverse_CDF_error": maximum_inverse[arm],
                            "maximum_control_tied_voxel_fraction": maximum_tie_fraction,
                            "maximum_post_DC_sorted_residual_multiset_error": maximum_post_dc,
                            "candidate_arm": arm == CANDIDATE,
                            "control_may_affect_pass_decision": False,
                            "sample_clipping": False,
                            "posthoc_Ak_used": False,
                            "development_sampling_authorized_by_V71_path_B_preflight": True,
                            "validation_truth_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
                            "gradient_computed": False,
                            "optimizer_constructed": False,
                            "optimizer_step_performed": False,
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
    if max(maximum_dc.values()) > 1.0e-7 or max(maximum_inverse.values()) > 2.0e-6:
        raise RuntimeError("V71 development DC or inverse-CDF tolerance differs")
    print(
        json.dumps(
            {
                "status": "complete_single_V71_ECC_development_sampling",
                "preflight_sha256": preflight_sha,
                "maximum_absolute_residual_dc": maximum_dc,
                "maximum_inverse_CDF_error": maximum_inverse,
                "maximum_control_tied_voxel_fraction": maximum_tie_fraction,
                "maximum_post_DC_sorted_residual_multiset_error": maximum_post_dc,
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sample_all(
        args.program, args.repo, args.preflight, args.preflight_sha256, args.out
    )


if __name__ == "__main__":
    main()
