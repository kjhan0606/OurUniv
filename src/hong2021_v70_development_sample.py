#!/usr/bin/env python
"""Run the single train-gate-authorized V70 locked development sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
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
    PROGRAM_SHA256 as TRAIN_GATE_PROGRAM_SHA256,
    _load_fit,
    heun_sample,
    load_program as load_train_gate_program,
    project_residual_dc,
    sigma_schedule,
)


PROGRAM_SCHEMA = "hong2021-v70-locked-three-domain-development-program-v1"
PROGRAM_SHA256 = "5417fceb29b42108b3f75cc00f0c2c3d9a8f3cc1977b778dbb9386cce1caa7fd"
PROGRAM_FREEZE_COMMIT = "a6429ee651a36ce88e3862b8e9de17523176b8c6"
ENSEMBLE_SCHEMA = "hong2021-v70-query-aligned-latent-spatial-score-ensemble-v1"
METHOD = "fixed_step30000_query_aligned_latent_spatial_score"
ARMS = (
    "query_aligned_latent_spatial_score",
    "independent_voxel_V63_marginal",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    """Validate only local/frozen programs; this does not touch development data."""
    repo = repo.resolve()
    if sha256_file(path.resolve()) != PROGRAM_SHA256:
        raise ValueError("V70 development program hash differs")
    program = _json(path.resolve())
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status")
        != "frozen_during_fixed_training_before_train_gate_result_or_development_access"
    ):
        raise ValueError("V70 development program schema or status differs")
    parent = program["parent_programs"]
    for key in ("v70_model_program", "v70_train_gate_program"):
        if sha256_file(_path(repo, parent[key])) != parent[f"{key}_sha256"]:
            raise ValueError(f"V70 development parent differs: {key}")
    if parent["v70_train_gate_program_sha256"] != TRAIN_GATE_PROGRAM_SHA256:
        raise ValueError("V70 development train-gate program binding differs")
    return program


def authorize_train_gate(
    program: dict[str, Any], repo: Path, gate_path: Path, gate_sha: str,
    commit: str,
) -> dict[str, Any]:
    parent = program["parent_programs"]
    if (
        gate_path.resolve() != Path(parent["required_train_gate_decision"]).resolve()
        or sha256_file(gate_path) != gate_sha
    ):
        raise ValueError("V70 development train-gate file or hash differs")
    gate = _json(gate_path)
    exact = {
        "schema": parent["required_train_gate_schema"],
        "status": parent["required_train_gate_status"],
        "program_sha256": parent["v70_train_gate_program_sha256"],
        "train_mechanism_pass": parent["required_train_mechanism_pass"],
        "candidate_selected": parent["required_candidate_selected"],
        "classification": parent["required_classification"],
        "next": parent["required_next"],
        "validation_accessed": False,
        "development_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    if any(gate.get(key) != value for key, value in exact.items()):
        raise ValueError("V70 development train-gate authorization differs")
    if (
        canonical_digest(gate) != gate.get("decision_digest_sha256")
        or not _is_ancestor(repo, str(gate.get("code_commit")), commit)
    ):
        raise ValueError("V70 development train-gate digest or ancestry differs")
    return gate


def load_development_definition(
    program: dict[str, Any], repo: Path
) -> dict[str, Any]:
    """Touch hash-bound development definitions only after gate authorization."""
    frozen = program["frozen_inputs"]
    path = _path(repo, frozen["v35_development_definition"])
    if sha256_file(path) != frozen["v35_development_definition_sha256"]:
        raise ValueError("V70 V35 development definition differs")
    v35 = _json(path)
    selected = program["immutable_development_selection"]
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        label = domain if domain != "Swift" else "Swift"
        selection_path = Path(selected[f"{label}_selection"]).resolve()
        selection_sha = selected[f"{label}_selection_sha256"]
        if (
            Path(row["phase_object_selection"]).resolve() != selection_path
            or row["phase_object_selection_sha256"] != selection_sha
            or sha256_file(selection_path) != selection_sha
        ):
            raise ValueError(f"V70 {domain} development selection differs")
        for key in ("validation_data", "validation_cache"):
            if sha256_file(Path(row[key])) != row[f"{key}_sha256"]:
                raise ValueError(f"V70 {domain} {key} differs")
    return v35


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
        "object_amplitude_prediction": handle.create_dataset(
            "object_amplitude_prediction", shape=(16,), dtype="f4"
        ),
        "initial_latent_sha256": handle.create_dataset(
            "initial_latent_sha256", shape=(16, 16, 32), dtype="u1"
        ),
        "maximum_inverse_CDF_error": handle.create_dataset(
            "maximum_inverse_CDF_error", shape=(16, 16), dtype="f4"
        ),
    }


@torch.inference_mode()
def sample_all(
    program_path: Path,
    repo: Path,
    train_gate_path: Path,
    train_gate_sha: str,
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
        raise RuntimeError("V70 development sampling requires clean frozen Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V70 development sampling requires the Lageunha Ada GPU")
    if output_root.exists():
        raise FileExistsError("V70 refuses an existing development output")

    gate = authorize_train_gate(
        program, repo, train_gate_path.resolve(), train_gate_sha, commit
    )
    # The following call is deliberately after the passing gate check.
    v35 = load_development_definition(program, repo)
    parent = program["parent_programs"]
    train_program_path = _path(repo, parent["v70_train_gate_program"])
    train_program, v70, inherited_v35, _ = load_train_gate_program(
        train_program_path, repo
    )
    if inherited_v35["development_domains"] != v35["development_domains"]:
        raise ValueError("V70 development domain definition differs")
    device = torch.device("cuda")
    model, training_report, checkpoint_sha, report_sha = _load_fit(
        train_program, repo, commit, device
    )
    if (
        gate.get("training_checkpoint_sha256") != checkpoint_sha
        or gate.get("training_report_sha256") != report_sha
    ):
        raise ValueError("V70 development fit differs from train gate")
    marginal, marginal_v35, prepared = _frozen_marginal(v70, repo, commit, device)
    if marginal_v35["development_domains"] != v35["development_domains"]:
        raise ValueError("V70 development marginal source differs")

    selection = _selection_arrays(v35)
    sampling = program["fixed_sampling"]
    if (
        int(sampling["members_per_query"]) != 16
        or int(sampling["inference_batch"]) != 4
        or int(sampling["sigma_steps"]) != 40
        or float(sampling["sigma_minimum"]) != 0.002
        or float(sampling["sigma_maximum"]) != 40.0
        or float(sampling["rho"]) != 7.0
    ):
        raise ValueError("V70 development frozen sampler differs")
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
    checkpoint_path = Path(program["frozen_inputs"]["expected_v70_checkpoint"])
    report_path = Path(program["frozen_inputs"]["expected_v70_training_report"])
    maximum_dc = {arm: 0.0 for arm in ARMS}
    maximum_inverse = {arm: 0.0 for arm in ARMS}
    peak = 0
    try:
        for domain in DOMAIN_ORDER:
            row = v35["development_domains"][domain]
            indices = np.asarray(selection[domain]["source_index"], dtype=np.int64)
            if indices.shape != (16,) or len(np.unique(indices)) != 16:
                raise ValueError(f"V70 {domain} development indices differ")
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
                    generated = {arm: [] for arm in ARMS}
                    inverse_errors = {arm: [] for arm in ARMS}
                    digests: list[np.ndarray] = []
                    for start in range(0, 16, int(sampling["inference_batch"])):
                        count = min(int(sampling["inference_batch"]), 16 - start)
                        innovation = torch.randn(
                            (count, 1, 64, 64, 64), device=device,
                            generator=generator,
                        )
                        for row_noise in innovation.detach().cpu().numpy():
                            digests.append(
                                np.frombuffer(hashlib.sha256(row_noise.tobytes()).digest(), dtype=np.uint8)
                            )
                        expanded = condition_tensor.expand(count, -1, -1, -1, -1)
                        latent = {
                            ARMS[0]: heun_sample(model, expanded, innovation, schedule),
                            ARMS[1]: innovation,
                        }
                        parameter_batch = parameters.expand(count, -1, -1, -1, -1)
                        for arm in ARMS:
                            uniform = standard_normal_cdf(latent[arm])
                            standardized = bounded_mixture_inverse(parameter_batch, uniform)
                            error = torch.amax(
                                torch.abs(
                                    bounded_mixture_cdf(parameter_batch, standardized)
                                    - uniform.clamp(1.0e-7, 1.0 - 1.0e-7)
                                ),
                                dim=(-4, -3, -2, -1),
                            )
                            residual = (
                                standardized.cpu().numpy().astype(np.float64)
                                * target_std + target_mean
                            )
                            residual, dc = project_residual_dc(residual)
                            maximum_dc[arm] = max(maximum_dc[arm], dc)
                            maximum_inverse[arm] = max(
                                maximum_inverse[arm], float(error.max().cpu())
                            )
                            generated[arm].append(
                                (backbone[None] + residual).astype(np.float32)
                            )
                            inverse_errors[arm].append(error.cpu().numpy())
                    for arm in ARMS:
                        sample = np.concatenate(generated[arm], axis=0)
                        error = np.concatenate(inverse_errors[arm], axis=0)
                        if sample.shape != (16, 1, 64, 64, 64) or not np.isfinite(sample).all():
                            raise RuntimeError("V70 development generated field differs")
                        datasets[arm]["sample"][object_position] = sample
                        datasets[arm]["conditional_mean"][object_position] = backbone
                        datasets[arm]["truth"][object_position] = truth
                        datasets[arm]["object_amplitude_prediction"][object_position] = float(
                            prepared[f"{domain}/validation/object_amplitude"][int(query_index)]
                        )
                        datasets[arm]["initial_latent_sha256"][object_position] = np.stack(digests)
                        datasets[arm]["maximum_inverse_CDF_error"][object_position] = error
                    print(
                        f"[v70-development] {domain} {object_position + 1}/16",
                        flush=True,
                    )
                peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
                for arm in ARMS:
                    handles[arm].attrs.update(
                        {
                            "schema": ENSEMBLE_SCHEMA,
                            "method": METHOD if arm == ARMS[0] else "independent_voxel_V63_marginal_control",
                            "arm": arm,
                            "v70_development_program_sha256": PROGRAM_SHA256,
                            "v70_train_gate_program_sha256": TRAIN_GATE_PROGRAM_SHA256,
                            "train_mechanism_gate": str(train_gate_path.resolve()),
                            "train_mechanism_gate_sha256": train_gate_sha,
                            "train_mechanism_pass": True,
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
                            "candidate_arm": arm == ARMS[0],
                            "control_may_affect_pass_decision": False,
                            "sample_clipping": False,
                            "posthoc_Ak_used": False,
                            "development_sampling_authorized_by_train_gate": True,
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
        raise RuntimeError("V70 development DC or inverse-CDF tolerance differs")
    print(
        json.dumps(
            {
                "status": "complete_locked_development_sampling",
                "train_gate_sha256": train_gate_sha,
                "maximum_absolute_residual_dc": maximum_dc,
                "maximum_inverse_CDF_error": maximum_inverse,
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
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--train-gate-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sample_all(
        args.program, args.repo, args.train_gate, args.train_gate_sha256,
        args.out,
    )


if __name__ == "__main__":
    main()
