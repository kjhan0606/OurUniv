#!/usr/bin/env python
"""Apply the frozen V83 marginal inverse to V72 SQT and paired control latents."""
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

from hong2021_v15_development_gate import git_state
from hong2021_v18_init import sha256_file
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v48_train import condition_cube, load_cache
from hong2021_v70_train_gate import (
    _load_fit,
    heun_sample,
    load_program as load_v70_train_gate_program,
    project_residual_dc,
    sigma_schedule,
)
from hong2021_v72_sqt import conditioning_strata, spatial_quantile_transport
from hong2021_v80_sample import innovation_digest_table, innovation_numpy
from hong2021_v83_contract import DOMAIN_ORDER, load_program
from hong2021_v83_network import conditional_forward, conditional_inverse
from hong2021_v83_train import CHECKPOINT_SCHEMA, seeded_model


SCHEMA = "hong2021-v83-v72-sqt-conditional-marginal-spline-ensemble-v1"
DOMAIN_KEYS = {"TNG100": "tng", "SIMBA": "simba_dev", "Swift": "swift_dev"}
ARMS = ("candidate", "control")
QUERIES = 32
MEMBERS = 16
GRID = 64
TRAIN_GATE_PROGRAM = Path("config/hong2021_v70_train_joint_structure_gate_program.json")


def _new_ensemble(handle: h5py.File) -> dict[str, h5py.Dataset]:
    return {
        "sample": handle.create_dataset(
            "sample",
            shape=(QUERIES, MEMBERS, 1, GRID, GRID, GRID),
            dtype="f4",
            chunks=(1, 1, 1, GRID, GRID, GRID),
            compression="lzf",
        ),
        "conditional_mean": handle.create_dataset(
            "conditional_mean",
            shape=(QUERIES, 1, GRID, GRID, GRID),
            dtype="f4",
            compression="lzf",
        ),
        "truth": handle.create_dataset(
            "truth",
            shape=(QUERIES, 1, GRID, GRID, GRID),
            dtype="f4",
            compression="lzf",
        ),
        "source_index": handle.create_dataset("source_index", shape=(QUERIES,), dtype="i8"),
        "initial_latent_sha256": handle.create_dataset(
            "initial_latent_sha256", shape=(QUERIES, MEMBERS, 32), dtype="u1"
        ),
        "maximum_inverse_error": handle.create_dataset(
            "maximum_inverse_error", shape=(QUERIES, MEMBERS), dtype="f4"
        ),
        "maximum_absolute_residual_DC": handle.create_dataset(
            "maximum_absolute_residual_DC", shape=(QUERIES, MEMBERS), dtype="f4"
        ),
        "pre_inverse_stratum_multiset_equal": handle.create_dataset(
            "pre_inverse_stratum_multiset_equal", shape=(QUERIES,), dtype="u1"
        ),
        "maximum_pre_inverse_stratum_multiset_error": handle.create_dataset(
            "maximum_pre_inverse_stratum_multiset_error", shape=(QUERIES,), dtype="f4"
        ),
        "marginal_tied_voxel_fraction": handle.create_dataset(
            "marginal_tied_voxel_fraction", shape=(QUERIES,), dtype="f4"
        ),
        "rank_disagreement_fraction_excluding_marginal_ties": handle.create_dataset(
            "rank_disagreement_fraction_excluding_marginal_ties",
            shape=(QUERIES,),
            dtype="f4",
        ),
    }


def expected_attrs(
    arm: str,
    seed: int,
    checkpoint_sha256: str,
    source_data_sha256: str,
    source_cache_sha256: str,
    commit: str,
    pairing_digest: str,
) -> dict[str, Any]:
    return {
        "ensemble_members": MEMBERS,
        "sampler": (
            "V70_EDM_V72_SQT_plus_V83_conditional_marginal_spline"
            if arm == "candidate"
            else "paired_independent_normal_plus_V83_conditional_marginal_spline"
        ),
        "sampler_steps": 40 if arm == "candidate" else 0,
        "sampling_batch_size": 4,
        "noise_seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
        "source_data_sha256": source_data_sha256,
        "source_cache_sha256": source_cache_sha256,
        "sampling_code_commit": commit,
        "innovation_pairing_digest": pairing_digest,
        "worktree_clean_at_sampling": True,
        "complete": True,
    }


def _load_v83_fit(
    checkpoint_path: Path,
    checkpoint_sha256: str,
    train_gate_path: Path,
    train_gate_sha256: str,
    program_sha256: str,
    device: torch.device,
) -> torch.nn.Module:
    if (
        sha256_file(checkpoint_path) != checkpoint_sha256
        or sha256_file(train_gate_path) != train_gate_sha256
    ):
        raise ValueError("V83 sampling fit artifact hash differs")
    gate = json.loads(train_gate_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        gate.get("status") != "pass"
        or gate.get("train_holdout_mechanism_pass") is not True
        or gate.get("checkpoint_sha256") != checkpoint_sha256
        or gate.get("program_sha256") != program_sha256
        or gate.get("validation_payload_accessed") is not False
        or gate.get("independent_gate_locked") is not True
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("program_sha256") != program_sha256
    ):
        raise ValueError("V83 train-only gate does not authorize consumed development")
    model = seeded_model(device)
    model.load_state_dict(checkpoint["ema_state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def sample(
    program_path: Path,
    repo: Path,
    conditioning_cache: Path,
    cache_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    train_gate_path: Path,
    train_gate_sha256: str,
    output_root: Path,
) -> None:
    repo = repo.resolve()
    commit, clean = git_state(repo)
    if (
        not clean
        or socket.gethostname().split(".")[0].lower() != "lageunha"
        or not torch.cuda.is_available()
        or "ada" not in torch.cuda.get_device_name(0).lower()
    ):
        raise RuntimeError("V83 sampling requires clean frozen Lageunha Ada")
    program, _, _ = load_program(program_path, repo, commit)
    if (
        output_root.resolve() != Path(program["output_roots"]["development"]).resolve()
        or output_root.exists()
    ):
        raise FileExistsError("V83 sampling refuses an existing or differing output")
    if program["consumed_development"]["authorized"] is not True:
        raise ValueError("V83 consumed development is not authorized")
    program_sha = sha256_file(program_path)
    device = torch.device("cuda")
    marginal = _load_v83_fit(
        checkpoint_path,
        checkpoint_sha256,
        train_gate_path,
        train_gate_sha256,
        program_sha,
        device,
    )
    train_program, _, v35, _ = load_v70_train_gate_program(
        (repo / TRAIN_GATE_PROGRAM).resolve(), repo
    )
    spatial_model, _, v70_checkpoint_sha, _ = _load_fit(
        train_program, repo, commit, device
    )
    if v70_checkpoint_sha != program["frozen_inputs"]["v70_checkpoint_sha256"]:
        raise ValueError("V83 V70 spatial checkpoint differs")
    if (
        conditioning_cache.resolve()
        != Path(program["frozen_inputs"]["conditioning_cache"]).resolve()
        or cache_sha256 != program["frozen_inputs"]["conditioning_cache_sha256"]
        or sha256_file(conditioning_cache) != cache_sha256
    ):
        raise ValueError("V83 sampling conditioning cache differs")
    prepared = load_cache(conditioning_cache, cache_sha256, commit)
    schedule = sigma_schedule(40, 0.002, 40.0, 7.0, device=device)
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    maximum_inverse = 0.0
    maximum_dc = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for domain in DOMAIN_ORDER:
            row = v35["development_domains"][domain]
            binding = program["consumed_development"]["file_bindings"][domain]
            if (
                Path(row["validation_data"]).resolve()
                != Path(binding["data"]).resolve()
                or Path(row["validation_cache"]).resolve()
                != Path(binding["cache"]).resolve()
                or sha256_file(Path(binding["data"])) != binding["data_sha256"]
                or sha256_file(Path(binding["cache"])) != binding["cache_sha256"]
            ):
                raise ValueError(f"V83 {domain} consumed file binding differs")
            indices = np.asarray(
                program["consumed_development"]["selection"][domain], dtype=np.int64
            )
            if len(indices) != QUERIES or len(np.unique(indices)) != QUERIES:
                raise ValueError(f"V83 {domain} consumed selection differs")
            seed = int(program["consumed_development"]["seeds"][domain])
            digest_table = innovation_digest_table(seed, domain)
            pairing_digest = hashlib.sha256(
                digest_table.tobytes(order="C")
            ).hexdigest()
            expected_pairing = program["consumed_development"]["pairing_sha256"][domain]
            if pairing_digest != expected_pairing:
                raise ValueError(f"V83 {domain} innovation pairing differs")
            data, cache = _open_split(row, "validation")
            handles: dict[str, h5py.File] = {}
            datasets: dict[str, dict[str, h5py.Dataset]] = {}
            partials: dict[str, Path] = {}
            try:
                for arm in ARMS:
                    final = output_root / arm / DOMAIN_KEYS[domain] / "ensemble16.h5"
                    final.parent.mkdir(parents=True, exist_ok=True)
                    partials[arm] = final.with_suffix(".h5.partial")
                    handles[arm] = h5py.File(partials[arm], "w")
                    datasets[arm] = _new_ensemble(handles[arm])
                    datasets[arm]["source_index"][:] = indices
                for position, source_index in enumerate(indices):
                    condition, _, backbone = condition_cube(
                        data,
                        cache,
                        prepared,
                        domain,
                        "validation",
                        int(source_index),
                    )
                    truth = np.asarray(data["target"][int(source_index)], dtype=np.float32)
                    condition_tensor = torch.from_numpy(condition[None]).to(device)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        parameters = marginal(condition_tensor)
                    parameters = parameters.float()
                    innovation_rows = [
                        innovation_numpy(seed, domain, position, member)
                        for member in range(MEMBERS)
                    ]
                    innovation = torch.from_numpy(np.stack(innovation_rows)).to(device)
                    raw_parts = []
                    for start in range(0, MEMBERS, 4):
                        expanded = condition_tensor.expand(4, -1, -1, -1, -1)
                        raw_parts.append(
                            heun_sample(
                                spatial_model,
                                expanded,
                                innovation[start : start + 4],
                                schedule,
                            )
                        )
                    raw_latent = torch.cat(raw_parts, dim=0)
                    score = torch.from_numpy(
                        (np.asarray(backbone[0], dtype=np.float32) + target_mean)[None]
                    ).to(device)
                    positions = conditioning_strata(score)
                    sqt_latent, diagnostics = spatial_quantile_transport(
                        raw_latent, innovation, positions
                    )
                    parameter_batch = parameters.expand(MEMBERS, -1, -1, -1, -1)
                    for arm, latent in (
                        ("candidate", sqt_latent),
                        ("control", innovation),
                    ):
                        standardized, _ = conditional_inverse(parameter_batch, latent)
                        recovered, _ = conditional_forward(parameter_batch, standardized)
                        inverse_error = torch.amax(
                            torch.abs(recovered - latent), dim=(-4, -3, -2, -1)
                        ).cpu().numpy()
                        maximum_inverse = max(
                            maximum_inverse, float(inverse_error.max())
                        )
                        physical = (
                            standardized.cpu().numpy().astype(np.float64) * target_std
                            + target_mean
                        )
                        projected, dc = project_residual_dc(physical)
                        total = (
                            np.asarray(backbone, dtype=np.float64)[None] + projected
                        ).astype(np.float32)
                        final_dc = np.abs(
                            (
                                total.astype(np.float64)
                                - np.asarray(backbone, dtype=np.float64)[None]
                            ).mean(axis=(-3, -2, -1), dtype=np.float64)
                        ).reshape(MEMBERS)
                        maximum_dc = max(maximum_dc, float(final_dc.max()))
                        if (
                            total.shape != (MEMBERS, 1, GRID, GRID, GRID)
                            or not np.isfinite(total).all()
                            or float(final_dc.max()) > 1.0e-7
                        ):
                            raise RuntimeError("V83 sampled field invariant differs")
                        datasets[arm]["sample"][position] = total
                        datasets[arm]["conditional_mean"][position] = backbone
                        datasets[arm]["truth"][position] = truth
                        datasets[arm]["initial_latent_sha256"][position] = digest_table[position]
                        datasets[arm]["maximum_inverse_error"][position] = inverse_error
                        datasets[arm]["maximum_absolute_residual_DC"][position] = final_dc
                        datasets[arm]["pre_inverse_stratum_multiset_equal"][position] = int(
                            diagnostics["pre_inverse_stratum_multiset_equal"]
                        )
                        datasets[arm]["maximum_pre_inverse_stratum_multiset_error"][position] = float(
                            diagnostics["maximum_pre_inverse_stratum_multiset_error"]
                        )
                        datasets[arm]["marginal_tied_voxel_fraction"][position] = float(
                            diagnostics["marginal_tied_voxel_fraction"]
                        )
                        datasets[arm]["rank_disagreement_fraction_excluding_marginal_ties"][position] = float(
                            diagnostics[
                                "rank_disagreement_fraction_excluding_marginal_ties"
                            ]
                        )
                    print(
                        f"[v83-sample] {domain} {position + 1}/{QUERIES}",
                        flush=True,
                    )
                for arm in ARMS:
                    handles[arm].attrs["schema"] = SCHEMA
                    handles[arm].attrs["arm"] = arm
                    handles[arm].attrs["program_sha256"] = program_sha
                    handles[arm].attrs["v70_checkpoint_sha256"] = v70_checkpoint_sha
                    for key, value in expected_attrs(
                        arm,
                        seed,
                        checkpoint_sha256,
                        binding["data_sha256"],
                        binding["cache_sha256"],
                        commit,
                        pairing_digest,
                    ).items():
                        handles[arm].attrs[key] = value
            finally:
                for handle in handles.values():
                    handle.close()
                data.close()
                cache.close()
            for arm in ARMS:
                os.replace(partials[arm], partials[arm].with_suffix(""))
    finally:
        prepared.close()
    print(
        json.dumps(
            {
                "status": "complete_V83_consumed_development_sampling",
                "maximum_inverse_error": maximum_inverse,
                "maximum_absolute_residual_DC": maximum_dc,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--conditioning-cache", type=Path, required=True)
    parser.add_argument("--conditioning-cache-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--train-gate-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    sample(
        args.program,
        args.repo,
        args.conditioning_cache,
        args.conditioning_cache_sha256,
        args.checkpoint,
        args.checkpoint_sha256,
        args.train_gate,
        args.train_gate_sha256,
        args.output_root,
    )


if __name__ == "__main__":
    main()
