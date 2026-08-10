#!/usr/bin/env python
"""V47 train-only audit of physical moment existence under V45 logistic mixtures."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v45_network import LocalMixtureUNet, mixture_parameters, parameter_count
from hong2021_v45_train import (
    CHECKPOINT_SCHEMA,
    PARAMETERS,
    PROGRAM_SHA256 as V45_PROGRAM_SHA256,
    condition_cube,
    load_cache,
    load_program as load_v45_program,
)
from hong2021_v46_tail_occupancy_audit import EXPECTED_OBJECTS, PROBE_VOXELS, _probe_indices


PROGRAM_SCHEMA = "hong2021-v47-logistic-physical-moment-existence-audit-program-v1"
PROGRAM_SHA256 = "8672726b6403c3718995619c31ad83eba0815d231f15f346288d23f1c43ed9f2"
RESULT_SCHEMA = "hong2021-v47-logistic-physical-moment-existence-audit-v1"
WEIGHT_EPSILON = 1.0e-12
MASS_THRESHOLD = 0.01
LARGE_MASS_THRESHOLD = 0.05
QUANTILES = (0.5, 0.9, 0.99, 0.999, 1.0)


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V47 {label} hash differs")
    return json.loads(path.read_text())


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_audit_implementation_or_execution"
    ):
        raise ValueError("V47 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        (repo / parent["v46_record"]).resolve(),
        parent["v46_record_sha256"],
        "V46 record",
    )
    if (
        record.get("audit", {}).get("classification")
        != parent["required_classification"]
        or record.get("audit", {}).get("next") != parent["required_next"]
        or record.get("firewall", {}).get("independent_gate_locked") is not True
    ):
        raise ValueError("V47 V46 conclusion differs")
    frozen = program["frozen_inputs"]
    for key, digest_key in (
        ("v45_checkpoint", "v45_checkpoint_sha256"),
        ("v45_conditioning_cache", "v45_conditioning_cache_sha256"),
        ("v45_program", "v45_program_sha256"),
        ("v46_audit", "v46_audit_sha256"),
    ):
        candidate = Path(frozen[key])
        if not candidate.is_absolute():
            candidate = repo / candidate
        if sha256_file(candidate.resolve()) != frozen[digest_key]:
            raise ValueError(f"V47 frozen {key} hash differs")
    _, v35, _ = load_v45_program((repo / frozen["v45_program"]).resolve(), repo)
    return program, v35


def classify(
    common_second_moment_mass: bool,
    any_first_moment_mass: bool,
    any_second_moment_mass: bool,
) -> tuple[str, str]:
    next_gaussian = "freeze_an_identifiable_train_only_Gaussian_mixture_likelihood_with_all_V45_data_sampling_and_gates_unchanged"
    if common_second_moment_mass:
        return (
            "logistic_components_make_the_physical_delta_squared_moment_divergent",
            next_gaussian,
        )
    if any_first_moment_mass:
        return (
            "logistic_components_make_even_the_mean_physical_density_divergent",
            next_gaussian,
        )
    if any_second_moment_mass:
        return (
            "rare_logistic_scale_excursions_are_mathematically_incompatible_with_Q4",
            next_gaussian,
        )
    return (
        "logistic_physical_moments_exist_on_train_probes",
        "audit_the_empirical_rank_cutoff_and_query_coupling_without_changing_the_likelihood",
    )


def _model(program: dict[str, Any], repo: Path, commit: str) -> tuple[LocalMixtureUNet, dict[str, Any]]:
    frozen = program["frozen_inputs"]
    checkpoint = torch.load(frozen["v45_checkpoint"], map_location="cpu", weights_only=False)
    source_commit = str(checkpoint.get("code_commit"))
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("program_sha256") != V45_PROGRAM_SHA256
        or checkpoint.get("step") != 12_000
        or checkpoint.get("parameters") != PARAMETERS
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, commit],
            cwd=repo,
            capture_output=True,
        ).returncode
    ):
        raise ValueError("V47 checkpoint binding differs")
    model = LocalMixtureUNet()
    if parameter_count(model) != PARAMETERS:
        raise RuntimeError("V47 parameter count differs")
    model.load_state_dict(checkpoint["ema_state_dict"])
    return model, checkpoint


def _summary(value: np.ndarray) -> list[float]:
    if value.size == 0 or not np.isfinite(value).all():
        raise RuntimeError("V47 summary input differs")
    return np.quantile(value.astype(np.float64, copy=False), QUANTILES).tolist()


@torch.inference_mode()
def _domain(
    model: LocalMixtureUNet,
    device: torch.device,
    v35: dict[str, Any],
    prepared: Any,
    domain: str,
    domain_index: int,
    first_limit: float,
    second_limit: float,
) -> dict[str, Any]:
    row = v35["development_domains"][domain]
    objects = int(row["train_objects"])
    if objects != EXPECTED_OBJECTS[domain]:
        raise RuntimeError("V47 object count differs")
    scales = [[] for _ in range(5)]
    first_component_mass = np.zeros(5, dtype=np.float64)
    second_component_mass = np.zeros(5, dtype=np.float64)
    first_component_count = np.zeros(5, dtype=np.int64)
    second_component_count = np.zeros(5, dtype=np.int64)
    total_first_mass: list[np.ndarray] = []
    total_second_mass: list[np.ndarray] = []
    finite_log10_first: list[np.ndarray] = []
    finite_log10_second: list[np.ndarray] = []
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    t_values = (
        4.5 * math.log(10.0) * target_std,
        9.0 * math.log(10.0) * target_std,
    )
    data, cache = _open_split(row, "train")
    try:
        for object_index in range(objects):
            condition, _, backbone = condition_cube(
                data, cache, prepared, domain, "train", object_index
            )
            parameter = model(torch.from_numpy(condition[None]).to(device)).float()
            indices = _probe_indices(domain_index, object_index)
            index_tensor = torch.from_numpy(indices).to(device)
            selected = parameter.reshape(1, 15, -1).index_select(2, index_tensor).reshape(1, 15, -1)
            logits, locations, component_scales = mixture_parameters(
                selected.reshape(1, 15, 1, 1, -1)
            )
            weights = torch.softmax(logits, dim=1).reshape(5, -1)
            locations = locations.reshape(5, -1)
            component_scales = component_scales.reshape(5, -1)
            if float(torch.max(torch.abs(weights.sum(dim=0) - 1.0))) > 1.0e-5:
                raise RuntimeError("V47 mixture weights differ")
            positive = weights > WEIGHT_EPSILON
            first_divergent = (component_scales >= first_limit) & positive
            second_divergent = (component_scales >= second_limit) & positive
            first_mass = torch.sum(torch.where(first_divergent, weights, 0.0), dim=0)
            second_mass = torch.sum(torch.where(second_divergent, weights, 0.0), dim=0)
            total_first_mass.append(first_mass.cpu().numpy())
            total_second_mass.append(second_mass.cpu().numpy())
            for component in range(5):
                scales[component].append(component_scales[component].cpu().numpy())
                first_component_count[component] += int(first_divergent[component].sum().cpu())
                second_component_count[component] += int(second_divergent[component].sum().cpu())
                first_component_mass[component] += float(
                    torch.where(first_divergent[component], weights[component], 0.0)
                    .double()
                    .sum()
                    .cpu()
                )
                second_component_mass[component] += float(
                    torch.where(second_divergent[component], weights[component], 0.0)
                    .double()
                    .sum()
                    .cpu()
                )
            selected_backbone = torch.from_numpy(backbone.reshape(-1)[indices]).to(device).float()
            for power, (t_value, limit, divergent, sink) in enumerate(
                (
                    (t_values[0], first_limit, first_divergent, finite_log10_first),
                    (t_values[1], second_limit, second_divergent, finite_log10_second),
                ),
                start=1,
            ):
                argument = math.pi * component_scales * t_value
                log_mgf = (
                    locations * t_value
                    + torch.log(argument.clamp_min(1.0e-30))
                    - torch.log(torch.sin(argument).clamp_min(1.0e-30))
                )
                log_weighted = (
                    torch.log(weights.clamp_min(WEIGHT_EPSILON))
                    + power * 4.5 * math.log(10.0) * (selected_backbone[None] + target_mean)
                    + log_mgf
                )
                log_weighted = torch.where(divergent, -torch.inf, log_weighted)
                finite_log = torch.logsumexp(log_weighted, dim=0) / math.log(10.0)
                if not torch.isfinite(finite_log).all():
                    raise RuntimeError("V47 finite analytic contribution differs")
                sink.append(finite_log.cpu().numpy())
            if (object_index + 1) % 32 == 0 or object_index + 1 == objects:
                print(f"[v47-audit] {domain} {object_index + 1}/{objects}", flush=True)
    finally:
        data.close()
        cache.close()
    first_mass = np.concatenate(total_first_mass)
    second_mass = np.concatenate(total_second_mass)
    probes = objects * PROBE_VOXELS
    if len(first_mass) != probes or len(second_mass) != probes:
        raise RuntimeError("V47 probe count differs")
    return {
        "train_objects": objects,
        "probe_voxels": probes,
        "component_scale_quantiles": [_summary(np.concatenate(row)) for row in scales],
        "component_fraction_above_first_moment_limit": (first_component_count / probes).tolist(),
        "component_fraction_above_second_moment_limit": (second_component_count / probes).tolist(),
        "component_mean_weight_above_first_moment_limit": (first_component_mass / probes).tolist(),
        "component_mean_weight_above_second_moment_limit": (second_component_mass / probes).tolist(),
        "mean_first_moment_divergent_mixture_mass": float(first_mass.mean(dtype=np.float64)),
        "mean_second_moment_divergent_mixture_mass": float(second_mass.mean(dtype=np.float64)),
        "first_moment_divergent_mass_quantiles": _summary(first_mass),
        "second_moment_divergent_mass_quantiles": _summary(second_mass),
        "fraction_voxels_with_first_moment_divergent_mass": {
            "above_zero": float(np.mean(first_mass > 0.0)),
            "above_0.01": float(np.mean(first_mass >= MASS_THRESHOLD)),
            "above_0.05": float(np.mean(first_mass >= LARGE_MASS_THRESHOLD)),
        },
        "fraction_voxels_with_second_moment_divergent_mass": {
            "above_zero": float(np.mean(second_mass > 0.0)),
            "above_0.01": float(np.mean(second_mass >= MASS_THRESHOLD)),
            "above_0.05": float(np.mean(second_mass >= LARGE_MASS_THRESHOLD)),
        },
        "finite_component_log10_rho_moment_contribution_quantiles": _summary(
            np.concatenate(finite_log10_first)
        ),
        "finite_component_log10_rho_squared_moment_contribution_quantiles": _summary(
            np.concatenate(finite_log10_second)
        ),
    }


def audit(program_path: Path, repo: Path, output: Path) -> dict[str, Any]:
    program, v35 = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V47 requires a clean committed worktree")
    if socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V47 requires Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V47 requires the Lageunha Ada GPU")
    if output.exists():
        raise FileExistsError("V47 refuses existing output")
    model, checkpoint = _model(program, repo.resolve(), commit)
    model = model.to("cuda").eval()
    frozen = program["frozen_inputs"]
    prepared = load_cache(
        Path(frozen["v45_conditioning_cache"]),
        frozen["v45_conditioning_cache_sha256"],
        str(checkpoint["code_commit"]),
    )
    target_std = float(prepared["target_std"][()])
    first_limit = 1.0 / (4.5 * math.log(10.0) * target_std)
    second_limit = 1.0 / (9.0 * math.log(10.0) * target_std)
    expected = program["analytic_definition"]
    if (
        abs(first_limit - expected["finite_mean_density_scale_limit_standardized"])
        > 1.0e-12
        or abs(second_limit - expected["finite_mean_delta_squared_scale_limit_standardized"])
        > 1.0e-12
    ):
        raise RuntimeError("V47 analytic scale limits differ")
    domains: dict[str, Any] = {}
    try:
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            domains[domain] = _domain(
                model,
                torch.device("cuda"),
                v35,
                prepared,
                domain,
                domain_index,
                first_limit,
                second_limit,
            )
    finally:
        prepared.close()
    common_second = all(
        row["mean_second_moment_divergent_mixture_mass"] >= MASS_THRESHOLD
        for row in domains.values()
    )
    any_first = any(
        row["mean_first_moment_divergent_mixture_mass"] >= MASS_THRESHOLD
        for row in domains.values()
    )
    any_second = any(
        row["fraction_voxels_with_second_moment_divergent_mass"]["above_zero"] > 0.0
        for row in domains.values()
    )
    classification, next_step = classify(common_second, any_first, any_second)
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete_train_only_analytic_moment_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "audit_code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "target_std": target_std,
        "finite_mean_density_scale_limit_standardized": first_limit,
        "finite_mean_delta_squared_scale_limit_standardized": second_limit,
        "domains": domains,
        "branch_conditions": {
            "common_domain_second_moment_divergent_mass_at_least_0.01": common_second,
            "any_domain_first_moment_divergent_mass_at_least_0.01": any_first,
            "any_second_moment_divergent_mass": any_second,
        },
        "classification": classification,
        "next": next_step,
        "fit_or_optimizer_performed": False,
        "new_sampling_performed": False,
        "validation_accessed": False,
        "development_arrays_accessed": False,
        "threshold_tuned": False,
        "scale_cap_or_clipping_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.program, args.repo, args.out)


if __name__ == "__main__":
    main()
