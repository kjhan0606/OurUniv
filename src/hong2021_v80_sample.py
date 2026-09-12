#!/usr/bin/env python
"""Preflight and sample the single frozen V80 candidate/control pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
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
from hong2021_v72_sqt import conditioning_strata, spatial_quantile_transport
from hong2021_v80_quantile_calibration import apply_monotone_map


PROGRAM_SCHEMA = "hong2021-v79-single-candidate-program-v1"
PROGRAM_STATUS = "frozen_and_pushed_before_V79_sampling_or_selected_payload_access"
PREFLIGHT_SCHEMA = "hong2021-v80-code-only-single-candidate-preflight-v1"
ENSEMBLE_SCHEMA = "hong2021-v80-calibrated-spatial-quantile-ensemble-v1"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
DOMAIN_KEYS = {"TNG100": "tng", "SIMBA": "simba_dev", "Swift": "swift_dev"}
ARMS = ("candidate", "control")
QUERIES = 32
MEMBERS = 16
GRID = 64
TRAIN_GATE_PROGRAM = Path("config/hong2021_v70_train_joint_structure_gate_program.json")


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def resolve_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def member_seed(base: int, domain: str, query: int, member: int) -> int:
    payload = f"V80-PCG64-v1\0{base}\0{domain}\0{query}\0{member}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def innovation_numpy(base: int, domain: str, query: int, member: int) -> np.ndarray:
    generator = np.random.Generator(
        np.random.PCG64(member_seed(base, domain, query, member))
    )
    return generator.standard_normal((1, GRID, GRID, GRID), dtype=np.float32)


def innovation_digest_table(base: int, domain: str) -> np.ndarray:
    output = np.empty((QUERIES, MEMBERS, 32), dtype=np.uint8)
    for query in range(QUERIES):
        for member in range(MEMBERS):
            value = innovation_numpy(base, domain, query, member)
            output[query, member] = np.frombuffer(
                hashlib.sha256(value.tobytes(order="C")).digest(), dtype=np.uint8
            )
    return output


def innovation_pairing_digest(base: int, domain: str) -> str:
    return hashlib.sha256(
        innovation_digest_table(base, domain).tobytes(order="C")
    ).hexdigest()


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program = strict_json(path.resolve())
    authorization = program.get("authorization", {})
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or program.get("candidate_count") != 1
        or authorization.get(
            "explicit_user_approval_for_candidate_design_and_single_use_execution"
        )
        is not True
        or authorization.get("V79_single_use_gate_execution") is not True
        or authorization.get("post_disclosure_candidate_change_or_retry") is not False
    ):
        raise ValueError("V80 candidate program or authorization differs")
    for label, row in program["implementation_sources"].items():
        path_value = resolve_path(repo, row["path"])
        if sha256_file(path_value) != row["sha256"]:
            raise ValueError(f"V80 implementation source differs: {label}")
    for label, row in program["frozen_artifacts"].items():
        path_value = resolve_path(repo, row["path"])
        if sha256_file(path_value) != row["sha256"]:
            raise ValueError(f"V80 frozen artifact differs: {label}")
    if set(program["single_use_fresh_selection"]) != set(DOMAIN_ORDER):
        raise ValueError("V80 fresh domain selection differs")
    for domain in DOMAIN_ORDER:
        indices = list(map(int, program["single_use_fresh_selection"][domain]))
        if len(indices) != QUERIES or len(set(indices)) != QUERIES:
            raise ValueError(f"V80 {domain} selected indices differ")
        contracts = program["frozen_domain_execution_contracts"][domain]
        digest = program["frozen_execution_provenance"][
            "innovation_pairing_digests"
        ][domain]
        for arm in ARMS:
            if contracts[f"{arm}_expected_attrs"]["innovation_pairing_digest"] != digest:
                raise ValueError(f"V80 {domain} pairing contract differs")
    return program


def candidate_freeze_commit(program_path: Path, repo: Path) -> str:
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(program_path.resolve())],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise ValueError("V80 candidate freeze commit cannot be resolved")
    return commit


def validate_without_selected_payload(
    program_path: Path, repo: Path, output_root: Path
) -> dict[str, Any]:
    repo = repo.resolve()
    program = load_program(program_path, repo)
    commit, clean = git_state(repo)
    freeze_commit = candidate_freeze_commit(program_path, repo)
    if not clean or not _is_ancestor(repo, freeze_commit, commit):
        raise RuntimeError("V80 preflight requires clean candidate-freeze ancestry")
    if output_root.exists():
        raise FileExistsError("V80 preflight refuses existing output")
    hashes = {}
    for label, row in program["fresh_file_bindings"].items():
        path = Path(row["path"]).resolve()
        digest = sha256_file(path)
        if digest != row["sha256"]:
            raise ValueError(f"V80 fresh file bytes differ: {label}")
        hashes[label] = digest
    result: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "complete_code_only_V80_single_candidate_preflight",
        "V79_program_sha256": program["V79_program_sha256"],
        "candidate_program": str(program_path.resolve()),
        "candidate_program_sha256": sha256_file(program_path),
        "candidate_freeze_commit": freeze_commit,
        "code_commit": commit,
        "worktree_clean": clean,
        "fresh_file_byte_hashes": hashes,
        "selected_payload_accessed_before_candidate_freeze": False,
        "candidate_program_frozen_and_pushed_before_sampling": True,
        "selected_input_or_target_dataset_read_by_preflight": False,
        "preflight_pass": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def _new_ensemble(handle: h5py.File) -> dict[str, h5py.Dataset]:
    return {
        "sample": handle.create_dataset(
            "sample", shape=(QUERIES, MEMBERS, 1, GRID, GRID, GRID), dtype="f4",
            chunks=(1, 1, 1, GRID, GRID, GRID), compression="lzf",
        ),
        "conditional_mean": handle.create_dataset(
            "conditional_mean", shape=(QUERIES, 1, GRID, GRID, GRID), dtype="f4",
            compression="lzf",
        ),
        "truth": handle.create_dataset(
            "truth", shape=(QUERIES, 1, GRID, GRID, GRID), dtype="f4",
            compression="lzf",
        ),
        "source_index": handle.create_dataset("source_index", shape=(QUERIES,), dtype="i8"),
        "initial_latent_sha256": handle.create_dataset(
            "initial_latent_sha256", shape=(QUERIES, MEMBERS, 32), dtype="u1"
        ),
        "maximum_inverse_CDF_error": handle.create_dataset(
            "maximum_inverse_CDF_error", shape=(QUERIES, MEMBERS), dtype="f4"
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
            "rank_disagreement_fraction_excluding_marginal_ties", shape=(QUERIES,), dtype="f4"
        ),
    }


def _load_calibration(program: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    path = Path(program["frozen_artifacts"]["V80_calibration"]["path"])
    output = {}
    with np.load(path, allow_pickle=False) as values:
        if str(values["program_sha256"]) != program["calibration_program_sha256"]:
            raise ValueError("V80 calibration program binding differs")
        for domain in DOMAIN_ORDER:
            source = np.asarray(values[f"{domain}__source_knots_y"], dtype=np.float64)
            mapped = np.asarray(values[f"{domain}__mapped_knots_y"], dtype=np.float64)
            if (
                len(source) < 2
                or source.shape != mapped.shape
                or not np.isfinite(source).all()
                or not np.isfinite(mapped).all()
                or np.any(np.diff(source) <= 0)
                or np.any(np.diff(mapped) < 0)
            ):
                raise ValueError(f"V80 {domain} calibration differs")
            output[domain] = source, mapped
    return output


def _calibrate_and_project(
    total_field: np.ndarray,
    mean: np.ndarray,
    source_knots: np.ndarray,
    mapped_knots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mapped = apply_monotone_map(total_field, source_knots, mapped_knots)
    residual = mapped - np.asarray(mean, dtype=np.float64)
    residual -= residual.mean(axis=(-3, -2, -1), keepdims=True, dtype=np.float64)
    output = (np.asarray(mean, dtype=np.float64) + residual).astype(np.float32)
    dc = np.abs(
        (output.astype(np.float64) - np.asarray(mean, dtype=np.float64)).mean(
            axis=(-3, -2, -1), dtype=np.float64
        )
    )
    return output, dc


@torch.inference_mode()
def sample(program_path: Path, preflight_path: Path, repo: Path, output_root: Path) -> None:
    repo = repo.resolve()
    program = load_program(program_path, repo)
    preflight = strict_json(preflight_path)
    commit, clean = git_state(repo)
    freeze_commit = candidate_freeze_commit(program_path, repo)
    if (
        not clean
        or not _is_ancestor(repo, freeze_commit, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
        or not torch.cuda.is_available()
        or "ada" not in torch.cuda.get_device_name(0).lower()
    ):
        raise RuntimeError("V80 sampling requires clean frozen Lageunha Ada")
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("preflight_pass") is not True
        or preflight.get("candidate_program_sha256") != sha256_file(program_path)
        or canonical_digest(preflight) != preflight.get("decision_digest_sha256")
    ):
        raise ValueError("V80 preflight differs")
    if output_root.resolve() != Path(program["outputs"]["ensemble_root"]).resolve() or output_root.exists():
        raise FileExistsError("V80 sampling refuses an existing or differing output")

    train_program, v70, v35, _ = load_v70_train_gate_program(
        (repo / TRAIN_GATE_PROGRAM).resolve(), repo
    )
    device = torch.device("cuda")
    model, _, checkpoint_sha, _ = _load_fit(train_program, repo, commit, device)
    marginal, inherited_v35, prepared = _frozen_marginal(v70, repo, commit, device)
    if inherited_v35["development_domains"] != v35["development_domains"]:
        raise ValueError("V80 inherited validation definition differs")
    artifacts = program["frozen_artifacts"]
    if checkpoint_sha != artifacts["V70_checkpoint"]["sha256"]:
        raise ValueError("V80 candidate checkpoint differs")
    calibration = _load_calibration(program)
    schedule = sigma_schedule(40, 0.002, 40.0, 7.0, device=device)
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    peak = 0
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for domain in DOMAIN_ORDER:
            indices = np.asarray(program["single_use_fresh_selection"][domain], dtype=np.int64)
            row = v35["development_domains"][domain]
            binding = program["fresh_file_bindings"]
            if (
                Path(row["validation_data"]).resolve() != Path(binding[f"{domain}_source_data"]["path"]).resolve()
                or Path(row["validation_cache"]).resolve() != Path(binding[f"{domain}_source_cache"]["path"]).resolve()
            ):
                raise ValueError(f"V80 {domain} validation path differs")
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
                seed = int(program["frozen_execution_provenance"]["seeds_by_domain"][domain]["candidate"])
                expected_digest = program["frozen_execution_provenance"]["innovation_pairing_digests"][domain]
                digest_table = innovation_digest_table(seed, domain)
                if hashlib.sha256(digest_table.tobytes()).hexdigest() != expected_digest:
                    raise ValueError(f"V80 {domain} fixed innovation digest differs")
                source_knots, mapped_knots = calibration[domain]
                for position, source_index in enumerate(indices):
                    condition, _, backbone = condition_cube(
                        data, cache, prepared, domain, "validation", int(source_index)
                    )
                    truth = np.asarray(data["target"][int(source_index)], dtype=np.float32)
                    condition_tensor = torch.from_numpy(condition[None]).to(device)
                    parameters = marginal(condition_tensor).float()
                    innovation_rows = [
                        innovation_numpy(seed, domain, position, member)
                        for member in range(MEMBERS)
                    ]
                    innovation = torch.from_numpy(np.stack(innovation_rows)).to(device)
                    raw_parts = []
                    for start in range(0, MEMBERS, 4):
                        expanded = condition_tensor.expand(4, -1, -1, -1, -1)
                        raw_parts.append(
                            heun_sample(model, expanded, innovation[start:start + 4], schedule)
                        )
                    raw_latent = torch.cat(raw_parts, dim=0)
                    score = torch.from_numpy(
                        (np.asarray(backbone[0], dtype=np.float32) + target_mean)[None]
                    ).to(device)
                    positions = conditioning_strata(score)
                    sqt_latent, sqt = spatial_quantile_transport(
                        raw_latent, innovation, positions
                    )
                    parameter_batch = parameters.expand(MEMBERS, -1, -1, -1, -1)
                    for arm, latent in (("candidate", sqt_latent), ("control", innovation)):
                        uniform = standard_normal_cdf(latent)
                        standardized = bounded_mixture_inverse(parameter_batch, uniform)
                        inverse_error = torch.amax(
                            torch.abs(
                                bounded_mixture_cdf(parameter_batch, standardized)
                                - uniform.clamp(1.0e-7, 1.0 - 1.0e-7)
                            ),
                            dim=(-4, -3, -2, -1),
                        ).cpu().numpy()
                        physical = standardized.cpu().numpy().astype(np.float64) * target_std + target_mean
                        projected, _ = project_residual_dc(physical)
                        total = np.asarray(backbone, dtype=np.float64)[None] + projected
                        calibrated, dc = _calibrate_and_project(
                            total, backbone, source_knots, mapped_knots
                        )
                        if (
                            calibrated.shape != (MEMBERS, 1, GRID, GRID, GRID)
                            or not np.isfinite(calibrated).all()
                            or float(dc.max()) > 1.0e-7
                        ):
                            raise RuntimeError("V80 calibrated field invariant differs")
                        datasets[arm]["sample"][position] = calibrated
                        datasets[arm]["conditional_mean"][position] = backbone
                        datasets[arm]["truth"][position] = truth
                        datasets[arm]["initial_latent_sha256"][position] = digest_table[position]
                        datasets[arm]["maximum_inverse_CDF_error"][position] = inverse_error
                        datasets[arm]["maximum_absolute_residual_DC"][position] = dc
                        datasets[arm]["pre_inverse_stratum_multiset_equal"][position] = int(sqt["pre_inverse_stratum_multiset_equal"])
                        datasets[arm]["maximum_pre_inverse_stratum_multiset_error"][position] = float(sqt["maximum_pre_inverse_stratum_multiset_error"])
                        datasets[arm]["marginal_tied_voxel_fraction"][position] = float(sqt["marginal_tied_voxel_fraction"])
                        datasets[arm]["rank_disagreement_fraction_excluding_marginal_ties"][position] = float(sqt["rank_disagreement_fraction_excluding_marginal_ties"])
                    print(f"[v80-sample] {domain} {position + 1}/{QUERIES}", flush=True)
                peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
                contracts = program["frozen_domain_execution_contracts"][domain]
                for arm in ARMS:
                    handles[arm].attrs["schema"] = ENSEMBLE_SCHEMA
                    handles[arm].attrs["arm"] = arm
                    handles[arm].attrs["candidate_program_sha256"] = sha256_file(program_path)
                    handles[arm].attrs["calibration_sha256"] = artifacts["V80_calibration"]["sha256"]
                    for key, value in contracts[f"{arm}_expected_attrs"].items():
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
    print(json.dumps({"status": "complete_single_V80_candidate_control_sampling", "peak_allocated_bytes": peak}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--program", type=Path, required=True)
    preflight_parser.add_argument("--repo", type=Path, required=True)
    preflight_parser.add_argument("--output-root", type=Path, required=True)
    preflight_parser.add_argument("--out", type=Path, required=True)
    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--program", type=Path, required=True)
    sample_parser.add_argument("--preflight", type=Path, required=True)
    sample_parser.add_argument("--repo", type=Path, required=True)
    sample_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = validate_without_selected_payload(args.program, args.repo, args.output_root)
        if args.out.exists():
            raise FileExistsError("V80 preflight refuses existing report")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    else:
        sample(args.program, args.preflight, args.repo, args.output_root)


if __name__ == "__main__":
    main()
