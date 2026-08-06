#!/usr/bin/env python
"""Create and verify the committed V20/Astrid one-shot artifact seal."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from hong2021_v14_edm import V20_E8_SCHEMA, decoder_upsampling_for_schema
from hong2021_v14_freeze import _file_row, _load_json, _run_git, _tracked_protocol_rows, sha256
from hong2021_v15_development_gate import canonical_digest
from hong2021_v20_edm import (
    FROZEN_REGISTRY_SHA256,
    P_MEAN,
    P_STD,
    _validate_checkpoint,
    load_frozen_registry,
)


SCHEMA = "hong2021-v20-astrid-one-shot-artifact-seal-v1"
DEFAULT_REPO = Path("/home/kjhan/BACKUP/CF4")
DEFAULT_TNG = Path("/gpfs/kjhan/IllustrisTNG/TNG100-1")
DEFAULT_ASTRID = Path("/gpfs/kjhan/CAMELS/Astrid/L25n256")
DEFAULT_DESTINATION = Path("config/hong2021_v20_astrid_one_shot_seal.json")
EXACT_ONE_SHOT_COMMAND = (
    "scripts/run_hong2021_v20_astrid_one_shot_lageunha.sh "
    "config/hong2021_v20_astrid_one_shot_seal.json"
)


def _selected_artifacts(
    tng: Path, repo: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Any]]:
    sequence_path = tng / "evaluation/tng100_simba_swift_v20_sequence/status.json"
    sequence = _load_json(sequence_path)
    if sequence.get("state") != "complete_e8_passed_astrid_still_unopened":
        raise RuntimeError("V20 has no completed passing development gate")
    if sequence.get("independent_data_paths_accessed") is not False:
        raise RuntimeError("V20 sequence does not preserve the independent-data firewall")
    decision_path = Path(sequence["detail"]).resolve()
    decision = _load_json(decision_path)
    if decision.get("experiment") != "e8_gaussianized_marginal_retrain":
        raise RuntimeError("V20 sequence refers to the wrong experiment")
    if decision.get("development_pass") is not True:
        raise RuntimeError("V20 development decision is not passing")
    if decision.get("next") != "freeze_exact_v20_hashes_before_astrid_one_shot":
        raise RuntimeError("V20 decision does not authorize the Astrid seal")
    if canonical_digest(decision) != decision.get("decision_digest_sha256"):
        raise RuntimeError("V20 development decision digest is invalid")
    if decision.get("Astrid_used") is not False or decision.get(
        "independent_data_paths_accessed_by_gate"
    ) is not False:
        raise RuntimeError("V20 decision does not prove independent data remained unused")
    registry_path = repo / "config/hong2021_v20_development_program.json"
    registry = load_frozen_registry(registry_path, repo)
    if decision.get("registry_sha256") != FROZEN_REGISTRY_SHA256:
        raise RuntimeError("V20 decision used a different frozen registry")
    selected_step = int(decision["selected_step"])
    selected = next(
        row for row in decision["candidates"] if int(row["step"]) == selected_step
    )
    checkpoint_path = Path(decision["selected_checkpoint"]).resolve()
    checkpoint, checkpoint_sha = _validate_checkpoint(
        checkpoint_path, step=selected_step, registry=registry
    )
    if selected["checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError("selected V20 checkpoint changed after the gate")
    correction_root = tng / "training/tng100_simba_swift_v14_mean_correction"
    model_root = tng / "derived/hong2021_v14/model"
    correction_selection_path = correction_root / "selection.json"
    preparation_path = model_root / "preparation_status.json"
    correction_selection = _load_json(correction_selection_path)
    preparation = _load_json(preparation_path)
    if correction_selection.get("development_pass") is not True:
        raise RuntimeError("frozen mean correction did not pass development selection")
    if preparation.get("state") != "complete":
        raise RuntimeError("V14 common residual preparation is incomplete")
    experiment = registry["e8_gaussianized_marginal_retrain"]
    training_root = Path(decision["training"]).resolve()
    artifacts = {
        "deterministic_v4": (
            tng / "training/tng100_v4_split00_l0_groupnorm_std_cosine"
            / "minimum_validation_loss.pt"
        ).resolve(),
        "mean_correction": Path(correction_selection["selected_checkpoint"]).resolve(),
        "location_scale": Path(preparation["location_scale_model"]).resolve(),
        "edm": checkpoint_path,
        "hop": Path("/home/kjhan/BACKUP/lagRamses-de-nonstd/utils/f90/hop_ramses/hop"),
        "regroup": Path("/home/kjhan/BACKUP/lagRamses-de-nonstd/utils/f90/hop_ramses/regroup"),
    }
    provenance = {
        "v20_sequence": sequence_path,
        "development_decision": decision_path,
        "v20_registry": registry_path,
        "v19_registry": repo / registry["parent_evidence"]["v19_registry"],
        "independent_audit": Path(registry["independent_audit"]["record"]),
        "hard_preflight": sequence_path.parent / "preflight.json",
        "training_run": training_root / "run.json",
        "gaussianization_transform": Path(experiment["gaussianization"]["path"]),
        "initialization_measurement": Path(
            experiment["initialization_and_normalization"]["measurement_report"]
        ),
        "mean_correction_selection": correction_selection_path,
        "model_preparation_status": preparation_path,
    }
    for domain in ("TNG100", "SIMBA", "Swift"):
        for split in ("train", "validation"):
            provenance[f"v20_cache_{domain}_{split}"] = Path(
                experiment["derived_caches"][domain][split]["path"]
            )
    return artifacts, provenance, {
        "decision": decision,
        "selected": selected,
        "registry": registry,
        "checkpoint": checkpoint,
    }


def create_seal(
    *, repo: Path, tng: Path, astrid_root: Path, destination: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    if _run_git(repo, "status", "--porcelain"):
        raise RuntimeError("refusing to seal a dirty worktree")
    output = destination if destination.is_absolute() else repo / destination
    if output.exists():
        raise RuntimeError(f"refusing to overwrite seal: {output}")
    # Deliberately do not stat/list/read astrid_root.  The V20 sequence and
    # decision are integrity-bound records of the previously established
    # zero-file state, and the independent-data path stays untouched until
    # the one-file seal commit exists.
    artifacts, provenance, selection = _selected_artifacts(tng.resolve(), repo)
    artifact_rows = {name: _file_row(path) for name, path in artifacts.items()}
    provenance_rows = {name: _file_row(path) for name, path in provenance.items()}
    checkpoint = selection["checkpoint"]
    correction = torch.load(artifacts["mean_correction"], map_location="cpu", weights_only=False)
    deterministic = torch.load(
        artifacts["deterministic_v4"], map_location="cpu", weights_only=False
    )
    location = _load_json(artifacts["location_scale"])
    if checkpoint.get("schema") != V20_E8_SCHEMA:
        raise ValueError("selected EDM is not the frozen V20-E8 model")
    if decoder_upsampling_for_schema(checkpoint["schema"]) != "nearest":
        raise ValueError("selected V20 model does not use nearest decoding")
    if checkpoint.get("experiment_registry_sha256") != FROZEN_REGISTRY_SHA256:
        raise ValueError("selected V20 model has wrong registry provenance")
    if checkpoint.get("edm_p_mean") != P_MEAN or checkpoint.get("edm_p_std") != P_STD:
        raise ValueError("selected V20 model has wrong E3 relative-noise constants")
    if checkpoint.get("edm_p_mean_mode") != "sigma_data_fraction":
        raise ValueError("selected V20 model has wrong p_mean mode")
    if correction.get("schema") != "hong2021-v14-three-domain-zero-dc-mean-correction-v1":
        raise ValueError("unexpected mean-correction checkpoint schema")
    if location.get("schema") != "hong2021-v14-three-domain-location-scale-model-v1":
        raise ValueError("unexpected location-scale model schema")
    code_commit = _run_git(repo, "rev-parse", "HEAD")
    decision = selection["decision"]
    for label, commit in (
        ("training", checkpoint["code_commit_at_launch"]),
        ("gate", decision["gate_code_commit"]),
    ):
        if subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, code_commit]
        ).returncode:
            raise RuntimeError(f"{label} commit is not an ancestor of seal code")
    preprocessing = deterministic.get("input_preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("deterministic checkpoint has no frozen preprocessing")
    preprocessing_bytes = json.dumps(
        preprocessing, sort_keys=True, separators=(",", ":")
    ).encode()
    experiment = selection["registry"]["e8_gaussianized_marginal_retrain"]
    initialization = experiment["initialization_and_normalization"]
    record = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit_before_seal_record": code_commit,
        "branch": _run_git(repo, "branch", "--show-current"),
        "development_gate": {
            "passed": True,
            "experiment": "e8_gaussianized_marginal_retrain",
            "selected_edm_step": int(checkpoint["step"]),
            "training_code_commit": checkpoint["code_commit_at_launch"],
            "gate_code_commit": decision["gate_code_commit"],
            "decision_digest_sha256": decision["decision_digest_sha256"],
            "selection": provenance_rows["development_decision"],
            "mechanism_10k": decision["mechanism_10k"],
            "diagnostics": selection["selected"]["domains"],
        },
        "artifacts": artifact_rows,
        "provenance": provenance_rows,
        "tracked_protocol_files": _tracked_protocol_rows(repo),
        "model_invariants": {
            "mean_correction_step": int(correction["step"]),
            "edm_step": int(checkpoint["step"]),
            "edm_schema": checkpoint["schema"],
            "decoder_upsampling": "nearest",
            "decoder_align_corners": None,
            "edm_sigma_data": float(checkpoint["sigma_data"]),
            "edm_p_mean": float(checkpoint["edm_p_mean"]),
            "edm_p_std": float(checkpoint["edm_p_std"]),
            "edm_p_mean_mode": checkpoint["edm_p_mean_mode"],
            "tail_exponent": float(
                checkpoint["tail_weight_fit"]["inverse_probability_exponent"]
            ),
            "gaussianization_transform_sha256": experiment["gaussianization"]["sha256"],
            "latent_dc_restored_at_sampling": False,
            "input_preprocessing": preprocessing,
            "input_preprocessing_canonical_sha256": hashlib.sha256(
                preprocessing_bytes
            ).hexdigest(),
            "simulation_identity_feature": False,
            "inference_target_feature": False,
            "posthoc_transfer": False,
        },
        "initialization": {
            "registry_sha256": FROZEN_REGISTRY_SHA256,
            "schema": "hong2021-v18-prior-matched-spectral-initialization-v1",
            "source_balanced_per_band_mode_variance": initialization[
                "source_balanced_band_mode_variance"
            ],
            "measurement_report_sha256": initialization["measurement_report_sha256"],
            "expected_mode_counts_by_grid": {
                "64": initialization["mode_counts_64"],
                "80": initialization["mode_counts_80"],
            },
            "inference_mapping": initialization["inference_mapping"],
            "additional_rng_draws": 0,
            "free_parameters_added": 0,
        },
        "astrid_preopen": {
            # Record the preregistered spelling without resolving or probing it.
            "root": str(astrid_root),
            "files_found_in_previously_integrity_bound_state": 0,
            "path_accessed_during_v20_development_or_seal_creation": False,
            "content_read": False,
        },
        "astrid_raw_units": {
            "snapshot_coordinates": "Gadget kpc/h",
            "catalog_coordinates": "Gadget kpc/h",
            "coordinate_scale_to_mpc_h": 0.001,
            "box_mpc_h": 25.0,
            "operator": "periodic cell-centred CIC on 80^3",
        },
        "one_shot": {
            "command": EXACT_ONE_SHOT_COMMAND,
            "realizations": list(range(27)),
            "observer_rule": "one per realization; closest in log stellar mass to geometric midpoint of (4e10,1e11) Msun; subhalo-index tie break",
            "indices": list(range(27)),
            "ensemble": 16,
            "sampling_steps": 40,
            "sigma_min": 0.002,
            "sigma_max": 40.0,
            "rho": 7.0,
            "seed": 28777,
            "initializer": "sealed V20 prior-matched 80^3 physical-k mapping",
            "gaussianization": "apply sealed train-only G and Addendum-B DC projection to the common 80^3 cache, then apply sealed G^-1 before V14 inverse",
            "field_checks": "unchanged eight checks from hong2021_v6_gate.field_gate",
            "grid_hop_only_after_field_pass": True,
            "hop_members": 16,
            "hop_objects": 27,
            "hop_omega_m": 0.3,
            "hop_bootstrap": 50000,
            "hop_bootstrap_seed": 2024,
            "no_alternate_after_opening": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "torch": torch.__version__,
        },
        "commit_requirement": "This record must be the only file in a commit directly atop the frozen V20 code before any Astrid path is accessed.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, output)
    return record


def verify_seal(
    seal: str | Path, *, repo: str | Path = DEFAULT_REPO,
    require_committed: bool = True, require_unopened: bool = False,
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    path = Path(seal)
    if not path.is_absolute():
        path = repo / path
    path = path.resolve()
    record = _load_json(path)
    if record.get("schema") != SCHEMA:
        raise ValueError("not a V20 Astrid one-shot seal")
    if record.get("development_gate", {}).get("passed") is not True:
        raise RuntimeError("V20 seal does not record a passing development gate")
    invariants = record.get("model_invariants", {})
    if invariants.get("edm_schema") != V20_E8_SCHEMA:
        raise RuntimeError("V20 seal does not retain the E8 checkpoint")
    if invariants.get("edm_p_mean") != P_MEAN or invariants.get("edm_p_std") != P_STD:
        raise RuntimeError("V20 seal does not retain the E3 relative-noise distribution")
    if invariants.get("edm_p_mean_mode") != "sigma_data_fraction":
        raise RuntimeError("V20 seal does not retain the relative p_mean mode")
    initialization = record.get("initialization", {})
    if initialization.get("registry_sha256") != FROZEN_REGISTRY_SHA256:
        raise RuntimeError("V20 seal does not retain the frozen registry")
    if initialization.get("expected_mode_counts_by_grid") != {
        "64": [146, 3596, 25296, 233105],
        "80": [250, 6872, 49928, 454949],
    }:
        raise RuntimeError("V20 seal does not retain both frozen Fourier grids")
    if _run_git(repo, "status", "--porcelain"):
        raise RuntimeError("the repository is dirty; sealed execution is forbidden")
    code_commit = record["code_commit_before_seal_record"]
    if subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", code_commit, "HEAD"]
    ).returncode:
        raise RuntimeError("the sealed V20 code commit is not an ancestor of HEAD")
    if require_committed:
        relative = path.relative_to(repo).as_posix()
        commits = _run_git(repo, "log", "--format=%H", "--follow", "--", relative).splitlines()
        if len(commits) != 1:
            raise RuntimeError("the V20 seal must be added once and never modified")
        seal_commit = commits[0]
        if _run_git(repo, "rev-parse", f"{seal_commit}^") != code_commit:
            raise RuntimeError("the seal commit is not directly atop frozen V20 code")
        changed = _run_git(
            repo, "diff-tree", "--no-commit-id", "--name-only", "-r", seal_commit
        ).splitlines()
        if changed != [relative]:
            raise RuntimeError("the seal commit must contain only the V20 seal record")
        committed = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{relative}"],
            check=True, capture_output=True,
        ).stdout
        if committed != path.read_bytes():
            raise RuntimeError("the working V20 seal differs from its committed blob")
    for group in ("artifacts", "provenance"):
        for name, row in record[group].items():
            artifact = Path(row["path"])
            if not artifact.is_file() or artifact.stat().st_size != row["bytes"]:
                raise RuntimeError(f"sealed {group} file changed or vanished: {name}")
            if sha256(artifact) != row["sha256"]:
                raise RuntimeError(f"sealed {group} hash mismatch: {name}")
    for row in record["tracked_protocol_files"]:
        artifact = repo / row["path"]
        if not artifact.is_file() or artifact.stat().st_size != row["bytes"]:
            raise RuntimeError(f"sealed protocol file changed: {row['path']}")
        if sha256(artifact) != row["sha256"]:
            raise RuntimeError(f"sealed protocol hash mismatch: {row['path']}")
    if record["one_shot"]["command"] != EXACT_ONE_SHOT_COMMAND:
        raise RuntimeError("V20 one-shot command differs from verifier")
    if require_unopened and record["astrid_preopen"].get(
        "path_accessed_during_v20_development_or_seal_creation"
    ) is not False:
        raise RuntimeError("V20 seal does not retain the no-path-access firewall")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    create = sub.add_parser("create")
    create.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    create.add_argument("--tng", type=Path, default=DEFAULT_TNG)
    create.add_argument("--astrid-root", type=Path, default=DEFAULT_ASTRID)
    create.add_argument("--out", type=Path, default=DEFAULT_DESTINATION)
    verify = sub.add_parser("verify")
    verify.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    verify.add_argument("--seal", type=Path, default=DEFAULT_DESTINATION)
    verify.add_argument("--require-unopened", action="store_true")
    args = parser.parse_args()
    if args.mode == "create":
        print(json.dumps(create_seal(
            repo=args.repo, tng=args.tng, astrid_root=args.astrid_root,
            destination=args.out,
        ), indent=2))
    else:
        record = verify_seal(
            args.seal, repo=args.repo, require_unopened=args.require_unopened
        )
        print(json.dumps({"schema": record["schema"], "verified": True}, indent=2))


if __name__ == "__main__":
    main()
