#!/usr/bin/env python
"""Create and verify the committed V19/Astrid one-shot artifact seal."""
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

from hong2021_v14_edm import V19_E7_SCHEMA, decoder_upsampling_for_schema
from hong2021_v14_freeze import (
    _file_row,
    _load_json,
    _run_git,
    _tracked_protocol_rows,
    astrid_files,
    sha256,
)
from hong2021_v15_development_gate import canonical_digest
from hong2021_v18_init import EXPECTED_MODE_COUNTS_BY_GRID
from hong2021_v19_edm import FROZEN_REGISTRY_SHA256, P_MEAN, P_STD, load_frozen_registry


SCHEMA = "hong2021-v19-astrid-one-shot-artifact-seal-v1"
DEFAULT_REPO = Path("/home/kjhan/BACKUP/CF4")
DEFAULT_TNG = Path("/gpfs/kjhan/IllustrisTNG/TNG100-1")
DEFAULT_ASTRID = Path("/gpfs/kjhan/CAMELS/Astrid/L25n256")
DEFAULT_DESTINATION = Path("config/hong2021_v19_astrid_one_shot_seal.json")
EXACT_ONE_SHOT_COMMAND = (
    "scripts/run_hong2021_v19_astrid_one_shot_lageunha.sh "
    "config/hong2021_v19_astrid_one_shot_seal.json"
)


def _selected_artifacts(
    tng: Path, repo: Path
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Any]]:
    sequence_path = tng / "evaluation/tng100_simba_swift_v19_sequence/status.json"
    sequence = _load_json(sequence_path)
    if sequence.get("state") != "complete_e7_passed_astrid_still_unopened":
        raise RuntimeError("V19 has no completed passing development gate")
    decision_path = Path(sequence["detail"]).resolve()
    decision = _load_json(decision_path)
    if decision.get("experiment") != "e7_band_anchored_noise_retrain":
        raise RuntimeError("V19 sequence refers to the wrong experiment")
    if decision.get("development_pass") is not True:
        raise RuntimeError("V19 development decision is not passing")
    if decision.get("next") != "freeze_exact_v19_hashes_before_astrid_one_shot":
        raise RuntimeError("V19 decision does not authorize the Astrid seal")
    if canonical_digest(decision) != decision.get("decision_digest_sha256"):
        raise RuntimeError("V19 development decision digest is invalid")
    if decision.get("Astrid_used") is not False:
        raise RuntimeError("V19 decision does not prove Astrid remained unused")
    registry_path = repo / "config/hong2021_v19_development_program.json"
    registry = load_frozen_registry(registry_path, repo)
    if decision.get("registry_sha256") != FROZEN_REGISTRY_SHA256:
        raise RuntimeError("V19 decision used a different frozen registry")
    selected_step = int(decision["selected_step"])
    selected = next(
        row for row in decision["candidates"] if int(row["step"]) == selected_step
    )
    checkpoint_path = Path(decision["selected_checkpoint"]).resolve()
    if selected["checkpoint_sha256"] != sha256(checkpoint_path):
        raise RuntimeError("selected V19 checkpoint changed after the gate")
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
    experiment = registry["e7_band_anchored_noise_retrain"]
    provenance = {
        "v19_sequence": sequence_path,
        "development_decision": decision_path,
        "v19_registry": registry_path,
        "v18_registry": repo / registry["parent_evidence"]["v18_registry"],
        "independent_audit": Path(registry["independent_audit"]["record"]),
        "hard_preflight": sequence_path.parent / "preflight.json",
        "training_run": training_root / "run.json",
        "initialization_measurement": Path(
            experiment["initialization"]["measurement_report"]
        ),
        "mean_correction_selection": correction_selection_path,
        "model_preparation_status": preparation_path,
    }
    return artifacts, provenance, {
        "decision": decision, "selected": selected,
        "registry": registry, "checkpoint_path": checkpoint_path,
    }


def create_seal(
    *, repo: Path, tng: Path, astrid_root: Path, destination: Path
) -> dict[str, Any]:
    repo = repo.resolve()
    if _run_git(repo, "status", "--porcelain"):
        raise RuntimeError("refusing to seal a dirty worktree")
    output = destination if destination.is_absolute() else repo / destination
    if output.exists():
        raise RuntimeError(f"refusing to overwrite seal: {output}")
    if astrid_files(astrid_root):
        raise RuntimeError("Astrid must remain unopened before the V19 seal commit")
    artifacts, provenance, selection = _selected_artifacts(tng.resolve(), repo)
    artifact_rows = {name: _file_row(path) for name, path in artifacts.items()}
    provenance_rows = {name: _file_row(path) for name, path in provenance.items()}
    if artifact_rows["edm"]["sha256"] != selection["selected"]["checkpoint_sha256"]:
        raise RuntimeError("selected V19 checkpoint hash differs from decision")
    correction = torch.load(
        artifacts["mean_correction"], map_location="cpu", weights_only=False
    )
    edm = torch.load(artifacts["edm"], map_location="cpu", weights_only=False)
    deterministic = torch.load(
        artifacts["deterministic_v4"], map_location="cpu", weights_only=False
    )
    location = _load_json(artifacts["location_scale"])
    if edm.get("schema") != V19_E7_SCHEMA:
        raise ValueError("selected EDM is not the frozen V19-E7 model")
    if decoder_upsampling_for_schema(edm["schema"]) != "nearest":
        raise ValueError("selected V19 model does not use nearest decoding")
    if edm.get("experiment_registry_sha256") != FROZEN_REGISTRY_SHA256:
        raise ValueError("selected V19 model has wrong registry provenance")
    if float(edm.get("edm_p_mean", np.nan)) != P_MEAN:
        raise ValueError("selected V19 model has wrong p_mean")
    if float(edm.get("edm_p_std", np.nan)) != P_STD:
        raise ValueError("selected V19 model has wrong p_std")
    if correction.get("schema") != "hong2021-v14-three-domain-zero-dc-mean-correction-v1":
        raise ValueError("unexpected mean-correction checkpoint schema")
    if location.get("schema") != "hong2021-v14-three-domain-location-scale-model-v1":
        raise ValueError("unexpected location-scale model schema")
    code_commit = _run_git(repo, "rev-parse", "HEAD")
    decision = selection["decision"]
    for label, commit in (
        ("training", edm["code_commit_at_launch"]),
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
    experiment = selection["registry"]["e7_band_anchored_noise_retrain"]
    init = experiment["initialization"]
    record = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit_before_seal_record": code_commit,
        "branch": _run_git(repo, "branch", "--show-current"),
        "development_gate": {
            "passed": True,
            "experiment": "e7_band_anchored_noise_retrain",
            "selected_edm_step": int(edm["step"]),
            "training_code_commit": edm["code_commit_at_launch"],
            "gate_code_commit": decision["gate_code_commit"],
            "decision_digest_sha256": decision["decision_digest_sha256"],
            "selection": provenance_rows["development_decision"],
            "score_spectroscopy_Q1": decision["score_spectroscopy_Q1"],
            "diagnostics": selection["selected"]["domains"],
        },
        "artifacts": artifact_rows,
        "provenance": provenance_rows,
        "tracked_protocol_files": _tracked_protocol_rows(repo),
        "model_invariants": {
            "mean_correction_step": int(correction["step"]),
            "edm_step": int(edm["step"]),
            "edm_schema": edm["schema"],
            "decoder_upsampling": "nearest",
            "decoder_align_corners": None,
            "edm_sigma_data": float(edm["sigma_data"]),
            "edm_p_mean": float(edm["edm_p_mean"]),
            "edm_p_std": float(edm["edm_p_std"]),
            "edm_p_mean_mode": edm["edm_p_mean_mode"],
            "tail_exponent": float(
                edm["tail_weight_fit"]["inverse_probability_exponent"]
            ),
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
            "schema": init["schema"],
            "source_balanced_per_band_mode_variance": init[
                "source_balanced_per_band_mode_variance"
            ],
            "measurement_report_sha256": init["measurement_report_sha256"],
            "expected_mode_counts_by_grid": init["expected_mode_counts_by_grid"],
            "inference_mapping": init["inference_mapping"],
            "additional_rng_draws": 0,
            "free_parameters_added": 0,
        },
        "astrid_preopen": {
            "root": str(astrid_root.resolve()),
            "files_found": 0,
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
            "initializer": "sealed V19 prior-matched 80^3 physical-k mapping",
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
            "numpy": np.__version__, "h5py": h5py.__version__, "torch": torch.__version__,
        },
        "commit_requirement": "This record must be the only file in a commit directly atop the frozen V19 code before any Astrid file is downloaded or opened.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(record, indent=2))
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
        raise ValueError("not a V19 Astrid one-shot seal")
    if record.get("development_gate", {}).get("passed") is not True:
        raise RuntimeError("seal does not record a passing development gate")
    invariants = record.get("model_invariants", {})
    if invariants.get("edm_schema") != V19_E7_SCHEMA:
        raise RuntimeError("seal does not retain the V19-E7 checkpoint")
    if invariants.get("decoder_upsampling") != "nearest":
        raise RuntimeError("seal does not retain nearest decoding")
    if invariants.get("edm_p_mean") != P_MEAN or invariants.get("edm_p_std") != P_STD:
        raise RuntimeError("seal does not retain the V19 noise distribution")
    init = record.get("initialization", {})
    if init.get("registry_sha256") != FROZEN_REGISTRY_SHA256:
        raise RuntimeError("seal does not retain the frozen V19 registry")
    if init.get("expected_mode_counts_by_grid") != {
        str(grid): list(counts) for grid, counts in EXPECTED_MODE_COUNTS_BY_GRID.items()
    }:
        raise RuntimeError("seal does not retain both frozen Fourier grids")
    if _run_git(repo, "status", "--porcelain"):
        raise RuntimeError("the repository is dirty; sealed execution is forbidden")
    code_commit = record["code_commit_before_seal_record"]
    if subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", code_commit, "HEAD"]
    ).returncode:
        raise RuntimeError("the sealed V19 code commit is not an ancestor of HEAD")
    if require_committed:
        relative = path.relative_to(repo).as_posix()
        commits = _run_git(repo, "log", "--format=%H", "--follow", "--", relative).splitlines()
        if len(commits) != 1:
            raise RuntimeError("the V19 seal must be added once and never modified")
        seal_commit = commits[0]
        if _run_git(repo, "rev-parse", f"{seal_commit}^") != code_commit:
            raise RuntimeError("the seal commit is not directly atop frozen V19 code")
        changed = _run_git(
            repo, "diff-tree", "--no-commit-id", "--name-only", "-r", seal_commit
        ).splitlines()
        if changed != [relative]:
            raise RuntimeError("the seal commit must contain only the V19 seal record")
        committed = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{relative}"],
            check=True, capture_output=True,
        ).stdout
        if committed != path.read_bytes():
            raise RuntimeError("the working V19 seal differs from its committed blob")
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
        raise RuntimeError("V19 one-shot command differs from verifier")
    if require_unopened and astrid_files(Path(record["astrid_preopen"]["root"])):
        raise RuntimeError("Astrid was already opened before V19 seal verification")
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
        create_seal(
            repo=args.repo, tng=args.tng, astrid_root=args.astrid_root,
            destination=args.out,
        )
    else:
        record = verify_seal(
            args.seal, repo=args.repo, require_unopened=args.require_unopened
        )
        print(json.dumps({"schema": record["schema"], "verified": True}, indent=2))


if __name__ == "__main__":
    main()
