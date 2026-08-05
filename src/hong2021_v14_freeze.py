#!/usr/bin/env python
"""Create and verify the committed V14/Astrid one-shot artifact seal.

The seal is created only after the predeclared three-domain development gate
passes.  Verification is target-blind: it hashes the selected model and code
artifacts but never opens an Astrid file.  The Astrid runner refuses its first
download unless this exact record is already committed in a clean worktree.
"""
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


SCHEMA = "hong2021-v14-astrid-one-shot-artifact-seal-v1"
DEFAULT_REPO = Path("/home/kjhan/BACKUP/CF4")
DEFAULT_TNG = Path("/gpfs/kjhan/IllustrisTNG/TNG100-1")
DEFAULT_ASTRID = Path("/gpfs/kjhan/CAMELS/Astrid/L25n256")
DEFAULT_DESTINATION = Path("config/hong2021_v14_astrid_one_shot_seal.json")
EXACT_ONE_SHOT_COMMAND = (
    "scripts/run_hong2021_v14_astrid_one_shot_lageunha.sh "
    "config/hong2021_v14_astrid_one_shot_seal.json"
)


def _run_git(repo: Path, *arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_row(path: Path, *, repo: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    label = str(path.resolve())
    if repo is not None:
        label = path.resolve().relative_to(repo.resolve()).as_posix()
    return {"path": label, "bytes": path.stat().st_size, "sha256": sha256(path)}


def astrid_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(str(path.resolve()) for path in root.rglob("*") if path.is_file())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _selected_artifacts(tng: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    evaluation = tng / "evaluation/tng100_simba_swift_v14_multiscale_edm"
    correction_root = tng / "training/tng100_simba_swift_v14_mean_correction"
    model_root = tng / "derived/hong2021_v14/model"
    development_path = evaluation / "development_decision.json"
    correction_selection_path = correction_root / "selection.json"
    preparation_path = model_root / "preparation_status.json"
    development = _load_json(development_path)
    correction_selection = _load_json(correction_selection_path)
    preparation = _load_json(preparation_path)
    if development.get("development_pass") is not True:
        raise RuntimeError("the frozen three-domain development gate did not pass")
    if correction_selection.get("development_pass") is not True:
        raise RuntimeError("the frozen mean-correction selection did not pass")
    if preparation.get("state") != "complete":
        raise RuntimeError("V14 model residual preparation is not complete")
    selected_step = int(development["selected_step"])
    edm = Path(development["selected_checkpoint"]).resolve()
    expected_edm = (
        tng
        / "training/tng100_simba_swift_v14_multiscale_edm"
        / "validation_checkpoints"
        / f"step_{selected_step:06d}.pt"
    ).resolve()
    if edm != expected_edm:
        raise RuntimeError("development decision selected an unexpected EDM path")
    artifacts = {
        "deterministic_v4": (
            tng
            / "training/tng100_v4_split00_l0_groupnorm_std_cosine"
            / "minimum_validation_loss.pt"
        ).resolve(),
        "mean_correction": Path(correction_selection["selected_checkpoint"]).resolve(),
        "location_scale": Path(preparation["location_scale_model"]).resolve(),
        "edm": edm,
        "hop": Path(
            "/home/kjhan/BACKUP/lagRamses-de-nonstd/utils/f90/hop_ramses/hop"
        ),
        "regroup": Path(
            "/home/kjhan/BACKUP/lagRamses-de-nonstd/utils/f90/hop_ramses/regroup"
        ),
    }
    provenance = {
        "development_decision": development_path,
        "mean_correction_selection": correction_selection_path,
        "model_preparation_status": preparation_path,
    }
    if Path(preparation["correction_checkpoint"]).resolve() != artifacts["mean_correction"]:
        raise RuntimeError("residual preparation used a different correction checkpoint")
    return artifacts, provenance


def _tracked_protocol_rows(repo: Path) -> list[dict[str, Any]]:
    names = _run_git(repo, "ls-files", "src", "scripts", "config").splitlines()
    rows = []
    for name in names:
        path = repo / name
        if path.is_file():
            rows.append(_file_row(path, repo=repo))
    if not rows:
        raise RuntimeError("no tracked protocol files were found")
    return rows


def create_seal(
    *, repo: Path, tng: Path, astrid_root: Path, destination: Path
) -> dict[str, Any]:
    repo = repo.resolve()
    if _run_git(repo, "status", "--porcelain"):
        raise RuntimeError("refusing to seal a dirty worktree")
    output = destination if destination.is_absolute() else repo / destination
    if output.exists():
        raise RuntimeError(f"refusing to overwrite seal: {output}")
    found = astrid_files(astrid_root)
    if found:
        raise RuntimeError(
            f"Astrid must remain unopened before the seal commit; found {len(found)} files"
        )
    artifacts, provenance = _selected_artifacts(tng.resolve())
    artifact_rows = {name: _file_row(path) for name, path in artifacts.items()}
    provenance_rows = {name: _file_row(path) for name, path in provenance.items()}
    # Validate checkpoint identities without consulting any independent truth.
    deterministic = torch.load(
        artifacts["deterministic_v4"], map_location="cpu", weights_only=False
    )
    correction = torch.load(artifacts["mean_correction"], map_location="cpu", weights_only=False)
    edm = torch.load(artifacts["edm"], map_location="cpu", weights_only=False)
    location = _load_json(artifacts["location_scale"])
    if correction.get("schema") != "hong2021-v14-three-domain-zero-dc-mean-correction-v1":
        raise ValueError("unexpected mean-correction checkpoint schema")
    if edm.get("schema") != "hong2021-v14-three-domain-multiscale-observable-context-edm-v1":
        raise ValueError("unexpected EDM checkpoint schema")
    if location.get("schema") != "hong2021-v14-three-domain-location-scale-model-v1":
        raise ValueError("unexpected location-scale model schema")
    protocol_rows = _tracked_protocol_rows(repo)
    preprocessing = deterministic.get("input_preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("deterministic checkpoint has no frozen input preprocessing")
    preprocessing_bytes = json.dumps(
        preprocessing, sort_keys=True, separators=(",", ":")
    ).encode()
    record = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit_before_seal_record": _run_git(repo, "rev-parse", "HEAD"),
        "branch": _run_git(repo, "branch", "--show-current"),
        "development_gate": {
            "passed": True,
            "selected_edm_step": int(edm["step"]),
            "selection": provenance_rows["development_decision"],
        },
        "artifacts": artifact_rows,
        "provenance": provenance_rows,
        "tracked_protocol_files": protocol_rows,
        "model_invariants": {
            "mean_correction_step": int(correction["step"]),
            "edm_step": int(edm["step"]),
            "edm_sigma_data": float(edm["sigma_data"]),
            "input_preprocessing": preprocessing,
            "input_preprocessing_canonical_sha256": hashlib.sha256(
                preprocessing_bytes
            ).hexdigest(),
            "correction_training_data_record_present": bool(correction.get("data")),
            "simulation_identity_feature": False,
            "inference_target_feature": False,
            "posthoc_transfer": False,
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
            "observer_rule": (
                "one per realization; closest in log stellar mass to geometric "
                "midpoint of (4e10,1e11) Msun; subhalo-index tie break"
            ),
            "indices": list(range(27)),
            "ensemble": 16,
            "sampling_steps": 40,
            "sigma_min": 0.002,
            "sigma_max": 40.0,
            "rho": 7.0,
            "seed": 28777,
            "field_checks": "unchanged eight checks from hong2021_v6_gate.field_gate",
            "grid_hop_only_after_field_pass": True,
            "hop_members": 16,
            "hop_objects": 27,
            "hop_omega_m": 0.3,
            "hop_bootstrap": 50_000,
            "hop_bootstrap_seed": 2024,
            "no_alternate_after_opening": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "torch": torch.__version__,
        },
        "commit_requirement": (
            "This record must be committed unchanged before the first Astrid file "
            "is downloaded or opened. The runner verifies the committed blob."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(record, indent=2))
    return record


def verify_seal(
    seal: str | Path,
    *,
    repo: str | Path = DEFAULT_REPO,
    require_committed: bool = True,
    require_unopened: bool = False,
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    path = Path(seal)
    if not path.is_absolute():
        path = repo / path
    path = path.resolve()
    record = _load_json(path)
    if record.get("schema") != SCHEMA:
        raise ValueError("not a V14 Astrid one-shot seal")
    if record.get("development_gate", {}).get("passed") is not True:
        raise RuntimeError("seal does not record a passing development gate")
    if _run_git(repo, "status", "--porcelain"):
        raise RuntimeError("the repository is dirty; sealed execution is forbidden")
    code_commit = record["code_commit_before_seal_record"]
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", code_commit, "HEAD"]
    )
    if ancestor.returncode:
        raise RuntimeError("the sealed code commit is not an ancestor of HEAD")
    if require_committed:
        relative = path.relative_to(repo).as_posix()
        committed = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if committed != path.read_bytes():
            raise RuntimeError("the working seal differs from the committed seal")
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
        raise RuntimeError("one-shot command differs from the verifier")
    if require_unopened:
        root = Path(record["astrid_preopen"]["root"])
        found = astrid_files(root)
        if found:
            raise RuntimeError(f"Astrid was already opened; found {len(found)} files")
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
        print(json.dumps({
            "schema": record["schema"], "verified": True,
            "seal": str(args.seal), "require_unopened": args.require_unopened,
        }, indent=2))


if __name__ == "__main__":
    main()
