#!/usr/bin/env python
"""Direct transport of frozen V28 donor residuals in physical y coordinates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_edm import git_state
from hong2021_v18_init import sha256_file
from hong2021_v26 import CACHE_KEYS
from hong2021_v28_empirical import DOMAIN_KEYS, DOMAIN_ORDER, DONOR_COUNTS
from hong2021_v28_empirical import load_frozen_program as load_v28_program


REGISTRY_SCHEMA = "hong2021-v29-direct-physical-residual-transport-development-program-v1"
REGISTRY_SHA256 = "a6f3561f2c2a96d24493109b1fc279bf9a4bf4d36e4c7cfdc5f678ae28796f93"
DESIGN_AUDIT_SHA256 = "e23f8cc3e58e71dc144dc0a72ded29d2f367708a17767ec3f417581ed30ccdb1"
FAILURE_AUDIT_SHA256 = "622c8e5ccee9d42b5cf428f6fb3ae3b9887365ac4e15d77f98692625da206949"
ENSEMBLE_SCHEMA = "hong2021-v29-direct-physical-residual-transport-ensemble-v1"
PREFLIGHT_SCHEMA = "hong2021-v29-direct-physical-residual-hard-preflight-v1"


def _resolve(value: str, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"{label} hash differs from V29 freeze")
    return json.loads(path.read_text())


def load_frozen_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    registry = _verified_json(path.resolve(), REGISTRY_SHA256, "V29 registry")
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("status")
        != "frozen_before_implementation_sampling_or_development_evaluation"
    ):
        raise ValueError("V29 registry schema or status differs")
    design = registry["design_audit"]
    design_payload = _verified_json(
        _resolve(design["path"], repo), DESIGN_AUDIT_SHA256, "V29 design audit"
    )
    if (
        design.get("sha256") != DESIGN_AUDIT_SHA256
        or design_payload.get("selected_change", {}).get("name")
        != "same_donors_direct_centered_physical_y_residual_transport"
        or design_payload.get("firewall", {}).get("donor_reselection") is not False
    ):
        raise ValueError("V29 selected design differs")
    parent = registry["parent_evidence"]
    v28_path = _resolve(parent["v28_registry"], repo)
    if sha256_file(v28_path) != parent["v28_registry_sha256"]:
        raise ValueError("V29 V28 parent registry differs")
    _, artifacts, v20 = load_v28_program(v28_path, repo)
    decision = _verified_json(
        Path(parent["v28_decision"]), parent["v28_decision_sha256"], "V28 decision"
    )
    audit = _verified_json(
        Path(parent["v28_failure_mechanism_audit"]),
        FAILURE_AUDIT_SHA256,
        "V28 failure mechanism audit",
    )
    if (
        parent["v28_failure_mechanism_audit_sha256"] != FAILURE_AUDIT_SHA256
        or decision.get("decision_digest_sha256")
        != parent["v28_decision_digest_sha256"]
        or audit.get("classification", {}).get("class")
        != parent["required_classification"]
        or audit.get("classification", {}).get("next") != parent["required_next"]
        or audit.get("Astrid_accessed") is not False
        or audit.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V29 parent failure conclusion or firewall differs")
    for domain in DOMAIN_KEYS.values():
        frozen = registry["frozen_v28_selections"][domain]
        if sha256_file(Path(frozen["ensemble"])) != frozen["sha256"]:
            raise ValueError(f"V29 frozen V28 {domain} ensemble differs")
    return registry, artifacts, v20


def centered_donor_residual(
    target_y: np.ndarray,
    corrected_mean: np.ndarray,
    predicted_location: float,
) -> np.ndarray:
    """Return a physical y residual with exact numerical cube DC projection."""
    residual = np.asarray(target_y, dtype=np.float32) - (
        np.asarray(corrected_mean, dtype=np.float32) + np.float32(predicted_location)
    )
    residual = residual.astype(np.float64)
    residual -= residual.mean(axis=(-3, -2, -1), keepdims=True)
    return residual.astype(np.float32)


def transport_residual(
    donor_residual: np.ndarray,
    query_baseline: np.ndarray,
    isometry: int,
) -> np.ndarray:
    if not 0 <= isometry < len(CUBE_ISOMETRIES):
        raise ValueError("V29 donor isometry is out of range")
    permutation, reflections = CUBE_ISOMETRIES[isometry]
    oriented = apply_cube_isometry(donor_residual, permutation, reflections)
    return np.asarray(query_baseline, dtype=np.float32) + oriented


def sample_all(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    registry, artifacts, v20 = load_frozen_program(args.registry.resolve(), repo)
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V29 sampling requires a clean committed worktree")
    preflight = _verified_json(
        args.preflight.resolve(), args.preflight_sha256, "V29 hard preflight"
    )
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("status") != "pass"
        or preflight.get("registry_sha256") != REGISTRY_SHA256
        or preflight.get("code_commit") != commit
    ):
        raise ValueError("V29 hard preflight differs at sampling")
    output_root = args.out.resolve()
    if output_root.exists():
        raise RuntimeError(f"V29 refuses pre-existing output: {output_root}")
    output_root.mkdir(parents=True)
    experiment = v20["e8_gaussianized_marginal_retrain"]
    train_data = {
        source: h5py.File(experiment["data"][source]["train_data"]["path"], "r")
        for source in DOMAIN_ORDER
    }
    train_cache = {
        source: h5py.File(
            artifacts["caches"][CACHE_KEYS[source]["train"]]["path"], "r"
        )
        for source in DOMAIN_ORDER
    }
    try:
        for source in DOMAIN_ORDER:
            domain = DOMAIN_KEYS[source]
            parent_info = registry["frozen_v28_selections"][domain]
            data_info = experiment["data"][source]["validation_data"]
            cache_info = artifacts["caches"][CACHE_KEYS[source]["validation"]]
            domain_root = output_root / domain
            domain_root.mkdir()
            output = domain_root / "ensemble16.h5"
            partial = output.with_suffix(".h5.partial")
            with h5py.File(parent_info["ensemble"], "r") as old, h5py.File(
                data_info["path"], "r"
            ) as query_data, h5py.File(cache_info["path"], "r") as query_cache, h5py.File(
                partial, "w"
            ) as new:
                sample_ds = new.create_dataset(
                    "sample", shape=(16, 16, 1, 64, 64, 64), dtype="f4",
                    chunks=(1, 1, 1, 64, 64, 64), compression="lzf",
                )
                mean_ds = new.create_dataset(
                    "conditional_mean", shape=(16, 1, 64, 64, 64),
                    dtype="f4", compression="lzf",
                )
                truth_ds = new.create_dataset(
                    "truth", shape=(16, 1, 64, 64, 64),
                    dtype="f4", compression="lzf",
                )
                for name in (
                    "source_index", "donor_source", "donor_index",
                    "donor_isometry", "donor_distance",
                    "predicted_residual_dc", "predicted_band_scales",
                ):
                    old.copy(name, new)
                maximum_dc = 0.0
                for object_index, data_index in enumerate(old["source_index"][:]):
                    query_mean = np.asarray(
                        query_cache["conditional_mean"][data_index], dtype=np.float32
                    )
                    query_location = float(
                        query_cache["predicted_residual_dc"][data_index]
                    )
                    query_baseline = query_mean + np.float32(query_location)
                    for member in range(16):
                        donor_source = DOMAIN_ORDER[
                            int(old["donor_source"][object_index, member])
                        ]
                        donor_index = int(old["donor_index"][object_index, member])
                        donor_mean = np.asarray(
                            train_cache[donor_source]["conditional_mean"][donor_index],
                            dtype=np.float32,
                        )
                        donor_location = float(
                            train_cache[donor_source]["predicted_residual_dc"][donor_index]
                        )
                        residual = centered_donor_residual(
                            np.asarray(
                                train_data[donor_source]["target"][donor_index],
                                dtype=np.float32,
                            ),
                            donor_mean,
                            donor_location,
                        )
                        maximum_dc = max(
                            maximum_dc,
                            float(np.max(np.abs(residual.mean(axis=(-3, -2, -1)))))
                        )
                        sample_ds[object_index, member] = transport_residual(
                            residual,
                            query_baseline,
                            int(old["donor_isometry"][object_index, member]),
                        )
                    mean_ds[object_index] = query_baseline
                    # Truth is copied only after donor selection and sample construction.
                    truth_ds[object_index] = np.asarray(
                        query_data["target"][data_index], dtype=np.float32
                    )
                    print(f"[sample] V29 {source} {object_index + 1}/16", flush=True)
                new.attrs.update(
                    {
                        "schema": ENSEMBLE_SCHEMA,
                        "method": "same_donors_direct_centered_physical_y_residual_transport",
                        "v29_registry_sha256": REGISTRY_SHA256,
                        "design_audit_sha256": DESIGN_AUDIT_SHA256,
                        "failure_audit_sha256": FAILURE_AUDIT_SHA256,
                        "parent_v28_ensemble": str(Path(parent_info["ensemble"]).resolve()),
                        "parent_v28_ensemble_sha256": parent_info["sha256"],
                        "source_cache_sha256": cache_info["sha256"],
                        "source_data_sha256": data_info["sha256"],
                        "ensemble_members": 16,
                        "diagnostic_k_h_mpc": 1.0,
                        "maximum_absolute_centered_donor_residual_dc": maximum_dc,
                        "donor_reselection": False,
                        "selection_uses_validation_truth": False,
                        "query_dependent_nonlinear_inverse": False,
                        "worktree_clean_at_sampling": clean,
                        "sampling_code_commit": commit,
                        "hard_preflight": str(args.preflight.resolve()),
                        "hard_preflight_sha256": args.preflight_sha256,
                        "Astrid_accessed": False,
                        "historical_EAGLE_accessed": False,
                        "complete": True,
                    }
                )
            os.replace(partial, output)
    finally:
        for handle in (*train_data.values(), *train_cache.values()):
            handle.close()
    print(f"[out] {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    sample_all(build_parser().parse_args())


if __name__ == "__main__":
    main()
