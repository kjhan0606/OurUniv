#!/usr/bin/env python
"""Build the hash-complete V79 single-use manifest for frozen V80 artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hong2021_v18_init import sha256_file
from hong2021_v80_sample import (
    DOMAIN_KEYS,
    DOMAIN_ORDER,
    candidate_freeze_commit,
    load_program,
)


MANIFEST_SCHEMA = "hong2021-v79-single-use-execution-manifest-v1"


def hash_row(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def build_manifest(
    candidate_program_path: Path,
    preflight_path: Path,
    repo: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    program = load_program(candidate_program_path.resolve(), repo)
    preflight = json.loads(preflight_path.read_text())
    candidate_sha = sha256_file(candidate_program_path)
    freeze_commit = candidate_freeze_commit(candidate_program_path, repo)
    if (
        preflight.get("preflight_pass") is not True
        or preflight.get("candidate_program_sha256") != candidate_sha
        or preflight.get("candidate_freeze_commit") != freeze_commit
        or preflight.get("selected_payload_accessed_before_candidate_freeze") is not False
    ):
        raise ValueError("V80 manifest preflight differs")
    domains = {}
    root = Path(program["outputs"]["ensemble_root"])
    for domain in DOMAIN_ORDER:
        key = DOMAIN_KEYS[domain]
        candidate = root / "candidate" / key / "ensemble16.h5"
        control = root / "control" / key / "ensemble16.h5"
        candidate_metrics = root / "candidate" / key / "ensemble_evaluation" / "metrics.json"
        control_metrics = root / "control" / key / "ensemble_evaluation" / "metrics.json"
        contracts = program["frozen_domain_execution_contracts"][domain]
        domains[domain] = {
            "candidate_ensemble": hash_row(candidate),
            "control_ensemble": hash_row(control),
            "candidate_metrics": hash_row(candidate_metrics),
            "control_metrics": hash_row(control_metrics),
            "candidate_expected_attrs": contracts["candidate_expected_attrs"],
            "control_expected_attrs": contracts["control_expected_attrs"],
            "pairing": contracts["pairing"],
        }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete_single_use_artifacts_bound_before_gate",
        "V79": {
            "program_sha256": program["V79_program_sha256"],
            "gate_source": program["implementation_sources"]["V79_gate"],
            "gate_implementation_commit": program["V79_gate_implementation_commit"],
        },
        "candidate_program": {
            "path": str(candidate_program_path.resolve()),
            "sha256": candidate_sha,
            "freeze_commit": freeze_commit,
        },
        "preflight": {
            "artifact": hash_row(preflight_path),
            "selected_payload_accessed_before_candidate_freeze": False,
            "candidate_program_frozen_and_pushed_before_sampling": True,
        },
        "source_indices": program["single_use_fresh_selection"],
        "domains": domains,
        "provenance": program["frozen_execution_provenance"],
        "single_use": {
            "candidate_count": 1,
            "prior_gate_disclosure": False,
            "retry_authorized": False,
            "post_disclosure_change_authorized": False,
        },
        "output_root": program["outputs"]["V79_gate_root"],
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-program", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("V80 manifest refuses an existing output")
    manifest = build_manifest(args.candidate_program, args.preflight, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(".json.partial")
    partial.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
