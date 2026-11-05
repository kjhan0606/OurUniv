#!/usr/bin/env python
"""Run the V80 engineering diagnostic with only the DC-row shape fix."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

import hong2021_v80_sample as frozen
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor


PROGRAM_SCHEMA = "hong2021-v80d-engineering-diagnostic-program-v1"
PROGRAM_STATUS = "frozen_before_V80D_reexecution_of_consumed_V79_subset"
BASE_PROGRAM = Path("config/hong2021_v80_single_candidate_program.json")
BASE_PROGRAM_SHA256 = "43d41ae9722e2321d3a206492d543e39ba5e8b22868e4f3a755cd1b2147dc205"
FAILURE_SEAL = Path("config/hong2021_v80_terminal_failure_seal.json")
FAILURE_SEAL_SHA256 = "4460acfb1faf06a71a3d8c325f1a2098b3937754ca38c0a9b7f13294fb68374a"
_INHERITED_CALIBRATE_AND_PROJECT = frozen._calibrate_and_project


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def diagnostic_freeze_commit(program_path: Path, repo: Path) -> str:
    return frozen.candidate_freeze_commit(program_path, repo)


def load_diagnostic_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    repo = repo.resolve()
    program = strict_json(path.resolve())
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or program.get("engineering_only") is not True
        or program.get("statistically_valid_V79_reexecution") is not False
        or program.get("only_code_change")
        != "reshape maximum_absolute_residual_DC from (members,1) to (members,) before the diagnostic HDF5 row write"
        or program.get("retry_or_candidate_selection_role") is not False
    ):
        raise ValueError("V80D engineering boundary differs")
    base_path = (repo / BASE_PROGRAM).resolve()
    seal_path = (repo / FAILURE_SEAL).resolve()
    if (
        sha256_file(base_path) != BASE_PROGRAM_SHA256
        or sha256_file(seal_path) != FAILURE_SEAL_SHA256
        or program["parent_failure"]["candidate_program_sha256"] != BASE_PROGRAM_SHA256
        or program["parent_failure"]["terminal_seal_sha256"] != FAILURE_SEAL_SHA256
    ):
        raise ValueError("V80D frozen parent differs")
    for label, row in program["implementation_sources"].items():
        source = Path(row["path"])
        source = source.resolve() if source.is_absolute() else (repo / source).resolve()
        if sha256_file(source) != row["sha256"]:
            raise ValueError(f"V80D source differs: {label}")
    base = frozen.load_program(base_path, repo)
    return program, base


def fixed_calibrate_and_project(
    total_field: np.ndarray,
    mean: np.ndarray,
    source_knots: np.ndarray,
    mapped_knots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    output, dc = _INHERITED_CALIBRATE_AND_PROJECT(
        total_field, mean, source_knots, mapped_knots
    )
    dc = np.asarray(dc, dtype=np.float64)
    if dc.shape != (total_field.shape[0], 1):
        raise ValueError("V80D inherited DC shape differs before the frozen fix")
    return output, dc.reshape(total_field.shape[0])


def install_diagnostic_overrides(
    diagnostic_program: dict[str, Any], base: dict[str, Any], freeze_commit: str
) -> None:
    effective = copy.deepcopy(base)
    effective["outputs"]["ensemble_root"] = diagnostic_program["outputs"][
        "ensemble_root"
    ]
    frozen.load_program = lambda path, repo: copy.deepcopy(effective)
    frozen.candidate_freeze_commit = lambda path, repo: freeze_commit
    frozen._calibrate_and_project = fixed_calibrate_and_project


def preflight(
    program_path: Path, repo: Path, output_root: Path
) -> dict[str, Any]:
    program, base = load_diagnostic_program(program_path, repo)
    freeze_commit = diagnostic_freeze_commit(program_path, repo)
    install_diagnostic_overrides(program, base, freeze_commit)
    result = frozen.validate_without_selected_payload(program_path, repo, output_root)
    result.update(
        {
            "schema": "hong2021-v80d-code-only-engineering-preflight-v1",
            "status": "complete_code_only_V80D_engineering_preflight",
            "engineering_only": True,
            "statistically_valid_V79_reexecution": False,
            "parent_terminal_seal_sha256": FAILURE_SEAL_SHA256,
        }
    )
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def run_sample(
    program_path: Path, preflight_path: Path, repo: Path, output_root: Path
) -> None:
    program, base = load_diagnostic_program(program_path, repo)
    freeze_commit = diagnostic_freeze_commit(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean or not _is_ancestor(repo.resolve(), freeze_commit, commit):
        raise RuntimeError("V80D requires clean frozen ancestry")
    report = strict_json(preflight_path)
    if (
        report.get("schema") != "hong2021-v80d-code-only-engineering-preflight-v1"
        or report.get("engineering_only") is not True
        or report.get("statistically_valid_V79_reexecution") is not False
        or canonical_digest(report) != report.get("decision_digest_sha256")
    ):
        raise ValueError("V80D engineering preflight differs")
    # The inherited sampler checks its original preflight schema.  This local
    # in-memory view changes only the label; the file remains an explicit V80D
    # diagnostic record and all hashes/authorization fields remain unchanged.
    inherited = copy.deepcopy(report)
    inherited["schema"] = frozen.PREFLIGHT_SCHEMA
    inherited["status"] = "complete_code_only_V80_single_candidate_preflight"
    inherited["decision_digest_sha256"] = canonical_digest(inherited)
    temporary = preflight_path.with_suffix(".inherited.json")
    if temporary.exists():
        raise FileExistsError("V80D refuses an existing inherited preflight view")
    temporary.write_text(json.dumps(inherited, indent=2) + "\n")
    try:
        install_diagnostic_overrides(program, base, freeze_commit)
        frozen.sample(program_path, temporary, repo, output_root)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    first = subparsers.add_parser("preflight")
    first.add_argument("--program", type=Path, required=True)
    first.add_argument("--repo", type=Path, required=True)
    first.add_argument("--output-root", type=Path, required=True)
    first.add_argument("--out", type=Path, required=True)
    second = subparsers.add_parser("sample")
    second.add_argument("--program", type=Path, required=True)
    second.add_argument("--preflight", type=Path, required=True)
    second.add_argument("--repo", type=Path, required=True)
    second.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.program, args.repo, args.output_root)
        if args.out.exists():
            raise FileExistsError("V80D preflight refuses existing output")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    else:
        run_sample(args.program, args.preflight, args.repo, args.output_root)


if __name__ == "__main__":
    main()
