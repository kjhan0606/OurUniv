#!/usr/bin/env python3
"""Audit a defensive latent-midpoint proposal without opening fresh phases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from cf4_lg_z0_likelihood import sha256_file
from cf4_lg_midpoint_proposal import (  # noqa: F401
    diagonal_normal_logpdf,
    mixture_logpdf,
    verify_defensive_component,
)


def importance_ess_fraction(likelihood: np.ndarray, proposal_over_prior: np.ndarray) -> float:
    likelihood = np.asarray(likelihood, dtype=np.float64)
    ratio = np.asarray(proposal_over_prior, dtype=np.float64)
    denominator = float(np.mean(likelihood**2 / ratio))
    return float(np.mean(likelihood) ** 2 / denominator) if denominator > 0.0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    program = json.loads(args.program.read_text())
    if program.get("status") != "frozen_before_conditional_proposal_audit":
        parser.error("proposal audit is not frozen")
    repo = Path(__file__).resolve().parents[1]

    parent_spec = program["parent_result"]
    parent_path = repo / parent_spec["path"]
    if sha256_file(parent_path) != parent_spec["sha256"]:
        parser.error("parent result hash mismatch")
    if json.loads(parent_path.read_text()).get("status") != parent_spec["required_status"]:
        parser.error("parent result status mismatch")

    loaded: dict[str, tuple[Path, dict[str, Any]]] = {}
    for name, specification in program["immutable_inputs"].items():
        path = Path(specification["path"])
        if not path.is_absolute():
            path = repo / path
        if sha256_file(path) != specification["sha256"]:
            parser.error(f"immutable input hash mismatch: {name}")
        loaded[name] = (path, json.loads(path.read_text()))
    likelihood_program = loaded["v7_likelihood_program"][1]
    development = loaded["v7_development_result"][1]
    manifest = loaded["v6_proposal_manifest"][1]

    bank = likelihood_program["prospective_v7_bank"]
    prior = bank["latent_midpoint_target_prior"]
    components = bank["latent_midpoint_sampling_proposal"]["components"]
    proposal_cfg = program["proposal"]
    analytic_bound = verify_defensive_component(
        prior, components, float(proposal_cfg["minimum_defensive_component_weight"]))
    if analytic_bound > float(proposal_cfg["maximum_analytic_target_prior_over_proposal_ratio"]):
        parser.error("analytic target-prior/proposal bound exceeds the frozen maximum")

    midpoint_by_seed = {
        int(row["proposal_seed"]): np.asarray(
            row["protohalo_midpoint_offset_draw_mpc_h"], dtype=np.float64)
        for row in manifest["entries"]
    }
    rows = development["rows"]
    finite_log = [float(row["log_likelihood"]) for row in rows
                  if np.isfinite(row["log_likelihood"])]
    if not finite_log:
        parser.error("development result contains no finite likelihood")
    shift = max(finite_log)
    likelihood = np.zeros(len(rows), dtype=np.float64)
    proposal_over_prior = np.empty(len(rows), dtype=np.float64)
    audit_rows = []
    for index, row in enumerate(rows):
        seed = int(row["small_scale_seed"])
        value = midpoint_by_seed[seed]
        log_prior = diagonal_normal_logpdf(value, prior)
        log_proposal = mixture_logpdf(value, components)
        proposal_over_prior[index] = math.exp(log_proposal - log_prior)
        if np.isfinite(row["log_likelihood"]):
            likelihood[index] = math.exp(float(row["log_likelihood"]) - shift)
        audit_rows.append({
            "small_scale_seed": seed,
            "midpoint_draw_mpc_h": value.tolist(),
            "relative_likelihood": float(likelihood[index]),
            "log_target_prior": log_prior,
            "log_sampling_proposal": log_proposal,
            "proposal_over_prior": float(proposal_over_prior[index]),
        })

    point_fraction = importance_ess_fraction(likelihood, proposal_over_prior)
    size_cfg = program["bank_size_selection"]
    rng = np.random.default_rng(int(size_cfg["bootstrap_seed"]))
    bootstrap = np.empty(int(size_cfg["bootstrap_replicates"]), dtype=np.float64)
    for replicate in range(bootstrap.size):
        indices = rng.integers(0, len(rows), len(rows))
        bootstrap[replicate] = importance_ess_fraction(
            likelihood[indices], proposal_over_prior[indices])
    lower_quantile = float(size_cfg["lower_quantile"])
    lower_fraction = float(np.quantile(bootstrap, lower_quantile))
    candidate_rows = []
    selected_size = None
    for size in size_cfg["candidate_sizes"]:
        size = int(size)
        row = {
            "bank_size": size,
            "point_expected_ESS": size * point_fraction,
            "bootstrap_lower_ESS": size * lower_fraction,
            "passes": size * lower_fraction >= float(size_cfg["minimum_lower_bound_ESS"]),
        }
        candidate_rows.append(row)
        if row["passes"] and selected_size is None:
            selected_size = size
    passed = (
        selected_size is not None
        and selected_size <= int(size_cfg["maximum_authorized_size"])
        and analytic_bound <= float(proposal_cfg["maximum_analytic_target_prior_over_proposal_ratio"])
    )
    report = {
        "schema": "ouruniv-lg-z0-forward-likelihood-v8-proposal-audit-result-v1",
        "status": "complete_pass_authorize_fresh_v8" if passed else "complete_fail_require_SMC",
        "program": str(args.program.resolve()),
        "program_sha256": sha256_file(args.program),
        "n_development_rows": len(rows),
        "n_finite_likelihood_rows": int(np.count_nonzero(likelihood)),
        "analytic_target_prior_over_proposal_bound": analytic_bound,
        "point_ESS_fraction": point_fraction,
        "bootstrap_lower_quantile": lower_quantile,
        "bootstrap_lower_ESS_fraction": lower_fraction,
        "candidate_bank_sizes": candidate_rows,
        "selected_bank_size": selected_size,
        "authorize_fresh_v8": passed,
        "audit_rows": audit_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "n_development_rows", "n_finite_likelihood_rows",
        "analytic_target_prior_over_proposal_bound", "point_ESS_fraction",
        "bootstrap_lower_ESS_fraction", "candidate_bank_sizes",
        "selected_bank_size", "authorize_fresh_v8",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
