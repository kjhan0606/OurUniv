#!/usr/bin/env python
"""Write the frozen automatic failure summary after a failed V24 gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    decision = json.loads(args.decision.read_text())
    if decision.get("development_pass") is not False:
        raise ValueError("V24 failure audit requires a failed decision")
    trajectory = {}
    for candidate in decision["candidates"]:
        trajectory[str(candidate["step"])] = {
            domain: {
                "field_pass": row["field_gate"]["pass"],
                "Q3_delta_q99_999_dex": row["mechanism_Q3_Q4"][
                    "delta_q99_999_dex"
                ],
                "Q4_generated_over_truth_mean_delta_squared": row[
                    "mechanism_Q3_Q4"
                ]["generated_over_truth_mean_delta_squared"],
                "Q6_latent_max_absolute_mean_error": row[
                    "conditional_Q6_latent"
                ]["maximum_absolute_generated_minus_truth_mean"],
                "comparison_to_v22": candidate["comparison_to_v22"][domain],
            }
            for domain, row in candidate["domains"].items()
        }
    report = {
        "schema": "hong2021-v24-automatic-failure-audit-v1",
        "decision": str(args.decision.resolve()),
        "decision_digest_sha256": decision["decision_digest_sha256"],
        "trajectory": trajectory,
        "capacity_diagnostic": decision["capacity_diagnostic"],
        "classification": decision["failure_classification"],
        "next": decision["next"],
        "capacity_increased_again_after_results": False,
        "horizon_extended_after_results": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite V24 failure audit: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
