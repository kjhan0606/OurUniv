#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    decision = json.loads(args.decision.read_text())
    if decision.get("development_pass") is not False:
        raise ValueError("V22 failure audit requires a failed decision")
    trajectory = {}
    for candidate in decision["candidates"]:
        trajectory[str(candidate["step"])] = {
            domain: {
                "field_pass": row["field_gate"]["pass"],
                "Q3_delta_q99_999_dex": row["mechanism_Q3_Q4"]["delta_q99_999_dex"],
                "Q4_generated_over_truth_mean_delta_squared": row["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"],
                "Q6_latent_max_absolute_mean_error": row["conditional_Q6_latent"]["maximum_absolute_generated_minus_truth_mean"],
            }
            for domain, row in candidate["domains"].items()
        }
    report = {
        "schema": "hong2021-v22-automatic-failure-audit-v1",
        "decision": str(args.decision.resolve()),
        "decision_digest_sha256": decision["decision_digest_sha256"],
        "trajectory": trajectory,
        "plateau_diagnostic": decision["plateau_diagnostic"],
        "classification": decision["failure_classification"],
        "recommended_next_design_audit": "conditional moment objective or capacity; no further horizon extension",
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite V22 failure audit: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n"); os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
