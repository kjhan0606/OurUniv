#!/usr/bin/env python
"""Write the frozen automatic failure summary after a failed V25 gate."""
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
        raise ValueError("V25 failure audit requires a failed decision")
    report = {
        "schema": "hong2021-v25-automatic-failure-audit-v1",
        "decision": str(args.decision.resolve()),
        "decision_digest_sha256": decision["decision_digest_sha256"],
        "candidate_summary": {
            str(candidate["step"]): {
                domain: {
                    "field_pass": row["field_gate"]["pass"],
                    "Q3_delta_q99_999_dex": row["mechanism_Q3_Q4"][
                        "delta_q99_999_dex"
                    ],
                    "Q3_generated_max_above_truth_max_dex": row[
                        "mechanism_Q3_Q4"
                    ]["generated_max_above_truth_max_dex"],
                    "Q4_generated_over_truth_mean_delta_squared": row[
                        "mechanism_Q3_Q4"
                    ]["generated_over_truth_mean_delta_squared"],
                    "comparison_to_v24": candidate["comparison_to_v24"][domain],
                }
                for domain, row in candidate["domains"].items()
            }
            for candidate in decision["candidates"]
        },
        "classification": decision["failure_classification"],
        "next": decision["next"],
        "mixture_coefficient_tuned_after_results": False,
        "capacity_increased_after_results": False,
        "horizon_extended_after_results": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite V25 failure audit: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
