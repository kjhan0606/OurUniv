#!/usr/bin/env python3
"""Write the Q4.0/Q4.1 seed and geometry preflight records."""

from __future__ import annotations

import json
from pathlib import Path

from cf4_q4_preflight import build_seed_manifest, geometry_preflight


def run(seed_manifest_path: Path, result_path: Path) -> None:
    seed_manifest = build_seed_manifest()
    geometry = geometry_preflight()
    seed_manifest_path.write_text(json.dumps(seed_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    result = {
        "schema": "ouruniv-cf4-q4-preflight-result-v1",
        "bundle": "Q4.0-Q4.1",
        "seed_manifest": str(seed_manifest_path),
        "seed_manifest_sha256": seed_manifest["manifest_sha256"],
        "seed_firewall": {
            "development_count": seed_manifest["development_count"],
            "heldout_count": seed_manifest["heldout_count"],
            "within_and_cross_namespace_disjoint": seed_manifest["cross_namespace_disjoint"],
            "heldout_evaluated": False,
        },
        "geometry": geometry,
        "status": "COMPLETE_Q4_0_Q4_1_TERMINAL_EXACT_GROUPING_NO_GO",
        "next": "Write candidate-2/3 design-only plan; do not run Q4.3 gradient timing or heldout evaluation.",
    }
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run(root / "config/cf4_q4_seed_manifest_v1.json", root / "config/cf4_q4_preflight_result_v1.json")
