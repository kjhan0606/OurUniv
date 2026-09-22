#!/usr/bin/env python3
"""Select the prespecified best passing LG row and reproduce it exactly."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-result", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    search = json.loads(args.search_result.read_text())
    rows = list(search.get("rows", []))
    if len(rows) != 256 or search.get("decision") not in {
        "CONDITIONED_LG_CANDIDATES_AVAILABLE",
        "NO_CONDITIONED_LG_CANDIDATE",
    }:
        raise RuntimeError("search result is incomplete or has an unknown decision")
    passing = [row for row in rows if row.get("screen_pass")]
    if not passing:
        print("NO_CONDITIONED_LG_CANDIDATE: exact reproduction not started", flush=True)
        raise SystemExit(3)

    selected = min(
        passing,
        key=lambda row: (float(row["best_pair"]["ranking_score"]), int(row["index"])),
    )
    base = json.loads(args.base_config.read_text())
    base["schema"] = "ouruniv-cf4-lg-highk-selected-candidate-v1"
    base["status"] = "selected_for_exact_reproduction"
    base["selection_source"] = str(args.search_result.resolve())
    base["selection_rule"] = (
        "minimum preregistered best-pair ranking_score among rows passing the "
        "hard P2 screen named by the base config; index breaks exact ties"
    )
    base["selected_search_index"] = int(selected["index"])
    base["seed_bank"] = {
        "count": 1,
        "field_seed_start": int(selected["field_seed"]),
        "geometry_seed_start": int(selected["geometry_seed"]),
        "likelihood_noise_seed_start": int(selected["likelihood_noise_seed"]),
        "midpoint_seed_start": int(selected["midpoint_seed"]),
    }
    base["expected_reproduction"] = {
        "field_sha256": selected["field_sha256"],
        "screen_pass": True,
        "n_screen_pairs": int(selected["n_screen_pairs"]),
    }
    base["output"] = str(args.output.resolve())
    base["retention"]["passing_conditioned_fields"] = True

    if args.selected_config.exists():
        raise RuntimeError(f"refusing to overwrite {args.selected_config}")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    args.selected_config.parent.mkdir(parents=True, exist_ok=True)
    args.selected_config.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "selected_search_index": selected["index"],
                "field_seed": selected["field_seed"],
                "ranking_score": selected["best_pair"]["ranking_score"],
                "selected_config": str(args.selected_config.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("cf4_lg_highk_conditioning_stream.py")),
            "--config",
            str(args.selected_config),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
