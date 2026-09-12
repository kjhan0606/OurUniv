#!/usr/bin/env python3
"""Freeze the best P2 pair that passes the recentered environment preview."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--p2-result", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    preview = json.loads(args.preview.read_text())
    p2 = json.loads(args.p2_result.read_text())
    if preview.get("p2_result_sha256") != sha256_file(args.p2_result):
        parser.error("preview/P2 SHA-256 mismatch")

    passing = []
    for row in preview.get("rows", []):
        pair_rows = row.get("pair_rows", [row])
        for pair_row in pair_rows:
            if pair_row.get("preview_pass"):
                passing.append({
                    "parent_seed": int(row["parent_seed"]),
                    "small_scale_seed": int(row["small_scale_seed"]),
                    **pair_row,
                })
    passing.sort(key=lambda row: (
        float(row["screen_pair"]["ranking_score"]),
        float(row["screen_pair"]["midpoint_offset_mpc_h"]),
        row["small_scale_seed"],
        row.get("pair_index", 0),
    ))

    args.outdir.mkdir(parents=True, exist_ok=True)
    selection_path = args.outdir / "promotion_selection.json"
    if not passing:
        result = {
            "schema": "ouruniv-recentered-promotion-selection-v1",
            "status": "no_passing_candidate",
            "preview": str(args.preview.resolve()),
            "preview_sha256": sha256_file(args.preview),
            "p2_result": str(args.p2_result.resolve()),
            "p2_result_sha256": sha256_file(args.p2_result),
            "n_passing_pairs": 0,
            "selected": None,
        }
        selection_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        (args.outdir / "AUTO_FAIL").write_text("no recentered P1 screen survivor\n")
        print("[select] no recentered P1 screen survivor", flush=True)
        return

    selected = passing[0]
    matches = [
        row for row in p2["results"]
        if int(row["parent_seed"]) == selected["parent_seed"]
        and int(row["small_scale_seed"]) == selected["small_scale_seed"]
    ]
    if len(matches) != 1:
        parser.error("cannot resolve exactly one selected P2 realization")
    selected_full = json.loads(json.dumps(matches[0]))
    selected_full["best_pair"] = selected["screen_pair"]
    selected_full["selection_status"] = "recentered_P1_preview_pass"
    selected_result_path = args.outdir / "selected_p2_result.json"
    selected_result_path.write_text(
        json.dumps(selected_full, indent=2, sort_keys=True) + "\n")

    # The trace needs the full-screen envelope and its config provenance, but
    # with the selected environment-valid pair promoted to best_pair.
    derived_full = json.loads(json.dumps(p2))
    for row in derived_full["results"]:
        if (int(row["parent_seed"]) == selected["parent_seed"]
                and int(row["small_scale_seed"]) == selected["small_scale_seed"]):
            row["best_pair"] = selected["screen_pair"]
            row["selection_status"] = "recentered_P1_preview_pass"
    derived_path = args.outdir / "selected_p2_full_result.json"
    derived_path.write_text(json.dumps(derived_full, indent=2, sort_keys=True) + "\n")

    result = {
        "schema": "ouruniv-recentered-promotion-selection-v1",
        "status": "selected",
        "preview": str(args.preview.resolve()),
        "preview_sha256": sha256_file(args.preview),
        "p2_result": str(args.p2_result.resolve()),
        "p2_result_sha256": sha256_file(args.p2_result),
        "n_passing_pairs": len(passing),
        "selected": selected,
        "selected_p2_result": str(selected_result_path.resolve()),
        "selected_p2_full_result": str(derived_path.resolve()),
    }
    selection_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.outdir / "AUTO_PASS").write_text(
        f"{selected['parent_seed']} {selected['small_scale_seed']}\n")
    print(f"[select] p{selected['parent_seed']} s{selected['small_scale_seed']} "
          f"rank={selected['screen_pair']['ranking_score']:.6f} "
          f"passing_pairs={len(passing)}", flush=True)


if __name__ == "__main__":
    main()
