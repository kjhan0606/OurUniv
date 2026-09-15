#!/usr/bin/env python
"""Correct the V27 audit's cross-domain numerical ratio without rerunning samples."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hong2021_v15_edm import git_state
from hong2021_v18_init import sha256_file
from hong2021_v27_latent_audit_logic import mechanism_summary


CORRECTION_SHA256 = (
    "9f97b972210275b58549c182187aa2139f629b370bc48780171f1871a1965bf3"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--v26-audit", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    commit, clean = git_state(args.repo.resolve())
    if not clean:
        raise RuntimeError("corrected V27 latent audit requires a clean worktree")
    if sha256_file(args.correction) != CORRECTION_SHA256:
        raise ValueError("V27 latent-audit correction file hash differs")
    correction = json.loads(args.correction.read_text())
    parent = Path(correction["superseded_audit"]["path"])
    if (
        correction.get("schema")
        != "hong2021-v27-latent-audit-cross-domain-ratio-correction-v1"
        or correction.get("status") != "frozen_before_corrected_classification"
        or sha256_file(parent) != correction["superseded_audit"]["sha256"]
        or correction.get("firewall", {}).get("thresholds_changed") is not False
        or correction.get("firewall", {}).get("Astrid_accessed") is not False
        or correction.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V27 latent-audit correction provenance differs")
    old = json.loads(parent.read_text())
    v26 = json.loads(args.v26_audit.read_text())
    compact = {
        step: row["domains"] for step, row in old["candidates"].items()
    }
    summary = mechanism_summary(compact, old["optimization"], v26)
    report = {
        "schema": "hong2021-v27-frozen-trained-flow-latent-audit-correction-v2",
        "status": "complete_corrected_classification",
        "correction": str(args.correction.resolve()),
        "correction_sha256": CORRECTION_SHA256,
        "superseded_audit": str(parent),
        "superseded_audit_sha256": correction["superseded_audit"]["sha256"],
        "v26_mechanism_audit": str(args.v26_audit.resolve()),
        "v26_mechanism_audit_sha256": sha256_file(args.v26_audit),
        "audit_code_commit": commit,
        "worktree_clean_at_audit": clean,
        "corrected_mechanism_summary": summary,
        "thresholds_changed": False,
        "model_or_samples_changed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False
    }
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite corrected audit: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
