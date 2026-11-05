#!/usr/bin/env python3
"""Run the one fixed R2 joint field-MCSE synthetic stress case."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cf4_r2_joint_field_mcse_stress import (
    DEFAULT_DRAWS,
    DEFAULT_GRID_SIZE,
    DEFAULT_ORACLE_CHUNK_SIZE,
    DEFAULT_ORACLE_HISTORIES,
    DEFAULT_ORACLE_SEED,
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    make_oracle_references,
    run_fixed_stress,
)


def _write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--oracle-seed", type=int, default=DEFAULT_ORACLE_SEED)
    parser.add_argument("--oracle-histories", type=int,
        default=DEFAULT_ORACLE_HISTORIES)
    parser.add_argument("--oracle-chunk-size", type=int,
        default=DEFAULT_ORACLE_CHUNK_SIZE)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve(strict=True)
    if not output_dir.is_dir():
        raise SystemExit("output directory must already exist and be a directory")
    oracle_path = output_dir / "oracle.json"
    replicate_path = output_dir / "replicates.jsonl"
    result_path = output_dir / "result.json"
    for path in (oracle_path, replicate_path, result_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")

    reference = make_oracle_references(histories=args.oracle_histories,
        chunk_size=args.oracle_chunk_size, seed=args.oracle_seed,
        size=DEFAULT_GRID_SIZE, draws=DEFAULT_DRAWS)
    _write_json_exclusive(oracle_path, reference)
    if not reference["gate_replicates_may_start"]:
        print(json.dumps(dict(status=reference["status"],
            reasons=reference["reasons"]), indent=2), file=sys.stderr)
        return 2

    with replicate_path.open("x", encoding="utf-8") as records:
        def persist_replicate(row):
            records.write(json.dumps(row, allow_nan=False) + "\n")
            records.flush()

        result = run_fixed_stress(seed=args.seed,
            replicates=args.replicates, size=DEFAULT_GRID_SIZE,
            draws=DEFAULT_DRAWS, reference=reference,
            on_replicate=persist_replicate)

    result["job_metadata"] = dict(
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        submit_host=os.environ.get("SLURM_SUBMIT_HOST"),
        hostname=socket.gethostname(),
        partition=os.environ.get("SLURM_JOB_PARTITION"),
        cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK"),
        memory_per_node_mb=os.environ.get("SLURM_MEM_PER_NODE"),
        source_commit=os.environ.get("CF4_EXPECTED_COMMIT"))
    result["paired_replicates_file"] = replicate_path.name
    result["oracle_file"] = oracle_path.name
    _write_json_exclusive(result_path, result)
    print(json.dumps(dict(status=result["status"],
        gate_counts=result["summary"]["gate_counts"],
        false_acceptances=result["summary"]["false_acceptances"],
        false_acceptance_upper95=(
            result["summary"]["one_sided_95_exact_clopper_pearson_upper"]),
        result=str(result_path)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
