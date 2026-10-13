#!/usr/bin/env python3
"""Build and validate the reusable exact high-k conditional covariance cache.

The expensive part of the exact conditional completion depends only on the
unique LG geometry keys, not on the parent or the independent fine-field seed.
This module deliberately runs on CPU and writes one self-contained ``.npz``
artifact atomically.  It never opens a parent field or generates a realization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cf4_aggregate_evidence_oracle import points_from_geometry_key
from cf4_peak_evidence_phase_cache import (
    covariance_for_point_sets,
    full_spectrum_from_rfft,
)


SCHEMA = "ouruniv-cf4-lg-highk-covariance-cache-v1"
RESULT_SCHEMA = "ouruniv-cf4-lg-highk-covariance-cache-result-v1"
ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    """Write an npz to a sibling temporary name, then replace it atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _program_inputs(program_path: Path) -> tuple[dict[str, Any], Path, Path, str, str]:
    """Read the streaming program and validate the two cache source pins."""
    program = json.loads(Path(program_path).read_text())
    if program.get("schema") != "ouruniv-cf4-lg-highk-streaming-forward-program-v1":
        raise ValueError("unexpected streaming-forward program schema")
    try:
        schedule_spec = program["inputs"]["schedule"]
        filter_spec = program["inputs"]["density_filter"]
        schedule_path = _resolve_path(schedule_spec["path"])
        filter_path = _resolve_path(filter_spec["path"])
        schedule_sha = str(schedule_spec["sha256"])
        filter_sha = str(filter_spec["sha256"])
    except (KeyError, TypeError) as error:
        raise ValueError("program lacks pinned schedule or density_filter input") from error
    if sha256_file(schedule_path) != schedule_sha:
        raise RuntimeError("program schedule SHA256 changed")
    if sha256_file(filter_path) != filter_sha:
        raise RuntimeError("program density_filter SHA256 changed")
    return program, schedule_path, filter_path, schedule_sha, filter_sha


def _schedule_keys(schedule_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(schedule_path, allow_pickle=False) as item:
        required = {"schedule_index", "keys"}
        missing = required.difference(item.files)
        if missing:
            raise ValueError(f"schedule is missing arrays: {sorted(missing)}")
        index = np.asarray(item["schedule_index"], dtype=np.int64)
        keys = np.asarray(item["keys"], dtype=np.int64)
    if index.ndim != 1 or not np.array_equal(index, np.arange(len(index))):
        raise ValueError("schedule indices must be a contiguous identity")
    if keys.shape != (len(index), 6):
        raise ValueError("schedule keys must have shape (row_count, 6)")
    return index, keys


def _key_lookup(unique_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Map every six-integer key to its row in lexicographically sorted keys."""
    table = {tuple(row.tolist()): number for number, row in enumerate(unique_keys)}
    try:
        result = np.asarray([table[tuple(row.tolist())] for row in keys], dtype=np.int32)
    except KeyError as error:
        raise ValueError("schedule key is absent from covariance cache") from error
    if not np.array_equal(unique_keys[result], keys):
        raise ValueError("covariance cache key mapping is not exact")
    return result


def build_covariance_cache(
    *,
    schedule_path: Path,
    filter_path: Path,
    output_path: Path,
    coarse_n: int = 192,
    result_path: Path | None = None,
    program_path: Path | None = None,
) -> dict[str, Any]:
    """Compute exact AQA* matrices once per unique scheduled geometry key."""
    schedule_path, filter_path = Path(schedule_path), Path(filter_path)
    output_path = Path(output_path)
    result_path = Path(result_path) if result_path is not None else None
    existing = [path for path in (output_path, result_path) if path is not None and path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite covariance-cache artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    schedule_index, keys = _schedule_keys(schedule_path)
    unique_keys = np.unique(keys, axis=0)
    filter_rfft = np.load(filter_path, allow_pickle=False)
    filter_full = full_spectrum_from_rfft(filter_rfft)
    fine_n = int(filter_full.shape[0])
    if fine_n % int(coarse_n):
        raise ValueError("filter mesh must be an integer multiple of coarse_n")
    points = np.asarray(
        [points_from_geometry_key(key, fine_n=fine_n) for key in unique_keys],
        dtype=np.int64,
    )
    covariance_list, covariance_diagnostics = covariance_for_point_sets(
        filter_full, int(coarse_n), list(points)
    )
    covariance = np.asarray(covariance_list, dtype=np.float64)
    row_to_unique = _key_lookup(unique_keys, keys)
    diagnostics: dict[str, Any] = {
        "schema": SCHEMA,
        "schedule_path": str(schedule_path.resolve()),
        "schedule_sha256": sha256_file(schedule_path),
        "filter_path": str(filter_path.resolve()),
        "filter_sha256": sha256_file(filter_path),
        "coarse_mesh": int(coarse_n),
        "fine_mesh": fine_n,
        "schedule_row_count": int(len(schedule_index)),
        "unique_key_count": int(len(unique_keys)),
        "constraint_count": int(points.shape[1]),
        "covariance": covariance_diagnostics,
    }
    _atomic_savez(
        output_path,
        schema=np.asarray(SCHEMA),
        schedule_index=schedule_index,
        schedule_keys=keys,
        row_to_unique=row_to_unique,
        unique_keys=unique_keys,
        points=points,
        covariance=covariance,
        diagnostics_json=np.asarray(json.dumps(diagnostics, sort_keys=True)),
    )
    # Re-open the artifact before returning; a successful replace alone is not
    # sufficient evidence that its serialized mapping is usable.
    validation = validate_covariance_cache(
        output_path, schedule_path=schedule_path, filter_path=filter_path
    )
    if result_path is not None:
        result = {
            "schema": RESULT_SCHEMA,
            "status": "complete_exact_covariance_cache",
            "cache": str(Path(output_path).resolve()),
            "cache_sha256": validation["sha256"],
            "inputs": {
                "schedule": {
                    "path": str(schedule_path.resolve()),
                    "sha256": diagnostics["schedule_sha256"],
                },
                "density_filter": {
                    "path": str(filter_path.resolve()),
                    "sha256": diagnostics["filter_sha256"],
                },
            },
            "diagnostics": validation["diagnostics"],
            "cache_validated": True,
            "field_generated": False,
            "science_selection_performed": False,
            "program": (
                str(Path(program_path).resolve()) if program_path is not None else None
            ),
            "program_sha256": (
                sha256_file(Path(program_path)) if program_path is not None else None
            ),
        }
        _atomic_json(Path(result_path), result)
    return validation


def validate_covariance_cache(
    path: Path,
    *,
    schedule_path: Path | None = None,
    filter_path: Path | None = None,
) -> dict[str, Any]:
    """Validate serialized key/point/covariance identity and optional input pins."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as item:
        required = {
            "schema", "schedule_index", "schedule_keys", "row_to_unique",
            "unique_keys", "points", "covariance", "diagnostics_json",
        }
        missing = required.difference(item.files)
        if missing:
            raise ValueError(f"covariance cache is missing arrays: {sorted(missing)}")
        schema = str(np.asarray(item["schema"]).item())
        schedule_index = np.asarray(item["schedule_index"], dtype=np.int64)
        schedule_keys = np.asarray(item["schedule_keys"], dtype=np.int64)
        row_to_unique = np.asarray(item["row_to_unique"], dtype=np.int64)
        unique_keys = np.asarray(item["unique_keys"], dtype=np.int64)
        points = np.asarray(item["points"], dtype=np.int64)
        covariance = np.asarray(item["covariance"], dtype=np.float64)
        diagnostics = json.loads(str(np.asarray(item["diagnostics_json"]).item()))
    if schema != SCHEMA or diagnostics.get("schema") != SCHEMA:
        raise ValueError("unexpected covariance cache schema")
    if schedule_keys.shape != (len(schedule_index), 6) \
            or row_to_unique.shape != (len(schedule_index),):
        raise ValueError("cache schedule mapping shape changed")
    if not np.array_equal(schedule_index, np.arange(len(schedule_index))):
        raise ValueError("cache schedule identity changed")
    if unique_keys.ndim != 2 or unique_keys.shape[1] != 6:
        raise ValueError("cache unique keys shape changed")
    if not np.array_equal(unique_keys, np.unique(unique_keys, axis=0)):
        raise ValueError("cache unique keys are not sorted and unique")
    if np.any(row_to_unique < 0) or np.any(row_to_unique >= len(unique_keys)) \
            or not np.array_equal(unique_keys[row_to_unique], schedule_keys):
        raise ValueError("cache row-to-key mapping changed")
    fine_n = int(diagnostics["fine_mesh"])
    expected_points = np.asarray(
        [points_from_geometry_key(key, fine_n=fine_n) for key in unique_keys],
        dtype=np.int64,
    )
    if points.shape != expected_points.shape or not np.array_equal(points, expected_points):
        raise ValueError("cache points do not reconstruct from geometry keys")
    if covariance.shape != (len(unique_keys), points.shape[1], points.shape[1]):
        raise ValueError("cache covariance shape changed")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("cache covariance contains non-finite values")
    asymmetry = float(np.max(np.abs(covariance - np.swapaxes(covariance, 1, 2))))
    if asymmetry > 1.0e-10:
        raise ValueError("cache covariance is not symmetric")
    if schedule_path is not None:
        schedule_index_check, schedule_keys_check = _schedule_keys(Path(schedule_path))
        if not np.array_equal(schedule_index, schedule_index_check) \
                or not np.array_equal(schedule_keys, schedule_keys_check):
            raise ValueError("cache does not map the requested schedule")
        if diagnostics.get("schedule_sha256") != sha256_file(Path(schedule_path)):
            raise ValueError("cache schedule SHA256 does not match requested schedule")
    if filter_path is not None and diagnostics.get("filter_sha256") != sha256_file(Path(filter_path)):
        raise ValueError("cache filter SHA256 does not match requested filter")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "diagnostics": diagnostics,
        "maximum_covariance_asymmetry": asymmetry,
    }


def load_covariance_for_schedule_row(
    path: Path, schedule_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (geometry key, points, AQA*) for an already validated cache row."""
    with np.load(path, allow_pickle=False) as item:
        rows = np.asarray(item["row_to_unique"], dtype=np.int64)
        keys = np.asarray(item["unique_keys"], dtype=np.int64)
        points = np.asarray(item["points"], dtype=np.int64)
        covariance = np.asarray(item["covariance"], dtype=np.float64)
    if schedule_index < 0 or schedule_index >= len(rows):
        raise IndexError("schedule index is outside covariance cache")
    key_index = int(rows[schedule_index])
    return keys[key_index].copy(), points[key_index].copy(), covariance[key_index].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--filter", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--coarse-n", type=int, default=192)
    parser.add_argument("--validate", action="store_true", help="validate an existing --output only")
    args = parser.parse_args()
    if args.program:
        program, schedule, density_filter, _, _ = _program_inputs(args.program)
        if not args.validate and program.get("authorization", {}).get("covariance_cache_execution") is not True:
            parser.error("covariance-cache execution is not authorized by the frozen program")
        cache_spec = program["covariance_cache"]
        output = args.output or _resolve_path(
            Path(cache_spec["artifact_root"]) / cache_spec["cache_file"]
        )
        result_path = args.result or _resolve_path(
            Path(cache_spec["artifact_root"]) / cache_spec["result_file"]
        )
    else:
        if not args.validate:
            parser.error("cache construction requires --program authorization")
        if args.schedule is None or args.filter is None or args.output is None:
            parser.error("--schedule, --filter, and --output are required without --program")
        schedule, density_filter, output, result_path = (
            args.schedule, args.filter, args.output, args.result)
    if args.validate:
        result = validate_covariance_cache(
            output, schedule_path=schedule, filter_path=density_filter
        )
    else:
        result = build_covariance_cache(
            schedule_path=schedule,
            filter_path=density_filter,
            output_path=output,
            coarse_n=args.coarse_n,
            result_path=result_path,
            program_path=args.program,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
