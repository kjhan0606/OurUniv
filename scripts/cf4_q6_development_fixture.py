#!/usr/bin/env python3
"""Reproduce the pre-registered Q6 development geometry and digest.

This generator is intentionally small and deterministic.  It writes no files,
does not read held-out data, and performs no Q1/operator evaluation.  The
digest scope is the concatenation of little-endian C-contiguous float64 bytes
for ``positions``, ``los``, ``scales`` and ``masses`` in that order.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np


def generate_fixture(
    *,
    seed: int = 62006,
    source_count: int = 192,
    grid_size: int = 8,
    box_size_cMpc_h: float = 8.0,
    include_cold: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate the exact Q6 smoke arrays from the declared draw sequence."""

    if source_count <= 0 or grid_size < 2 or box_size_cMpc_h <= 0.0:
        raise ValueError("source_count/grid_size must be positive and box size >= 0")
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, box_size_cMpc_h, size=(source_count, 3))
    los = rng.normal(size=(source_count, 3))
    los /= np.linalg.norm(los, axis=1)[:, None]
    scales = rng.uniform(0.05, 1.25, size=source_count)
    if include_cold:
        spacing = box_size_cMpc_h / grid_size
        scales[0] = 0.0
        scales[1] = 64.0 * np.finfo(np.float64).eps * spacing * 0.5
    masses = np.abs(rng.normal(size=(6, source_count)))
    return positions, los, scales, masses


def fixture_sha256(
    positions: np.ndarray,
    los: np.ndarray,
    scales: np.ndarray,
    masses: np.ndarray,
) -> str:
    """Hash the declared little-endian C-contiguous float64 byte stream."""

    payload = b"".join(
        np.asarray(array, dtype="<f8", order="C").tobytes()
        for array in (positions, los, scales, masses)
    )
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=62006)
    parser.add_argument("--source-count", type=int, default=192)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--box-size", type=float, default=8.0)
    parser.add_argument("--include-cold", action="store_true")
    args = parser.parse_args()
    arrays = generate_fixture(
        seed=args.seed,
        source_count=args.source_count,
        grid_size=args.grid_size,
        box_size_cMpc_h=args.box_size,
        include_cold=args.include_cold,
    )
    print(
        json.dumps(
            {
                "seed": args.seed,
                "source_count": args.source_count,
                "grid_size": args.grid_size,
                "box_size_cMpc_h": args.box_size,
                "include_cold": args.include_cold,
                "digest_scope": "sha256(LE-float64-C-contiguous positions||los||scales||masses)",
                "fixture_sha256": fixture_sha256(*arrays),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

