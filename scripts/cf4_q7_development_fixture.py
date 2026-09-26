#!/usr/bin/env python3
"""Deterministic Q7 development fixtures reusing the frozen Q6 seed set.

The seven family names and seeds are copied from the Q6 manifest.  Geometry
draws use the Q6 PCG64 stream; family transforms are deterministic and are
applied before the declared little-endian float64 digest is computed.  These
fixtures are development-only and never open the sealed Q4 held-out set.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np


FAMILIES: tuple[tuple[str, int, int], ...] = (
    ("same_knot_different_coefficient", 62001, 32),
    ("translated_cell", 62002, 32),
    ("near_coincident", 62003, 32),
    ("interval_permutation", 62004, 32),
    ("seam_sliver_clip", 62005, 32),
    ("los_sigma_variation", 62006, 192),
    ("cold_zero_near_zero", 62007, 32),
)


def _base_fixture(
    *, seed: int, source_count: int, grid_size: int, box_size_cMpc_h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if source_count <= 0 or grid_size < 2 or box_size_cMpc_h <= 0.0:
        raise ValueError("source_count/grid_size must be positive and box size > 0")
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, box_size_cMpc_h, size=(source_count, 3))
    los = rng.normal(size=(source_count, 3))
    los /= np.linalg.norm(los, axis=1)[:, None]
    scales = rng.uniform(0.05, 1.25, size=source_count)
    masses = np.abs(rng.normal(size=(6, source_count)))
    return positions, los, scales, masses


def generate_fixture(
    *,
    family: str,
    seed: int | None = None,
    source_count: int | None = None,
    grid_size: int = 8,
    box_size_cMpc_h: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate one of the seven pre-registered Q7 families."""

    declared = {name: (declared_seed, declared_count) for name, declared_seed, declared_count in FAMILIES}
    if family not in declared:
        raise ValueError(f"unknown family {family!r}; choose one of {sorted(declared)}")
    declared_seed, declared_count = declared[family]
    if seed is None:
        seed = declared_seed
    if source_count is None:
        source_count = declared_count
    if seed != declared_seed or source_count != declared_count:
        raise ValueError("Q7 fixture seeds and source counts are frozen by the manifest")
    positions, los, scales, masses = _base_fixture(
        seed=seed,
        source_count=source_count,
        grid_size=grid_size,
        box_size_cMpc_h=box_size_cMpc_h,
    )
    spacing = box_size_cMpc_h / grid_size
    if family == "same_knot_different_coefficient":
        # Identical geometry with independent population masses: geometry-only
        # grouping must still preserve every population/state coefficient.
        half = source_count // 2
        positions[half:] = positions[:half]
        los[half:] = los[:half]
        scales[half:] = scales[:half]
    elif family == "translated_cell":
        half = source_count // 2
        positions[half:] = (positions[:half] + np.asarray([spacing, 0.0, 0.0])) % box_size_cMpc_h
    elif family == "near_coincident":
        positions[1] = np.nextafter(positions[0], np.inf)
        los[1] = los[0]
        scales[1] = scales[0]
    elif family == "interval_permutation":
        # Reverse LOS for alternating rows and use a different sigma scale so
        # the ordered epsilon intervals cannot be inferred from position only.
        los[::2] *= -1.0
        scales[1::2] = np.flip(scales[1::2])
    elif family == "seam_sliver_clip":
        positions[0] = np.asarray(
            [np.nextafter(box_size_cMpc_h, 0.0), 0.5 * spacing, 0.5 * spacing]
        )
        positions[1] = np.asarray([0.0, 0.5 * spacing, 0.5 * spacing])
        positions[2] = np.asarray([0.5 * spacing, 0.5 * spacing, 0.5 * spacing])
        scales[0] = 64.0 * np.finfo(np.float64).eps * spacing * 0.5
        scales[1] = 0.25
        scales[2] = 1.0
    elif family == "cold_zero_near_zero":
        scales[0] = 0.0
        scales[1] = 64.0 * np.finfo(np.float64).eps * spacing * 0.5
    # los remains normalized after sign changes and all transforms preserve the
    # Q1 domain contract.
    return positions, los, scales, masses


def fixture_sha256(
    positions: np.ndarray,
    los: np.ndarray,
    scales: np.ndarray,
    masses: np.ndarray,
) -> str:
    payload = b"".join(
        np.asarray(array, dtype="<f8", order="C").tobytes()
        for array in (positions, los, scales, masses)
    )
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=[name for name, _, _ in FAMILIES])
    args = parser.parse_args()
    arrays = generate_fixture(family=args.family)
    seed = next(seed for name, seed, _ in FAMILIES if name == args.family)
    source_count = next(count for name, _, count in FAMILIES if name == args.family)
    print(
        json.dumps(
            {
                "family": args.family,
                "seed": seed,
                "source_count": source_count,
                "grid_size": 8,
                "box_size_cMpc_h": 8.0,
                "digest_scope": "sha256(LE-float64-C-contiguous positions||los||scales||masses)",
                "fixture_sha256": fixture_sha256(*arrays),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

