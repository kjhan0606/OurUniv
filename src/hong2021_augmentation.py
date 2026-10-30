#!/usr/bin/env python
"""Exact cube symmetries for Hong-style observer-centred fields.

The observer is fixed at the cube centre, so translations are not a symmetry of
the radial-velocity observable.  Signed axis permutations are exact: the six
axis permutations combined with the eight independent axis reflections form
the full 48-element octahedral group, including the 24 mirror transformations.
"""
from __future__ import annotations

from itertools import permutations, product
from typing import TypeVar

import numpy as np


Array = TypeVar("Array", bound=np.ndarray)
AXIS_PERMUTATIONS = tuple(permutations((0, 1, 2)))
CUBE_ISOMETRIES = tuple(
    (permutation, reflections)
    for permutation in AXIS_PERMUTATIONS
    for reflections in product((False, True), repeat=3)
)


def permutation_parity(permutation: tuple[int, int, int]) -> int:
    """Return +1 for even and -1 for odd three-axis permutations."""
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(3)
        for j in range(i + 1, 3)
    )
    return -1 if inversions % 2 else 1


def is_mirror_isometry(
    permutation: tuple[int, int, int],
    reflections: tuple[bool, bool, bool],
) -> bool:
    """Whether a signed permutation reverses orientation (determinant -1)."""
    determinant = permutation_parity(permutation)
    determinant *= -1 if sum(reflections) % 2 else 1
    return determinant < 0


def apply_cube_isometry(
    value: Array,
    permutation: tuple[int, int, int],
    reflections: tuple[bool, bool, bool],
) -> Array:
    """Transform the final three axes of an array by one cube isometry."""
    array = np.asarray(value)
    if array.ndim < 3:
        raise ValueError("cube isometry requires at least three dimensions")
    if sorted(permutation) != [0, 1, 2]:
        raise ValueError(f"invalid spatial permutation: {permutation}")
    if len(reflections) != 3:
        raise ValueError("reflections must contain three booleans")
    leading = tuple(range(array.ndim - 3))
    spatial = tuple(array.ndim - 3 + axis for axis in permutation)
    transformed = np.transpose(array, leading + spatial)
    flip_axes = tuple(
        transformed.ndim - 3 + axis
        for axis, enabled in enumerate(reflections)
        if enabled
    )
    if flip_axes:
        transformed = np.flip(transformed, axis=flip_axes)
    return np.ascontiguousarray(transformed)


def random_cube_isometry(
    value: Array, rng: np.random.Generator
) -> tuple[Array, int]:
    """Apply a uniformly sampled member of the full 48-element group."""
    index = int(rng.integers(len(CUBE_ISOMETRIES)))
    permutation, reflections = CUBE_ISOMETRIES[index]
    return apply_cube_isometry(value, permutation, reflections), index
