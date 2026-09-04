"""Exact knot-aligned aggregation contract for the frozen Q1 operator.

Q1 is piecewise polynomial in the scalar Gaussian displacement ``epsilon``.
This module groups sources only when *all* data that determine that piecewise
polynomial are bitwise identical: the absolute target-cell/27-stencil map,
ordered interval boundaries and midpoints, degree-six local coefficients,
sliver handling, clipping, tail cutoff and per-source renormalisation.  The
contract is deliberately conservative.  It makes no cross-knot Taylor claim;
sources that do not share a complete signature are left as singleton groups.

The implementation is intended for small Q6 development fixtures.  It does
not allocate full PM fields, launch JAX/GPFS/Slurm jobs, or authorize posterior
or IC generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.special import ndtr

from cf4_2mpp_joint_likelihood_local import LikelihoodInputError
from cf4_q1_cell_integrated_convolution import (
    Q1_DEFAULT_TAIL_CUTOFF,
    Q1_ORACLE_NEGATIVE_CLIP_TOLERANCE,
    Q1_SLIVER_EPSILON,
    _axis_breakpoints,
    _interval_stencil,
    _normal_moments,
    tsc_deposit,
)


DEGREE = 6
POPULATIONS = 6
DERIVATIVE_DIRECTIONS = 23
FLOAT64_EPSILON = np.finfo(np.float64).eps


@dataclass(frozen=True)
class KnotContract:
    """The complete immutable-by-convention per-source Q1 signature payload."""

    key: bytes
    grid_size: int
    box_size_cMpc_h: float
    tail_cutoff: float
    mode: str
    breaks: np.ndarray
    midpoints: np.ndarray
    target_cells: np.ndarray
    coefficients: np.ndarray
    clip_mask: np.ndarray
    pre_renormalization_total: float
    dropped_sliver_probability: float
    clipped_negative_mass: float


@dataclass(frozen=True)
class KnotCompressedSources:
    """Complete knot-signature groups and deterministic source ownership."""

    positions: np.ndarray
    los_unit_vectors: np.ndarray
    displacement_scales: np.ndarray
    contracts: tuple[KnotContract, ...]
    source_to_group: np.ndarray
    source_count: int
    group_count: int

    @property
    def compression_ratio(self) -> float:
        return float(self.source_count / self.group_count)


def _validate_geometry(
    positions: Iterable[Iterable[float]],
    los_unit_vectors: Iterable[Iterable[float]],
    displacement_scales: Iterable[float],
    *,
    grid_size: int,
    box_size_cMpc_h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(grid_size, int) or grid_size < 2:
        raise LikelihoodInputError("grid_size must be an integer >= 2")
    if not math.isfinite(box_size_cMpc_h) or box_size_cMpc_h <= 0.0:
        raise LikelihoodInputError("box_size_cMpc_h must be positive and finite")
    pos = np.asarray(positions, dtype=np.float64)
    los = np.asarray(los_unit_vectors, dtype=np.float64)
    scale = np.asarray(displacement_scales, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] == 0:
        raise LikelihoodInputError("positions must have shape (M, 3)")
    if los.shape != pos.shape or scale.shape != (pos.shape[0],):
        raise LikelihoodInputError("LOS and displacement arrays must match positions")
    if (
        not np.all(np.isfinite(pos))
        or not np.all(np.isfinite(los))
        or not np.all(np.isfinite(scale))
    ):
        raise LikelihoodInputError("geometry arrays must be finite")
    if np.any(pos < 0.0) or np.any(pos >= box_size_cMpc_h):
        raise LikelihoodInputError("positions must lie in [0, box_size)")
    if np.any(scale < 0.0):
        raise LikelihoodInputError("displacement_scales must be non-negative")
    norms = np.linalg.norm(los, axis=1)
    if np.any(norms <= 0.0) or not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-12):
        raise LikelihoodInputError("los_unit_vectors must be finite unit vectors")
    return pos, los, scale


def _merge_breakpoints(
    position: np.ndarray,
    displacement: np.ndarray,
    spacing: float,
    box_size: float,
    grid_size: int,
    tail_cutoff: float,
) -> tuple[np.ndarray, float]:
    """Reproduce Q1's sorted break/sliver rule byte-for-byte in float64."""

    boundaries = [-tail_cutoff, tail_cutoff]
    for axis in range(3):
        boundaries.extend(
            _axis_breakpoints(position[axis], displacement[axis], spacing, tail_cutoff)
        )
    raw_breaks = np.unique(np.asarray(boundaries, dtype=np.float64))
    breaks = [float(raw_breaks[0])]
    dropped = 0.0
    for boundary in raw_breaks[1:]:
        gap = float(boundary) - breaks[-1]
        if gap > Q1_SLIVER_EPSILON:
            breaks.append(float(boundary))
        else:
            dropped += float(ndtr(boundary) - ndtr(breaks[-1]))
    return np.asarray(breaks, dtype=np.float64), float(dropped)


def _pack_array(array: np.ndarray, dtype: str) -> bytes:
    """Canonical little-endian payload for collision-free dictionary keys."""

    return np.asarray(array, dtype=dtype, order="C").tobytes()


def _build_contract(
    position: np.ndarray,
    los: np.ndarray,
    scale: float,
    grid_size: int,
    box_size: float,
    tail_cutoff: float,
) -> KnotContract:
    spacing = box_size / grid_size
    displacement = float(scale) * los
    if scale == 0.0:
        # Q1's zero-scale branch is a deterministic point TSC deposit.  Keep
        # its complete target support and field bytes in the signature so no
        # different cell or coefficient can be merged accidentally.
        nearest = np.floor(position / spacing).astype(np.int64) % grid_size
        target = np.asarray(
            [
                [(nearest[0] + dx) % grid_size,
                 (nearest[1] + dy) % grid_size,
                 (nearest[2] + dz) % grid_size]
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
            ],
            dtype=np.int64,
        )
        unit_field = tsc_deposit(
            position[None, :], np.asarray([1.0]), grid_size, box_size
        )
        key = b"zero-scale-v1" + b"".join(
            (
                _pack_array(np.asarray([grid_size], dtype=np.int64), "<i8"),
                _pack_array(np.asarray([box_size, tail_cutoff], dtype=np.float64), "<f8"),
                _pack_array(target, "<i8"),
                _pack_array(unit_field, "<f8"),
            )
        )
        return KnotContract(
            key=key,
            grid_size=grid_size,
            box_size_cMpc_h=box_size,
            tail_cutoff=tail_cutoff,
            mode="zero_scale_tsc",
            breaks=np.asarray([], dtype=np.float64),
            midpoints=np.asarray([], dtype=np.float64),
            target_cells=target,
            coefficients=np.asarray([], dtype=np.float64).reshape(0, 27, 7),
            clip_mask=np.asarray([], dtype=np.bool_).reshape(0, 27),
            pre_renormalization_total=1.0,
            dropped_sliver_probability=0.0,
            clipped_negative_mass=0.0,
        )

    breaks, dropped_sliver_probability = _merge_breakpoints(
        position, displacement, spacing, box_size, grid_size, tail_cutoff
    )
    interval_count = max(0, breaks.size - 1)
    midpoints = np.empty(interval_count, dtype=np.float64)
    targets = np.empty((interval_count, 27, 3), dtype=np.int64)
    coefficients = np.zeros((interval_count, 27, 7), dtype=np.float64)
    clip_mask = np.zeros((interval_count, 27), dtype=np.bool_)
    total = 0.0
    clipped_negative_mass = 0.0
    for interval, (left, right) in enumerate(zip(breaks[:-1], breaks[1:])):
        if not right > left:
            continue
        midpoint = 0.5 * (float(left) + float(right))
        midpoints[interval] = midpoint
        nearest, axis_polynomials = _interval_stencil(
            position, displacement, midpoint, spacing, box_size, grid_size
        )
        moments = _normal_moments(float(left), float(right), DEGREE, center=midpoint)
        entry = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    polynomial = np.polynomial.polynomial.polymul(
                        np.polynomial.polynomial.polymul(
                            axis_polynomials[0][dx + 1], axis_polynomials[1][dy + 1]
                        ),
                        axis_polynomials[2][dz + 1],
                    )
                    coefficients[interval, entry, : polynomial.size] = polynomial
                    targets[interval, entry, :] = (
                        (nearest[0] + dx) % grid_size,
                        (nearest[1] + dy) % grid_size,
                        (nearest[2] + dz) % grid_size,
                    )
                    value = float(np.dot(polynomial, moments[: polynomial.size]))
                    if value < -Q1_ORACLE_NEGATIVE_CLIP_TOLERANCE:
                        raise LikelihoodInputError(
                            "Q1 knot contract produced a negative weight"
                        )
                    if value < 0.0:
                        clip_mask[interval, entry] = True
                        clipped_negative_mass += -value
                    total += max(0.0, value)
                    entry += 1
    if not math.isfinite(total) or total <= 0.0:
        raise LikelihoodInputError("Q1 knot contract has zero or non-finite mass")
    key = b"knot-v2" + b"".join(
        (
            _pack_array(np.asarray([grid_size], dtype=np.int64), "<i8"),
            _pack_array(np.asarray([box_size, tail_cutoff, Q1_SLIVER_EPSILON, Q1_ORACLE_NEGATIVE_CLIP_TOLERANCE], dtype=np.float64), "<f8"),
            _pack_array(breaks, "<f8"),
            _pack_array(midpoints, "<f8"),
            _pack_array(targets, "<i8"),
            _pack_array(coefficients, "<f8"),
            _pack_array(clip_mask, "|u1"),
            _pack_array(np.asarray([total, dropped_sliver_probability, clipped_negative_mass], dtype=np.float64), "<f8"),
        )
    )
    return KnotContract(
        key=key,
        grid_size=grid_size,
        box_size_cMpc_h=box_size,
        tail_cutoff=tail_cutoff,
        mode="gaussian_t8_piecewise_degree6",
        breaks=breaks,
        midpoints=midpoints,
        target_cells=targets,
        coefficients=coefficients,
        clip_mask=clip_mask,
        pre_renormalization_total=float(total),
        dropped_sliver_probability=float(dropped_sliver_probability),
        clipped_negative_mass=float(clipped_negative_mass),
    )


def exact_knot_compress(
    positions: Iterable[Iterable[float]],
    los_unit_vectors: Iterable[Iterable[float]],
    displacement_scales: Iterable[float],
    grid_size: int,
    box_size_cMpc_h: float,
    *,
    tail_cutoff: float = Q1_DEFAULT_TAIL_CUTOFF,
) -> KnotCompressedSources:
    """Group only complete, bitwise-identical Q1 knot contracts."""

    if not math.isfinite(tail_cutoff) or tail_cutoff <= 0.0:
        raise LikelihoodInputError("tail_cutoff must be positive and finite")
    pos, los, scale = _validate_geometry(
        positions,
        los_unit_vectors,
        displacement_scales,
        grid_size=grid_size,
        box_size_cMpc_h=box_size_cMpc_h,
    )
    key_to_group: dict[bytes, int] = {}
    source_to_group = np.empty(pos.shape[0], dtype=np.int64)
    group_positions: list[np.ndarray] = []
    group_los: list[np.ndarray] = []
    group_scales: list[float] = []
    contracts: list[KnotContract] = []
    for source in range(pos.shape[0]):
        contract = _build_contract(
            pos[source],
            los[source],
            float(scale[source]),
            grid_size,
            float(box_size_cMpc_h),
            float(tail_cutoff),
        )
        group = key_to_group.get(contract.key)
        if group is None:
            group = len(contracts)
            key_to_group[contract.key] = group
            group_positions.append(pos[source].copy())
            group_los.append(los[source].copy())
            group_scales.append(float(scale[source]))
            contracts.append(contract)
        source_to_group[source] = group
    return KnotCompressedSources(
        positions=np.asarray(group_positions, dtype=np.float64),
        los_unit_vectors=np.asarray(group_los, dtype=np.float64),
        displacement_scales=np.asarray(group_scales, dtype=np.float64),
        contracts=tuple(contracts),
        source_to_group=source_to_group,
        source_count=int(pos.shape[0]),
        group_count=len(contracts),
    )


def contract_field(contract: KnotContract) -> np.ndarray:
    """Evaluate one retained knot contract using Q1's exact moments."""

    field = np.zeros(
        (contract.grid_size, contract.grid_size, contract.grid_size), dtype=np.float64
    )
    if contract.mode == "zero_scale_tsc":
        # Reconstructing zero-scale coefficients is unnecessary and would risk
        # changing Q1's point-TSC arithmetic.  The representative route below
        # is used by ``evaluate_grouped_q1_operator`` for this branch.
        raise LikelihoodInputError("zero-scale contract requires representative evaluation")
    for interval, (left, right) in enumerate(zip(contract.breaks[:-1], contract.breaks[1:])):
        moments = _normal_moments(float(left), float(right), DEGREE, center=float(contract.midpoints[interval]))
        for entry in range(27):
            value = float(np.dot(contract.coefficients[interval, entry], moments))
            if value < 0.0:
                value = 0.0
            target = tuple(int(value) for value in contract.target_cells[interval, entry])
            field[target] += value
    field /= contract.pre_renormalization_total
    if not np.all(np.isfinite(field)) or np.any(field < 0.0):
        raise LikelihoodInputError("contract field is not finite and non-negative")
    return field


def aggregate_population_masses(
    compressed: KnotCompressedSources,
    population_masses: Iterable[Iterable[float]],
) -> np.ndarray:
    """Aggregate all population masses while preserving state dependence."""

    masses = np.asarray(population_masses, dtype=np.float64)
    if masses.ndim != 2 or masses.shape[1] != compressed.source_count:
        raise LikelihoodInputError("population_masses must have shape (P, M)")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
        raise LikelihoodInputError("population_masses must be finite and non-negative")
    grouped = np.zeros((masses.shape[0], compressed.group_count), dtype=np.float64)
    for source, group in enumerate(compressed.source_to_group):
        grouped[:, group] += masses[:, source]
    return grouped


def aggregate_mass_basis(
    compressed: KnotCompressedSources,
    mass_basis: Iterable[Iterable[Iterable[float]]],
) -> np.ndarray:
    """Apply the exact group transpose to an arbitrary source basis."""

    basis = np.asarray(mass_basis, dtype=np.float64)
    if basis.ndim != 3 or basis.shape[1] != compressed.source_count:
        raise LikelihoodInputError("mass_basis must have shape (P, M, B)")
    if not np.all(np.isfinite(basis)):
        raise LikelihoodInputError("mass_basis must be finite")
    grouped = np.zeros((basis.shape[0], compressed.group_count, basis.shape[2]), dtype=np.float64)
    for source, group in enumerate(compressed.source_to_group):
        grouped[:, group, :] += basis[:, source, :]
    return grouped


def scatter_group_cotangent(
    compressed: KnotCompressedSources,
    group_cotangent: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Scatter any group cotangent back to sources (the exact transpose map)."""

    cotangent = np.asarray(group_cotangent, dtype=np.float64)
    if cotangent.ndim == 0 or cotangent.shape[0] != compressed.group_count:
        raise LikelihoodInputError("group_cotangent first dimension must be group_count")
    if not np.all(np.isfinite(cotangent)):
        raise LikelihoodInputError("group_cotangent must be finite")
    return cotangent[compressed.source_to_group]


def evaluate_grouped_q1_operator(
    compressed: KnotCompressedSources,
    population_masses: Iterable[Iterable[float]],
) -> np.ndarray:
    """Evaluate grouped Q1 fields using retained exact contracts."""

    grouped_masses = aggregate_population_masses(compressed, population_masses)
    result = np.zeros(
        (grouped_masses.shape[0], compressed.contracts[0].grid_size,
         compressed.contracts[0].grid_size, compressed.contracts[0].grid_size),
        dtype=np.float64,
    )
    for group, contract in enumerate(compressed.contracts):
        if contract.mode == "zero_scale_tsc":
            field = tsc_deposit(
                compressed.positions[group : group + 1],
                np.asarray([1.0], dtype=np.float64),
                contract.grid_size,
                contract.box_size_cMpc_h,
            )
        else:
            field = contract_field(contract)
        for population in range(grouped_masses.shape[0]):
            result[population] += grouped_masses[population, group] * field
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise LikelihoodInputError("grouped knot result is not finite and non-negative")
    return result


def candidate_metadata(compressed: KnotCompressedSources) -> dict[str, object]:
    """Return auditable metadata without authorising a science claim."""

    interval_counts = [max(0, contract.breaks.size - 1) for contract in compressed.contracts]
    return {
        "representation": "exact_knot_aligned_q1_contract",
        "source_count": compressed.source_count,
        "group_count": compressed.group_count,
        "compression_ratio": compressed.compression_ratio,
        "signature_components": [
            "absolute_target_cell_and_27_stencil_mapping",
            "ordered_breakpoints_and_midpoints",
            "degree_6_local_epsilon_coefficients",
            "sliver_merge_rule",
            "negative_clip_mask",
            "tail_cutoff_and_per_source_renormalization",
        ],
        "interval_count_min": min(interval_counts),
        "interval_count_max": max(interval_counts),
        "arbitrary_cotangent_transpose_preserved": True,
        "approximation": False,
        "science_claim_authorized": False,
    }
