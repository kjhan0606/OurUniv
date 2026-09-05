"""Development-only exact source compression for the frozen Q1 operator.

The first Q3 candidate is deliberately conservative: sources are grouped only
when position, LOS direction and displacement scale are bitwise identical.
Because the Q1 response is linear in source mass, summing masses (or a complete
state-dependent mass/gradient basis) within such a group is an exact sufficient
statistic.  No quantisation or geometry approximation is hidden here; if a
catalog has no duplicate response signatures, the compression ratio is one and
the candidate is a scientifically safe but computationally unsuccessful
NO-GO, rather than a lossy shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit
from cf4_2mpp_joint_likelihood_local import LikelihoodInputError


@dataclass(frozen=True)
class ExactCompressedSources:
    """Geometry groups and deterministic source-to-group ownership."""

    positions: np.ndarray
    los_unit_vectors: np.ndarray
    displacement_scales: np.ndarray
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(positions, dtype=np.float64)
    los = np.asarray(los_unit_vectors, dtype=np.float64)
    scale = np.asarray(displacement_scales, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] == 0:
        raise LikelihoodInputError("positions must have shape (M, 3)")
    if los.shape != pos.shape or scale.shape != (pos.shape[0],):
        raise LikelihoodInputError("LOS and displacement arrays must match positions")
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(los)) or not np.all(np.isfinite(scale)):
        raise LikelihoodInputError("geometry arrays must be finite")
    if np.any(scale < 0.0):
        raise LikelihoodInputError("displacement_scales must be non-negative")
    norms = np.linalg.norm(los, axis=1)
    if np.any(norms <= 0.0) or not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-12):
        raise LikelihoodInputError("los_unit_vectors must be finite unit vectors")
    return pos, los, scale


def exact_geometry_compress(
    positions: Iterable[Iterable[float]],
    los_unit_vectors: Iterable[Iterable[float]],
    displacement_scales: Iterable[float],
) -> ExactCompressedSources:
    """Group only bitwise-identical complete Q1 response signatures."""

    pos, los, scale = _validate_geometry(positions, los_unit_vectors, displacement_scales)
    key_to_group: dict[bytes, int] = {}
    group_for_source = np.empty(pos.shape[0], dtype=np.int64)
    group_positions: list[np.ndarray] = []
    group_los: list[np.ndarray] = []
    group_scales: list[float] = []
    for index in range(pos.shape[0]):
        # C-contiguous byte keys preserve signed zero and every float64 bit;
        # no tolerance-based geometry merge is allowed by this exact candidate.
        key = pos[index].tobytes() + los[index].tobytes() + scale[index].tobytes()
        group = key_to_group.get(key)
        if group is None:
            group = len(group_positions)
            key_to_group[key] = group
            group_positions.append(pos[index].copy())
            group_los.append(los[index].copy())
            group_scales.append(float(scale[index]))
        group_for_source[index] = group
    return ExactCompressedSources(
        positions=np.asarray(group_positions, dtype=np.float64),
        los_unit_vectors=np.asarray(group_los, dtype=np.float64),
        displacement_scales=np.asarray(group_scales, dtype=np.float64),
        source_to_group=group_for_source,
        source_count=int(pos.shape[0]),
        group_count=len(group_positions),
    )


def aggregate_population_masses(
    compressed: ExactCompressedSources,
    population_masses: Iterable[Iterable[float]],
) -> np.ndarray:
    """Aggregate current state masses without dropping population dimensions."""

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
    compressed: ExactCompressedSources,
    mass_basis: Iterable[Iterable[Iterable[float]]],
) -> np.ndarray:
    """Aggregate a complete state-dependent mass/gradient basis exactly.

    The input shape is ``(population, source, basis_coordinate)``.  Summing
    this basis, rather than only a fiducial mass vector, is the contract that
    preserves latent-field and bias derivatives for every posterior state.
    """

    basis = np.asarray(mass_basis, dtype=np.float64)
    if basis.ndim != 3 or basis.shape[1] != compressed.source_count:
        raise LikelihoodInputError("mass_basis must have shape (P, M, B)")
    if not np.all(np.isfinite(basis)):
        raise LikelihoodInputError("mass_basis must be finite")
    grouped = np.zeros((basis.shape[0], compressed.group_count, basis.shape[2]), dtype=np.float64)
    for source, group in enumerate(compressed.source_to_group):
        grouped[:, group, :] += basis[:, source, :]
    return grouped


def evaluate_grouped_q1_operator(
    compressed: ExactCompressedSources,
    population_masses: Iterable[Iterable[float]],
    grid_size: int,
    box_size_cMpc_h: float,
    *,
    tail_cutoff: float = 8.0,
) -> np.ndarray:
    """Evaluate the frozen Q1 NumPy operator using exact geometry groups."""

    grouped_masses = aggregate_population_masses(compressed, population_masses)
    result = np.zeros((grouped_masses.shape[0], grid_size, grid_size, grid_size), dtype=np.float64)
    for group in range(compressed.group_count):
        for population in range(grouped_masses.shape[0]):
            mass = float(grouped_masses[population, group])
            if mass == 0.0:
                continue
            field = cell_integrated_tsc_deposit(
                compressed.positions[group : group + 1],
                np.asarray([mass], dtype=np.float64),
                compressed.los_unit_vectors[group : group + 1],
                compressed.displacement_scales[group : group + 1],
                grid_size,
                box_size_cMpc_h,
                tail_cutoff=tail_cutoff,
            )
            result[population] += field
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise LikelihoodInputError("grouped Q1 result is not finite and non-negative")
    return result


def candidate_metadata(compressed: ExactCompressedSources) -> dict[str, object]:
    """Return auditable compression metadata; no science claim is emitted."""

    return {
        "representation": "exact_geometry_grouping",
        "source_count": compressed.source_count,
        "group_count": compressed.group_count,
        "compression_ratio": compressed.compression_ratio,
        "bitwise_geometry_key": True,
        "state_dependent_mass_basis_preserved": True,
        "approximation": False,
        "science_claim_authorized": False,
    }
