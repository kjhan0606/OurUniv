"""Fail-closed topology-aware approximation for the frozen Q1 operator.

Q6 showed that a complete (bitwise) knot signature gives essentially one
group per source.  Q7 deliberately removes the coefficient bytes from the
group key, while retaining the discrete TSC topology (ordered stencil map,
clip regime and sigma-zero branch).  A group may be approximated only when an
interval enclosure is supplied for its response.  In the absence of a
continuous topology-cell certificate the public evaluator falls back to the
sourcewise Q1 response and records the source as overflow.  This prevents a
finite fixture envelope from being silently promoted to a continuous error
claim.

The module is an in-memory development component.  It has no JAX, posterior,
IC, PM/HOP, GPFS or Slurm path.  Geometry derivatives are intentionally out of
scope; the directional basis covers only mass/state directions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np

from cf4_2mpp_joint_likelihood_local import LikelihoodInputError
from cf4_q1_cell_integrated_convolution import (
    Q1_DEFAULT_TAIL_CUTOFF,
    cell_integrated_tsc_deposit,
    tsc_deposit,
)
from cf4_q6_knot_aligned_operator import (
    KnotCompressedSources,
    contract_field,
    exact_knot_compress,
)


Q7_ROUTE_NAME = "adaptive_topology_aware_response_atlas"
Q7_DERIVATIVE_DIRECTIONS = 23
Q7_CERTIFIED_COVERAGE = "continuous_topology_cell"
Q7_FINITE_COVERAGE = "finite_source_set"


@dataclass(frozen=True)
class TopologyBin:
    """A deterministic set of sources sharing the discrete Q1 topology."""

    index: int
    key: tuple[object, ...]
    members: np.ndarray
    representative: int
    coverage: str
    continuous_certified: bool


@dataclass(frozen=True)
class TopologyAwareResponseAtlas:
    """Retained representative responses and certified response enclosures."""

    compressed: KnotCompressedSources
    bins: tuple[TopologyBin, ...]
    source_to_bin: np.ndarray
    representative_fields: np.ndarray
    lower_enclosures: tuple[np.ndarray, ...]
    upper_enclosures: tuple[np.ndarray, ...]
    finite_spread_l1: np.ndarray
    finite_spread_linf: np.ndarray
    q1_tail_cutoff: float
    max_host_bytes: int

    @property
    def source_count(self) -> int:
        return self.compressed.source_count

    @property
    def bin_count(self) -> int:
        return len(self.bins)

    @property
    def compression_ratio(self) -> float:
        return float(self.source_count / self.bin_count)


@dataclass(frozen=True)
class ApproximationResult:
    """Candidate fields and separate certified/finite/measured error budgets."""

    fields: np.ndarray
    gradients: np.ndarray
    overflow_source_count: int
    overflow_fraction: float
    used_uncertified_finite_enclosure: bool
    certificate_status: str
    certified_value_l1_per_population: np.ndarray
    certified_value_linf_per_population: np.ndarray
    finite_value_l1_per_population: np.ndarray
    finite_value_linf_per_population: np.ndarray
    certified_gradient_l1: np.ndarray
    finite_gradient_l1: np.ndarray
    measured: Mapping[str, float] | None


def _topology_key(contract: object) -> tuple[object, ...]:
    """Return only discrete topology/regime bytes, never coefficients/breaks."""

    mode = getattr(contract, "mode")
    target = np.asarray(getattr(contract, "target_cells"), dtype="<i8", order="C")
    clip = np.asarray(getattr(contract, "clip_mask"), dtype="|u1", order="C")
    if mode == "zero_scale_tsc":
        return (mode, target.tobytes())
    breaks = np.asarray(getattr(contract, "breaks"), dtype=np.float64)
    # The sliver merge rule is a discrete regime boundary even when the final
    # interval count happens to remain unchanged; retain its activation bit in
    # the topology key so a bin cannot silently straddle that boundary.
    sliver_active = bool(getattr(contract, "dropped_sliver_probability") > 0.0)
    return (mode, int(breaks.size), sliver_active, target.tobytes(), clip.tobytes())


def _validate_enclosure(
    lower: np.ndarray,
    upper: np.ndarray,
    shape: tuple[int, int, int],
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.shape != shape or upper.shape != shape:
        raise LikelihoodInputError(f"{label} enclosure must have shape {shape}")
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower < 0.0)
        or np.any(upper < lower)
    ):
        raise LikelihoodInputError(f"{label} enclosure must be finite, ordered and non-negative")
    return lower.copy(), upper.copy()


def build_topology_aware_atlas(
    positions: Iterable[Iterable[float]],
    los_unit_vectors: Iterable[Iterable[float]],
    displacement_scales: Iterable[float],
    grid_size: int,
    box_size_cMpc_h: float,
    *,
    tail_cutoff: float = Q1_DEFAULT_TAIL_CUTOFF,
    continuous_enclosures: Mapping[int, tuple[Iterable[Iterable[Iterable[float]]], Iterable[Iterable[Iterable[float]]]]] | None = None,
    max_host_bytes: int = 64 * 1024**3,
) -> TopologyAwareResponseAtlas:
    """Build one deterministic topology atlas.

    ``continuous_enclosures`` is keyed by the prospective bin index and must
    enclose every response component over that *continuous* topology cell.  If
    omitted, bounds are constructed over the finite input source set solely as
    a diagnostic; those bins are not certified and the evaluator falls back to
    sourcewise Q1 unless explicitly asked to use the finite envelope.
    """

    if not isinstance(max_host_bytes, int) or max_host_bytes <= 0:
        raise LikelihoodInputError("max_host_bytes must be a positive integer")
    if not math.isfinite(tail_cutoff) or tail_cutoff <= 0.0:
        raise LikelihoodInputError("tail_cutoff must be positive and finite")
    compressed = exact_knot_compress(
        positions,
        los_unit_vectors,
        displacement_scales,
        grid_size,
        box_size_cMpc_h,
        tail_cutoff=tail_cutoff,
    )
    topology_to_bin: dict[tuple[object, ...], int] = {}
    source_to_bin = np.empty(compressed.source_count, dtype=np.int64)
    members: list[list[int]] = []
    representative_exact_group: list[int] = []
    keys: list[tuple[object, ...]] = []
    for source, exact_group in enumerate(compressed.source_to_group):
        key = _topology_key(compressed.contracts[int(exact_group)])
        bin_index = topology_to_bin.get(key)
        if bin_index is None:
            bin_index = len(members)
            topology_to_bin[key] = bin_index
            members.append([])
            representative_exact_group.append(int(exact_group))
            keys.append(key)
        members[bin_index].append(source)
        source_to_bin[source] = bin_index

    representative_fields: list[np.ndarray] = []
    finite_lower: list[np.ndarray] = []
    finite_upper: list[np.ndarray] = []
    finite_l1: list[float] = []
    finite_linf: list[float] = []
    bins: list[TopologyBin] = []
    field_shape = (grid_size, grid_size, grid_size)
    estimated_bytes = len(members) * int(np.prod(field_shape)) * 3 * 8
    if estimated_bytes > max_host_bytes:
        # Do not allocate an atlas that violates the frozen host limit.  The
        # sourcewise fallback remains available through ``evaluate_atlas``.
        raise LikelihoodInputError(
            f"topology atlas estimate {estimated_bytes} bytes exceeds max_host_bytes {max_host_bytes}"
        )
    for bin_index, source_members in enumerate(members):
        exact_group = representative_exact_group[bin_index]
        contract = compressed.contracts[exact_group]
        if contract.mode == "zero_scale_tsc":
            representative = tsc_deposit(
                compressed.positions[exact_group : exact_group + 1],
                np.asarray([1.0], dtype=np.float64),
                grid_size,
                box_size_cMpc_h,
            )
        else:
            representative = contract_field(contract)
        representative = np.asarray(representative, dtype=np.float64)
        if representative.shape != field_shape or not np.all(np.isfinite(representative)):
            raise LikelihoodInputError("representative Q1 response is invalid")
        if np.any(representative < 0.0) or not np.isclose(
            np.sum(representative), 1.0, rtol=0.0, atol=3.0e-13
        ):
            raise LikelihoodInputError("representative Q1 response violates non-negativity/conservation")
        lower = representative.copy()
        upper = representative.copy()
        # Exact-knot contracts are retained once per exact group; evaluating
        # those fields supplies a deterministic finite-source diagnostic hull.
        for source in source_members:
            source_group = int(compressed.source_to_group[source])
            source_contract = compressed.contracts[source_group]
            if source_contract.mode == "zero_scale_tsc":
                source_field = tsc_deposit(
                    compressed.positions[source_group : source_group + 1],
                    np.asarray([1.0], dtype=np.float64),
                    grid_size,
                    box_size_cMpc_h,
                )
            else:
                source_field = contract_field(source_contract)
            lower = np.minimum(lower, source_field)
            upper = np.maximum(upper, source_field)
        lower, upper = _validate_enclosure(lower, upper, field_shape, label=f"bin {bin_index}")
        if continuous_enclosures is not None and bin_index in continuous_enclosures:
            lower, upper = _validate_enclosure(
                *continuous_enclosures[bin_index], field_shape, label=f"continuous bin {bin_index}"
            )
            # A declared continuous certificate must at least enclose the
            # actual fixture members; otherwise fail closed immediately.
            for source in source_members:
                source_group = int(compressed.source_to_group[source])
                source_contract = compressed.contracts[source_group]
                source_field = (
                    tsc_deposit(
                        compressed.positions[source_group : source_group + 1],
                        np.asarray([1.0], dtype=np.float64),
                        grid_size,
                        box_size_cMpc_h,
                    )
                    if source_contract.mode == "zero_scale_tsc"
                    else contract_field(source_contract)
                )
                if np.any(source_field < lower) or np.any(source_field > upper):
                    raise LikelihoodInputError(
                        f"continuous enclosure for bin {bin_index} misses a fixture response"
                    )
            coverage = Q7_CERTIFIED_COVERAGE
            continuous_certified = True
        else:
            coverage = Q7_FINITE_COVERAGE
            continuous_certified = False
        spread = np.maximum(np.abs(lower - representative), np.abs(upper - representative))
        representative_fields.append(representative)
        finite_lower.append(lower)
        finite_upper.append(upper)
        finite_l1.append(float(np.sum(spread)))
        finite_linf.append(float(np.max(spread)))
        bins.append(
            TopologyBin(
                index=bin_index,
                key=keys[bin_index],
                members=np.asarray(source_members, dtype=np.int64),
                representative=representative_exact_group[bin_index],
                coverage=coverage,
                continuous_certified=continuous_certified,
            )
        )
    return TopologyAwareResponseAtlas(
        compressed=compressed,
        bins=tuple(bins),
        source_to_bin=source_to_bin,
        representative_fields=np.stack(representative_fields, axis=0),
        lower_enclosures=tuple(finite_lower),
        upper_enclosures=tuple(finite_upper),
        finite_spread_l1=np.asarray(finite_l1, dtype=np.float64),
        finite_spread_linf=np.asarray(finite_linf, dtype=np.float64),
        q1_tail_cutoff=float(tail_cutoff),
        max_host_bytes=int(max_host_bytes),
    )


def _validate_mass_inputs(
    atlas: TopologyAwareResponseAtlas,
    population_masses: Iterable[Iterable[float]],
    directional_mass_basis: Iterable[Iterable[float]] | Iterable[Iterable[Iterable[float]]] | None,
) -> tuple[np.ndarray, np.ndarray | None, str | None]:
    masses = np.asarray(population_masses, dtype=np.float64)
    if masses.ndim != 2 or masses.shape[1] != atlas.source_count or masses.shape[0] == 0:
        raise LikelihoodInputError("population_masses must have shape (P, M)")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
        raise LikelihoodInputError("population_masses must be finite and non-negative")
    if directional_mass_basis is None:
        return masses, None, None
    basis = np.asarray(directional_mass_basis, dtype=np.float64)
    if not np.all(np.isfinite(basis)):
        raise LikelihoodInputError("directional_mass_basis must be finite")
    if basis.ndim == 2 and basis.shape[1] == atlas.source_count:
        if basis.shape[0] != Q7_DERIVATIVE_DIRECTIONS:
            raise LikelihoodInputError("directional_mass_basis must have 23 directions")
        return masses, basis, "direction_source"
    if basis.ndim == 3 and basis.shape[:2] == masses.shape and basis.shape[2] == Q7_DERIVATIVE_DIRECTIONS:
        return masses, basis, "population_direction_source"
    raise LikelihoodInputError("directional_mass_basis must have shape (23,M) or (P,M,23)")


def _source_field(atlas: TopologyAwareResponseAtlas, source: int) -> np.ndarray:
    group = int(atlas.compressed.source_to_group[source])
    contract = atlas.compressed.contracts[group]
    if contract.mode == "zero_scale_tsc":
        return tsc_deposit(
            atlas.compressed.positions[group : group + 1],
            np.asarray([1.0], dtype=np.float64),
            contract.grid_size,
            contract.box_size_cMpc_h,
        )
    return contract_field(contract)


def evaluate_atlas(
    atlas: TopologyAwareResponseAtlas,
    population_masses: Iterable[Iterable[float]],
    *,
    directional_mass_basis: Iterable[Iterable[float]] | Iterable[Iterable[Iterable[float]]] | None = None,
    allow_uncertified_finite_enclosure: bool = False,
    oracle_fields: Iterable[Iterable[Iterable[Iterable[float]]]] | None = None,
) -> ApproximationResult:
    """Evaluate the route and return certified, finite and optional measured budgets."""

    masses, basis, basis_mode = _validate_mass_inputs(
        atlas, population_masses, directional_mass_basis
    )
    populations = masses.shape[0]
    field_shape = atlas.representative_fields.shape[1:]
    fields = np.zeros((populations,) + field_shape, dtype=np.float64)
    if basis_mode == "direction_source":
        gradients = np.zeros((Q7_DERIVATIVE_DIRECTIONS,) + field_shape, dtype=np.float64)
    elif basis_mode == "population_direction_source":
        gradients = np.zeros((populations, Q7_DERIVATIVE_DIRECTIONS) + field_shape, dtype=np.float64)
    else:
        gradients = np.zeros((0,) + field_shape, dtype=np.float64)
    overflow = 0
    used_uncertified = False
    cert_value_l1 = np.zeros(populations, dtype=np.float64)
    cert_value_linf = np.zeros(populations, dtype=np.float64)
    finite_value_l1 = np.zeros(populations, dtype=np.float64)
    finite_value_linf = np.zeros(populations, dtype=np.float64)
    cert_grad_l1 = np.zeros(Q7_DERIVATIVE_DIRECTIONS, dtype=np.float64)
    finite_grad_l1 = np.zeros(Q7_DERIVATIVE_DIRECTIONS, dtype=np.float64)
    for bin_index, topology_bin in enumerate(atlas.bins):
        members = topology_bin.members
        certified = topology_bin.continuous_certified
        approximate = certified or allow_uncertified_finite_enclosure
        if approximate:
            if not certified:
                used_uncertified = True
            rep = atlas.representative_fields[bin_index]
            grouped_mass = np.sum(masses[:, members], axis=1)
            fields += grouped_mass[(slice(None),) + (None,) * 3] * rep[None, ...]
            spread_l1 = float(atlas.finite_spread_l1[bin_index])
            spread_linf = float(atlas.finite_spread_linf[bin_index])
            finite_value_l1 += np.sum(masses[:, members], axis=1) * spread_l1
            finite_value_linf += np.sum(masses[:, members], axis=1) * spread_linf
            if certified:
                cert_value_l1 += np.sum(masses[:, members], axis=1) * spread_l1
                cert_value_linf += np.sum(masses[:, members], axis=1) * spread_linf
            if basis_mode == "direction_source":
                grouped_basis = np.sum(basis[:, members], axis=1)
                gradients += grouped_basis[(slice(None),) + (None,) * 3] * rep[None, ...]
                finite_grad_l1 += np.sum(np.abs(basis[:, members]), axis=1) * spread_l1
                if certified:
                    cert_grad_l1 += np.sum(np.abs(basis[:, members]), axis=1) * spread_l1
            elif basis_mode == "population_direction_source":
                grouped_basis = np.sum(basis[:, members, :], axis=1)
                gradients += grouped_basis[(slice(None), slice(None)) + (None,) * 3] * rep[None, None, ...]
                finite_grad_l1 += np.sum(np.abs(basis[:, members, :]), axis=(0, 1)) * spread_l1
                if certified:
                    cert_grad_l1 += np.sum(np.abs(basis[:, members, :]), axis=(0, 1)) * spread_l1
            continue
        overflow += int(members.size)
        for source in members:
            field = _source_field(atlas, int(source))
            fields += masses[:, source, None, None, None] * field[None, ...]
            if basis_mode == "direction_source":
                gradients += basis[:, source, None, None, None] * field[None, ...]
            elif basis_mode == "population_direction_source":
                gradients += basis[:, source, :, None, None, None] * field[None, None, ...]
    if not np.all(np.isfinite(fields)) or np.any(fields < 0.0):
        raise LikelihoodInputError("topology-aware fields are not finite and non-negative")
    measured: dict[str, float] | None = None
    if oracle_fields is not None:
        oracle = np.asarray(oracle_fields, dtype=np.float64)
        if oracle.shape != fields.shape or not np.all(np.isfinite(oracle)) or np.any(oracle < 0.0):
            raise LikelihoodInputError("oracle_fields must match non-negative field shape")
        delta = np.abs(fields - oracle)
        measured = {
            "value_l1": float(np.sum(delta)),
            "value_linf": float(np.max(delta)),
            "value_l1_per_population_max": float(np.max(np.sum(delta, axis=(1, 2, 3)))),
        }
    if used_uncertified:
        certificate_status = "MEASURED_ONLY_FINITE_SOURCE_SET"
        # A finite envelope cannot certify a continuous cell.  Keep the useful
        # finite numbers above, but make the promoted certificate unmistakable.
        cert_value_l1[:] = np.inf
        cert_value_linf[:] = np.inf
        cert_grad_l1[:] = np.inf
    elif overflow:
        certificate_status = "CERTIFIED_WITH_SOURCEWISE_OVERFLOW"
    else:
        certificate_status = "CERTIFIED"
    return ApproximationResult(
        fields=fields,
        gradients=gradients,
        overflow_source_count=overflow,
        overflow_fraction=float(overflow / atlas.source_count),
        used_uncertified_finite_enclosure=used_uncertified,
        certificate_status=certificate_status,
        certified_value_l1_per_population=cert_value_l1,
        certified_value_linf_per_population=cert_value_linf,
        finite_value_l1_per_population=finite_value_l1,
        finite_value_linf_per_population=finite_value_linf,
        certified_gradient_l1=cert_grad_l1,
        finite_gradient_l1=finite_grad_l1,
        measured=measured,
    )


def candidate_metadata(atlas: TopologyAwareResponseAtlas) -> dict[str, object]:
    """Return provenance and scope metadata for an auditable Q7 candidate."""

    certified_bins = sum(item.continuous_certified for item in atlas.bins)
    return {
        "route": Q7_ROUTE_NAME,
        "representation": "topology_key_without_numeric_knot_coefficients",
        "source_count": atlas.source_count,
        "bin_count": atlas.bin_count,
        "compression_ratio": atlas.compression_ratio,
        "certified_bin_count": certified_bins,
        "finite_only_bin_count": atlas.bin_count - certified_bins,
        "topology_key_components": [
            "Q1 mode (including sigma-zero branch)",
            "ordered interval count",
            "absolute 27-cell stencil mapping",
            "negative-clip mask",
            "sliver-merge activation bit",
        ],
        "numeric_knot_coefficients_in_key": False,
        "tail_cutoff_sigma": atlas.q1_tail_cutoff,
        "frozen_direction_count": Q7_DERIVATIVE_DIRECTIONS,
        "geometry_derivatives": False,
        "overflow_policy": "sourcewise Q1 fallback; no clipping or tolerance relaxation",
        "continuous_certificate_required_for_promotion": True,
        "science_claim_authorized": False,
    }


def compare_to_q1(
    candidate: Iterable[Iterable[Iterable[Iterable[float]]]],
    oracle: Iterable[Iterable[Iterable[Iterable[float]]]],
) -> dict[str, float]:
    """Compute measured value errors without applying a promotion decision."""

    candidate_array = np.asarray(candidate, dtype=np.float64)
    oracle_array = np.asarray(oracle, dtype=np.float64)
    if candidate_array.shape != oracle_array.shape or candidate_array.ndim != 4:
        raise LikelihoodInputError("candidate and oracle must share shape (P,N,N,N)")
    if not np.all(np.isfinite(candidate_array)) or not np.all(np.isfinite(oracle_array)):
        raise LikelihoodInputError("candidate and oracle must be finite")
    delta = np.abs(candidate_array - oracle_array)
    return {
        "value_l1": float(np.sum(delta)),
        "value_linf": float(np.max(delta)),
        "value_l1_per_population_max": float(np.max(np.sum(delta, axis=(1, 2, 3)))),
        "oracle_l1": float(np.sum(np.abs(oracle_array))),
        "oracle_field_max": float(np.max(oracle_array)),
    }


def sourcewise_q1_fields(
    atlas: TopologyAwareResponseAtlas,
    population_masses: Iterable[Iterable[float]],
) -> np.ndarray:
    """Return the frozen Q1 reference for the same atlas inputs."""

    masses, _, _ = _validate_mass_inputs(atlas, population_masses, None)
    result = np.zeros((masses.shape[0],) + atlas.representative_fields.shape[1:], dtype=np.float64)
    for source in range(atlas.source_count):
        result += masses[:, source, None, None, None] * _source_field(atlas, source)[None, ...]
    return result
