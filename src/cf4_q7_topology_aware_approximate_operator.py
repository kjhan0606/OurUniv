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
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from cf4_2mpp_joint_likelihood_local import LikelihoodInputError
from cf4_q1_cell_integrated_convolution import (
    Q1_DEFAULT_TAIL_CUTOFF,
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
Q7_Q1_SOURCE_SHA256 = "74ae1bb12171a2baac76c8052d592b4dc5098043bf7c11bca6ffb9eea852d6b2"
Q7_Q6_SOURCE_SHA256 = "c7e18760312a17b608957ca344c00c4099846e9d80c878f34ccc07e8b7e1ffae"
Q7_SIGMA_NEAR_ZERO_FACTOR = 64.0
Q7_SUMMARY_COUNT = 175


def _assert_frozen_q1_q6_provenance() -> None:
    """Pin both frozen dependencies locally, rather than transitively."""

    root = Path(__file__).resolve().parents[1]
    q1 = root / "src" / "cf4_q1_cell_integrated_convolution.py"
    q6 = root / "src" / "cf4_q6_knot_aligned_operator.py"
    q1_sha = hashlib.sha256(q1.read_bytes()).hexdigest()
    q6_sha = hashlib.sha256(q6.read_bytes()).hexdigest()
    if q1_sha != Q7_Q1_SOURCE_SHA256:
        raise RuntimeError(f"Q7 frozen Q1 SHA mismatch: expected {Q7_Q1_SOURCE_SHA256}, got {q1_sha}")
    if q6_sha != Q7_Q6_SOURCE_SHA256:
        raise RuntimeError(f"Q7 frozen Q6 SHA mismatch: expected {Q7_Q6_SOURCE_SHA256}, got {q6_sha}")


_assert_frozen_q1_q6_provenance()


@dataclass(frozen=True)
class OutwardInterval:
    """A small directed-rounding interval used for the Q7 certificate."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)) or self.lo > self.hi:
            raise LikelihoodInputError("invalid finite interval")

    @staticmethod
    def point(value: float) -> "OutwardInterval":
        value = float(value)
        return OutwardInterval(np.nextafter(value, -math.inf), np.nextafter(value, math.inf))

    @staticmethod
    def _down(value: float) -> float:
        return float(np.nextafter(float(value), -math.inf))

    @staticmethod
    def _up(value: float) -> float:
        return float(np.nextafter(float(value), math.inf))

    def __add__(self, other: "OutwardInterval") -> "OutwardInterval":
        return OutwardInterval(self._down(self.lo + other.lo), self._up(self.hi + other.hi))

    def __sub__(self, other: "OutwardInterval") -> "OutwardInterval":
        return OutwardInterval(self._down(self.lo - other.hi), self._up(self.hi - other.lo))

    def __mul__(self, other: "OutwardInterval") -> "OutwardInterval":
        values = (self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi)
        return OutwardInterval(self._down(min(values)), self._up(max(values)))

    def __truediv__(self, other: "OutwardInterval") -> "OutwardInterval":
        if other.lo <= 0.0 <= other.hi:
            raise LikelihoodInputError("interval division crosses zero")
        values = (self.lo / other.lo, self.lo / other.hi, self.hi / other.lo, self.hi / other.hi)
        return OutwardInterval(self._down(min(values)), self._up(max(values)))

    def widen(self, amount: float) -> "OutwardInterval":
        if not math.isfinite(amount) or amount < 0.0:
            raise LikelihoodInputError("interval widening must be finite and non-negative")
        return OutwardInterval(self._down(self.lo - amount), self._up(self.hi + amount))


def _frozen_direction_names() -> tuple[str, ...]:
    """Load the registered 23-direction semantic contract and verify its hash."""

    path = Path(__file__).resolve().parents[1] / "config" / "cf4_q7_frozen_23_mass_basis_v1.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = tuple(payload["direction_names"])
        expected = str(payload["direction_names_sha256"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("frozen Q7 23-direction basis contract is unreadable") from exc
    digest = hashlib.sha256(json.dumps(names, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    if len(names) != Q7_DERIVATIVE_DIRECTIONS or digest != expected:
        raise RuntimeError("frozen Q7 23-direction basis contract hash/count mismatch")
    return names


FROZEN_DIRECTION_NAMES = _frozen_direction_names()


def _read_registered_manifest(value: str | Path, *, label: str) -> dict[str, object]:
    """Read a repository-owned registry manifest, never an arbitrary caller map."""

    root = Path(__file__).resolve().parents[1]
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    config_root = (root / "config").resolve()
    try:
        path.relative_to(config_root)
    except ValueError as exc:
        raise LikelihoodInputError(f"{label} manifest must be inside the repository config directory") from exc
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise LikelihoodInputError(f"{label} manifest is unreadable") from exc
    if not isinstance(payload, dict):
        raise LikelihoodInputError(f"{label} manifest must contain a JSON object")
    status = str(payload.get("status", ""))
    if status not in {"REGISTERED", "REGISTERED_DEVELOPMENT"}:
        raise LikelihoodInputError(f"{label} manifest status is not registered: {status!r}")
    return payload


@dataclass(frozen=True)
class TopologyBin:
    """A deterministic set of sources sharing the discrete Q1 topology."""

    index: int
    key: tuple[object, ...]
    members: np.ndarray
    representative: int
    coverage: str
    continuous_certified: bool
    certificate_method: str


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
    certified_spread_l1: np.ndarray
    certified_spread_linf: np.ndarray
    finite_lower_enclosures: tuple[np.ndarray, ...]
    finite_upper_enclosures: tuple[np.ndarray, ...]
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
    certified_summary_l1_bounds: np.ndarray | None
    measured: Mapping[str, float] | None


def _topology_key(contract: object) -> tuple[object, ...]:
    """Return only discrete topology/regime bytes, never coefficients/breaks."""

    mode = getattr(contract, "mode")
    target = np.asarray(getattr(contract, "target_cells"), dtype="<i8", order="C")
    clip = np.asarray(getattr(contract, "clip_mask"), dtype="|u1", order="C")
    if mode == "zero_scale_tsc":
        # The target neighborhood alone is insufficient at sigma=0: point-TSC
        # weights vary continuously within a cell.  Include the exact unit
        # deposit payload (via the frozen contract key) so distinct subcell
        # responses cannot merge.
        return (mode, target.tobytes(), hashlib.sha256(getattr(contract, "key")).digest())
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


def _continuous_lipschitz_enclosure(
    compressed: KnotCompressedSources,
    source_members: list[int],
    representative_exact_group: int,
    representative: np.ndarray,
    *,
    grid_size: int,
    box_size: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Construct a validated enclosure over the observed topology cell.

    The response is a product of periodic TSC weights.  Each component has
    coordinate derivative bounded by ``2 / dx``.  For the normalized frozen
    Gaussian on ``[-T,T]``, ``E|epsilon| <= min(T, 1)``; this is an upper bound,
    not the truncated expectation used as a lower estimate.  Directed-rounded
    geometry radii therefore give a conservative componentwise Lipschitz
    enclosure.  If the observed hull touches a periodic seam, the sigma-zero
    branch, or a near-zero scale regime, no continuous claim is made and the
    public evaluator uses sourcewise Q1.
    """

    if not source_members:
        return None
    source_array = np.asarray(source_members, dtype=np.int64)
    groups = compressed.source_to_group[source_array]
    positions = compressed.positions[groups]
    scales = compressed.displacement_scales[groups]
    los = compressed.los_unit_vectors[groups]
    spacing = float(box_size / grid_size)
    if mode == "zero_scale_tsc":
        # The topology key deliberately includes the exact point-deposit field;
        # distinct subcell locations must never be merged into one certificate.
        return None
    if np.any(scales <= Q7_SIGMA_NEAR_ZERO_FACTOR * np.finfo(np.float64).eps * spacing):
        return None
    d = scales[:, None] * los
    position_lo = np.min(positions, axis=0)
    position_hi = np.max(positions, axis=0)
    # A topology cell is certified only while its z=0 position hull remains in
    # one periodic grid cell.  This prevents an absolute stencil index from
    # silently changing across a seam/boundary.
    for axis in range(3):
        lo_cell = math.floor(float(position_lo[axis] / spacing))
        hi_cell = math.floor(float(position_hi[axis] / spacing))
        if lo_cell != hi_cell or lo_cell < 0 or hi_cell >= grid_size:
            return None
        if position_lo[axis] <= lo_cell * spacing or position_hi[axis] >= (lo_cell + 1) * spacing:
            return None
    representative_position = compressed.positions[representative_exact_group]
    representative_scale = float(compressed.displacement_scales[representative_exact_group])
    representative_los = compressed.los_unit_vectors[representative_exact_group]
    representative_d = representative_scale * representative_los
    delta_position = np.max(np.abs(positions - representative_position[None, :]), axis=0)
    delta_scale = float(np.max(np.abs(scales - representative_scale)))
    # Bound changes in sigma explicitly in addition to the observed vector
    # displacement hull.  The extra term remains valid when LOS and sigma vary
    # together, rather than assuming a fixed direction.
    delta_d = np.max(np.abs(d - representative_d[None, :]), axis=0) + delta_scale
    if np.any(delta_position > 0.0) or np.any(delta_d > 0.0):
        # A raw support crossing 0/L would require a union of wrapped interval
        # branches.  Refuse certification until that union is represented.
        for position, displacement in zip(positions, d):
            support_lo = position - float(compressed.contracts[representative_exact_group].tail_cutoff) * np.abs(displacement)
            support_hi = position + float(compressed.contracts[representative_exact_group].tail_cutoff) * np.abs(displacement)
            if np.any(support_lo < 0.0) or np.any(support_hi >= box_size):
                return None
    try:
        radius = OutwardInterval.point(0.0)
        for axis in range(3):
            radius = radius + (OutwardInterval.point(float(delta_position[axis])) + OutwardInterval.point(float(delta_d[axis]))) / OutwardInterval.point(spacing)
        # The factor 2*min(T,1) is a directed-rounded upper bound for the
        # expected displacement contribution under the normalized Q1 tail.
        epsilon_abs_bound = min(float(compressed.contracts[representative_exact_group].tail_cutoff), 1.0)
        widening = (radius * OutwardInterval.point(2.0 * epsilon_abs_bound)).hi
        lower = np.empty_like(representative)
        upper = np.empty_like(representative)
        for index, value in np.ndenumerate(representative):
            interval = OutwardInterval.point(float(value)).widen(widening)
            lower[index] = max(0.0, interval.lo)
            upper[index] = min(1.0, interval.hi)
    except (LikelihoodInputError, FloatingPointError):
        return None
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(upper < lower):
        return None
    return lower, upper


def induced_summary_l1_bounds(
    per_bin_response_l1: Iterable[float],
    summary_operator_l1_norms: Iterable[Iterable[float]],
    *,
    summary_operator_l1_norms_sha256: str | None = None,
    summary_operator_registry_id: str | None = None,
    summary_operator_registry_manifest: str | Path | None = None,
) -> np.ndarray:
    """Propagate certified per-bin response bounds through 175 fixed maps.

    ``summary_operator_l1_norms`` is an externally frozen non-negative matrix
    with shape ``(175, number_of_bins)``.  The Q7 module validates its shape
    and finite values but does not invent those science-specific norms.
    """

    response = np.asarray(per_bin_response_l1, dtype=np.float64)
    norms = np.asarray(summary_operator_l1_norms, dtype=np.float64)
    if response.ndim != 1 or response.size == 0 or not np.all(np.isfinite(response)) or np.any(response < 0.0):
        raise LikelihoodInputError("per_bin_response_l1 must be finite and non-negative with shape (G,)")
    if norms.shape != (Q7_SUMMARY_COUNT, response.size) or not np.all(np.isfinite(norms)) or np.any(norms < 0.0):
        raise LikelihoodInputError(f"summary_operator_l1_norms must have shape ({Q7_SUMMARY_COUNT}, {response.size}) and be non-negative")
    if summary_operator_l1_norms_sha256 is None or summary_operator_registry_id is None or not str(summary_operator_registry_id).strip():
        raise LikelihoodInputError("registered summary-operator digest and registry id are required")
    norms_digest = hashlib.sha256(np.asarray(norms, dtype="<f8", order="C").tobytes()).hexdigest()
    if str(summary_operator_l1_norms_sha256) != norms_digest:
        raise LikelihoodInputError("summary operator numeric-content SHA256 mismatch")
    if summary_operator_registry_manifest is None:
        raise LikelihoodInputError("repository-owned summary-operator registry manifest is required")
    manifest = _read_registered_manifest(summary_operator_registry_manifest, label="summary operator")
    if str(manifest.get("registry_id", "")) != str(summary_operator_registry_id):
        raise LikelihoodInputError("summary operator registry id does not match manifest")
    if tuple(manifest.get("matrix_shape", ())) != norms.shape:
        raise LikelihoodInputError("summary operator shape does not match its registry manifest")
    if str(manifest.get("matrix_sha256", "")) != norms_digest:
        raise LikelihoodInputError("summary operator digest does not match its registry manifest")
    if str(manifest.get("norm_definition", "")) != "max_column_absolute_sum_induced_l1_norm":
        raise LikelihoodInputError("summary operator manifest does not define the required induced L1 norm")
    if not str(manifest.get("units", "")).strip() or not str(manifest.get("coordinate_convention", "")).strip():
        raise LikelihoodInputError("summary operator manifest must declare units and coordinate convention")
    # Accumulate each non-negative product with a nextafter-upward step.  A
    # plain BLAS matmul is not an upper bound in floating point and cannot
    # serve as a certified induced norm propagation.
    result = np.zeros(Q7_SUMMARY_COUNT, dtype=np.float64)
    for row in range(Q7_SUMMARY_COUNT):
        total = 0.0
        for column in range(response.size):
            term = float(norms[row, column] * response[column])
            term = float(np.nextafter(term, math.inf))
            total = float(np.nextafter(total + term, math.inf))
        result[row] = total
    if not np.all(np.isfinite(result)):
        raise LikelihoodInputError("induced summary bounds are non-finite")
    return result


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

    A continuous enclosure is generated internally with directed-rounded
    interval geometry and a frozen TSC Lipschitz bound.  Caller-supplied finite
    arrays are rejected: a finite fixture hull is not a continuous certificate.
    Bins that touch a cell seam or near-zero-sigma branch remain finite-only and
    are evaluated sourcewise by the fail-closed public route.
    """

    if continuous_enclosures is not None:
        raise LikelihoodInputError(
            "caller-supplied enclosures are not certification evidence; use the internal interval certificate"
        )
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
    certified_l1: list[float] = []
    certified_linf: list[float] = []
    finite_lower_diagnostic: list[np.ndarray] = []
    finite_upper_diagnostic: list[np.ndarray] = []
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
        # Keep the observed finite hull separately for diagnostics.  It is
        # never used as a promoted certificate.
        finite_lower_source = lower.copy()
        finite_upper_source = upper.copy()
        certified = _continuous_lipschitz_enclosure(
            compressed,
            source_members,
            exact_group,
            representative,
            grid_size=grid_size,
            box_size=box_size_cMpc_h,
            mode=contract.mode,
        )
        if certified is None:
            coverage = Q7_FINITE_COVERAGE
            continuous_certified = False
            certificate_method = "none_sourcewise_q1_fallback"
            certified_lower = representative.copy()
            certified_upper = representative.copy()
        else:
            coverage = Q7_CERTIFIED_COVERAGE
            continuous_certified = True
            certificate_method = "outward_lipschitz_interval"
            certified_lower, certified_upper = _validate_enclosure(
                *certified, field_shape, label=f"certified bin {bin_index}"
            )
        finite_spread = np.maximum(
            np.abs(finite_lower_source - representative),
            np.abs(finite_upper_source - representative),
        )
        certified_spread = np.maximum(
            np.abs(certified_lower - representative),
            np.abs(certified_upper - representative),
        )
        representative_fields.append(representative)
        # Public lower/upper fields always refer to the certified enclosure;
        # finite hulls are named explicitly and remain diagnostics only.
        finite_lower.append(certified_lower)
        finite_upper.append(certified_upper)
        finite_lower_diagnostic.append(finite_lower_source)
        finite_upper_diagnostic.append(finite_upper_source)
        finite_l1.append(float(np.sum(finite_spread)))
        finite_linf.append(float(np.max(finite_spread)))
        certified_l1.append(float(np.sum(certified_spread)))
        certified_linf.append(float(np.max(certified_spread)))
        bins.append(
            TopologyBin(
                index=bin_index,
                key=keys[bin_index],
                members=np.asarray(source_members, dtype=np.int64),
                representative=representative_exact_group[bin_index],
                coverage=coverage,
                continuous_certified=continuous_certified,
                certificate_method=certificate_method,
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
        certified_spread_l1=np.asarray(certified_l1, dtype=np.float64),
        certified_spread_linf=np.asarray(certified_linf, dtype=np.float64),
        finite_lower_enclosures=tuple(finite_lower_diagnostic),
        finite_upper_enclosures=tuple(finite_upper_diagnostic),
        q1_tail_cutoff=float(tail_cutoff),
        max_host_bytes=int(max_host_bytes),
    )


def _validate_mass_inputs(
    atlas: TopologyAwareResponseAtlas,
    population_masses: Iterable[Iterable[float]],
    directional_mass_basis: Iterable[Iterable[float]] | Iterable[Iterable[Iterable[float]]] | None,
    *,
    directional_basis_labels: Iterable[str] | None = None,
    directional_basis_sha256: str | None = None,
    directional_basis_content_sha256: str | None = None,
    directional_basis_registry_id: str | None = None,
    directional_basis_registry_manifest: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None, str | None, dict[str, str] | None]:
    masses = np.asarray(population_masses, dtype=np.float64)
    if masses.ndim != 2 or masses.shape[1] != atlas.source_count or masses.shape[0] == 0:
        raise LikelihoodInputError("population_masses must have shape (P, M)")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
        raise LikelihoodInputError("population_masses must be finite and non-negative")
    if directional_mass_basis is None:
        if any(value is not None for value in (directional_basis_labels, directional_basis_sha256, directional_basis_content_sha256, directional_basis_registry_id, directional_basis_registry_manifest)):
            raise LikelihoodInputError("basis metadata supplied without directional_mass_basis")
        return masses, None, None, None
    basis = np.asarray(directional_mass_basis, dtype=np.float64)
    if not np.all(np.isfinite(basis)):
        raise LikelihoodInputError("directional_mass_basis must be finite")
    if directional_basis_labels is None or directional_basis_sha256 is None:
        raise LikelihoodInputError("directional basis semantic labels and their SHA256 are required")
    labels = tuple(str(item) for item in directional_basis_labels)
    if labels != FROZEN_DIRECTION_NAMES:
        raise LikelihoodInputError("directional basis labels do not match the frozen 23-direction contract")
    labels_digest = hashlib.sha256(json.dumps(labels, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    if str(directional_basis_sha256) != labels_digest:
        raise LikelihoodInputError("directional basis semantic-label SHA256 mismatch")
    if directional_basis_content_sha256 is None or directional_basis_registry_id is None or not str(directional_basis_registry_id).strip():
        raise LikelihoodInputError("a registered numeric directional-basis digest and registry id are required")
    if directional_basis_registry_manifest is None:
        raise LikelihoodInputError("repository-owned directional-basis registry manifest is required")
    canonical_basis_digest = hashlib.sha256(np.asarray(basis, dtype="<f8", order="C").tobytes()).hexdigest()
    if str(directional_basis_content_sha256) != canonical_basis_digest:
        raise LikelihoodInputError("directional basis numeric-content SHA256 mismatch")
    manifest = _read_registered_manifest(directional_basis_registry_manifest, label="directional basis")
    if str(manifest.get("registry_id", "")) != str(directional_basis_registry_id):
        raise LikelihoodInputError("directional basis registry id does not match manifest")
    allowed = tuple(str(item) for item in manifest.get("allowed_content_sha256", ()))
    if canonical_basis_digest not in allowed:
        raise LikelihoodInputError("directional basis content digest is not registered in the manifest")
    if not str(manifest.get("units", "")).strip() or not str(manifest.get("coordinate_convention", "")).strip():
        raise LikelihoodInputError("directional basis manifest must declare units and coordinate convention")
    metadata = {
        "direction_names_sha256": labels_digest,
        "numeric_basis_sha256": canonical_basis_digest,
        "numeric_basis_registry_id": str(directional_basis_registry_id),
        "numeric_basis_registry_status": str(manifest.get("status")),
    }
    if basis.ndim == 2 and basis.shape[1] == atlas.source_count:
        if basis.shape[0] != Q7_DERIVATIVE_DIRECTIONS:
            raise LikelihoodInputError("directional_mass_basis must have 23 directions")
        return masses, basis, "direction_source", metadata
    if basis.ndim == 3 and basis.shape[:2] == masses.shape and basis.shape[2] == Q7_DERIVATIVE_DIRECTIONS:
        return masses, basis, "population_direction_source", metadata
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
    directional_basis_labels: Iterable[str] | None = None,
    directional_basis_sha256: str | None = None,
    directional_basis_content_sha256: str | None = None,
    directional_basis_registry_id: str | None = None,
    directional_basis_registry_manifest: str | Path | None = None,
    allow_uncertified_finite_enclosure: bool = False,
    oracle_fields: Iterable[Iterable[Iterable[Iterable[float]]]] | None = None,
    summary_operator_l1_norms: Iterable[Iterable[float]] | None = None,
    summary_operator_l1_norms_sha256: str | None = None,
    summary_operator_registry_id: str | None = None,
    summary_operator_registry_manifest: str | Path | None = None,
) -> ApproximationResult:
    """Evaluate the route and return certified, finite and optional measured budgets."""

    if allow_uncertified_finite_enclosure:
        raise LikelihoodInputError(
            "finite-source enclosures cannot be promoted; use sourcewise Q1 fallback"
        )
    if summary_operator_l1_norms is None and any(
        value is not None for value in (summary_operator_l1_norms_sha256, summary_operator_registry_id, summary_operator_registry_manifest)
    ):
        raise LikelihoodInputError("summary-operator metadata supplied without summary norms")
    masses, basis, basis_mode, _basis_metadata = _validate_mass_inputs(
        atlas,
        population_masses,
        directional_mass_basis,
        directional_basis_labels=directional_basis_labels,
        directional_basis_sha256=directional_basis_sha256,
        directional_basis_content_sha256=directional_basis_content_sha256,
        directional_basis_registry_id=directional_basis_registry_id,
        directional_basis_registry_manifest=directional_basis_registry_manifest,
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
    certified_bin_l1 = np.zeros(atlas.bin_count, dtype=np.float64)
    for bin_index, topology_bin in enumerate(atlas.bins):
        members = topology_bin.members
        certified = topology_bin.continuous_certified
        approximate = certified
        if approximate:
            rep = atlas.representative_fields[bin_index]
            grouped_mass = np.sum(masses[:, members], axis=1)
            fields += grouped_mass[(slice(None),) + (None,) * 3] * rep[None, ...]
            spread_l1 = float(atlas.finite_spread_l1[bin_index])
            spread_linf = float(atlas.finite_spread_linf[bin_index])
            certified_spread_l1 = float(atlas.certified_spread_l1[bin_index])
            certified_spread_linf = float(atlas.certified_spread_linf[bin_index])
            finite_value_l1 += np.sum(masses[:, members], axis=1) * spread_l1
            finite_value_linf += np.sum(masses[:, members], axis=1) * spread_linf
            if certified:
                cert_value_l1 += np.sum(masses[:, members], axis=1) * certified_spread_l1
                cert_value_linf += np.sum(masses[:, members], axis=1) * certified_spread_linf
                certified_bin_l1[bin_index] = float(np.sum(masses[:, members]) * certified_spread_l1)
            if basis_mode == "direction_source":
                grouped_basis = np.sum(basis[:, members], axis=1)
                gradients += grouped_basis[(slice(None),) + (None,) * 3] * rep[None, ...]
                finite_grad_l1 += np.sum(np.abs(basis[:, members]), axis=1) * spread_l1
                if certified:
                    cert_grad_l1 += np.sum(np.abs(basis[:, members]), axis=1) * certified_spread_l1
            elif basis_mode == "population_direction_source":
                grouped_basis = np.sum(basis[:, members, :], axis=1)
                gradients += grouped_basis[(slice(None), slice(None)) + (None,) * 3] * rep[None, None, ...]
                finite_grad_l1 += np.sum(np.abs(basis[:, members, :]), axis=(0, 1)) * spread_l1
                if certified:
                    cert_grad_l1 += np.sum(np.abs(basis[:, members, :]), axis=(0, 1)) * certified_spread_l1
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
    summary_bounds = None
    if summary_operator_l1_norms is not None:
        if overflow:
            raise LikelihoodInputError("summary bounds require every source to be covered by a continuous certificate")
        summary_bounds = induced_summary_l1_bounds(
            certified_bin_l1,
            summary_operator_l1_norms,
            summary_operator_l1_norms_sha256=summary_operator_l1_norms_sha256,
            summary_operator_registry_id=summary_operator_registry_id,
            summary_operator_registry_manifest=summary_operator_registry_manifest,
        )
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
        certified_summary_l1_bounds=summary_bounds,
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
            "exact zero-scale deposit payload hash",
        ],
        "numeric_knot_coefficients_in_key": False,
        "tail_cutoff_sigma": atlas.q1_tail_cutoff,
        "frozen_direction_count": Q7_DERIVATIVE_DIRECTIONS,
        "frozen_direction_names": list(FROZEN_DIRECTION_NAMES),
        "directional_basis_provenance": "repository-owned registry manifest with units, convention and consumed-array digest",
        "summary_bound_count": Q7_SUMMARY_COUNT,
        "summary_bound_api": "induced_summary_l1_bounds",
        "summary_operator_provenance": "registered numeric digest and registry id required; accumulation rounded upward",
        "certificate_method": "outward_lipschitz_interval",
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

    masses, _, _, _ = _validate_mass_inputs(atlas, population_masses, None)
    result = np.zeros((masses.shape[0],) + atlas.representative_fields.shape[1:], dtype=np.float64)
    for source in range(atlas.source_count):
        result += masses[:, source, None, None, None] * _source_field(atlas, source)[None, ...]
    return result
