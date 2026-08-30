"""Deterministic ROI-window leakage calibration for the frozen CF4 k design.

This module consumes no truth or candidate field.  It computes integer-lattice
mode counts and window-only shell mixing, then publishes a hash-bound artifact
directory only for a numerically passing v2 precheck.  This precheck makes no
scientific leakage decision; invalid inputs, numerics, or provenance publish
neither an artifact directory nor COMPLETE.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.signal import fftconvolve


FROZEN_DESIGN_SHA256 = (
    "76b71a482a1d92b146e335e231c5b4430f06df009566f22ce1efb739c5c96da9"
)
ROUNDING_ABS_TOLERANCE = 1.0e-5
CONVERGENCE_ABS_TOLERANCE = 5.0e-4
COARSE_NUMERICS = {
    "moment_order": 96,
    "shell_order": 6,
    "q_period_samples": 16,
    "q_panel_order": 4,
    "parseval_tail_x": (384.0, 768.0),
}
FINE_NUMERICS = {
    "moment_order": 160,
    "shell_order": 10,
    "q_period_samples": 32,
    "q_panel_order": 8,
    "parseval_tail_x": (512.0, 1024.0),
}


class LeakageError(ValueError):
    """A fail-closed contract or numerical-integrity violation."""


@dataclass(frozen=True)
class NativeBin:
    index: int
    lower: float
    upper: float
    representative: float
    terminal: bool


@dataclass(frozen=True)
class WindowSpec:
    key: str
    radius: float
    centers: tuple[tuple[float, float, float], ...]
    box_size: float

    @property
    def multiplicity(self) -> int:
        return max(1, len(self.centers))

    @property
    def is_union(self) -> bool:
        return len(self.centers) > 1


@dataclass(frozen=True)
class MixingEvaluation:
    matrix: np.ndarray
    containment: np.ndarray
    column_sum: np.ndarray
    signed_normalization_residual: np.ndarray
    localized_neff: np.ndarray
    normalization_valid: np.ndarray
    raw_supported: np.ndarray
    suffix: dict[str, object]
    moments: dict[str, float]
    numerical_audit: dict[str, object]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Return timestamp-free, key-sorted NPZ bytes."""

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                raise LeakageError(f"invalid deterministic NPZ key {name!r}")
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise LeakageError("object arrays are forbidden in mixing artifact")
            payload = io.BytesIO()
            np.lib.format.write_array(payload, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100444 << 16
            archive.writestr(info, payload.getvalue())
    return output.getvalue()


def load_frozen_design(path: str | Path) -> tuple[dict[str, object], str]:
    design_path = Path(path)
    raw = design_path.read_bytes()
    digest = _sha256(raw)
    if digest != FROZEN_DESIGN_SHA256:
        raise LeakageError(
            f"frozen design SHA256 mismatch: {digest} != {FROZEN_DESIGN_SHA256}"
        )
    try:
        design = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LeakageError("cannot parse frozen design JSON") from exc
    if design.get("schema") != "ouruniv-cf4-kf-bin-manifest-design-v1":
        raise LeakageError("unexpected frozen design schema")
    if design.get("status") != (
        "user_approved_design_frozen_ROI_leakage_and_manifest_materialization_pending"
    ):
        raise LeakageError("frozen design status is not approved/pending")

    source = design["ROI_geometry"]["source"]
    repository_root = design_path.resolve().parents[1]
    source_path = repository_root / source["path"]
    if _sha256(source_path.read_bytes()) != source["raw_sha256"]:
        raise LeakageError("frozen ROI source SHA256 mismatch")
    return design, digest


def build_native_bins(design: Mapping[str, object]) -> list[NativeBin]:
    lattice = design["analysis_lattice"]
    specification = design["native_bin_design"]
    box_size = float(lattice["box_size_cMpc_h"])
    spacing = float(lattice["grid_spacing_cMpc_h"])
    grid_size = int(lattice["cells_per_axis_N"])
    if grid_size * spacing != box_size:
        raise LeakageError("frozen lattice does not satisfy N*dx=L exactly")
    fundamental = 2.0 * math.pi / box_size
    nyquist = math.pi / spacing
    if fundamental != float(lattice["fundamental_h_Mpc"]):
        raise LeakageError("frozen fundamental value mismatch")
    if nyquist != float(lattice["isotropic_analysis_Nyquist_h_Mpc"]):
        raise LeakageError("frozen Nyquist value mismatch")

    ratio = 2.0 ** 0.25
    edges = [fundamental]
    while edges[-1] * ratio <= nyquist:
        edges.append(edges[-1] * ratio)
    if edges[-1] != nyquist:
        edges.append(nyquist)
    bins = [
        NativeBin(
            index=index,
            lower=lower,
            upper=upper,
            representative=math.sqrt(lower * upper),
            terminal=index == len(edges) - 2,
        )
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]))
    ]
    expected = int(specification["complete_native_bin_count_for_frozen_lattice"])
    expected += int(specification["terminal_truncated_bin_count_for_frozen_lattice"])
    if len(bins) != expected:
        raise LeakageError(f"native bin count {len(bins)} != frozen {expected}")
    if bins[-1].upper != nyquist:
        raise LeakageError("terminal native bin does not end exactly at k_Ny")
    return bins


def squared_frequency_degeneracy(grid_size: int) -> np.ndarray:
    if grid_size <= 0 or grid_size % 2:
        raise LeakageError("grid_size must be a positive even integer")
    half = grid_size // 2
    frequencies = np.arange(-half, half, dtype=np.int64)
    return np.bincount(frequencies * frequencies, minlength=half * half + 1)


def squared_radius_histogram_fft(
    grid_size: int, rounding_tolerance: float = ROUNDING_ABS_TOLERANCE
) -> tuple[np.ndarray, dict[str, object]]:
    """Count full 3-D DFT vectors by integer squared radius using two FFT convs."""

    one_dimensional = squared_frequency_degeneracy(grid_size).astype(np.float64)
    convolution_2_float = fftconvolve(one_dimensional, one_dimensional, mode="full")
    convolution_2_round = np.rint(convolution_2_float)
    error_2 = float(np.max(np.abs(convolution_2_float - convolution_2_round)))
    if error_2 > rounding_tolerance:
        raise LeakageError(f"2-D convolution rounding error {error_2} exceeds tolerance")

    convolution_3_float = fftconvolve(
        convolution_2_round, one_dimensional, mode="full"
    )
    convolution_3_round = np.rint(convolution_3_float)
    error_3 = float(np.max(np.abs(convolution_3_float - convolution_3_round)))
    if error_3 > rounding_tolerance:
        raise LeakageError(f"3-D convolution rounding error {error_3} exceeds tolerance")
    if np.any(convolution_3_round < 0):
        raise LeakageError("rounded 3-D convolution contains negative counts")
    counts = convolution_3_round.astype(np.int64)
    total = int(counts.sum(dtype=np.int64))
    expected_total = grid_size**3
    if total != expected_total:
        raise LeakageError(f"3-D convolution total {total} != N^3 {expected_total}")
    return counts, {
        "method": "1D signed-DFT squared-radius degeneracy and two scipy.signal.fftconvolve operations",
        "rounding_abs_tolerance": rounding_tolerance,
        "convolution_2_max_abs_rounding_error": error_2,
        "convolution_3_max_abs_rounding_error": error_3,
        "max_abs_rounding_error": max(error_2, error_3),
        "full_lattice_count": total,
        "expected_N_cubed": expected_total,
        "total_count_assertion_pass": True,
    }


def self_conjugate_squared_radius_histogram(grid_size: int) -> dict[int, int]:
    if grid_size <= 0 or grid_size % 2:
        raise LeakageError("grid_size must be a positive even integer")
    half = grid_size // 2
    histogram: dict[int, int] = {}
    for vector in itertools.product((0, -half), repeat=3):
        squared_radius = sum(component * component for component in vector)
        if squared_radius == 0 or squared_radius > half * half:
            continue
        histogram[squared_radius] = histogram.get(squared_radius, 0) + 1
    return histogram


def _q_at_or_above_quarter_octave_edge(
    squared_radius: int, edge_index: int
) -> bool:
    """Compare integer q to 2^(edge_index/2) without floating arithmetic."""

    if edge_index % 2 == 0:
        return squared_radius >= 1 << (edge_index // 2)
    return squared_radius * squared_radius >= 1 << edge_index


def _bin_for_squared_radius(squared_radius: int, bin_count: int) -> int:
    if squared_radius < 1 or bin_count < 1:
        raise LeakageError("squared radius and bin count must be positive")
    for index in range(bin_count - 1):
        if not _q_at_or_above_quarter_octave_edge(squared_radius, index + 1):
            return index
    return bin_count - 1


def mode_counts_by_native_bin(
    grid_size: int,
    bins: Sequence[NativeBin],
    fundamental: float,
    rounding_tolerance: float = ROUNDING_ABS_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    radius_histogram, convolution_audit = squared_radius_histogram_fft(
        grid_size, rounding_tolerance
    )
    half = grid_size // 2
    maximum_q = half * half
    if fundamental <= 0 or not math.isfinite(fundamental):
        raise LeakageError("fundamental must be finite and positive")
    full_counts = np.zeros(len(bins), dtype=np.int64)
    for squared_radius, count in enumerate(radius_histogram[: maximum_q + 1]):
        if squared_radius == 0 or count == 0:
            continue
        index = _bin_for_squared_radius(squared_radius, len(bins))
        full_counts[index] += int(count)

    self_counts = np.zeros(len(bins), dtype=np.int64)
    for squared_radius, count in self_conjugate_squared_radius_histogram(
        grid_size
    ).items():
        index = _bin_for_squared_radius(squared_radius, len(bins))
        self_counts[index] += count
    if np.any((full_counts - self_counts) % 2):
        raise LeakageError("non-self full-vector counts are not conjugate-pair even")
    independent_counts = (full_counts + self_counts) // 2
    audit = {
        **convolution_audit,
        "radial_domain_q_max_inclusive": maximum_q,
        "DC_excluded": True,
        "full_vectors_in_isotropic_analysis_sphere": int(full_counts.sum()),
        "self_conjugate_vectors_in_analysis_sphere": int(self_counts.sum()),
        "independent_real_modes_in_analysis_sphere": int(
            independent_counts.sum()
        ),
        "conjugate_formula": "independent=(full_vectors+self_conjugate)/2",
    }
    return full_counts, independent_counts, audit


def greedy_merge_bins(
    bins: Sequence[NativeBin], independent_counts: Sequence[int], minimum: int = 32
) -> list[dict[str, object]]:
    if len(bins) != len(independent_counts) or not bins:
        raise LeakageError("bins/counts must be nonempty and have equal length")
    merged: list[dict[str, object]] = []
    members: list[int] = []
    total = 0
    for item, count in zip(bins, independent_counts):
        members.append(item.index)
        total += int(count)
        if total >= minimum:
            merged.append(
                {
                    "native_bin_indices": members,
                    "independent_real_mode_count": total,
                }
            )
            members = []
            total = 0
    if members:
        if not merged:
            raise LeakageError("terminal underfill has no preceding merged bin")
        merged[-1]["native_bin_indices"].extend(members)
        merged[-1]["independent_real_mode_count"] += total
        merged[-1]["terminal_underfill_absorbed"] = True
    for index, item in enumerate(merged):
        item["merged_bin_index"] = index
    return merged


def _gauss_interval(order: int, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    if order < 2 or not (math.isfinite(lower) and math.isfinite(upper)) or upper <= lower:
        raise LeakageError("invalid Gauss-Legendre interval/order")
    nodes, weights = leggauss(order)
    scale = 0.5 * (upper - lower)
    return lower + scale * (nodes + 1.0), scale * weights


def raised_cosine_window(radius: np.ndarray | float, outer_radius: float) -> np.ndarray:
    values = np.asarray(radius, dtype=np.float64)
    result = np.zeros_like(values)
    core = values <= 0.75 * outer_radius
    taper = (values > 0.75 * outer_radius) & (values <= outer_radius)
    result[core] = 1.0
    result[taper] = 0.5 * (
        1.0
        + np.cos(
            math.pi
            * (values[taper] - 0.75 * outer_radius)
            / (0.25 * outer_radius)
        )
    )
    return result


def single_sphere_window_moment(
    outer_radius: float, power: int, radial_order: int
) -> float:
    pieces = ((0.0, 0.75 * outer_radius), (0.75 * outer_radius, outer_radius))
    total = 0.0
    for lower, upper in pieces:
        radius, weights = _gauss_interval(radial_order, lower, upper)
        window = raised_cosine_window(radius, outer_radius)
        total += float(np.sum(weights * radius * radius * window**power))
    return 4.0 * math.pi * total


def _dimensionless_cos_moment(power: int, lower: float, upper: float) -> float:
    """Return integral_lower^upper t^power cos(4*pi*t) dt analytically."""

    beta = 4.0 * math.pi
    cosine: list[float] = []
    sine: list[float] = []
    for degree in range(power + 1):
        boundary_cosine = (
            upper**degree * math.sin(beta * upper)
            - lower**degree * math.sin(beta * lower)
        ) / beta
        boundary_sine = (
            -upper**degree * math.cos(beta * upper)
            + lower**degree * math.cos(beta * lower)
        ) / beta
        cosine.append(
            boundary_cosine
            if degree == 0
            else boundary_cosine - degree * sine[degree - 1] / beta
        )
        sine.append(
            boundary_sine
            if degree == 0
            else boundary_sine + degree * cosine[degree - 1] / beta
        )
    return cosine[power]


def _dimensionless_window_moment(power: int) -> float:
    core_edge = 0.75
    polynomial = core_edge ** (power + 1) / (power + 1)
    taper_polynomial = (1.0 - core_edge ** (power + 1)) / (power + 1)
    return (
        polynomial
        + 0.5 * taper_polynomial
        - 0.5 * _dimensionless_cos_moment(power, core_edge, 1.0)
    )


def _integral_t_sin(frequency: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Stable analytic integral of t*sin(frequency*t) on [lower, upper]."""

    values = np.asarray(frequency, dtype=np.float64)
    result = np.empty_like(values)
    small = np.abs(values) < 1.0e-3
    if np.any(small):
        c = values[small]
        series = np.zeros_like(c)
        for index in range(8):
            degree = 2 * index + 1
            coefficient = (-1.0) ** index / math.factorial(degree)
            moment = (upper ** (degree + 2) - lower ** (degree + 2)) / (
                degree + 2
            )
            series += coefficient * c**degree * moment
        result[small] = series
    if np.any(~small):
        c = values[~small]

        def primitive(boundary: float) -> np.ndarray:
            argument = c * boundary
            return (np.sin(argument) - argument * np.cos(argument)) / (c * c)

        result[~small] = primitive(upper) - primitive(lower)
    return result


def sphere_radial_transform(
    wavenumber: np.ndarray | float,
    outer_radius: float,
) -> np.ndarray:
    """Analytic 4*pi*int r^2 W(r) j0(qr) dr with stable singular limits."""

    q = np.asarray(wavenumber, dtype=np.float64)
    if np.any(~np.isfinite(q)) or np.any(q < 0.0) or outer_radius <= 0.0:
        raise LeakageError("sphere transform requires finite q>=0 and R>0")
    x = (q * outer_radius).reshape(-1)
    transformed = np.empty_like(x)
    small = np.abs(x) < 5.0e-2
    if np.any(small):
        xs = x[small]
        series = np.zeros_like(xs)
        for index in range(9):
            coefficient = (-1.0) ** index / math.factorial(2 * index + 1)
            series += (
                coefficient
                * xs ** (2 * index)
                * _dimensionless_window_moment(2 * index + 2)
            )
        transformed[small] = 4.0 * math.pi * outer_radius**3 * series
    if np.any(~small):
        xs = x[~small]
        beta = 4.0 * math.pi
        numerator = (
            _integral_t_sin(xs, 0.0, 0.75)
            + 0.5 * _integral_t_sin(xs, 0.75, 1.0)
            - 0.25 * _integral_t_sin(xs + beta, 0.75, 1.0)
            - 0.25 * _integral_t_sin(xs - beta, 0.75, 1.0)
        )
        transformed[~small] = 4.0 * math.pi * outer_radius**3 * numerator / xs
    return transformed.reshape(q.shape)


def _minimum_image_distance(
    first: Sequence[float], second: Sequence[float], box_size: float
) -> float:
    displacement = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    displacement -= box_size * np.rint(displacement / box_size)
    return float(np.linalg.norm(displacement))


def verify_disjoint_union(specification: WindowSpec) -> tuple[float, ...]:
    if not specification.is_union:
        return ()
    distances = tuple(
        _minimum_image_distance(first, second, specification.box_size)
        for first, second in itertools.combinations(specification.centers, 2)
    )
    if any(distance <= 2.0 * specification.radius for distance in distances):
        raise LeakageError("union component spheres are not strictly disjoint")
    return distances


def orientation_averaged_structure_factor(
    wavenumber: np.ndarray | float, specification: WindowSpec
) -> np.ndarray:
    q = np.asarray(wavenumber, dtype=np.float64)
    if not specification.is_union:
        return np.ones_like(q)
    distances = verify_disjoint_union(specification)
    factor = np.full_like(q, float(specification.multiplicity))
    for distance in distances:
        factor += 2.0 * np.sinc(q * distance / math.pi)
    if np.any(factor < -1.0e-10):
        raise LeakageError("orientation-averaged structure factor became negative")
    return factor


def _window_moments(specification: WindowSpec, radial_order: int) -> dict[str, float]:
    multiplicity = specification.multiplicity
    moments = {
        f"int_W{power}_dV": multiplicity
        * single_sphere_window_moment(specification.radius, power, radial_order)
        for power in (1, 2, 4)
    }
    moments["V_eff"] = moments["int_W2_dV"] ** 2 / moments["int_W4_dV"]
    return moments


def _window_frequency_scale(specification: WindowSpec) -> float:
    distances = verify_disjoint_union(specification)
    # A(q)^2 carries frequencies through 2R.  Multiplication by the union
    # structure factor adds each pair-distance frequency, so 2R+max(d_ij)
    # is the conservative shortest-period scale.
    return 2.0 * specification.radius + (max(distances) if distances else 0.0)


def _window_power(q: np.ndarray, specification: WindowSpec) -> np.ndarray:
    amplitude = sphere_radial_transform(q, specification.radius)
    return amplitude * amplitude * orientation_averaged_structure_factor(
        q, specification
    )


class QPowerCumulative:
    """Deterministic composite-GL cumulative of q |W-tilde(q)|^2.

    Uniform panels are fixed by the shortest declared window/structure-factor
    period.  Arbitrary endpoints use the same fixed GL rule on only the final
    fractional panel, so the angular integral is evaluated in q, not mu.
    """

    def __init__(
        self,
        specification: WindowSpec,
        q_max: float,
        *,
        period_samples: int,
        panel_order: int,
    ) -> None:
        if q_max <= 0.0 or period_samples < 4 or panel_order < 2:
            raise LeakageError("invalid q-cumulative numerical contract")
        frequency_scale = _window_frequency_scale(specification)
        maximum_width = 2.0 * math.pi / (frequency_scale * period_samples)
        self.panel_count = max(1, math.ceil(q_max / maximum_width))
        self.panel_width = q_max / self.panel_count
        self.q_max = float(q_max)
        self.specification = specification
        self.nodes, self.weights = leggauss(panel_order)
        lower = self.panel_width * np.arange(self.panel_count, dtype=np.float64)
        midpoint = lower + 0.5 * self.panel_width
        q = midpoint[:, None] + 0.5 * self.panel_width * self.nodes[None, :]
        values = q * _window_power(q, specification)
        panel_integrals = 0.5 * self.panel_width * np.sum(
            self.weights[None, :] * values, axis=1
        )
        self.cumulative = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(panel_integrals))
        )
        if not np.all(np.isfinite(self.cumulative)):
            raise LeakageError("nonfinite q-cumulative integral")

    def __call__(self, q: np.ndarray | float) -> np.ndarray:
        values = np.asarray(q, dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise LeakageError("q-cumulative endpoints must be finite and nonnegative")
        if np.any(values > self.q_max * (1.0 + 8.0 * np.finfo(float).eps)):
            raise LeakageError("q-cumulative endpoint exceeds frozen q_max")
        flat = np.minimum(values.reshape(-1), self.q_max)
        indices = np.minimum(
            np.floor(flat / self.panel_width).astype(np.int64), self.panel_count - 1
        )
        lower = indices * self.panel_width
        widths = flat - lower
        q_nodes = lower[:, None] + 0.5 * widths[:, None] * (
            self.nodes[None, :] + 1.0
        )
        partial = 0.5 * widths * np.sum(
            self.weights[None, :]
            * q_nodes
            * _window_power(q_nodes, self.specification),
            axis=1,
        )
        result = self.cumulative[indices] + partial
        at_upper = flat == self.q_max
        result[at_upper] = self.cumulative[-1]
        return result.reshape(values.shape)

    def angular_integral(
        self, output_k: np.ndarray, input_k: np.ndarray
    ) -> np.ndarray:
        ko = np.asarray(output_k, dtype=np.float64)
        ki = np.asarray(input_k, dtype=np.float64)
        if np.any(ko <= 0.0) or np.any(ki <= 0.0):
            raise LeakageError("shell nodes must be positive")
        upper = ko + ki
        lower = np.abs(ko - ki)
        return (self(upper) - self(lower)) / (ko * ki)


def _composite_q_power_integral(
    specification: WindowSpec,
    upper: float,
    *,
    period_samples: int,
    panel_order: int,
) -> float:
    """Compute integral_0^upper q^2 |W-tilde(q)|^2 dq."""

    frequency_scale = _window_frequency_scale(specification)
    maximum_width = 2.0 * math.pi / (frequency_scale * period_samples)
    panel_count = max(1, math.ceil(upper / maximum_width))
    width = upper / panel_count
    nodes, weights = leggauss(panel_order)
    lower = width * np.arange(panel_count, dtype=np.float64)
    q = lower[:, None] + 0.5 * width * (nodes[None, :] + 1.0)
    return float(
        0.5
        * width
        * np.sum(weights[None, :] * q * q * _window_power(q, specification))
    )


def parseval_audit(
    specification: WindowSpec,
    moments: Mapping[str, float],
    numerics: Mapping[str, object],
) -> dict[str, object]:
    """Independent real/q-space Parseval comparison with a finite-tail audit."""

    lower_x, upper_x = (float(value) for value in numerics["parseval_tail_x"])
    if not (0.0 < lower_x < upper_x):
        raise LeakageError("invalid frozen Parseval tail bounds")
    values = []
    for cutoff_x in (lower_x, upper_x):
        q_integral = _composite_q_power_integral(
            specification,
            cutoff_x / specification.radius,
            period_samples=int(numerics["q_period_samples"]),
            panel_order=int(numerics["q_panel_order"]),
        )
        values.append(q_integral / (2.0 * math.pi**2))
    real_value = float(moments["int_W2_dV"])
    relative_error = abs(values[1] - real_value) / real_value
    tail_increment = abs(values[1] - values[0]) / real_value
    return {
        "identity": "int_W2_dV=(1/(2*pi^2))*int_0^infinity q^2*|A(q)|^2*S(q)dq",
        "real_space_int_W2_dV": real_value,
        "q_space_lower_cutoff_x": lower_x,
        "q_space_upper_cutoff_x": upper_x,
        "q_space_lower_value": values[0],
        "q_space_upper_value": values[1],
        "relative_parseval_error": relative_error,
        "relative_finite_tail_increment": tail_increment,
        "pass": relative_error <= CONVERGENCE_ABS_TOLERANCE
        and tail_increment <= CONVERGENCE_ABS_TOLERANCE,
    }


def compute_mixing_matrix(
    bins: Sequence[NativeBin],
    specification: WindowSpec,
    *,
    numerics: Mapping[str, object],
) -> tuple[np.ndarray, dict[str, float], dict[str, object]]:
    """Integrate M[out,in] using dmu=q*dq/(k*k')."""

    verify_disjoint_union(specification)
    moment_order = int(numerics["moment_order"])
    shell_order = int(numerics["shell_order"])
    moments = _window_moments(specification, moment_order)
    parseval_denominator = (2.0 * math.pi) ** 3 * moments["int_W2_dV"]
    q_cumulative = QPowerCumulative(
        specification,
        2.0 * max(item.upper for item in bins),
        period_samples=int(numerics["q_period_samples"]),
        panel_order=int(numerics["q_panel_order"]),
    )
    shell_nodes: list[np.ndarray] = []
    shell_weights: list[np.ndarray] = []
    shell_norms: list[float] = []
    for item in bins:
        nodes, weights = _gauss_interval(shell_order, item.lower, item.upper)
        shell_nodes.append(nodes)
        shell_weights.append(weights)
        shell_norms.append((item.upper**3 - item.lower**3) / 3.0)

    matrix = np.empty((len(bins), len(bins)), dtype=np.float64)
    for input_index, (input_k, input_weight, input_norm) in enumerate(
        zip(shell_nodes, shell_weights, shell_norms)
    ):
        ki = input_k[None, :]
        for output_index, (output_k, output_weight) in enumerate(
            zip(shell_nodes, shell_weights)
        ):
            ko = output_k[:, None]
            angular = q_cumulative.angular_integral(ko, ki)
            integrand_weights = (
                output_weight[:, None]
                * ko
                * ko
                * input_weight[None, :]
                * ki
                * ki
            )
            numerator = 2.0 * math.pi * float(
                np.sum(integrand_weights * angular)
            )
            matrix[output_index, input_index] = (
                numerator / input_norm / parseval_denominator
            )
    if not np.all(np.isfinite(matrix)) or np.any(matrix < -1.0e-12):
        raise LeakageError("mixing matrix contains nonfinite or materially negative values")
    transform_zero = float(sphere_radial_transform(np.array([0.0]), specification.radius)[0])
    single_moment_one = single_sphere_window_moment(
        specification.radius, 1, moment_order
    )
    column_sum = matrix.sum(axis=0)
    audit: dict[str, object] = {
        "analytic_radial_transform": True,
        "moment_order": moment_order,
        "shell_order": shell_order,
        "q_period_samples": int(numerics["q_period_samples"]),
        "q_panel_order": int(numerics["q_panel_order"]),
        "q_panel_count_for_mixing": q_cumulative.panel_count,
        "q_panel_width_h_Mpc": q_cumulative.panel_width,
        "sphere_A0_relative_error": abs(transform_zero - single_moment_one)
        / single_moment_one,
        "parseval_denominator": parseval_denominator,
        "minimum_analysis_column_sum": float(np.min(column_sum)),
        "maximum_analysis_column_sum": float(np.max(column_sum)),
        "maximum_analysis_column_overshoot": float(max(0.0, np.max(column_sum) - 1.0)),
        "parseval_q_space": parseval_audit(specification, moments, numerics),
    }
    return matrix, moments, audit


def supported_suffix(mask: Sequence[bool]) -> dict[str, object]:
    values = np.asarray(mask)
    if values.ndim != 1 or values.dtype != np.bool_ or len(values) == 0:
        raise LeakageError("supported suffix input must be nonempty exact booleans")
    passing = np.flatnonzero(values)
    if len(passing) == 0:
        return {
            "pass": False,
            "first_supported_bin": None,
            "failed_after_first_supported": [],
            "reason": "no_supported_native_bin",
        }
    first = int(passing[0])
    holes = [int(index) for index in np.flatnonzero(~values[first:]) + first]
    if holes:
        return {
            "pass": False,
            "first_supported_bin": None,
            "first_raw_supported_bin": first,
            "failed_after_first_supported": holes,
            "reason": "supported_suffix_hole_fail_closed",
        }
    return {
        "pass": True,
        "first_supported_bin": first,
        "failed_after_first_supported": [],
        "reason": "contiguous_supported_suffix_through_Nyquist",
    }


def evaluate_support(
    matrix: np.ndarray,
    independent_counts: Sequence[int],
    moments: Mapping[str, float],
    box_size: float,
    *,
    containment_minimum: float = 0.9,
    outside_maximum: float = 0.01,
    localized_neff_minimum: float = 32.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    matrix = np.asarray(matrix, dtype=float)
    counts = np.asarray(independent_counts, dtype=float)
    if matrix.shape != (len(counts), len(counts)):
        raise LeakageError("mixing matrix/count shape mismatch")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise LeakageError("invalid mixing matrix for support evaluation")
    containment = np.empty(len(counts), dtype=float)
    for index in range(len(counts)):
        lower = max(0, index - 1)
        upper = min(len(counts), index + 2)
        containment[index] = float(matrix[lower:upper, index].sum())
    column_sums = matrix.sum(axis=0)
    signed_residual = 1.0 - column_sums
    normalization_valid = signed_residual >= -CONVERGENCE_ABS_TOLERANCE
    effective_volume = float(moments["V_eff"])
    localized_neff = counts * effective_volume / box_size**3
    supported = (
        (containment >= containment_minimum)
        & (signed_residual >= 0.0)
        & (signed_residual <= outside_maximum)
        & (localized_neff >= localized_neff_minimum)
        & normalization_valid
    )
    return (
        containment,
        column_sums,
        signed_residual,
        localized_neff,
        normalization_valid,
        supported,
        supported_suffix(supported),
    )


def _window_specs(design: Mapping[str, object]) -> tuple[dict[str, WindowSpec], dict[str, str]]:
    geometry = design["ROI_geometry"]
    box_size = float(design["analysis_lattice"]["box_size_cMpc_h"])
    specifications: dict[str, WindowSpec] = {}
    semantic_to_key: dict[str, str] = {}
    for roi in geometry["ROIs"]:
        roi_id = roi["id"]
        if roi["geometry"] == "sphere":
            radius = float(roi["radius_cMpc_h"])
            key = f"sphere_R{radius:g}".replace(".", "p")
            specification = WindowSpec(key, radius, (), box_size)
        elif roi["geometry"] == "union_max_of_spheres":
            radius = float(roi["component_radius_cMpc_h"])
            centers = tuple(
                tuple(float(value) for value in item["center_cMpc_h"])
                for item in roi[
                    "component_centers_from_observer_plus_source_offsets_cMpc_h"
                ]
            )
            key = f"union_R{radius:g}_M{len(centers)}".replace(".", "p")
            specification = WindowSpec(key, radius, centers, box_size)
            verify_disjoint_union(specification)
        else:
            raise LeakageError(f"unknown ROI geometry for {roi_id}")
        if key in specifications and specifications[key] != specification:
            raise LeakageError(f"numeric window key collision for {key}")
        specifications[key] = specification
        semantic_to_key[roi_id] = key
    if semantic_to_key["Local_Group"] != semantic_to_key["observer_environment"]:
        raise LeakageError("Local Group and observer numeric windows must be shared")
    return specifications, semantic_to_key


def _evaluate_window(
    bins: Sequence[NativeBin],
    specification: WindowSpec,
    independent_counts: np.ndarray,
    numerics: Mapping[str, object],
) -> MixingEvaluation:
    matrix, moments, audit = compute_mixing_matrix(
        bins, specification, numerics=numerics
    )
    containment, column_sum, residual, neff, valid, supported, suffix = evaluate_support(
        matrix, independent_counts, moments, specification.box_size
    )
    return MixingEvaluation(
        matrix,
        containment,
        column_sum,
        residual,
        neff,
        valid,
        supported,
        suffix,
        moments,
        audit,
    )


def _roi_result(
    roi_id: str, key: str, evaluation: MixingEvaluation
) -> dict[str, object]:
    return {
        "ROI_id": roi_id,
        "numeric_product_key": key,
        "containment": evaluation.containment.tolist(),
        "analysis_column_sum": evaluation.column_sum.tolist(),
        "signed_outside_analysis_residual": (
            evaluation.signed_normalization_residual.tolist()
        ),
        "normalization_valid": evaluation.normalization_valid.tolist(),
        "localized_effective_independent_mode_count": (
            evaluation.localized_neff.tolist()
        ),
        "native_bin_supported": evaluation.raw_supported.tolist(),
        "supported_suffix": evaluation.suffix,
        "window_moments": evaluation.moments,
        "numerical_audit": evaluation.numerical_audit,
    }


def _mode_counts_document(
    design_sha: str,
    grid_size: int,
    bins: Sequence[NativeBin],
    full_counts: np.ndarray,
    independent_counts: np.ndarray,
    audit: Mapping[str, object],
) -> dict[str, object]:
    merged = greedy_merge_bins(bins, independent_counts, minimum=32)
    return {
        "schema": "ouruniv-cf4-kf-roi-leakage-mode-counts-v2",
        "design_raw_sha256": design_sha,
        "grid_size_N": grid_size,
        "native_bins": [
            {
                "index": item.index,
                "lower_h_Mpc": item.lower,
                "upper_h_Mpc": item.upper,
                "representative_h_Mpc": item.representative,
                "terminal_upper_inclusive": item.terminal,
                "full_vector_count": int(full_counts[item.index]),
                "independent_real_mode_count": int(independent_counts[item.index]),
            }
            for item in bins
        ],
        "merged_bins": merged,
        "count_audit": dict(audit),
    }


def load_execution_grant(
    path: str | Path, design_sha256: str
) -> tuple[dict[str, object], str]:
    payload = Path(path).read_bytes()
    digest = _sha256(payload)
    try:
        grant = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LeakageError("cannot parse v2 execution grant") from exc
    if grant.get("schema") != "ouruniv-cf4-kf-roi-leakage-execution-v2":
        raise LeakageError("unexpected v2 execution grant schema")
    if grant.get("status") != "user_approved_single_v2_preflight_only":
        raise LeakageError("v2 execution grant is not active")
    if grant.get("scope", {}).get("design_raw_sha256") != design_sha256:
        raise LeakageError("execution grant/design SHA256 mismatch")
    authorization = grant.get("authorization", {})
    required_true = {
        "v2_implementation_authorized",
        "Slurm_preflight_authorized",
    }
    if any(authorization.get(key) is not True for key in required_true):
        raise LeakageError("v2 implementation/preflight authority is missing")
    if authorization.get("maximum_preflight_submissions") != 1:
        raise LeakageError("v2 grant must authorize exactly one preflight")
    required_false = {
        "Slurm_production_authorized",
        "retry_authorized",
        "numeric_retuning_authorized",
        "replacement_run_authorized",
        "final_manifest_materialization_authorized",
        "KF_EXPAND_authorized",
        "all_D_mock_execution_authorized",
        "production_science_inference_authorized",
        "scientific_leakage_decision_authorized",
        "network_access_authorized",
    }
    if any(authorization.get(key) is not False for key in required_false):
        raise LeakageError("v2 grant contains forbidden downstream authority")
    numerical_contract = grant.get("numerical_contract", {})
    expected_coarse = json.loads(canonical_json_bytes(COARSE_NUMERICS))
    expected_fine = json.loads(canonical_json_bytes(FINE_NUMERICS))
    if numerical_contract.get("coarse") != expected_coarse:
        raise LeakageError("v2 grant frozen coarse numerics mismatch")
    if numerical_contract.get("fine") != expected_fine:
        raise LeakageError("v2 grant frozen fine numerics mismatch")
    return grant, digest


def _compare_evaluations(
    coarse: Mapping[str, MixingEvaluation], fine: Mapping[str, MixingEvaluation]
) -> dict[str, object]:
    containment_difference = 0.0
    residual_difference = 0.0
    moment_difference = 0.0
    classification_match = True
    suffix_match = True
    normalization_overshoot = 0.0
    transform_zero_error = 0.0
    parseval_error = 0.0
    parseval_tail = 0.0
    normalization_valid = True
    per_window: dict[str, object] = {}
    for key in sorted(fine):
        first = coarse[key]
        second = fine[key]
        containment_delta = float(
            np.max(np.abs(first.containment - second.containment))
        )
        residual_delta = float(
            np.max(
                np.abs(
                    first.signed_normalization_residual
                    - second.signed_normalization_residual
                )
            )
        )
        relative_moments = max(
            abs(first.moments[name] - second.moments[name]) / second.moments[name]
            for name in ("int_W1_dV", "int_W2_dV", "int_W4_dV", "V_eff")
        )
        same_classification = bool(
            np.array_equal(first.raw_supported, second.raw_supported)
        )
        same_suffix = first.suffix["pass"] == second.suffix["pass"]
        overshoot = float(max(
            first.numerical_audit["maximum_analysis_column_overshoot"],
            second.numerical_audit["maximum_analysis_column_overshoot"],
        ))
        a0_error = max(
            first.numerical_audit["sphere_A0_relative_error"],
            second.numerical_audit["sphere_A0_relative_error"],
        )
        containment_difference = max(containment_difference, containment_delta)
        first_parseval = first.numerical_audit["parseval_q_space"]
        second_parseval = second.numerical_audit["parseval_q_space"]
        window_parseval_error = float(
            max(
                first_parseval["relative_parseval_error"],
                second_parseval["relative_parseval_error"],
            )
        )
        window_parseval_tail = float(
            max(
                first_parseval["relative_finite_tail_increment"],
                second_parseval["relative_finite_tail_increment"],
            )
        )
        window_normalization_valid = bool(
            np.all(first.normalization_valid) and np.all(second.normalization_valid)
        )
        residual_difference = max(residual_difference, residual_delta)
        moment_difference = max(moment_difference, relative_moments)
        classification_match &= same_classification
        suffix_match &= same_suffix
        normalization_overshoot = max(normalization_overshoot, overshoot)
        transform_zero_error = max(transform_zero_error, a0_error)
        parseval_error = max(parseval_error, window_parseval_error)
        parseval_tail = max(parseval_tail, window_parseval_tail)
        normalization_valid &= window_normalization_valid
        per_window[key] = {
            "max_abs_containment_difference": containment_delta,
            "max_abs_signed_normalization_residual_difference": residual_delta,
            "max_relative_window_moment_difference": relative_moments,
            "native_classification_identical": same_classification,
            "suffix_classification_identical": same_suffix,
            "maximum_analysis_column_overshoot": overshoot,
            "coarse_failed_normalization_bins": np.flatnonzero(
                ~first.normalization_valid
            ).tolist(),
            "fine_failed_normalization_bins": np.flatnonzero(
                ~second.normalization_valid
            ).tolist(),
            "coarse_column_sum": first.column_sum.tolist(),
            "fine_column_sum": second.column_sum.tolist(),
            "coarse_signed_normalization_residual": (
                first.signed_normalization_residual.tolist()
            ),
            "fine_signed_normalization_residual": (
                second.signed_normalization_residual.tolist()
            ),
            "max_relative_parseval_error": window_parseval_error,
            "max_relative_parseval_tail_increment": window_parseval_tail,
            "coarse_sphere_A0_relative_error": first.numerical_audit[
                "sphere_A0_relative_error"
            ],
            "fine_sphere_A0_relative_error": second.numerical_audit[
                "sphere_A0_relative_error"
            ],
        }
    passed = (
        containment_difference <= CONVERGENCE_ABS_TOLERANCE
        and residual_difference <= CONVERGENCE_ABS_TOLERANCE
        and moment_difference <= CONVERGENCE_ABS_TOLERANCE
        and normalization_overshoot <= CONVERGENCE_ABS_TOLERANCE
        and transform_zero_error <= CONVERGENCE_ABS_TOLERANCE
        and parseval_error <= CONVERGENCE_ABS_TOLERANCE
        and parseval_tail <= CONVERGENCE_ABS_TOLERANCE
        and normalization_valid
        and classification_match
        and suffix_match
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "absolute_tolerance": CONVERGENCE_ABS_TOLERANCE,
        "max_abs_containment_difference": containment_difference,
        "max_abs_signed_normalization_residual_difference": residual_difference,
        "max_relative_parseval_denominator_or_window_moment_difference": (
            moment_difference
        ),
        "max_analysis_column_overshoot": normalization_overshoot,
        "max_sphere_A0_relative_error": transform_zero_error,
        "max_relative_parseval_error": parseval_error,
        "max_relative_parseval_tail_increment": parseval_tail,
        "all_analysis_column_normalizations_valid": normalization_valid,
        "native_classification_identical": classification_match,
        "suffix_classification_identical": suffix_match,
        "threshold_margin_safety_pass": classification_match and suffix_match,
        "per_numeric_window": per_window,
    }


def calculate(
    design: Mapping[str, object],
    design_sha: str,
    implementation_commit: str,
    *,
    execution_grant_sha256: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, np.ndarray]]:
    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise LeakageError("implementation commit must be lowercase 40-hex")
    if not re.fullmatch(r"[0-9a-f]{64}", execution_grant_sha256):
        raise LeakageError("execution grant SHA256 must be lowercase 64-hex")
    bins = build_native_bins(design)
    lattice = design["analysis_lattice"]
    grid_size = int(lattice["cells_per_axis_N"])
    fundamental = float(lattice["fundamental_h_Mpc"])
    full_counts, independent_counts, count_audit = mode_counts_by_native_bin(
        grid_size, bins, fundamental
    )
    mode_counts = _mode_counts_document(
        design_sha,
        grid_size,
        bins,
        full_counts,
        independent_counts,
        count_audit,
    )
    specifications, semantic_to_key = _window_specs(design)
    implementation_sha = _sha256(Path(__file__).read_bytes())
    arrays: dict[str, np.ndarray] = {}

    coarse = {
        key: _evaluate_window(bins, spec, independent_counts, COARSE_NUMERICS)
        for key, spec in specifications.items()
    }
    fine = {
        key: _evaluate_window(bins, spec, independent_counts, FINE_NUMERICS)
        for key, spec in specifications.items()
    }
    convergence = _compare_evaluations(coarse, fine)
    convergence["mode_count_convolution_rounding_max_abs_error"] = count_audit[
        "max_abs_rounding_error"
    ]
    if count_audit["max_abs_rounding_error"] > ROUNDING_ABS_TOLERANCE:
        convergence["status"] = "FAIL"
    evaluations = fine
    for key in sorted(specifications):
        arrays[f"coarse__{key}"] = coarse[key].matrix
        arrays[f"fine__{key}"] = fine[key].matrix
    status = "PRECHECK_PASS" if convergence["status"] == "PASS" else "PRECHECK_FAIL"

    roi_results = [
        _roi_result(roi_id, semantic_to_key[roi_id], evaluations[semantic_to_key[roi_id]])
        for roi_id in (
            "Local_Group",
            "Virgo",
            "Coma",
            "Local_Void",
            "Bootes_Void",
            "observer_environment",
        )
    ]
    leakage_gate_pass = all(item["supported_suffix"]["pass"] for item in roi_results)
    result = {
        "schema": "ouruniv-cf4-kf-roi-leakage-result-v2",
        "status": status,
        "mode": "preflight",
        "design_raw_sha256": design_sha,
        "execution_grant_raw_sha256": execution_grant_sha256,
        "implementation_path": "src/cf4_kf_roi_leakage.py",
        "implementation_sha256": implementation_sha,
        "implementation_commit": implementation_commit,
        "truth_or_candidate_data_consumed": False,
        "frozen_coarse_numerics": COARSE_NUMERICS,
        "frozen_fine_numerics": FINE_NUMERICS,
        "numerical_convergence": convergence,
        "ROI_results": roi_results,
        "Local_Group_observer_numeric_product_shared": True,
        "Local_Group_observer_semantic_results_separate": True,
        "Local_Group_observer_scores_summed": False,
        "overall_leakage_gate_pass": leakage_gate_pass,
        "scientific_disposition": "numerical_PRECHECK_only_no_scientific_leakage_decision_authorized",
        "scientific_leakage_decision_authorized": False,
        "final_manifest_materialized": False,
        "k_boundary_claim_created": False,
    }
    return result, mode_counts, arrays


def _write_and_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _inode_identity(path: Path) -> tuple[int, int]:
    metadata = path.stat(follow_symlinks=False)
    return metadata.st_dev, metadata.st_ino


def publish_artifacts(
    output_path: str | Path,
    result: Mapping[str, object],
    mode_counts: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> None:
    if result.get("status") != "PRECHECK_PASS":
        raise LeakageError("only a fully passing v2 precheck may be published")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing overwrite of existing output {output}")
    if not output.parent.is_dir():
        raise LeakageError("output parent directory must already exist")
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".staging", dir=output.parent
        )
    )
    stage_identity = _inode_identity(stage)
    published = False
    try:
        payloads = {
            "result.json": canonical_json_bytes(result),
            "mode_counts.json": canonical_json_bytes(mode_counts),
            "mixing_matrices.npz": deterministic_npz_bytes(arrays),
        }
        for filename, payload in payloads.items():
            _write_and_fsync(stage / filename, payload)
        manifest = {
            "schema": "ouruniv-cf4-kf-roi-leakage-artifact-manifest-v2",
            "status": "PRECHECK_PASS",
            "mode": result["mode"],
            "design_raw_sha256": result["design_raw_sha256"],
            "execution_grant_raw_sha256": result[
                "execution_grant_raw_sha256"
            ],
            "implementation_commit": result["implementation_commit"],
            "payloads": {
                filename: {"sha256": _sha256(payload), "bytes": len(payload)}
                for filename, payload in sorted(payloads.items())
            },
        }
        manifest_payload = canonical_json_bytes(manifest)
        _write_and_fsync(stage / "manifest.json", manifest_payload)
        complete = {
            "schema": "ouruniv-cf4-kf-roi-leakage-complete-v2",
            "status": manifest["status"],
            "mode": result["mode"],
            "manifest_sha256": _sha256(manifest_payload),
            "design_raw_sha256": result["design_raw_sha256"],
            "execution_grant_raw_sha256": result[
                "execution_grant_raw_sha256"
            ],
            "implementation_commit": result["implementation_commit"],
        }
        _write_and_fsync(stage / "COMPLETE", canonical_json_bytes(complete))
        directory_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output.exists():
            raise FileExistsError(f"refusing overwrite of raced output {output}")
        os.rename(stage, output)
        published = True
        if _inode_identity(output) != stage_identity:
            raise LeakageError("published output inode does not match staging inode")
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        elif published:
            try:
                if _inode_identity(output) == stage_identity:
                    shutil.rmtree(output)
            except FileNotFoundError:
                pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execution-grant", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("preflight",))
    parser.add_argument("--implementation-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"refusing overwrite of existing output {args.output}")
        design, design_sha = load_frozen_design(args.design)
        grant, grant_sha = load_execution_grant(args.execution_grant, design_sha)
        if str(args.output) != grant["scope"]["preflight_output"]:
            raise LeakageError("output does not equal the v2 grant preflight path")
        result, mode_counts, arrays = calculate(
            design,
            design_sha,
            args.implementation_commit,
            execution_grant_sha256=grant_sha,
        )
        if result["status"] != "PRECHECK_PASS":
            diagnostic = canonical_json_bytes(result).decode("utf-8").rstrip()
            print(diagnostic)
            print(diagnostic, file=sys.stderr)
            return 2
        publish_artifacts(args.output, result, mode_counts, arrays)
    except (OSError, LeakageError, json.JSONDecodeError) as exc:
        print(
            json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "mode": "preflight",
                "overall_leakage_gate_pass": result["overall_leakage_gate_pass"],
                "scientific_disposition": result["scientific_disposition"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
