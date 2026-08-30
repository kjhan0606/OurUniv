"""Deterministic ROI-window leakage calibration for the frozen CF4 k design.

This module consumes no truth or candidate field.  It computes integer-lattice
mode counts and window-only shell mixing, then publishes a hash-bound artifact
directory.  A scientifically valid leakage NO-GO is a successful calculation;
invalid inputs, numerics, or provenance are errors and never publish COMPLETE.
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
COARSE_ORDERS = {"radial": 96, "shell": 6, "mu": 12}
FINE_ORDERS = {"radial": 160, "shell": 10, "mu": 20}


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
    outside_analysis: np.ndarray
    localized_neff: np.ndarray
    raw_supported: np.ndarray
    suffix: dict[str, object]
    moments: dict[str, float]
    numerical_audit: dict[str, float]


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
    radius, weights = _gauss_interval(radial_order, 0.0, outer_radius)
    window = raised_cosine_window(radius, outer_radius)
    return float(4.0 * math.pi * np.sum(weights * radius * radius * window**power))


def sphere_radial_transform(
    wavenumber: np.ndarray | float, outer_radius: float, radial_order: int
) -> np.ndarray:
    q = np.asarray(wavenumber, dtype=np.float64)
    radius, weights = _gauss_interval(radial_order, 0.0, outer_radius)
    window = raised_cosine_window(radius, outer_radius)
    radial_weight = weights * radius * radius * window
    flattened = q.reshape(-1)
    transformed = np.empty_like(flattened)
    chunk_size = 8192
    for start in range(0, len(flattened), chunk_size):
        chunk = flattened[start : start + chunk_size]
        j0 = np.sinc(np.multiply.outer(chunk, radius) / math.pi)
        transformed[start : start + len(chunk)] = 4.0 * math.pi * (
            j0 @ radial_weight
        )
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
    return np.maximum(factor, 0.0)


def _window_moments(specification: WindowSpec, radial_order: int) -> dict[str, float]:
    multiplicity = specification.multiplicity
    moments = {
        f"int_W{power}_dV": multiplicity
        * single_sphere_window_moment(specification.radius, power, radial_order)
        for power in (1, 2, 4)
    }
    moments["V_eff"] = moments["int_W2_dV"] ** 2 / moments["int_W4_dV"]
    return moments


def compute_mixing_matrix(
    bins: Sequence[NativeBin],
    specification: WindowSpec,
    *,
    radial_order: int,
    shell_order: int,
    mu_order: int,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    """Integrate M[out,in] with k'^2 input-shell averaging."""

    verify_disjoint_union(specification)
    moments = _window_moments(specification, radial_order)
    parseval_denominator = (2.0 * math.pi) ** 3 * moments["int_W2_dV"]
    mu_nodes, mu_weights = leggauss(mu_order)
    shell_nodes = []
    shell_weights = []
    shell_norms = []
    for item in bins:
        nodes, weights = _gauss_interval(shell_order, item.lower, item.upper)
        shell_nodes.append(nodes)
        shell_weights.append(weights)
        shell_norms.append((item.upper**3 - item.lower**3) / 3.0)

    matrix = np.empty((len(bins), len(bins)), dtype=np.float64)
    for input_index, (input_k, input_weight, input_norm) in enumerate(
        zip(shell_nodes, shell_weights, shell_norms)
    ):
        ki = input_k[None, :, None]
        weighted_input = input_weight[None, :, None] * ki * ki
        for output_index, (output_k, output_weight) in enumerate(
            zip(shell_nodes, shell_weights)
        ):
            ko = output_k[:, None, None]
            q_squared = ko * ko + ki * ki - 2.0 * ko * ki * mu_nodes[None, None, :]
            q = np.sqrt(np.maximum(q_squared, 0.0))
            amplitude = sphere_radial_transform(
                q, specification.radius, radial_order
            )
            structure = orientation_averaged_structure_factor(q, specification)
            power = amplitude * amplitude * structure
            integrand_weights = (
                output_weight[:, None, None]
                * ko
                * ko
                * weighted_input
                * mu_weights[None, None, :]
            )
            numerator = 2.0 * math.pi * float(np.sum(integrand_weights * power))
            matrix[output_index, input_index] = (
                numerator / input_norm / parseval_denominator
            )
    if not np.all(np.isfinite(matrix)) or np.any(matrix < -1.0e-12):
        raise LeakageError("mixing matrix contains nonfinite or negative values")
    matrix = np.maximum(matrix, 0.0)
    transform_zero = float(
        sphere_radial_transform(np.array([0.0]), specification.radius, radial_order)[
            0
        ]
    )
    single_moment_one = single_sphere_window_moment(
        specification.radius, 1, radial_order
    )
    audit = {
        "radial_order": float(radial_order),
        "shell_order": float(shell_order),
        "mu_order": float(mu_order),
        "sphere_A0_relative_error": abs(transform_zero - single_moment_one)
        / single_moment_one,
        "parseval_denominator": parseval_denominator,
        "maximum_analysis_column_sum": float(np.max(matrix.sum(axis=0))),
        "maximum_analysis_column_overshoot": float(
            max(0.0, np.max(matrix.sum(axis=0)) - 1.0)
        ),
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
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
    if np.any(column_sums > 1.0 + CONVERGENCE_ABS_TOLERANCE):
        raise LeakageError("analysis-bin mixing probability exceeds unity tolerance")
    outside = np.maximum(0.0, 1.0 - column_sums)
    effective_volume = float(moments["V_eff"])
    localized_neff = counts * effective_volume / box_size**3
    supported = (
        (containment >= containment_minimum)
        & (outside <= outside_maximum)
        & (localized_neff >= localized_neff_minimum)
    )
    return containment, outside, localized_neff, supported, supported_suffix(supported)


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
    orders: Mapping[str, int],
) -> MixingEvaluation:
    matrix, moments, audit = compute_mixing_matrix(
        bins,
        specification,
        radial_order=int(orders["radial"]),
        shell_order=int(orders["shell"]),
        mu_order=int(orders["mu"]),
    )
    containment, outside, neff, supported, suffix = evaluate_support(
        matrix, independent_counts, moments, specification.box_size
    )
    return MixingEvaluation(
        matrix, containment, outside, neff, supported, suffix, moments, audit
    )


def _roi_result(
    roi_id: str, key: str, evaluation: MixingEvaluation
) -> dict[str, object]:
    return {
        "ROI_id": roi_id,
        "numeric_product_key": key,
        "containment": evaluation.containment.tolist(),
        "outside_analysis": evaluation.outside_analysis.tolist(),
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
        "schema": "ouruniv-cf4-kf-roi-leakage-mode-counts-v1",
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


def validate_preflight_result(
    path: str | Path,
    expected_sha256: str,
    design_sha256: str,
    implementation_commit: str,
) -> dict[str, object]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise LeakageError("production requires a lowercase 64-hex preflight SHA256")
    payload = Path(path).read_bytes()
    actual = _sha256(payload)
    if actual != expected_sha256:
        raise LeakageError("preflight result SHA256 mismatch")
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LeakageError("cannot parse preflight result") from exc
    if result.get("status") != "PRECHECK" or result.get("mode") != "preflight":
        raise LeakageError("production requires a PRECHECK preflight result")
    if result.get("design_raw_sha256") != design_sha256:
        raise LeakageError("preflight design SHA256 mismatch")
    if result.get("implementation_commit") != implementation_commit:
        raise LeakageError("preflight implementation commit mismatch")
    if result.get("numerical_convergence", {}).get("status") != "PASS":
        raise LeakageError("production requires numerical preflight PASS")
    return result


def _compare_evaluations(
    coarse: Mapping[str, MixingEvaluation], fine: Mapping[str, MixingEvaluation]
) -> dict[str, object]:
    containment_difference = 0.0
    outside_difference = 0.0
    moment_difference = 0.0
    classification_match = True
    suffix_match = True
    parseval_overshoot = 0.0
    transform_zero_error = 0.0
    per_window: dict[str, object] = {}
    for key in sorted(fine):
        first = coarse[key]
        second = fine[key]
        containment_delta = float(
            np.max(np.abs(first.containment - second.containment))
        )
        outside_delta = float(
            np.max(np.abs(first.outside_analysis - second.outside_analysis))
        )
        relative_moments = max(
            abs(first.moments[name] - second.moments[name]) / second.moments[name]
            for name in ("int_W1_dV", "int_W2_dV", "int_W4_dV", "V_eff")
        )
        same_classification = bool(
            np.array_equal(first.raw_supported, second.raw_supported)
        )
        same_suffix = first.suffix["pass"] == second.suffix["pass"]
        overshoot = max(
            first.numerical_audit["maximum_analysis_column_overshoot"],
            second.numerical_audit["maximum_analysis_column_overshoot"],
        )
        a0_error = max(
            first.numerical_audit["sphere_A0_relative_error"],
            second.numerical_audit["sphere_A0_relative_error"],
        )
        containment_difference = max(containment_difference, containment_delta)
        outside_difference = max(outside_difference, outside_delta)
        moment_difference = max(moment_difference, relative_moments)
        classification_match &= same_classification
        suffix_match &= same_suffix
        parseval_overshoot = max(parseval_overshoot, overshoot)
        transform_zero_error = max(transform_zero_error, a0_error)
        per_window[key] = {
            "max_abs_containment_difference": containment_delta,
            "max_abs_outside_analysis_difference": outside_delta,
            "max_relative_window_moment_difference": relative_moments,
            "native_classification_identical": same_classification,
            "suffix_classification_identical": same_suffix,
            "max_parseval_probability_overshoot": overshoot,
            "coarse_sphere_A0_relative_error": first.numerical_audit[
                "sphere_A0_relative_error"
            ],
            "fine_sphere_A0_relative_error": second.numerical_audit[
                "sphere_A0_relative_error"
            ],
        }
    passed = (
        containment_difference <= CONVERGENCE_ABS_TOLERANCE
        and outside_difference <= CONVERGENCE_ABS_TOLERANCE
        and moment_difference <= CONVERGENCE_ABS_TOLERANCE
        and parseval_overshoot <= CONVERGENCE_ABS_TOLERANCE
        and transform_zero_error <= CONVERGENCE_ABS_TOLERANCE
        and classification_match
        and suffix_match
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "absolute_tolerance": CONVERGENCE_ABS_TOLERANCE,
        "max_abs_containment_difference": containment_difference,
        "max_abs_outside_analysis_difference": outside_difference,
        "max_relative_parseval_denominator_or_window_moment_difference": (
            moment_difference
        ),
        "max_parseval_probability_overshoot": parseval_overshoot,
        "max_sphere_A0_relative_error": transform_zero_error,
        "native_classification_identical": classification_match,
        "suffix_classification_identical": suffix_match,
        "threshold_margin_safety_pass": classification_match and suffix_match,
        "per_numeric_window": per_window,
    }


def calculate(
    design: Mapping[str, object],
    design_sha: str,
    mode: str,
    implementation_commit: str,
    *,
    preflight_result: str | Path | None = None,
    preflight_result_sha256: str | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, np.ndarray]]:
    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise LeakageError("implementation commit must be lowercase 40-hex")
    validated_preflight = None
    if mode == "production":
        if preflight_result is None or preflight_result_sha256 is None:
            raise LeakageError(
                "production requires --preflight-result and its exact SHA256"
            )
        validated_preflight = validate_preflight_result(
            preflight_result,
            preflight_result_sha256,
            design_sha,
            implementation_commit,
        )
    elif mode != "preflight":
        raise LeakageError(f"unknown mode {mode!r}")
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

    if mode == "preflight":
        coarse = {
            key: _evaluate_window(bins, spec, independent_counts, COARSE_ORDERS)
            for key, spec in specifications.items()
        }
        fine = {
            key: _evaluate_window(bins, spec, independent_counts, FINE_ORDERS)
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
        status = "PRECHECK"
        scientific_disposition = "PRECHECK_only_not_scientific"
        preflight_binding = None
    elif mode == "production":
        if validated_preflight is None:
            raise LeakageError("internal preflight validation state is missing")
        evaluations = {
            key: _evaluate_window(bins, spec, independent_counts, FINE_ORDERS)
            for key, spec in specifications.items()
        }
        for key in sorted(specifications):
            arrays[f"production__{key}"] = evaluations[key].matrix
        convergence = {
            "status": "PASS",
            "source": "validated_preflight_result",
            "preflight_result_sha256": preflight_result_sha256,
        }
        status = "COMPLETE"
        preflight_binding = {
            "path": str(preflight_result),
            "sha256": preflight_result_sha256,
        }
        scientific_disposition = "pending_leakage_gate"

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
    if mode == "production":
        scientific_disposition = (
            "leakage_PASS_final_manifest_still_requires_separate_materialization_authority"
            if leakage_gate_pass
            else "scientific_NO_GO_final_manifest_blocked"
        )
    result = {
        "schema": "ouruniv-cf4-kf-roi-leakage-result-v1",
        "status": status,
        "mode": mode,
        "design_raw_sha256": design_sha,
        "implementation_path": "src/cf4_kf_roi_leakage.py",
        "implementation_sha256": implementation_sha,
        "implementation_commit": implementation_commit,
        "truth_or_candidate_data_consumed": False,
        "quadrature_orders": FINE_ORDERS,
        "preflight_binding": preflight_binding,
        "numerical_convergence": convergence,
        "ROI_results": roi_results,
        "Local_Group_observer_numeric_product_shared": True,
        "Local_Group_observer_semantic_results_separate": True,
        "Local_Group_observer_scores_summed": False,
        "overall_leakage_gate_pass": leakage_gate_pass,
        "scientific_disposition": scientific_disposition,
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
            "schema": "ouruniv-cf4-kf-roi-leakage-artifact-manifest-v1",
            "status": "COMPLETE" if result["status"] == "COMPLETE" else "PRECHECK",
            "mode": result["mode"],
            "design_raw_sha256": result["design_raw_sha256"],
            "implementation_commit": result["implementation_commit"],
            "payloads": {
                filename: {"sha256": _sha256(payload), "bytes": len(payload)}
                for filename, payload in sorted(payloads.items())
            },
        }
        manifest_payload = canonical_json_bytes(manifest)
        _write_and_fsync(stage / "manifest.json", manifest_payload)
        complete = {
            "schema": "ouruniv-cf4-kf-roi-leakage-complete-v1",
            "status": manifest["status"],
            "mode": result["mode"],
            "manifest_sha256": _sha256(manifest_payload),
            "design_raw_sha256": result["design_raw_sha256"],
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
    parser.add_argument("--mode", required=True, choices=("preflight", "production"))
    parser.add_argument("--preflight-result", type=Path)
    parser.add_argument("--preflight-result-sha256")
    parser.add_argument("--implementation-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"refusing overwrite of existing output {args.output}")
        design, design_sha = load_frozen_design(args.design)
        result, mode_counts, arrays = calculate(
            design,
            design_sha,
            args.mode,
            args.implementation_commit,
            preflight_result=args.preflight_result,
            preflight_result_sha256=args.preflight_result_sha256,
        )
        publish_artifacts(args.output, result, mode_counts, arrays)
    except (OSError, LeakageError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "mode": result["mode"],
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
