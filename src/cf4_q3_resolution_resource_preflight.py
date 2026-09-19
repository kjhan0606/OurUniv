"""Read-only Q3 resolution/resource envelope calculations.

This module does not allocate a PM field, run JAX, or launch infrastructure.  It
only derives the frozen static-shape envelope for the full PM source basis at
the Q3 low-k resolution and at the deferred high-resolution projection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


PM_SOURCE_PARTICLES = 192**3
BOX_SIZE_CMPCH = 384.0
MAX_DISPLACEMENT_CMPCH = 2.316206864681996
TAIL_CUTOFF_SIGMA = 8.0
CHUNK_SIZE_PARTICLES = 1024
POPULATIONS = 6
FLOAT64_BYTES = 8
MAX_HOST_MEMORY_BYTES = 64 * 1024**3
MAX_DEVICE_MEMORY_BYTES = 16 * 1024**3


class PreflightInputError(ValueError):
    """Raised when a resolution/resource input is not physically valid."""


@dataclass(frozen=True)
class ResolutionPoint:
    label: str
    cell_spacing_cMpc_h: float
    grid_size: int
    source_particles: int
    chunk_size_particles: int
    tail_cutoff_sigma: float
    k_max: int
    padded_breaks_bytes: int
    padded_polynomial_coefficients_bytes: int
    padded_interval_mask_bytes: int
    padded_positions_bytes: int
    padded_los_bytes: int
    padded_masses_bytes: int
    padded_output_bytes: int
    padded_array_bytes_total: int
    padded_array_MiB: float
    fits_device_memory: bool
    fits_host_memory: bool
    observational_information_claim_authorized: bool


def _positive_finite(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise PreflightInputError(f"{name} must be positive and finite")
    return float(value)


def derive_k_max(
    cell_spacing_cMpc_h: float,
    *,
    max_displacement_cMpc_h: float = MAX_DISPLACEMENT_CMPCH,
    tail_cutoff_sigma: float = TAIL_CUTOFF_SIGMA,
) -> int:
    """Return the declared fixed-shape interval bound for a cubic mesh."""

    spacing = _positive_finite(cell_spacing_cMpc_h, "cell_spacing_cMpc_h")
    displacement = _positive_finite(max_displacement_cMpc_h, "max_displacement_cMpc_h")
    tail = _positive_finite(tail_cutoff_sigma, "tail_cutoff_sigma")
    crossing = math.ceil(2.0 * tail * displacement / spacing)
    return 1 + 3 * (crossing + 2)


def _grid_size(cell_spacing_cMpc_h: float, box_size_cMpc_h: float) -> int:
    spacing = _positive_finite(cell_spacing_cMpc_h, "cell_spacing_cMpc_h")
    box = _positive_finite(box_size_cMpc_h, "box_size_cMpc_h")
    ratio = box / spacing
    grid = int(round(ratio))
    if grid < 2 or not math.isclose(ratio, grid, rel_tol=0.0, abs_tol=1.0e-12):
        raise PreflightInputError("box_size/cell_spacing must be an integer grid size >= 2")
    return grid


def _padded_bytes(grid_size: int, k_max: int, chunk_size: int) -> dict[str, int]:
    if grid_size < 2 or k_max < 1 or chunk_size < 1:
        raise PreflightInputError("grid_size, k_max and chunk_size must be positive")
    # These shapes mirror the frozen Q2 static-shape accounting: interval
    # breaks, 27 TSC polynomial coefficients of degree six, a boolean mask,
    # source positions/LOS/masses, and six output fields.
    return {
        "padded_breaks_bytes": (k_max + 1) * chunk_size * FLOAT64_BYTES,
        "padded_polynomial_coefficients_bytes": k_max * 27 * 7 * chunk_size * FLOAT64_BYTES,
        "padded_interval_mask_bytes": k_max * chunk_size,
        "padded_positions_bytes": 3 * chunk_size * FLOAT64_BYTES,
        "padded_los_bytes": 3 * chunk_size * FLOAT64_BYTES,
        "padded_masses_bytes": POPULATIONS * chunk_size * FLOAT64_BYTES,
        "padded_output_bytes": POPULATIONS * grid_size**3 * FLOAT64_BYTES,
    }


def resolution_point(
    label: str,
    cell_spacing_cMpc_h: float,
    *,
    box_size_cMpc_h: float = BOX_SIZE_CMPCH,
    source_particles: int = PM_SOURCE_PARTICLES,
    chunk_size_particles: int = CHUNK_SIZE_PARTICLES,
    max_displacement_cMpc_h: float = MAX_DISPLACEMENT_CMPCH,
    tail_cutoff_sigma: float = TAIL_CUTOFF_SIGMA,
    observational_information_claim_authorized: bool = False,
) -> ResolutionPoint:
    if not label:
        raise PreflightInputError("label must be non-empty")
    if not isinstance(source_particles, int) or source_particles <= 0:
        raise PreflightInputError("source_particles must be a positive integer")
    if not isinstance(chunk_size_particles, int) or chunk_size_particles <= 0:
        raise PreflightInputError("chunk_size_particles must be a positive integer")
    grid = _grid_size(cell_spacing_cMpc_h, box_size_cMpc_h)
    k_max = derive_k_max(
        cell_spacing_cMpc_h,
        max_displacement_cMpc_h=max_displacement_cMpc_h,
        tail_cutoff_sigma=tail_cutoff_sigma,
    )
    sizes = _padded_bytes(grid, k_max, chunk_size_particles)
    total = sum(sizes.values())
    return ResolutionPoint(
        label=label,
        cell_spacing_cMpc_h=float(cell_spacing_cMpc_h),
        grid_size=grid,
        source_particles=source_particles,
        chunk_size_particles=chunk_size_particles,
        tail_cutoff_sigma=float(tail_cutoff_sigma),
        k_max=k_max,
        **sizes,
        padded_array_bytes_total=total,
        padded_array_MiB=total / 1024.0**2,
        fits_device_memory=total <= MAX_DEVICE_MEMORY_BYTES,
        fits_host_memory=total <= MAX_HOST_MEMORY_BYTES,
        observational_information_claim_authorized=bool(observational_information_claim_authorized),
    )


def q3_resolution_preflight() -> dict[str, object]:
    """Return the frozen Q3 ladder and explicit deferred-target policy."""

    points = [
        resolution_point("development_12_cMpc_h", 12.0),
        resolution_point("q3_release_2_cMpc_h", 2.0),
        resolution_point(
            "deferred_0p3_cMpc_h_projection",
            0.3,
            observational_information_claim_authorized=False,
        ),
    ]
    return {
        "schema": "ouruniv-cf4-q3-resolution-resource-preflight-v1",
        "source_particles_are_full_pm_basis": True,
        "secure_cf4_object_count_substitution_forbidden": True,
        "points": [asdict(point) for point in points],
        "q3_release": "Only development_12_cMpc_h and q3_release_2_cMpc_h are Q3 operator points; 2 cMpc/h is the release resolution.",
        "deferred_target": "0.3 cMpc/h is a separate conditional high-resolution forward-model/resource projection, not an observational-information validation.",
        "gpfs_used": False,
        "slurm_used": False,
        "jax_executed": False,
    }
