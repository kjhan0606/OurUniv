"""Small, host-side statistics for one bounded R2 VJP geometry sketch."""

import math

import numpy as np
from scipy.stats import chi2


def leading_left_basis(sketch, rank, *, relative_singular_tolerance=100.0 * np.finfo(np.float64).eps):
    """Return leading left singular vectors using a stable tall-skinny QR.

    ``sketch`` is coordinates-by-probes, with each column a VJP vector. The
    full-size QR/SVD is deliberately host-side; only the small R factor is
    decomposed. No target or rows are modified by this transformation.
    """
    matrix = np.asarray(sketch, dtype=np.float64)
    rank = int(rank)
    tolerance = float(relative_singular_tolerance)
    if (matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < rank
            or rank < 1 or not np.isfinite(matrix).all()):
        raise ValueError("finite coordinates-by-probes sketch and valid rank required")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("positive finite singular-value tolerance required")

    q, r = np.linalg.qr(matrix, mode="reduced")
    left, singular_values, _ = np.linalg.svd(r, full_matrices=False)
    if (not np.isfinite(singular_values).all() or singular_values[0] <= 0.0
            or singular_values[rank - 1] <= tolerance * singular_values[0]):
        raise np.linalg.LinAlgError("candidate sketch does not support the requested numerical rank")
    basis = q @ left[:, :rank]
    orthogonality_error = float(np.linalg.norm(basis.T @ basis - np.eye(rank), ord="fro"))
    if not math.isfinite(orthogonality_error):
        raise FloatingPointError("candidate basis orthogonality check is nonfinite")
    return basis, singular_values, orthogonality_error


def subspace_canonical_correlations(first, second):
    """Return descending canonical-correlation singular values for two spans."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if (first.ndim != 2 or second.shape != first.shape
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError("aligned finite subspace bases required")
    correlations = np.linalg.svd(first.T @ second, compute_uv=False)
    return np.sort(np.clip(correlations, 0.0, 1.0))[::-1]


def project_probe(basis, probe):
    """Project one full VJP and return coordinates plus residual/full norms."""
    basis = np.asarray(basis, dtype=np.float64)
    probe = np.asarray(probe, dtype=np.float64)
    if (basis.ndim != 2 or probe.shape != (basis.shape[0],)
            or not np.isfinite(basis).all() or not np.isfinite(probe).all()):
        raise ValueError("finite basis and aligned finite VJP vector required")
    coordinates = basis.T @ probe
    residual = probe - basis @ coordinates
    residual_squared_norm = float(np.dot(residual, residual))
    full_squared_norm = float(np.dot(probe, probe))
    if (not np.isfinite(coordinates).all()
            or not math.isfinite(residual_squared_norm)
            or not math.isfinite(full_squared_norm)):
        raise FloatingPointError("projected VJP summary is nonfinite")
    return coordinates, residual_squared_norm, full_squared_norm


def chi_square_second_moment_interval(values, *, family_alpha=0.05,
                                      family_size=4):
    """Exact Bonferroni interval for an uncentered zero-mean Gaussian second moment."""
    values = np.asarray(values, dtype=np.float64)
    alpha = float(family_alpha)
    family_size = int(family_size)
    if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
            or not math.isfinite(alpha) or not 0.0 < alpha < 1.0
            or family_size < 1):
        raise ValueError("finite samples and valid family-wise interval settings required")
    estimate = float(np.mean(values * values, dtype=np.float64))
    if estimate <= 0.0:
        raise ValueError("zero directional second moment has no relative interval")
    per_tail = alpha / (2.0 * family_size)
    degrees = values.size
    lower = degrees * estimate / float(chi2.ppf(1.0 - per_tail, degrees))
    upper = degrees * estimate / float(chi2.ppf(per_tail, degrees))
    relative_half_width = max(estimate - lower, upper - estimate) / estimate
    return dict(estimate=estimate, lower=float(lower), upper=float(upper),
                relative_half_width=float(relative_half_width),
                degrees_of_freedom=int(degrees),
                per_interval_confidence=1.0 - alpha / family_size)


def _matrix_relative_difference(first, second):
    average_norm = float(np.linalg.norm(0.5 * (first + second), ord="fro"))
    if not math.isfinite(average_norm) or average_norm <= 0.0:
        return None
    difference = float(np.linalg.norm(first - second, ord="fro"))
    return difference / average_norm


def _bootstrap_projected_uncertainty(projections, *, replicates, seed, alpha):
    n, rank = projections.shape
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, n, size=(int(replicates), n), dtype=np.int32)
    resampled = projections[indices]
    matrices = np.einsum("bni,bnj->bij", resampled, resampled, optimize=True) / n
    low = np.quantile(matrices, alpha / 2.0, axis=0)
    high = np.quantile(matrices, 1.0 - alpha / 2.0, axis=0)
    eigenvalues = np.linalg.eigvalsh(matrices)
    eigen_low = np.quantile(eigenvalues, alpha / 2.0, axis=0)
    eigen_high = np.quantile(eigenvalues, 1.0 - alpha / 2.0, axis=0)
    return dict(
        method="percentile bootstrap over independent validation probes; conditional on frozen U",
        seed=int(seed), replicates=int(replicates), confidence=float(1.0 - alpha),
        off_diagonal_lower=low.tolist(), off_diagonal_upper=high.tolist(),
        sorted_eigenvalue_lower=eigen_low.tolist(),
        sorted_eigenvalue_upper=eigen_high.tolist(),
    )


def summarize_validation(projections, residual_squared_norms, full_squared_norms,
                         *, batch_size=32, theta_min=0.10, residual_limit_nats=1.0,
                         matrix_disagreement_limit=0.50,
                         directional_relative_half_width_limit=0.50,
                         bootstrap_replicates=10000, bootstrap_seed=20261102):
    """Evaluate the frozen-rank validation gates without altering its basis."""
    projections = np.asarray(projections, dtype=np.float64)
    residuals = np.asarray(residual_squared_norms, dtype=np.float64)
    full = np.asarray(full_squared_norms, dtype=np.float64)
    batch_size = int(batch_size)
    theta_min = float(theta_min)
    residual_limit_nats = float(residual_limit_nats)
    matrix_limit = float(matrix_disagreement_limit)
    width_limit = float(directional_relative_half_width_limit)
    if (projections.ndim != 2 or projections.shape[0] < 2 * batch_size
            or projections.shape[0] != 2 * batch_size or projections.shape[1] < 1
            or residuals.shape != (projections.shape[0],)
            or full.shape != residuals.shape
            or not np.isfinite(projections).all()
            or not np.isfinite(residuals).all() or np.any(residuals < 0.0)
            or not np.isfinite(full).all() or np.any(full < 0.0)):
        raise ValueError("two aligned finite validation batches are required")
    if (not math.isfinite(theta_min) or theta_min <= 0.0
            or not math.isfinite(residual_limit_nats) or residual_limit_nats <= 0.0
            or not math.isfinite(matrix_limit) or matrix_limit <= 0.0
            or not math.isfinite(width_limit) or width_limit <= 0.0):
        raise ValueError("positive finite gate thresholds required")

    n, rank = projections.shape
    first, second = projections[:batch_size], projections[batch_size:]
    matrix_first = first.T @ first / batch_size
    matrix_second = second.T @ second / batch_size
    matrix_all = projections.T @ projections / n
    matrix_disagreement = _matrix_relative_difference(matrix_first, matrix_second)
    directional = []
    for axis in range(rank):
        try:
            interval = chi_square_second_moment_interval(
                projections[:, axis], family_alpha=0.05, family_size=rank)
        except ValueError:
            interval = dict(estimate=0.0, lower=0.0, upper=0.0,
                            relative_half_width=None,
                            degrees_of_freedom=int(n),
                            per_interval_confidence=1.0 - 0.05 / rank)
        interval["passed"] = bool(
            interval["estimate"] > 0.0
            and interval["relative_half_width"] is not None
            and interval["relative_half_width"] <= width_limit)
        directional.append(interval)

    mean_residual = float(np.mean(residuals, dtype=np.float64))
    epsilon = 2.0 * math.sqrt(math.log(1.0 / 0.05) / n)
    if epsilon >= 1.0:
        raise ValueError("validation sample count is too small for the residual UCL")
    residual_ucl = mean_residual / (1.0 - epsilon)
    mean_full = float(np.mean(full, dtype=np.float64))
    residual_share = (mean_residual / mean_full) if mean_full > 0.0 else None
    residual_nats = 0.5 * theta_min * theta_min * residual_ucl

    bootstrap = _bootstrap_projected_uncertainty(
        projections, replicates=bootstrap_replicates,
        seed=bootstrap_seed, alpha=0.05)
    gates = dict(
        projected_batch_disagreement=(matrix_disagreement is not None
                                      and matrix_disagreement <= matrix_limit),
        all_directional_intervals_passed=all(row["passed"] for row in directional),
        residual_absolute_information=(residual_nats <= residual_limit_nats),
    )
    return dict(
        validation_probe_count=int(n), batch_size=int(batch_size),
        projected_second_moment=matrix_all.tolist(),
        projected_second_moment_batch1=matrix_first.tolist(),
        projected_second_moment_batch2=matrix_second.tolist(),
        projected_batch_relative_frobenius_difference=matrix_disagreement,
        projected_batch_difference_limit=matrix_limit,
        directional_second_moment_intervals=directional,
        directional_relative_half_width_limit=width_limit,
        projected_uncertainty=bootstrap,
        residual_trace_estimate=mean_residual,
        residual_trace_one_sided_95_ucl=float(residual_ucl),
        residual_trace_fraction_of_total=residual_share,
        residual_ucl_method=(
            "Gaussian quadratic-form lower-tail bound; U fixed independently of validation; "
            "mean/(1-2*sqrt(log(20)/n))"),
        theta_min_rad=theta_min,
        residual_quadratic_cost_nats=float(residual_nats),
        residual_quadratic_cost_limit_nats=residual_limit_nats,
        gates=gates,
        passed=bool(all(gates.values())),
    )
