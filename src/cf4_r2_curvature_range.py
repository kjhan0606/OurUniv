"""Small matrix-free helpers for the conditional-v6 curvature range screen.

The screen estimates local negative-log-likelihood Hessian actions in
prior-whitened IC coordinates. It is a diagnostic, not an LIS sampler or
posterior covariance estimator.
"""

import numpy as np


def local_probe_schedule(directions, *, step=0.1):
    """Freeze the 4-sketch, 2-validation, and 2 half-step-replay pairs."""
    directions = tuple(np.asarray(vector, dtype=np.float64) for vector in directions)
    if (len(directions) != 6 or not np.isfinite(step) or step <= 0
            or any(vector.ndim != 1 or vector.shape != directions[0].shape
                   or not np.isfinite(vector).all() for vector in directions)):
        raise ValueError("six finite, same-shape directions and a positive step required")
    half = step / 2.0
    return tuple(
        [(f"sketch_{i + 1}", "range_sketch", directions[i], float(step))
         for i in range(4)]
        + [(f"validation_{i + 1}", "independent_random_direction_validation",
            directions[4 + i], float(step)) for i in range(2)]
        + [("sketch_1_half_step", "step_stability_repeat", directions[0], half),
           ("validation_1_half_step", "step_stability_repeat", directions[4], half)]
    )


def orthonormal_random_directions(n_coordinates, count, *, seed):
    """Return deterministic orthonormal Gaussian directions, one at a time."""
    if (not isinstance(n_coordinates, int) or n_coordinates < 1
            or not isinstance(count, int) or count < 1 or count > n_coordinates):
        raise ValueError("positive compatible coordinate and direction counts required")
    rng = np.random.default_rng(seed)
    basis = []
    for _ in range(count):
        vector = rng.standard_normal(n_coordinates)
        # Re-orthogonalize once to keep the finite probe well conditioned.
        for _ in range(2):
            for previous in basis:
                vector -= np.dot(previous, vector) * previous
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise FloatingPointError("random direction lost numerical rank")
        basis.append(vector / norm)
    return tuple(basis)


def likelihood_hessian_action(gradient_plus, gradient_minus, ic_direction,
                              *, step, n_ic):
    """Centered exact-target gradient difference minus the unit IC prior.

    The canonical target uses standard-normal coordinates, so the prior
    Hessian is identity. ``ic_direction`` is unit norm and nuisance-fixed;
    the returned tail retains the IC-to-nuisance cross response.
    """
    plus = np.asarray(gradient_plus, dtype=np.float64)
    minus = np.asarray(gradient_minus, dtype=np.float64)
    direction = np.asarray(ic_direction, dtype=np.float64)
    if (not np.isfinite(step) or step <= 0 or not isinstance(n_ic, int)
            or plus.ndim != 1 or minus.shape != plus.shape
            or plus.size != n_ic + 24 or direction.shape != (n_ic,)
            or not np.isfinite(plus).all() or not np.isfinite(minus).all()
            or not np.isfinite(direction).all()):
        raise ValueError("finite GL2 gradients and a matching IC direction required")
    action = (plus - minus) / (2.0 * step)
    action[:n_ic] -= direction
    if not np.isfinite(action).all():
        raise FloatingPointError("nonfinite likelihood-Hessian action")
    return action


def relative_action_change(reference, comparison):
    """Relative L2 change between two finite-difference actions."""
    reference = np.asarray(reference, dtype=np.float64)
    comparison = np.asarray(comparison, dtype=np.float64)
    if (reference.shape != comparison.shape or reference.ndim != 1
            or not np.isfinite(reference).all() or not np.isfinite(comparison).all()):
        raise ValueError("finite same-shape actions required")
    denominator = max(float(np.linalg.norm(reference)),
                      float(np.linalg.norm(comparison)),
                      np.finfo(np.float64).tiny)
    return float(np.linalg.norm(reference - comparison) / denominator)


def support_summary_difference(plus, minus):
    """Compare stable support-size summaries, not volatile timing metadata.

    This does not identify exact cell membership; it only reports differences
    in the summary quantities emitted by the active support builder.
    """
    fields = {
        "component_count_delta": "components",
        "largest_candidate_cells_delta": "largest_candidate_cells",
        "radius_delta_cMpc_h": "radius",
    }
    if any(key not in summary or not np.isfinite(summary[key])
           for summary in (plus, minus) for key in fields.values()):
        raise ValueError("support summaries must contain finite comparison fields")
    return {
        output: float(plus[source] - minus[source])
        for output, source in fields.items()
    }


def response_range(responses, *, relative_rank_tolerance=1e-10):
    """Build a re-orthogonalized range and singular values from few responses."""
    responses = tuple(np.asarray(row, dtype=np.float64) for row in responses)
    if (not responses or any(row.ndim != 1 or row.shape != responses[0].shape
                             or not np.isfinite(row).all() for row in responses)
            or not np.isfinite(relative_rank_tolerance)
            or relative_rank_tolerance <= 0):
        raise ValueError("finite same-shape response vectors required")

    count = len(responses)
    gram = np.empty((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i, count):
            value = float(np.dot(responses[i], responses[j]))
            gram[i, j] = gram[j, i] = value
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = relative_rank_tolerance * largest

    basis = []
    for response in responses:
        residual = response.copy()
        for _ in range(2):
            for previous in basis:
                residual -= np.dot(previous, residual) * previous
        norm = float(np.linalg.norm(residual))
        if norm > threshold:
            basis.append(residual / norm)
    return tuple(basis), singular_values


def outside_range_energy_fraction(response, basis):
    """Fraction of response squared norm not captured by an orthonormal range."""
    response = np.asarray(response, dtype=np.float64)
    basis = tuple(np.asarray(vector, dtype=np.float64) for vector in basis)
    if (response.ndim != 1 or not np.isfinite(response).all()
            or any(vector.shape != response.shape or not np.isfinite(vector).all()
                   for vector in basis)):
        raise ValueError("finite response and compatible finite basis required")
    norm2 = float(np.dot(response, response))
    if norm2 == 0.0:
        return None
    residual = response.copy()
    for vector in basis:
        residual -= np.dot(vector, residual) * vector
    return float(np.clip(np.dot(residual, residual) / norm2, 0.0, 1.0))
