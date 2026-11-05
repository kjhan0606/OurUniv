"""Small NumPy-only summaries for the bounded R2 VJP tail screen."""

import math

import numpy as np


def stable_norm(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("gradient must be a finite one-dimensional vector")
    scale = float(np.max(np.abs(array))) if array.size else 0.0
    if scale == 0.0:
        return 0.0
    norm = scale * float(np.linalg.norm(array / scale))
    if not math.isfinite(norm):
        raise FloatingPointError("gradient norm is not representable in float64")
    return norm


def split_low_intensity_tail(positive_rows, positive_values, *, fraction=0.01):
    """Return position indices for the low-intensity tail and its complement.

    Ties are deterministic: lower flat row IDs sort first. This only partitions
    a diagnostic cotangent; it does not remove rows from the model or target.
    """
    rows = np.asarray(positive_rows)
    values = np.asarray(positive_values, dtype=np.float64)
    if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
            or values.shape != rows.shape or not rows.size):
        raise ValueError("positive-row IDs and intensities must be aligned vectors")
    if (np.unique(rows).size != rows.size or np.any(rows < 0)
            or not np.isfinite(values).all() or np.any(values <= 0.0)):
        raise ValueError("positive rows must be unique and have finite positive intensity")
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("tail fraction must lie strictly between zero and one")
    order = np.lexsort((rows, values))
    n_tail = max(1, int(math.ceil(fraction * rows.size)))
    return order[:n_tail].astype(np.int64), order[n_tail:].astype(np.int64)


def summarize_vjp_split(full, tail, rest, *, tolerance=1.0e-8):
    """Check linear recombination and report per-probe norm/share/cross term."""
    full = np.asarray(full, dtype=np.float64)
    tail = np.asarray(tail, dtype=np.float64)
    rest = np.asarray(rest, dtype=np.float64)
    if (full.ndim != 1 or tail.shape != full.shape or rest.shape != full.shape
            or not all(np.isfinite(v).all() for v in (full, tail, rest))):
        raise ValueError("full, tail and rest gradients must be aligned and finite")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    full_norm = stable_norm(full)
    tail_norm = stable_norm(tail)
    rest_norm = stable_norm(rest)
    denominator = tail_norm + rest_norm
    decomposition_error = stable_norm(full - tail - rest)
    relative_error = decomposition_error / max(denominator, np.finfo(np.float64).tiny)

    scale = max(tail_norm, rest_norm)
    if scale == 0.0:
        tail_share = 0.0
    else:
        tail_scaled = (tail_norm / scale) ** 2
        rest_scaled = (rest_norm / scale) ** 2
        tail_share = tail_scaled / (tail_scaled + rest_scaled)
    with np.errstate(over="ignore", invalid="ignore"):
        cross_term = 2.0 * float(np.dot(tail, rest))
    if not math.isfinite(cross_term):
        raise FloatingPointError("tail/rest cross term is not representable in float64")
    return dict(
        full_norm=full_norm,
        tail_norm=tail_norm,
        rest_norm=rest_norm,
        tail_squared_norm_share=tail_share,
        doubled_tail_rest_cross_term=cross_term,
        decomposition_relative_error=relative_error,
        decomposition_passed=bool(relative_error <= tolerance),
        decomposition_tolerance=tolerance,
    )


def normalized_repeat_error(first, repeat):
    """Relative norm difference for deterministic replay of one cotangent."""
    first = np.asarray(first, dtype=np.float64)
    repeat = np.asarray(repeat, dtype=np.float64)
    if first.shape != repeat.shape or first.ndim != 1:
        raise ValueError("repeat gradients must be aligned one-dimensional vectors")
    first_norm = stable_norm(first)
    repeat_norm = stable_norm(repeat)
    difference = stable_norm(first - repeat)
    scale = max(first_norm, repeat_norm, np.finfo(np.float64).tiny)
    return dict(
        first_norm=first_norm,
        repeat_norm=repeat_norm,
        difference_norm=difference,
        relative_error=difference / scale,
    )
