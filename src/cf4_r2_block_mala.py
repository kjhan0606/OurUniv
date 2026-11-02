"""Exact Metropolis-adjusted Langevin updates on fixed prior-white blocks."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlockMALAResult:
    position: np.ndarray
    potential: float
    gradient: np.ndarray
    proposed_position: np.ndarray
    proposed_potential: float
    proposed_gradient: np.ndarray
    accepted: bool
    log_q_forward: float
    log_q_reverse: float
    log_acceptance: float
    proposal_displacement_rms: float
    accepted_displacement_rms: float
    retained_displacement_rms: float


def _finite_vector(name, value, shape):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with shape {shape}")
    return result


def _block_dimension(project_block):
    dimension = getattr(project_block, "dimension", None)
    if not isinstance(dimension, (int, np.integer)) or dimension <= 0:
        raise ValueError("block projector must declare a positive dimension")
    return int(dimension)


def block_gaussian_logpdf(destination, origin, origin_gradient,
                          project_block, step_size):
    """Log density of a projected MALA proposal, normalized on its block."""
    step = float(step_size)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("MALA step size must be finite and positive")
    origin = np.asarray(origin, dtype=np.float64)
    if origin.ndim != 1 or not np.isfinite(origin).all():
        raise ValueError("MALA origin must be a finite vector")
    shape = origin.shape
    destination = _finite_vector("MALA destination", destination, shape)
    gradient = _finite_vector("MALA origin gradient", origin_gradient, shape)
    dimension = _block_dimension(project_block)
    displacement = destination - origin
    block_displacement = _finite_vector(
        "projected displacement", project_block(displacement), shape)
    if not np.allclose(displacement, block_displacement, rtol=1e-11, atol=1e-11):
        raise ValueError("MALA proposal changes coordinates outside its block")
    block_gradient = _finite_vector(
        "projected gradient", project_block(gradient), shape)
    residual = displacement + 0.5 * step * step * block_gradient
    return float(-0.5 * np.dot(residual, residual) / (step * step)
                 - dimension * np.log(step)
                 - 0.5 * dimension * np.log(2.0 * np.pi))


def block_mala_log_acceptance(current_position, current_potential,
                              current_gradient, proposed_position,
                              proposed_potential, proposed_gradient,
                              project_block, step_size):
    """Full-target Metropolis log ratio for one fixed-block MALA proposal."""
    x = np.asarray(current_position, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("current position must be a finite vector")
    y = _finite_vector("proposed position", proposed_position, x.shape)
    gx = _finite_vector("current gradient", current_gradient, x.shape)
    gy = _finite_vector("proposed gradient", proposed_gradient, x.shape)
    ux, uy = float(current_potential), float(proposed_potential)
    if not np.isfinite([ux, uy]).all():
        raise FloatingPointError("MALA target potentials must be finite")
    log_forward = block_gaussian_logpdf(y, x, gx, project_block, step_size)
    log_reverse = block_gaussian_logpdf(x, y, gy, project_block, step_size)
    log_alpha = -uy + ux + log_reverse - log_forward
    if not np.isfinite(log_alpha):
        raise FloatingPointError("MALA Metropolis ratio is nonfinite")
    return float(log_alpha), float(log_forward), float(log_reverse)


def block_mala_step(position, potential, gradient, evaluate, project_block,
                    rng, *, step_size):
    """Propose and MH-correct one conditional MALA move in an orthogonal block.

    ``evaluate(candidate)`` must return the complete negative-log-target value
    and its complete gradient. A rejected proposal returns the unchanged
    current state and gradient; the caller can safely retain its cached target
    evaluation. No likelihood factor or prior term is dropped.
    """
    step = float(step_size)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("MALA step size must be finite and positive")
    x = np.asarray(position, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("current position must be a finite vector")
    shape = x.shape
    gx = _finite_vector("current gradient", gradient, shape)
    ux = float(potential)
    if not np.isfinite(ux):
        raise FloatingPointError("current target potential must be finite")
    dimension = _block_dimension(project_block)
    block_gradient = _finite_vector(
        "projected current gradient", project_block(gx), shape)
    noise = _finite_vector(
        "projected standard-normal noise",
        project_block(rng.standard_normal(shape)), shape)
    displacement = -0.5 * step * step * block_gradient + step * noise
    y = x + displacement

    proposed_value, proposed_gradient = evaluate(y)
    uy = float(proposed_value)
    gy = _finite_vector("proposed gradient", proposed_gradient, shape)
    if not np.isfinite(uy):
        raise FloatingPointError("proposed target potential must be finite")
    log_alpha, log_forward, log_reverse = block_mala_log_acceptance(
        x, ux, gx, y, uy, gy, project_block, step)
    uniform = max(float(rng.random()), np.finfo(np.float64).tiny)
    accepted = bool(np.log(uniform) < min(0.0, log_alpha))
    proposed_rms = float(np.linalg.norm(displacement) / np.sqrt(dimension))
    if accepted:
        return BlockMALAResult(
            position=y.copy(), potential=uy, gradient=gy.copy(),
            proposed_position=y.copy(), proposed_potential=uy,
            proposed_gradient=gy.copy(), accepted=True,
            log_q_forward=log_forward, log_q_reverse=log_reverse,
            log_acceptance=log_alpha,
            proposal_displacement_rms=proposed_rms,
            accepted_displacement_rms=proposed_rms,
            retained_displacement_rms=proposed_rms,
        )
    return BlockMALAResult(
        position=x.copy(), potential=ux, gradient=gx.copy(),
        proposed_position=y.copy(), proposed_potential=uy,
        proposed_gradient=gy.copy(), accepted=False,
        log_q_forward=log_forward, log_q_reverse=log_reverse,
        log_acceptance=log_alpha,
        proposal_displacement_rms=proposed_rms,
        accepted_displacement_rms=0.0,
        retained_displacement_rms=0.0,
    )
