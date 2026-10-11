"""Coarse-force proposals, fixed fine-target Metropolis correction.

This is not likelihood tempering. Both oracles are deterministic functions
of the current state, with fresh support. The fixed metric and symmetric
split map preserve phase-space volume/reversibility; fine energies, not
coarse values, determine acceptance. Production tuning is forbidden.
"""
import numpy as np
from cf4_r2_prior_split_hmc import split_trajectory


def corrected_split_step(force_oracle,fine_value,metric,q,energy,force_value,gradient,rng,*,step,steps):
    """Return accepted state, FINE energy and its matching COARSE force cache."""
    p=metric.momentum(rng)
    initial_fine=float(energy);initial_force=float(force_value)
    initial_kinetic=metric.kinetic(p)
    start=initial_fine+initial_kinetic
    if not np.isfinite(start):raise FloatingPointError('finite fine initial Hamiltonian required')
    calls=0
    def counted(position):
        nonlocal calls
        calls+=1
        return force_oracle(position)
    proposal,pend,coarse,derivative=split_trajectory(counted,metric,q,p,step,steps,
        initial_evaluation=(force_value,gradient))
    # An undefined force path is rejected, never clipped or continued with a
    # substituted gradient. Runtime/capacity errors still propagate.
    fine=float(fine_value(proposal)) if np.isfinite(coarse) else np.inf
    if np.isnan(fine) or fine==-np.inf:raise FloatingPointError('undefined fine endpoint energy')
    final_kinetic=metric.kinetic(pend) if np.isfinite(fine) else None
    delta=fine+final_kinetic-start if np.isfinite(fine) else np.inf
    if np.isfinite(fine) and not np.isfinite(delta):raise FloatingPointError('nonfinite fine Hamiltonian difference')
    log_acceptance=min(0.,-delta)
    accepted=bool(np.log(rng.uniform())<log_acceptance)
    force_error=(coarse+final_kinetic-initial_force-initial_kinetic
                 if final_kinetic is not None else None)
    correction_change=((fine-coarse)-(initial_fine-initial_force)
                       if final_kinetic is not None else None)
    if accepted:q,energy,force_value,gradient=proposal,fine,coarse,derivative
    return q,energy,force_value,gradient,dict(accepted=accepted,energy_error=delta,
        log_acceptance=log_acceptance,force_evaluations=calls,
        fine_endpoint_evaluations=int(np.isfinite(coarse)),
        initial_fine_energy=initial_fine,initial_force_energy=initial_force,
        proposed_fine_energy=fine,proposed_force_energy=float(coarse),
        force_hamiltonian_error=force_error,fine_force_correction_change=correction_change,
        fine_minus_coarse=float(energy-force_value))
