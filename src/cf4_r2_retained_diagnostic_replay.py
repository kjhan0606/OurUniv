"""Rebuild field accumulators from lossless retained canonical chain states.

This path deliberately reevaluates only the deterministic PM field, not the
observation likelihood or held-out data. It is suitable for reconstructing
posterior-moment and field-MCSE bookkeeping after a chain interruption.
"""
from __future__ import annotations


def replay_retained_field_diagnostics(state_store, *, field_moments,
                                      posterior_moments, field_mcse):
    """Replay every retained q, including exact repeats after rejection.

    ``field_moments(q)`` must return aligned ``rho``,
    ``mean_velocity_km_s``, ``physical_velocity_variance_km2_s2`` and
    ``velocity_valid`` maps using the frozen PM forward model.
    Accumulators must be empty; partially restored state is rejected so a
    caller cannot silently double-count an unknown suffix.
    """
    if state_store.format_version != state_store.VERSION or state_store.VERSION < 2:
        raise ValueError("RNG-audited canonical-state store v2 or newer required")
    if not hasattr(posterior_moments, "n") or posterior_moments.n != 0:
        raise ValueError("posterior-moment accumulator must be empty before replay")
    if field_mcse.retained_states != 0:
        raise ValueError("field-MCSE accumulator must be empty before replay")

    replayed = 0
    accepted = 0
    rejected = 0
    for q, record in state_store.iter_states(verify=True):
        field = field_moments(q)
        required = ("rho", "mean_velocity_km_s",
                    "physical_velocity_variance_km2_s2", "velocity_valid")
        if not isinstance(field, dict) or any(key not in field for key in required):
            raise ValueError("PM replay must return all retained-field moments")
        if record["transition"] != replayed + 1:
            raise ValueError("retained-state replay order is not consecutive")
        posterior_moments.update(field)
        field_mcse.update(dict(rho=field["rho"],
            mean_velocity_km_s=field["mean_velocity_km_s"],
            velocity_valid=field["velocity_valid"]),
            transition=record["transition"], accepted=record["accepted"])
        replayed += 1
        accepted += int(record["accepted"])
        rejected += int(not record["accepted"])
        del field, q

    if (posterior_moments.n != replayed
            or field_mcse.retained_states != replayed
            or field_mcse.accepted_transitions != accepted
            or field_mcse.rejected_transitions != rejected):
        raise AssertionError("replayed field diagnostics disagree with canonical chain index")
    return dict(retained_states=replayed, accepted_transitions=accepted,
        rejected_transitions=rejected,
        field_MCSE_interpretation="replayed bookkeeping only; not convergence or posterior validation")
