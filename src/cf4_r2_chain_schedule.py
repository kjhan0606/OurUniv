"""Declared schedule with diminishing warmup-only step-size adaptation."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ChainSchedule:
    proposals: int = 48
    warmup: int = 12
    warmup_steps: int = 4
    production_steps: int = 8
    initial_step: float = .05220457782250625
    maximum_step: float = .08
    target_acceptance: float = .65

    def __post_init__(self):
        if not 0 <= self.warmup < self.proposals:
            raise ValueError('at least one retained proposal required')
        if min(self.warmup_steps, self.production_steps) < 1:
            raise ValueError('positive integration lengths required')
        if not 0 < self.initial_step <= self.maximum_step or not math.isfinite(self.maximum_step):
            raise ValueError('finite positive step bounds required')
        if not 0 < self.target_acceptance < 1:
            raise ValueError('interior acceptance target required')

    def steps(self, iteration):
        # Tune the step against the eventual trajectory length before freezing.
        return self.warmup_steps if iteration < self.warmup//2 else self.production_steps

    def next_step(self, step, iteration, log_acceptance):
        if iteration >= self.warmup:
            return step
        log_acceptance = float(log_acceptance)
        if math.isnan(log_acceptance) or log_acceptance > 0:
            raise ValueError('Metropolis log acceptance must be nonpositive and not NaN')
        acceptance = math.exp(log_acceptance) if math.isfinite(log_acceptance) else 0.
        return min(self.maximum_step, max(1e-6,
            step * math.exp((acceptance-self.target_acceptance)/math.sqrt(iteration+1))))

    def adaptation_contract(self):
        return dict(method='diminishing Robbins-Monro on log step size',
            equation='s[t+1]=clip(s[t]*exp((min(1,exp(log_alpha[t]))-target)/sqrt(t+1)),1e-6,maximum_step)',
            target_acceptance=self.target_acceptance,
            initial_step=self.initial_step, minimum_step=1e-6,
            maximum_step=self.maximum_step,
            warmup_steps_first_half=self.warmup_steps,
            warmup_steps_second_half=self.production_steps,
            production_step_size='last warmup update; frozen thereafter',
            metric='fixed throughout warmup and retained sampling')


def review_warmup(transitions, *, expected_transitions, final_step,
                  minimum_step=1e-6, maximum_step=0.08):
    """Fail only on invalid/incomplete warmup or a completely stuck chain.

    This is an operational viability check, not a stationarity test. Retained
    draws still require the separate cross-chain scalar and field diagnostics.
    """
    if (not isinstance(expected_transitions, int)
            or isinstance(expected_transitions, bool) or expected_transitions < 1
            or len(transitions) != expected_transitions):
        raise ValueError('warmup review requires the complete declared transition window')
    if (not math.isfinite(final_step)
            or not minimum_step <= final_step <= maximum_step):
        raise ValueError('final warmup step must remain inside declared bounds')
    accepted = 0
    probabilities = []
    for row in transitions:
        log_acceptance = float(row['log_acceptance'])
        if math.isnan(log_acceptance) or log_acceptance > 0:
            raise ValueError('warmup contains invalid Metropolis log acceptance')
        if not isinstance(row.get('accepted'), bool):
            raise ValueError('warmup transition must record boolean acceptance')
        step = float(row['step_size'])
        if not math.isfinite(step) or not minimum_step <= step <= maximum_step:
            raise ValueError('warmup transition step is outside declared bounds')
        probabilities.append(math.exp(log_acceptance)
            if math.isfinite(log_acceptance) else 0.)
        accepted += int(row['accepted'])
    status = ('STOP_ZERO_ACCEPTED_TRANSITIONS' if accepted == 0 else
        'READY_FOR_RETAINED_DIAGNOSTIC_NOT_EQUILIBRATION')
    return dict(status=status, transitions=len(transitions), accepted=accepted,
        rejected=len(transitions)-accepted, acceptance_probability_mean=
            math.fsum(probabilities)/len(probabilities), final_step=float(final_step),
        posterior_equilibration_assessed=False)
