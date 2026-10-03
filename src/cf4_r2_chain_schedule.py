"""Declared finite warmup/production schedule; no production adaptation."""
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
        acceptance = math.exp(min(0., log_acceptance))
        return min(self.maximum_step, max(1e-6,
            step * math.exp((acceptance-self.target_acceptance)/math.sqrt(iteration+1))))
