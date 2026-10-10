"""Equal-weight predictive density over retained states, including repeats.

This numerical utility does not establish mixing, calibration or independence.
Pass one JOINT heldout log likelihood per retained post-warmup state. Summing
separately marginalized cell scores would define a different predictive law.
"""
import math
import numpy as np


class JointPredictiveScore:
    def __init__(self):
        self.states = 0
        self.log_sum = -math.inf

    def update(self, joint_log_likelihood):
        value = np.asarray(joint_log_likelihood)
        if value.shape != ():
            raise ValueError('one joint log likelihood per retained state required')
        value = float(value)
        if math.isnan(value) or value == math.inf:
            raise ValueError('finite log likelihood or negative infinity required')
        # Negative infinity is legitimate zero predictive probability.
        self.log_sum = float(np.logaddexp(self.log_sum, value))
        self.states += 1

    def result(self):
        if not self.states:
            raise ValueError('no retained states')
        return self.log_sum - math.log(self.states)
