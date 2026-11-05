"""Equal-weight predictive density over retained states, including repeats.

This numerical utility does not establish mixing, calibration or independence.
Pass one JOINT heldout log likelihood per retained post-warmup state. Summing
separately marginalized cell scores would define a different predictive law.
"""
import math
import numpy as np

import jax
import jax.numpy as jnp
from cf4_r2_marked_tracer_jax import sparse_marked_poisson_log_likelihood
from cf4_r2_count_predictive import count_intensity_from_field


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


class JointCountMarkHoldoutPredictive:
    """Jointly score held-out counts and marks on the same retained states.

    ``mark_log_likelihoods`` contains one already-normalized conditional-v6
    mark factor per manifest row for a single state. The class sums those
    factors with that state's count likelihood *before* the posterior
    log-mean-exp. Repeated states and zero-probability states remain in the
    equal-weight mixture. This utility does not certify posterior validity or
    observation-law calibration.
    """

    def __init__(self, *, expected_mark_rows):
        n = int(expected_mark_rows)
        if n < 1:
            raise ValueError('a nonempty certified mark manifest is required')
        self.expected_mark_rows = n
        self.joint = JointPredictiveScore()
        self.count = JointPredictiveScore()

    @staticmethod
    def _log_likelihood(value, label):
        array = np.asarray(value)
        if array.shape != ():
            raise ValueError(f'{label} must be one scalar log likelihood')
        result = float(array)
        if math.isnan(result) or result == math.inf:
            raise ValueError(f'{label} must be finite or negative infinity')
        return result

    def update(self, count_log_likelihood, mark_log_likelihoods):
        count = self._log_likelihood(count_log_likelihood, 'count likelihood')
        marks = np.asarray(mark_log_likelihoods, dtype=np.float64)
        if marks.shape != (self.expected_mark_rows,):
            raise ValueError('mark likelihood vector must match the frozen manifest')
        if np.isnan(marks).any() or np.isposinf(marks).any():
            raise ValueError('mark likelihoods must be finite or negative infinity')
        joint = -math.inf if count == -math.inf or np.isneginf(marks).any() else (
            count + float(np.sum(marks, dtype=np.float64)))
        self.count.update(count)
        self.joint.update(joint)
        return joint

    @property
    def states(self):
        return self.joint.states

    def result(self):
        """Return the joint held-out predictive log density."""
        return self.joint.result()

    def count_result(self):
        """Return the count-only marginal over the same retained states."""
        return self.count.result()

    def mark_given_count_result(self):
        """Return log p(held-out marks | held-out counts) under this mixture."""
        joint = self.result()
        count = self.count_result()
        if count == -math.inf:
            raise ValueError('held-out counts have zero predictive probability under all states')
        return joint - count


class ConditionalCountHoldoutPredictive:
    """Score the frozen held-out count window state-by-state after fitting.

    Construct this only in a post-fit evaluator. It checks that training and
    held-out exposures/occupied keys are disjoint, then contributes one full
    joint held-out Poisson log likelihood per retained state to the equal-
    weight posterior-predictive mixture. Repeated states are retained. This
    utility does not certify that supplied states are posterior draws or that
    the observation law is calibrated.
    """

    def __init__(self, *, intensity_shape, train_keys, train_exposure,
                 heldout_keys, heldout_counts, heldout_exposure):
        shape = tuple(int(value) for value in intensity_shape)
        if len(shape) != 4 or shape[0] != 6 or len(set(shape[1:])) != 1 or shape[1] < 1:
            raise ValueError('intensity shape must be (six populations, cubic grid, cubic grid, cubic grid)')
        size = int(np.prod(shape))
        train_keys = self._integer_vector(train_keys, 'training keys')
        heldout_keys = self._integer_vector(heldout_keys, 'held-out keys')
        counts = self._integer_vector(heldout_counts, 'held-out counts')
        if counts.shape != heldout_keys.shape:
            raise ValueError('held-out keys and sparse counts must align')
        if np.any(counts <= 0):
            raise ValueError('sparse held-out occupied-cell counts must be positive')
        if (np.any(train_keys < 0) or np.any(train_keys >= size)
                or np.any(heldout_keys < 0) or np.any(heldout_keys >= size)):
            raise ValueError('sparse key lies outside the declared intensity shape')
        if (np.unique(train_keys).size != train_keys.size
                or np.unique(heldout_keys).size != heldout_keys.size):
            raise ValueError('sparse training/held-out keys must be unique')

        train_exposure = np.asarray(train_exposure)
        heldout_exposure = np.asarray(heldout_exposure)
        if (train_exposure.dtype != np.bool_ or heldout_exposure.dtype != np.bool_
                or train_exposure.shape != (size,) or heldout_exposure.shape != (size,)):
            raise ValueError('training and held-out exposure masks must be aligned boolean vectors')
        if not np.any(heldout_exposure):
            raise ValueError('held-out exposure must contain at least one observed cell')
        if np.any(train_exposure & heldout_exposure):
            raise ValueError('training and held-out exposure masks overlap')
        if (not np.all(train_exposure[train_keys])
                or not np.all(heldout_exposure[heldout_keys])):
            raise ValueError('occupied keys do not belong to their declared exposure role')
        if np.intersect1d(train_keys, heldout_keys, assume_unique=True).size:
            raise ValueError('training and held-out occupied keys overlap')

        self.intensity_shape = shape
        self.heldout_keys = jnp.asarray(heldout_keys)
        self.heldout_counts = jnp.asarray(counts)
        self.heldout_exposure = jnp.asarray(heldout_exposure)
        self.predictive = JointPredictiveScore()

    @staticmethod
    def _integer_vector(values, label):
        array = np.asarray(values)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f'{label} must be a one-dimensional integer vector')
        return array

    def update(self, intensity):
        """Add one complete held-out count log likelihood for one state."""
        intensity = jnp.asarray(intensity)
        if intensity.shape != self.intensity_shape:
            raise ValueError('state intensity shape differs from the frozen hold-out geometry')
        joint = sparse_marked_poisson_log_likelihood(
            intensity, self.heldout_keys, self.heldout_counts,
            selected_voxel_mask=self.heldout_exposure)
        value = float(jax.device_get(joint))
        self.predictive.update(value)
        return value

    def update_field(self, density, mean_velocity_km_s, tracer_white,
                     source, geometry, *, volume_order,
                     source_chunk_size=2097152):
        """Forward one retained field/nuisance state, then score held-out counts.

        Call only in the declared post-fit evaluation phase. No held-out values
        are loaded or scored by constructing this class or importing this code.
        The supplied state is not thereby certified as a posterior draw.
        """
        intensity = count_intensity_from_field(
            density, mean_velocity_km_s, tracer_white, source, geometry,
            volume_order=volume_order, source_chunk_size=source_chunk_size)
        return self.update(intensity)

    def update_coordinates(self, q, field_from_white, *, native_grid_size,
                           source, geometry, volume_order,
                           source_chunk_size=2097152):
        """Evolve one canonical conditional-v6 draw and score its count holdout.

        The coordinate contract is ``[white_IC(N^3), tracer_white(9),
        population_white(15)]``. Population coordinates affect the CF4 mark
        factor, not this 2M++ count factor. ``field_from_white`` must be the
        unchanged PMWD forward map returning ``(density, mean_velocity)``.
        Invoke only after a valid post-fit draw has been selected.
        """
        n = int(native_grid_size)
        if n < 1:
            raise ValueError('native field grid size must be positive')
        n_ic = n ** 3
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (n_ic + 24,) or not np.isfinite(q).all():
            raise ValueError('canonical conditional-v6 draw must contain N^3+24 finite coordinates')
        density, velocity = field_from_white(jnp.asarray(q[:n_ic]))
        return self.update_field(
            density, velocity, q[n_ic:n_ic + 9], source, geometry,
            volume_order=volume_order, source_chunk_size=source_chunk_size)

    @property
    def states(self):
        return self.predictive.states

    def result(self):
        """Equal-weight posterior-predictive log density over supplied states."""
        return self.predictive.result()
