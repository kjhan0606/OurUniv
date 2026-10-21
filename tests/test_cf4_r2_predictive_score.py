"""Scoring arithmetic only; not survey or posterior validation."""
import math
import unittest
from unittest import mock

import numpy as np

from cf4_r2_predictive_score import (
    ConditionalCountHoldoutPredictive,
    JointCountMarkHoldoutPredictive,
    JointPredictiveScore,
)
import cf4_r2_predictive_score as predictive_score_module


class PredictiveScoreTests(unittest.TestCase):
    def test_equal_weight_mixture_not_average_log_score(self):
        score = JointPredictiveScore()
        for probability in (.2, .8, .8):
            score.update(math.log(probability))
        self.assertAlmostEqual(score.result(), math.log(.6))
        self.assertEqual(score.states, 3)  # Rejection repeat remains a state.

    def test_zero_probability_and_large_negative_scores(self):
        score = JointPredictiveScore()
        score.update(-math.inf)
        score.update(-10000.)
        self.assertAlmostEqual(score.result(), -10000.-math.log(2))
        zero = JointPredictiveScore()
        zero.update(-math.inf)
        self.assertEqual(zero.result(), -math.inf)

    def test_invalid_input_does_not_mutate(self):
        score = JointPredictiveScore()
        with self.assertRaises(ValueError): score.result()
        for invalid in ([1., 2.], math.nan, math.inf):
            with self.assertRaises(ValueError): score.update(invalid)
        self.assertEqual(score.states, 0)


class JointCountMarkHoldoutPredictiveTests(unittest.TestCase):
    def test_sum_factors_per_state_before_equal_weight_mixture(self):
        scorer = JointCountMarkHoldoutPredictive(expected_mark_rows=2)
        scorer.update(math.log(.2), [math.log(.5), math.log(.25)])
        scorer.update(math.log(.8), [math.log(.5), math.log(.25)])
        # Each state's mark likelihood is .125; joint probabilities are .025,.1.
        self.assertAlmostEqual(scorer.result(), math.log(.0625))
        self.assertAlmostEqual(scorer.count_result(), math.log(.5))
        self.assertAlmostEqual(scorer.mark_given_count_result(), math.log(.125))
        self.assertEqual(scorer.states, 2)

    def test_repeats_are_retained_and_zero_probability_state_is_not_dropped(self):
        scorer = JointCountMarkHoldoutPredictive(expected_mark_rows=1)
        scorer.update(-math.inf, [-1.])
        scorer.update(-2., [-3.])
        scorer.update(-2., [-3.])  # a retained rejection repeat still has weight
        self.assertAlmostEqual(scorer.result(), -5.+math.log(2.)-math.log(3.))
        self.assertEqual(scorer.states, 3)

    def test_mark_given_count_is_joint_minus_count_not_mark_only_mixture(self):
        scorer = JointCountMarkHoldoutPredictive(expected_mark_rows=1)
        scorer.update(math.log(.01), [math.log(.9)])
        scorer.update(math.log(.99), [math.log(.1)])
        mark_only = float(np.logaddexp(math.log(.9), math.log(.1))-math.log(2.))
        conditional = scorer.mark_given_count_result()
        self.assertNotAlmostEqual(conditional, mark_only)
        expected = scorer.result()-scorer.count_result()
        self.assertAlmostEqual(conditional, expected)

    def test_invalid_state_does_not_mutate_either_mixture(self):
        scorer = JointCountMarkHoldoutPredictive(expected_mark_rows=2)
        with self.assertRaises(ValueError): scorer.result()
        for count, marks in ((math.nan, [0., 0.]), (0., [0.]),
                             (0., [0., math.inf]), (0., [0., math.nan])):
            with self.assertRaises(ValueError): scorer.update(count, marks)
        self.assertEqual(scorer.states, 0)
        with self.assertRaises(ValueError): scorer.count_result()

    def test_conditional_mark_score_rejects_zero_count_predictive_probability(self):
        scorer = JointCountMarkHoldoutPredictive(expected_mark_rows=1)
        scorer.update(-math.inf, [-1.])
        self.assertEqual(scorer.result(), -math.inf)
        with self.assertRaisesRegex(ValueError, 'zero predictive probability'):
            scorer.mark_given_count_result()
class ConditionalCountHoldoutPredictiveTests(unittest.TestCase):
    @staticmethod
    def scorer():
        train_exposure = np.zeros(48, dtype=bool)
        heldout_exposure = np.zeros(48, dtype=bool)
        train_exposure[[0, 4]] = True
        heldout_exposure[[3, 12]] = True
        return ConditionalCountHoldoutPredictive(
            intensity_shape=(6, 2, 2, 2),
            train_keys=np.array([0, 4], dtype=np.int32),
            train_exposure=train_exposure,
            heldout_keys=np.array([3, 12], dtype=np.int32),
            heldout_counts=np.array([2, 1], dtype=np.int32),
            heldout_exposure=heldout_exposure,
        )

    def test_scores_one_joint_exposure_including_empty_cells_per_state(self):
        scorer = self.scorer()
        intensity = np.ones((6, 2, 2, 2), dtype=np.float32)
        self.assertAlmostEqual(scorer.update(intensity), -2.0-math.log(2.0), places=5)
        scorer.update(intensity)  # A repeated retained chain state remains a state.
        self.assertEqual(scorer.states, 2)
        self.assertAlmostEqual(scorer.result(), -2.0-math.log(2.0), places=5)

    def test_zero_predictive_probability_is_retained_as_negative_infinity(self):
        scorer = self.scorer()
        intensity = np.ones((6, 2, 2, 2), dtype=np.float32)
        intensity.reshape(-1)[[3, 12]] = 0.0
        self.assertEqual(scorer.update(intensity), -math.inf)
        self.assertEqual(scorer.result(), -math.inf)

    def test_empty_occupied_key_list_still_scores_the_exposure_integral(self):
        train_exposure = np.zeros(48, dtype=bool)
        heldout_exposure = np.zeros(48, dtype=bool)
        train_exposure[[0, 4]] = True
        heldout_exposure[[3, 12]] = True
        scorer = ConditionalCountHoldoutPredictive(
            intensity_shape=(6, 2, 2, 2),
            train_keys=np.array([0, 4], dtype=np.int32),
            train_exposure=train_exposure,
            heldout_keys=np.array([], dtype=np.int32),
            heldout_counts=np.array([], dtype=np.int32),
            heldout_exposure=heldout_exposure,
        )
        self.assertAlmostEqual(scorer.update(np.ones((6, 2, 2, 2), dtype=np.float32)),
                               -2.0, places=6)

    def test_rejects_training_heldout_key_or_exposure_overlap(self):
        kwargs = dict(
            intensity_shape=(6, 2, 2, 2),
            train_keys=np.array([0], dtype=np.int32),
            train_exposure=np.isin(np.arange(48), [0, 3]),
            heldout_keys=np.array([3], dtype=np.int32),
            heldout_counts=np.array([1], dtype=np.int32),
            heldout_exposure=np.isin(np.arange(48), [3, 12]),
        )
        with self.assertRaisesRegex(ValueError, 'exposure masks overlap'):
            ConditionalCountHoldoutPredictive(**kwargs)

    def test_rejects_empty_heldout_exposure(self):
        with self.assertRaisesRegex(ValueError, 'at least one observed cell'):
            ConditionalCountHoldoutPredictive(
                intensity_shape=(6, 2, 2, 2),
                train_keys=np.array([], dtype=np.int32),
                train_exposure=np.zeros(48, dtype=bool),
                heldout_keys=np.array([], dtype=np.int32),
                heldout_counts=np.array([], dtype=np.int32),
                heldout_exposure=np.zeros(48, dtype=bool),
            )

    def test_rejects_occupied_key_outside_heldout_role(self):
        kwargs = dict(
            intensity_shape=(6, 2, 2, 2),
            train_keys=np.array([0], dtype=np.int32),
            train_exposure=np.isin(np.arange(48), [0]),
            heldout_keys=np.array([3], dtype=np.int32),
            heldout_counts=np.array([1], dtype=np.int32),
            heldout_exposure=np.isin(np.arange(48), [12]),
        )
        with self.assertRaisesRegex(ValueError, 'declared exposure role'):
            ConditionalCountHoldoutPredictive(**kwargs)

    def test_field_entrypoint_forwards_then_adds_one_joint_state_score(self):
        scorer = self.scorer()
        field = np.zeros((6, 2, 2, 2), dtype=np.float32)
        with mock.patch.object(
                predictive_score_module, 'count_intensity_from_field',
            return_value=np.ones((6, 2, 2, 2), dtype=np.float32)) as forward:
            value = scorer.update_field(
                field[0], np.zeros((3, 2, 2, 2)), np.zeros(9),
                {'positions': np.zeros((8, 3)), 'angular': np.ones((2, 8))},
                {'box_size_cMpc_h': 384.0, 'grid_size': 2}, volume_order=2)
        self.assertAlmostEqual(value, -2.0-math.log(2.0), places=5)
        self.assertEqual(scorer.states, 1)
        forward.assert_called_once()

    def test_canonical_state_entrypoint_evolves_ic_and_selects_tracer_block(self):
        scorer = self.scorer()
        q = np.arange(8 + 24, dtype=np.float64)
        field = (np.ones((2, 2, 2), dtype=np.float32),
                 np.zeros((3, 2, 2, 2), dtype=np.float32))
        source = {'positions': np.zeros((8, 3)), 'angular': np.ones((2, 8))}
        geometry = {'box_size_cMpc_h': 384.0, 'grid_size': 2}
        evolve = mock.Mock(return_value=field)
        with mock.patch.object(
                predictive_score_module, 'count_intensity_from_field',
                return_value=np.ones((6, 2, 2, 2), dtype=np.float32)) as forward:
            value = scorer.update_coordinates(
                q, evolve, native_grid_size=2, source=source, geometry=geometry,
                volume_order=2)
        np.testing.assert_array_equal(np.asarray(evolve.call_args.args[0]), q[:8])
        args, kwargs = forward.call_args
        np.testing.assert_array_equal(args[0], field[0])
        np.testing.assert_array_equal(args[1], field[1])
        np.testing.assert_array_equal(np.asarray(args[2]), q[8:17])
        self.assertIs(args[3], source)
        self.assertIs(args[4], geometry)
        self.assertEqual(kwargs['volume_order'], 2)
        self.assertAlmostEqual(value, -2.0-math.log(2.0), places=5)
        self.assertEqual(scorer.states, 1)

    def test_canonical_state_entrypoint_rejects_wrong_coordinate_count(self):
        scorer = self.scorer()
        with self.assertRaisesRegex(ValueError, r'N\^3\+24'):
            scorer.update_coordinates(
                np.zeros(31), lambda white: None, native_grid_size=2,
                source={}, geometry={}, volume_order=2)


if __name__ == '__main__': unittest.main()
