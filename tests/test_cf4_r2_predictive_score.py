"""Scoring arithmetic only; not survey or posterior validation."""
import math
import unittest
from cf4_r2_predictive_score import JointPredictiveScore


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


if __name__ == '__main__': unittest.main()
