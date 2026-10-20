import time
import unittest
import json

import numpy as np

from cf4_r2_conditional_map import (
    BudgetedObjective,
    BudgetedScoreObjective,
    EvaluationBudgetStop,
)
from cf4_affine_objective import AffineObjective
from cf4_r2_raw_volume_target import checked_padded_support_width


class ConditionalMapBudgetTest(unittest.TestCase):
    def test_affine_replay_is_cached_and_trials_obey_same_hard_budget(self):
        origin = np.array([1., .02])
        objective = BudgetedObjective(
            lambda q: (float(q @ q), 2 * q, {}),
            max_evaluations=2, deadline=time.monotonic() + 30)
        value, _ = objective(origin)  # Replay consumes the first call.
        scaled = AffineObjective(objective, origin, [1., .001], value)
        self.assertEqual(scaled(np.zeros(2))[0], 0.)
        self.assertEqual(len(objective.records), 1)
        scaled(np.ones(2))  # A trial consumes the second call.
        with self.assertRaises(EvaluationBudgetStop):
            scaled(-np.ones(2))
        self.assertEqual(len(objective.records), 2)

    def test_support_workspace_ceiling_raises_without_truncating(self):
        self.assertEqual(checked_padded_support_width([32768]), 32768)
        with self.assertRaisesRegex(MemoryError, 'candidate support was not truncated'):
            checked_padded_support_width([32769])
        self.assertEqual(
            checked_padded_support_width([32769], max_cells=65536), 32832)

    def test_duplicate_request_is_cached_and_budget_counts_trials(self):
        calls = []

        def quadratic(x):
            calls.append(x.copy())
            return float(x @ x), 2.0 * x, {'component': float(x @ x)}

        objective = BudgetedObjective(
            quadratic, max_evaluations=2, deadline=time.monotonic() + 30)
        value, gradient = objective(np.array([2.0]))
        duplicate_value, duplicate_gradient = objective(np.array([2.0]))
        self.assertEqual(value, duplicate_value)
        np.testing.assert_array_equal(gradient, duplicate_gradient)
        objective(np.array([1.0]))
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(objective.records), 2)
        self.assertEqual(objective.best['objective'], 1.0)
        with self.assertRaises(EvaluationBudgetStop):
            objective(np.array([0.5]))

    def test_deadline_stops_before_calling_target(self):
        calls = []
        objective = BudgetedObjective(
            lambda x: calls.append(x) or (0.0, np.zeros_like(x), {}),
            max_evaluations=4, deadline=time.monotonic() - 1)
        with self.assertRaises(EvaluationBudgetStop):
            objective(np.array([1.0]))
        self.assertEqual(calls, [])

    def test_evaluation_callback_receives_candidate_record_and_best(self):
        callbacks = []
        objective = BudgetedObjective(
            lambda x: (float(x @ x), 2.0 * x, {'component': float(x @ x)}),
            max_evaluations=1, deadline=time.monotonic() + 30,
            on_evaluation=lambda x, row, best: callbacks.append(
                (x.copy(), row['evaluation'], best['objective'])))
        objective(np.array([3.0]))
        self.assertEqual(len(callbacks), 1)
        np.testing.assert_array_equal(callbacks[0][0], [3.0])
        self.assertEqual(callbacks[0][1:], (1, 9.0))
        self.assertEqual(len(objective.records), 1)

    def test_scalar_score_appends_without_mutating_gradient_cache(self):
        callbacks = []
        objective = BudgetedObjective(
            lambda q: (float(q @ q), 2.0 * q, {'kind': 'gradient'}),
            max_evaluations=3, deadline=time.monotonic() + 30,
            on_evaluation=lambda x, row, best: callbacks.append(
                (row['evaluation'], row.get('gradient_evaluated'), best is None)))
        anchor = np.array([1.0, -2.0])
        objective(anchor)
        gradient_x = objective._last_x.copy()
        gradient_result = objective._last_result
        score_objective = BudgetedScoreObjective(
            lambda q: (float(q @ q), -0.5 * float(q @ q), {'kind': 'scalar'}),
            objective)

        candidate = np.array([0.5, 0.25])
        score = score_objective(candidate)
        self.assertEqual(score, -0.5 * float(candidate @ candidate))
        self.assertEqual(len(objective.records), 2)
        self.assertFalse(objective.records[-1]['gradient_evaluated'])
        self.assertTrue(objective.records[-1]['scalar_score_evaluator'])
        np.testing.assert_array_equal(objective._last_x, gradient_x)
        self.assertIs(objective._last_result, gradient_result)
        self.assertEqual(callbacks[-1], (2, False, True))
        self.assertEqual(score_objective(candidate), score)  # Cached exact duplicate.
        self.assertEqual(len(objective.records), 2)

    def test_scalar_score_obeys_shared_evaluation_budget(self):
        objective = BudgetedObjective(
            lambda q: (float(q @ q), 2.0 * q, {}),
            max_evaluations=1, deadline=time.monotonic() + 30)
        objective(np.array([1.0]))
        score_objective = BudgetedScoreObjective(
            lambda q: (float(q @ q), float(q[0]), {}), objective)
        with self.assertRaises(EvaluationBudgetStop):
            score_objective(np.array([2.0]))
        self.assertEqual(len(objective.records), 1)

    def test_true_zero_likelihood_is_recorded_as_json_safe_negative_infinity(self):
        objective = BudgetedObjective(
            lambda q: (float(q @ q), 2.0 * q, {}),
            max_evaluations=2, deadline=time.monotonic() + 30)
        objective(np.array([1.0]))
        score_objective = BudgetedScoreObjective(
            lambda q: (np.inf, -np.inf,
                       {'count_log_likelihood': -np.inf,
                        'conditional_FP_log_likelihood': 0.0,
                        'total_negative_log_target': np.inf}),
            objective)
        self.assertEqual(score_objective(np.array([2.0])), -np.inf)
        row = objective.records[-1]
        self.assertEqual(row['objective'], 'Infinity')
        self.assertEqual(row['count_log_likelihood'], '-Infinity')
        self.assertEqual(json.loads(json.dumps(row, allow_nan=False))['objective'], 'Infinity')


if __name__ == '__main__':
    unittest.main()
