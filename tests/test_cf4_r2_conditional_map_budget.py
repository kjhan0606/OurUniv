import time
import unittest

import numpy as np

from cf4_r2_conditional_map import BudgetedObjective, EvaluationBudgetStop
from cf4_r2_raw_volume_target import checked_padded_support_width


class ConditionalMapBudgetTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
