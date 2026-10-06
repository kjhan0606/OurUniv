import unittest

import numpy as np

from cf4_r2_conditional_map import assemble_conditional_objective, conditional_target_terms
from cf4_r2_conditional_optimizer_diagnostic import (
    block_gradient_summary, diagnostic_status, finite_difference_agreement, gradient_agreement,
    reproduction_failures,
)


class ConditionalObjectiveAlgebraTest(unittest.TestCase):
    def test_standard_normal_prior_minus_score_gradient(self):
        q = np.zeros(26, dtype=np.float64)
        q[:4] = (0.2, -0.4, 0.5, 1.5)
        score_gradient = np.zeros(26, dtype=np.float64)
        score_gradient[:4] = (0.0, 1.0, -2.0, 0.25)
        value, detail = conditional_target_terms(
            q, score=3.0, components=np.array([2.0, 1.0]), support_info={'cells': 4}, n_ic=2)
        self.assertAlmostEqual(value, 0.5 * float(q @ q) - 3.0)
        self.assertAlmostEqual(detail['count_log_likelihood'], 2.0)
        self.assertAlmostEqual(detail['conditional_FP_log_likelihood'], 1.0)
        objective, gradient, assembled = assemble_conditional_objective(
            q, 3.0, np.array([2.0, 1.0]), score_gradient[:2], score_gradient[2:11],
            score_gradient[11:], {'cells': 4}, n_ic=2)
        self.assertAlmostEqual(objective, value)
        np.testing.assert_allclose(gradient, q - score_gradient)
        self.assertAlmostEqual(assembled['total_negative_log_target'], value)

    def test_rejects_a_mismatched_block(self):
        q = np.zeros(4)
        with self.assertRaisesRegex(ValueError, 'canonical coordinates'):
            conditional_target_terms(q, 0.0, np.array([0.0]), {}, n_ic=2)


class ConditionalDiagnosticGateTest(unittest.TestCase):
    def saved(self):
        return dict(objective=10.0, IC_prior_NLL=8.0, tracer_nuisance_prior_NLL=0.0,
                    population_nuisance_prior_NLL=0.0, count_log_likelihood=-1.5,
                    conditional_FP_log_likelihood=0.25,
                    total_negative_log_target=10.0)

    def test_matching_replay_passes(self):
        self.assertEqual(reproduction_failures(self.saved(), self.saved()), [])

    def test_term_error_above_one_thousandth_fails_closed(self):
        replay = self.saved()
        replay['count_log_likelihood'] = -1.5 + 1.1e-3
        failures = reproduction_failures(self.saved(), replay)
        self.assertEqual([item['term'] for item in failures], ['count_log_likelihood'])

    def test_objective_relative_error_fails_even_when_terms_pass(self):
        saved = self.saved()
        replay = self.saved()
        replay['total_negative_log_target'] = 10.0 + 2e-7
        failures = reproduction_failures(saved, replay)
        self.assertEqual([item['kind'] for item in failures], ['objective'])

    def test_infinity_norm_location_names_its_block(self):
        gradient = np.zeros(26)
        gradient[2] = -5.0
        summary = block_gradient_summary(gradient, n_ic=2)
        self.assertEqual(summary['infinity_norm_location']['block'], 'tracer')
        self.assertEqual(summary['infinity_norm_location']['index_in_block'], 0)
        self.assertAlmostEqual(summary['tracer']['inf'], 5.0)
        self.assertAlmostEqual(summary['IC']['rms'], 0.0)

    def test_finite_difference_uses_the_declared_relative_limit(self):
        passed = finite_difference_agreement(10.0, 10.001, 10.0, 1e-4)
        failed = finite_difference_agreement(10.0, 10.003, 10.0, 1e-4)
        self.assertTrue(passed['passed'])
        self.assertFalse(failed['passed'])
        self.assertAlmostEqual(passed['relative_error'], 0.0)

    def test_component_sum_must_rebuild_the_joint_gradient(self):
        reference = np.array([1.0, -2.0, 0.5, 0.25])
        self.assertTrue(gradient_agreement(reference, reference)['passed'])
        shifted = reference + np.array([0.1, 0.0, 0.0, 0.0])
        self.assertFalse(gradient_agreement(shifted, reference)['passed'])

    def test_only_a_complete_attribution_has_the_complete_status(self):
        self.assertEqual(diagnostic_status(
            reproduction_passed=False, component_split=None, finite_difference=None,
            budget_stopped_early=False), 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED')
        self.assertEqual(diagnostic_status(
            reproduction_passed=True, component_split='mismatch', finite_difference='passed',
            budget_stopped_early=False), 'CONDITIONAL_OPTIMIZER_COMPONENT_SPLIT_MISMATCH')
        self.assertEqual(diagnostic_status(
            reproduction_passed=True, component_split='passed', finite_difference='failed',
            budget_stopped_early=False), 'CONDITIONAL_OPTIMIZER_FD_FAILED')
        self.assertEqual(diagnostic_status(
            reproduction_passed=True, component_split='passed', finite_difference=None,
            budget_stopped_early=True), 'CONDITIONAL_OPTIMIZER_DIAGNOSTIC_INCOMPLETE_BUDGET')
        self.assertEqual(diagnostic_status(
            reproduction_passed=True, component_split='skipped', finite_difference='passed',
            budget_stopped_early=False), 'CONDITIONAL_OPTIMIZER_DIAGNOSTIC_COMPLETE_BLOCK_ONLY')
        self.assertEqual(diagnostic_status(
            reproduction_passed=True, component_split='passed', finite_difference='passed',
            budget_stopped_early=False), 'CONDITIONAL_OPTIMIZER_DIAGNOSTIC_COMPLETE')


if __name__ == '__main__':
    unittest.main()
