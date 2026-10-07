import unittest

import numpy as np

from cf4_r2_conditional_map import assemble_conditional_objective, conditional_target_terms
from cf4_r2_conditional_optimizer_diagnostic import (
    block_gradient_summary, diagnostic_status, finite_difference_agreement,
    finite_difference_status, gradient_agreement, nuisance_block_status, nuisance_scale,
    coordinate_line_action, full_gradient_record_status, pop9_line_status, pop9_newton_step,
    tracer2_line_status, tracer_pair_newton_step, tracer_pair_status,
    pop3_line_status, pop0_line_status, pop_pair_status, pop2_line_status,
    pop12_line_status, pop14_line_status,
    reproduction_failures,
    scaled_nuisance,
    tracer0_line_status, tracer0_secant,
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

    def test_finite_difference_completion_does_not_invent_a_pass(self):
        self.assertEqual(finite_difference_status(
            reproduction_passed=False, finite_difference='passed'),
            'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED')
        self.assertEqual(finite_difference_status(
            reproduction_passed=True, finite_difference='failed'),
            'CONDITIONAL_OPTIMIZER_FD_FAILED')
        self.assertEqual(finite_difference_status(
            reproduction_passed=True, finite_difference=None),
            'CONDITIONAL_OPTIMIZER_DIAGNOSTIC_INCOMPLETE_BUDGET')
        self.assertEqual(finite_difference_status(
            reproduction_passed=True, finite_difference='passed'),
            'CONDITIONAL_OPTIMIZER_FD_PASSED')

    def test_nuisance_scale_preserves_the_saved_point_at_zero_step(self):
        gradient = np.array([13524., -2., 0.1] + [0.] * 21)
        scale = nuisance_scale(gradient)
        self.assertAlmostEqual(scale[0], 1. / 13524.)
        self.assertAlmostEqual(scale[2], 1.)
        origin = np.linspace(-1., 1., 24)
        np.testing.assert_allclose(scaled_nuisance(origin, scale, np.zeros(24)), origin)
        self.assertEqual(nuisance_block_status(1000., 100., -10., -9.),
                         'CONDITIONAL_NUISANCE_BLOCK_AMPLITUDE_REDUCED')
        self.assertEqual(nuisance_block_status(1000., 100., -10., -11.),
                         'CONDITIONAL_NUISANCE_BLOCK_NOT_REDUCED')
        self.assertEqual(nuisance_block_status(1000., 200., -10., -9.),
                         'CONDITIONAL_NUISANCE_BLOCK_NOT_REDUCED')

    def test_tracer0_line_requires_both_improvement_and_a_tenfold_drop(self):
        self.assertEqual(tracer0_line_status(10., 9., 1000., 50., -10., -9.),
                         'CONDITIONAL_TRACER0_LINE_AMPLITUDE_REDUCED')
        self.assertEqual(tracer0_line_status(10., 9., 1000., 500., -10., -9.),
                         'CONDITIONAL_TRACER0_LINE_IMPROVED')
        self.assertEqual(tracer0_line_status(10., 10., 1000., 50., -10., -9.),
                         'CONDITIONAL_TRACER0_LINE_NO_IMPROVEMENT')
        self.assertEqual(tracer0_line_status(10., 9., 1000., 50., -10., -11.),
                         'CONDITIONAL_TRACER0_LINE_IMPROVED')

    def test_secant_stays_inside_the_sign_bracket(self):
        theta = tracer0_secant(0.2993783248581184, 13524.69411623714,
                               0.1993783248581184, -6010.1391)
        self.assertGreater(theta, 0.1993783248581184)
        self.assertLess(theta, 0.2993783248581184)
        with self.assertRaisesRegex(ValueError, 'opposite derivative'):
            tracer0_secant(0.3, 1., 0.2, 1.)

    def test_a_sign_change_stops_before_another_outward_step(self):
        self.assertEqual(coordinate_line_action(True, True, False, False), 'stop')
        self.assertEqual(coordinate_line_action(True, False, False, False), 'continue')
        self.assertEqual(coordinate_line_action(False, False, False, False), 'midpoint')
        self.assertEqual(coordinate_line_action(False, False, False, True), 'stop')
        self.assertEqual(coordinate_line_action(True, False, True, False), 'stop')

    def test_population_coordinate_gate_uses_the_fp_term(self):
        self.assertEqual(pop9_line_status(10., 9., 700., 50., 1., 1.1),
                         'CONDITIONAL_POP9_LINE_REDUCED')
        self.assertEqual(pop9_line_status(10., 9., 700., 200., 1., 1.1),
                         'CONDITIONAL_POP9_LINE_IMPROVED')
        self.assertEqual(pop9_line_status(10., 9., 700., 50., 1., 0.),
                         'CONDITIONAL_POP9_LINE_IMPROVED')

    def test_population_newton_step_is_capped_and_stays_downslope(self):
        theta = pop9_newton_step(-0.2212557363, 627.94236, -0.3212557363, 525.56259)
        self.assertLess(theta, -0.3212557363)
        self.assertGreater(theta, -1.3212557363)
        self.assertAlmostEqual(pop9_newton_step(0.0, 21.0, -0.1, 20.0), -1.1)
        with self.assertRaisesRegex(ValueError, 'does not fall'):
            pop9_newton_step(-0.2, 100., -0.3, 150.)
        with self.assertRaisesRegex(ValueError, 'cap'):
            pop9_newton_step(0.0, 2.0, -0.1, 1.0, max_abs=0.0)

    def test_population_secant_stays_inside_the_newton_bracket(self):
        positive, negative = -0.32125573632738025, -0.8346018455909067
        theta = tracer0_secant(positive, 525.5625891982911, negative, -742.5802154827692)
        self.assertLess(theta, positive)
        self.assertGreater(theta, negative)
        self.assertGreater(theta, 0.5 * (positive + negative))

    def test_second_population_secant_stays_closer_to_the_smaller_derivative(self):
        positive, negative = -0.5340042606603805, -0.8346018455909067
        theta = tracer0_secant(positive, 179.299297990836, negative, -742.5802154827693)
        self.assertLess(theta, positive)
        self.assertGreater(theta, negative)
        self.assertGreater(theta, 0.5 * (positive + negative))

    def test_tracer0_revisit_secant_stays_closer_to_the_smaller_derivative(self):
        negative, positive = 0.23014459265495463, 0.33014459265495466
        theta = tracer0_secant(positive, 13714.015001370202, negative, -5855.130693568598)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(theta, 0.5 * (negative + positive))

    def test_tracer_pair_newton_uses_the_recorded_columns(self):
        current = (0.26006480881995014, -404.55082315761325, 2141.346117378757)
        earlier = (0.23014459265495463, -5855.130693568599, 1578.4178189668382)
        column = (0.9017512357648408, -5855.1306935686, 1578.4178189668382)
        column_prev = (1.0017512357648408, -4016.864850526628, 3363.161778842157)
        arguments = (
            current[0], current[1], earlier[0], earlier[1], current[2], earlier[2],
            column[0], column[1], column[2], column_prev[0], column_prev[1], column_prev[2])
        tracer0, tracer2 = tracer_pair_newton_step(*arguments)
        self.assertGreater(tracer0, 0.260065)
        self.assertLess(tracer0, 0.280065)
        self.assertGreater(tracer2, 0.751751)
        self.assertLess(tracer2, 0.901751)
        capped0, capped2 = tracer_pair_newton_step(*arguments, max_abs=0.01)
        self.assertAlmostEqual(abs(capped2 - column[0]), 0.01)
        self.assertLess(abs(capped0 - current[0]), 0.01)
        with self.assertRaisesRegex(ValueError, 'do not agree'):
            tracer_pair_newton_step(
                1., -1., 0., -2., 1., 0.,
                1., 0., 1., 0., 10., 0.)

    def test_tracer_pair_status_requires_both_gradients(self):
        self.assertEqual(tracer_pair_status(10., 9., 100., 5., 200., 10., 3., 3.),
                         'CONDITIONAL_TRACER_PAIR_REDUCED')
        self.assertEqual(tracer_pair_status(10., 9., 100., 50., 200., 10., 3., 3.),
                         'CONDITIONAL_TRACER_PAIR_IMPROVED')
        self.assertEqual(tracer_pair_status(10., 9., 100., 5., 200., 10., 3., 2.),
                         'CONDITIONAL_TRACER_PAIR_IMPROVED')
        self.assertEqual(tracer_pair_status(10., 10., 100., 5., 200., 10., 3., 4.),
                         'CONDITIONAL_TRACER_PAIR_NO_IMPROVEMENT')

    def test_population3_secant_stays_inside_the_sign_bracket(self):
        positive, negative = -0.3520684535369951, -0.4520684535369951
        theta = tracer0_secant(positive, 413.43912366672095, negative, -356.21633707957614)
        self.assertLess(theta, positive)
        self.assertGreater(theta, negative)
        self.assertLess(theta, 0.5 * (positive + negative))

    def test_population_pair_newton_uses_the_recorded_columns(self):
        current0, failed0 = -0.016861714226106356, -0.11686171422610636
        current3, earlier3 = -0.40578588227658063, -0.15206845353699508
        pop0, pop3 = tracer_pair_newton_step(
            current0, 1111.6576005525103, failed0, -3774.87620888411,
            0.7789983915350208, 988.8199200151512,
            current3, 1111.6576005525098, 0.7789983915350919,
            earlier3, -1352.885111834647, 1913.0246939274975)
        self.assertLess(pop0, current0)
        self.assertGreater(pop0, failed0)
        self.assertLess(pop3, current3)
        self.assertGreater(pop3, current3 - 0.1)
        self.assertEqual(pop_pair_status(10., 9., 1100., 100., 50., 191., 3., 3.),
                         'CONDITIONAL_POP_PAIR_REDUCED')
        self.assertEqual(pop_pair_status(10., 9., 1100., 100., 200., 191., 3., 3.),
                         'CONDITIONAL_POP_PAIR_IMPROVED')
        self.assertEqual(pop_pair_status(10., 10., 1100., 100., 50., 191., 3., 4.),
                         'CONDITIONAL_POP_PAIR_NO_IMPROVEMENT')

    def test_population2_secant_stays_inside_the_sign_bracket(self):
        positive, negative = 0.07370632659783677, -0.026293673402163237
        theta = tracer0_secant(positive, 1189.844649145757, negative, -830.9757089189494)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))

    def test_population14_gate_uses_count_plus_fp(self):
        self.assertEqual(pop14_line_status(10., 9., 1058., 100., 3., 3.1),
                         'CONDITIONAL_POP14_LINE_REDUCED')
        self.assertEqual(pop14_line_status(10., 9., 1058., 110., 3., 3.1),
                         'CONDITIONAL_POP14_LINE_IMPROVED')
        self.assertEqual(pop14_line_status(10., 9., 1058., 100., 3., 2.),
                         'CONDITIONAL_POP14_LINE_IMPROVED')
        self.assertEqual(pop14_line_status(10., 10., 1058., 100., 3., 4.),
                         'CONDITIONAL_POP14_LINE_NO_IMPROVEMENT')

    def test_population12_gate_uses_count_plus_fp(self):
        self.assertEqual(pop12_line_status(10., 9., 828., 80., 3., 3.1),
                         'CONDITIONAL_POP12_LINE_REDUCED')
        self.assertEqual(pop12_line_status(10., 9., 828., 90., 3., 3.1),
                         'CONDITIONAL_POP12_LINE_IMPROVED')
        self.assertEqual(pop12_line_status(10., 9., 828., 80., 3., 2.),
                         'CONDITIONAL_POP12_LINE_IMPROVED')
        self.assertEqual(pop12_line_status(10., 10., 828., 80., 3., 4.),
                         'CONDITIONAL_POP12_LINE_NO_IMPROVEMENT')

    def test_population2_gate_uses_count_plus_fp(self):
        self.assertEqual(pop2_line_status(10., 9., 830., 80., 3., 3.1),
                         'CONDITIONAL_POP2_LINE_REDUCED')
        self.assertEqual(pop2_line_status(10., 9., 830., 90., 3., 3.1),
                         'CONDITIONAL_POP2_LINE_IMPROVED')
        self.assertEqual(pop2_line_status(10., 9., 830., 80., 3., 2.),
                         'CONDITIONAL_POP2_LINE_IMPROVED')
        self.assertEqual(pop2_line_status(10., 10., 830., 80., 3., 4.),
                         'CONDITIONAL_POP2_LINE_NO_IMPROVEMENT')

    def test_population0_gate_uses_count_plus_fp(self):
        self.assertEqual(pop0_line_status(10., 9., 1100., 100., 3., 3.1),
                         'CONDITIONAL_POP0_LINE_REDUCED')
        self.assertEqual(pop0_line_status(10., 9., 1100., 200., 3., 3.1),
                         'CONDITIONAL_POP0_LINE_IMPROVED')
        self.assertEqual(pop0_line_status(10., 9., 1100., 100., 3., 2.),
                         'CONDITIONAL_POP0_LINE_IMPROVED')
        self.assertEqual(pop0_line_status(10., 10., 1100., 100., 3., 4.),
                         'CONDITIONAL_POP0_LINE_NO_IMPROVEMENT')

    def test_population3_gate_uses_count_plus_fp(self):
        self.assertEqual(pop3_line_status(10., 9., 1900., 100., 3., 3.1),
                         'CONDITIONAL_POP3_LINE_REDUCED')
        self.assertEqual(pop3_line_status(10., 9., 1900., 200., 3., 3.1),
                         'CONDITIONAL_POP3_LINE_IMPROVED')
        self.assertEqual(pop3_line_status(10., 9., 1900., 100., 3., 2.),
                         'CONDITIONAL_POP3_LINE_IMPROVED')
        self.assertEqual(pop3_line_status(10., 10., 1900., 100., 3., 4.),
                         'CONDITIONAL_POP3_LINE_NO_IMPROVEMENT')

    def test_tracer2_gate_uses_the_combined_likelihood(self):
        self.assertEqual(tracer2_line_status(10., 9., 7000., 500., 3., 3.1),
                         'CONDITIONAL_TRACER2_LINE_REDUCED')
        self.assertEqual(tracer2_line_status(10., 9., 7000., 2000., 3., 3.1),
                         'CONDITIONAL_TRACER2_LINE_IMPROVED')
        self.assertEqual(tracer2_line_status(10., 9., 7000., 500., 3., 2.),
                         'CONDITIONAL_TRACER2_LINE_IMPROVED')
        self.assertEqual(tracer2_line_status(10., 10., 7000., 500., 3., 4.),
                         'CONDITIONAL_TRACER2_LINE_NO_IMPROVEMENT')

    def test_full_gradient_record_requires_the_reduced_point(self):
        self.assertEqual(full_gradient_record_status(True, True, True),
                         'CONDITIONAL_FULL_GRADIENT_RECORDED')
        self.assertEqual(full_gradient_record_status(True, False, True),
                         'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED')
        self.assertEqual(full_gradient_record_status(True, True, True, False),
                         'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED')


if __name__ == '__main__':
    unittest.main()
