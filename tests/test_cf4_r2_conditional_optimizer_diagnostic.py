import unittest

import numpy as np

from cf4_r2_conditional_map import assemble_conditional_objective, conditional_target_terms
from cf4_r2_conditional_optimizer_diagnostic import (
    block_gradient_summary, diagnostic_status, finite_difference_agreement,
    finite_difference_status, gradient_agreement, nuisance_block_status, nuisance_scale,
    coordinate_line_action, full_gradient_record_status, pop9_line_status, pop9_newton_step,
    tracer2_line_status, tracer3_line_status, tracer4_line_status, tracer4_revisit_status, tracer_pair_newton_step,
    tracer_pair_status,
    pop3_line_status, pop0_line_status, pop_pair_status, pop2_line_status, pop2_revisit_step, pop2_revisit_status,
    pop12_line_status, pop14_line_status, pop6_line_status, pop12_revisit_status,
    pop12_revisit2_status, pop11_line_status, pop11_revisit_status, pop11_revisit2_status, pop11_revisit3_status, pop11_revisit4_status, pop13_line_status, pop10_line_status, pop1_line_status, pop4_line_status, pop4_revisit_status, pop14_revisit_status,
    pop9_revisit_status, pop14_revisit2_status, pop14_revisit3_status,
    pop14_revisit4_status, pop14_revisit5_status, pop14_revisit6_status,
    pop0_revisit_status, pop0_revisit2_status, pop0_revisit3_status, pop0_revisit4_status, pop0_revisit5_status, pop0_revisit6_status, pop0_revisit7_status, pop0_revisit8_status,
    pop3_revisit_status, pop3_revisit2_status, pop3_revisit3_status, pop3_revisit4_status,
    reproduction_failures,
    scaled_nuisance,
    tracer0_line_status, tracer0_revisit2_status, tracer0_revisit3_status, tracer0_revisit4_status, tracer0_secant,
    tracer6_line_status, tracer6_revisit_first_step, tracer6_revisit_status,
    tracer6_revisit2_first_step, tracer6_revisit2_continuation_step, tracer6_revisit2_status,
    tracer6_support_step_size, tracer6_support_step_status, tracer6_revisit3_status,
    tracer6_support_step2_status, tracer6_revisit4_status, tracer6_revisit5_status, tracer6_support_step4_status, tracer6_revisit6_status, tracer6_support_step5_status, tracer6_revisit7_status, tracer6_support_step6_status,
    tracer6_support_step3_status, tracer6_revisit8_status, tracer6_support_step7_status,
    pop5_line_step, pop5_line_status, pop5_revisit_status, pop5_revisit2_status, pop5_revisit3_status,
    pop0_revisit9_step, pop0_revisit9_status,
    pop0_revisit10_step, pop0_revisit10_status, pop0_revisit11_status, pop0_revisit12_status,
    pop0_revisit13_status, pop0_revisit14_status,
    pop0_revisit15_step, pop0_revisit15_status, pop0_revisit16_status,
    pop3_revisit5_step, pop3_revisit5_status, pop3_revisit6_status, pop3_revisit7_status,
    pop3_revisit8_status,
    pop13_revisit_step, pop13_revisit_status,
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

    def test_population0_revisit_gate_uses_count_plus_fp(self):
        self.assertEqual(pop0_revisit_status(10., 9., 1732., 173., 3., 3.1),
                         'CONDITIONAL_POP0_REVISIT_REDUCED')
        self.assertEqual(pop0_revisit_status(10., 9., 1732., 174., 3., 3.1),
                         'CONDITIONAL_POP0_REVISIT_IMPROVED')
        self.assertEqual(pop0_revisit_status(10., 9., 1732., 173., 3., 2.),
                         'CONDITIONAL_POP0_REVISIT_IMPROVED')
        self.assertEqual(pop0_revisit_status(10., 10., 1732., 173., 3., 4.),
                         'CONDITIONAL_POP0_REVISIT_NO_IMPROVEMENT')

    def test_population0_revisit4_gate_uses_count_plus_fp(self):
        self.assertEqual(pop0_revisit4_status(10., 9., 1321.908, 132.190, 3., 3.1),
                         'CONDITIONAL_POP0_REVISIT4_REDUCED')
        self.assertEqual(pop0_revisit4_status(10., 9., 1321.908, 132.191, 3., 3.1),
                         'CONDITIONAL_POP0_REVISIT4_IMPROVED')
        self.assertEqual(pop0_revisit4_status(10., 9., 1321.908, 132.190, 3., 2.),
                         'CONDITIONAL_POP0_REVISIT4_IMPROVED')
        self.assertEqual(pop0_revisit4_status(10., 10., 1321.908, 132.190, 3., 4.),
                         'CONDITIONAL_POP0_REVISIT4_NO_IMPROVEMENT')

    def test_pop0_revisit5_gate_uses_count_plus_fp(self):
        self.assertEqual(pop0_revisit5_status(10., 9., 872.695798833002, 87.2695, 3., 3.1),
                         'CONDITIONAL_POP0_REVISIT5_REDUCED')
        self.assertEqual(pop0_revisit5_status(10., 9., 872.695798833002, 87.2696, 3., 3.1),
                         'CONDITIONAL_POP0_REVISIT5_IMPROVED')
        self.assertEqual(pop0_revisit5_status(10., 9., 872.695798833002, 87.2695, 3., 2.),
                         'CONDITIONAL_POP0_REVISIT5_IMPROVED')
        self.assertEqual(pop0_revisit5_status(10., 10., 872.695798833002, 87.2695, 3., 4.),
                         'CONDITIONAL_POP0_REVISIT5_NO_IMPROVEMENT')


    def test_pop0_revisit6_gate_uses_count_plus_fp(self):
        self.assertEqual(
            pop0_revisit6_status(10., 9., 842.0593085668357, 84.2059, 3., 3.1),
            'CONDITIONAL_POP0_REVISIT6_REDUCED')
        self.assertEqual(
            pop0_revisit6_status(10., 9., 842.0593085668357, 84.2060, 3., 3.1),
            'CONDITIONAL_POP0_REVISIT6_IMPROVED')
        self.assertEqual(
            pop0_revisit6_status(10., 9., 842.0593085668357, 84.2059, 3., 2.),
            'CONDITIONAL_POP0_REVISIT6_IMPROVED')
        self.assertEqual(
            pop0_revisit6_status(10., 10., 842.0593085668357, 84.2059, 3., 4.),
            'CONDITIONAL_POP0_REVISIT6_NO_IMPROVEMENT')


    def test_pop0_revisit7_gate_uses_count_plus_fp(self):
        self.assertEqual(
            pop0_revisit7_status(10., 9., 991.4788710836873, 99.1478, 3., 3.1),
            'CONDITIONAL_POP0_REVISIT7_REDUCED')
        self.assertEqual(
            pop0_revisit7_status(10., 9., 991.4788710836873, 99.1479, 3., 3.1),
            'CONDITIONAL_POP0_REVISIT7_IMPROVED')
        self.assertEqual(
            pop0_revisit7_status(10., 9., 991.4788710836873, 99.1478, 3., 2.),
            'CONDITIONAL_POP0_REVISIT7_IMPROVED')
        self.assertEqual(
            pop0_revisit7_status(10., 10., 991.4788710836873, 99.1478, 3., 4.),
            'CONDITIONAL_POP0_REVISIT7_NO_IMPROVEMENT')


    def test_pop0_revisit7_rejected_step_does_not_improve(self):
        self.assertEqual(
            pop0_revisit7_status(8208651.1227584565, 8224802.805308412,
                                 991.4788707891084, 324024.51306475233, 1., 0.),
            'CONDITIONAL_POP0_REVISIT7_NO_IMPROVEMENT')

    def test_pop0_revisit7_secant_stays_inside_the_sign_bracket(self):
        positive, negative = 0.06371840835870071, -0.0362815916412993
        theta = tracer0_secant(positive, 324024.51306475233, negative, -991.4788707891084)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))
        self.assertAlmostEqual(theta, -0.03597653623006103)
        with self.assertRaises(ValueError):
            tracer0_secant(negative, -991.4788707891084, positive, 324024.51306475233)

    def test_pop0_revisit7_secant_meets_the_accepted_end_gate(self):
        initial = -142420.23671111895 + 5915.757788626992
        best = -142420.23671111895 + 5915.909004574992
        self.assertEqual(
            pop0_revisit7_status(8208651.1227584565, 8208650.971531487,
                                 991.4788710836744, 0.006731083806192473,
                                 initial, best),
            'CONDITIONAL_POP0_REVISIT7_REDUCED')


    def test_pop0_revisit6_rejected_step_does_not_improve(self):
        self.assertEqual(
            pop0_revisit6_status(8208706.210870936, 8219699.266073297,
                                 842.0593085668335, 220703.35722785108, 1., 0.),
            'CONDITIONAL_POP0_REVISIT6_NO_IMPROVEMENT')

    def test_pop0_revisit6_secant_stays_inside_the_sign_bracket(self):
        positive, negative = -0.035901507358026205, -0.1359015073580262
        theta = tracer0_secant(positive, 842.0593085668335, negative, -220703.35722785108)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - positive), abs(theta - negative))
        self.assertAlmostEqual(theta, -0.0362815916412993)
        with self.assertRaises(ValueError):
            tracer0_secant(negative, -220703.35722785108, positive, 842.0593085668335)


    def test_pop0_revisit6_secant_meets_the_accepted_end_gate(self):
        initial = -142420.23671111895 + 5860.262418762844
        best = -142420.23671111895 + 5860.422459658225
        self.assertEqual(
            pop0_revisit6_status(8208706.210870936, 8208706.0508437585,
                                 842.059308566838, 0.0022238618678747032,
                                 initial, best),
            'CONDITIONAL_POP0_REVISIT6_REDUCED')

    def test_pop0_revisit5_rejected_step_does_not_improve(self):
        self.assertEqual(
            pop0_revisit5_status(8209252.606482146, 8220063.228338363,
                                 872.695798832995, 217084.46761286855, 1., 0.),
            'CONDITIONAL_POP0_REVISIT5_NO_IMPROVEMENT')

    def test_pop0_revisit5_secant_stays_inside_the_sign_bracket(self):
        positive, negative = 0.06369809479077561, -0.0363019052092244
        theta = tracer0_secant(positive, 217084.46761286855, negative, -872.695798832995)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))
        self.assertAlmostEqual(theta, -0.035901507358026205)

    def test_pop0_revisit5_secant_meets_the_tenfold_gate(self):
        initial_likelihood = -142420.23671111895 + 5313.276050871325
        best_likelihood = -142420.23671111895 + 5313.450747129417
        self.assertEqual(
            pop0_revisit5_status(
                8209252.606482146, 8209252.431771433,
                872.6957988329941, 0.01021211907156118,
                initial_likelihood, best_likelihood),
            'CONDITIONAL_POP0_REVISIT5_REDUCED')

    def test_population0_revisit4_secant_stays_inside_the_sign_bracket(self):
        negative, positive = -0.03689363492481951, 0.0631063650751805
        theta = tracer0_secant(positive, 222075.27977415553, negative, -1321.9075411483052)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))
        self.assertAlmostEqual(theta, -0.0363019052092244)

    def test_tracer0_revisit2_gate_uses_the_count_log_likelihood(self):
        self.assertEqual(tracer0_revisit2_status(10., 9., 895.970, 89.597, 3., 3.1),
                         'CONDITIONAL_TRACER0_REVISIT2_REDUCED')
        self.assertEqual(tracer0_revisit2_status(10., 9., 895.970, 89.598, 3., 3.1),
                         'CONDITIONAL_TRACER0_REVISIT2_IMPROVED')
        self.assertEqual(tracer0_revisit2_status(10., 9., 895.970, 89.597, 3., 2.),
                         'CONDITIONAL_TRACER0_REVISIT2_IMPROVED')
        self.assertEqual(tracer0_revisit2_status(10., 10., 895.970, 89.597, 3., 4.),
                         'CONDITIONAL_TRACER0_REVISIT2_NO_IMPROVEMENT')

    def test_tracer0_revisit3_gate_uses_the_count_log_likelihood(self):
        self.assertEqual(tracer0_revisit3_status(10., 9., 1134.545, 113.4545, 3., 3.1),
                         'CONDITIONAL_TRACER0_REVISIT3_REDUCED')
        self.assertEqual(tracer0_revisit3_status(10., 9., 1134.545, 113.4546, 3., 3.1),
                         'CONDITIONAL_TRACER0_REVISIT3_IMPROVED')
        self.assertEqual(tracer0_revisit3_status(10., 9., 1134.545, 113.4545, 3., 2.),
                         'CONDITIONAL_TRACER0_REVISIT3_IMPROVED')
        self.assertEqual(tracer0_revisit3_status(10., 10., 1134.545, 113.4545, 3., 4.),
                         'CONDITIONAL_TRACER0_REVISIT3_NO_IMPROVEMENT')

    def test_tracer0_revisit3_secant_stays_inside_the_sign_bracket(self):
        negative, positive = 0.2709028932654769, 0.3709028932654769
        theta = tracer0_secant(positive, 19479.742103700915, negative, -1134.54521280877)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))
        self.assertAlmostEqual(theta, 0.27640657716872846)

    def test_tracer0_revisit3_secant_meets_the_tenfold_gate(self):
        self.assertEqual(
            tracer0_revisit3_status(8209758.453135235, 8209755.03961515,
                                    1134.5452128087713, 104.01340469296662,
                                    -142423.65173731328, -142420.236711119),
            'CONDITIONAL_TRACER0_REVISIT3_REDUCED')

    def test_pop11_line_gate_uses_count_plus_fp(self):
        self.assertEqual(pop11_line_status(10., 9., 578.646, 57.8645, 3., 3.1),
                         'CONDITIONAL_POP11_LINE_REDUCED')
        self.assertEqual(pop11_line_status(10., 9., 578.646, 57.8646, 3., 3.1),
                         'CONDITIONAL_POP11_LINE_IMPROVED')
        self.assertEqual(pop11_line_status(10., 9., 578.646, 57.8645, 3., 2.),
                         'CONDITIONAL_POP11_LINE_IMPROVED')
        self.assertEqual(pop11_line_status(10., 10., 578.646, 57.8645, 3., 4.),
                         'CONDITIONAL_POP11_LINE_NO_IMPROVEMENT')

    def test_pop11_line_three_steps_improved_without_a_tenfold_drop(self):
        initial = -142420.236711119 + 4810.587073462018
        best = -142420.236711119 + 5013.605979035632
        self.assertEqual(
            pop11_line_status(8209755.03961515, 8209552.090364812,
                              578.6464599997995, 762.4874205196342,
                              initial, best),
            'CONDITIONAL_POP11_LINE_IMPROVED')

    def test_pop11_revisit_gate_uses_the_new_derivative(self):
        self.assertEqual(
            pop11_revisit_status(10., 9., 762.4874205196338, 76.2487, 3., 3.1),
            'CONDITIONAL_POP11_REVISIT_REDUCED')
        self.assertEqual(
            pop11_revisit_status(10., 9., 762.4874205196338, 76.2488, 3., 3.1),
            'CONDITIONAL_POP11_REVISIT_IMPROVED')
        self.assertEqual(
            pop11_revisit_status(10., 9., 762.4874205196338, 76.2487, 3., 2.),
            'CONDITIONAL_POP11_REVISIT_IMPROVED')
        self.assertEqual(
            pop11_revisit_status(10., 10., 762.4874205196338, 76.2487, 3., 4.),
            'CONDITIONAL_POP11_REVISIT_NO_IMPROVEMENT')

    def test_pop11_revisit2_gate_uses_count_plus_fp(self):
        self.assertEqual(
            pop11_revisit2_status(10., 9., 793.5140851613787, 79.3514, 3., 3.1),
            'CONDITIONAL_POP11_REVISIT2_REDUCED')
        self.assertEqual(
            pop11_revisit2_status(10., 9., 793.5140851613787, 79.3515, 3., 3.1),
            'CONDITIONAL_POP11_REVISIT2_IMPROVED')
        self.assertEqual(
            pop11_revisit2_status(10., 9., 793.5140851613787, 79.3514, 3., 2.),
            'CONDITIONAL_POP11_REVISIT2_IMPROVED')
        self.assertEqual(
            pop11_revisit2_status(10., 10., 793.5140851613787, 79.3514, 3., 4.),
            'CONDITIONAL_POP11_REVISIT2_NO_IMPROVEMENT')

    def test_pop11_revisit3_gate_uses_count_plus_fp(self):
        self.assertEqual(
            pop11_revisit3_status(10., 9., 824.7243300726143, 82.4724, 3., 3.1),
            'CONDITIONAL_POP11_REVISIT3_REDUCED')
        self.assertEqual(
            pop11_revisit3_status(10., 9., 824.7243300726143, 82.4725, 3., 3.1),
            'CONDITIONAL_POP11_REVISIT3_IMPROVED')
        self.assertEqual(
            pop11_revisit3_status(10., 9., 824.7243300726143, 82.4724, 3., 2.),
            'CONDITIONAL_POP11_REVISIT3_IMPROVED')
        self.assertEqual(
            pop11_revisit3_status(10., 10., 824.7243300726143, 82.4724, 3., 4.),
            'CONDITIONAL_POP11_REVISIT3_NO_IMPROVEMENT')

    def test_pop11_revisit4_gate_uses_count_plus_fp(self):
        self.assertEqual(
            pop11_revisit4_status(10., 9., 480.1873616849716, 48.0187, 3., 3.1),
            'CONDITIONAL_POP11_REVISIT4_REDUCED')
        self.assertEqual(
            pop11_revisit4_status(10., 9., 480.1873616849716, 48.0188, 3., 3.1),
            'CONDITIONAL_POP11_REVISIT4_IMPROVED')
        self.assertEqual(
            pop11_revisit4_status(10., 9., 480.1873616849716, 48.0187, 3., 2.),
            'CONDITIONAL_POP11_REVISIT4_IMPROVED')
        self.assertEqual(
            pop11_revisit4_status(10., 10., 480.1873616849716, 48.0187, 3., 4.),
            'CONDITIONAL_POP11_REVISIT4_NO_IMPROVEMENT')

    def test_pop11_revisit4_sign_bracket_is_improved(self):
        initial = -142420.236711119 + 5915.909004574992
        best = -142420.236711119 + 5973.527200671365
        self.assertEqual(
            pop11_revisit4_status(8208650.971531487, 8208593.629772215,
                                  480.1873616849718, 65.5573929047156,
                                  initial, best),
            'CONDITIONAL_POP11_REVISIT4_IMPROVED')

    def test_pop11_revisit4_secant_stays_inside_the_sign_bracket(self):
        positive, negative = -1.4821841183726576, -1.5821841183726577
        theta = tracer0_secant(positive, 65.5573929047156, negative, -209.99064377010126)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - positive), abs(theta - negative))
        self.assertAlmostEqual(theta, -1.505975756918093)
        with self.assertRaises(ValueError):
            tracer0_secant(negative, -209.99064377010126, positive, 65.5573929047156)

    def test_pop11_revisit4_secant_uses_the_accepted_end_gate(self):
        self.assertEqual(
            pop11_revisit4_status(8208593.629772215, 8208593.0,
                                  65.5573929047156, 6.5557, 1., 1.),
            'CONDITIONAL_POP11_REVISIT4_REDUCED')
        self.assertEqual(
            pop11_revisit4_status(8208593.629772215, 8208593.0,
                                  65.5573929047156, 6.5558, 1., 1.),
            'CONDITIONAL_POP11_REVISIT4_IMPROVED')

    def test_pop11_revisit4_secant_meets_the_accepted_end_gate(self):
        initial = -142420.236711119 + 5973.527200671365
        best = -142420.236711119 + 5974.402139239697
        self.assertEqual(
            pop11_revisit4_status(8208593.629772191, 8208592.790380232,
                                  65.5573929047159, 4.549920068464542,
                                  initial, best),
            'CONDITIONAL_POP11_REVISIT4_REDUCED')

    def test_pop11_revisit3_three_steps_improved_without_a_tenfold_drop(self):
        initial = -142420.23671111895 + 5645.364285825175
        best = -142420.23671111895 + 5846.452918299063
        self.assertEqual(
            pop11_revisit3_status(8208920.769355999, 8208720.020378761,
                                  824.724330072614, 467.0405898321668,
                                  initial, best),
            'CONDITIONAL_POP11_REVISIT3_IMPROVED')

    def test_pop13_line_gate_uses_count_plus_fp(self):
        self.assertEqual(
            pop13_line_status(10., 9., 1091.1301573851867, 109.1130, 3., 3.1),
            'CONDITIONAL_POP13_LINE_REDUCED')
        self.assertEqual(
            pop13_line_status(10., 9., 1091.1301573851867, 109.1131, 3., 3.1),
            'CONDITIONAL_POP13_LINE_IMPROVED')
        self.assertEqual(
            pop13_line_status(10., 9., 1091.1301573851867, 109.1130, 3., 2.),
            'CONDITIONAL_POP13_LINE_IMPROVED')
        self.assertEqual(
            pop13_line_status(10., 10., 1091.1301573851867, 109.1130, 3., 4.),
            'CONDITIONAL_POP13_LINE_NO_IMPROVEMENT')

    def test_pop13_rejected_step_does_not_improve(self):
        self.assertEqual(
            pop13_line_status(8208720.020378785, 8208826.302583098,
                              1091.1301573851856, 3201.1612935614044, 1., 0.),
            'CONDITIONAL_POP13_LINE_NO_IMPROVEMENT')

    def test_pop13_secant_stays_inside_the_sign_bracket(self):
        positive, negative = 0.013000910407807214, -0.0869990895921928
        theta = tracer0_secant(positive, 1091.1301573851856, negative, -3201.1612935614044)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - positive), abs(theta - negative))
        self.assertAlmostEqual(theta, -0.012419780844287421)
        with self.assertRaises(ValueError):
            tracer0_secant(negative, -3201.1612935614044, positive, 1091.1301573851856)

    def test_pop13_secant_meets_the_accepted_end_gate(self):
        initial = -142420.236711119 + 5846.452918299063
        best = -142420.236711119 + 5860.2624187628435
        self.assertEqual(
            pop13_line_status(8208720.020378785, 8208706.210870936,
                              1091.130157385182, 5.431992665149094,
                              initial, best),
            'CONDITIONAL_POP13_LINE_REDUCED')

    def test_pop11_revisit2_three_steps_improved_without_a_tenfold_drop(self):
        initial = -142420.23671111895 + 5313.450747129416
        best = -142420.23671111895 + 5526.00578201406
        self.assertEqual(
            pop11_revisit2_status(8209252.431771433, 8209040.126391783,
                                  793.5140851613793, 575.7497278965304,
                                  initial, best),
            'CONDITIONAL_POP11_REVISIT2_IMPROVED')

    def test_pop10_line_gate_uses_count_plus_fp(self):
        self.assertEqual(
            pop10_line_status(10., 9., 1460.4235570371668, 146.0423, 3., 3.1),
            'CONDITIONAL_POP10_LINE_REDUCED')
        self.assertEqual(
            pop10_line_status(10., 9., 1460.4235570371668, 146.0424, 3., 3.1),
            'CONDITIONAL_POP10_LINE_IMPROVED')
        self.assertEqual(
            pop10_line_status(10., 9., 1460.4235570371668, 146.0423, 3., 2.),
            'CONDITIONAL_POP10_LINE_IMPROVED')
        self.assertEqual(
            pop10_line_status(10., 10., 1460.4235570371668, 146.0423, 3., 4.),
            'CONDITIONAL_POP10_LINE_NO_IMPROVEMENT')

    def test_pop10_accepted_step_does_not_meet_the_line_gate(self):
        self.assertEqual(
            pop10_line_status(8209040.126391808, 8208943.950005639,
                              1460.4235570371677, 462.82762569735536, 1., 1.),
            'CONDITIONAL_POP10_LINE_IMPROVED')

    def test_pop10_secant_stays_inside_the_sign_bracket(self):
        positive, negative = -0.05730666969573986, -0.15730666969573986
        theta = tracer0_secant(positive, 462.82762569735536, negative, -527.7267124017568)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertAlmostEqual(theta, -0.10403077234169145)

    def test_pop10_secant_meets_the_accepted_end_gate(self):
        initial = -142420.236711119 + 5622.1828988496945
        best = -142420.236711119 + 5632.946139178732
        self.assertEqual(
            pop10_line_status(8208943.950005615, 8208933.190534459,
                              462.82762569735536, 1.861064299446706,
                              initial, best),
            'CONDITIONAL_POP10_LINE_REDUCED')

    def test_pop1_line_gate_uses_count_plus_fp(self):
        self.assertEqual(
            pop1_line_status(10., 9., 809.7459785875657, 80.9745, 3., 3.1),
            'CONDITIONAL_POP1_LINE_REDUCED')
        self.assertEqual(
            pop1_line_status(10., 9., 809.7459785875657, 80.9746, 3., 3.1),
            'CONDITIONAL_POP1_LINE_IMPROVED')
        self.assertEqual(
            pop1_line_status(10., 9., 809.7459785875657, 80.9745, 3., 2.),
            'CONDITIONAL_POP1_LINE_IMPROVED')
        self.assertEqual(
            pop1_line_status(10., 10., 809.7459785875657, 80.9745, 3., 4.),
            'CONDITIONAL_POP1_LINE_NO_IMPROVEMENT')

    def test_pop1_rejected_step_does_not_improve(self):
        self.assertEqual(
            pop1_line_status(8208933.190534484, 8208984.786896571,
                             809.7459785875661, 1848.6781202056213, 1., 0.),
            'CONDITIONAL_POP1_LINE_NO_IMPROVEMENT')

    def test_pop1_secant_stays_inside_the_sign_bracket(self):
        positive, negative = 0.11476529714749317, 0.014765297147493167
        theta = tracer0_secant(positive, 809.7459785875661, negative, -1848.6781202056213)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - positive), abs(theta - negative))
        self.assertAlmostEqual(theta, 0.08430567338185109)
        with self.assertRaises(ValueError):
            tracer0_secant(negative, -1848.6781202056213, positive, 809.7459785875661)

    def test_pop1_secant_meets_the_accepted_end_gate(self):
        initial = -142420.23671111895 + 5632.946139178732
        best = -142420.23671111895 + 5645.364285825175
        self.assertEqual(
            pop1_line_status(8208933.190534459, 8208920.769355999,
                             809.7459785875636, 4.864094740441916,
                             initial, best),
            'CONDITIONAL_POP1_LINE_REDUCED')

    def test_pop11_revisit_three_steps_improved_without_a_tenfold_drop(self):
        initial = -142420.23671111895 + 5013.605979035632
        best = -142420.23671111895 + 5253.047414546184
        self.assertEqual(
            pop11_revisit_status(8209552.090364812, 8209312.8085845355,
                                 762.4874205196336, 793.340778356706,
                                 initial, best),
            'CONDITIONAL_POP11_REVISIT_IMPROVED')

    def test_pop4_line_gate_uses_count_plus_fp(self):
        self.assertEqual(pop4_line_status(10., 9., 971.9560505156035, 97.1956, 3., 3.1),
                         'CONDITIONAL_POP4_LINE_REDUCED')
        self.assertEqual(pop4_line_status(10., 9., 971.9560505156035, 97.1957, 3., 3.1),
                         'CONDITIONAL_POP4_LINE_IMPROVED')
        self.assertEqual(pop4_line_status(10., 9., 971.9560505156035, 97.1956, 3., 2.),
                         'CONDITIONAL_POP4_LINE_IMPROVED')
        self.assertEqual(pop4_line_status(10., 10., 971.9560505156035, 97.1956, 3., 4.),
                         'CONDITIONAL_POP4_LINE_NO_IMPROVEMENT')

    def test_pop4_revisit_gate_uses_count_plus_fp(self):
        self.assertEqual(
            pop4_revisit_status(10., 9., 542.8941177183742, 54.2894, 3., 3.1),
            'CONDITIONAL_POP4_REVISIT_REDUCED')
        self.assertEqual(
            pop4_revisit_status(10., 9., 542.8941177183742, 54.2895, 3., 3.1),
            'CONDITIONAL_POP4_REVISIT_IMPROVED')
        self.assertEqual(
            pop4_revisit_status(10., 9., 542.8941177183742, 54.2894, 3., 2.),
            'CONDITIONAL_POP4_REVISIT_IMPROVED')
        self.assertEqual(
            pop4_revisit_status(10., 10., 542.8941177183742, 54.2894, 3., 4.),
            'CONDITIONAL_POP4_REVISIT_NO_IMPROVEMENT')

    def test_pop4_revisit_rejected_step_does_not_improve(self):
        self.assertEqual(
            pop4_revisit_status(8208592.790380232, 8208777.425503423,
                                542.8941177183739, 4261.167895863255, 1., 0.),
            'CONDITIONAL_POP4_REVISIT_NO_IMPROVEMENT')

    def test_pop4_revisit_secant_stays_inside_the_sign_bracket(self):
        positive, negative = -0.17172582151837482, -0.2717258215183748
        theta = tracer0_secant(positive, 4261.167895863255, negative, -542.8941177183739)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))
        self.assertAlmostEqual(theta, -0.260425090674694)
        with self.assertRaises(ValueError):
            tracer0_secant(negative, -542.8941177183739, positive, 4261.167895863255)

    def test_pop4_revisit_secant_meets_the_accepted_end_gate(self):
        initial = -142420.236711119 + 5974.402139239697
        best = -142420.236711119 + 5977.523608313021
        self.assertEqual(
            pop4_revisit_status(8208592.790380232, 8208589.665904312,
                                542.8941177183742, 9.47192682401596,
                                initial, best),
            'CONDITIONAL_POP4_REVISIT_REDUCED')


    def test_pop0_revisit8_gate_uses_count_plus_fp(self):
        self.assertEqual(
            pop0_revisit8_status(10., 9., 336.25482118341034, 33.6254, 3., 3.1),
            'CONDITIONAL_POP0_REVISIT8_REDUCED')
        self.assertEqual(
            pop0_revisit8_status(10., 9., 336.25482118341034, 33.6255, 3., 3.1),
            'CONDITIONAL_POP0_REVISIT8_IMPROVED')
        self.assertEqual(
            pop0_revisit8_status(10., 9., 336.25482118341034, 33.6254, 3., 2.),
            'CONDITIONAL_POP0_REVISIT8_IMPROVED')
        self.assertEqual(
            pop0_revisit8_status(10., 10., 336.25482118341034, 33.6254, 3., 4.),
            'CONDITIONAL_POP0_REVISIT8_NO_IMPROVEMENT')


    def test_pop0_revisit8_rejected_step_does_not_improve(self):
        self.assertEqual(
            pop0_revisit8_status(8208589.665904312, 8224897.223151519,
                                 336.2548211834142, 326486.8995091744, 1., 0.),
            'CONDITIONAL_POP0_REVISIT8_NO_IMPROVEMENT')

    def test_pop0_revisit8_secant_stays_inside_the_sign_bracket(self):
        positive, negative = 0.06402346376993898, -0.03597653623006103
        theta = tracer0_secant(positive, 326486.8995091744, negative, -336.2548211834142)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))
        self.assertEqual(theta, -0.03587365036756075)
        with self.assertRaises(ValueError):
            tracer0_secant(negative, -336.2548211834142, positive, 326486.8995091744)


    def test_pop0_revisit8_secant_meets_the_accepted_end_gate(self):
        initial = -142420.236711119 + 5977.523608313021
        best = -142420.236711119 + 5977.5409024630535
        self.assertEqual(
            pop0_revisit8_status(8208589.665904312, 8208589.648606465,
                                 336.25482118341364, 0.0016989132560376458,
                                 initial, best),
            'CONDITIONAL_POP0_REVISIT8_REDUCED')


    def test_tracer4_revisit_gate_uses_count_plus_fp(self):
        self.assertEqual(
            tracer4_revisit_status(10., 9., 329.8449370157337, 32.9844, 3., 3.1),
            'CONDITIONAL_TRACER4_REVISIT_REDUCED')
        self.assertEqual(
            tracer4_revisit_status(10., 9., 329.8449370157337, 32.9845, 3., 3.1),
            'CONDITIONAL_TRACER4_REVISIT_IMPROVED')
        self.assertEqual(
            tracer4_revisit_status(10., 9., 329.8449370157337, 32.9844, 3., 2.),
            'CONDITIONAL_TRACER4_REVISIT_IMPROVED')
        self.assertEqual(
            tracer4_revisit_status(10., 10., 329.8449370157337, 32.9844, 3., 4.),
            'CONDITIONAL_TRACER4_REVISIT_NO_IMPROVEMENT')

    def test_tracer4_revisit_sign_change_stays_improved(self):
        initial = -142420.236711119 + 5977.5409024630535
        best = -142394.81260779343 + 5978.423264787913
        self.assertEqual(
            tracer4_revisit_status(8208589.648606465, 8208563.429289561,
                                   329.8449370157307, 93.06800982891079,
                                   initial, best),
            'CONDITIONAL_TRACER4_REVISIT_IMPROVED')

    def test_tracer4_revisit_secant_stays_inside_the_sign_bracket(self):
        positive, negative = 0.5357437290186818, 0.43574372901868186
        theta = tracer0_secant(positive, 93.06800982891079, negative, -137.4683568046625)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - positive), abs(theta - negative))
        self.assertEqual(theta, 0.4953735214076467)
        with self.assertRaises(ValueError):
            tracer0_secant(negative, -137.4683568046625, positive, 93.06800982891079)

    def test_tracer4_revisit_accepted_end_gate(self):
        self.assertEqual(
            tracer4_revisit_status(8208563.429289561, 8208563.4,
                                   93.06800982891079, 9.3068, 3., 3.1),
            'CONDITIONAL_TRACER4_REVISIT_REDUCED')
        self.assertEqual(
            tracer4_revisit_status(8208563.429289561, 8208563.4,
                                   93.06800982891079, 9.3069, 3., 3.1),
            'CONDITIONAL_TRACER4_REVISIT_IMPROVED')

    def test_tracer4_revisit_secant_meets_the_accepted_end_gate(self):
        initial = -142394.81260779343 + 5978.423264787913
        best = -142392.90580909158 + 5978.259938462888
        self.assertEqual(
            tracer4_revisit_status(8208563.429289561, 8208561.665003976,
                                   93.06800982891079, 4.623539209986719,
                                   initial, best),
            'CONDITIONAL_TRACER4_REVISIT_REDUCED')

    def test_tracer0_revisit4_gate_uses_the_count_log_likelihood(self):
        self.assertEqual(
            tracer0_revisit4_status(10., 9., 505.4950486895277, 50.5495, 3., 3.1),
            'CONDITIONAL_TRACER0_REVISIT4_REDUCED')
        self.assertEqual(
            tracer0_revisit4_status(10., 9., 505.4950486895277, 50.5496, 3., 3.1),
            'CONDITIONAL_TRACER0_REVISIT4_IMPROVED')
        self.assertEqual(
            tracer0_revisit4_status(10., 9., 505.4950486895277, 50.5495, 3., 2.),
            'CONDITIONAL_TRACER0_REVISIT4_IMPROVED')
        self.assertEqual(
            tracer0_revisit4_status(10., 10., 505.4950486895277, 50.5495, 3., 4.),
            'CONDITIONAL_TRACER0_REVISIT4_NO_IMPROVEMENT')

    def test_tracer0_revisit4_rejected_step_does_not_improve(self):
        self.assertEqual(
            tracer0_revisit4_status(8208561.665003976, 8209398.463877712,
                                    505.49504868952965, 16669.361922521985,
                                    -142392.90580909158, -143229.72732348583),
            'CONDITIONAL_TRACER0_REVISIT4_NO_IMPROVEMENT')

    def test_tracer0_revisit4_secant_stays_inside_the_sign_bracket(self):
        positive, negative = 0.27640657716872846, 0.17640657716872846
        theta = tracer0_secant(positive, 505.49504868952965, negative, -16669.361922521985)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - positive), abs(theta - negative))
        self.assertEqual(theta, 0.273463350045866)
        with self.assertRaises(ValueError):
            tracer0_secant(negative, -16669.361922521985, positive, 505.49504868952965)

    def test_tracer0_revisit4_secant_improves_without_the_tenfold_drop(self):
        initial_g = 505.49504868952863
        best_g = 50.59476630458172
        initial_obj = 8208561.665003976
        best_obj = 8208560.996369408
        initial_count = -142392.90580909158
        best_count = -142392.2379837209
        self.assertGreater(best_g, initial_g / 10.)
        self.assertEqual(
            tracer0_revisit4_status(initial_obj, best_obj, initial_g, best_g,
                                    initial_count, best_count),
            'CONDITIONAL_TRACER0_REVISIT4_IMPROVED')
        delta_obj = best_obj - initial_obj
        delta_count = best_count - initial_count
        delta_tracer = 0.8111208630960242 - 0.8119300591379364
        self.assertLess(abs(delta_obj - (-delta_count + delta_tracer)), 1e-9)
        white_pos, white_neg = 0.27640657716872846, 0.273463350045866
        self.assertLess(
            abs(delta_tracer - 0.5 * (white_neg ** 2 - white_pos ** 2)), 1e-12)

    def test_tracer6_line_gate_uses_count_plus_fp(self):
        initial = 270.74828769716277
        self.assertEqual(
            tracer6_line_status(10., 9., initial, 27.0748, 3., 3.1),
            'CONDITIONAL_TRACER6_LINE_REDUCED')
        self.assertEqual(
            tracer6_line_status(10., 9., initial, 27.0749, 3., 3.1),
            'CONDITIONAL_TRACER6_LINE_IMPROVED')
        self.assertEqual(
            tracer6_line_status(10., 9., initial, 27.0748, 3., 2.),
            'CONDITIONAL_TRACER6_LINE_IMPROVED')
        self.assertEqual(
            tracer6_line_status(10., 10., initial, 1., 3., 4.),
            'CONDITIONAL_TRACER6_LINE_NO_IMPROVEMENT')
        self.assertLess(27.0748, initial / 10.)
        self.assertGreater(27.0749, initial / 10.)

    def test_tracer6_support_stop_does_not_improve_or_repeat_the_step(self):
        initial = 270.74828769716277
        objective = 8208560.996369408
        self.assertEqual(
            tracer6_line_status(objective, objective, initial, initial, 1., 1.),
            'CONDITIONAL_TRACER6_LINE_NO_IMPROVEMENT')
        self.assertEqual(tracer6_revisit_first_step(-initial), 0.05)
        with self.assertRaises(ValueError):
            tracer6_revisit_first_step(initial)
        self.assertEqual(
            tracer6_revisit_status(objective, objective - 1., initial, 27.0748, 3., 3.),
            'CONDITIONAL_TRACER6_REVISIT_REDUCED')
        self.assertEqual(
            tracer6_revisit_status(objective, objective - 1., initial, 27.0749, 3., 3.),
            'CONDITIONAL_TRACER6_REVISIT_IMPROVED')

    def test_tracer6_revisit_half_step_improves_without_the_tenfold_drop(self):
        initial_g = 270.74828769716316
        best_g = 261.94891913129055
        initial_obj = 8208560.996369408
        best_obj = 8208547.671551434
        initial_like = -142392.2379837209 + 5978.259938462888
        best_like = -142378.99792921095 + 5978.3447506951325
        self.assertGreater(best_g, initial_g / 10.)
        self.assertGreater(best_like, initial_like)
        self.assertEqual(
            tracer6_revisit_status(initial_obj, best_obj, initial_g, best_g,
                                   initial_like, best_like),
            'CONDITIONAL_TRACER6_REVISIT_IMPROVED')
        delta_count = -142378.99792921095 - (-142392.2379837209)
        delta_fp = 5978.3447506951325 - 5978.259938462888
        delta_tracer = 0.8111696309108478 - 0.8111208630960242
        self.assertLess(abs((best_obj - initial_obj) - (-delta_count - delta_fp + delta_tracer)), 1e-9)
        white0, white1 = -0.02402464370352757, 0.02597535629647243
        self.assertEqual(white1, white0 + 0.05)
        self.assertLess(abs(delta_tracer - 0.5 * (white1 ** 2 - white0 ** 2)), 1e-12)

    def test_tracer6_revisit2_keeps_the_support_safe_step(self):
        gradient = -261.948919131281
        self.assertEqual(tracer6_revisit2_first_step(gradient), 0.05)
        self.assertEqual(tracer6_revisit2_continuation_step(gradient), 0.05)
        self.assertEqual(tracer6_revisit2_continuation_step(-gradient), -0.05)
        self.assertGreater(abs(gradient), 270.74828769716277 / 10.)
        with self.assertRaises(ValueError):
            tracer6_revisit2_first_step(-gradient)
        with self.assertRaises(ValueError):
            tracer6_revisit2_continuation_step(0.)
        objective = 8208547.671551459
        gate = abs(gradient) / 10.
        self.assertEqual(
            tracer6_revisit2_status(objective, objective - 1., abs(gradient), gate + 0.001, 1., 1.1),
            'CONDITIONAL_TRACER6_REVISIT2_IMPROVED')
        self.assertEqual(
            tracer6_revisit2_status(objective, objective - 1., abs(gradient), gate - 0.001, 1., 1.1),
            'CONDITIONAL_TRACER6_REVISIT2_REDUCED')
        self.assertEqual(
            tracer6_revisit2_status(objective, objective - 1., abs(gradient), gate - 0.001, 1., 0.),
            'CONDITIONAL_TRACER6_REVISIT2_IMPROVED')

    def test_tracer6_support_step_scores_the_same_half_step(self):
        gradient = -261.94891913129004
        self.assertEqual(tracer6_support_step_size(gradient), 0.05)
        with self.assertRaises(ValueError):
            tracer6_support_step_size(-gradient)
        objective = 8208547.671551434
        gate = abs(gradient) / 10.
        self.assertEqual(
            tracer6_support_step_status(objective, objective - 1., abs(gradient), gate + 0.001, 1., 1.1),
            'CONDITIONAL_TRACER6_SUPPORT_STEP_IMPROVED')
        self.assertEqual(
            tracer6_support_step_status(objective, objective - 1., abs(gradient), gate - 0.001, 1., 1.1),
            'CONDITIONAL_TRACER6_SUPPORT_STEP_REDUCED')
        self.assertEqual(
            tracer6_support_step_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_TRACER6_SUPPORT_STEP_NO_IMPROVEMENT')

    def test_tracer6_support_step_improved_without_a_sign_change(self):
        initial_g = -261.9489191312816
        best_g = -251.26375904165042
        initial_obj = 8208547.671551434
        best_obj = 8208534.83278695
        initial_like = -136400.65317851581
        best_like = -136387.8118652634
        self.assertLess(best_obj, initial_obj)
        self.assertLess(best_g, 0.)
        self.assertLess(initial_g, 0.)
        self.assertGreater(abs(best_g), abs(initial_g) / 10.)
        self.assertGreater(best_like, initial_like)
        self.assertEqual(
            tracer6_support_step_status(initial_obj, best_obj, abs(initial_g), abs(best_g),
                                        initial_like, best_like),
            'CONDITIONAL_TRACER6_SUPPORT_STEP_IMPROVED')
        with self.assertRaises(ValueError):
            tracer0_secant(0.02597535629647243, initial_g, 0.07597535629647244, best_g)

    def test_tracer6_revisit3_uses_the_forty_ninth_gradient_gate(self):
        gradient = -251.26375904165926
        self.assertEqual(tracer6_revisit2_first_step(gradient), 0.05)
        self.assertEqual(tracer6_revisit2_continuation_step(gradient), 0.05)
        self.assertGreater(abs(gradient), 261.9489191312816 / 10.)
        objective = 8208534.83278695
        gate = abs(gradient) / 10.
        self.assertEqual(
            tracer6_revisit3_status(objective, objective - 1., abs(gradient), gate + 0.001, 1., 1.1),
            'CONDITIONAL_TRACER6_REVISIT3_IMPROVED')
        self.assertEqual(
            tracer6_revisit3_status(objective, objective - 1., abs(gradient), gate - 0.001, 1., 1.1),
            'CONDITIONAL_TRACER6_REVISIT3_REDUCED')

    def test_tracer6_support_step2_scores_the_refused_step(self):
        gradient = -251.2637590416596
        self.assertEqual(tracer6_support_step_size(gradient), 0.05)
        self.assertEqual(0.07597535629647244 + 0.05, 0.12597535629647244)
        objective = 8208534.83278695
        gate = abs(gradient) / 10.
        self.assertEqual(
            tracer6_support_step2_status(objective, objective - 1., abs(gradient), gate + 0.001, 1., 1.1),
            'CONDITIONAL_TRACER6_SUPPORT_STEP2_IMPROVED')
        self.assertEqual(
            tracer6_support_step2_status(objective, objective - 1., abs(gradient), gate - 0.001, 1., 1.1),
            'CONDITIONAL_TRACER6_SUPPORT_STEP2_REDUCED')
        self.assertEqual(
            tracer6_support_step2_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_TRACER6_SUPPORT_STEP2_NO_IMPROVEMENT')

    def test_tracer6_support_step2_improved_without_a_sign_change(self):
        initial_g = -251.26375904165948
        best_g = -238.50953348555262
        initial_obj = 8208534.83278695
        best_obj = 8208522.5797250355
        initial_like = -136387.8118652634
        best_like = -136375.5537545813
        white0 = 0.07597535629647244
        white1 = 0.12597535629647244
        delta_count = 12.169717923330609
        delta_fp = 0.08839275872378494
        delta_tracer = 0.0050487678148236625
        self.assertEqual(white0 + 0.05, white1)
        self.assertLess(abs((best_obj - initial_obj) - (-delta_count - delta_fp + delta_tracer)), 1e-9)
        self.assertLess(abs(delta_tracer - 0.5 * (white1 ** 2 - white0 ** 2)), 1e-12)
        self.assertLess(best_obj, initial_obj)
        self.assertLess(best_g, 0.)
        self.assertLess(initial_g, 0.)
        self.assertGreater(abs(best_g), abs(initial_g) / 10.)
        self.assertGreater(best_like, initial_like)
        self.assertEqual(
            tracer6_support_step2_status(initial_obj, best_obj, abs(initial_g), abs(best_g),
                                         initial_like, best_like),
            'CONDITIONAL_TRACER6_SUPPORT_STEP2_IMPROVED')
        with self.assertRaises(ValueError):
            tracer0_secant(white0, initial_g, white1, best_g)

    def test_tracer6_revisit4_uses_the_fiftieth_gradient_gate(self):
        gradient = -238.5095334855527
        self.assertEqual(tracer6_revisit2_first_step(gradient), 0.05)
        self.assertEqual(tracer6_revisit2_continuation_step(gradient), 0.05)
        self.assertGreater(abs(gradient), 251.26375904165926 / 10.)
        self.assertEqual(0.07597535629647244 + 0.05, 0.12597535629647244)
        objective = 8208522.5797250355
        gate = abs(gradient) / 10.
        self.assertEqual(
            tracer6_revisit4_status(objective, objective - 1., abs(gradient), gate + 0.001, 1., 1.1),
            'CONDITIONAL_TRACER6_REVISIT4_IMPROVED')
        self.assertEqual(
            tracer6_revisit4_status(objective, objective - 1., abs(gradient), gate - 0.001, 1., 1.1),
            'CONDITIONAL_TRACER6_REVISIT4_REDUCED')
        self.assertEqual(
            tracer6_revisit4_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_TRACER6_REVISIT4_NO_IMPROVEMENT')
        with self.assertRaises(ValueError):
            tracer0_secant(0.12597535629647244, gradient, 0.17597535629647244, gradient)

    def test_tracer6_support_step3_scores_the_refused_step(self):
        gradient = -238.50953348555274
        white = 0.12597535629647244
        self.assertEqual(tracer6_support_step_size(gradient), 0.05)
        self.assertEqual(white + 0.05, 0.17597535629647243)
        self.assertGreater(abs(gradient), 251.26375904165926 / 10.)
        objective = 8208522.5797250355
        gate = abs(gradient) / 10.
        self.assertEqual(
            tracer6_support_step3_status(objective, objective - 1., abs(gradient), gate + 0.001, 1., 1.1),
            'CONDITIONAL_TRACER6_SUPPORT_STEP3_IMPROVED')
        self.assertEqual(
            tracer6_support_step3_status(objective, objective - 1., abs(gradient), gate - 0.001, 1., 1.1),
            'CONDITIONAL_TRACER6_SUPPORT_STEP3_REDUCED')
        self.assertEqual(
            tracer6_support_step3_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_TRACER6_SUPPORT_STEP3_NO_IMPROVEMENT')
        with self.assertRaises(ValueError):
            tracer0_secant(white, gradient, white + 0.05, gradient)

    def test_tracer6_support_step3_improved_without_a_sign_change(self):
        initial_g = -238.50953348555262
        best_g = -223.56095110860338
        initial_obj = 8208522.5797250355
        best_obj = 8208511.01850836
        initial_like = -136375.5537545813
        best_like = -136363.98498913736
        white0 = 0.12597535629647244
        white1 = 0.17597535629647243
        delta_count = 11.478750340611441
        delta_fp = 0.09001510333564511
        delta_tracer = 0.007548767814823609
        self.assertEqual(white0 + 0.05, white1)
        self.assertLess(abs((best_obj - initial_obj) - (-delta_count - delta_fp + delta_tracer)), 1e-9)
        self.assertLess(abs(delta_tracer - 0.5 * (white1 ** 2 - white0 ** 2)), 1e-12)
        self.assertLess(best_obj, initial_obj)
        self.assertLess(best_g, 0.)
        self.assertLess(initial_g, 0.)
        self.assertGreater(abs(best_g), abs(initial_g) / 10.)
        self.assertGreater(best_like, initial_like)
        self.assertEqual(
            tracer6_support_step3_status(initial_obj, best_obj, abs(initial_g), abs(best_g),
                                         initial_like, best_like),
            'CONDITIONAL_TRACER6_SUPPORT_STEP3_IMPROVED')
        with self.assertRaises(ValueError):
            tracer0_secant(white0, initial_g, white1, best_g)

    def test_pop5_line_uses_the_fifty_first_gradient_gate(self):
        gradient = -238.34390670220037
        white = 0.07311500623629187
        self.assertEqual(pop5_line_step(gradient), 0.1)
        self.assertEqual(pop5_line_step(-gradient), -0.1)
        self.assertEqual(0.5 * white, 0.036557503118145936)
        self.assertEqual(white + 0.1, 0.1731150062362919)
        objective = 8208511.01850836
        gate = abs(gradient) / 10.
        self.assertEqual(
            pop5_line_status(objective, objective - 1., abs(gradient), gate + 0.001, 1., 1.1),
            'CONDITIONAL_POP5_LINE_IMPROVED')
        self.assertEqual(
            pop5_line_status(objective, objective - 1., abs(gradient), gate - 0.001, 1., 1.1),
            'CONDITIONAL_POP5_LINE_REDUCED')
        self.assertEqual(
            pop5_line_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_POP5_LINE_NO_IMPROVEMENT')
        with self.assertRaises(ValueError):
            pop5_line_step(0.)
        with self.assertRaises(ValueError):
            tracer0_secant(white, gradient, white + 0.1, gradient)

    def test_pop5_line_sign_change_has_one_secant(self):
        negative = 0.07311500623629187
        positive = 0.1731150062362919
        gradient_negative = -238.34390670220088
        gradient_positive = 16845.120014479984
        self.assertEqual(negative + 0.1, positive)
        theta = tracer0_secant(positive, gradient_positive, negative, gradient_negative)
        self.assertEqual(theta, 0.07451017941546588)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))
        self.assertEqual(0.5 * positive, 0.08655750311814595)
        self.assertEqual(0.5 * theta, 0.03725508970773294)
        count = -142342.5947882921
        fp0 = 5978.609799154731
        fp1 = 5144.173483611274
        pop0 = 4.180281801233866
        pop1 = 4.192593301857495
        obj0 = 8208511.01850836
        obj1 = 8209345.467135403
        self.assertLess(abs((obj1 - obj0) - ((pop1 - pop0) - (fp1 - fp0))), 1e-9)
        self.assertLess(abs((pop1 - pop0) - 0.5 * (positive ** 2 - negative ** 2)), 1e-16)
        self.assertGreater(obj1, obj0)
        self.assertGreater(abs(gradient_positive), abs(gradient_negative) / 10.)
        self.assertEqual(
            pop5_line_status(obj0, obj1, abs(gradient_negative), abs(gradient_positive),
                             count + fp0, count + fp1),
            'CONDITIONAL_POP5_LINE_NO_IMPROVEMENT')
        with self.assertRaises(ValueError):
            tracer0_secant(negative, gradient_negative, positive, gradient_positive)

    def test_pop5_secant_meets_the_accepted_end_gate(self):
        negative = 0.07311500623629187
        theta = 0.07451017941546588
        initial_g = -238.34390670220077
        best_g = 2.6689179176258726
        obj0 = 8208511.01850836
        obj1 = 8208510.854108464
        count = -142342.5947882921
        fp0 = 5978.60979915473
        fp1 = 5978.774302031556
        pop0 = 4.180281801233866
        pop1 = 4.180384782583661
        self.assertLess(negative, theta)
        self.assertLess(theta, negative + 0.1)
        self.assertEqual(0.5 * theta, 0.03725508970773294)
        self.assertLess(obj1, obj0)
        self.assertGreater(best_g, 0.)
        self.assertLess(initial_g, 0.)
        self.assertLess(abs(best_g), abs(initial_g) / 10.)
        self.assertLess(abs((obj1 - obj0) - ((pop1 - pop0) - (fp1 - fp0))), 1e-9)
        self.assertLess(abs((pop1 - pop0) - 0.5 * (theta ** 2 - negative ** 2)), 1e-15)
        self.assertEqual(
            pop5_line_status(obj0, obj1, abs(initial_g), abs(best_g), count + fp0, count + fp1),
            'CONDITIONAL_POP5_LINE_REDUCED')

    def test_pop0_revisit9_uses_the_fifty_second_gradient_gate(self):
        gradient = -512.4156757903945
        white = -0.03587365036756075
        self.assertEqual(pop0_revisit9_step(gradient), 0.1)
        self.assertEqual(pop0_revisit9_step(-gradient), -0.1)
        self.assertGreater(abs(gradient), 336.25482118341364 / 10.)
        self.assertEqual(abs(gradient) / 10., 51.241567579039454)
        self.assertEqual(0.3 + white, 0.26412634963243925)
        self.assertEqual(white + 0.1, 0.06412634963243925)
        self.assertEqual(0.3 + (white + 0.1), 0.36412634963243923)
        self.assertLess(abs(2.6689179176247357), abs(-238.34390670220037) / 10.)
        objective = 8208510.854108488
        self.assertEqual(
            pop0_revisit9_status(objective, objective - 1., abs(gradient),
                                 abs(gradient) / 10. + 0.001, 1., 1.1),
            'CONDITIONAL_POP0_REVISIT9_IMPROVED')
        self.assertEqual(
            pop0_revisit9_status(objective, objective - 1., abs(gradient),
                                 abs(gradient) / 10. - 0.001, 1., 1.1),
            'CONDITIONAL_POP0_REVISIT9_REDUCED')
        self.assertEqual(
            pop0_revisit9_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_POP0_REVISIT9_NO_IMPROVEMENT')
        with self.assertRaises(ValueError):
            pop0_revisit9_step(0.)

    def test_pop0_revisit9_secant_stays_inside_the_sign_bracket(self):
        negative = -0.03587365036756075
        positive = 0.06412634963243925
        initial_g = -512.4156757903997
        failed_g = 326283.52567259746
        theta = tracer0_secant(positive, failed_g, negative, initial_g)
        obj0 = 8208510.854108488
        obj1 = 8224799.428132071
        count = -142342.5947882921
        fp0 = 5978.774302031556
        fp1 = -10309.798308915426
        prior0 = 4.180384782583662
        prior1 = 4.181797417546905
        ic = 8072142.026921511
        tracer_prior = 0.8263159343553188
        rise = obj1 - obj0
        residual = rise - (-(fp1 - fp0) + (prior1 - prior0))
        prior_err = (prior1 - prior0) - 0.5 * (positive ** 2 - negative ** 2)
        self.assertEqual(theta, -0.03571685047779585)
        self.assertEqual(0.3 + theta, 0.2642831495222041)
        self.assertLess(negative, theta)
        self.assertLess(theta, positive)
        self.assertLess(theta - negative, positive - theta)
        self.assertEqual(rise, 16288.574023582973)
        self.assertEqual(residual, 1.027729013003409e-09)
        self.assertLess(abs(residual), 2e-9)
        self.assertEqual(prior_err, -8.604228440844963e-16)
        self.assertLess(abs(prior_err), 1e-15)
        self.assertEqual(ic + tracer_prior + prior0 - count - fp0, obj0)
        self.assertEqual(ic + tracer_prior + prior1 - count - fp1, obj1)
        self.assertGreater(abs(failed_g), abs(initial_g) / 10.)
        self.assertEqual(
            pop0_revisit9_status(obj0, obj0, abs(initial_g), abs(initial_g),
                                 count + fp0, count + fp0),
            'CONDITIONAL_POP0_REVISIT9_NO_IMPROVEMENT')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        with self.assertRaises(ValueError):
            tracer0_secant(negative, initial_g, positive, failed_g)

    def test_pop0_revisit9_secant_reduced_stops_the_coordinate(self):
        negative = -0.03587365036756075
        theta = -0.03571685047779585
        line_g = -512.4156757903997
        secant_g = 0.0020660503449805023
        obj0 = 8208510.854108488
        obj1 = 8208510.81393529
        count = -142342.5947882921
        fp0 = 5978.774302031556
        fp1 = 5978.814469617511
        prior0 = 4.180384782583662
        prior1 = 4.180379169892341
        ic = 8072142.026921511
        tracer_prior = 0.8263159343553188
        drop = obj0 - obj1
        residual = (obj1 - obj0) - (-(fp1 - fp0) + (prior1 - prior0))
        prior_err = (prior1 - prior0) - 0.5 * (theta ** 2 - negative ** 2)
        self.assertEqual(0.3 + theta, 0.2642831495222041)
        self.assertLess(negative, theta)
        self.assertLess(theta, negative + 0.1)
        self.assertEqual(abs(line_g) / 10., 51.24156757903997)
        self.assertLess(abs(secant_g), abs(line_g) / 10.)
        self.assertLess(abs(secant_g), 1.)
        self.assertEqual(drop, 0.040173198096454144)
        self.assertEqual(residual, 5.505809141936879e-10)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(prior_err, -2.17057274931598e-16)
        self.assertEqual(ic + tracer_prior + prior0 - count - fp0, obj0)
        self.assertEqual(ic + tracer_prior + prior1 - count - fp1, obj1)
        self.assertEqual(
            pop0_revisit9_status(obj0, obj1, abs(line_g), abs(secant_g),
                                 count + fp0, count + fp1),
            'CONDITIONAL_POP0_REVISIT9_REDUCED')

    def test_pop3_revisit5_uses_the_fifty_third_gradient_gate(self):
        gradient = 273.552509195402
        white = -0.4097629673888512
        self.assertEqual(pop3_revisit5_step(gradient), -0.1)
        self.assertEqual(pop3_revisit5_step(-gradient), 0.1)
        self.assertGreater(abs(gradient), 620.8935209944603 / 10.)
        self.assertEqual(abs(gradient) / 10., 27.355250919540204)
        self.assertEqual(0.5 * white, -0.2048814836944256)
        self.assertEqual(white - 0.1, -0.5097629673888512)
        self.assertEqual(0.5 * (white - 0.1), -0.2548814836944256)
        self.assertLess(abs(0.002066050339750908), 1.)
        self.assertLess(abs(0.002066050339750908), abs(-512.4156757903997) / 10.)
        objective = 8208510.81393529
        self.assertEqual(
            pop3_revisit5_status(objective, objective - 1., abs(gradient),
                                 abs(gradient) / 10. + 0.001, 1., 1.1),
            'CONDITIONAL_POP3_REVISIT5_IMPROVED')
        self.assertEqual(
            pop3_revisit5_status(objective, objective - 1., abs(gradient),
                                 abs(gradient) / 10. - 0.001, 1., 1.1),
            'CONDITIONAL_POP3_REVISIT5_REDUCED')
        self.assertEqual(
            pop3_revisit5_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_POP3_REVISIT5_NO_IMPROVEMENT')
        with self.assertRaises(ValueError):
            pop3_revisit5_step(0.)

    def test_pop3_revisit5_secant_stays_inside_the_sign_bracket(self):
        positive = -0.4097629673888512
        negative = -0.5097629673888512
        g_pos = 273.5525091954001
        g_neg = -49068.05941797826
        theta = tracer0_secant(positive, g_pos, negative, g_neg)
        obj0 = 8208510.81393529
        obj1 = 8210968.881699661
        count = -142342.59478829207
        fp0 = 5978.814469617512
        fp1 = 3520.792681543325
        prior0 = 4.180379169892341
        prior1 = 4.226355466631226
        ic = 8072142.026921511
        tracer_prior = 0.8263159343553188
        rise = obj1 - obj0
        residual = rise - (-(fp1 - fp0) + (prior1 - prior0))
        prior_err = (prior1 - prior0) - 0.5 * (negative ** 2 - positive ** 2)
        failed_gap = ic + tracer_prior + prior1 - count - fp1 - obj1
        self.assertEqual(theta, -0.41031737268391766)
        self.assertEqual(0.5 * theta, -0.20515868634195883)
        self.assertLess(negative, theta)
        self.assertLess(theta, positive)
        self.assertLess(positive - theta, theta - negative)
        self.assertEqual(rise, 2458.067764370702)
        self.assertEqual(residual, -2.2373569663614035e-10)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(prior_err, -5.273559366969494e-16)
        self.assertEqual(ic + tracer_prior + prior0 - count - fp0, obj0)
        self.assertEqual(failed_gap, 9.313225746154785e-10)
        self.assertGreater(abs(g_neg), abs(g_pos) / 10.)
        self.assertLess(abs(0.002066050347254239), 1.)
        self.assertEqual(
            pop3_revisit5_status(obj0, obj0, abs(g_pos), abs(g_pos), count + fp0, count + fp0),
            'CONDITIONAL_POP3_REVISIT5_NO_IMPROVEMENT')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        with self.assertRaises(ValueError):
            tracer0_secant(negative, g_neg, positive, g_pos)

    def test_pop3_revisit5_secant_reduced_stops_the_coordinate(self):
        positive = -0.4097629673888512
        negative = -0.5097629673888512
        theta = tracer0_secant(positive, 273.5525091954001, negative, -49068.05941797826)
        obj0 = 8208510.81393529
        obj1 = 8208510.739070199
        count = -142342.5947882921
        fp0 = 5978.814469617511
        fp1 = 5978.88956203627
        prior0 = 4.180379169892341
        prior1 = 4.180606498333799
        ic = 8072142.026921511
        tracer_prior = 0.8263159343553188
        delta = obj1 - obj0
        dfp = fp1 - fp0
        dprior = prior1 - prior0
        residual = delta - (-dfp + dprior)
        saved_gap = ic + tracer_prior + prior1 - count - fp1 - obj1
        derivatives = (
            -56.67514861203325, 35.29548877332872, 28.211046509271345,
            -31.183485327233313, -223.56921281996932, 362.2367269677417,
            -3.4822793321745538, -14.657305206069516, -216.5385939036381)
        self.assertEqual(theta, -0.41031737268391766)
        self.assertEqual(0.5 * theta, -0.20515868634195883)
        self.assertLess(negative, theta)
        self.assertLess(theta, positive)
        self.assertEqual(delta, -0.07486509066075087)
        self.assertEqual(residual, -3.4352254374425684e-10)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(dprior - 0.5 * (theta ** 2 - positive ** 2), -3.3306690738754696e-16)
        self.assertEqual(ic + tracer_prior + prior0 - count - fp0, obj0)
        self.assertEqual(saved_gap, 9.313225746154785e-10)
        self.assertLessEqual(abs(-3.4822793321745538), abs(273.5525091954018) / 10.)
        self.assertEqual(abs(273.5525091954018) / 10., 27.35525091954018)
        self.assertGreaterEqual(min(abs(value) for value in derivatives), 1.)
        self.assertLess(abs(-14.657305206069516), 542.8941177183742 / 10.)
        self.assertGreater(abs(362.2367269677417), abs(-512.4156757903997) / 10.)
        self.assertEqual(
            pop3_revisit5_status(
                obj0, obj1, abs(273.5525091954018), abs(-3.4822793321745538),
                count + fp0, count + fp1),
            'CONDITIONAL_POP3_REVISIT5_REDUCED')

    def test_pop0_revisit10_uses_the_fifty_fourth_gradient_gate(self):
        gradient = 362.23672696774145
        white = -0.03571685047779585
        derivatives = (
            -56.675148612040786, 35.29548877332851, 28.211046509269252,
            -31.18348532723347, -223.56921281996927, gradient,
            -3.4822793321738716, -14.65730520606909, -216.53859390363849)
        diffs = (
            -7.538858426414663e-12, -2.1316282072803006e-13, -2.092548356813495e-12,
            -1.5631940186722204e-13, 5.684341886080802e-14, -2.2737367544323206e-13,
            6.821210263296962e-13, 4.263256414560601e-13, -3.979039320256561e-13)
        objective = 8208510.739070199
        gap = (8072142.026921511 + 0.8263159343553188 + 4.180606498333799
               - (-142342.5947882921) - 5978.88956203627 - objective)
        self.assertEqual(pop0_revisit10_step(gradient), -0.1)
        self.assertEqual(pop0_revisit10_step(-gradient), 0.1)
        self.assertGreater(abs(gradient), abs(-512.4156757903997) / 10.)
        self.assertEqual(abs(gradient) / 10., 36.223672696774145)
        self.assertEqual(0.3 + white, 0.2642831495222041)
        self.assertEqual(white - 0.1, -0.13571685047779586)
        self.assertEqual(0.3 + (white - 0.1), 0.16428314952220413)
        self.assertLessEqual(abs(-3.4822793321738716), abs(273.5525091954018) / 10.)
        self.assertLess(abs(-14.65730520606909), 542.8941177183742 / 10.)
        self.assertGreater(abs(-216.53859390363849), 23.834)
        self.assertGreaterEqual(min(abs(value) for value in derivatives), 1.)
        self.assertTrue(all(abs(value) < 1e-6 for value in diffs))
        self.assertEqual(gap, 9.313225746154785e-10)
        self.assertEqual(
            pop0_revisit10_status(objective, objective - 1., abs(gradient),
                                  abs(gradient) / 10. + 0.001, 1., 1.1),
            'CONDITIONAL_POP0_REVISIT10_IMPROVED')
        self.assertEqual(
            pop0_revisit10_status(objective, objective - 1., abs(gradient),
                                  abs(gradient) / 10. - 0.001, 1., 1.1),
            'CONDITIONAL_POP0_REVISIT10_REDUCED')
        self.assertEqual(
            pop0_revisit10_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_POP0_REVISIT10_NO_IMPROVEMENT')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        with self.assertRaises(ValueError):
            pop0_revisit10_step(0.)

    def test_pop0_revisit10_secant_stays_inside_the_sign_bracket(self):
        positive = -0.03571685047779585
        negative = -0.13571685047779586
        g_pos = 362.2367269677417
        g_neg = -326446.38296246435
        theta = tracer0_secant(positive, g_pos, negative, g_neg)
        obj0 = 8208510.739070199
        obj1 = 8224814.914291897
        count = -142342.5947882921
        fp0 = 5978.88956203627
        fp1 = -10325.27708797582
        prior0 = 4.180606498333799
        prior1 = 4.189178183381578
        ic = 8072142.026921511
        tracer_prior = 0.8263159343553188
        rise = obj1 - obj0
        dfp = fp1 - fp0
        dprior = prior1 - prior0
        residual = rise - (-dfp + dprior)
        derivatives = (
            -56.67514861204005, 35.295488773328586, 28.211046509269472,
            -31.183485327233306, -223.56921281996955, g_pos,
            -3.4822793321725074, -14.657305206069886, -216.5385939036377)
        self.assertEqual(theta, -0.03582769110596287)
        self.assertEqual(0.3 + theta, 0.2641723088940371)
        self.assertEqual(0.3 + positive, 0.2642831495222041)
        self.assertEqual(0.3 + negative, 0.16428314952220413)
        self.assertLess(negative, theta)
        self.assertLess(theta, positive)
        self.assertLess(positive - theta, theta - negative)
        self.assertEqual(rise, 16304.175221697427)
        self.assertEqual(residual, 2.874003257602453e-10)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(dprior - 0.5 * (negative ** 2 - positive ** 2), -3.1051550219984847e-16)
        self.assertEqual(ic + tracer_prior + prior0 - count - fp0 - obj0, 9.313225746154785e-10)
        self.assertEqual(ic + tracer_prior + prior1 - count - fp1 - obj1, 9.313225746154785e-10)
        self.assertGreater(abs(g_neg), abs(g_pos) / 10.)
        self.assertEqual(abs(g_pos) / 10., 36.223672696774166)
        self.assertGreaterEqual(min(abs(value) for value in derivatives), 1.)
        self.assertLessEqual(abs(-3.4822793321725074), abs(273.5525091954018) / 10.)
        self.assertEqual(
            pop0_revisit10_status(obj0, obj0, abs(g_pos), abs(g_pos), count + fp0, count + fp0),
            'CONDITIONAL_POP0_REVISIT10_NO_IMPROVEMENT')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        with self.assertRaises(ValueError):
            tracer0_secant(negative, g_neg, positive, g_pos)

    def test_full_gradient68_names_population5_line(self):
        gradient = -165.653905578772
        own_gate = abs(2.070679793428482) / 10.
        line_gate = abs(gradient) / 10.
        white = 0.0765438051588915
        pop8 = -0.5570775454883079
        diffs = {
            9: -7.503331289626658e-12,
            14: 4.547473508864641e-13,
            22: 3.2316371800789057e-12,
        }
        objective = 8208466.823045053
        ic = 8072142.0269214865
        tracer_prior = 0.9015597734294367
        population_prior = 4.181215025367132
        count = -142299.81701650965
        fp = 5980.1036677424645
        support_fp = 5980.103667742464
        self.assertEqual(16777216 + 14, 16777230)
        self.assertEqual(own_gate, 0.2070679793428482)
        self.assertNotEqual(own_gate, 0.207)
        self.assertEqual(format(own_gate, '.3f'), '0.207')
        self.assertEqual(line_gate, 16.5653905578772)
        self.assertEqual(format(line_gate, '.3f'), '16.565')
        self.assertGreater(abs(gradient), own_gate)
        self.assertEqual(pop5_line_step(gradient), 0.1)
        self.assertEqual(white + 0.1, 0.1765438051588915)
        self.assertEqual(0.5 * white, 0.03827190257944575)
        self.assertEqual(0.5 * (white + 0.1), 0.08827190257944575)
        with self.assertRaises(ValueError):
            pop5_line_step(0.)
        self.assertEqual(sorted(index for index, value in ((14, gradient), (17, pop8), (9, 6.440485892080976)) if abs(value) < 1.), [17])
        self.assertEqual(1e-4 * max(abs(pop8), 1.), 1e-4)
        self.assertGreater(abs(6.440485892080976), 1.)
        self.assertGreater(abs(-110.35181616702789), abs(-110.35181616702818) / 10.)
        self.assertGreater(abs(66.31587008741337), 0.3690110978312417)
        self.assertLessEqual(abs(-11.360197536685128), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(1.7168722314640068), abs(-233.42232564563201) / 10.)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 9)
        self.assertEqual(gradient - -165.65390557877245, 4.547473508864641e-13)
        self.assertEqual(fp - support_fp, 9.094947017729282e-13)
        self.assertNotEqual(fp, support_fp)
        self.assertEqual(ic + tracer_prior + population_prior - count - fp - objective, 0.0)
        self.assertEqual(format(1.7928503032390806, '.6f'), '1.792850')
        self.assertEqual(format(12.509202375426138, '.6f'), '12.509202')
        self.assertEqual(format(69.56273125699573, '.6f'), '69.562731')
        self.assertEqual(format(110.35181616702789, '.6f'), '110.351816')
        self.assertEqual(format(90.47937312748165, '.6f'), '90.479373')
        self.assertEqual(format(165.653905578772, '.6f'), '165.653906')
        self.assertEqual(float(100 * np.exp(0.5 * 0.4259753562964724)), 123.73694044672878)
        self.assertEqual(int(601.8449366189307), 601)
        self.assertEqual(format(10.903480529785156, '.2f'), '10.90')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')
        self.assertEqual(
            pop5_revisit3_status(2., 1., abs(gradient), abs(gradient), 0., 0.),
            'CONDITIONAL_POP5_REVISIT3_IMPROVED')
        self.assertEqual(
            pop5_revisit3_status(1., 1., 10., 10., 0., 0.),
            'CONDITIONAL_POP5_REVISIT3_NO_IMPROVEMENT')

    def test_pop5_revisit3_sign_change_queues_one_secant(self):
        accepted = 0.0765438051588915
        rejected = 0.1765438051588915
        g_neg = -165.6539055787712
        g_pos = 16860.96068764376
        pop8 = -0.5570775454889048
        pop8_rejected = -1424.4985310512973
        objective0 = 8208466.823045053
        objective1 = 8209306.294278104
        ic = 8072142.0269214865
        tracer_prior = 0.9015597734294367
        prior0 = 4.181215025367132
        prior1 = 4.193869405883021
        count = -142299.8170165097
        gradient_count = -142299.81701650965
        fp0 = 5980.103667742464
        fp1 = 5140.64508907185
        gradient_fp = 5980.1036677424645
        theta = tracer0_secant(rejected, g_pos, accepted, g_neg)
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (rejected ** 2 - accepted ** 2)
        identity0 = ic + tracer_prior + prior0 - count - fp0 - objective0
        identity1 = ic + tracer_prior + prior1 - count - fp1 - objective1
        verify = {
            0: -66.10804596100041,
            2: -67.80237395862164,
            3: -76.2790856241235,
            4: -51.73031764479333,
            5: -2.2213935103044147,
            6: -110.35181616703981,
            9: 6.440485892082568,
            12: 66.31587008741462,
            13: -11.360197536686833,
            14: g_neg,
            17: pop8,
            22: 1.716872231461031,
        }
        gradient = {
            0: -66.10804596106486,
            2: -67.80237395864611,
            3: -76.2790856241497,
            4: -51.730317644797694,
            5: -2.221393510304418,
            6: -110.35181616702789,
            9: 6.440485892080976,
            12: 66.31587008741337,
            13: -11.360197536685128,
            14: -165.653905578772,
            17: -0.5570775454883079,
            22: 1.7168722314640068,
        }
        rejected_values = {
            0: -66.10804596100066,
            2: -68.63924565087328,
            3: -78.98311628395416,
            4: -49.73374387637489,
            5: -2.2189803947761337,
            6: -121.54590170803087,
            9: -37564.863520596686,
            12: 28482.125710102035,
            13: 811.1032099212416,
            14: g_pos,
            17: pop8_rejected,
            22: -98.35997217469247,
        }
        diffs = {index: verify[index] - gradient[index] for index in verify}
        self.assertEqual(accepted + 0.1, rejected)
        self.assertEqual(pop5_line_step(g_neg), 0.1)
        self.assertEqual(pop5_line_step(g_pos), -0.1)
        self.assertEqual(theta, 0.07751671674194167)
        self.assertEqual(0.5 * accepted, 0.03827190257944575)
        self.assertEqual(0.5 * rejected, 0.08827190257944575)
        self.assertEqual(0.5 * theta, 0.03875835837097084)
        self.assertLess(accepted, theta)
        self.assertLess(theta, rejected)
        self.assertLess(theta - accepted, rejected - theta)
        self.assertEqual(theta - accepted, 0.0009729115830501706)
        self.assertEqual(rejected - theta, 0.09902708841694983)
        with self.assertRaises(ValueError):
            tracer0_secant(accepted, g_neg, rejected, g_pos)
        self.assertEqual(g_neg * g_pos, -2793083.989718313)
        self.assertLess(g_neg, 0.)
        self.assertGreater(g_pos, 0.)
        self.assertEqual(delta_objective, 839.4712330512702)
        self.assertGreater(delta_objective, 0.)
        self.assertEqual(delta_fp, -839.4585786706139)
        self.assertEqual(delta_prior, 0.01265438051588852)
        self.assertEqual(prior_residual, -6.314393452555578e-16)
        self.assertLess(prior_residual, 0.)
        self.assertEqual(residual, 1.405169314239174e-10)
        self.assertGreater(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, 0.15087890625)
        self.assertEqual(identity0, 0.0)
        self.assertEqual(identity1, -9.313225746154785e-10)
        self.assertLess(identity1, 0.)
        self.assertEqual(count - gradient_count, -5.820766091346741e-11)
        self.assertEqual(fp0 - gradient_fp, -9.094947017729282e-13)
        self.assertNotEqual(count, gradient_count)
        self.assertNotEqual(fp0, gradient_fp)
        self.assertEqual(
            ic + tracer_prior + prior0 - gradient_count - gradient_fp - objective0, 0.0)
        self.assertEqual(abs(g_neg) / 10., 16.56539055787712)
        self.assertEqual(abs(-165.653905578772) / 10., 16.5653905578772)
        self.assertNotEqual(abs(g_neg) / 10., abs(-165.653905578772) / 10.)
        self.assertEqual(format(abs(g_neg) / 10., '.3f'), '16.565')
        self.assertEqual(format(abs(-165.653905578772) / 10., '.3f'), '16.565')
        self.assertNotEqual(abs(g_neg) / 10., 16.565)
        self.assertEqual(abs(2.070679793428482) / 10., 0.2070679793428482)
        self.assertNotEqual(abs(2.070679793428482) / 10., 0.207)
        self.assertEqual(format(abs(2.070679793428482) / 10., '.3f'), '0.207')
        self.assertGreater(abs(g_neg), abs(2.070679793428482) / 10.)
        self.assertGreater(abs(g_pos), abs(g_neg) / 10.)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 0)
        self.assertEqual(diffs[0], 6.444622613344109e-11)
        self.assertEqual(sorted(index for index, value in verify.items() if abs(value) < 1.), [17])
        self.assertEqual(
            sorted(index for index, value in rejected_values.items() if abs(value) < 1.), [])
        self.assertLess(abs(pop8), 1.)
        self.assertGreater(abs(pop8_rejected), 1.)
        self.assertEqual(1e-4 * max(abs(pop8), 1.), 1e-4)
        self.assertGreater(abs(verify[9]), 1.)
        self.assertGreater(abs(verify[9]), 0.00022566027606149425)
        self.assertGreater(abs(verify[12]), 0.3690110978312417)
        self.assertLessEqual(abs(verify[13]), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(verify[22]), abs(-233.42232564563201) / 10.)
        self.assertGreater(abs(verify[6]), abs(-110.35181616702818) / 10.)
        self.assertEqual(16777216 + 14, 16777230)
        self.assertEqual(0.3 + -0.03583179970308488, 0.2641682002969151)
        self.assertEqual(0.5 * -0.4114823435353389, -0.20574117176766946)
        self.assertEqual(0.5 * -0.010011571272851271, -0.005005785636425636)
        self.assertEqual(float(100 * np.exp(0.5 * 0.4259753562964724)), 123.73694044672878)
        self.assertEqual(int(1088.8237474779598), 1088)
        self.assertEqual(format(13.91021728515625, '.2f'), '13.91')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        self.assertEqual(coordinate_line_action(False, False, False, False), 'midpoint')
        self.assertEqual(
            pop5_revisit3_status(objective0, objective1, abs(g_neg), abs(g_pos), 0., 0.),
            'CONDITIONAL_POP5_REVISIT3_NO_IMPROVEMENT')

    def test_pop5_revisit3_secant_reduced_queues_one_full_gradient(self):
        verified = 0.0765438051588915
        saved = 0.07751671674194167
        g_verify = -165.65390557877075
        g_saved = 2.1509295454801833
        pop8_verify = -0.5570775454882226
        pop8_saved = -14.555927116785622
        objective0 = 8208466.823045053
        objective1 = 8208466.743509629
        ic = 8072142.0269214865
        tracer_prior = 0.9015597734294367
        prior0 = 4.181215025367132
        prior1 = 4.181289969000256
        count = -142299.8170165097
        gradient_count = -142299.81701650965
        fp0 = 5980.103667742464
        fp1 = 5980.183278110035
        gradient_fp = 5980.1036677424645
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (saved ** 2 - verified ** 2)
        identity0 = ic + tracer_prior + prior0 - count - fp0 - objective0
        identity1 = ic + tracer_prior + prior1 - count - fp1 - objective1
        gradient_identity = ic + tracer_prior + prior0 - gradient_count - gradient_fp - objective0
        verify = {
            0: -66.10804596100076,
            2: -67.80237395862169,
            3: -76.27908562412361,
            4: -51.73031764479332,
            5: -2.2213935103044147,
            6: -110.35181616703979,
            9: 6.440485892077793,
            12: 66.31587008741485,
            13: -11.360197536686435,
            14: g_verify,
            17: pop8_verify,
            22: 1.7168722314703164,
        }
        line = {
            0: -66.10804596100041,
            2: -67.80237395862164,
            3: -76.2790856241235,
            4: -51.73031764479333,
            5: -2.2213935103044147,
            6: -110.35181616703981,
            9: 6.440485892082568,
            12: 66.31587008741462,
            13: -11.360197536686833,
            14: -165.6539055787712,
            17: -0.5570775454889048,
            22: 1.716872231461031,
        }
        saved_values = {
            0: -66.10804596100064,
            2: -67.80949119947883,
            3: -76.29400957892567,
            4: -51.7266329316885,
            5: -2.2213653011763306,
            6: -110.38270963860407,
            9: -359.7448550468642,
            12: 346.25405065266125,
            13: -3.701132775470633,
            14: g_saved,
            17: pop8_saved,
            22: 1.6557243220193445,
        }
        diffs = {index: verify[index] - line[index] for index in verify}
        self.assertEqual(tracer0_secant(0.1765438051588915, 16860.96068764376, verified, -165.6539055787712), saved)
        self.assertEqual(0.5 * saved, 0.03875835837097084)
        self.assertEqual(0.5 * verified, 0.03827190257944575)
        self.assertLess(verified, saved)
        self.assertLess(saved, 0.1765438051588915)
        self.assertLess(g_verify, 0.)
        self.assertGreater(g_saved, 0.)
        self.assertEqual(g_verify * g_saved, -356.30987983356255)
        self.assertGreater(abs(g_saved), 1.)
        self.assertGreater(abs(pop8_saved), 1.)
        self.assertLess(abs(pop8_verify), 1.)
        self.assertEqual(1e-4 * max(abs(pop8_verify), 1.), 1e-4)
        self.assertEqual(1e-4 * max(abs(g_saved), 1.), 0.00021509295454801834)
        self.assertNotEqual(1e-4 * max(abs(g_saved), 1.), 1e-4)
        self.assertEqual(1e-4 * max(abs(pop8_saved), 1.), 0.0014555927116785622)
        self.assertEqual(sorted(index for index, value in verify.items() if abs(value) < 1.), [17])
        self.assertEqual(sorted(index for index, value in saved_values.items() if abs(value) < 1.), [])
        self.assertEqual(abs(g_verify) / 10., 16.565390557877073)
        self.assertEqual(abs(-165.6539055787712) / 10., 16.56539055787712)
        self.assertEqual(abs(-165.653905578772) / 10., 16.5653905578772)
        self.assertNotEqual(abs(g_verify) / 10., abs(-165.6539055787712) / 10.)
        self.assertNotEqual(abs(g_verify) / 10., abs(-165.653905578772) / 10.)
        self.assertEqual(format(abs(g_verify) / 10., '.3f'), '16.565')
        self.assertEqual(format(abs(-165.6539055787712) / 10., '.3f'), '16.565')
        self.assertEqual(format(abs(-165.653905578772) / 10., '.3f'), '16.565')
        self.assertNotEqual(abs(g_verify) / 10., 16.565)
        self.assertEqual(abs(g_saved) / 10., 0.21509295454801833)
        self.assertNotEqual(abs(g_saved) / 10., 0.215)
        self.assertEqual(format(abs(g_saved) / 10., '.3f'), '0.215')
        self.assertEqual(abs(2.070679793428482) / 10., 0.2070679793428482)
        self.assertGreater(abs(g_saved), abs(2.070679793428482) / 10.)
        self.assertLessEqual(abs(g_saved), abs(g_verify) / 10.)
        self.assertEqual(delta_objective, -0.0795354237779975)
        self.assertLess(delta_objective, 0.)
        self.assertEqual(delta_fp, 0.07961036757114925)
        self.assertEqual(delta_prior, 7.494363312332553e-05)
        self.assertEqual(prior_residual, -7.116703060194851e-16)
        self.assertLess(prior_residual, 0.)
        self.assertEqual(residual, 1.6002843494788976e-10)
        self.assertGreater(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, 0.1718292236328125)
        self.assertEqual(identity0, 0.0)
        self.assertEqual(identity1, 0.0)
        self.assertEqual(gradient_identity, 0.0)
        self.assertEqual(count - gradient_count, -5.820766091346741e-11)
        self.assertEqual(fp0 - gradient_fp, -9.094947017729282e-13)
        self.assertNotEqual(count, gradient_count)
        self.assertNotEqual(fp0, gradient_fp)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 22)
        self.assertEqual(diffs[22], 9.28546128875496e-12)
        self.assertGreater(abs(saved_values[9]), 0.00022566027606149425)
        self.assertGreater(abs(saved_values[12]), 0.3690110978312417)
        self.assertLessEqual(abs(saved_values[13]), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(saved_values[22]), abs(-233.42232564563201) / 10.)
        self.assertGreater(abs(saved_values[6]), abs(-110.35181616702818) / 10.)
        self.assertEqual(16777216 + 14, 16777230)
        self.assertEqual(float(100 * np.exp(0.5 * 0.4259753562964724)), 123.73694044672878)
        self.assertEqual(int(1063.5091512040235), 1063)
        self.assertEqual(format(13.773536682128906, '.2f'), '13.77')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')
        self.assertEqual(coordinate_line_action(True, True, True, False), 'stop')
        self.assertEqual(coordinate_line_action(True, False, True, False), 'stop')
        self.assertEqual(
            pop5_revisit3_status(objective0, objective1, abs(g_verify), abs(g_saved), count + fp0, count + fp1),
            'CONDITIONAL_POP5_REVISIT3_REDUCED')
        self.assertEqual(
            full_gradient_record_status(True, True, True, True),
            'CONDITIONAL_FULL_GRADIENT_RECORDED')

    def test_full_gradient69_names_population2_line(self):
        gradient = -361.4068442163105
        own_gate = abs(-2.598336989703646) / 10.
        later = abs(-2.5983369897036246) / 10.
        line_gate = abs(gradient) / 10.
        white = 0.014827038075365416
        pop5 = 2.1509295454798423
        pop8 = -14.555927116784627
        pop13 = 1.6557243220254922
        secant = {
            0: -66.10804596100064,
            2: -67.80949119947883,
            3: -76.29400957892567,
            4: -51.7266329316885,
            5: -2.2213653011763306,
            6: -110.38270963860407,
            9: -359.7448550468642,
            12: 346.25405065266125,
            13: -3.701132775470633,
            14: 2.1509295454801833,
            17: -14.555927116785622,
            22: 1.6557243220193445,
        }
        measured = {
            0: -66.10804596100269,
            2: -67.80949119947897,
            3: -76.29400957892582,
            4: -51.726632931688926,
            5: -2.221365301176333,
            6: -110.38270963860411,
            9: -359.74485504686123,
            11: gradient,
            12: 346.25405065266136,
            13: -3.7011327754710877,
            14: pop5,
            17: pop8,
            22: pop13,
        }
        diffs = {index: measured[index] - secant[index] for index in secant}
        objective = 8208466.743509629
        ic = 8072142.0269214865
        tracer_prior = 0.9015597734294367
        population_prior = 4.181289969000256
        count = -142299.8170165097
        fp = 5980.183278110035
        self.assertEqual(16777216 + 11, 16777227)
        self.assertEqual(own_gate, 0.25983369897036457)
        self.assertNotEqual(own_gate, 0.260)
        self.assertEqual(format(own_gate, '.3f'), '0.260')
        self.assertNotEqual(own_gate, later)
        self.assertEqual(line_gate, 36.14068442163105)
        self.assertNotEqual(line_gate, 36.141)
        self.assertEqual(format(line_gate, '.3f'), '36.141')
        self.assertGreater(abs(gradient), own_gate)
        self.assertGreater(abs(gradient), 1.)
        self.assertEqual(pop2_revisit_step(gradient), 0.1)
        self.assertEqual(white + 0.1, 0.11482703807536543)
        self.assertEqual(2.7 + white, 2.7148270380753656)
        self.assertEqual(2.7 + white + 0.1, 2.8148270380753657)
        with self.assertRaises(ValueError):
            pop2_revisit_step(0.)
        self.assertEqual(sorted(index for index, value in measured.items() if abs(value) < 1.), [])
        self.assertGreater(abs(pop13), 1.)
        self.assertGreater(abs(pop8), 1.)
        self.assertGreater(abs(pop5), 1.)
        self.assertGreater(abs(pop5), 0.2070679793428482)
        self.assertLessEqual(abs(pop5), 16.565390557877073)
        self.assertGreater(abs(measured[9]), 0.00022566027606149425)
        self.assertGreater(abs(measured[12]), 0.3690110978312417)
        self.assertLessEqual(abs(measured[13]), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(pop13), abs(-233.42232564563201) / 10.)
        self.assertGreater(abs(measured[6]), abs(-110.35181616702818) / 10.)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 22)
        self.assertEqual(diffs[22], 6.147748976559342e-12)
        self.assertEqual(ic + tracer_prior + population_prior - count - fp - objective, 0.0)
        self.assertEqual(format(1.7928608629419727, '.6f'), '1.792861')
        self.assertEqual(format(12.509156068190354, '.6f'), '12.509156')
        self.assertEqual(format(69.57035421572427, '.6f'), '69.570354')
        self.assertEqual(format(110.38270963860411, '.6f'), '110.382710')
        self.assertEqual(format(174.13833661181405, '.6f'), '174.138337')
        self.assertEqual(format(abs(gradient), '.6f'), '361.406844')
        self.assertEqual(float(100 * np.exp(0.5 * 0.4259753562964724)), 123.73694044672878)
        self.assertEqual(int(595.6773382080719), 595)
        self.assertEqual(format(10.898468017578125, '.2f'), '10.90')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')
        self.assertEqual(
            pop2_revisit_status(1., 1., abs(gradient), abs(gradient), 0., 0.),
            'CONDITIONAL_POP2_REVISIT_NO_IMPROVEMENT')
        self.assertEqual(
            pop2_revisit_status(2., 1., abs(gradient), abs(gradient) / 10., 0., 0.),
            'CONDITIONAL_POP2_REVISIT_REDUCED')
        self.assertEqual(
            pop2_revisit_status(2., 1., 10., 9., 0., 0.),
            'CONDITIONAL_POP2_REVISIT_IMPROVED')

    def test_pop2_revisit_sign_change_queues_one_secant(self):
        accepted = 0.014827038075365416
        rejected = 0.11482703807536543
        g_neg = -361.40684421630937
        g_pos = 112580.59949570238
        objective0 = 8208466.743509629
        objective1 = 8214077.707584847
        ic = 8072142.0269214865
        tracer_prior = 0.9015597734294367
        prior0 = 4.181289969000256
        prior1 = 4.187772672807792
        count0 = -142299.81701650968
        gradient_count = -142299.8170165097
        fp0 = 5980.183278110035
        fp1 = 369.22568559562205
        theta = tracer0_secant(rejected, g_pos, accepted, g_neg)
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (rejected ** 2 - accepted ** 2)
        identity0 = ic + tracer_prior + prior0 - count0 - fp0 - objective0
        identity1 = ic + tracer_prior + prior1 - count0 - fp1 - objective1
        gradient_identity = ic + tracer_prior + prior0 - gradient_count - fp0 - objective0
        verify = {
            0: -66.10804596100081,
            2: -67.80949119947864,
            3: -76.29400957892557,
            4: -51.72663293168863,
            5: -2.2213653011763324,
            6: -110.38270963860398,
            9: -359.74485504686317,
            11: g_neg,
            12: 346.25405065266125,
            13: -3.701132775470576,
            14: 2.150929545479956,
            17: -14.555927116785139,
            22: 1.655724322020564,
        }
        gradient = {
            0: -66.10804596100269,
            2: -67.80949119947897,
            3: -76.29400957892582,
            4: -51.726632931688926,
            5: -2.221365301176333,
            6: -110.38270963860411,
            9: -359.74485504686123,
            11: -361.4068442163105,
            12: 346.25405065266136,
            13: -3.7011327754710877,
            14: 2.1509295454798423,
            17: -14.555927116784627,
            22: 1.6557243220254922,
        }
        rejected_values = {
            0: -66.10804596100091,
            2: -66.6608938559818,
            3: -72.28615266877242,
            4: -50.262242310357706,
            5: -2.2200581069183056,
            6: -108.51064119269658,
            9: 188060.34455785493,
            11: g_pos,
            12: -37755.41005556856,
            13: -1031.7345711048367,
            14: -22838.530550860014,
            17: -16.322422136518806,
            22: 24.459043609187965,
        }
        diffs = {index: verify[index] - gradient[index] for index in verify}
        self.assertEqual(accepted + 0.1, rejected)
        self.assertEqual(pop2_revisit_step(g_neg), 0.1)
        self.assertEqual(pop2_revisit_step(g_pos), -0.1)
        self.assertEqual(theta, 0.015147031367434796)
        self.assertEqual(2.7 + accepted, 2.7148270380753656)
        self.assertEqual(2.7 + rejected, 2.8148270380753657)
        self.assertEqual(2.7 + theta, 2.715147031367435)
        self.assertLess(accepted, theta)
        self.assertLess(theta, rejected)
        self.assertLess(theta - accepted, rejected - theta)
        self.assertEqual(theta - accepted, 0.00031999329206938015)
        self.assertEqual(rejected - theta, 0.09968000670793063)
        with self.assertRaises(ValueError):
            tracer0_secant(accepted, g_neg, rejected, g_pos)
        self.assertEqual(g_neg * g_pos, -40687399.18372203)
        self.assertLess(g_neg, 0.)
        self.assertGreater(g_pos, 0.)
        self.assertEqual(delta_objective, 5610.964075217955)
        self.assertGreater(delta_objective, 0.)
        self.assertEqual(delta_fp, -5610.957592514413)
        self.assertEqual(delta_prior, 0.006482703807536794)
        self.assertEqual(prior_residual, 2.5066754227864863e-16)
        self.assertGreater(prior_residual, 0.)
        self.assertEqual(residual, -2.6557245291769505e-10)
        self.assertLess(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(identity0, 0.0)
        self.assertEqual(identity1, 0.0)
        self.assertEqual(gradient_identity, 0.0)
        self.assertEqual(count0 - gradient_count, 2.9103830456733704e-11)
        self.assertNotEqual(count0, gradient_count)
        self.assertEqual(abs(g_neg) / 10., 36.14068442163094)
        self.assertEqual(abs(-361.4068442163105) / 10., 36.14068442163105)
        self.assertNotEqual(abs(g_neg) / 10., abs(-361.4068442163105) / 10.)
        self.assertEqual(format(abs(g_neg) / 10., '.3f'), '36.141')
        self.assertNotEqual(abs(g_neg) / 10., 36.141)
        self.assertEqual(abs(-2.598336989703646) / 10., 0.25983369897036457)
        self.assertGreater(abs(g_neg), 0.25983369897036457)
        self.assertGreater(abs(g_pos), abs(g_neg) / 10.)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 22)
        self.assertEqual(diffs[22], -4.92828000631107e-12)
        self.assertEqual(max(rejected_values, key=lambda index: abs(rejected_values[index])), 9)
        self.assertEqual(sorted(index for index, value in verify.items() if abs(value) < 1.), [])
        self.assertEqual(
            sorted(index for index, value in rejected_values.items() if abs(value) < 1.), [])
        self.assertGreater(abs(verify[22]), 1.)
        self.assertNotEqual(1e-4 * max(abs(verify[22]), 1.), 1e-4)
        self.assertEqual(16777216 + 11, 16777227)
        self.assertEqual(float(100 * np.exp(0.5 * 0.4259753562964724)), 123.73694044672878)
        self.assertEqual(int(1054.1034470399609), 1054)
        self.assertEqual(format(14.055519104003906, '.2f'), '14.06')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        self.assertEqual(coordinate_line_action(False, False, False, False), 'midpoint')
        self.assertEqual(
            pop2_revisit_status(objective0, objective1, abs(g_neg), abs(g_pos), 0., 0.),
            'CONDITIONAL_POP2_REVISIT_NO_IMPROVEMENT')

    def test_pop0_revisit16_secant_reduced_queues_one_full_gradient(self):
        white = -0.03590646609168676
        secant_white = -0.03601817746630569
        gradient = 363.8563766673434
        line_gradient = 363.8563766673452
        saved_gradient = 0.0019222290782157786
        self.assertEqual(secant_white, -0.03601817746630569)
        self.assertEqual(white - secant_white, 0.0001117113746189266)
        self.assertEqual(secant_white - white, -0.0001117113746189266)
        self.assertEqual(0.3 + secant_white, 0.2639818225336943)
        self.assertEqual((0.3 + white) + (secant_white - white), 0.26398182253369434)
        self.assertNotEqual(0.3 + secant_white, (0.3 + white) + (secant_white - white))
        self.assertGreater(saved_gradient, 0.)
        self.assertGreater(gradient, 0.)
        self.assertEqual(line_gradient * saved_gradient, 0.6994153075242041)
        self.assertEqual(gradient * saved_gradient, 0.6994153075242007)
        self.assertNotEqual(line_gradient * saved_gradient, gradient * saved_gradient)
        self.assertGreater(line_gradient * saved_gradient, 0.)
        self.assertEqual(8208466.580615268 - 8208466.600938824, -0.020323555916547775)
        self.assertEqual(-(0.020327572798123583) + 4.017400400790905e-06 + (-5.188249829757297e-10), -0.020323555916547775)
        self.assertEqual(0.020327572798123583, 0.020327572798123583)
        self.assertEqual(-5.188249829757297e-10 / 9.313225746154785e-10, -0.5570840835571289)
        self.assertLess(-5.188249829757297e-10, 0.)
        self.assertGreater(3.710139834245396e-16, 0.)
        self.assertEqual(-142299.81701650968 - -142299.81701650965, -2.9103830456733704e-11)
        self.assertEqual(-142299.81701650968 - -142299.8170165097, 2.9103830456733704e-11)
        self.assertEqual(abs(-2.9103830456733704e-11) / 9.313225746154785e-10, 0.03125)
        self.assertNotEqual(-142299.81701650968, -142299.81701650965)
        self.assertLess(abs(saved_gradient), 1.)
        self.assertEqual(1e-4 * max(abs(saved_gradient), 1.), 1e-4)
        self.assertNotEqual(1e-4 * max(abs(gradient), 1.), 1e-4)
        self.assertGreaterEqual(abs(-1.1346355032254354), 1.)
        self.assertLessEqual(abs(saved_gradient), abs(gradient) / 10.)
        self.assertGreater(abs(saved_gradient), 0.00022566027606149425)
        self.assertGreater(abs(saved_gradient), 0.00013553813665528116)
        self.assertEqual(format(saved_gradient, '.3f'), '0.002')
        self.assertNotEqual(saved_gradient, 0.002)
        self.assertEqual(abs(saved_gradient) / 10., 0.00019222290782157787)
        self.assertEqual(format(abs(saved_gradient) / 10., '.3f'), '0.000')
        self.assertNotEqual(abs(gradient) / 10., abs(line_gradient) / 10.)
        self.assertEqual(gradient - line_gradient, -1.8189894035458565e-12)
        self.assertGreater(abs(6.80842049405328e-11), abs(-1.8189894035458565e-12))
        self.assertEqual(coordinate_line_action(True, False, True, False), 'stop')
        self.assertEqual(coordinate_line_action(True, False, False, False), 'continue')
        self.assertEqual(
            pop0_revisit16_status(
                8208466.600938824, 8208466.580615268, abs(gradient), abs(saved_gradient),
                0., 0.020327572798123583),
            'CONDITIONAL_POP0_REVISIT16_REDUCED')
        self.assertGreater(abs(-160.82954304664483), 1.)
        self.assertGreater(abs(68.89362536612903), 0.3690110978312417)
        self.assertEqual(int(1073.2541597329546), 1073)
        self.assertEqual(format(13.795707702636719, '.2f'), '13.80')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')

    def test_pop0_revisit16_sign_stop_queues_one_secant(self):
        white = -0.03590646609168676
        failed_white = -0.13590646609168677
        gradient = 363.8563766673452
        failed_gradient = -325347.270094148
        theta = tracer0_secant(white, gradient, failed_white, failed_gradient)
        self.assertEqual(theta, -0.03601817746630569)
        self.assertEqual(pop0_revisit15_step(gradient), -0.1)
        self.assertEqual(pop0_revisit15_step(failed_gradient), 0.1)
        self.assertEqual(failed_white - white, -0.1)
        self.assertEqual(failed_white, white - 0.1)
        self.assertEqual(white - theta, 0.0001117113746189266)
        self.assertEqual(theta - failed_white, 0.09988828862538109)
        self.assertEqual((white - theta) + (theta - failed_white), 0.1)
        self.assertLess(white - theta, theta - failed_white)
        self.assertEqual(0.3 + theta, 0.2639818225336943)
        self.assertEqual((0.3 + white) + (theta - white), 0.26398182253369434)
        self.assertNotEqual(0.3 + theta, (0.3 + white) + (theta - white))
        self.assertEqual(gradient * failed_gradient, -118379678.85506882)
        self.assertLess(gradient * failed_gradient, 0.)
        with self.assertRaises(ValueError):
            tracer0_secant(failed_white, failed_gradient, white, gradient)
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        self.assertEqual(coordinate_line_action(False, False, False, False), 'midpoint')
        self.assertEqual(8224715.739591803 - 8208466.600938824, 16249.138652979396)
        self.assertEqual(-(-16249.13006233332) + 0.00859064660916875 + (-5.329638952389359e-10), 16249.138652979396)
        self.assertEqual(-5.329638952389359e-10 / 9.313225746154785e-10, -0.572265625)
        self.assertLess(-5.329638952389359e-10, 0.)
        self.assertLess(abs(-5.329638952389359e-10), 1e-9)
        self.assertEqual(7.28583859910259e-17, 0.00859064660916875 - 0.5 * (failed_white ** 2 - white ** 2))
        self.assertGreater(7.28583859910259e-17, 0.)
        self.assertEqual(-142299.81701650965 - -142299.8170165097, 5.820766091346741e-11)
        self.assertEqual(5.820766091346741e-11, 2 * 2.9103830456733704e-11)
        self.assertEqual(5.820766091346741e-11 / 9.313225746154785e-10, 0.0625)
        self.assertNotEqual(-142299.81701650965, -142299.8170165097)
        self.assertGreater(abs(-6.369305083353538e-11), abs(1.3642420526593924e-12))
        self.assertLess(abs(0.5899675730594862), 1.)
        self.assertGreaterEqual(abs(-1.1368095880749212), 1.)
        self.assertEqual(1e-4 * max(abs(0.5899675730594862), 1.), 1e-4)
        self.assertNotEqual(1e-4 * max(abs(gradient), 1.), 1e-4)
        self.assertNotEqual(1e-4 * max(abs(-1.1368095880749212), 1.), 1e-4)
        self.assertEqual(abs(gradient) / 10., 36.385637666734524)
        self.assertEqual(abs(363.85637666734385) / 10., 36.38563766673438)
        self.assertNotEqual(abs(gradient) / 10., abs(363.85637666734385) / 10.)
        self.assertEqual(format(abs(gradient) / 10., '.3f'), '36.386')
        self.assertNotEqual(abs(gradient) / 10., 36.386)
        self.assertGreater(abs(failed_gradient), abs(gradient) / 10.)
        self.assertGreater(abs(-3.7831335430786783), 0.3690110978312417)
        self.assertGreater(abs(-188359.656881561), 1.)
        self.assertEqual(
            pop0_revisit16_status(8208466.600938824, 8208466.600938824, abs(gradient), abs(gradient), 0., 0.),
            'CONDITIONAL_POP0_REVISIT16_NO_IMPROVEMENT')
        self.assertEqual(int(1066.4839881359367), 1066)
        self.assertEqual(format(13.803665161132812, '.2f'), '13.80')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')

    def test_full_gradient72_names_one_population0_line(self):
        white = -0.03590646609168676
        gradient = 363.85637666734385
        secant_gradient = 363.8563766673384
        step_white = white - 0.1
        count = -142299.8170165097
        count_secant = -142299.81701650968
        residual = 0. - (-(count - count_secant) - 0.)
        self.assertEqual(pop0_revisit15_step(gradient), -0.1)
        self.assertEqual(step_white, -0.13590646609168677)
        self.assertEqual(step_white, white - 0.1)
        self.assertEqual(step_white - white, -0.1)
        self.assertEqual(0.3 + white, 0.26409353390831325)
        self.assertEqual(0.3 + step_white, 0.16409353390831322)
        self.assertEqual((0.3 + white) - 0.1, 0.16409353390831324)
        self.assertNotEqual(0.3 + step_white, (0.3 + white) - 0.1)
        self.assertEqual(16777216 + 9, 16777225)
        self.assertGreater(gradient, 0.)
        self.assertGreater(abs(gradient), 0.00022566027606149425)
        self.assertGreater(abs(gradient), 0.00013553813665528116)
        self.assertNotEqual(0.00022566027606149425, 0.00013553813665528116)
        self.assertEqual(format(0.00022566027606149425, '.3f'), '0.000')
        self.assertNotEqual(0.00022566027606149425, 0.000)
        self.assertEqual(abs(gradient) / 10., 36.38563766673438)
        self.assertEqual(abs(secant_gradient) / 10., 36.38563766673384)
        self.assertNotEqual(abs(gradient) / 10., abs(secant_gradient) / 10.)
        self.assertEqual(format(abs(gradient) / 10., '.3f'), '36.386')
        self.assertNotEqual(abs(gradient) / 10., 36.386)
        self.assertEqual(gradient - secant_gradient, 5.4569682106375694e-12)
        self.assertEqual(2.052449931187877 - 2.0524499311933515, -5.474287689821722e-12)
        self.assertGreater(abs(2.052449931187877 - 2.0524499311933515), abs(gradient - secant_gradient))
        self.assertEqual(residual, -2.9103830456733704e-11)
        self.assertLess(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, -0.03125)
        self.assertNotEqual(count, count_secant)
        self.assertGreaterEqual(abs(-1.136809588076456), 1.)
        self.assertNotEqual(1e-4 * max(abs(gradient), 1.), 1e-4)
        self.assertNotEqual(1e-4 * max(abs(-1.136809588076456), 1.), 1e-4)
        self.assertGreater(abs(-3.783133543077769), 0.3690110978312417)
        self.assertGreater(abs(69.85096613742863), 1.)
        self.assertGreater(abs(-202.8769333836278), abs(-165.65390557877075) / 10.)
        self.assertEqual(
            pop0_revisit16_status(8208466.600938824, 8208466.600938824, abs(gradient), abs(gradient), 0., 0.),
            'CONDITIONAL_POP0_REVISIT16_NO_IMPROVEMENT')
        self.assertEqual(int(594.8791151460027), 594)
        self.assertEqual(format(10.918670654296875, '.2f'), '10.92')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')
        self.assertEqual(format(1.7928202389528087, '.6f'), '1.792820')
        self.assertEqual(format(12.509033652817559, '.6f'), '12.509034')
        self.assertEqual(format(69.5599358316341, '.6f'), '69.559936')
        self.assertEqual(format(110.34629557211788, '.6f'), '110.346296')
        self.assertEqual(format(128.42481811473633, '.6f'), '128.424818')
        self.assertEqual(format(363.85637666734385, '.6f'), '363.856377')

    def test_pop3_revisit8_secant_reduced_queues_one_full_gradient(self):
        accepted = -0.4114823435353389
        theta = -0.41204161762975566
        g_line = 274.3729540603854
        g_verify = 274.37295406038595
        g_saved = -3.7831335430764046
        objective0 = 8208466.676606872
        objective1 = 8208466.600938824
        count = -142299.81701650968
        fp0 = 5980.250188340764
        fp1 = 5980.326086677034
        prior0 = 4.18129744296945
        prior1 = 4.181527730778256
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = (objective1 - objective0) - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (theta ** 2 - accepted ** 2)
        like = (count + fp1) - (count + fp0)
        saved = (
            363.8563766673384, -202.87693338362763, -110.34629557211787,
            -76.27410325584113, 69.85096613742567, -67.80200415103899,
            -66.10804596099972, -51.726764962676086, -18.577442868289673,
            g_saved, -2.2213843274495666, 2.0524499311933515, -1.1368095880757738)
        self.assertEqual(0.5 * theta, -0.20602080881487783)
        self.assertEqual(accepted - theta, 0.0005592740944167507)
        self.assertLess(g_saved, 0.)
        self.assertGreater(g_line, 0.)
        self.assertGreater(g_verify, 0.)
        self.assertEqual(g_saved * g_line, -1037.9895258188053)
        self.assertEqual(g_saved * g_verify, -1037.9895258188076)
        self.assertNotEqual(g_saved * g_line, g_saved * g_verify)
        self.assertLess(g_saved * g_line, 0.)
        self.assertLess(abs(g_saved), abs(g_line) / 10.)
        self.assertLess(abs(g_saved), abs(g_verify) / 10.)
        self.assertLess(abs(g_saved), abs(274.37295406038663) / 10.)
        self.assertGreater(abs(g_saved), 0.3690110978312417)
        self.assertGreaterEqual(abs(g_saved), 1.)
        self.assertEqual(abs(g_saved) / 10., 0.37831335430764046)
        self.assertEqual(format(abs(g_saved) / 10., '.3f'), '0.378')
        self.assertNotEqual(abs(g_saved) / 10., 0.378)
        self.assertEqual(format(0.3690110978312417, '.3f'), '0.369')
        self.assertNotEqual(0.3690110978312417, 0.369)
        self.assertNotEqual(abs(g_saved) / 10., 0.3690110978312417)
        self.assertEqual(objective1 - objective0, -0.07566804811358452)
        self.assertLess(objective1, objective0)
        self.assertEqual(delta_fp, 0.07589833627025655)
        self.assertEqual(like, 0.07589833627571352)
        self.assertNotEqual(like, delta_fp)
        self.assertEqual(delta_prior, 0.00023028780880540722)
        self.assertEqual(residual, 3.4786662439501015e-10)
        self.assertGreater(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, 0.3735189437866211)
        self.assertEqual(prior_residual, -1.5265566588595902e-16)
        self.assertLess(prior_residual, 0.)
        self.assertEqual(coordinate_line_action(True, True, True, False), 'stop')
        self.assertEqual(coordinate_line_action(True, False, True, False), 'stop')
        self.assertEqual(coordinate_line_action(True, False, False, False), 'continue')
        self.assertEqual(
            pop3_revisit8_status(objective0, objective1, abs(g_line), abs(g_saved), count + fp0, count + fp1),
            'CONDITIONAL_POP3_REVISIT8_REDUCED')
        self.assertEqual(
            pop3_revisit8_status(objective0, objective1, abs(g_verify), abs(g_saved), count + fp0, count + fp1),
            'CONDITIONAL_POP3_REVISIT8_REDUCED')
        self.assertTrue(all(abs(value) >= 1. for value in saved))
        self.assertGreater(abs(363.8563766673384), abs(g_saved))
        self.assertNotEqual(1e-4 * max(abs(g_saved), 1.), 1e-4)
        self.assertNotEqual(1e-4 * max(abs(363.8563766673384), 1.), 1e-4)
        self.assertEqual(0.0013553813713276588 - 0.0013553813615505908, 9.777068044058979e-12)
        self.assertEqual(g_verify - g_line, 5.684341886080801e-13)
        self.assertEqual(int(1070.2300533570815), 1070)
        self.assertEqual(format(13.798660278320312, '.2f'), '13.80')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')

    def test_pop3_revisit8_sign_stop_queues_one_secant(self):
        accepted = -0.4114823435353389
        rejected = -0.511482343535339
        g_pos = 274.3729540603854
        g_neg = -48784.3903248845
        g_gradient = 274.37295406038595
        g_stored = 274.37295406038663
        objective0 = 8208466.676606872
        objective1 = 8210912.244360311
        fp0 = 5980.250188340764
        fp_gradient = 5980.250188340765
        fp1 = 3534.728583135924
        prior0 = 4.18129744296945
        prior1 = 4.227445677322985
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = (objective1 - objective0) - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (rejected ** 2 - accepted ** 2)
        theta = tracer0_secant(accepted, g_pos, rejected, g_neg)
        pop0 = 0.0013553813615505908
        pop0_gradient = 0.0013553813672349327
        pop0_saved = 0.0013553813665528117
        self.assertEqual(rejected, accepted - 0.1)
        self.assertEqual(rejected - accepted, -0.10000000000000003)
        self.assertNotEqual(rejected - accepted, -0.1)
        self.assertEqual(0.5 * accepted, -0.20574117176766946)
        self.assertEqual(0.5 * rejected, -0.2557411717676695)
        self.assertEqual(0.5 * accepted - 0.05, -0.2557411717676695)
        self.assertEqual(0.5 * rejected, 0.5 * accepted - 0.05)
        self.assertEqual(pop3_revisit5_step(g_pos), -0.1)
        self.assertEqual(pop3_revisit5_step(g_neg), 0.1)
        self.assertGreater(g_pos, 0.)
        self.assertLess(g_neg, 0.)
        self.assertEqual(g_pos * g_neg, -13385117.285473445)
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        self.assertEqual(coordinate_line_action(False, False, False, False), 'midpoint')
        self.assertEqual(theta, -0.41204161762975566)
        self.assertEqual(0.5 * theta, -0.20602080881487783)
        self.assertLess(rejected, theta)
        self.assertLess(theta, accepted)
        self.assertLess(accepted - theta, theta - rejected)
        self.assertEqual(accepted - theta, 0.0005592740944167507)
        self.assertEqual(theta - rejected, 0.09944072590558328)
        with self.assertRaises(ValueError):
            tracer0_secant(rejected, g_neg, accepted, g_pos)
        self.assertEqual(objective1 - objective0, 2445.567753438838)
        self.assertGreater(objective1, objective0)
        self.assertEqual(delta_fp, -2445.52160520484)
        self.assertEqual(delta_prior, 0.046148234353534434)
        self.assertEqual(residual, -3.5561242839321494e-10)
        self.assertLess(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, -0.3818359375)
        self.assertEqual(prior_residual, 5.134781488891349e-16)
        self.assertGreater(prior_residual, 0.)
        self.assertEqual(fp0 - fp_gradient, -9.094947017729282e-13)
        self.assertNotEqual(fp0, fp_gradient)
        self.assertEqual(abs(g_pos) / 10., 27.437295406038537)
        self.assertEqual(abs(g_gradient) / 10., 27.437295406038594)
        self.assertEqual(abs(g_stored) / 10., 27.437295406038665)
        self.assertNotEqual(abs(g_pos) / 10., abs(g_gradient) / 10.)
        self.assertNotEqual(abs(g_pos) / 10., abs(g_stored) / 10.)
        self.assertEqual(format(abs(g_pos) / 10., '.3f'), '27.437')
        self.assertNotEqual(abs(g_pos) / 10., 27.437)
        self.assertGreater(abs(g_pos), 0.3690110978312417)
        self.assertGreater(abs(g_neg), abs(g_pos) / 10.)
        self.assertLess(pop0, 1.)
        self.assertGreater(pop0, 0.)
        self.assertEqual(pop0 - pop0_gradient, -5.6843418860808015e-12)
        self.assertNotEqual(pop0, pop0_gradient)
        self.assertNotEqual(pop0, pop0_saved)
        self.assertNotEqual(pop0_gradient, pop0_saved)
        self.assertGreater(abs(pop0), 0.00022566027606149425)
        self.assertGreater(abs(pop0), 0.00013553813665528116)
        self.assertEqual(1e-4 * max(abs(pop0), 1.), 1e-4)
        self.assertNotEqual(1e-4 * max(abs(g_pos), 1.), 1e-4)
        self.assertGreater(abs(-140.68574842391158), 1.)
        self.assertGreater(abs(-140.68574842391158), abs(-2.598336989703646) / 10.)
        self.assertGreater(abs(64946.48867484529), abs(g_neg))
        self.assertEqual(
            pop3_revisit8_status(objective0, objective0, abs(g_pos), abs(g_pos), 0., 0.),
            'CONDITIONAL_POP3_REVISIT8_NO_IMPROVEMENT')
        self.assertEqual(int(1072.1561975330114), 1072)
        self.assertEqual(format(13.800285339355469, '.2f'), '13.80')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')

    def test_full_gradient71_names_one_population3_line(self):
        white = -0.4114823435353389
        step_white = white - 0.1
        gradient = 274.37295406038595
        secant_gradient = 274.37295406038663
        pop0 = 0.0013553813672349327
        pop0_secant = 0.0013553813665528117
        objective = 8208466.676606872
        count = -142299.81701650968
        count_secant = -142299.8170165097
        fp = 5980.250188340765
        fp_secant = 5980.250188340764
        delta_count = count - count_secant
        delta_fp = fp - fp_secant
        residual = 0. - (-delta_count - delta_fp)
        self.assertEqual(pop3_revisit5_step(gradient), -0.1)
        self.assertEqual(step_white, -0.511482343535339)
        self.assertEqual(0.5 * white, -0.20574117176766946)
        self.assertEqual(0.5 * step_white, -0.2557411717676695)
        self.assertEqual(0.5 * white - 0.05, -0.2557411717676695)
        self.assertEqual(0.5 * step_white, 0.5 * white - 0.05)
        self.assertEqual(16777216 + 12, 16777228)
        self.assertGreater(gradient, 0.)
        self.assertGreater(abs(gradient), 0.3690110978312417)
        self.assertEqual(abs(gradient) / 10., 27.437295406038594)
        self.assertEqual(abs(secant_gradient) / 10., 27.437295406038665)
        self.assertNotEqual(abs(gradient) / 10., abs(secant_gradient) / 10.)
        self.assertEqual(format(abs(gradient) / 10., '.3f'), '27.437')
        self.assertNotEqual(abs(gradient) / 10., 27.437)
        self.assertEqual(gradient - secant_gradient, -6.821210263296962e-13)
        self.assertLess(pop0, 1.)
        self.assertGreater(pop0, 0.)
        self.assertGreater(pop0, 0.00022566027606149425)
        self.assertEqual(pop0 - pop0_secant, 6.821210263296962e-13)
        self.assertEqual(1e-4 * max(abs(pop0), 1.), 1e-4)
        self.assertNotEqual(1e-4 * max(abs(gradient), 1.), 1e-4)
        self.assertNotEqual(1e-4 * max(abs(-14.560811461035936), 1.), 1e-4)
        self.assertGreater(abs(-140.6857484239125), 1.)
        self.assertGreater(abs(-140.6857484239125), abs(-2.598336989703646) / 10.)
        self.assertEqual(delta_count, 2.9103830456733704e-11)
        self.assertEqual(delta_fp, 9.094947017729282e-13)
        self.assertNotEqual(count, count_secant)
        self.assertNotEqual(fp, fp_secant)
        self.assertEqual(residual, 3.001332515850663e-11)
        self.assertGreater(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, 0.0322265625)
        self.assertEqual(
            pop3_revisit8_status(objective, objective, abs(gradient), abs(gradient), 0., 0.),
            'CONDITIONAL_POP3_REVISIT8_NO_IMPROVEMENT')
        self.assertEqual(int(596.1693409350701), 596)
        self.assertEqual(format(10.909072875976562, '.2f'), '10.91')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')
        self.assertEqual(format(1.7928244037861076, '.6f'), '1.792824')
        self.assertEqual(format(12.5089728409645, '.6f'), '12.508973')
        self.assertEqual(format(69.56626969259275, '.6f'), '69.566270')
        self.assertEqual(format(110.37166665321197, '.6f'), '110.371667')
        self.assertEqual(format(106.74517704542696, '.6f'), '106.745177')
        self.assertEqual(format(274.37295406038595, '.6f'), '274.372954')

    def test_pop0_revisit15_secant_reduced_queues_one_full_gradient(self):
        accepted = -0.03583179970308488
        saved = -0.03590646609168676
        g_line = 243.18828792633946
        g_verify = 243.18828792632854
        g_saved = 0.0013553813665528117
        objective0 = 8208466.685685918
        objective1 = 8208466.676606872
        ic = 8072142.0269214865
        tracer_prior = 0.9015597734294367
        prior0 = 4.1812947647508345
        prior1 = 4.18129744296945
        count_line = -142299.81701650965
        count = -142299.8170165097
        count_gradient = -142299.81701650968
        fp0 = 5980.241106616332
        fp1 = 5980.250188340764
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (saved ** 2 - accepted ** 2)
        identity0 = ic + tracer_prior + prior0 - count - fp0 - objective0
        identity1 = ic + tracer_prior + prior1 - count - fp1 - objective1
        likelihood_line = count_line + fp0
        likelihood_verify = count + fp0
        likelihood_saved = count + fp1
        self.assertEqual(saved, -0.03590646609168676)
        self.assertEqual(0.3 + saved, 0.26409353390831325)
        self.assertEqual(accepted - saved, 7.466638860188085e-05)
        self.assertNotEqual(saved, -0.1358317997030849)
        self.assertEqual(16777216 + 9, 16777225)
        self.assertGreater(g_line, 0.)
        self.assertGreater(g_verify, 0.)
        self.assertGreater(g_saved, 0.)
        self.assertLess(g_saved, 1.)
        self.assertEqual(g_line * g_saved, 0.3296128740192406)
        self.assertEqual(g_verify * g_saved, 0.3296128740192258)
        self.assertNotEqual(g_line * g_saved, g_verify * g_saved)
        self.assertEqual(delta_objective, -0.009079045616090298)
        self.assertLess(delta_objective, 0.)
        self.assertEqual(delta_fp, 0.009081724431780458)
        self.assertEqual(delta_prior, 2.6782186157703336e-06)
        self.assertEqual(residual, 5.970743899297304e-10)
        self.assertGreater(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, 0.6411037445068359)
        self.assertEqual(prior_residual, 4.163336342344337e-17)
        self.assertGreater(prior_residual, 0.)
        self.assertEqual(identity0, 0.0)
        self.assertEqual(identity1, 0.0)
        self.assertEqual(count - count_line, -5.820766091346741e-11)
        self.assertEqual(count - count_gradient, -2.9103830456733704e-11)
        self.assertNotEqual(count, count_line)
        self.assertNotEqual(count, count_gradient)
        self.assertNotEqual(fp0, 5980.241106616331)
        self.assertEqual(likelihood_saved - likelihood_line, 0.009081724361749366)
        self.assertEqual(likelihood_saved - likelihood_verify, 0.009081724419957027)
        self.assertGreater(likelihood_saved, likelihood_line)
        self.assertEqual(abs(g_line) / 10., 24.318828792633944)
        self.assertEqual(abs(g_verify) / 10., 24.318828792632853)
        self.assertEqual(abs(243.188287926329) / 10., 24.3188287926329)
        self.assertNotEqual(abs(g_line) / 10., abs(g_verify) / 10.)
        self.assertNotEqual(abs(g_line) / 10., abs(243.188287926329) / 10.)
        self.assertNotEqual(abs(g_verify) / 10., abs(243.188287926329) / 10.)
        self.assertEqual(format(abs(g_line) / 10., '.3f'), '24.319')
        self.assertNotEqual(abs(g_line) / 10., 24.319)
        self.assertEqual(abs(g_saved) / 10., 0.00013553813665528116)
        self.assertEqual(format(abs(g_saved) / 10., '.3f'), '0.000')
        self.assertNotEqual(abs(g_saved) / 10., 0.000)
        self.assertEqual(abs(0.0022566027606149425) / 10., 0.00022566027606149425)
        self.assertGreater(abs(g_saved), abs(0.0022566027606149425) / 10.)
        self.assertLess(abs(g_saved), abs(g_line) / 10.)
        self.assertEqual(1e-4 * max(abs(g_saved), 1.), 1e-4)
        self.assertNotEqual(1e-4 * max(abs(-14.560811461036163), 1.), 1e-4)
        self.assertNotEqual(1e-4 * max(abs(-140.6857484239093), 1.), 1e-4)
        self.assertGreater(abs(-140.6857484239093), 1.)
        self.assertGreater(abs(-140.6857484239093), abs(-2.598336989703646) / 10.)
        self.assertGreater(0.001075686027008818, 0.)
        self.assertLess(abs(0.001075686027008818), 1.)
        self.assertEqual(
            0.001075686027008818 - 0.0010756860341710889, -7.16227077646181e-12)
        self.assertEqual(coordinate_line_action(True, False, True, False), 'stop')
        self.assertEqual(coordinate_line_action(True, False, False, False), 'continue')
        self.assertEqual(
            pop0_revisit15_status(
                objective0, objective1, abs(g_line), abs(g_saved),
                likelihood_line, likelihood_saved),
            'CONDITIONAL_POP0_REVISIT15_REDUCED')
        self.assertEqual(
            pop0_revisit15_status(
                objective0, objective1, abs(g_verify), abs(g_saved),
                likelihood_verify, likelihood_saved),
            'CONDITIONAL_POP0_REVISIT15_REDUCED')
        self.assertEqual(int(1071.4857079059584), 1071)
        self.assertEqual(format(13.779590606689453, '.2f'), '13.78')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')

    def test_pop0_revisit15_sign_change_queues_one_secant(self):
        accepted = -0.03583179970308488
        rejected = -0.1358317997030849
        g_pos = 243.18828792633946
        g_neg = -325456.62454621913
        g_gradient = 243.188287926329
        objective0 = 8208466.685685918
        objective1 = 8224727.323640072
        count = -142299.81701650965
        count_g = -142299.81701650968
        fp0 = 5980.241106616332
        fp1 = -10280.388264356945
        prior0 = 4.1812947647508345
        prior1 = 4.189877944721143
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = (objective1 - objective0) - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (rejected ** 2 - accepted ** 2)
        theta = tracer0_secant(accepted, g_pos, rejected, g_neg)
        pop2 = 0.0010756860341710889
        self.assertEqual(rejected, accepted - 0.1)
        self.assertEqual(0.3 + accepted, 0.2641682002969151)
        self.assertEqual(0.3 + rejected, 0.1641682002969151)
        self.assertEqual(0.3 + accepted - 0.1, 0.16416820029691512)
        self.assertNotEqual(0.3 + rejected, 0.3 + accepted - 0.1)
        self.assertGreater(g_pos, 0.)
        self.assertLess(g_neg, 0.)
        self.assertEqual(g_pos * g_neg, -79147239.3176805)
        self.assertEqual(pop0_revisit15_step(g_pos), -0.1)
        self.assertEqual(pop0_revisit15_step(g_neg), 0.1)
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        self.assertEqual(coordinate_line_action(False, False, False, False), 'midpoint')
        self.assertEqual(objective1 - objective0, 16260.637954154052)
        self.assertGreater(objective1, objective0)
        self.assertEqual(delta_fp, -16260.629370973278)
        self.assertEqual(delta_prior, 0.008583179970308663)
        self.assertEqual(residual, 8.039933163672686e-10)
        self.assertGreater(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, 0.86328125)
        self.assertEqual(prior_residual, 1.717376241217039e-16)
        self.assertGreater(prior_residual, 0.)
        self.assertEqual(count - count_g, 2.9103830456733704e-11)
        self.assertNotEqual(count, count_g)
        self.assertNotEqual(fp0, 5980.241106616331)
        self.assertEqual(abs(g_pos) / 10., 24.318828792633944)
        self.assertEqual(abs(g_gradient) / 10., 24.3188287926329)
        self.assertNotEqual(abs(g_pos) / 10., abs(g_gradient) / 10.)
        self.assertEqual(format(abs(g_pos) / 10., '.3f'), '24.319')
        self.assertNotEqual(abs(g_pos) / 10., 24.319)
        self.assertEqual(abs(0.0022566027606149425) / 10., 0.00022566027606149425)
        self.assertEqual(format(abs(0.0022566027606149425) / 10., '.3f'), '0.000')
        self.assertNotEqual(abs(0.0022566027606149425) / 10., 0.000)
        self.assertGreater(abs(g_neg), abs(g_pos) / 10.)
        self.assertLess(abs(pop2), 1.)
        self.assertGreater(pop2, 0.)
        self.assertEqual(pop2 - 0.0010756860287141207, 5.4569682106375694e-12)
        self.assertEqual(pop2 - 0.0010756860295099285, 4.661160346586257e-12)
        self.assertEqual(1e-4 * max(abs(pop2), 1.), 1e-4)
        self.assertGreater(abs(-188421.8761422185), 1.)
        self.assertGreater(abs(-14.562432763914497), 1.)
        self.assertEqual(theta, -0.03590646609168676)
        self.assertEqual(0.3 + theta, 0.26409353390831325)
        self.assertLess(rejected, theta)
        self.assertLess(theta, accepted)
        self.assertEqual(accepted - theta, 7.466638860188085e-05)
        self.assertEqual(theta - rejected, 0.09992533361139813)
        self.assertLess(accepted - theta, theta - rejected)
        with self.assertRaises(ValueError):
            tracer0_secant(rejected, g_neg, accepted, g_pos)
        self.assertEqual(
            pop0_revisit15_status(objective0, objective1, abs(g_pos), abs(g_neg), 0., 0.),
            'CONDITIONAL_POP0_REVISIT15_NO_IMPROVEMENT')
        self.assertEqual(int(1061.5339182310272), 1061)
        self.assertEqual(format(13.78182601928711, '.2f'), '13.78')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')

    def test_full_gradient70_names_one_population0_line(self):
        white = -0.03583179970308488
        step_white = -0.1358317997030849
        gradient = 243.188287926329
        own_gate = abs(0.0022566027606149425) / 10.
        line_gate = abs(gradient) / 10.
        pop2 = 0.0010756860287141207
        pop2_saved = 0.0010756860295099285
        pop2_own = abs(-2.598336989703646) / 10.
        pop2_later = abs(-2.5983369897036246) / 10.
        objective = 8208466.685685918
        count_g = -142299.81701650968
        count_s = -142299.8170165097
        fp_g = 5980.241106616332
        fp_s = 5980.241106616331
        delta_count = count_g - count_s
        delta_fp = fp_g - fp_s
        residual = 0.0 - (-delta_count - delta_fp)
        diffs = {
            0: 2.9416469260468148e-12,
            2: 1.5631940186722204e-13,
            3: 3.552713678800501e-13,
            4: 3.979039320256561e-13,
            5: 2.220446049250313e-15,
            6: -2.2737367544323206e-13,
            9: 2.2737367544323206e-12,
            11: -7.958078640513122e-13,
            12: 1.8189894035458565e-12,
            13: 2.8421709430404007e-13,
            14: -4.547473508864641e-13,
            17: -1.1368683772161603e-12,
            22: -3.410605131648481e-13,
        }
        self.assertEqual(white - 0.1, step_white)
        self.assertEqual(0.3 + white, 0.2641682002969151)
        self.assertEqual(0.3 + white - 0.1, 0.16416820029691512)
        self.assertEqual(0.3 + (white - 0.1), 0.1641682002969151)
        self.assertEqual(0.3 + step_white, 0.1641682002969151)
        self.assertNotEqual(0.3 + white - 0.1, 0.3 + step_white)
        self.assertEqual(16777216 + 9, 16777225)
        self.assertEqual(2.7 + 0.015147031367434796, 2.715147031367435)
        self.assertGreater(gradient, 0.)
        self.assertEqual(pop0_revisit15_step(gradient), -0.1)
        self.assertEqual(pop0_revisit15_step(-gradient), 0.1)
        with self.assertRaises(ValueError):
            pop0_revisit15_step(0.)
        self.assertEqual(own_gate, 0.00022566027606149425)
        self.assertEqual(format(own_gate, '.3f'), '0.000')
        self.assertNotEqual(own_gate, 0.000)
        self.assertGreater(abs(gradient), own_gate)
        self.assertEqual(line_gate, 24.3188287926329)
        self.assertEqual(format(line_gate, '.3f'), '24.319')
        self.assertNotEqual(line_gate, 24.319)
        self.assertGreater(pop2, 0.)
        self.assertGreater(pop2_saved, 0.)
        self.assertLess(abs(pop2), 1.)
        self.assertLess(abs(pop2_saved), 1.)
        self.assertEqual(pop2 - pop2_saved, -7.958078640513122e-13)
        self.assertNotEqual(pop2, pop2_saved)
        self.assertEqual(1e-4 * max(abs(pop2), 1.), 1e-4)
        self.assertEqual(pop2_own, 0.25983369897036457)
        self.assertEqual(format(pop2_own, '.3f'), '0.260')
        self.assertNotEqual(pop2_own, 0.260)
        self.assertNotEqual(pop2_later, pop2_own)
        self.assertLess(abs(pop2), pop2_own)
        self.assertGreater(abs(-14.562432763915378), 1.)
        self.assertNotEqual(1e-4 * max(abs(-14.562432763915378), 1.), 1e-4)
        self.assertGreater(abs(-70.06021094703644), abs(2.070679793428482) / 10.)
        self.assertGreater(abs(-70.06021094703644), abs(-165.65390557877075) / 10.)
        self.assertGreater(abs(225.7958163209987), 0.3690110978312417)
        self.assertLessEqual(abs(-6.948312067057209), 542.8941177183742 / 10.)
        self.assertGreater(abs(2.4645925264204296), 1.)
        self.assertLessEqual(abs(2.4645925264204296), abs(-233.42232564563201) / 10.)
        self.assertGreater(abs(-110.36677846097515), abs(-110.35181616702818) / 10.)
        self.assertEqual(float(100 * np.exp(0.5 * 0.4259753562964724)), 123.73694044672878)
        self.assertEqual(delta_count, 2.9103830456733704e-11)
        self.assertNotEqual(count_g, count_s)
        self.assertEqual(delta_fp, 9.094947017729282e-13)
        self.assertNotEqual(fp_g, fp_s)
        self.assertEqual(residual, 3.001332515850663e-11)
        self.assertGreater(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 0)
        self.assertEqual(diffs[0], 2.9416469260468148e-12)
        self.assertEqual(coordinate_line_action(True, False, False, False), 'continue')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        self.assertEqual(coordinate_line_action(False, False, False, False), 'midpoint')
        self.assertEqual(coordinate_line_action(False, False, False, True), 'stop')
        self.assertEqual(coordinate_line_action(True, False, True, False), 'stop')
        self.assertEqual(
            pop0_revisit15_status(2., 1., 10., 1., 0., 0.),
            'CONDITIONAL_POP0_REVISIT15_REDUCED')
        self.assertEqual(
            pop0_revisit15_status(2., 1., 10., 2., 0., 0.),
            'CONDITIONAL_POP0_REVISIT15_IMPROVED')
        self.assertEqual(
            pop0_revisit15_status(2., 1., 10., 1., 1., 0.),
            'CONDITIONAL_POP0_REVISIT15_IMPROVED')
        self.assertEqual(
            pop0_revisit15_status(1., 1., abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_POP0_REVISIT15_NO_IMPROVEMENT')
        self.assertEqual(
            full_gradient_record_status(True, True, True, True),
            'CONDITIONAL_FULL_GRADIENT_RECORDED')
        self.assertEqual(int(594.6435992759652), 594)
        self.assertEqual(format(10.904472351074219, '.2f'), '10.90')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')
        self.assertEqual(objective, 8208466.685685918)

    def test_pop2_revisit_secant_reduced_queues_one_full_gradient(self):
        verified = 0.014827038075365416
        saved = 0.015147031367434796
        g_verify = -361.4068442163104
        g_saved = 0.0010756860295099285
        objective0 = 8208466.743509629
        objective1 = 8208466.685685918
        ic = 8072142.0269214865
        tracer_prior = 0.9015597734294367
        prior0 = 4.181289969000256
        prior1 = 4.1812947647508345
        count = -142299.8170165097
        line_count = -142299.81701650968
        fp0 = 5980.183278110035
        fp1 = 5980.241106616331
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (saved ** 2 - verified ** 2)
        identity0 = ic + tracer_prior + prior0 - count - fp0 - objective0
        identity1 = ic + tracer_prior + prior1 - count - fp1 - objective1
        likelihood0 = count + fp0
        likelihood1 = count + fp1
        self.assertEqual(saved, 0.015147031367434796)
        self.assertEqual(2.7 + saved, 2.715147031367435)
        self.assertEqual(16777216 + 11, 16777227)
        self.assertLess(verified, saved)
        self.assertLess(g_verify, 0.)
        self.assertGreater(g_saved, 0.)
        self.assertEqual(g_verify * g_saved, -0.3887602932927562)
        self.assertEqual(delta_objective, -0.057823711074888706)
        self.assertLess(delta_objective, 0.)
        self.assertEqual(delta_fp, 0.05782850629657332)
        self.assertEqual(delta_prior, 4.79575057887871e-06)
        self.assertEqual(residual, -5.2889426171987e-10)
        self.assertLess(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(prior_residual, 1.9786689647860456e-17)
        self.assertGreater(prior_residual, 0.)
        self.assertEqual(identity0, 0.0)
        self.assertEqual(identity1, 0.0)
        self.assertEqual(count - line_count, -2.9103830456733704e-11)
        self.assertNotEqual(count, line_count)
        self.assertGreater(likelihood1, likelihood0)
        self.assertEqual(abs(g_verify) / 10., 36.140684421631036)
        self.assertEqual(abs(-361.40684421630937) / 10., 36.14068442163094)
        self.assertEqual(abs(-361.4068442163105) / 10., 36.14068442163105)
        self.assertNotEqual(abs(g_verify) / 10., abs(-361.40684421630937) / 10.)
        self.assertNotEqual(abs(g_verify) / 10., abs(-361.4068442163105) / 10.)
        self.assertEqual(format(abs(g_verify) / 10., '.3f'), '36.141')
        self.assertNotEqual(abs(g_verify) / 10., 36.141)
        self.assertEqual(abs(g_saved) / 10., 0.00010756860295099286)
        self.assertEqual(format(abs(g_saved) / 10., '.3f'), '0.000')
        self.assertNotEqual(abs(g_saved) / 10., 0.000)
        self.assertLess(abs(g_saved), 1.)
        self.assertLess(abs(g_saved), abs(g_verify) / 10.)
        self.assertEqual(1e-4 * max(abs(g_saved), 1.), 1e-4)
        self.assertGreater(abs(-14.562432763914241), 1.)
        self.assertEqual(coordinate_line_action(True, True, True, False), 'stop')
        self.assertEqual(coordinate_line_action(True, False, True, False), 'stop')
        self.assertEqual(
            pop2_revisit_status(
                objective0, objective1, abs(g_verify), abs(g_saved), likelihood0, likelihood1),
            'CONDITIONAL_POP2_REVISIT_REDUCED')
        self.assertEqual(int(1050.2945015360601), 1050)
        self.assertEqual(format(13.77047348022461, '.2f'), '13.77')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')

    def test_tracer6_support_step7_improved_queues_one_full_gradient(self):
        verified = 0.3759753562964724
        saved = 0.4259753562964724
        g0 = -138.72354222799916
        g1 = -110.35181616702818
        objective0 = 8208473.062978312
        objective1 = 8208466.823045053
        count_delta = 6.160812665882986
        fp_delta = 0.0991693617143028
        prior_delta = 0.020048767814823565
        residual = (objective1 - objective0) - (-count_delta - fp_delta + prior_delta)
        prior_residual = prior_delta - 0.5 * (saved ** 2 - verified ** 2)
        pop8 = -0.55707754548845
        pop5 = -165.65390557877245
        diffs = {
            5: 0.0,
            14: 0.0,
            22: 4.808597964256478e-12,
        }
        self.assertEqual(saved, verified + 0.05)
        self.assertEqual(0.3259753562964724 + 0.05 + 0.05, saved)
        self.assertEqual(16777216 + 6, 16777222)
        self.assertLess(g0, 0.)
        self.assertLess(g1, 0.)
        self.assertGreater(g1 * g0, 0.)
        self.assertEqual(abs(g0) / 10., 13.872354222799917)
        self.assertEqual(abs(-138.7235422279993) / 10., 13.872354222799931)
        self.assertNotEqual(abs(g0) / 10., abs(-138.7235422279993) / 10.)
        self.assertEqual(format(abs(g0) / 10., '.3f'), '13.872')
        self.assertEqual(format(abs(-138.7235422279993) / 10., '.3f'), '13.872')
        self.assertEqual(abs(g1) / 10., 11.035181616702818)
        self.assertEqual(format(abs(g1) / 10., '.3f'), '11.035')
        self.assertGreater(abs(g1), abs(g0) / 10.)
        self.assertEqual(objective1 - objective0, -6.239933259785175)
        self.assertEqual(residual, -2.709832358505082e-12)
        self.assertLess(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, -0.0029096603393554688)
        self.assertEqual(prior_residual, -4.163336342344337e-17)
        self.assertLess(prior_residual, 0.)
        self.assertGreater(abs(pop5), abs(g1))
        self.assertLess(abs(pop8), 1.)
        self.assertEqual(1e-4 * max(abs(pop8), 1.), 1e-4)
        self.assertGreater(abs(6.440485892088479), 1.)
        self.assertGreater(abs(66.31587008741167), 0.3690110978312417)
        self.assertLessEqual(abs(-11.360197536685469), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(1.7168722314607752), abs(-233.42232564563201) / 10.)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 22)
        self.assertEqual(float(100 * np.exp(0.5 * verified)), 120.68186450175409)
        self.assertEqual(float(100 * np.exp(0.5 * saved)), 123.73694044672878)
        self.assertEqual(0.5 * -0.010011571272851271, -0.005005785636425636)
        self.assertEqual(int(1402.1596948930528), 1402)
        self.assertEqual(format(14.946136474609375, '.2f'), '14.95')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')
        self.assertEqual(
            tracer6_support_step7_status(objective0, objective1, abs(g0), abs(g1), 0., 1.),
            'CONDITIONAL_TRACER6_SUPPORT_STEP7_IMPROVED')
        self.assertEqual(
            full_gradient_record_status(True, True, True, True),
            'CONDITIONAL_FULL_GRADIENT_RECORDED')

    def test_tracer6_revisit8_support_change_queues_one_recompiled_step(self):
        verified = 0.3259753562964724
        improved = 0.3759753562964724
        g0 = -164.04053987282455
        g1 = -138.7235422279993
        objective0 = 8208480.64461356
        objective1 = 8208473.062978312
        count_delta = 7.50021337158978
        fp_delta = 0.09897064404503908
        prior_delta = 0.017548767814823618
        residual = (objective1 - objective0) - (-count_delta - fp_delta + prior_delta)
        prior_residual = prior_delta - 0.5 * (improved ** 2 - verified ** 2)
        pop8 = -0.20522195267576707
        pop5 = -164.08178749991677
        self.assertEqual(improved, verified + 0.05)
        self.assertEqual(improved + 0.05, 0.4259753562964724)
        self.assertLess(g0, 0.)
        self.assertLess(g1, 0.)
        self.assertGreater(g1 * g0, 0.)
        self.assertEqual(abs(g0) / 10., 16.404053987282456)
        self.assertEqual(abs(-164.04053987283538) / 10., 16.40405398728354)
        self.assertNotEqual(abs(g0) / 10., abs(-164.04053987283538) / 10.)
        self.assertEqual(format(abs(g0) / 10., '.3f'), '16.404')
        self.assertEqual(format(abs(-164.04053987283538) / 10., '.3f'), '16.404')
        self.assertEqual(abs(g1) / 10., 13.872354222799931)
        self.assertEqual(format(abs(g1) / 10., '.3f'), '13.872')
        self.assertGreater(abs(g1), abs(g0) / 10.)
        self.assertEqual(tracer6_support_step_size(g1), 0.05)
        self.assertEqual(coordinate_line_action(True, False, False, False), 'continue')
        self.assertEqual(objective1 - objective0, -7.581635247915983)
        self.assertEqual(residual, -9.598810635225163e-11)
        self.assertLess(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, -0.10306644439697266)
        self.assertEqual(prior_residual, 0.0)
        self.assertGreater(abs(pop5), abs(g1))
        self.assertLess(abs(pop8), 1.)
        self.assertEqual(1e-4 * max(abs(pop8), 1.), 1e-4)
        self.assertGreater(abs(3.2273272702225926), 1.)
        self.assertGreater(abs(3.2273272702225926), 0.00022566027606149425)
        self.assertGreater(abs(68.57256312118398), 0.3690110978312417)
        self.assertLessEqual(abs(-13.438845828578117), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(3.1992017453670263), abs(-233.42232564563201) / 10.)
        self.assertEqual(float(100 * np.exp(0.5 * verified)), 117.70221865062094)
        self.assertEqual(float(100 * np.exp(0.5 * improved)), 120.68186450175409)
        self.assertEqual(max(
            {0: -6.177458544698311e-11, 22: 8.646416915780719e-12},
            key=lambda index: abs({0: -6.177458544698311e-11, 22: 8.646416915780719e-12}[index])), 0)
        self.assertEqual(int(1191.5061188359978), 1191)
        self.assertEqual(format(14.097980499267578, '.2f'), '14.10')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')
        self.assertEqual(
            tracer6_revisit8_status(objective0, objective1, abs(g0), abs(g1), 0., 1.),
            'CONDITIONAL_TRACER6_REVISIT8_IMPROVED')
        self.assertEqual(
            tracer6_support_step7_status(1., 1., 10., 10., 0., 0.),
            'CONDITIONAL_TRACER6_SUPPORT_STEP7_NO_IMPROVEMENT')

    def test_full_gradient67_names_tracer6_line(self):
        gradient = -164.04053987283538
        own_gate = abs(-164.02895850096368) / 10.
        line_gate = abs(gradient) / 10.
        pop0 = 0.002256602760160195
        pop0_saved = 0.0022566027606149425
        pop8 = 0.1606061944962431
        values = {
            0: -62.062407505713885,
            2: -25.314642258495613,
            3: -33.139662793973436,
            4: -43.21332584155364,
            5: -2.1499250303300657,
            6: gradient,
            9: pop0,
            12: 70.85747294138264,
            13: -15.492147921434373,
            14: -162.4927762699186,
            17: pop8,
            22: 4.6766209476625615,
        }
        saved = {
            0: -62.06240750571477,
            6: -164.0405398728352,
            9: pop0_saved,
            12: 70.85747294138196,
            17: 0.16060619449763577,
            22: 4.676620947666595,
        }
        diffs = {index: values[index] - saved[index] for index in saved}
        below = sorted(index for index, value in (
            (9, pop0), (17, pop8), (5, values[5]), (6, gradient)) if abs(value) < 1.)
        objective = 8208480.64461356
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        population_prior = 4.181215025367132
        count = -142313.47804254713
        fp = 5979.905527736704
        secant_fp = 5979.905527736705
        self.assertEqual(16777216 + 6, 16777222)
        self.assertEqual(own_gate, 16.402895850096368)
        self.assertEqual(line_gate, 16.40405398728354)
        self.assertNotEqual(line_gate, own_gate)
        self.assertEqual(format(own_gate, '.3f'), '16.403')
        self.assertEqual(format(line_gate, '.3f'), '16.404')
        self.assertGreater(abs(gradient), own_gate)
        self.assertGreater(abs(gradient), 18.655009368073785)
        self.assertGreater(abs(gradient), 1.)
        self.assertEqual(tracer6_support_step_size(gradient), 0.05)
        self.assertEqual(0.3259753562964724 + 0.05, 0.3759753562964724)
        with self.assertRaises(ValueError):
            tracer6_support_step_size(0.)
        self.assertEqual(below, [9, 17])
        self.assertEqual(min((9, 17), key=lambda index: abs((pop0, pop8)[index == 17])), 9)
        self.assertEqual(1e-4 * max(abs(pop0), 1.), 1e-4)
        self.assertEqual(1e-4 * max(abs(pop8), 1.), 1e-4)
        self.assertEqual(abs(pop0) / 10., 0.0002256602760160195)
        self.assertEqual(abs(pop0_saved) / 10., 0.00022566027606149425)
        self.assertNotEqual(abs(pop0) / 10., abs(pop0_saved) / 10.)
        self.assertGreater(abs(pop0), abs(pop0_saved) / 10.)
        self.assertLess(abs(pop0), 1.)
        self.assertLess(abs(pop8), 1.)
        self.assertGreater(abs(values[12]), 0.3690110978312417)
        self.assertGreater(abs(values[14]), 0.2070679793428482)
        self.assertLessEqual(abs(values[13]), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(values[22]), abs(-233.42232564563201) / 10.)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 22)
        self.assertEqual(diffs[22], -4.033218203858269e-12)
        self.assertEqual(fp - secant_fp, -9.094947017729282e-13)
        self.assertNotEqual(fp, secant_fp)
        self.assertEqual(ic + tracer_prior + population_prior - count - fp - objective, 0.0)
        self.assertEqual(format(1.8209067972071895, '.6f'), '1.820907')
        self.assertEqual(format(12.683474196080232, '.6f'), '12.683474')
        self.assertEqual(format(73.16082313977495, '.6f'), '73.160823')
        self.assertEqual(format(abs(gradient), '.6f'), '164.040540')
        self.assertEqual(format(90.52578672081717, '.6f'), '90.525787')
        self.assertEqual(format(162.4927762699186, '.6f'), '162.492776')
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(0.3 + -0.03583179970308488, 0.2641682002969151)
        self.assertEqual(0.5 * -0.010011571272851271, -0.005005785636425636)
        self.assertEqual(int(596.6815483910032), 596)
        self.assertEqual(format(10.702079772949219, '.2f'), '10.70')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')
        self.assertEqual(
            tracer6_revisit8_status(1., 1., 10., 10., 0., 0.),
            'CONDITIONAL_TRACER6_REVISIT8_NO_IMPROVEMENT')
        self.assertEqual(
            tracer6_revisit8_status(2., 1., 10., 9., 0., 0.),
            'CONDITIONAL_TRACER6_REVISIT8_IMPROVED')
        self.assertEqual(
            tracer6_revisit8_status(2., 1., 10., 1., 0., 0.),
            'CONDITIONAL_TRACER6_REVISIT8_REDUCED')

    def test_pop0_revisit14_secant_reduced_queues_one_full_gradient(self):
        verified = -0.035717259497908965
        saved = -0.03583179970308488
        g_verify = 373.0820398941962
        g_saved = 0.0022566027606149425
        pop8 = 0.16060619449763577
        objective0 = 8208480.665980136
        objective1 = 8208480.64461356
        count = -142313.47804254713
        fp0 = 5979.88415706311
        fp1 = 5979.905527736705
        prior0 = 4.1812109277451714
        prior1 = 4.181215025367132
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (saved ** 2 - verified ** 2)
        identity0 = ic + tracer_prior + prior0 - count - fp0 - objective0
        identity1 = ic + tracer_prior + prior1 - count - fp1 - objective1
        saved_values = {
            9: g_saved,
            17: pop8,
            5: -2.1499250303300688,
            12: 70.85747294138196,
        }
        below = sorted(index for index, value in saved_values.items() if abs(value) < 1.)
        diffs = {
            0: -2.1032064978498966e-12,
            22: -3.1885605267234496e-12,
            9: -4.547473508864641e-13,
        }
        self.assertEqual(0.3 + saved, 0.2641682002969151)
        self.assertEqual(0.3 + verified, 0.264282740502091)
        self.assertLess(saved, verified)
        self.assertGreater(g_saved, 0.)
        self.assertGreater(g_verify, 0.)
        self.assertLess(abs(g_saved), 1.)
        self.assertLess(abs(pop8), 1.)
        self.assertGreater(abs(saved_values[5]), 1.)
        self.assertEqual(below, [9, 17])
        self.assertEqual(min(saved_values, key=lambda index: abs(saved_values[index])), 9)
        self.assertEqual(1e-4 * max(abs(g_saved), 1.), 1e-4)
        self.assertEqual(1e-4 * max(abs(pop8), 1.), 1e-4)
        self.assertEqual(abs(g_verify) / 10., 37.308203989419624)
        self.assertEqual(abs(373.08203989419667) / 10., 37.30820398941967)
        self.assertEqual(abs(373.08203989420394) / 10., 37.30820398942039)
        self.assertNotEqual(abs(g_verify) / 10., 37.30820398941967)
        self.assertNotEqual(abs(g_verify) / 10., 37.30820398942039)
        self.assertNotEqual(abs(g_verify) / 10., 37.308)
        self.assertEqual(format(abs(g_verify) / 10., '.3f'), '37.308')
        self.assertEqual(abs(g_saved) / 10., 0.00022566027606149425)
        self.assertNotEqual(abs(g_saved) / 10., 0.00012529367029458783)
        self.assertNotEqual(abs(g_saved) / 10., 0.0002332932001004387)
        self.assertNotEqual(abs(g_saved) / 10., 0.000)
        self.assertEqual(format(abs(g_saved) / 10., '.3f'), '0.000')
        self.assertLessEqual(abs(g_saved), abs(g_verify) / 10.)
        self.assertGreater(abs(g_saved), abs(g_saved) / 10.)
        self.assertEqual(delta_objective, -0.021366575732827187)
        self.assertLess(delta_objective, 0.)
        self.assertEqual(delta_fp, 0.021370673594901746)
        self.assertEqual(delta_prior, 4.097621960852393e-06)
        self.assertEqual(prior_residual, 3.3968054063970854e-16)
        self.assertGreater(prior_residual, 0.)
        self.assertEqual(residual, 2.4011370669541066e-10)
        self.assertGreater(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, 0.25782012939453125)
        self.assertEqual(identity0, 0.0)
        self.assertEqual(identity1, 0.0)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 22)
        self.assertGreater(abs(saved_values[12]), 0.3690110978312417)
        self.assertGreater(abs(-162.4927762699182), 0.2070679793428482)
        self.assertLessEqual(abs(-15.492147921433748), 54.28941177183742)
        self.assertLessEqual(abs(4.676620947666595), abs(-233.42232564563201) / 10.)
        self.assertEqual(16777216 + 9, 16777225)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(int(1045.9836078759981), 1045)
        self.assertEqual(format(13.581108093261719, '.2f'), '13.58')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')
        self.assertEqual(coordinate_line_action(True, False, True, False), 'stop')
        self.assertEqual(
            full_gradient_record_status(True, True, True, True),
            'CONDITIONAL_FULL_GRADIENT_RECORDED')

    def test_pop0_revisit14_sign_change_queues_one_secant(self):
        positive = -0.035717259497908965
        negative = -0.13571725949790897
        g_pos = 373.08203989419667
        g_neg = -325348.38783281425
        pop8 = 0.15848909803117237
        objective0 = 8208480.665980136
        objective1 = 8224729.395645589
        count0 = -142313.47804254713
        count_gradient = -142313.4780425471
        fp0 = 5979.88415706311
        fp1 = -10268.836936663192
        prior0 = 4.1812109277451714
        prior1 = 4.189782653694962
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        theta = tracer0_secant(positive, g_pos, negative, g_neg)
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (negative ** 2 - positive ** 2)
        identity0 = ic + tracer_prior + prior0 - count0 - fp0 - objective0
        identity1 = ic + tracer_prior + prior1 - count0 - fp1 - objective1
        diffs = {
            0: 5.519495971384458e-11,
            2: 2.4577673229941865e-11,
            3: 2.4201085579989012e-11,
            4: 4.149569576838985e-12,
            5: 1.7763568394002505e-15,
            6: -1.0913936421275139e-11,
            9: -7.275957614183426e-12,
            12: 1.2505552149377763e-12,
            13: -1.4210854715202004e-12,
            14: 1.3073986337985843e-12,
            17: 5.400124791776761e-13,
            22: 1.55964130499342e-12,
        }
        self.assertEqual(positive - 0.1, negative)
        self.assertEqual(pop0_revisit10_step(g_pos), -0.1)
        self.assertEqual(theta, -0.03583179970308488)
        self.assertEqual(0.3 + positive, 0.264282740502091)
        self.assertEqual(0.3 + negative, 0.16428274050209102)
        self.assertEqual(0.3 + positive - 0.1, 0.164282740502091)
        self.assertNotEqual(0.3 + negative, 0.3 + positive - 0.1)
        self.assertEqual(0.3 + theta, 0.2641682002969151)
        self.assertLess(negative, theta)
        self.assertLess(theta, positive)
        self.assertLess(positive - theta, theta - negative)
        self.assertEqual(positive - theta, 0.00011454020517591423)
        self.assertEqual(theta - negative, 0.09988545979482409)
        with self.assertRaises(ValueError):
            tracer0_secant(negative, g_neg, positive, g_pos)
        self.assertEqual(delta_objective, 16248.729665452614)
        self.assertGreater(delta_objective, 0.)
        self.assertEqual(delta_fp, -16248.721093726303)
        self.assertEqual(delta_prior, 0.008571725949790832)
        self.assertEqual(prior_residual, -6.591949208711867e-17)
        self.assertLess(prior_residual, 0.)
        self.assertEqual(residual, 3.6197889130562544e-10)
        self.assertGreater(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, 0.388671875)
        self.assertEqual(identity0, 0.0)
        self.assertEqual(identity1, -9.313225746154785e-10)
        self.assertLess(identity1, 0.)
        self.assertGreater(g_pos, 0.)
        self.assertLess(g_neg, 0.)
        self.assertGreater(abs(g_neg), abs(g_pos) / 10.)
        self.assertEqual(abs(g_pos) / 10., 37.30820398941967)
        self.assertEqual(abs(373.08203989420394) / 10., 37.30820398942039)
        self.assertNotEqual(abs(g_pos) / 10., abs(373.08203989420394) / 10.)
        self.assertEqual(format(abs(g_pos) / 10., '.3f'), '37.308')
        self.assertEqual(format(abs(373.08203989420394) / 10., '.3f'), '37.308')
        self.assertNotEqual(abs(g_pos) / 10., 37.308)
        self.assertLess(abs(pop8), 1.)
        self.assertGreater(abs(1.6050525909992683), 1.)
        self.assertEqual(1e-4 * max(abs(pop8), 1.), 1e-4)
        self.assertEqual(count0 - count_gradient, -2.9103830456733704e-11)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 0)
        self.assertGreater(abs(-3.6901109783123034), 0.3690110978312417)
        self.assertGreater(abs(-205.62275200870184), 0.2070679793428482)
        self.assertLessEqual(abs(-19.253646718848195), 54.28941177183742)
        self.assertLessEqual(abs(5.137547373828277), abs(-233.42232564563201) / 10.)
        self.assertEqual(16777216 + 9, 16777225)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(int(1058.8851914659608), 1058)
        self.assertEqual(format(13.703689575195312, '.2f'), '13.70')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        self.assertEqual(coordinate_line_action(False, False, False, False), 'midpoint')
        self.assertEqual(
            pop0_revisit14_status(objective0, objective0, abs(g_pos), abs(g_pos), 1., 1.),
            'CONDITIONAL_POP0_REVISIT14_NO_IMPROVEMENT')

    def test_full_gradient66_names_population0_and_steps_by_minus01(self):
        gradient = 373.08203989420394
        white = -0.035717259497908965
        pop8 = 0.15848909803063235
        objective = 8208480.665980136
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        population_prior = 4.1812109277451714
        count = -142313.4780425471
        secant_count = -142313.47804254713
        fp = 5979.88415706311
        identity = ic + tracer_prior + population_prior - count - fp - objective
        diffs = {
            0: -5.6772364587232005e-11,
            2: -2.475886162756069e-11,
            3: -2.4421353828074643e-11,
            4: -4.263256414560601e-12,
            6: 1.1226575225009583e-11,
            9: 1.0913936421275139e-11,
            12: -1.1368683772161603e-12,
            13: 1.4779288903810084e-12,
            14: -1.0800249583553523e-12,
            22: 4.5279335836312384e-12,
        }
        blocks = {
            (1.820881592477015, '1.820882'),
            (12.683349331929325, '12.683349'),
            (73.15804678934661, '73.158047'),
            (164.03314414086424, '164.033144'),
            (130.17597764626632, '130.175978'),
            (373.08203989420394, '373.082040'),
        }
        self.assertEqual(pop0_revisit10_step(gradient), -0.1)
        self.assertEqual(pop0_revisit10_step(-gradient), 0.1)
        self.assertEqual(white - 0.1, -0.13571725949790897)
        self.assertEqual(0.3 + white, 0.264282740502091)
        self.assertEqual(0.3 + (white - 0.1), 0.16428274050209102)
        self.assertEqual(0.3 + white - 0.1, 0.164282740502091)
        self.assertNotEqual(0.3 + (white - 0.1), 0.3 + white - 0.1)
        self.assertGreater(gradient, 0.)
        self.assertGreater(abs(gradient), 1.)
        self.assertGreater(abs(gradient), 0.00012529367029458783)
        self.assertEqual(abs(0.0012529367029458782) / 10., 0.00012529367029458783)
        self.assertNotEqual(0.00012529367029458783, 0.0002332932001004387)
        self.assertEqual(format(0.00012529367029458783, '.3f'), '0.000')
        self.assertEqual(format(0.0002332932001004387, '.3f'), '0.000')
        self.assertEqual(abs(gradient) / 10., 37.30820398942039)
        self.assertEqual(format(abs(gradient) / 10., '.3f'), '37.308')
        self.assertNotEqual(abs(gradient) / 10., 37.308)
        self.assertLess(abs(pop8), 1.)
        self.assertLess(abs(pop8), abs(-2.149923006156464))
        self.assertGreater(abs(-2.149923006156464), 1.)
        self.assertEqual(1e-4 * max(abs(pop8), 1.), 1e-4)
        self.assertNotEqual(1e-4 * max(abs(gradient), 1.), 1e-4)
        self.assertEqual(0.5 * -0.010011571272851271, -0.0050057856364256355)
        self.assertEqual(0.5 * -0.4114823435353389, -0.20574117176766946)
        self.assertEqual(0.5 * 0.0765438051588915, 0.03827190257944575)
        self.assertGreater(abs(-3.690110978313554), 0.3690110978312417)
        self.assertEqual(abs(-3.690110978313554) / 10., 0.3690110978313554)
        self.assertNotEqual(abs(-3.690110978313554) / 10., 0.3690110978312417)
        self.assertNotEqual(0.3690110978312417, 0.369)
        self.assertEqual(format(0.3690110978312417, '.3f'), '0.369')
        self.assertEqual(format(0.3690110978313554, '.3f'), '0.369')
        self.assertGreater(abs(-205.62275200870315), 0.2070679793428482)
        self.assertLessEqual(abs(-19.253646718846774), 54.28941177183742)
        self.assertEqual(542.8941177183742 / 10., 54.28941177183742)
        self.assertLessEqual(abs(5.137547373826718), abs(-233.42232564563201) / 10.)
        self.assertEqual(abs(-233.42232564563201) / 10., 23.3422325645632)
        self.assertEqual(16777216 + 9, 16777225)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(identity, -9.313225746154785e-10)
        self.assertLess(identity, 0.)
        self.assertEqual(count - secant_count, 2.9103830456733704e-11)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 0)
        self.assertEqual(int(588.8646958660102), 588)
        self.assertEqual(format(10.705066680908203, '.2f'), '10.71')
        self.assertEqual(format(104.85063171386719, '.2f'), '104.85')
        for value, display in blocks:
            self.assertEqual(format(value, '.6f'), display)
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        self.assertEqual(coordinate_line_action(False, False, False, False), 'midpoint')
        self.assertEqual(
            pop0_revisit14_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_POP0_REVISIT14_NO_IMPROVEMENT')
        self.assertEqual(
            full_gradient_record_status(True, True, True, True),
            'CONDITIONAL_FULL_GRADIENT_RECORDED')

    def test_pop3_revisit7_secant_reduced_queues_one_full_gradient(self):
        verified = -0.41090912289159987
        saved = -0.4114823435353389
        g_verify = 281.5682173355289
        g_saved = -3.690110978312417
        objective0 = 8208480.745624014
        objective1 = 8208480.665980136
        count = -142313.47804254713
        fp0 = 5979.8042774792775
        fp1 = 5979.88415706311
        prior0 = 4.180975221862276
        prior1 = 4.1812109277451714
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (saved ** 2 - verified ** 2)
        identity0 = ic + tracer_prior + prior0 - count - fp0 - objective0
        identity1 = ic + tracer_prior + prior1 - count - fp1 - objective1
        checked = (
            -62.06240750571126, -25.313375400686567, -33.13544453830395,
            -43.21160960813981, -164.03314414087546, 373.08203989419303, g_saved,
            -19.253646718848252, -205.62275200870207, 5.13754737382219)
        self.assertEqual(0.5 * saved, -0.20574117176766946)
        self.assertEqual(0.5 * verified, -0.20545456144579993)
        self.assertLess(saved, verified)
        self.assertLess(g_saved, 0.)
        self.assertGreater(g_verify, 0.)
        self.assertGreater(abs(g_saved), 1.)
        self.assertEqual(min(abs(value) for value in checked), abs(g_saved))
        self.assertEqual(abs(g_saved) / 10., 0.3690110978312417)
        self.assertEqual(format(abs(g_saved) / 10., '.3f'), '0.369')
        self.assertNotEqual(abs(g_saved) / 10., 0.369)
        self.assertNotEqual(abs(g_saved) / 10., 0.3805631199799269)
        self.assertGreater(abs(g_saved), 0.3805631199799269)
        self.assertLessEqual(abs(g_saved), abs(g_verify) / 10.)
        self.assertEqual(abs(g_verify) / 10., 28.156821733552892)
        self.assertEqual(abs(281.5682173355288) / 10., 28.15682173355288)
        self.assertEqual(abs(281.5682173355295) / 10., 28.15682173355295)
        self.assertNotEqual(abs(g_verify) / 10., abs(281.5682173355288) / 10.)
        self.assertNotEqual(abs(g_verify) / 10., abs(281.5682173355295) / 10.)
        self.assertEqual(format(abs(g_verify) / 10., '.3f'), '28.157')
        self.assertEqual(delta_objective, -0.07964387815445662)
        self.assertLess(delta_objective, 0.)
        self.assertEqual(delta_fp, 0.07987958383273508)
        self.assertEqual(delta_prior, 0.00023570588289523187)
        self.assertEqual(prior_residual, -1.3877787807814457e-16)
        self.assertLess(prior_residual, 0.)
        self.assertEqual(residual, -2.0461676797367545e-10)
        self.assertLess(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, -0.21970558166503906)
        self.assertEqual(identity0, -9.313225746154785e-10)
        self.assertLess(identity0, 0.)
        self.assertEqual(identity1, 0.0)
        self.assertEqual(1e-4 * max(abs(g_saved), 1.), 0.00036901109783124173)
        self.assertNotEqual(1e-4 * max(abs(g_saved), 1.), 1e-4)
        self.assertGreater(abs(373.08203989419303), 0.00012529367029458783)
        self.assertGreater(abs(-205.62275200870207), 0.2070679793428482)
        self.assertLessEqual(abs(-19.253646718848252), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(5.13754737382219), abs(-233.42232564563201) / 10.)
        self.assertEqual(16777216 + 12, 16777228)
        self.assertEqual(coordinate_line_action(True, True, True, False), 'stop')
        self.assertEqual(
            pop3_revisit7_status(objective0, objective1, abs(g_verify), abs(g_saved), 1., 1.),
            'CONDITIONAL_POP3_REVISIT7_REDUCED')
        self.assertEqual(
            full_gradient_record_status(True, True, True, True),
            'CONDITIONAL_FULL_GRADIENT_RECORDED')

    def test_pop3_revisit7_sign_change_queues_one_secant(self):
        positive = -0.41090912289159987
        negative = -0.5109091228915998
        g_pos = 281.5682173355288
        g_neg = -48838.82205662482
        objective0 = 8208480.745624014
        objective1 = 8210927.722253151
        count = -142313.47804254713
        fp0 = 5979.8042774792775
        fp1 = 3532.873739254298
        prior0 = 4.180975221862276
        prior1 = 4.227066134151436
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        pop0 = 0.0012529367029458782
        theta = tracer0_secant(positive, g_pos, negative, g_neg)
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (negative ** 2 - positive ** 2)
        identity0 = ic + tracer_prior + prior0 - count - fp0 - objective0
        identity1 = ic + tracer_prior + prior1 - count - fp1 - objective1
        self.assertEqual(positive - 0.1, negative)
        self.assertEqual(pop3_revisit5_step(g_pos), -0.1)
        self.assertEqual(theta, -0.4114823435353389)
        self.assertEqual(0.5 * positive, -0.20545456144579993)
        self.assertEqual(0.5 * negative, -0.2554545614457999)
        self.assertEqual(0.5 * theta, -0.20574117176766946)
        self.assertLess(negative, theta)
        self.assertLess(theta, positive)
        self.assertLess(positive - theta, theta - negative)
        with self.assertRaises(ValueError):
            tracer0_secant(negative, g_neg, positive, g_pos)
        self.assertEqual(delta_objective, 2446.9766291370615)
        self.assertGreater(delta_objective, 0.)
        self.assertEqual(delta_fp, -2446.9305382249795)
        self.assertEqual(delta_prior, 0.04609091228916018)
        self.assertEqual(prior_residual, 2.0816681711721685e-16)
        self.assertGreater(prior_residual, 0.)
        self.assertEqual(residual, -2.0691004465334117e-10)
        self.assertLess(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, -0.22216796875)
        self.assertEqual(identity0, -9.313225746154785e-10)
        self.assertEqual(identity1, -9.313225746154785e-10)
        self.assertLess(identity0, 0.)
        self.assertLess(identity1, 0.)
        self.assertGreater(g_pos, 0.)
        self.assertLess(g_neg, 0.)
        self.assertGreater(abs(g_neg), abs(g_pos) / 10.)
        self.assertEqual(abs(g_pos) / 10., 28.15682173355288)
        self.assertEqual(abs(281.5682173355295) / 10., 28.15682173355295)
        self.assertNotEqual(abs(g_pos) / 10., abs(281.5682173355295) / 10.)
        self.assertEqual(format(abs(g_pos) / 10., '.3f'), '28.157')
        self.assertEqual(format(abs(281.5682173355295) / 10., '.3f'), '28.157')
        self.assertLess(abs(pop0), 1.)
        self.assertGreater(abs(pop0), 0.00012529367029458783)
        self.assertEqual(1e-4 * max(abs(pop0), 1.), 1e-4)
        self.assertEqual(-62.06240750570811 - (-62.06240750571303), 4.924061158817494e-12)
        self.assertEqual(g_pos - 281.5682173355295, -6.821210263296962e-13)
        self.assertGreater(abs(-40.591049057191256), 0.2070679793428482)
        self.assertLessEqual(abs(-4.821083100508879), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(5.257512894146075), abs(-233.42232564563201) / 10.)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        self.assertEqual(coordinate_line_action(False, False, False, False), 'midpoint')
        self.assertEqual(
            pop3_revisit7_status(objective0, objective0, abs(g_pos), abs(g_pos), 1., 1.),
            'CONDITIONAL_POP3_REVISIT7_NO_IMPROVEMENT')

    def test_full_gradient65_names_population3_and_steps_by_minus01(self):
        gradient = 281.5682173355295
        white = -0.41090912289159987
        pop0 = 0.0012529367013542625
        objective = 8208480.745624014
        checked = (
            -62.06240750571303, -25.319409002910383, -33.14921218299066,
            -43.2087493874708, -164.05977003369506, pop0, gradient,
            -4.821083100508709, -40.59104905719103, 5.257512894145648)
        diffs = {
            0: 5.300648808770347e-12,
            2: 1.3002932064409833e-12,
            3: 1.8758328224066645e-12,
            4: 3.126388037344441e-13,
            6: -9.379164112033322e-13,
            9: -1.5916157281026244e-12,
            12: 5.684341886080801e-13,
            13: 5.684341886080802e-14,
            14: 3.410605131648481e-13,
            22: -4.945377440890297e-12,
        }
        self.assertEqual(pop3_revisit5_step(gradient), -0.1)
        self.assertEqual(pop3_revisit5_step(-gradient), 0.1)
        self.assertEqual(white - 0.1, -0.5109091228915998)
        self.assertEqual(0.5 * white, -0.20545456144579993)
        self.assertEqual(0.5 * (white - 0.1), -0.2554545614457999)
        self.assertGreater(gradient, 0.)
        self.assertGreater(abs(gradient), 1.)
        self.assertEqual(abs(gradient) / 10., 28.15682173355295)
        self.assertEqual(format(abs(gradient) / 10., '.3f'), '28.157')
        self.assertNotEqual(abs(gradient) / 10., 28.157)
        self.assertNotEqual(abs(gradient) / 10., 29.06757060377401)
        self.assertEqual(0.3805631199799269, abs(-3.805631199799269) / 10.)
        self.assertGreater(abs(gradient), 0.3805631199799269)
        self.assertLess(abs(pop0), 1.)
        self.assertGreater(abs(pop0), 0.00012529367029458783)
        self.assertEqual(1e-4 * max(abs(pop0), 1.), 1e-4)
        self.assertNotEqual(1e-4 * max(abs(gradient), 1.), 1e-4)
        self.assertEqual(min(abs(value) for value in checked), abs(pop0))
        self.assertGreater(abs(-40.59104905719103), 0.2070679793428482)
        self.assertLessEqual(abs(-4.821083100508709), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(5.257512894145648), abs(-233.42232564563201) / 10.)
        self.assertGreater(abs(-13.58799257709165), 1.)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 0)
        self.assertEqual(objective - 8208480.745624014, 0.0)
        self.assertEqual(5979.804277479278 - 5979.8042774792775, 9.094947017729282e-13)
        self.assertEqual(16777216 + 12, 16777228)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(
            pop3_revisit7_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_POP3_REVISIT7_NO_IMPROVEMENT')
        self.assertEqual(
            full_gradient_record_status(True, True, True, True),
            'CONDITIONAL_FULL_GRADIENT_RECORDED')

    def test_pop0_revisit13_secant_reduced_queues_one_full_gradient(self):
        white = -0.035717259497908965
        verified = -0.035830557565609475
        g_verify = -369.01994600970926
        g_saved = 0.0012529367029458782
        objective0 = 8208480.766528566
        objective1 = 8208480.745624014
        fp0 = 5979.783376979996
        fp1 = 5979.8042774792775
        prior0 = 4.180979274976987
        prior1 = 4.180975221862276
        count = -142313.47804254713
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (white ** 2 - verified ** 2)
        identity0 = ic + tracer_prior + prior0 - count - fp0 - objective0
        identity1 = ic + tracer_prior + prior1 - count - fp1 - objective1
        checked = (
            -62.06240750571833, -25.319409002911684, -33.149212182992535,
            -43.208749387471116, -164.05977003369412, g_saved,
            281.5682173355289, -4.8210831005087655, -40.59104905719137,
            5.257512894150594)
        diffs = {
            0: -4.810374321095878e-12,
            9: 6.0254023992456496e-12,
            14: -9.094947017729282e-13,
        }
        self.assertEqual(0.3 + white, 0.264282740502091)
        self.assertEqual(0.3 + verified, 0.2641694424343905)
        self.assertEqual(delta_objective, -0.020904552191495895)
        self.assertLess(delta_objective, 0.)
        self.assertEqual(delta_fp, 0.02090049928119697)
        self.assertEqual(delta_prior, -4.0531147105227205e-06)
        self.assertEqual(prior_residual, 2.2041830166630305e-16)
        self.assertGreater(prior_residual, 0.)
        self.assertEqual(residual, 2.0441159875872472e-10)
        self.assertGreater(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, 0.21948528289794922)
        self.assertEqual(identity0, 0.0)
        self.assertEqual(identity1, -9.313225746154785e-10)
        self.assertLess(identity1, 0.)
        self.assertLess(g_verify, 0.)
        self.assertGreater(g_saved, 0.)
        self.assertLess(abs(g_saved), 1.)
        self.assertLessEqual(abs(g_saved), abs(g_verify) / 10.)
        self.assertGreater(abs(g_saved), 0.0002332932001004387)
        self.assertEqual(abs(g_verify) / 10., 36.90199460097092)
        self.assertEqual(abs(-369.0199460097153) / 10., 36.90199460097153)
        self.assertEqual(abs(-369.0199460097031) / 10., 36.90199460097031)
        self.assertNotEqual(abs(g_verify) / 10., 36.90199460097153)
        self.assertNotEqual(abs(g_verify) / 10., 36.90199460097031)
        self.assertEqual(format(abs(g_verify) / 10., '.3f'), '36.902')
        self.assertEqual(abs(g_saved) / 10., 0.00012529367029458783)
        self.assertEqual(format(abs(g_saved) / 10., '.3f'), '0.000')
        self.assertNotEqual(abs(g_saved) / 10., 0.0002332932001004387)
        self.assertEqual(1e-4 * max(abs(g_saved), 1.), 1e-4)
        self.assertEqual(min(abs(value) for value in checked), abs(g_saved))
        self.assertGreater(abs(-40.59104905719137), 0.2070679793428482)
        self.assertGreater(abs(281.5682173355289), 0.3805631199799269)
        self.assertLessEqual(abs(-4.8210831005087655), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(5.257512894150594), abs(-233.42232564563201) / 10.)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 9)
        self.assertEqual(diffs[9], 6.0254023992456496e-12)
        self.assertEqual(objective0 - 8208480.766528566, 0.0)
        self.assertEqual(16777216 + 9, 16777225)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(coordinate_line_action(True, True, True, False), 'stop')
        self.assertEqual(
            pop0_revisit13_status(
                objective0, objective1, abs(g_verify), abs(g_saved), count + fp0, count + fp1),
            'CONDITIONAL_POP0_REVISIT13_REDUCED')
        self.assertEqual(
            full_gradient_record_status(True, True, True, True),
            'CONDITIONAL_FULL_GRADIENT_RECORDED')

    def test_pop0_revisit13_sign_change_queues_one_secant(self):
        negative = -0.035830557565609475
        positive = 0.06416944243439053
        g_neg = -369.0199460097153
        g_pos = 325338.1642093004
        objective0 = 8208480.766528566
        objective1 = 8224729.238770537
        fp0 = 5979.783376979996
        fp1 = -10268.687448046829
        prior0 = 4.180979274976987
        prior1 = 4.182396219220426
        count = -142313.47804254713
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        pop4 = -1.100052189061735
        theta = tracer0_secant(positive, g_pos, negative, g_neg)
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (positive ** 2 - negative ** 2)
        identity0 = ic + tracer_prior + prior0 - count - fp0 - objective0
        identity1 = ic + tracer_prior + prior1 - count - fp1 - objective1
        checked = (
            -62.06240750571333, -25.320681980953875, -33.153483480892596,
            -43.21049369264317, -164.06726104265297, g_neg,
            355.30668512865435, pop4, 2.070679793428596, 4.746771779558472)
        diffs = {
            0: 6.248512818274321e-11,
            2: 2.48228104737791e-11,
            3: 2.6048496692965273e-11,
            4: 4.206412995699793e-12,
            6: -1.0885514711844735e-11,
            9: -1.2164491636212915e-11,
            12: -1.1368683772161603e-13,
            13: -1.7053025658242404e-12,
            14: 3.410605131648481e-13,
            22: 4.348521542851813e-12,
        }
        self.assertEqual(positive, negative + 0.1)
        self.assertEqual(pop0_revisit10_step(g_neg), 0.1)
        self.assertEqual(theta, -0.035717259497908965)
        self.assertEqual(0.3 + negative, 0.2641694424343905)
        self.assertEqual(0.3 + positive, 0.36416944243439053)
        self.assertEqual(0.3 + theta, 0.264282740502091)
        self.assertLess(negative, theta)
        self.assertLess(theta, positive)
        self.assertLess(abs(negative - theta), abs(positive - theta))
        self.assertEqual(delta_objective, 16248.47224197071)
        self.assertGreater(delta_objective, 0.)
        self.assertEqual(delta_fp, -16248.470825026825)
        self.assertEqual(delta_prior, 0.0014169442434388557)
        self.assertEqual(prior_residual, -1.973247953923618e-16)
        self.assertLess(prior_residual, 0.)
        self.assertEqual(residual, -3.583409124985337e-10)
        self.assertLess(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, -0.384765625)
        self.assertEqual(identity0, 0.0)
        self.assertEqual(identity1, 0.0)
        self.assertLess(g_neg, 0.)
        self.assertGreater(g_pos, 0.)
        self.assertGreater(abs(g_pos), abs(g_neg) / 10.)
        self.assertEqual(abs(g_neg) / 10., 36.90199460097153)
        self.assertEqual(abs(-369.0199460097031) / 10., 36.90199460097031)
        self.assertNotEqual(abs(g_neg) / 10., abs(-369.0199460097031) / 10.)
        self.assertEqual(format(abs(g_neg) / 10., '.3f'), '36.902')
        self.assertEqual(format(abs(-369.0199460097031) / 10., '.3f'), '36.902')
        self.assertGreaterEqual(min(abs(value) for value in checked), abs(pop4))
        self.assertGreaterEqual(abs(pop4), 1.)
        self.assertNotEqual(1e-4 * max(abs(pop4), 1.), 1e-4)
        self.assertGreater(abs(g_neg), 1.)
        self.assertGreater(abs(g_neg), 0.0002332932001004387)
        self.assertGreater(abs(2.070679793428596), 0.2070679793428482)
        self.assertEqual(abs(2.070679793428482) / 10., 0.2070679793428482)
        self.assertGreater(abs(355.30668512865435), 0.3805631199799269)
        self.assertLessEqual(abs(pop4), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(4.746771779558472), abs(-233.42232564563201) / 10.)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 0)
        self.assertEqual(diffs[0], 6.248512818274321e-11)
        self.assertEqual(diffs[9], -1.2164491636212915e-11)
        self.assertEqual(-142313.47804254713 - -142313.4780425471, -2.9103830456733704e-11)
        self.assertEqual(objective0 - 8208480.766528566, 0.0)
        self.assertEqual(16777216 + 9, 16777225)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        with self.assertRaises(ValueError):
            tracer0_secant(negative, g_neg, positive, g_pos)
        self.assertEqual(
            pop0_revisit13_status(objective0, objective1, abs(g_neg), abs(g_pos), 1., 1.),
            'CONDITIONAL_POP0_REVISIT13_NO_IMPROVEMENT')

    def test_full_gradient64_names_population0_and_steps_by_plus01(self):
        gradient = -369.0199460097031
        white = -0.035830557565609475
        pop4 = -1.1000521890600297
        pop5 = 2.070679793428255
        pop8 = -13.585636501089791
        objective = 8208480.766528566
        checked = (
            -62.062407505775816, -25.320681980978698, -33.153483480918645,
            -43.210493692647375, -164.0672610426421, gradient,
            355.30668512865446, pop4, pop5, 4.746771779554123)
        diffs = {
            0: -6.81197320773208e-11,
            2: -2.4215296434704214e-11,
            3: -2.673061771929497e-11,
            4: -4.092726157978177e-12,
            6: 1.057287590811029e-11,
            9: 6.480149750132114e-12,
            12: -2.2737367544323206e-13,
            13: 2.1032064978498966e-12,
            14: -2.2737367544323206e-13,
            22: -4.595435143528448e-12,
        }
        self.assertEqual(pop0_revisit10_step(gradient), 0.1)
        self.assertEqual(pop0_revisit10_step(-gradient), -0.1)
        self.assertEqual(white + 0.1, 0.06416944243439053)
        self.assertEqual(0.3 + white, 0.2641694424343905)
        self.assertEqual(0.3 + white + 0.1, 0.36416944243439053)
        self.assertLess(gradient, 0.)
        self.assertGreater(abs(gradient), 1.)
        self.assertEqual(abs(gradient) / 10., 36.90199460097031)
        self.assertEqual(format(abs(gradient) / 10., '.3f'), '36.902')
        self.assertNotEqual(abs(gradient) / 10., 36.902)
        self.assertEqual(abs(0.002332932001004387) / 10., 0.0002332932001004387)
        self.assertEqual(format(abs(0.002332932001004387) / 10., '.3f'), '0.000')
        self.assertGreater(abs(gradient), 0.0002332932001004387)
        self.assertGreaterEqual(min(abs(value) for value in checked), abs(pop4))
        self.assertGreaterEqual(abs(pop4), 1.)
        self.assertNotEqual(1e-4 * max(abs(pop4), 1.), 1e-4)
        self.assertGreater(abs(pop5), abs(2.070679793428482) / 10.)
        self.assertEqual(abs(2.070679793428482) / 10., 0.2070679793428482)
        self.assertNotEqual(pop5, gradient)
        self.assertGreater(abs(pop8), 1.)
        self.assertGreater(abs(355.30668512865446), 0.3805631199799269)
        self.assertLessEqual(abs(pop4), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(4.746771779554123), abs(-233.42232564563201) / 10.)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 0)
        self.assertEqual(diffs[0], -6.81197320773208e-11)
        self.assertEqual(diffs[9], 6.480149750132114e-12)
        self.assertEqual(objective - 8208480.766528566, 0.0)
        self.assertEqual(-142313.4780425471 - -142313.47804254713, 2.9103830456733704e-11)
        self.assertEqual(16777216 + 9, 16777225)
        self.assertEqual(16777216 + 14, 16777230)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(
            pop0_revisit13_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_POP0_REVISIT13_NO_IMPROVEMENT')
        self.assertEqual(
            full_gradient_record_status(True, True, True, True),
            'CONDITIONAL_FULL_GRADIENT_RECORDED')

    def test_pop5_revisit2_secant_reduced_queues_one_full_gradient(self):
        white = 0.0765438051588915
        verified = 0.07556377229186596
        g_verify = -167.0647738069621
        g_saved = 2.070679793428482
        objective0 = 8208480.847376873
        objective1 = 8208480.766528566
        fp0 = 5979.702454137936
        fp1 = 5979.783376979996
        prior0 = 4.180904739764374
        prior1 = 4.180979274976987
        count = -142313.47804254713
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (white ** 2 - verified ** 2)
        identity0 = ic + tracer_prior + prior0 - count - fp0 - objective0
        identity1 = ic + tracer_prior + prior1 - count - fp1 - objective1
        checked = (
            -62.0624075057077, -25.320681980954483, -33.153483480891914,
            -43.21049369264328, -164.06726104265266, -369.0199460097096,
            355.3066851286547, -1.1000521890621329, g_saved, 4.746771779558719)
        self.assertEqual(0.5 * white, 0.03827190257944575)
        self.assertEqual(delta_objective, -0.08084830641746521)
        self.assertLess(delta_objective, 0.)
        self.assertEqual(delta_fp, 0.08092284206031763)
        self.assertEqual(delta_prior, 7.453521261258089e-05)
        self.assertEqual(prior_residual, -1.0711917464156784e-16)
        self.assertLess(prior_residual, 0.)
        self.assertEqual(residual, 4.3023984375167856e-10)
        self.assertGreater(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, 0.46196651458740234)
        self.assertEqual(identity0, 0.0)
        self.assertEqual(identity1, 0.0)
        self.assertLess(g_verify, 0.)
        self.assertGreater(g_saved, 0.)
        self.assertLessEqual(abs(g_saved), abs(g_verify) / 10.)
        self.assertGreaterEqual(abs(g_saved), 1.)
        self.assertEqual(abs(g_verify) / 10., 16.70647738069621)
        self.assertEqual(abs(-167.06477380696307) / 10., 16.706477380696306)
        self.assertEqual(abs(-167.06477380696177) / 10., 16.706477380696178)
        self.assertNotEqual(abs(g_verify) / 10., 16.706477380696306)
        self.assertNotEqual(abs(g_verify) / 10., 16.706477380696178)
        self.assertEqual(format(abs(g_verify) / 10., '.3f'), '16.706')
        self.assertEqual(abs(g_saved) / 10., 0.2070679793428482)
        self.assertEqual(format(abs(g_saved) / 10., '.3f'), '0.207')
        self.assertGreater(abs(g_saved), abs(2.2229824052020826) / 10.)
        self.assertEqual(min(abs(value) for value in checked), abs(-1.1000521890621329))
        self.assertGreaterEqual(min(abs(value) for value in checked), 1.)
        self.assertNotEqual(1e-4 * max(abs(-1.1000521890621329), 1.), 1e-4)
        self.assertGreater(abs(-369.0199460097096), 1.)
        self.assertGreater(abs(-369.0199460097096), 0.0002332932001004387)
        self.assertGreater(abs(355.3066851286547), 0.3805631199799269)
        self.assertLessEqual(abs(-1.1000521890621329), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(4.746771779558719), abs(-233.42232564563201) / 10.)
        diffs = {
            0: 7.425171588693047e-12,
            14: 9.663381206337363e-13,
            9: 2.0463630789890885e-12,
        }
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 0)
        self.assertEqual(objective0 - 8208480.847376873, 0.0)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(coordinate_line_action(True, True, True, False), 'stop')
        self.assertEqual(
            pop5_revisit2_status(
                objective0, objective1, abs(g_verify), abs(g_saved), count + fp0, count + fp1),
            'CONDITIONAL_POP5_REVISIT2_REDUCED')
        self.assertEqual(
            full_gradient_record_status(True, True, True, True),
            'CONDITIONAL_FULL_GRADIENT_RECORDED')

    def test_pop5_revisit2_sign_change_queues_one_secant(self):
        negative = 0.07556377229186596
        positive = 0.17556377229186598
        g_neg = -167.06477380696307
        g_pos = 16879.789411197682
        objective0 = 8208480.847376873
        objective1 = 8209320.980869919
        fp0 = 5979.702454137936
        fp1 = 5139.581517469729
        prior0 = 4.180904739764374
        prior1 = 4.193461116993561
        count = -142313.47804254713
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        pop0 = 0.0023329319880440874
        theta = tracer0_secant(positive, g_pos, negative, g_neg)
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (positive ** 2 - negative ** 2)
        identity0 = ic + tracer_prior + prior0 - count - fp0 - objective0
        identity1 = ic + tracer_prior + prior1 - count - fp1 - objective1
        self.assertEqual(positive, negative + 0.1)
        self.assertEqual(pop5_line_step(g_neg), 0.1)
        self.assertEqual(theta, 0.0765438051588915)
        self.assertEqual(0.5 * negative, 0.03778188614593298)
        self.assertEqual(0.5 * positive, 0.08778188614593299)
        self.assertEqual(0.5 * theta, 0.03827190257944575)
        self.assertLess(negative, theta)
        self.assertLess(theta, positive)
        self.assertLess(abs(negative - theta), abs(positive - theta))
        with self.assertRaises(ValueError):
            tracer0_secant(negative, g_neg, positive, g_pos)
        self.assertEqual(delta_objective, 840.1334930462763)
        self.assertGreater(delta_objective, 0.)
        self.assertEqual(delta_fp, -840.1209366682069)
        self.assertEqual(delta_prior, 0.012556377229186566)
        self.assertEqual(prior_residual, -3.469446951953614e-17)
        self.assertLess(prior_residual, 0.)
        self.assertEqual(residual, 8.401457307627425e-10)
        self.assertGreater(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, 0.902099609375)
        self.assertEqual(identity0, 0.0)
        self.assertEqual(identity1, -9.313225746154785e-10)
        self.assertLess(identity1, 0.)
        self.assertGreater(g_pos, 0.)
        self.assertLess(g_neg, 0.)
        self.assertGreater(abs(g_pos), abs(g_neg) / 10.)
        self.assertEqual(abs(g_neg) / 10., 16.706477380696306)
        self.assertEqual(abs(-167.06477380696177) / 10., 16.706477380696178)
        self.assertNotEqual(abs(g_neg) / 10., 16.706477380696178)
        self.assertEqual(format(abs(g_neg) / 10., '.3f'), '16.706')
        self.assertEqual(format(16.706477380696178, '.3f'), '16.706')
        self.assertEqual(abs(g_pos) / 10., 1687.9789411197683)
        self.assertLess(abs(pop0), 1.)
        self.assertGreater(abs(pop0), 0.0002332932001004387)
        self.assertEqual(1e-4 * max(abs(pop0), 1.), 1e-4)
        self.assertGreater(abs(73.15236986684813), 0.3805631199799269)
        self.assertLessEqual(abs(-8.809128948968358), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(4.818644532559582), abs(-233.42232564563201) / 10.)
        self.assertGreater(abs(g_neg), abs(2.2229824052020826) / 10.)
        diffs = {
            0: -8.498091119690798e-12,
            2: 6.643574579356937e-13,
            3: -1.7692514120426495e-12,
            4: -4.263256414560601e-13,
            6: -1.9895196601282805e-13,
            9: -9.094947017729282e-13,
            12: -2.2737367544323206e-13,
            13: 5.684341886080802e-14,
            14: -1.3073986337985843e-12,
            22: 9.121592370320286e-13,
        }
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 0)
        self.assertEqual(diffs[0], -8.498091119690798e-12)
        self.assertEqual(diffs[14], -1.3073986337985843e-12)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        self.assertEqual(
            pop5_revisit2_status(
                objective0, objective1, abs(g_neg), abs(g_pos), count + fp0, count + fp1),
            'CONDITIONAL_POP5_REVISIT2_NO_IMPROVEMENT')

    def test_full_gradient63_names_population5_and_steps_by_plus01(self):
        gradient = -167.06477380696177
        white = 0.07556377229186596
        pop0 = 0.002332931988953582
        pop8 = 0.5032382410476584
        objective = 8208480.847376873
        checked = (
            -62.06240750570625, -25.31329691582682, -33.138602369952025,
            -43.21403617399602, -164.03552834970117, pop0,
            73.15236986684836, -8.809128948968414, gradient,
            4.81864453255867)
        diffs = {
            0: 7.194955742306774e-11,
            2: 2.509992214072554e-11,
            3: 2.8315128020039992e-11,
            4: 4.391154106997419e-12,
            6: -1.1226575225009583e-11,
            9: -1.2050804798491299e-11,
            12: 3.524291969370097e-12,
            13: -1.8189894035458565e-12,
            14: 6.821210263296962e-13,
            22: -9.456435634547233e-12,
        }
        self.assertEqual(pop5_line_step(gradient), 0.1)
        self.assertEqual(white + 0.1, 0.17556377229186598)
        self.assertEqual(0.5 * white, 0.03778188614593298)
        self.assertEqual(0.5 * (white + 0.1), 0.08778188614593299)
        self.assertLess(gradient, 0.)
        self.assertEqual(abs(gradient) / 10., 16.706477380696178)
        self.assertEqual(format(abs(gradient) / 10., '.3f'), '16.706')
        self.assertEqual(abs(2.2229824052020826) / 10., 0.22229824052020825)
        self.assertEqual(format(abs(2.2229824052020826) / 10., '.3f'), '0.222')
        self.assertGreater(abs(gradient), abs(2.2229824052020826) / 10.)
        self.assertLess(abs(pop0), 1.)
        self.assertGreater(abs(pop0), 0.0002332932001004387)
        self.assertEqual(abs(0.002332932001004387) / 10., 0.0002332932001004387)
        self.assertEqual(1e-4 * max(abs(pop0), 1.), 1e-4)
        self.assertLess(abs(pop8), 1.)
        self.assertNotEqual(pop8, gradient)
        self.assertEqual(min(abs(value) for value in checked), abs(pop0))
        others = tuple(value for value in checked if value != pop0)
        self.assertGreaterEqual(min(abs(value) for value in others), 4.81864453255867)
        self.assertGreater(abs(73.15236986684836), 0.3805631199799269)
        self.assertLessEqual(abs(-8.809128948968414), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(4.81864453255867), abs(-233.42232564563201) / 10.)
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 0)
        self.assertEqual(diffs[0], 7.194955742306774e-11)
        self.assertEqual(diffs[14], 6.821210263296962e-13)
        self.assertEqual(objective - 8208480.847376873, 0.0)
        self.assertEqual(-142313.47804254713 - -142313.4780425471, -2.9103830456733704e-11)
        self.assertEqual(5979.702454137936 - 5979.702454137935, 9.094947017729282e-13)
        self.assertEqual(16777216 + 14, 16777230)
        self.assertEqual(16777216 + 9, 16777225)
        self.assertEqual(16777216 + 17, 16777233)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(
            pop5_revisit2_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_POP5_REVISIT2_NO_IMPROVEMENT')
        self.assertEqual(
            full_gradient_record_status(True, True, True, True),
            'CONDITIONAL_FULL_GRADIENT_RECORDED')

    def test_pop0_revisit12_secant_reduced_queues_one_full_gradient(self):
        white = -0.035830557565609475
        verified = -0.035712314270852585
        g_verify = 385.14365477992845
        g_saved = 0.002332932001004387
        objective0 = 8208480.870147339
        objective1 = 8208480.847376873
        count_line = -142313.47804254713
        count = -142313.4780425471
        fp0 = 5979.679679442886
        fp1 = 5979.702454137935
        prior0 = 4.1809005100319325
        prior1 = 4.180904739764374
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (white ** 2 - verified ** 2)
        identity0 = ic + tracer_prior + prior0 - count - fp0 - objective0
        identity1 = ic + tracer_prior + prior1 - count - fp1 - objective1
        self.assertEqual(white, -0.035830557565609475)
        self.assertNotEqual(white, verified)
        self.assertNotEqual(white, -0.1357123142708526)
        self.assertNotEqual(white, -0.03582769110596287)
        self.assertEqual(0.3 + white, 0.2641694424343905)
        self.assertEqual(delta_objective, -0.022770466282963753)
        self.assertLess(delta_objective, 0.)
        self.assertEqual(delta_fp, 0.022774695049520233)
        self.assertGreater(delta_fp, 0.)
        self.assertEqual(delta_prior, 4.229732441629608e-06)
        self.assertGreater(delta_prior, 0.)
        self.assertEqual(prior_residual, 4.730374078554256e-16)
        self.assertGreater(prior_residual, 0.)
        self.assertEqual(residual, -9.658851496396892e-10)
        self.assertLess(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertGreater(abs(residual), 9.313225746154785e-10)
        self.assertEqual(residual / 9.313225746154785e-10, -1.0371112823486328)
        self.assertEqual(identity0, -9.313225746154785e-10)
        self.assertLess(identity0, 0.)
        self.assertEqual(identity1, 0.0)
        self.assertEqual(objective0 - 8208480.870147339, 0.0)
        self.assertEqual(count - count_line, 2.9103830456733704e-11)
        self.assertGreater(g_verify, 0.)
        self.assertGreater(g_saved, 0.)
        self.assertLess(abs(g_saved), 1.)
        self.assertLessEqual(abs(g_saved), abs(g_verify) / 10.)
        self.assertEqual(abs(g_verify) / 10., 38.51436547799285)
        self.assertEqual(abs(385.14365477991754) / 10., 38.51436547799175)
        self.assertEqual(abs(385.14365477991436) / 10., 38.51436547799143)
        self.assertNotEqual(abs(g_verify) / 10., 38.51436547799175)
        self.assertNotEqual(abs(g_verify) / 10., 38.51436547799143)
        self.assertNotEqual(abs(g_verify) / 10., 37.57900670731469)
        self.assertNotEqual(abs(g_verify) / 10., 37.579006707314704)
        self.assertNotEqual(abs(g_verify) / 10., 37.579006707314626)
        self.assertEqual(format(abs(g_verify) / 10., '.3f'), '38.514')
        self.assertEqual(format(38.51436547799175, '.3f'), '38.514')
        self.assertEqual(format(38.51436547799143, '.3f'), '38.514')
        self.assertEqual(abs(g_saved) / 10., 0.0002332932001004387)
        self.assertEqual(format(abs(g_saved) / 10., '.3f'), '0.000')
        self.assertEqual(1e-4 * max(abs(g_saved), 1.), 1e-4)
        diffs = {
            0: -6.535572083521402e-11,
            2: -2.5476509790678392e-11,
            3: -2.6908253403234994e-11,
            4: -4.305888978706207e-12,
            6: 1.1311840353300795e-11,
            9: 1.0913936421275139e-11,
            12: -2.6147972675971687e-12,
            13: 2.1032064978498966e-12,
            14: -4.547473508864641e-13,
            22: 3.5980107782052073e-12,
        }
        self.assertEqual(max(diffs, key=lambda index: abs(diffs[index])), 0)
        self.assertEqual(diffs[0], -6.535572083521402e-11)
        self.assertEqual(diffs[9], 1.0913936421275139e-11)
        saved = (
            -62.0624075057782, -25.31329691585192, -33.13860236998034,
            -43.21403617400041, -164.03552834968994, g_saved,
            73.15236986684484, -8.809128948966595, -167.06477380696245,
            4.818644532568126)
        others = tuple(value for value in saved if value != g_saved)
        self.assertEqual(min(abs(value) for value in saved), abs(g_saved))
        self.assertGreaterEqual(min(abs(value) for value in others), 4.818644532568126)
        self.assertGreater(abs(73.15236986684484), 0.3805631199799269)
        self.assertGreater(abs(-167.06477380696245), abs(2.2229824052020826) / 10.)
        self.assertLessEqual(abs(-8.809128948966595), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(4.818644532568126), abs(-233.42232564563201) / 10.)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(
            pop0_revisit12_status(
                objective0, objective1, abs(g_verify), abs(g_saved), count + fp0, count + fp1),
            'CONDITIONAL_POP0_REVISIT12_REDUCED')
        self.assertEqual(coordinate_line_action(True, False, True, False), 'stop')
        self.assertEqual(
            full_gradient_record_status(True, True, True, True),
            'CONDITIONAL_FULL_GRADIENT_RECORDED')

    def test_pop0_revisit12_sign_change_queues_one_secant(self):
        positive = -0.035712314270852585
        negative = -0.1357123142708526
        g_pos = 385.14365477991754
        g_neg = -325336.2053416151
        objective0 = 8208480.870147339
        objective1 = 8224728.387560036
        count = -142313.47804254713
        fp0 = 5979.679679442886
        fp1 = -10267.82916202327
        prior0 = 4.1809005100319325
        prior1 = 4.189471741459018
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        theta = tracer0_secant(positive, g_pos, negative, g_neg)
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (negative ** 2 - positive ** 2)
        failed_identity = ic + tracer_prior + prior1 - count - fp1 - objective1
        self.assertEqual(positive - 0.1, negative)
        self.assertEqual(pop0_revisit10_step(g_pos), -0.1)
        self.assertEqual(theta, -0.035830557565609475)
        self.assertEqual(0.3 + positive, 0.2642876857291474)
        self.assertEqual(0.3 + negative, 0.1642876857291474)
        self.assertEqual(0.3 + theta, 0.2641694424343905)
        self.assertLess(negative, theta)
        self.assertLess(theta, positive)
        self.assertLess(positive - theta, theta - negative)
        self.assertNotEqual(theta, -0.03582769110596287)
        self.assertEqual(delta_objective, 16247.517412696965)
        self.assertGreater(delta_objective, 0.)
        self.assertEqual(delta_fp, -16247.508841466155)
        self.assertEqual(delta_prior, 0.00857123142708538)
        self.assertEqual(prior_residual, 1.214306433183765e-16)
        self.assertGreater(prior_residual, 0.)
        self.assertEqual(residual, -6.166374078020453e-10)
        self.assertLess(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(ic + tracer_prior + prior0 - count - fp0 - objective0, 0.0)
        self.assertEqual(failed_identity, -9.313225746154785e-10)
        self.assertLess(failed_identity, 0.)
        self.assertGreater(g_pos, 0.)
        self.assertLess(g_neg, 0.)
        self.assertGreater(abs(g_neg), abs(g_pos) / 10.)
        self.assertEqual(abs(g_pos) / 10., 38.51436547799175)
        self.assertEqual(abs(385.14365477991436) / 10., 38.51436547799143)
        self.assertNotEqual(abs(g_pos) / 10., 38.51436547799143)
        self.assertEqual(format(abs(g_pos) / 10., '.3f'), '38.514')
        self.assertEqual(format(abs(385.14365477991436) / 10., '.3f'), '38.514')
        self.assertEqual(g_pos - 385.14365477991436, 3.183231456205249e-12)
        self.assertEqual(5.320867990585626 - 5.320867990581763, 3.862687947275845e-12)
        checked = (
            -62.06240750571262, -25.311987434439292, -33.134245760489684,
            -43.21226450868933, -164.02789424763094, g_pos,
            -3.8056311997976775, -12.692243444729861, -211.58932288561138,
            5.320867990585626)
        self.assertEqual(min(abs(value) for value in checked), abs(-3.8056311997976775))
        self.assertEqual(abs(-3.8056311997976775), 3.8056311997976775)
        self.assertGreaterEqual(min(abs(value) for value in checked), 1.)
        self.assertGreater(abs(-3.8056311997976775), 0.3805631199799269)
        self.assertGreater(abs(-211.58932288561138), abs(2.2229824052020826) / 10.)
        self.assertLessEqual(abs(-12.692243444729861), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(5.320867990585626), abs(-233.42232564563201) / 10.)
        self.assertLess(abs(0.5010543086435217), 1.)
        with self.assertRaises(ValueError):
            tracer0_secant(negative, g_neg, positive, g_pos)
        self.assertEqual(
            pop0_revisit12_status(
                objective0, objective1, abs(g_pos), abs(g_neg), count + fp0, count + fp1),
            'CONDITIONAL_POP0_REVISIT12_NO_IMPROVEMENT')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')

    def test_full_gradient62_names_population0_and_steps_by_minus01(self):
        gradient = 385.14365477991436
        white = -0.035712314270852585
        pop3 = -3.805631199798928
        pop8 = 0.5010543086435217
        objective = 8208480.870147339
        checked = (
            -62.06240750571436, -25.31198743443897, -33.13424576048997,
            -43.212264508689586, -164.02789424763097, gradient,
            pop3, -12.69224344472952, -211.58932288561132, 5.320867990581763)
        self.assertEqual(abs(gradient) / 10., 38.51436547799143)
        self.assertEqual(format(abs(gradient) / 10., '.3f'), '38.514')
        self.assertEqual(abs(0.001279714049275571) / 10., 0.0001279714049275571)
        self.assertGreater(abs(gradient), abs(0.001279714049275571) / 10.)
        self.assertNotEqual(abs(gradient) / 10., 37.57900670731469)
        self.assertNotEqual(abs(gradient) / 10., 37.579006707314704)
        self.assertNotEqual(abs(gradient) / 10., 37.579006707314626)
        self.assertEqual(format(37.57900670731469, '.3f'), '37.579')
        self.assertEqual(pop0_revisit10_step(gradient), -0.1)
        self.assertEqual(white - 0.1, -0.1357123142708526)
        self.assertEqual(0.3 + white, 0.2642876857291474)
        self.assertEqual(0.3 + (white - 0.1), 0.1642876857291474)
        self.assertGreater(gradient, 1.)
        self.assertLess(abs(pop8), 1.)
        self.assertNotEqual(pop8, gradient)
        self.assertEqual(min(abs(value) for value in checked), abs(pop3))
        self.assertEqual(abs(pop3), 3.805631199798928)
        self.assertGreaterEqual(min(abs(value) for value in checked), 1.)
        self.assertGreater(abs(pop3), 0.3805631199799269)
        self.assertGreater(abs(-211.58932288561132), abs(2.2229824052020826) / 10.)
        self.assertLessEqual(abs(-12.69224344472952), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(5.320867990581763), abs(-233.42232564563201) / 10.)
        self.assertEqual(gradient - 385.14365477991845, -4.092726157978177e-12)
        self.assertEqual(objective - 8208480.870147339, 0.0)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(16777216 + 9, 16777225)
        self.assertEqual(16777216 + 17, 16777233)
        self.assertEqual(
            pop0_revisit12_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_POP0_REVISIT12_NO_IMPROVEMENT')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        self.assertEqual(
            full_gradient_record_status(True, True, True, True),
            'CONDITIONAL_FULL_GRADIENT_RECORDED')

    def test_pop3_revisit6_secant_reduced_queues_one_full_gradient(self):
        positive = -0.41031737268391766
        negative = -0.5103173726839176
        theta = -0.41090912289159987
        g_verify = 290.6757060377417
        g_saved = -3.805631199799269
        objective0 = 8208480.955026312
        objective1 = 8208480.870147339
        count = -142313.47804254713
        fp0 = 5979.594557488588
        fp1 = 5979.679679442886
        prior0 = 4.180657529557277
        prior1 = 4.1809005100319325
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        prior_residual = delta_prior - 0.5 * (theta ** 2 - positive ** 2)
        self.assertEqual(0.5 * theta, -0.20545456144579993)
        self.assertLess(negative, theta)
        self.assertLess(theta, positive)
        self.assertLess(positive - theta, theta - negative)
        self.assertEqual(delta_objective, -0.08487897273153067)
        self.assertLess(delta_objective, 0.)
        self.assertEqual(delta_fp, 0.08512195429739222)
        self.assertGreater(delta_fp, 0.)
        self.assertEqual(delta_prior, 0.0002429804746553188)
        self.assertGreater(delta_prior, 0.)
        self.assertEqual(prior_residual, -1.5265566588595902e-16)
        self.assertLess(prior_residual, 0.)
        self.assertEqual(residual, 1.0912062364809572e-09)
        self.assertGreater(residual, 0.)
        self.assertGreater(abs(residual), 1e-9)
        self.assertEqual(residual / 9.313225746154785e-10, 1.1716737747192383)
        self.assertEqual(ic + tracer_prior + prior0 - count - fp0 - objective0, 0.0)
        self.assertEqual(ic + tracer_prior + prior1 - count - fp1 - objective1, 0.0)
        self.assertGreater(g_verify, 0.)
        self.assertLess(g_saved, 0.)
        self.assertGreater(abs(g_saved), 1.)
        self.assertLessEqual(abs(g_saved), abs(g_verify) / 10.)
        self.assertEqual(abs(g_verify) / 10., 29.06757060377417)
        self.assertEqual(abs(290.67570603774055) / 10., 29.067570603774055)
        self.assertEqual(abs(290.6757060377401) / 10., 29.06757060377401)
        self.assertNotEqual(abs(g_verify) / 10., 29.067570603774055)
        self.assertNotEqual(abs(g_verify) / 10., 29.06757060377401)
        self.assertEqual(format(abs(g_verify) / 10., '.3f'), '29.068')
        self.assertEqual(format(abs(290.67570603774055) / 10., '.3f'), '29.068')
        self.assertEqual(format(abs(290.6757060377401) / 10., '.3f'), '29.068')
        self.assertEqual(abs(g_saved) / 10., 0.3805631199799269)
        self.assertEqual(format(abs(g_saved) / 10., '.3f'), '0.381')
        self.assertEqual(g_verify - 290.67570603774055, 1.1368683772161603e-12)
        self.assertEqual(5.337583066630342 - 5.33758306662161, 8.731682044071931e-12)
        derivatives = (
            -62.062407505713416, -25.311987434439242, -33.13424576049,
            -43.21226450868937, -164.02789424763083, 385.14365477991845,
            g_saved, -12.692243444729463, -211.58932288561115,
            5.320867990579572)
        self.assertEqual(min(abs(value) for value in derivatives), abs(g_saved))
        self.assertEqual(min(abs(value) for value in derivatives), 3.805631199799269)
        self.assertGreaterEqual(min(abs(value) for value in derivatives), 1.)
        self.assertGreater(385.14365477991845, 1.)
        self.assertGreater(abs(385.14365477991845), abs(0.001279714049275571) / 10.)
        self.assertGreater(abs(-211.58932288561115), abs(2.2229824052020826) / 10.)
        self.assertLessEqual(abs(-12.692243444729463), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(5.320867990579572), abs(-233.42232564563201) / 10.)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(
            pop3_revisit6_status(
                objective0, objective1, abs(g_verify), abs(g_saved), count + fp0, count + fp1),
            'CONDITIONAL_POP3_REVISIT6_REDUCED')
        self.assertEqual(coordinate_line_action(True, True, True, False), 'stop')
        self.assertEqual(
            full_gradient_record_status(True, True, True, True),
            'CONDITIONAL_FULL_GRADIENT_RECORDED')

    def test_pop3_revisit6_sign_change_queues_one_secant(self):
        positive = -0.41031737268391766
        negative = -0.5103173726839176
        g_pos = 290.67570603774055
        g_neg = -48830.676895809076
        objective0 = 8208480.955026312
        objective1 = 8210927.057550839
        count = -142313.47804254713
        fp0 = 5979.594557488588
        fp1 = 3533.5380646989197
        prior0 = 4.180657529557277
        prior1 = 4.2266892668256695
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        theta = tracer0_secant(positive, g_pos, negative, g_neg)
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        self.assertEqual(positive - 0.1, negative)
        self.assertEqual(pop3_revisit5_step(g_pos), -0.1)
        self.assertEqual(theta, -0.41090912289159987)
        self.assertEqual(0.5 * positive, -0.20515868634195883)
        self.assertEqual(0.5 * negative, -0.2551586863419588)
        self.assertEqual(0.5 * theta, -0.20545456144579993)
        self.assertLess(negative, theta)
        self.assertLess(theta, positive)
        self.assertLess(positive - theta, theta - negative)
        self.assertEqual(delta_objective, 2446.1025245273486)
        self.assertGreater(delta_objective, 0.)
        self.assertEqual(delta_fp, -2446.0564927896685)
        self.assertEqual(delta_prior, 0.04603173726839227)
        self.assertEqual(residual, 4.1154635255225003e-10)
        self.assertGreater(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(
            delta_prior - 0.5 * (negative ** 2 - positive ** 2), 5.273559366969494e-16)
        self.assertGreater(delta_prior - 0.5 * (negative ** 2 - positive ** 2), 0.)
        self.assertEqual(ic + tracer_prior + prior0 - count - fp0 - objective0, 0.0)
        self.assertEqual(ic + tracer_prior + prior1 - count - fp1 - objective1, 0.0)
        self.assertGreater(g_pos, 0.)
        self.assertLess(g_neg, 0.)
        self.assertGreater(abs(g_neg), abs(g_pos) / 10.)
        self.assertEqual(abs(g_pos) / 10., 29.067570603774055)
        self.assertNotEqual(abs(g_pos) / 10., abs(290.6757060377401) / 10.)
        self.assertEqual(format(abs(g_pos) / 10., '.3f'), '29.068')
        self.assertEqual(format(abs(290.6757060377401) / 10., '.3f'), '29.068')
        self.assertEqual(g_pos - 290.6757060377401, 4.547473508864641e-13)
        self.assertEqual(-62.06240750571306 - (-62.062407505775354), 6.229328164408798e-11)
        self.assertEqual(0.0012797140488208236 - 0.0012797140533682971, -4.547473508864641e-12)
        self.assertLess(abs(0.0012797140488208236), 1.)
        self.assertEqual(1e-4 * max(abs(0.0012797140488208236), 1.), 1e-4)
        self.assertGreater(abs(-41.22166316079629), abs(2.2229824052020826) / 10.)
        self.assertLessEqual(abs(2.207010726045546), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(5.33758306662161), abs(-233.42232564563201) / 10.)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(
            pop3_revisit6_status(objective0, objective0, abs(g_pos), abs(g_pos), 1., 1.),
            'CONDITIONAL_POP3_REVISIT6_NO_IMPROVEMENT')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        self.assertEqual(coordinate_line_action(False, False, False, False), 'midpoint')
        with self.assertRaises(ValueError):
            tracer0_secant(negative, g_neg, positive, g_pos)

    def test_full_gradient61_names_population3_and_steps_by_minus01(self):
        gradient = 290.6757060377401
        white = -0.41031737268391766
        pop0 = 0.0012797140533682971
        objective = 8208480.955026312
        self.assertEqual(abs(gradient) / 10., 29.06757060377401)
        self.assertEqual(format(abs(gradient) / 10., '.3f'), '29.068')
        self.assertEqual(abs(273.5525091954018) / 10., 27.35525091954018)
        self.assertGreater(abs(gradient), abs(273.5525091954018) / 10.)
        self.assertNotEqual(abs(gradient) / 10., abs(273.5525091954018) / 10.)
        self.assertEqual(pop3_revisit5_step(gradient), -0.1)
        self.assertEqual(white - 0.1, -0.5103173726839176)
        self.assertEqual(0.5 * white, -0.20515868634195883)
        self.assertEqual(0.5 * (white - 0.1), -0.2551586863419588)
        self.assertLess(abs(pop0), 1.)
        self.assertEqual(1e-4 * max(abs(pop0), 1.), 1e-4)
        self.assertGreater(abs(pop0), abs(0.001279714049275571) / 10.)
        self.assertEqual(abs(0.001279714049275571) / 10., 0.0001279714049275571)
        self.assertEqual(pop0 - 0.001279714049275571, 4.092726157978177e-12)
        self.assertEqual(-62.062407505775354 - (-62.0624075057131), -6.225775450729998e-11)
        self.assertEqual(gradient - 290.67570603774215, -2.0463630789890885e-12)
        self.assertGreater(abs(-41.22166316079686), abs(2.2229824052020826) / 10.)
        self.assertLessEqual(abs(2.207010726047706), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(5.337583066622327), abs(-233.42232564563201) / 10.)
        self.assertGreater(abs(gradient), abs(pop0))
        self.assertGreater(abs(gradient), abs(-164.05538348949278))
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        self.assertEqual(
            pop3_revisit6_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_POP3_REVISIT6_NO_IMPROVEMENT')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')

    def test_pop0_revisit11_secant_reduced_queues_one_full_gradient(self):
        negative = -0.03582769110596287
        theta = -0.035712314270852585
        g_neg = -375.79006707314625
        g_saved = 0.001279714049275571
        objective0 = 8208480.976704973
        objective1 = 8208480.955026312
        count = -142313.47804254713
        fp0 = 5979.572882955182
        fp1 = 5979.594557488588
        prior0 = 4.180661656586979
        prior1 = 4.180657529557277
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        self.assertEqual(0.3 + theta, 0.2642876857291474)
        self.assertLess(negative, theta)
        self.assertLess(theta, 0.06417230889403713)
        self.assertEqual(delta_objective, -0.02167866099625826)
        self.assertLess(delta_objective, 0.)
        self.assertEqual(delta_fp, 0.021674533406439878)
        self.assertEqual(delta_prior, -4.127029701983531e-06)
        self.assertLess(delta_prior, 0.)
        self.assertEqual(residual, -5.601163977075885e-10)
        self.assertLess(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(
            delta_prior - 0.5 * (theta ** 2 - negative ** 2), 9.139824314052802e-17)
        self.assertGreater(delta_prior - 0.5 * (theta ** 2 - negative ** 2), 0.)
        self.assertEqual(ic + tracer_prior + prior0 - count - fp0 - objective0, 0.0)
        self.assertEqual(ic + tracer_prior + prior1 - count - fp1 - objective1, 0.0)
        self.assertLess(g_neg, 0.)
        self.assertGreater(g_saved, 0.)
        self.assertLess(abs(g_saved), 1.)
        self.assertLessEqual(abs(g_saved), abs(g_neg) / 10.)
        self.assertEqual(abs(g_neg) / 10., 37.579006707314626)
        self.assertNotEqual(abs(g_neg) / 10., 37.579006707314704)
        self.assertNotEqual(abs(g_neg) / 10., 37.57900670731469)
        self.assertEqual(format(abs(g_neg) / 10., '.3f'), '37.579')
        self.assertEqual(abs(g_saved) / 10., 0.0001279714049275571)
        self.assertEqual(format(abs(g_saved) / 10., '.3f'), '0.000')
        self.assertEqual(1e-4 * max(abs(g_saved), 1.), 1e-4)
        self.assertEqual(g_neg - (-375.79006707314704), 7.958078640513122e-13)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        derivatives = (
            -62.0624075057131, -25.318224454812352, -33.14846458804153,
            -43.20930577869501, -164.05538348950353, g_saved,
            290.67570603774215, 2.207010726045489, -41.22166316079538,
            5.337583066630572)
        self.assertEqual(min(abs(value) for value in derivatives), abs(g_saved))
        self.assertGreaterEqual(min(abs(value) for value in derivatives if value != g_saved), 1.)
        self.assertGreater(abs(-41.22166316079538), abs(2.2229824052020826) / 10.)
        self.assertGreater(abs(290.67570603774215), abs(273.5525091954018) / 10.)
        self.assertLessEqual(abs(2.207010726045489), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(5.337583066630572), abs(-233.42232564563201) / 10.)
        self.assertEqual(
            pop0_revisit11_status(
                objective0, objective1, abs(g_neg), abs(g_saved), count + fp0, count + fp1),
            'CONDITIONAL_POP0_REVISIT11_REDUCED')
        self.assertEqual(coordinate_line_action(True, True, True, False), 'stop')
        self.assertEqual(
            full_gradient_record_status(True, True, True, True),
            'CONDITIONAL_FULL_GRADIENT_RECORDED')

    def test_pop0_revisit11_sign_change_queues_one_secant(self):
        negative = -0.03582769110596287
        positive = 0.06417230889403713
        g_neg = -375.79006707314704
        g_pos = 325330.8968202149
        objective0 = 8208480.976704973
        objective1 = 8224728.747110141
        count = -142313.47804254713
        fp0 = 5979.572882955182
        fp1 = -10268.19610498194
        prior0 = 4.180661656586979
        prior1 = 4.182078887476384
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        theta = tracer0_secant(positive, g_pos, negative, g_neg)
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        self.assertEqual(negative + 0.1, positive)
        self.assertEqual(pop0_revisit10_step(g_neg), 0.1)
        self.assertEqual(theta, -0.035712314270852585)
        self.assertEqual(0.3 + negative, 0.2641723088940371)
        self.assertEqual(0.3 + positive, 0.3641723088940371)
        self.assertEqual(0.3 + theta, 0.2642876857291474)
        self.assertLess(negative, theta)
        self.assertLess(theta, positive)
        self.assertGreater(positive - theta, theta - negative)
        self.assertEqual(delta_objective, 16247.770405168645)
        self.assertEqual(delta_fp, -16247.768987937121)
        self.assertEqual(delta_prior, 0.001417230889404486)
        self.assertEqual(residual, 6.348273018375039e-10)
        self.assertGreater(residual, 0.)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(
            delta_prior - 0.5 * (positive ** 2 - negative ** 2), 7.732529894166618e-16)
        self.assertGreater(delta_prior - 0.5 * (positive ** 2 - negative ** 2), 0.)
        self.assertEqual(ic + tracer_prior + prior0 - count - fp0 - objective0, 0.0)
        self.assertEqual(
            ic + tracer_prior + prior1 - count - fp1 - objective1, -9.313225746154785e-10)
        self.assertLess(ic + tracer_prior + prior1 - count - fp1 - objective1, 0.)
        self.assertLess(g_neg, 0.)
        self.assertGreater(g_pos, 0.)
        self.assertGreater(abs(g_pos), abs(g_neg) / 10.)
        self.assertEqual(abs(g_neg) / 10., 37.579006707314704)
        self.assertNotEqual(abs(g_neg) / 10., abs(-375.79006707314693) / 10.)
        self.assertEqual(abs(-375.79006707314693) / 10., 37.57900670731469)
        self.assertEqual(format(abs(g_neg) / 10., '.3f'), '37.579')
        self.assertEqual(format(abs(-375.79006707314693) / 10., '.3f'), '37.579')
        self.assertEqual(g_neg - (-375.79006707314693), -1.1368683772161603e-13)
        self.assertEqual(fp0 - 5979.572882955183, -9.094947017729282e-13)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        derivatives = (
            -62.062407505708, -25.319523104424974, -33.15281949135617,
            -43.211083572049716, -164.0630168431221, g_neg,
            365.7673904138078, 5.996331726480541, 2.2229824052012868,
            4.790307172798412)
        self.assertEqual(min(abs(value) for value in derivatives), 2.2229824052012868)
        self.assertGreaterEqual(min(abs(value) for value in derivatives), 1.)
        self.assertGreater(abs(2.2229824052012868), abs(2.2229824052020826) / 10.)
        self.assertGreater(abs(365.7673904138078), abs(273.5525091954018) / 10.)
        self.assertLessEqual(abs(5.996331726480541), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(4.790307172798412), abs(-233.42232564563201) / 10.)
        self.assertEqual(-25.319523104424974 - (-25.31952310442535), 3.765876499528531e-13)
        self.assertEqual(-33.15281949135617 - (-33.152819491356674), 5.044853423896711e-13)
        self.assertEqual(-43.211083572049716 - (-43.21108357204946), -2.5579538487363607e-13)
        self.assertEqual(-164.0630168431221 - (-164.06301684312194), -1.7053025658242404e-13)
        self.assertEqual(365.7673904138078 - 365.76739041380927, -1.4779288903810084e-12)
        self.assertEqual(5.996331726480541 - 5.996331726480484, 5.684341886080802e-14)
        self.assertEqual(2.2229824052012868 - 2.2229824052014004, -1.1368683772161603e-13)
        self.assertEqual(4.790307172798412 - 4.79030717280601, -7.597478202114871e-12)
        self.assertEqual(
            pop0_revisit11_status(objective0, objective0, abs(g_neg), abs(g_neg), 1., 1.),
            'CONDITIONAL_POP0_REVISIT11_NO_IMPROVEMENT')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        self.assertEqual(coordinate_line_action(False, False, False, False), 'midpoint')
        with self.assertRaises(ValueError):
            tracer0_secant(negative, g_neg, positive, g_pos)

    def test_full_gradient60_names_population0_and_steps_by_01(self):
        gradient = -375.79006707314693
        white = -0.03582769110596287
        objective = 8208480.976704973
        count = -142313.47804254713
        fp = 5979.572882955183
        step_fp = 5979.572882955182
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        population = 4.180661656586979
        self.assertEqual(abs(gradient) / 10., 37.57900670731469)
        self.assertEqual(format(abs(gradient) / 10., '.3f'), '37.579')
        self.assertEqual(abs(362.2367269677485) / 10., 36.22367269677485)
        self.assertGreater(abs(gradient), abs(362.2367269677485) / 10.)
        self.assertEqual(pop0_revisit10_step(gradient), 0.1)
        self.assertEqual(white + 0.1, 0.06417230889403713)
        self.assertEqual(0.3 + white, 0.2641723088940371)
        self.assertEqual(0.3 + (white + 0.1), 0.3641723088940371)
        self.assertEqual(ic + tracer_prior + population - count - fp - objective, 0.0)
        self.assertEqual(fp - step_fp, 9.094947017729282e-13)
        self.assertEqual(2.2229824052014004 - 2.2229824052020826, -6.821210263296962e-13)
        self.assertGreater(abs(2.2229824052014004), abs(2.2229824052020826) / 10.)
        self.assertGreater(abs(365.76739041380927), abs(273.5525091954018) / 10.)
        self.assertLessEqual(abs(5.996331726480484), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(4.79030717280601), abs(-233.42232564563201) / 10.)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        derivatives = (
            -62.062407505708, -25.31952310442535, -33.152819491356674,
            -43.21108357204946, -164.06301684312194, gradient,
            365.76739041380927, 5.996331726480484, 2.2229824052014004, 4.79030717280601)
        self.assertGreaterEqual(min(abs(value) for value in derivatives), 1.)
        self.assertGreaterEqual(2.149910772598043, 1.)
        self.assertEqual(
            pop0_revisit11_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_POP0_REVISIT11_NO_IMPROVEMENT')
        self.assertEqual(
            pop0_revisit11_status(10., 9., abs(gradient), abs(gradient) / 10., 1., 1.),
            'CONDITIONAL_POP0_REVISIT11_REDUCED')
        self.assertEqual(
            pop0_revisit11_status(10., 9., abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_POP0_REVISIT11_IMPROVED')
        self.assertEqual(coordinate_line_action(True, False, False, False), 'continue')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        with self.assertRaises(ValueError):
            pop0_revisit10_step(0.)

    def test_pop5_revisit_reduced_queues_one_full_gradient(self):
        white0 = 0.07451017941546588
        white1 = 0.07556377229186596
        gradient0 = -179.60957974176367
        gradient1 = 2.2229824052020826
        objective0 = 8208481.070149725
        objective1 = 8208480.976704973
        count = -142313.47804254713
        fp0 = 5979.479359143986
        fp1 = 5979.572882955182
        prior0 = 4.180582598163753
        prior1 = 4.180661656586979
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        self.assertEqual(white1, 0.07556377229186596)
        self.assertEqual(0.5 * white1, 0.03778188614593298)
        self.assertLess(white0, white1)
        self.assertEqual(delta_objective, -0.09344475250691175)
        self.assertEqual(delta_fp, 0.09352381119606434)
        self.assertEqual(delta_prior, 7.905842322575296e-05)
        self.assertEqual(residual, 2.659268361071554e-10)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(
            delta_prior - 0.5 * (white1 ** 2 - white0 ** 2), -2.745199900733297e-16)
        self.assertEqual(ic + tracer_prior + prior0 - count - fp0 - objective0, 0.0)
        self.assertEqual(ic + tracer_prior + prior1 - count - fp1 - objective1, 0.0)
        self.assertEqual(fp0 - 5979.479359143987, -9.094947017729282e-13)
        self.assertGreater(gradient1, 0.)
        self.assertLessEqual(abs(gradient1), abs(gradient0) / 10.)
        self.assertEqual(abs(gradient0) / 10., 17.960957974176367)
        self.assertNotEqual(abs(gradient0) / 10., 17.960957974176385)
        self.assertNotEqual(abs(gradient0) / 10., 17.960957974176328)
        self.assertEqual(format(abs(gradient0) / 10., '.3f'), '17.961')
        self.assertEqual(abs(gradient1) / 10., 0.22229824052020825)
        self.assertEqual(format(abs(gradient1) / 10., '.3f'), '0.222')
        self.assertEqual(gradient0 - (-179.60957974176384), 1.7053025658242404e-13)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        derivatives = (
            -62.06240750570924, -25.319523104425386, -33.15281949135641,
            -43.211083572049716, -164.063016843122, -375.7900670731468,
            365.76739041381035, 5.996331726480484, gradient1, 4.790307172812899)
        self.assertEqual(min(abs(value) for value in derivatives), abs(gradient1))
        self.assertGreaterEqual(min(abs(value) for value in derivatives), 1.)
        self.assertGreater(abs(-375.7900670731468), abs(362.2367269677485) / 10.)
        self.assertGreater(abs(365.76739041381035), abs(273.5525091954018) / 10.)
        self.assertLessEqual(abs(5.996331726480484), 542.8941177183742 / 10.)
        self.assertLessEqual(abs(4.790307172812899), abs(-233.42232564563201) / 10.)
        self.assertEqual(
            pop5_revisit_status(
                objective0, objective1, abs(gradient0), abs(gradient1), count + fp0, count + fp1),
            'CONDITIONAL_POP5_REVISIT_REDUCED')
        self.assertEqual(coordinate_line_action(True, True, True, False), 'stop')

    def test_pop5_revisit_sign_change_queues_one_secant(self):
        negative = 0.07451017941546588
        positive = 0.17451017941546587
        g_neg = -179.60957974176384
        g_pos = 16867.732307710565
        objective0 = 8208481.070149725
        objective1 = 8209319.969881736
        count = -142313.47804254713
        fp0 = 5979.479359143987
        fp1 = 5140.592078152069
        prior0 = 4.180582598163753
        prior1 = 4.1930336161053
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        theta = tracer0_secant(positive, g_pos, negative, g_neg)
        delta_objective = objective1 - objective0
        delta_fp = fp1 - fp0
        delta_prior = prior1 - prior0
        residual = delta_objective - (-delta_fp + delta_prior)
        self.assertEqual(negative + 0.1, positive)
        self.assertEqual(theta, 0.07556377229186596)
        self.assertEqual(0.5 * theta, 0.03778188614593298)
        self.assertLess(negative, theta)
        self.assertLess(theta, positive)
        self.assertGreater(positive - theta, theta - negative)
        self.assertEqual(delta_objective, 838.899732010439)
        self.assertEqual(delta_fp, -838.8872809919176)
        self.assertEqual(delta_prior, 0.01245101794154646)
        self.assertEqual(residual, 5.799165592179634e-10)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(delta_prior - 0.5 * (positive ** 2 - negative ** 2), -1.2663481374630692e-16)
        self.assertEqual(ic + tracer_prior + prior0 - count - fp0 - objective0, 0.0)
        self.assertEqual(ic + tracer_prior + prior1 - count - fp1 - objective1, -9.313225746154785e-10)
        self.assertLess(g_neg, 0.)
        self.assertGreater(g_pos, 0.)
        self.assertGreater(abs(g_pos), abs(g_neg) / 10.)
        self.assertEqual(abs(g_neg) / 10., 17.960957974176385)
        self.assertNotEqual(abs(g_neg) / 10., abs(-179.60957974176327) / 10.)
        self.assertEqual(format(abs(g_neg) / 10., '.3f'), '17.961')
        self.assertEqual(g_neg - (-179.60957974176327), -5.684341886080801e-13)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        derivatives = (
            -62.06240750571353, -25.311584894600227, -33.13682957743554,
            -43.21488298594798, -164.02895850096385, 20.932194608194894,
            62.43204014002461, -2.2909095827464583, g_neg, 4.986584760929821)
        self.assertEqual(min(abs(value) for value in derivatives), 2.2909095827464583)
        self.assertGreaterEqual(min(abs(value) for value in derivatives), 1.)
        self.assertLessEqual(abs(20.932194608194894), abs(362.2367269677485) / 10.)
        self.assertLessEqual(abs(4.986584760929821), abs(-233.42232564563201) / 10.)
        self.assertLessEqual(abs(-2.2909095827464583), 542.8941177183742 / 10.)
        self.assertGreater(abs(62.43204014002461), abs(273.5525091954018) / 10.)
        self.assertEqual(
            pop5_revisit_status(objective0, objective0, abs(g_neg), abs(g_neg), 1., 1.),
            'CONDITIONAL_POP5_REVISIT_NO_IMPROVEMENT')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        with self.assertRaises(ValueError):
            tracer0_secant(negative, g_neg, positive, g_pos)

    def test_full_gradient59_names_population5_and_steps_by_01(self):
        gradient = -179.60957974176327
        white = 0.07451017941546588
        objective = 8208481.070149725
        count = -142313.47804254713
        fp = 5979.479359143987
        step_fp = 5979.479359143986
        ic = 8072142.0269214865
        tracer_prior = 0.8639622377997895
        population = 4.180582598163753
        self.assertEqual(abs(gradient) / 10., 17.960957974176328)
        self.assertEqual(format(abs(gradient) / 10., '.3f'), '17.961')
        self.assertEqual(abs(-238.34390670220037) / 10., 23.83439067022004)
        self.assertGreater(abs(gradient), abs(-238.34390670220037) / 10.)
        self.assertEqual(pop5_line_step(gradient), 0.1)
        self.assertNotEqual(pop5_line_step(gradient), 0.05)
        self.assertEqual(white + 0.1, 0.17451017941546587)
        self.assertEqual(0.5 * white, 0.03725508970773294)
        self.assertEqual(0.5 * (white + 0.1), 0.08725508970773294)
        self.assertEqual(ic + tracer_prior + population - count - fp - objective, 0.0)
        self.assertEqual(fp - step_fp, 9.094947017729282e-13)
        self.assertEqual(gradient - (-179.60957974176344), 1.7053025658242404e-13)
        self.assertEqual(-164.02895850096354 - (-164.02895850096368), 1.4210854715202004e-13)
        self.assertEqual(float(100 * np.exp(0.5 * 0.3259753562964724)), 117.70221865062094)
        derivatives = (
            -62.062407505716166, -25.311584894600266, -33.136829577436046,
            -43.21488298594845, -164.02895850096354, 20.932194608196486,
            62.43204014002643, -2.290909582745435, gradient, 4.9865847609320975)
        self.assertGreaterEqual(min(abs(value) for value in derivatives), 1.)
        self.assertEqual(min(abs(value) for value in derivatives + (1.4590439168912934,)), 1.4590439168912934)
        self.assertLessEqual(abs(20.932194608196486), abs(362.2367269677485) / 10.)
        self.assertLessEqual(abs(4.9865847609320975), abs(-233.42232564563201) / 10.)
        self.assertLessEqual(abs(-2.290909582745435), 542.8941177183742 / 10.)
        self.assertGreater(abs(62.43204014002643), abs(273.5525091954018) / 10.)
        self.assertGreater(abs(gradient), abs(-164.02895850096354))
        self.assertEqual(
            pop5_revisit_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_POP5_REVISIT_NO_IMPROVEMENT')
        self.assertEqual(
            pop5_revisit_status(10., 9., abs(gradient), abs(gradient) / 10., 1., 1.),
            'CONDITIONAL_POP5_REVISIT_REDUCED')
        self.assertEqual(
            pop5_revisit_status(10., 9., abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_POP5_REVISIT_IMPROVED')
        self.assertEqual(coordinate_line_action(True, False, False, False), 'continue')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        self.assertEqual(coordinate_line_action(True, False, True, False), 'stop')
        with self.assertRaises(ValueError):
            pop5_line_step(0.)
        with self.assertRaises(ValueError):
            tracer0_secant(white, gradient, white + 0.1, gradient)

    def test_tracer6_support_step6_improved_queues_one_full_gradient(self):
        white0 = 0.2759753562964724
        white1 = 0.3259753562964724
        gradient0 = -186.55009368073783
        gradient1 = -164.02895850096368
        objective0 = 8208489.846138006
        objective1 = 8208481.070149725
        count0 = -142322.17127641622
        count1 = -142313.47804254713
        fp0 = 5979.381555965149
        fp1 = 5979.479359143986
        tracer0 = 0.8489134699849659
        tracer1 = 0.8639622377997895
        ic = 8072142.0269214865
        population = 4.180582598163753
        delta_objective = objective1 - objective0
        delta_count = count1 - count0
        delta_fp = fp1 - fp0
        delta_tracer = tracer1 - tracer0
        residual = delta_objective - (-delta_count - delta_fp + delta_tracer)
        self.assertEqual(white0 + 0.05, white1)
        self.assertEqual(delta_objective, -8.775988280773163)
        self.assertEqual(residual, -6.549267794753177e-10)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(delta_tracer - 0.5 * (white1 ** 2 - white0 ** 2), 5.551115123125783e-17)
        self.assertLess(abs(delta_tracer - 0.5 * (white1 ** 2 - white0 ** 2)), 1e-12)
        self.assertEqual(ic + tracer0 + population - count0 - fp0 - objective0, 0.0)
        self.assertEqual(ic + tracer1 + population - count1 - fp1 - objective1, 0.0)
        self.assertLess(gradient0, 0.)
        self.assertLess(gradient1, 0.)
        self.assertGreater(abs(gradient1), abs(gradient0) / 10.)
        self.assertEqual(abs(gradient0) / 10., 18.655009368073785)
        self.assertEqual(float(100 * np.exp(0.5 * white1)), 117.70221865062094)
        self.assertGreaterEqual(min(abs(value) for value in (-62.06240750571341, -25.311584894600365, -33.136829577435485, -43.21488298594812, -164.02895850096368, 20.93219460819694, 62.432040140025975, -2.290909582745776, -179.60957974176344, 4.98658476093249)), 1.)
        self.assertGreater(abs(-179.60957974176344), abs(gradient1))
        self.assertLessEqual(abs(20.93219460819694), abs(362.2367269677485) / 10.)
        self.assertLessEqual(abs(4.98658476093249), abs(-233.42232564563201) / 10.)
        self.assertLessEqual(abs(-2.290909582745776), 542.8941177183742 / 10.)
        self.assertGreater(abs(62.432040140025975), 273.5525091954018 / 10.)
        self.assertGreater(abs(-179.60957974176344), 23.834)
        self.assertEqual(
            tracer6_support_step6_status(
                objective0, objective1, abs(gradient0), abs(gradient1), count0 + fp0, count1 + fp1),
            'CONDITIONAL_TRACER6_SUPPORT_STEP6_IMPROVED')
        with self.assertRaises(ValueError):
            tracer6_support_step_size(0.)

    def test_tracer6_revisit7_support_change_keeps_the_005_step(self):
        white = 0.2759753562964724
        gradient = -186.5500936807379
        proposal = 0.3259753562964724
        objective = 8208489.846138006
        self.assertEqual(white + 0.05, proposal)
        self.assertEqual(tracer6_support_step_size(gradient), 0.05)
        self.assertNotEqual(tracer6_support_step_size(gradient), 0.025)
        self.assertGreater(abs(gradient), abs(-186.55009368072817) / 10.)
        self.assertEqual(abs(gradient) / 10., 18.655009368073788)
        self.assertEqual(gradient - (-186.5500936807381), 1.9895196601282805e-13)
        self.assertEqual(5979.381555965149 - 5979.38155596515, -9.094947017729282e-13)
        self.assertEqual(8072142.0269214865 + 0.8489134699849659 + 4.180582598163753 - -142322.17127641622 - 5979.381555965149 - objective, 0.0)
        self.assertEqual(float(100 * np.exp(0.5 * white)), 114.79614051767668)
        self.assertGreaterEqual(min(abs(value) for value in (-60.18071087617793, -4.68132084276926, -12.228800052945918, -39.105541092839985, -186.5500936807379, 17.698540430076154, 64.73533884303359, -4.310166105389108, -178.0139997709454, 6.448524547737966)), 1.)
        self.assertLessEqual(abs(17.698540430076154), abs(362.2367269677485) / 10.)
        self.assertLessEqual(abs(6.448524547737966), abs(-233.42232564563201) / 10.)
        self.assertLessEqual(abs(-4.310166105389108), 542.8941177183742 / 10.)
        self.assertGreater(abs(64.73533884303359), 273.5525091954018 / 10.)
        self.assertGreater(abs(-178.0139997709454), 23.834)
        self.assertEqual(
            tracer6_revisit7_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_TRACER6_REVISIT7_NO_IMPROVEMENT')
        self.assertEqual(
            tracer6_support_step6_status(10., 9., abs(gradient), abs(gradient) / 10., 1., 1.),
            'CONDITIONAL_TRACER6_SUPPORT_STEP6_REDUCED')
        self.assertEqual(
            tracer6_support_step6_status(10., 11., abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_TRACER6_SUPPORT_STEP6_NO_IMPROVEMENT')
        with self.assertRaises(ValueError):
            tracer6_support_step_size(0.)

    def test_full_gradient58_names_tracer6_and_steps_by_005(self):
        gradient = -186.5500936807381
        white = 0.2759753562964724
        self.assertEqual(abs(gradient) / 10., 18.65500936807381)
        self.assertGreater(abs(gradient), abs(-186.55009368072817) / 10.)
        self.assertNotEqual(abs(gradient) / 10., abs(-186.55009368072817) / 10.)
        self.assertEqual(tracer6_support_step_size(gradient), 0.05)
        self.assertNotEqual(tracer6_support_step_size(gradient), 0.025)
        self.assertEqual(float(100 * np.exp(0.5 * white)), 114.79614051767668)
        self.assertEqual(-142322.17127641622 - -142322.17127641616, -5.820766091346741e-11)
        self.assertEqual(8072142.0269214865 + 0.8489134699849659 + 4.180582598163753 - -142322.17127641622 - 5979.38155596515 - 8208489.846138006, 0.0)
        self.assertGreaterEqual(min(abs(value) for value in (-60.18071087617701, -4.6813208427692405, -12.228800052946037, -39.105541092839914, -186.5500936807381, 17.69854043007479, 64.73533884303495, -4.310166105388426, -178.01399977094437, 6.448524547732416)), 1.)
        self.assertLessEqual(abs(17.69854043007479), abs(362.2367269677485) / 10.)
        self.assertLessEqual(abs(6.448524547732416), abs(-233.42232564563201) / 10.)
        self.assertLessEqual(abs(-4.310166105388426), 542.8941177183742 / 10.)
        self.assertGreater(abs(64.73533884303495), 273.5525091954018 / 10.)
        self.assertGreater(abs(-178.01399977094437), 23.834)
        self.assertEqual(
            tracer6_revisit7_status(8208489.846138006, 8208489.846138006, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_TRACER6_REVISIT7_NO_IMPROVEMENT')
        self.assertEqual(
            tracer6_revisit7_status(10., 9., abs(gradient), abs(gradient) / 10., 1., 1.),
            'CONDITIONAL_TRACER6_REVISIT7_REDUCED')
        with self.assertRaises(ValueError):
            tracer6_support_step_size(0.)

    def test_tracer6_support_step5_improved_queues_one_full_gradient(self):
        white0 = 0.22597535629647242
        white1 = 0.2759753562964724
        gradient0 = -206.35780567850628
        gradient1 = -186.55009368072817
        objective0 = 8208499.679400425
        objective1 = 8208489.846138006
        count0 = -142331.92021652748
        count1 = -142322.17127641616
        fp0 = 5979.2846848888175
        fp1 = 5979.38155596515
        tracer0 = 0.8363647021701422
        tracer1 = 0.8489134699849659
        ic = 8072142.0269214865
        population = 4.180582598163753
        delta_objective = objective1 - objective0
        delta_count = count1 - count0
        delta_fp = fp1 - fp0
        delta_tracer = tracer1 - tracer0
        residual = delta_objective - (-delta_count - delta_fp + delta_tracer)
        self.assertEqual(white0 + 0.05, white1)
        self.assertEqual(delta_objective, -9.833262419328094)
        self.assertEqual(residual, 5.02350161468712e-10)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(delta_tracer - 0.5 * (white1 ** 2 - white0 ** 2), -3.469446951953614e-18)
        self.assertLess(abs(delta_tracer - 0.5 * (white1 ** 2 - white0 ** 2)), 1e-12)
        self.assertEqual(ic + tracer0 + population - count0 - fp0 - objective0, 0.0)
        self.assertEqual(ic + tracer1 + population - count1 - fp1 - objective1, 0.0)
        self.assertLess(gradient0, 0.)
        self.assertLess(gradient1, 0.)
        self.assertGreater(abs(gradient1), abs(gradient0) / 10.)
        self.assertEqual(abs(gradient0) / 10., 20.635780567850627)
        self.assertEqual(float(100 * np.exp(0.5 * white1)), 114.79614051767668)
        self.assertGreaterEqual(min(abs(value) for value in (-60.180710876229305, -4.6813208427929105, -12.228800052969259, -39.105541092843865, -186.55009368072817, 17.69854043008775, 64.7353388430337, -4.3101661053878, -178.0139997709459, 6.448524547733004)), 1.)
        self.assertLessEqual(abs(17.69854043008775), abs(362.2367269677485) / 10.)
        self.assertLessEqual(abs(6.448524547733004), abs(-233.42232564563201) / 10.)
        self.assertLessEqual(abs(-4.3101661053878), 542.8941177183742 / 10.)
        self.assertGreater(abs(64.7353388430337), 273.5525091954018 / 10.)
        self.assertGreater(abs(-178.0139997709459), 23.834)
        likelihood0 = count0 + fp0
        likelihood1 = count1 + fp1
        self.assertEqual(
            tracer6_support_step5_status(
                objective0, objective1, abs(gradient0), abs(gradient1), likelihood0, likelihood1),
            'CONDITIONAL_TRACER6_SUPPORT_STEP5_IMPROVED')
        with self.assertRaises(ValueError):
            tracer6_support_step_size(0.)

    def test_tracer6_revisit6_support_change_keeps_the_005_step(self):
        white = 0.22597535629647242
        gradient = -206.3578056785051
        proposal = 0.2759753562964724
        objective = 8208499.679400425
        self.assertEqual(white + 0.05, proposal)
        self.assertEqual(tracer6_support_step_size(gradient), 0.05)
        self.assertNotEqual(tracer6_support_step_size(gradient), 0.025)
        self.assertGreater(abs(gradient), abs(-223.6444195175734) / 10.)
        self.assertEqual(abs(gradient) / 10., 20.63578056785051)
        self.assertEqual(gradient - (-206.35780567850585), 7.673861546209082e-13)
        self.assertEqual(objective, 8208499.679400425)
        self.assertEqual(8072142.0269214865 + 0.8363647021701422 + 4.180582598163753 - -142331.92021652748 - 5979.2846848888175 - objective, 0.0)
        self.assertEqual(float(100 * np.exp(0.5 * white)), 111.96181370948737)
        self.assertGreaterEqual(min(abs(value) for value in (-58.38640719632564, 15.523176897016274, 8.222730202769744, -35.09717124474311, -206.3578056785051, 14.452494734392646, 67.05636075989051, -6.299512002810964, -176.40902005037756, 7.900497708527646)), 1.)
        self.assertLessEqual(abs(14.452494734392646), abs(362.2367269677485) / 10.)
        self.assertLessEqual(abs(7.900497708527646), abs(-233.42232564563201) / 10.)
        self.assertLessEqual(abs(-6.299512002810964), 542.8941177183742 / 10.)
        self.assertGreater(abs(67.05636075989051), 273.5525091954018 / 10.)
        self.assertGreater(abs(-176.40902005037756), 23.834)
        self.assertEqual(
            tracer6_revisit6_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_TRACER6_REVISIT6_NO_IMPROVEMENT')
        self.assertEqual(
            tracer6_support_step5_status(10., 9., abs(gradient), abs(gradient) / 10., 1., 1.),
            'CONDITIONAL_TRACER6_SUPPORT_STEP5_REDUCED')
        self.assertEqual(
            tracer6_support_step5_status(10., 11., abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_TRACER6_SUPPORT_STEP5_NO_IMPROVEMENT')
        with self.assertRaises(ValueError):
            tracer6_support_step_size(0.)

    def test_full_gradient57_names_tracer6_and_steps_by_005(self):
        gradient = -206.35780567850585
        white = 0.22597535629647242
        self.assertEqual(abs(gradient) / 10., 20.635780567850585)
        self.assertGreater(abs(gradient), abs(-223.6444195175734) / 10.)
        self.assertEqual(tracer6_support_step_size(gradient), 0.05)
        self.assertEqual(float(100 * np.exp(0.5 * white)), 111.96181370948737)
        self.assertEqual(8208499.679400425, 8208499.679400425)
        self.assertEqual(5979.2846848888175 - 5979.284684888817, 9.094947017729282e-13)
        self.assertEqual(8072142.0269214865 + 0.8363647021701422 + 4.180582598163753 - -142331.92021652748 - 5979.2846848888175 - 8208499.679400425, 0.0)
        self.assertGreaterEqual(min(abs(value) for value in (-58.386407196321336, 15.52317689701714, 8.222730202770839, -35.097171244742796, -206.35780567850585, 14.45249473439401, 67.05636075989074, -6.299512002810907, -176.40902005037586, 7.900497708524671)), 1.)
        self.assertLessEqual(abs(14.45249473439401), abs(362.2367269677485) / 10.)
        self.assertLessEqual(abs(7.900497708524671), abs(-233.42232564563201) / 10.)
        self.assertLessEqual(abs(-6.299512002810907), 542.8941177183742 / 10.)
        self.assertGreater(abs(67.05636075989074), 273.5525091954018 / 10.)
        self.assertGreater(abs(-176.40902005037586), 23.834)
        self.assertEqual(
            tracer6_revisit6_status(10., 9., abs(gradient), abs(gradient) / 10., 1., 1.),
            'CONDITIONAL_TRACER6_REVISIT6_REDUCED')
        self.assertEqual(
            tracer6_revisit6_status(10., 10., abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_TRACER6_REVISIT6_NO_IMPROVEMENT')
        with self.assertRaises(ValueError):
            tracer6_support_step_size(0.)

    def test_tracer6_support_step4_improved_queues_one_full_gradient(self):
        white0 = 0.17597535629647243
        white1 = 0.22597535629647242
        gradient0 = -223.6444195175734
        gradient1 = -206.35780567850546
        objective0 = 8208510.439566547
        objective1 = 8208499.679400425
        count0 = -142342.59478829207
        count1 = -142331.92021652748
        fp0 = 5979.189041764286
        fp1 = 5979.284684888817
        tracer0 = 0.8263159343553187
        tracer1 = 0.8363647021701422
        ic = 8072142.0269214865
        population = 4.180582598163753
        delta_objective = objective1 - objective0
        delta_count = count1 - count0
        delta_fp = fp1 - fp0
        delta_tracer = tracer1 - tracer0
        residual = delta_objective - (-delta_count - delta_fp + delta_tracer)
        self.assertEqual(white0 + 0.05, white1)
        self.assertEqual(delta_objective, -10.760166121646762)
        self.assertEqual(residual, -3.339835075166775e-10)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(delta_tracer - 0.5 * (white1 ** 2 - white0 ** 2), -6.418476861114186e-17)
        self.assertLess(abs(delta_tracer - 0.5 * (white1 ** 2 - white0 ** 2)), 1e-12)
        self.assertEqual(ic + tracer0 + population - count0 - fp0 - objective0, 0.0)
        self.assertEqual(ic + tracer1 + population - count1 - fp1 - objective1, 0.0)
        self.assertLess(gradient0, 0.)
        self.assertLess(gradient1, 0.)
        self.assertGreater(abs(gradient1), abs(gradient0) / 10.)
        self.assertEqual(float(100 * np.exp(0.5 * white1)), 111.96181370948737)
        self.assertGreaterEqual(min(abs(value) for value in (
            -58.386407196323354, 15.523176897016842, 8.222730202770098,
            -35.09717124474294, gradient1, 14.452494734397648,
            67.05636075989085, -6.299512002810339,
            -176.40902005037654, 7.900497708530869)), 1.)
        self.assertLessEqual(abs(14.452494734397648), abs(362.2367269677485) / 10.)
        self.assertLessEqual(abs(7.900497708530869), abs(-233.42232564563201) / 10.)
        self.assertLessEqual(abs(-6.299512002810339), 542.8941177183742 / 10.)
        self.assertGreater(abs(67.05636075989085), 273.5525091954018 / 10.)
        self.assertGreater(abs(-176.40902005037654), 23.834)
        likelihood0 = count0 + fp0
        likelihood1 = count1 + fp1
        self.assertEqual(
            tracer6_support_step4_status(
                objective0, objective1, abs(gradient0), abs(gradient1), likelihood0, likelihood1),
            'CONDITIONAL_TRACER6_SUPPORT_STEP4_IMPROVED')

    def test_tracer6_revisit5_support_change_keeps_the_005_step(self):
        white = 0.17597535629647243
        gradient = -223.64441951757334
        objective = 8208510.439566547
        reference = 8208510.439566571
        count = -142342.59478829207
        fp = 5979.189041764287
        ic = 8072142.0269214865
        tracer_prior = 0.8263159343553187
        population_prior = 4.180582598163753
        self.assertEqual(white + 0.05, 0.22597535629647242)
        self.assertEqual(tracer6_support_step_size(gradient), 0.05)
        self.assertNotEqual(tracer6_support_step_size(gradient), 0.025)
        self.assertGreater(abs(gradient), 238.5095334855527 / 10.)
        self.assertEqual(abs(gradient) / 10., 22.364441951757335)
        self.assertEqual(objective - reference, -2.421438694000244e-08)
        self.assertEqual(ic - 8072142.026921511, objective - reference)
        self.assertEqual(count - (-142342.5947882921), 2.9103830456733704e-11)
        self.assertEqual(ic + tracer_prior + population_prior - count - fp - objective, 0.0)
        self.assertLess(abs(objective - reference) / abs(reference), 1e-8)
        self.assertGreaterEqual(min(abs(value) for value in (
            -56.675148612104074, 35.29162831006507, 28.20730682622264,
            -31.191689724550102, gradient, 11.19453726704444,
            69.38902776660976, -8.25688540973767,
            -174.79881499945637, 9.339767162872464)), 1.)
        self.assertLessEqual(abs(11.19453726704444), abs(362.2367269677485) / 10.)
        self.assertLessEqual(abs(9.339767162872464), abs(-233.42232564563201) / 10.)
        self.assertLessEqual(abs(-8.25688540973767), 542.8941177183742 / 10.)
        self.assertGreater(abs(69.38902776660976), 273.5525091954018 / 10.)
        self.assertGreater(abs(-174.79881499945637), 23.834)
        self.assertEqual(
            tracer6_revisit5_status(objective, objective, abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_TRACER6_REVISIT5_NO_IMPROVEMENT')
        self.assertEqual(
            tracer6_support_step4_status(10., 9., abs(gradient), abs(gradient) / 10., 1., 1.),
            'CONDITIONAL_TRACER6_SUPPORT_STEP4_REDUCED')
        self.assertEqual(
            tracer6_support_step4_status(10., 11., abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_TRACER6_SUPPORT_STEP4_NO_IMPROVEMENT')
        with self.assertRaises(ValueError):
            tracer6_support_step_size(0.)

    def test_full_gradient56_names_tracer6_and_steps_by_005(self):
        gradient = -223.64441951758337
        white = 0.17597535629647243
        self.assertEqual(gradient, -223.64441951758337)
        self.assertGreater(abs(gradient), 238.5095334855527 / 10.)
        self.assertEqual(abs(gradient) / 10., 22.364441951758337)
        self.assertEqual(tracer6_support_step_size(gradient), 0.05)
        self.assertNotEqual(tracer6_support_step_size(gradient), 0.1)
        self.assertEqual(white, 0.17597535629647243)
        self.assertEqual(float(100 * np.exp(0.5 * white)), 109.1974666795327)
        self.assertEqual(8208510.439566571, 8208510.439566571)
        self.assertEqual(-142342.5947882921 - -142342.59478829207, -2.9103830456733704e-11)
        self.assertEqual(
            8072142.026921511 + 0.8263159343553188 + 4.180582598163753 - -142342.5947882921 - 5979.189041764286 - 8208510.439566571, 0.0)
        self.assertGreaterEqual(abs(11.194537267040348), 1.)
        self.assertLessEqual(abs(11.194537267040348), abs(362.2367269677485) / 10.)
        self.assertGreaterEqual(abs(9.339767162865073), 1.)
        self.assertLessEqual(abs(9.339767162865073), abs(-233.42232564563201) / 10.)
        self.assertGreaterEqual(abs(-8.256885409739375), 1.)
        self.assertLessEqual(abs(-8.256885409739375), 542.8941177183742 / 10.)
        self.assertGreater(abs(69.38902776661021), 273.5525091954018 / 10.)
        self.assertGreater(abs(-174.798814999455), 23.834)
        self.assertEqual(coordinate_line_action(True, False, False, False), 'continue')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        self.assertEqual(
            tracer6_revisit5_status(10., 9., abs(gradient), abs(gradient) / 10., 1., 1.),
            'CONDITIONAL_TRACER6_REVISIT5_REDUCED')
        self.assertEqual(
            tracer6_revisit5_status(10., 9., abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_TRACER6_REVISIT5_IMPROVED')
        self.assertEqual(
            tracer6_revisit5_status(10., 10., abs(gradient), abs(gradient), 1., 1.),
            'CONDITIONAL_TRACER6_REVISIT5_NO_IMPROVEMENT')
        with self.assertRaises(ValueError):
            tracer6_support_step_size(0.)
        with self.assertRaises(ValueError):
            tracer6_support_step_size(9.339767162865073)

    def test_pop13_revisit_secant_reduced_stops_the_coordinate(self):
        accepted, rejected = -0.012419780844287421, 0.08758021915571258
        g_line, g_rejected = -233.42232564563201, 9125.82896064206
        theta = tracer0_secant(rejected, g_rejected, accepted, g_line)
        self.assertEqual(theta, -0.00992575307961334)
        self.assertEqual(0.3 * theta, -0.0029777259238840015)
        drop = 8208510.439566571 - 8208510.718994812
        self.assertEqual(drop, -0.27942824084311724)
        residual = drop - (-0.2794003755734593 + -2.78651910115002e-05)
        self.assertEqual(residual, -7.864642270760669e-11)
        self.assertLess(abs(residual), 1e-9)
        self.assertGreater(abs(9.339767162867238), 1.)
        self.assertLessEqual(abs(9.339767162867238), abs(g_line) / 10.)
        self.assertGreaterEqual(abs(11.19453726704444), 1.)
        self.assertLessEqual(abs(11.19453726704444), abs(362.2367269677485) / 10.)
        self.assertGreater(abs(69.38902776660828), abs(273.5525091954018) / 10.)
        self.assertGreater(abs(-174.79881499945654), 23.834)
        self.assertGreaterEqual(8.25688540973767, 1.)
        initial = -142342.59478829207 + 5978.909641388713
        best = -142342.59478829207 + 5979.189041764286
        self.assertEqual(
            pop13_revisit_status(8208510.718994812, 8208510.439566571,
                                 abs(g_line), abs(9.339767162867238), initial, best),
            'CONDITIONAL_POP13_REVISIT_REDUCED')

    def test_pop13_revisit_secant_uses_the_positive_rejected_end_first(self):
        accepted, rejected = -0.012419780844287421, 0.08758021915571258
        g_accepted, g_rejected = -233.42232564563201, 9125.82896064206
        theta = tracer0_secant(rejected, g_rejected, accepted, g_accepted)
        self.assertEqual(theta, -0.00992575307961334)
        self.assertEqual(0.3 * theta, -0.0029777259238840015)
        self.assertLess(accepted, theta)
        self.assertLess(theta, rejected)
        self.assertLess(abs(theta - accepted), abs(theta - rejected))
        self.assertEqual(abs(g_accepted) / 10., 23.3422325645632)
        self.assertGreater(abs(g_rejected), abs(g_accepted) / 10.)
        rise = 8208964.524410955 - 8208510.718994812
        self.assertEqual(rise, 453.805416142568)
        residual = rise - (453.80165812017367 + 0.0037580219155710637)
        self.assertEqual(residual, 4.787352736457251e-10)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(
            0.0037580219155710637 - 0.5 * (rejected ** 2 - accepted ** 2),
            -1.942890293094024e-16)
        self.assertLess(abs(0.0020606823308089994), 1.)
        self.assertGreater(abs(68.93745432354117), abs(273.5525091954018) / 10.)
        self.assertEqual(
            pop13_revisit_status(8208510.718994812, 8208964.524410955,
                                 abs(g_accepted), abs(g_rejected), 1., 0.),
            'CONDITIONAL_POP13_REVISIT_NO_IMPROVEMENT')
        with self.assertRaises(ValueError):
            tracer0_secant(accepted, g_accepted, rejected, g_rejected)

    def test_pop13_revisit_steps_against_the_negative_derivative(self):
        white = -0.012419780844287421
        gradient = -233.42232564563966
        step = pop13_revisit_step(gradient)
        self.assertEqual(step, 0.1)
        self.assertEqual(white + step, 0.08758021915571258)
        self.assertEqual(0.3 * white, -0.0037259342532862264)
        self.assertEqual(0.3 * (white + step), 0.026274065746713773)
        self.assertGreater(abs(gradient), 1091.1301573851867 / 10.)
        self.assertEqual(abs(gradient) / 10., 23.342232564563965)
        self.assertLess(abs(0.0020606823328553625), 1.)
        self.assertLessEqual(abs(0.0020606823328553625), abs(362.2367269677485) / 10.)
        self.assertGreater(abs(68.93745432353856), abs(273.5525091954018) / 10.)
        self.assertGreater(abs(-174.70479066883163), 23.834)
        self.assertEqual(
            pop13_revisit_status(10., 9., abs(gradient), abs(gradient) / 10., 1., 1.),
            'CONDITIONAL_POP13_REVISIT_REDUCED')
        self.assertEqual(coordinate_line_action(False, True, False, False), 'stop')
        with self.assertRaises(ValueError):
            pop13_revisit_step(0.)

    def test_pop0_revisit10_secant_reduced_stops_the_coordinate(self):
        positive = -0.03571685047779585
        negative = -0.13571685047779586
        theta = tracer0_secant(positive, 362.2367269677417, negative, -326446.38296246435)
        obj0 = 8208510.739070199
        obj1 = 8208510.718994812
        count = -142342.59478829207
        fp0 = 5978.88956203627
        fp1 = 5978.909641388713
        prior0 = 4.180606498333799
        prior1 = 4.180610463354765
        ic = 8072142.026921511
        tracer_prior = 0.8263159343553188
        delta = obj1 - obj0
        dfp = fp1 - fp0
        dprior = prior1 - prior0
        residual = delta - (-dfp + dprior)
        g_saved = 0.0020606823469525304
        self.assertEqual(theta, -0.03582769110596287)
        self.assertEqual(0.3 + theta, 0.2641723088940371)
        self.assertLess(negative, theta)
        self.assertLess(theta, positive)
        self.assertEqual(delta, -0.020075387321412563)
        self.assertEqual(residual, 1.0035705599875655e-10)
        self.assertLess(abs(residual), 1e-9)
        self.assertEqual(
            dprior - 0.5 * (theta ** 2 - positive ** 2), 2.4665599424045226e-16)
        self.assertEqual(ic + tracer_prior + prior0 - count - fp0 - obj0, 9.313225746154785e-10)
        self.assertEqual(ic + tracer_prior + prior1 - count - fp1 - obj1, 0.)
        self.assertLess(abs(g_saved), 1.)
        self.assertLessEqual(abs(g_saved), abs(362.2367269677485) / 10.)
        self.assertEqual(abs(362.2367269677485) / 10., 36.22367269677485)
        self.assertGreater(abs(68.9374543235389), abs(273.5525091954018) / 10.)
        self.assertGreater(abs(-174.70479066883175), 23.834)
        self.assertEqual(
            pop0_revisit10_status(
                obj0, obj1, abs(362.2367269677485), abs(g_saved), count + fp0, count + fp1),
            'CONDITIONAL_POP0_REVISIT10_REDUCED')

    def test_pop4_line_improved_before_the_sign_change(self):
        initial = -142420.23671111898 + 5253.047414546184
        accepted = -142420.23671111898 + 5310.71470520702
        self.assertEqual(
            pop4_line_status(8209312.8085845355, 8209255.160704653,
                             971.9560505156032, 193.59921272852307,
                             initial, accepted),
            'CONDITIONAL_POP4_LINE_IMPROVED')

    def test_pop4_secant_stays_inside_the_sign_bracket(self):
        positive, negative = -0.24410777812331236, -0.3441077781233124
        theta = tracer0_secant(positive, 193.59921272852307, negative, -507.3889418600625)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - positive), abs(theta - negative))
        self.assertAlmostEqual(theta, -0.2717258215183748)

    def test_pop4_secant_meets_the_accepted_end_gate(self):
        initial = -142420.236711119 + 5310.71470520702
        best = -142420.236711119 + 5313.276050871325
        self.assertEqual(
            pop4_line_status(8209255.160704653, 8209252.606482146,
                             193.59921272852313, 7.621834032470206,
                             initial, best),
            'CONDITIONAL_POP4_LINE_REDUCED')

    def test_tracer0_revisit2_secant_stays_inside_the_sign_bracket(self):
        positive, negative = 0.2760982361105669, 0.17609823611056688
        theta = tracer0_secant(positive, 895.9700267980062, negative, -16349.668105529856)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - positive), abs(theta - negative))
        self.assertAlmostEqual(theta, 0.2709028932654769)

    def test_tracer3_line_gate_uses_count_plus_fp(self):
        self.assertEqual(tracer3_line_status(10., 9., 598.735, 59.873, 3., 3.1),
                         'CONDITIONAL_TRACER3_LINE_REDUCED')
        self.assertEqual(tracer3_line_status(10., 9., 598.735, 59.874, 3., 3.1),
                         'CONDITIONAL_TRACER3_LINE_IMPROVED')
        self.assertEqual(tracer3_line_status(10., 9., 598.735, 59.873, 3., 2.),
                         'CONDITIONAL_TRACER3_LINE_IMPROVED')
        self.assertEqual(tracer3_line_status(10., 10., 598.735, 59.873, 3., 4.),
                         'CONDITIONAL_TRACER3_LINE_NO_IMPROVEMENT')

    def test_tracer3_line_secant_stays_inside_the_sign_bracket(self):
        positive, negative = 0.5732460450693487, 0.47324604506934875
        theta = tracer0_secant(positive, 598.7345278345143, negative, -732.9181370449785)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - positive), abs(theta - negative))
        self.assertAlmostEqual(theta, 0.5282842811180337)

    def test_tracer3_secant_meets_the_tenfold_gate(self):
        initial = -142436.3914841816 + 4810.707946311214
        best = -142423.65173731328 + 4810.587073462019
        self.assertEqual(
            tracer3_line_status(8209771.096772627, 8209758.453135235,
                                598.7345278345147, 28.544918313510692,
                                initial, best),
            'CONDITIONAL_TRACER3_LINE_REDUCED')

    def test_tracer4_line_gate_uses_count_plus_fp(self):
        self.assertEqual(tracer4_line_status(10., 9., 682.934, 68.293, 3., 3.1),
                         'CONDITIONAL_TRACER4_LINE_REDUCED')
        self.assertEqual(tracer4_line_status(10., 9., 682.934, 68.294, 3., 3.1),
                         'CONDITIONAL_TRACER4_LINE_IMPROVED')
        self.assertEqual(tracer4_line_status(10., 9., 682.934, 68.293, 3., 2.),
                         'CONDITIONAL_TRACER4_LINE_IMPROVED')
        self.assertEqual(tracer4_line_status(10., 10., 682.934, 68.293, 3., 4.),
                         'CONDITIONAL_TRACER4_LINE_NO_IMPROVEMENT')

    def test_population3_revisit3_gate_uses_count_plus_fp(self):
        self.assertEqual(pop3_revisit3_status(10., 9., 852.294, 85.229, 3., 3.1),
                         'CONDITIONAL_POP3_REVISIT3_REDUCED')
        self.assertEqual(pop3_revisit3_status(10., 9., 852.294, 85.230, 3., 3.1),
                         'CONDITIONAL_POP3_REVISIT3_IMPROVED')
        self.assertEqual(pop3_revisit3_status(10., 9., 852.294, 85.229, 3., 2.),
                         'CONDITIONAL_POP3_REVISIT3_IMPROVED')
        self.assertEqual(pop3_revisit3_status(10., 10., 852.294, 85.229, 3., 4.),
                         'CONDITIONAL_POP3_REVISIT3_NO_IMPROVEMENT')


    def test_pop3_revisit4_gate_uses_count_plus_fp(self):
        self.assertEqual(
            pop3_revisit4_status(10., 9., 620.8935209944585, 62.0893, 3., 3.1),
            'CONDITIONAL_POP3_REVISIT4_REDUCED')
        self.assertEqual(
            pop3_revisit4_status(10., 9., 620.8935209944585, 62.0894, 3., 3.1),
            'CONDITIONAL_POP3_REVISIT4_IMPROVED')
        self.assertEqual(
            pop3_revisit4_status(10., 9., 620.8935209944585, 62.0893, 3., 2.),
            'CONDITIONAL_POP3_REVISIT4_IMPROVED')
        self.assertEqual(
            pop3_revisit4_status(10., 10., 620.8935209944585, 62.0893, 3., 4.),
            'CONDITIONAL_POP3_REVISIT4_NO_IMPROVEMENT')


    def test_pop3_revisit4_rejected_step_does_not_improve(self):
        self.assertEqual(
            pop3_revisit4_status(8208651.510122626, 8211045.417978643,
                                 620.8935209944603, 47989.92103017981, 1., 0.),
            'CONDITIONAL_POP3_REVISIT4_NO_IMPROVEMENT')

    def test_pop3_revisit4_secant_stays_inside_the_sign_bracket(self):
        positive, negative = -0.3110402418528304, -0.41104024185283045
        theta = tracer0_secant(positive, 47989.92103017981, negative, -620.8935209944603)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))
        self.assertAlmostEqual(theta, -0.4097629673888512)
        with self.assertRaises(ValueError):
            tracer0_secant(negative, -620.8935209944603, positive, 47989.92103017981)


    def test_pop3_revisit4_secant_meets_the_accepted_end_gate(self):
        count = -142420.23671111895
        self.assertEqual(
            pop3_revisit4_status(
                8208651.510122626, 8208651.1227584565,
                620.8935209944598, 14.3287979345126,
                count + 5915.370948653073, count + 5915.757788626992),
            'CONDITIONAL_POP3_REVISIT4_REDUCED')

    def test_population3_revisit3_secant_stays_inside_the_sign_bracket(self):
        negative, positive = -0.41357431852137316, -0.3135743185213732
        theta = tracer0_secant(positive, 32781.02780396851, negative, -852.2941648681355)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))
        self.assertAlmostEqual(theta, -0.41104024185283045)

    def test_population14_revisit5_gate_uses_count_plus_fp(self):
        self.assertEqual(pop14_revisit5_status(10., 9., 889.528, 88.952, 3., 3.1),
                         'CONDITIONAL_POP14_REVISIT5_REDUCED')
        self.assertEqual(pop14_revisit5_status(10., 9., 889.528, 88.953, 3., 3.1),
                         'CONDITIONAL_POP14_REVISIT5_IMPROVED')
        self.assertEqual(pop14_revisit5_status(10., 9., 889.528, 88.952, 3., 2.),
                         'CONDITIONAL_POP14_REVISIT5_IMPROVED')
        self.assertEqual(pop14_revisit5_status(10., 10., 889.528, 88.952, 3., 4.),
                         'CONDITIONAL_POP14_REVISIT5_NO_IMPROVEMENT')


    def test_pop14_revisit6_gate_uses_count_plus_fp(self):
        self.assertEqual(
            pop14_revisit6_status(10., 9., 484.7936133463083, 48.4793, 3., 3.1),
            'CONDITIONAL_POP14_REVISIT6_REDUCED')
        self.assertEqual(
            pop14_revisit6_status(10., 9., 484.7936133463083, 48.4794, 3., 3.1),
            'CONDITIONAL_POP14_REVISIT6_IMPROVED')
        self.assertEqual(
            pop14_revisit6_status(10., 9., 484.7936133463083, 48.4793, 3., 2.),
            'CONDITIONAL_POP14_REVISIT6_IMPROVED')
        self.assertEqual(
            pop14_revisit6_status(10., 10., 484.7936133463083, 48.4793, 3., 4.),
            'CONDITIONAL_POP14_REVISIT6_NO_IMPROVEMENT')


    def test_pop14_revisit6_two_steps_meet_the_gate(self):
        initial = -142420.236711119 + 5860.422459658225
        best = -142420.236711119 + 5915.370948653073
        self.assertEqual(
            pop14_revisit6_status(8208706.0508437585, 8208651.510122626,
                                  484.79361334630585, 30.663013792200143,
                                  initial, best),
            'CONDITIONAL_POP14_REVISIT6_REDUCED')

    def test_population14_revisit4_gate_uses_count_plus_fp(self):
        self.assertEqual(pop14_revisit4_status(10., 9., 1096., 109., 3., 3.1),
                         'CONDITIONAL_POP14_REVISIT4_REDUCED')
        self.assertEqual(pop14_revisit4_status(10., 9., 1096., 110., 3., 3.1),
                         'CONDITIONAL_POP14_REVISIT4_IMPROVED')
        self.assertEqual(pop14_revisit4_status(10., 9., 1096., 109., 3., 2.),
                         'CONDITIONAL_POP14_REVISIT4_IMPROVED')
        self.assertEqual(pop14_revisit4_status(10., 10., 1096., 109., 3., 4.),
                         'CONDITIONAL_POP14_REVISIT4_NO_IMPROVEMENT')

    def test_population14_revisit3_gate_uses_count_plus_fp(self):
        self.assertEqual(pop14_revisit3_status(10., 9., 1239., 123., 3., 3.1),
                         'CONDITIONAL_POP14_REVISIT3_REDUCED')
        self.assertEqual(pop14_revisit3_status(10., 9., 1239., 124., 3., 3.1),
                         'CONDITIONAL_POP14_REVISIT3_IMPROVED')
        self.assertEqual(pop14_revisit3_status(10., 9., 1239., 123., 3., 2.),
                         'CONDITIONAL_POP14_REVISIT3_IMPROVED')
        self.assertEqual(pop14_revisit3_status(10., 10., 1239., 123., 3., 4.),
                         'CONDITIONAL_POP14_REVISIT3_NO_IMPROVEMENT')

    def test_population0_revisit2_gate_uses_count_plus_fp(self):
        self.assertEqual(pop0_revisit2_status(10., 9., 2065., 206., 3., 3.1),
                         'CONDITIONAL_POP0_REVISIT2_REDUCED')
        self.assertEqual(pop0_revisit2_status(10., 9., 2065., 207., 3., 3.1),
                         'CONDITIONAL_POP0_REVISIT2_IMPROVED')
        self.assertEqual(pop0_revisit2_status(10., 9., 2065., 206., 3., 2.),
                         'CONDITIONAL_POP0_REVISIT2_IMPROVED')
        self.assertEqual(pop0_revisit2_status(10., 10., 2065., 206., 3., 4.),
                         'CONDITIONAL_POP0_REVISIT2_NO_IMPROVEMENT')

    def test_population0_revisit3_gate_uses_count_plus_fp(self):
        self.assertEqual(pop0_revisit3_status(10., 9., 2903., 290., 3., 3.1),
                         'CONDITIONAL_POP0_REVISIT3_REDUCED')
        self.assertEqual(pop0_revisit3_status(10., 9., 2903., 291., 3., 3.1),
                         'CONDITIONAL_POP0_REVISIT3_IMPROVED')
        self.assertEqual(pop0_revisit3_status(10., 9., 2903., 290., 3., 2.),
                         'CONDITIONAL_POP0_REVISIT3_IMPROVED')
        self.assertEqual(pop0_revisit3_status(10., 10., 2903., 290., 3., 4.),
                         'CONDITIONAL_POP0_REVISIT3_NO_IMPROVEMENT')

    def test_population0_revisit3_secant_stays_inside_the_sign_bracket(self):
        negative, positive = -0.03918101393817476, 0.060818986061825245
        theta = tracer0_secant(positive, 124032.62236529504, negative, -2903.510462671655)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))
        self.assertAlmostEqual(theta, -0.03689363492481951)

    def test_population6_line_gate_uses_count_plus_fp(self):
        self.assertEqual(pop6_line_status(10., 9., 1419., 141., 3., 3.1),
                         'CONDITIONAL_POP6_LINE_REDUCED')
        self.assertEqual(pop6_line_status(10., 9., 1419., 142., 3., 3.1),
                         'CONDITIONAL_POP6_LINE_IMPROVED')
        self.assertEqual(pop6_line_status(10., 9., 1419., 141., 3., 2.),
                         'CONDITIONAL_POP6_LINE_IMPROVED')
        self.assertEqual(pop6_line_status(10., 10., 1419., 141., 3., 4.),
                         'CONDITIONAL_POP6_LINE_NO_IMPROVEMENT')

    def test_population6_line_secant_stays_inside_the_sign_bracket(self):
        positive, negative = 0.024305908871722418, -0.07569409112827759
        theta = tracer0_secant(positive, 1419.1104521292946, negative, -7674.834610795968)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - positive), abs(theta - negative))
        self.assertAlmostEqual(theta, 0.008700905297258255)

    def test_population0_revisit2_secant_stays_inside_the_sign_bracket(self):
        negative, positive = -0.043790935948505985, 0.05620906405149402
        theta = tracer0_secant(positive, 42742.355172397256, negative, -2065.612357544863)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))
        self.assertAlmostEqual(theta, -0.03918101393817476)

    def test_population3_revisit2_gate_uses_count_plus_fp(self):
        self.assertEqual(pop3_revisit2_status(10., 9., 1863., 186., 3., 3.1),
                         'CONDITIONAL_POP3_REVISIT2_REDUCED')
        self.assertEqual(pop3_revisit2_status(10., 9., 1863., 187., 3., 3.1),
                         'CONDITIONAL_POP3_REVISIT2_IMPROVED')
        self.assertEqual(pop3_revisit2_status(10., 9., 1863., 186., 3., 2.),
                         'CONDITIONAL_POP3_REVISIT2_IMPROVED')
        self.assertEqual(pop3_revisit2_status(10., 10., 1863., 186., 3., 4.),
                         'CONDITIONAL_POP3_REVISIT2_NO_IMPROVEMENT')

    def test_population3_revisit2_secant_stays_inside_the_sign_bracket(self):
        negative, positive = -0.42322182348159887, -0.3232218234815989
        theta = tracer0_secant(positive, 17448.409965125433, negative, -1863.0766268546038)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))
        self.assertAlmostEqual(theta, -0.41357431852137316)

    def test_population3_revisit_gate_uses_count_plus_fp(self):
        self.assertEqual(pop3_revisit_status(10., 9., 1580., 158., 3., 3.1),
                         'CONDITIONAL_POP3_REVISIT_REDUCED')
        self.assertEqual(pop3_revisit_status(10., 9., 1580., 159., 3., 3.1),
                         'CONDITIONAL_POP3_REVISIT_IMPROVED')
        self.assertEqual(pop3_revisit_status(10., 9., 1580., 158., 3., 2.),
                         'CONDITIONAL_POP3_REVISIT_IMPROVED')
        self.assertEqual(pop3_revisit_status(10., 10., 1580., 158., 3., 4.),
                         'CONDITIONAL_POP3_REVISIT_NO_IMPROVEMENT')

    def test_population3_revisit_secant_stays_inside_the_sign_bracket(self):
        negative, positive = -0.44625950123612534, -0.34625950123612537
        theta = tracer0_secant(positive, 5280.034350755502, negative, -1580.5101295873344)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))
        self.assertAlmostEqual(theta, -0.42322182348159887)

    def test_population0_revisit_secant_stays_inside_the_sign_bracket(self):
        negative, positive = -0.04765670069317916, 0.05234329930682084
        theta = tracer0_secant(positive, 43077.1754455704, negative, -1732.226045126417)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))
        self.assertAlmostEqual(theta, -0.043790935948505985)

    def test_population14_revisit2_gate_uses_count_plus_fp(self):
        self.assertEqual(pop14_revisit2_status(10., 9., 1286., 128., 3., 3.1),
                         'CONDITIONAL_POP14_REVISIT2_REDUCED')
        self.assertEqual(pop14_revisit2_status(10., 9., 1286., 130., 3., 3.1),
                         'CONDITIONAL_POP14_REVISIT2_IMPROVED')
        self.assertEqual(pop14_revisit2_status(10., 9., 1286., 128., 3., 2.),
                         'CONDITIONAL_POP14_REVISIT2_IMPROVED')
        self.assertEqual(pop14_revisit2_status(10., 10., 1286., 128., 3., 4.),
                         'CONDITIONAL_POP14_REVISIT2_NO_IMPROVEMENT')

    def test_population9_revisit_gate_uses_count_plus_fp(self):
        self.assertEqual(pop9_revisit_status(10., 9., 1152., 115., 3., 3.1),
                         'CONDITIONAL_POP9_REVISIT_REDUCED')
        self.assertEqual(pop9_revisit_status(10., 9., 1152., 120., 3., 3.1),
                         'CONDITIONAL_POP9_REVISIT_IMPROVED')
        self.assertEqual(pop9_revisit_status(10., 9., 1152., 115., 3., 2.),
                         'CONDITIONAL_POP9_REVISIT_IMPROVED')
        self.assertEqual(pop9_revisit_status(10., 10., 1152., 115., 3., 4.),
                         'CONDITIONAL_POP9_REVISIT_NO_IMPROVEMENT')

    def test_population14_revisit_gate_uses_count_plus_fp(self):
        self.assertEqual(pop14_revisit_status(10., 9., 1243., 120., 3., 3.1),
                         'CONDITIONAL_POP14_REVISIT_REDUCED')
        self.assertEqual(pop14_revisit_status(10., 9., 1243., 130., 3., 3.1),
                         'CONDITIONAL_POP14_REVISIT_IMPROVED')
        self.assertEqual(pop14_revisit_status(10., 9., 1243., 120., 3., 2.),
                         'CONDITIONAL_POP14_REVISIT_IMPROVED')
        self.assertEqual(pop14_revisit_status(10., 10., 1243., 120., 3., 4.),
                         'CONDITIONAL_POP14_REVISIT_NO_IMPROVEMENT')

    def test_population12_revisit_gate_uses_count_plus_fp(self):
        self.assertEqual(pop12_revisit_status(10., 9., 1006., 100., 3., 3.1),
                         'CONDITIONAL_POP12_REVISIT_REDUCED')
        self.assertEqual(pop12_revisit_status(10., 9., 1006., 110., 3., 3.1),
                         'CONDITIONAL_POP12_REVISIT_IMPROVED')
        self.assertEqual(pop12_revisit_status(10., 9., 1006., 100., 3., 2.),
                         'CONDITIONAL_POP12_REVISIT_IMPROVED')
        self.assertEqual(pop12_revisit_status(10., 10., 1006., 100., 3., 4.),
                         'CONDITIONAL_POP12_REVISIT_NO_IMPROVEMENT')

    def test_population12_revisit2_gate_uses_count_plus_fp(self):
        self.assertEqual(pop12_revisit2_status(10., 9., 659.664, 65.966, 3., 3.1),
                         'CONDITIONAL_POP12_REVISIT2_REDUCED')
        self.assertEqual(pop12_revisit2_status(10., 9., 659.664, 65.967, 3., 3.1),
                         'CONDITIONAL_POP12_REVISIT2_IMPROVED')
        self.assertEqual(pop12_revisit2_status(10., 9., 659.664, 65.966, 3., 2.),
                         'CONDITIONAL_POP12_REVISIT2_IMPROVED')
        self.assertEqual(pop12_revisit2_status(10., 10., 659.664, 65.966, 3., 4.),
                         'CONDITIONAL_POP12_REVISIT2_NO_IMPROVEMENT')

    def test_population12_revisit2_secant_stays_inside_the_sign_bracket(self):
        positive, negative = -0.5877651208441997, -0.6877651208441997
        theta = tracer0_secant(positive, 6455.915574916689, negative, -659.6640246738126)
        self.assertGreater(theta, negative)
        self.assertLess(theta, positive)
        self.assertLess(abs(theta - negative), abs(theta - positive))
        self.assertAlmostEqual(theta, -0.6784944210306276)

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
