import unittest

import numpy as np

from cf4_r2_conditional_map import assemble_conditional_objective, conditional_target_terms
from cf4_r2_conditional_optimizer_diagnostic import (
    block_gradient_summary, diagnostic_status, finite_difference_agreement,
    finite_difference_status, gradient_agreement, nuisance_block_status, nuisance_scale,
    coordinate_line_action, full_gradient_record_status, pop9_line_status, pop9_newton_step,
    tracer2_line_status, tracer3_line_status, tracer4_line_status, tracer4_revisit_status, tracer_pair_newton_step,
    tracer_pair_status,
    pop3_line_status, pop0_line_status, pop_pair_status, pop2_line_status,
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
    tracer6_support_step2_status, tracer6_revisit4_status,
    tracer6_support_step3_status,
    pop5_line_step, pop5_line_status,
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
