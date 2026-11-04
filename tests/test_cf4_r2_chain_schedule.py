import math
import unittest
from cf4_r2_chain_schedule import ChainSchedule, review_warmup


class ScheduleTests(unittest.TestCase):
    def test_warmup_then_frozen_production(self):
        schedule=ChainSchedule()
        step=schedule.initial_step
        for i in range(schedule.warmup):
            self.assertEqual(schedule.steps(i),4 if i<schedule.warmup//2 else 8)
            step=schedule.next_step(step,i,-10. if i%2 else 0.)
        frozen=step
        for i in range(schedule.warmup,schedule.proposals):
            self.assertEqual(schedule.steps(i),8)
            step=schedule.next_step(step,i,-100. if i%2 else 0.)
            self.assertEqual(step,frozen)

    def test_invalid_and_bounded_settings(self):
        with self.assertRaises(ValueError):ChainSchedule(proposals=12,warmup=12)
        with self.assertRaises(ValueError):ChainSchedule(production_steps=0)
        schedule=ChainSchedule()
        self.assertEqual(schedule.next_step(.08,0,0.),.08)

    def test_warmup_update_uses_declared_diminishing_log_step_rule(self):
        schedule=ChainSchedule()
        step=.04
        log_acceptance=-.2
        expected=step*math.exp(
            (math.exp(log_acceptance)-schedule.target_acceptance))
        self.assertAlmostEqual(schedule.next_step(step,0,log_acceptance),expected)
        self.assertEqual(schedule.next_step(step,schedule.warmup,log_acceptance),step)
        contract=schedule.adaptation_contract()
        self.assertEqual(contract['method'],'diminishing Robbins-Monro on log step size')
        self.assertEqual(contract['metric'],'fixed throughout warmup and retained sampling')

    def test_invalid_log_acceptance_fails_closed(self):
        schedule=ChainSchedule()
        with self.assertRaises(ValueError):schedule.next_step(.04,0,float('nan'))
        with self.assertRaises(ValueError):schedule.next_step(.04,0,.01)

    def test_warmup_review_allows_retention_screen_not_equilibration(self):
        rows=[dict(log_acceptance=-.2,accepted=True,step_size=.04),
              dict(log_acceptance=-2.,accepted=False,step_size=.05)]
        result=review_warmup(rows,expected_transitions=2,final_step=.05,
            minimum_step=1e-6,maximum_step=.08)
        self.assertEqual(result['status'],'READY_FOR_RETAINED_DIAGNOSTIC_NOT_EQUILIBRATION')
        self.assertEqual(result['accepted'],1)
        self.assertFalse(result['posterior_equilibration_assessed'])

    def test_warmup_review_stops_when_no_transition_was_accepted(self):
        rows=[dict(log_acceptance=-float('inf'),accepted=False,step_size=.04)
              for _ in range(2)]
        result=review_warmup(rows,expected_transitions=2,final_step=.04)
        self.assertEqual(result['status'],'STOP_ZERO_ACCEPTED_TRANSITIONS')
        self.assertEqual(result['acceptance_probability_mean'],0.)

    def test_warmup_review_requires_complete_valid_window(self):
        row=dict(log_acceptance=-.1,accepted=True,step_size=.04)
        with self.assertRaises(ValueError):
            review_warmup([row],expected_transitions=2,final_step=.04)
        with self.assertRaises(ValueError):
            review_warmup([dict(row,log_acceptance=float('nan'))],
                expected_transitions=1,final_step=.04)


if __name__=='__main__':unittest.main()
