import unittest
from cf4_r2_chain_schedule import ChainSchedule


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


if __name__=='__main__':unittest.main()
