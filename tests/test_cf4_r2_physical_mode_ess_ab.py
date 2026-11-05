"""Pure mechanics regression for the bounded physical-mode A/B runner."""

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cf4_r2_physical_mode_ess_ab import _run_arm  # noqa: E402


def one_coordinate_block():
    def project(vector):
        result = np.zeros_like(vector)
        result[0] = vector[0]
        return result

    def draw_auxiliary(rng):
        result = np.zeros(2, dtype=np.float64)
        result[0] = rng.standard_normal()
        return result

    project.dimension = 1
    project.block_name = "one_coordinate"
    project.draw_auxiliary = draw_auxiliary
    return project


class PhysicalModeESSABTests(unittest.TestCase):
    def test_arm_records_active_name_and_saves_before_transition(self):
        block = one_coordinate_block()
        objective = type("Objective", (), {"records": [{}]})()
        report = {}
        saved_active_arms = []

        def save():
            saved_active_arms.append(report.get("active_arm"))

        def evaluate(_position):
            objective.records.append({"evaluation": len(objective.records) + 1})
            return 0.0

        result = _run_arm(
            "unit_arm", (block,), (1,), 7, np.array([0.2, -0.3]), 0.0,
            (block,), evaluate, objective, report, save)

        self.assertEqual(saved_active_arms, ["unit_arm"])
        self.assertEqual(result["arm"], "unit_arm")
        self.assertTrue(result["complete_fixed_sweep"])
        self.assertEqual(result["proposal_target_evaluations"], 1)
        self.assertEqual(objective.records[-1]["evaluation"], 2)


if __name__ == "__main__":
    unittest.main()
