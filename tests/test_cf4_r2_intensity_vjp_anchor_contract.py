"""Metadata contract for the bounded R2 intensity-VJP feasibility gate."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cf4_r2_intensity_vjp_feasibility as vjp


class AnchorResultContractTests(unittest.TestCase):
    def setUp(self):
        self.result = dict(
            status="CONDITIONAL_HMC_TRANSPORT_DIAGNOSTIC_NOT_POSTERIOR",
            source_commit=vjp.ANCHOR_SOURCE_COMMIT,
            N=256,
            box_cMpc_h=384.0,
            target_rules=vjp.ANCHOR_TARGET_RULES,
            target_orders=dict(value=2, gradient=2, matched_primal=True),
            checkpoint_replay_passed=True,
            conditional_count_rows=47121,
            count_training_keys=37951,
            heldout_count_values_loaded=0,
            heldout_FP_marks_loaded=0,
            R2_complete=False,
            posterior_sample=False,
            posterior_uncertainty=False,
        )

    def test_accepts_recorded_anchor_contract_without_unwritten_field(self):
        self.assertTrue(vjp._anchor_result_matches_contract(self.result))
        self.assertNotIn("likelihood_changed", self.result)

    def test_rejects_target_or_holdout_contract_mismatch(self):
        for key, value in (("target_rules", "changed target"),
                           ("heldout_count_values_loaded", 1),
                           ("checkpoint_replay_passed", False)):
            with self.subTest(key=key):
                changed = dict(self.result, **{key: value})
                self.assertFalse(vjp._anchor_result_matches_contract(changed))


if __name__ == "__main__":
    unittest.main()
