import unittest
from unittest.mock import patch

import numpy as np

from cf4_r2_retained_field_mcse import (RetainedFieldMCSEDiagnostics,
    validate_field_mcse_schedule)


def field(value, *, empty=False):
    rho = np.full((2, 2, 2), 0. if empty else value, dtype=np.float64)
    velocity = np.full((3, 2, 2, 2), value, dtype=np.float64)
    if empty:
        velocity.fill(0.)
    return dict(rho=rho, mean_velocity_km_s=velocity,
        velocity_valid=rho > 0)


def diagnostics(keys=None):
    if keys is None:
        # The first coarse cell in each of two populations.
        keys = np.asarray([0, 15], dtype=np.int64)
    return RetainedFieldMCSEDiagnostics((2, 2, 2), train_keys=keys,
        batch_sizes=(2, 4), minimum_batches=2,
        count_grid_size=2, population_count=2)


class RetainedFieldMCSEDiagnosticsTests(unittest.TestCase):
    def test_schedule_requires_explicit_complete_scales_without_choosing_them(self):
        contract = validate_field_mcse_schedule((4, 6), 4, 36)
        self.assertEqual(contract, dict(batch_sizes=(4, 6),
            minimum_batches=4, retained_capacity=36))
        with self.assertRaisesRegex(ValueError, "exactly two distinct"):
            validate_field_mcse_schedule((4, 4), 4, 36)
        with self.assertRaisesRegex(ValueError, "complete batch"):
            validate_field_mcse_schedule((5, 6), 4, 36)
        with self.assertRaisesRegex(ValueError, "minimum batches"):
            validate_field_mcse_schedule((4, 6), 7, 36)

    def test_rejected_transitions_count_retained_repeats_and_report_domains(self):
        acc = diagnostics()
        retained = [field(1.), field(1.), field(2.), field(2.),
                    field(3.), field(3.), field(4.), field(4.)]
        accepted = [True, False, True, False, True, False, True, False]
        for transition, (item, did_accept) in enumerate(zip(retained, accepted), 1):
            acc.update(item, transition=transition, accepted=did_accept)

        self.assertEqual(acc.retained_states, 8)
        self.assertEqual(acc.accepted_transitions, 4)
        self.assertEqual(acc.rejected_transitions, 4)
        self.assertEqual(acc.estimated_accumulator_bytes,
            2 * 37 * 2 ** 3 * np.dtype(np.float64).itemsize)
        result = acc.finalize_scale(4)
        self.assertEqual(result["status"], "FIELD_MCSE_AVAILABLE_NOT_CONVERGENCE_EVIDENCE")
        density = result["regional_summary"]["density_mean_mcse_over_sd"]
        self.assertEqual(density["full_box"]["region_cells"], 8)
        self.assertEqual(density["training_count_cells"]["region_cells"], 2)
        self.assertEqual(result["retained_states"], 8)

    def test_checkpoint_resume_is_exact_and_training_identity_is_enforced(self):
        states = [field(value) for value in (1., 1., 2., 2., 3., 3., 4., 4.)]
        uninterrupted = diagnostics()
        for transition, item in enumerate(states, 1):
            uninterrupted.update(item, transition=transition,
                accepted=(transition % 2 == 1))

        partial = diagnostics()
        for transition, item in enumerate(states[:5], 1):
            partial.update(item, transition=transition,
                accepted=(transition % 2 == 1))
        checkpoint = {key: np.array(value, copy=True)
            for key, value in partial.checkpoint_arrays().items()}
        with patch.object(RetainedFieldMCSEDiagnostics, "__init__",
                          side_effect=AssertionError("restore must not reallocate")), \
             patch("cf4_r2_field_batch_mcse.MultiScaleFieldMCSEAccumulator.__init__",
                   side_effect=AssertionError("restore must not reallocate")), \
             patch("cf4_r2_field_batch_mcse.FieldBatchMCSEAccumulator.__init__",
                   side_effect=AssertionError("restore must not reallocate")):
            resumed = RetainedFieldMCSEDiagnostics.from_checkpoint(
                checkpoint, train_keys=np.asarray([0, 15], dtype=np.int64))
        for transition, item in enumerate(states[5:], 6):
            resumed.update(item, transition=transition,
                accepted=(transition % 2 == 1))

        self.assertEqual(resumed.summary(), uninterrupted.summary())
        for size in (2, 4):
            expected = uninterrupted.finalize_scale(size)
            actual = resumed.finalize_scale(size)
            for name in ("density_mean", "density_posterior_sd",
                         "density_mean_mcse", "velocity_conditional_mean",
                         "velocity_conditional_posterior_sd",
                         "velocity_occupancy_fraction"):
                np.testing.assert_array_equal(actual[name], expected[name])
            self.assertEqual(actual["regional_summary"], expected["regional_summary"])

        with self.assertRaisesRegex(ValueError, "training-key identity"):
            RetainedFieldMCSEDiagnostics.from_checkpoint(
                checkpoint, train_keys=np.asarray([1, 15], dtype=np.int64))

    def test_invalid_transition_or_field_does_not_advance_chain_diagnostics(self):
        acc = diagnostics()
        with self.assertRaisesRegex(ValueError, "consecutive"):
            acc.update(field(1.), transition=2, accepted=True)
        with self.assertRaisesRegex(ValueError, "occupancy-aligned"):
            acc.update(dict(rho=np.ones((2, 2, 2)),
                mean_velocity_km_s=np.ones((3, 2, 2, 2)),
                velocity_valid=np.zeros((2, 2, 2)),
            ), transition=1, accepted=True)
        self.assertEqual(acc.retained_states, 0)

    def test_each_scale_can_be_finalized_without_retaining_other_scale_maps(self):
        acc = diagnostics()
        for transition, value in enumerate((1., 1., 2., 2., 3., 3., 4., 4.), 1):
            acc.update(field(value), transition=transition,
                accepted=(transition % 2 == 1))
        self.assertEqual(acc.finalize_scale(2)["status"],
            "FIELD_MCSE_AVAILABLE_NOT_CONVERGENCE_EVIDENCE")
        self.assertEqual(acc.finalize_scale(4)["status"],
            "FIELD_MCSE_AVAILABLE_NOT_CONVERGENCE_EVIDENCE")
        short = diagnostics()
        for transition, value in enumerate((1., 1., 2., 2., 3., 3.), 1):
            short.update(field(value), transition=transition,
                accepted=(transition % 2 == 1))
        insufficient = short.finalize_scale(4)
        self.assertEqual(insufficient["status"], "INSUFFICIENT_COMPLETE_BATCHES")
        self.assertEqual(insufficient["reporting_region_cell_denominators"],
            dict(full_box=8, training_count_cells=2))


if __name__ == "__main__":
    unittest.main()
