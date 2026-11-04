import tempfile
import unittest

import numpy as np

from cf4_r2_canonical_state_store import CanonicalStateStore
from cf4_r2_field_batch_mcse import MultiScaleFieldMCSEAccumulator
from cf4_r2_posterior_moments import PresentMomentAccumulator
from cf4_r2_retained_diagnostic_replay import replay_retained_field_diagnostics
from cf4_r2_retained_field_mcse import RetainedFieldMCSEDiagnostics


class RetainedDiagnosticReplayTests(unittest.TestCase):
    @staticmethod
    def field_moments(q):
        shape = (2, 2, 2)
        rho = np.full(shape, max(float(q[0]), .25))
        velocity = np.stack([np.full(shape, q[1]),
            np.full(shape, 2*q[1]), np.full(shape, -q[1])])
        physical = np.full((3,) + shape, .5 + abs(float(q[1])))
        return dict(rho=rho, mean_velocity_km_s=velocity,
            physical_velocity_variance_km2_s2=physical,
            velocity_valid=rho > 0)

    @staticmethod
    def accumulators(train_keys):
        moments = PresentMomentAccumulator((2, 2, 2))
        mcse = RetainedFieldMCSEDiagnostics((2, 2, 2), train_keys=train_keys,
            batch_sizes=(2, 4), minimum_batches=2,
            count_grid_size=1, population_count=1)
        return moments, mcse

    def test_replay_matches_direct_accumulation_including_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = CanonicalStateStore(temporary, chain_id="replay-test",
                capacity=4, dimension=2, contract_id="replay-contract")
            q0 = np.array([1., 0.], dtype=np.float64)
            states = [np.array([1., .1]), np.array([1., .1]),
                      np.array([2., .3]), np.array([1.5, -.2])]
            accepted = [True, False, True, True]
            previous = q0
            previous_target = 12.
            for index, (q, is_accepted) in enumerate(zip(states, accepted), start=1):
                target = previous_target - .25 if is_accepted else previous_target
                store.append(q, previous_state=previous,
                    previous_target=previous_target, transition=index,
                    accepted=is_accepted, target_value=target,
                    rng_state=np.random.default_rng(index).bit_generator.state)
                previous = q
                previous_target = target

            train_keys = np.array([0], dtype=np.int64)
            direct_moments, direct_mcse = self.accumulators(train_keys)
            for q, record in store.iter_states(verify=True):
                field = self.field_moments(q)
                direct_moments.update(field)
                direct_mcse.update(field, transition=record["transition"],
                    accepted=record["accepted"])

            replay_moments, replay_mcse = self.accumulators(train_keys)
            summary = replay_retained_field_diagnostics(store,
                field_moments=self.field_moments,
                posterior_moments=replay_moments, field_mcse=replay_mcse)
            self.assertEqual(summary["retained_states"], 4)
            self.assertEqual(summary["accepted_transitions"], 3)
            self.assertEqual(summary["rejected_transitions"], 1)
            self.assertEqual(replay_moments.n, direct_moments.n)
            for key, expected in direct_moments.checkpoint_arrays().items():
                np.testing.assert_array_equal(replay_moments.checkpoint_arrays()[key], expected)
            for key, expected in direct_mcse.checkpoint_arrays().items():
                np.testing.assert_array_equal(replay_mcse.checkpoint_arrays()[key], expected)

    def test_replay_refuses_nonempty_accumulators(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = CanonicalStateStore(temporary, chain_id="replay-test",
                capacity=2, dimension=2, contract_id="replay-contract")
            moments, mcse = self.accumulators(np.array([0], dtype=np.int64))
            moments.update(self.field_moments(np.array([1., .2])))
            with self.assertRaisesRegex(ValueError, "must be empty"):
                replay_retained_field_diagnostics(store,
                    field_moments=self.field_moments,
                    posterior_moments=moments, field_mcse=mcse)


if __name__ == "__main__":
    unittest.main()
