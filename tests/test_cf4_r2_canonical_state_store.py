import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cf4_r2_canonical_state_store import CanonicalStateStore
from cf4_r2_corrected_split_hmc import corrected_split_step
from cf4_r2_prior_split_hmc import FixedSplitMetric


class CanonicalStateStoreTests(unittest.TestCase):
    def make_store(self, directory, *, capacity=4, dimension=3, chain="a"):
        return CanonicalStateStore(directory, chain_id=chain, capacity=capacity,
            dimension=dimension, contract_id="conditional-v6-test-contract")

    @staticmethod
    def rng_state(seed=0):
        return np.random.default_rng(seed).bit_generator.state

    def test_rejection_repeat_is_a_distinct_retained_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(temporary)
            q = np.array([1., -2., 3.], dtype=np.float64)
            q0 = np.array([0., 0., 0.], dtype=np.float64)
            store.append(q, previous_state=q0, previous_target=14.,
                transition=1, accepted=True, target_value=12.5,
                rng_state=self.rng_state(1))
            store.append(q, previous_state=q, previous_target=12.5,
                transition=2, accepted=False, target_value=12.5,
                rng_state=self.rng_state(2))
            self.assertEqual(len(store), 2)
            np.testing.assert_array_equal(store.read_state(0), q)
            np.testing.assert_array_equal(store.read_state(1), q)
            self.assertTrue(store.records[1]["accepted"] is False)

    def test_resume_continues_the_committed_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = self.make_store(temporary)
            q1 = np.array([0., 1., 2.])
            first.append(q1, previous_state=np.array([-1., -1., -1.]),
                previous_target=5., transition=1, accepted=True, target_value=4.,
                rng_state=self.rng_state(1))
            resumed = self.make_store(temporary)
            q2 = np.array([3., 4., 5.])
            resumed.append(q2, previous_state=q1, previous_target=4.,
                transition=2, accepted=True, target_value=5.,
                rng_state=self.rng_state(2))
            self.assertEqual(resumed.verify()["states"], 2)
            np.testing.assert_array_equal(resumed.read_state(1), q2)

    def test_partial_trailing_index_record_is_discarded_before_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(temporary)
            q1 = np.array([1., 2., 3.])
            q0 = np.array([0., 0., 0.])
            store.append(q1, previous_state=q0, previous_target=8.,
                transition=1, accepted=True, target_value=7.,
                rng_state=self.rng_state(1))
            with (Path(temporary) / CanonicalStateStore.INDEX_FILE).open("ab") as stream:
                stream.write(b'{"index":1,"partial":')
            resumed = self.make_store(temporary)
            self.assertEqual(len(resumed), 1)
            q2 = np.array([4., 5., 6.])
            resumed.append(q2, previous_state=q1, previous_target=7.,
                transition=2, accepted=True, target_value=8.,
                rng_state=self.rng_state(2))
            self.assertEqual(len(resumed), 2)
            lines = (Path(temporary) / CanonicalStateStore.INDEX_FILE).read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[-1])["transition"], 2)

    def test_corrupt_committed_state_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(temporary)
            store.append(np.array([1., 2., 3.]), previous_state=np.zeros(3),
                previous_target=8., transition=1, accepted=True, target_value=7.,
                rng_state=self.rng_state(1))
            array = np.load(store.state_path, mmap_mode="r+", allow_pickle=False)
            array[0, 0] = 99.
            array.flush()
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                self.make_store(temporary)

    def test_capacity_and_nonfinite_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(temporary, capacity=1)
            with self.assertRaisesRegex(ValueError, "finite canonical state"):
                store.append(np.array([np.nan, 1., 2.]), transition=1,
                    previous_state=np.zeros(3), previous_target=2.,
                    accepted=True, target_value=1.,rng_state=self.rng_state(1))
            q = np.array([1., 2., 3.])
            store.append(q, previous_state=np.zeros(3), previous_target=2.,
                transition=1, accepted=True, target_value=1.,
                rng_state=self.rng_state(1))
            with self.assertRaisesRegex(ValueError, "capacity exhausted"):
                store.append(q, previous_state=q, previous_target=1.,
                    transition=2, accepted=False, target_value=1.,
                    rng_state=self.rng_state(2))

    def test_rejection_must_repeat_the_previous_state_and_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(temporary)
            q = np.array([1., 2., 3.])
            store.append(q, previous_state=np.zeros(3), previous_target=4.,
                transition=1, accepted=True, target_value=3.,
                rng_state=self.rng_state(1))
            with self.assertRaisesRegex(ValueError, "rejected transition"):
                store.append(np.array([1., 2., 4.]), previous_state=q,
                    previous_target=3., transition=2, accepted=False, target_value=3.,
                    rng_state=self.rng_state(2))
            with self.assertRaisesRegex(ValueError, "float64"):
                store.append(q.astype(np.float32), previous_state=q,
                    previous_target=3., transition=2, accepted=True, target_value=2.,
                    rng_state=self.rng_state(2))

    def test_corrected_hmc_sequence_persists_accepted_and_rejected_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(temporary, capacity=2, dimension=9)
            metric = FixedSplitMetric(np.ones((2, 2, 2)), np.eye(1))

            def gaussian(q):
                return .5 * float(q @ q), np.asarray(q, dtype=np.float64)

            q = np.linspace(-.4, .4, 9)
            value, gradient = gaussian(q)
            force_value = value
            rng = np.random.default_rng(18)
            previous = q.copy()
            previous_value = value
            q, value, force_value, gradient, info = corrected_split_step(
                gaussian, lambda x: gaussian(x)[0], metric, q, value,
                force_value, gradient, rng, step=.2, steps=2)
            self.assertTrue(info["accepted"])
            store.append(q, previous_state=previous, previous_target=previous_value,
                transition=1, accepted=info["accepted"], target_value=value,
                rng_state=rng.bit_generator.state)

            previous = q.copy()
            previous_value = value
            q, value, force_value, gradient, info = corrected_split_step(
                gaussian, lambda _: np.inf, metric, q, value,
                force_value, gradient, rng, step=.2, steps=2)
            self.assertFalse(info["accepted"])
            store.append(q, previous_state=previous, previous_target=previous_value,
                transition=2, accepted=info["accepted"], target_value=value,
                rng_state=rng.bit_generator.state)
            np.testing.assert_array_equal(store.read_state(1), store.read_state(0))
            self.assertEqual(store.records[1]["accepted"], False)

    def test_rng_state_is_persisted_checksummed_and_resume_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(temporary)
            q = np.array([1., 2., 3.])
            rng = np.random.default_rng(123)
            rng.standard_normal(10)
            rng_state = rng.bit_generator.state
            expected_next = rng.standard_normal(5)
            store.append(q, previous_state=np.zeros(3), previous_target=4.,
                transition=1, accepted=True, target_value=3.,rng_state=rng_state)
            record = store.records[0]
            self.assertEqual(record["rng_state"], rng_state)
            self.assertEqual(len(record["rng_state_sha256"]), 64)
            resumed = self.make_store(temporary)
            self.assertEqual(resumed.records[0]["rng_state"], rng_state)
            self.assertTrue(resumed.verify()["all_rng_state_checksums_valid"])
            replay = np.random.default_rng()
            replay.bit_generator.state = resumed.records[0]["rng_state"]
            np.testing.assert_array_equal(replay.standard_normal(5), expected_next)

    def test_corrupt_rng_state_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(temporary)
            store.append(np.array([1., 2., 3.]), previous_state=np.zeros(3),
                previous_target=4., transition=1, accepted=True, target_value=3.,
                rng_state=self.rng_state(1))
            index_path = Path(temporary) / CanonicalStateStore.INDEX_FILE
            record = json.loads(index_path.read_text())
            record["rng_state"]["state"]["state"] += 1
            index_path.write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "invalid canonical-state metadata"):
                self.make_store(temporary)

    def test_non_json_rng_state_is_rejected_before_state_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(temporary)
            with self.assertRaisesRegex(ValueError, "RNG state must be finite"):
                store.append(np.array([1., 2., 3.]), previous_state=np.zeros(3),
                    previous_target=4., transition=1, accepted=True,
                    target_value=3.,rng_state={"bad": np.nan})
            self.assertEqual(len(store), 0)

    def test_legacy_v1_store_remains_readable_but_not_extendable(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(temporary)
            q = np.array([1., 2., 3.])
            store.append(q, previous_state=np.zeros(3), previous_target=4.,
                transition=1, accepted=True, target_value=3.,
                rng_state=self.rng_state(1))
            meta_path = Path(temporary) / CanonicalStateStore.META_FILE
            metadata = json.loads(meta_path.read_text())
            metadata["version"] = 1
            meta_path.write_text(json.dumps(metadata) + "\n")
            index_path = Path(temporary) / CanonicalStateStore.INDEX_FILE
            record = json.loads(index_path.read_text())
            record.pop("rng_state")
            record.pop("rng_state_sha256")
            index_path.write_text(json.dumps(record) + "\n")

            legacy = self.make_store(temporary)
            self.assertEqual(legacy.verify()["states"], 1)
            self.assertFalse(legacy.verify()["all_rng_state_checksums_valid"])
            with self.assertRaisesRegex(ValueError, "readable but cannot be extended"):
                legacy.append(q, previous_state=q, previous_target=3.,
                    transition=2, accepted=False, target_value=3.,
                    rng_state=self.rng_state(2))


if __name__ == "__main__":
    unittest.main()
