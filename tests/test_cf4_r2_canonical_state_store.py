import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cf4_r2_canonical_state_store import CanonicalStateStore


class CanonicalStateStoreTests(unittest.TestCase):
    def make_store(self, directory, *, capacity=4, dimension=3, chain="a"):
        return CanonicalStateStore(directory, chain_id=chain, capacity=capacity,
            dimension=dimension, contract_id="conditional-v6-test-contract")

    def test_rejection_repeat_is_a_distinct_retained_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(temporary)
            q = np.array([1., -2., 3.], dtype=np.float64)
            q0 = np.array([0., 0., 0.], dtype=np.float64)
            store.append(q, previous_state=q0, previous_target=14.,
                transition=1, accepted=True, target_value=12.5)
            store.append(q, previous_state=q, previous_target=12.5,
                transition=2, accepted=False, target_value=12.5)
            self.assertEqual(len(store), 2)
            np.testing.assert_array_equal(store.read_state(0), q)
            np.testing.assert_array_equal(store.read_state(1), q)
            self.assertTrue(store.records[1]["accepted"] is False)

    def test_resume_continues_the_committed_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = self.make_store(temporary)
            q1 = np.array([0., 1., 2.])
            first.append(q1, previous_state=np.array([-1., -1., -1.]),
                previous_target=5., transition=1, accepted=True, target_value=4.)
            resumed = self.make_store(temporary)
            q2 = np.array([3., 4., 5.])
            resumed.append(q2, previous_state=q1, previous_target=4.,
                transition=2, accepted=True, target_value=5.)
            self.assertEqual(resumed.verify()["states"], 2)
            np.testing.assert_array_equal(resumed.read_state(1), q2)

    def test_partial_trailing_index_record_is_discarded_before_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(temporary)
            q1 = np.array([1., 2., 3.])
            q0 = np.array([0., 0., 0.])
            store.append(q1, previous_state=q0, previous_target=8.,
                transition=1, accepted=True, target_value=7.)
            with (Path(temporary) / CanonicalStateStore.INDEX_FILE).open("ab") as stream:
                stream.write(b'{"index":1,"partial":')
            resumed = self.make_store(temporary)
            self.assertEqual(len(resumed), 1)
            q2 = np.array([4., 5., 6.])
            resumed.append(q2, previous_state=q1, previous_target=7.,
                transition=2, accepted=True, target_value=8.)
            self.assertEqual(len(resumed), 2)
            lines = (Path(temporary) / CanonicalStateStore.INDEX_FILE).read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[-1])["transition"], 2)

    def test_corrupt_committed_state_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(temporary)
            store.append(np.array([1., 2., 3.]), previous_state=np.zeros(3),
                previous_target=8., transition=1, accepted=True, target_value=7.)
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
                    accepted=True, target_value=1.)
            q = np.array([1., 2., 3.])
            store.append(q, previous_state=np.zeros(3), previous_target=2.,
                transition=1, accepted=True, target_value=1.)
            with self.assertRaisesRegex(ValueError, "capacity exhausted"):
                store.append(q, previous_state=q, previous_target=1.,
                    transition=2, accepted=False, target_value=1.)

    def test_rejection_must_repeat_the_previous_state_and_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = self.make_store(temporary)
            q = np.array([1., 2., 3.])
            store.append(q, previous_state=np.zeros(3), previous_target=4.,
                transition=1, accepted=True, target_value=3.)
            with self.assertRaisesRegex(ValueError, "rejected transition"):
                store.append(np.array([1., 2., 4.]), previous_state=q,
                    previous_target=3., transition=2, accepted=False, target_value=3.)
            with self.assertRaisesRegex(ValueError, "float64"):
                store.append(q.astype(np.float32), previous_state=q,
                    previous_target=3., transition=2, accepted=True, target_value=2.)


if __name__ == "__main__":
    unittest.main()
