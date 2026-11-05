import unittest

from src.cf4_r2_graph_closure import close_graph_roles


class GraphClosureTests(unittest.TestCase):
    def test_mixed_train_holdout_buffers_full_transitive_component(self):
        roles = {"train": 0, "heldout": 1}
        final, summary = close_graph_roles(
            roles, [("train", "parent"), ("parent", "gid"),
                    ("gid", "heldout")])
        self.assertEqual({final[k] for k in ("train", "parent", "gid", "heldout")}, {2})
        self.assertEqual(summary["mixed_train_heldout_components"], 1)

    def test_disjoint_roles_remain_separate(self):
        final, summary = close_graph_roles(
            {"train": 0, "heldout": 1}, [("train", "train-parent")])
        self.assertEqual(final["train-parent"], 0)
        self.assertEqual(final["heldout"], 1)
        self.assertEqual(summary["mixed_train_heldout_components"], 0)

    def test_existing_buffer_propagates_to_entire_component(self):
        final, summary = close_graph_roles(
            {"buffer": 2, "other": 0}, [("buffer", "other")])
        self.assertEqual(final["other"], 2)
        self.assertEqual(summary["components_buffered_by_mixing_or_existing_buffer"], 1)

    def test_invalid_role_rejected(self):
        with self.assertRaises(ValueError):
            close_graph_roles({"bad": 4}, [])


if __name__ == "__main__":
    unittest.main()
