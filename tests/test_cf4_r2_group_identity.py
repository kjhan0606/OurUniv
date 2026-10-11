import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cf4_r2_t10106_source_identity import catalogue_relation, summarize_bridge


class GroupIdentityTests(unittest.TestCase):
    def test_catalogue_namespaces_are_classified_separately(self):
        result = summarize_bridge(
            (10, 11), (100, 101),
            ((100, "700", 10), (101, "700", 11)),
            {100: {"700"}, 101: {"700"}}, {10: "700", 11: "701"})
        self.assertEqual(result["CF4_catalogue_group_relation"], "shared_catalogue_group")
        self.assertEqual(result["2mpp_catalogue_group_relation"], "distinct_catalogue_groups")
        self.assertFalse(result["cross_catalogue_IDs_equated"])
        self.assertEqual([row["2mpp_GID"] for row in result["members"]], ["700", "701"])

    def test_unassigned_group_and_duplicate_secure_edge_are_not_hidden(self):
        self.assertEqual(catalogue_relation(((), ("9",))), "partly_unassigned")
        with self.assertRaisesRegex(ValueError, "duplicate secure edges"):
            summarize_bridge(
                (10,), (100,), ((100, "700", 10), (100, "700", 10)),
                {100: {"700"}}, {10: "700"})


if __name__ == "__main__":
    unittest.main()
