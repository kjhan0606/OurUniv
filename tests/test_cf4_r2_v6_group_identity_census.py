import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cf4_r2_v6_group_identity_census import group_relation_census


class GroupIdentityCensusTests(unittest.TestCase):
    def test_census_keeps_catalogue_partitions_separate(self):
        rows = group_relation_census(
            {"T1": {10, 11}, "T2": {12, 13}},
            {"T1": [(100, "700", 10), (101, "700", 11)],
             "T2": [(102, "701", 12), (103, "701", 13)]},
            {100: {"700"}, 101: {"700"}, 102: {"701"}, 103: {"702"}},
            {10: "900", 11: "900", 12: "901", 13: "902"})
        self.assertEqual(rows[0]["CF4_catalogue_group_relation"], "shared_catalogue_group")
        self.assertEqual(rows[0]["2mpp_catalogue_group_relation"], "shared_catalogue_group")
        self.assertEqual(rows[1]["CF4_catalogue_group_relation"], "distinct_catalogue_groups")
        self.assertEqual(rows[1]["2mpp_catalogue_group_relation"], "distinct_catalogue_groups")
        self.assertEqual(rows[0]["crossmatch_vs_CF4_member_table"], {"match": 2})
        self.assertEqual(rows[1]["crossmatch_vs_CF4_member_table"], {"match": 1, "mismatch": 1})


if __name__ == "__main__":
    unittest.main()
