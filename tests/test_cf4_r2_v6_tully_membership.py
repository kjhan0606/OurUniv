import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cf4_r2_v6_tully_membership import aggregate_by_relation, summarize_tully_membership


class V6TullyMembershipTests(unittest.TestCase):
    def test_linked_members_match_the_tully_parent(self):
        rows = summarize_tully_membership(
            {"T1": [(1, "100", 10, "shared|shared"),
                    (2, "100", 11, "shared|shared")]},
            {"100": {500}}, {1: {500}, 2: {500}}, {1: {500}, 2: {500}},
        )
        row = rows[0]
        self.assertEqual(row["Tully_member_Nest_relation"], "shared_catalogue_group")
        self.assertEqual(row["Tully_parent_Nest_relation_from_CF4_1PGC"],
                         "shared_catalogue_group")
        self.assertEqual(row["member_vs_parent_status"],
                         {"member_matches_CF4_parent_Nest": 2})
        self.assertEqual(row["Tully_table5_vs_table4_membership"], {"match": 2})

    def test_mismatch_and_ambiguous_grouping_are_not_resolved_by_choice(self):
        rows = summarize_tully_membership(
            {"T2": [(3, "200", 12, "shared|partly")],
             "T3": [(4, "300", 13, "distinct|shared")]},
            {"200": {700}, "300": {800, 801}},
            {3: {701}, 4: {800, 801}},
            {3: {701}, 4: {800, 801}},
        )
        by_label = {row["source_group_label"]: row for row in rows}
        self.assertEqual(by_label["T2"]["member_vs_parent_status"],
                         {"member_differs_from_CF4_parent_Nest": 1})
        self.assertEqual(by_label["T3"]["member_vs_parent_status"],
                         {"member_ambiguous": 1})
        self.assertEqual(by_label["T3"]["Tully_parent_Nest_relation_from_CF4_1PGC"],
                         "ambiguous_member_grouping")
        self.assertEqual(by_label["T3"]["Tully_member_Nest_relation"],
                         "ambiguous_member_grouping")

    def test_relation_aggregation_preserves_all_group_counts(self):
        rows = summarize_tully_membership(
            {"T1": [(1, "100", 10, "shared|shared")],
             "T2": [(2, "100", 11, "shared|shared")]},
            {"100": {500}}, {1: {500}, 2: {500}}, {1: {500}, 2: {500}},
        )
        result = aggregate_by_relation(rows)["shared|shared"]
        self.assertEqual(result["group_count"], 2)
        self.assertEqual(result["secure_member_pair_count"], 2)
        self.assertEqual(result["member_vs_parent_status_counts"],
                         {"member_matches_CF4_parent_Nest": 2})


if __name__ == "__main__":
    unittest.main()
