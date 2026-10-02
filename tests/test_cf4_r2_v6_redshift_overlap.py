import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cf4_r2_v6_redshift_overlap import (
    aggregate_relations,
    summarize_group_velocity_overlap,
)


class V6RedshiftOverlapTests(unittest.TestCase):
    def test_group_and_member_velocities_stay_separate(self):
        rows = summarize_group_velocity_overlap(
            {"T1": [(1, "100", 10), (2, "100", 11)]},
            {"100": {"Vcmb": "1000"}},
            {1: {"100"}, 2: {"100"}},
            {1: ["990"], 2: ["1010"]},
            {10: {"GID": "100", "Vcmb": "980"},
             11: {"GID": "100", "Vcmb": "1020"}},
            {"100": {"Vcmb": "950"}},
        )
        row = rows[0]
        self.assertEqual(row["relation_class"], "shared_catalogue_group|shared_catalogue_group")
        self.assertEqual(row["abs_CF4_individual_Vcmb_minus_2mpp_individual_Vcmb_km_s"]["median"], 10)
        self.assertEqual(row["abs_CF4_group_Vcmb_minus_linked_CF4_member_Vcmb_km_s"]["median"], 10)
        self.assertEqual(row["abs_2mpp_group_Vcmb_minus_linked_2mpp_member_Vcmb_km_s"]["median"], 50)
        self.assertEqual(row["abs_CF4_group_Vcmb_minus_linked_2mpp_group_Vcmb_km_s"]["n"], 1)
        self.assertEqual(row["abs_CF4_group_Vcmb_minus_linked_2mpp_group_Vcmb_km_s"]["median"], 50)

    def test_relation_aggregation_pools_member_pairs_but_not_group_pairs(self):
        first = summarize_group_velocity_overlap(
            {"T1": [(1, "cf4-1", 10), (2, "cf4-1", 11)]},
            {"cf4-1": {"Vcmb": "100"}},
            {1: {"cf4-1"}, 2: {"cf4-1"}},
            {1: ["90"], 2: ["80"]},
            {10: {"GID": "mpp-1", "Vcmb": "90"},
             11: {"GID": "mpp-1", "Vcmb": "80"}},
            {"mpp-1": {"Vcmb": "70"}},
        )[0]
        second = summarize_group_velocity_overlap(
            {"T2": [(3, "cf4-1", 12)]},
            {"cf4-1": {"Vcmb": "100"}},
            {3: {"cf4-1"}}, {3: ["190"]},
            {12: {"GID": "mpp-1", "Vcmb": "190"}},
            {"mpp-1": {"Vcmb": "70"}},
        )[0]
        aggregate = aggregate_relations([first, second])[first["relation_class"]]
        self.assertEqual(aggregate["group_count"], 2)
        self.assertEqual(aggregate["secure_member_pair_count"], 3)
        self.assertEqual(aggregate[
            "abs_CF4_group_Vcmb_minus_linked_2mpp_group_Vcmb_km_s"]["n"], 1)

    def test_ambiguous_cf4_member_velocity_is_not_arbitrarily_selected(self):
        row = summarize_group_velocity_overlap(
            {"T1": [(1, "100", 10)]}, {"100": {"Vcmb": "1000"}},
            {1: {"100"}}, {1: ["990", "995"]},
            {10: {"GID": "200", "Vcmb": "990"}},
            {"200": {"Vcmb": "980"}},
        )[0]
        self.assertEqual(row["ambiguous_CF4_member_velocity_records"], 1)
        self.assertEqual(row[
            "abs_CF4_individual_Vcmb_minus_2mpp_individual_Vcmb_km_s"]["n"], 0)


if __name__ == "__main__":
    unittest.main()
