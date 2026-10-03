import io
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cf4_r2_v6_tully_velocity_sources import (
    parse_2mrs_table3, summarize_2mrs_id_source_join, summarize_source_rows,
    summarize_tully_member_coverage,
)


class TullyVelocitySourceTests(unittest.TestCase):
    def test_explicit_2mrs_sources_and_rounding_counts_are_separate(self):
        summary = summarize_source_rows([
            {"reference": "20112MRS.FLWO.0000H", "abs_delta_km_s": 0.0},
            {"reference": "20112MRS.FLWO.0000H", "abs_delta_km_s": 0.4},
            {"reference": "20112MRS.ZMA..0000H", "abs_delta_km_s": 1.2},
            {"reference": "1999ApJS..121..287H", "abs_delta_km_s": 8.0},
        ])
        self.assertEqual(summary["linked_member_count"], 4)
        self.assertEqual(summary["reference_class_counts"], {
            "explicit_2MRS_source_code": 3,
            "other_or_unresolved_reference": 1,
        })
        self.assertEqual(summary["explicit_2MRS_reference_codes"], {
            "20112MRS.FLWO.0000H": 2,
            "20112MRS.ZMA..0000H": 1,
        })
        self.assertEqual(summary["by_reference_class"]["explicit_2MRS_source_code"]
                         ["equal_within_0p5_km_s_count"], 2)
        self.assertEqual(summary["absolute_velocity_agreement_counts"]["exact_equal"], 1)

    def test_other_reference_is_not_called_non_2mrs(self):
        summary = summarize_source_rows([
            {"reference": "1999ApJS..121..287H", "abs_delta_km_s": 3.0},
        ])
        self.assertEqual(summary["reference_class_counts"],
                         {"other_or_unresolved_reference": 1})

    def test_missing_reference_is_preserved(self):
        summary = summarize_source_rows([
            {"reference": "", "abs_delta_km_s": 3.0},
        ])
        self.assertEqual(summary["reference_counts"], {"MISSING": 1})
        self.assertEqual(summary["reference_class_counts"], {"missing_reference": 1})
        self.assertEqual(summary["by_reference"]["MISSING"]["count"], 1)

    def test_empty_source_rows_have_explicit_empty_statistics(self):
        summary = summarize_source_rows([])
        self.assertEqual(summary["linked_member_count"], 0)
        self.assertEqual(summary["reference_counts"], {})
        self.assertEqual(summary["absolute_CF4_individual_minus_2mpp_point_Vcmb_km_s"],
                         {"n": 0, "median": None, "p90": None, "maximum": None})

    def test_2mrs_fixed_width_join_uses_id_and_r_cz_columns(self):
        row = bytearray(b" " * 233)
        row[0:16] = b"12345678+1234567"
        row[185:204] = b"20112MRS.FLWO.0000H"
        table = parse_2mrs_table3(io.BytesIO(bytes(row) + b"\n"))
        self.assertEqual(table, {"12345678+1234567": "20112MRS.FLWO.0000H"})

    def test_2mrs_fixed_width_join_pads_omitted_blank_tail_fields(self):
        row = bytearray(b" " * 172)
        row[0:16] = b"12345678+1234567"
        self.assertEqual(parse_2mrs_table3(io.BytesIO(bytes(row) + b"\n")),
                         {"12345678+1234567": ""})

    def test_2mrs_fixed_width_join_rejects_duplicate_and_too_short_id_rows(self):
        row = bytearray(b" " * 233)
        row[0:16] = b"12345678+1234567"
        data = bytes(row) + b"\n" + bytes(row) + b"\n"
        with self.assertRaisesRegex(ValueError, "duplicate 2MRS ID"):
            parse_2mrs_table3(io.BytesIO(data))
        with self.assertRaisesRegex(ValueError, "short 2MRS table3 ID row"):
            parse_2mrs_table3(io.BytesIO(b"too short\n"))

    def test_2mrs_join_distinguishes_matching_publication_from_id_absence(self):
        summary = summarize_2mrs_id_source_join([
            {"2mpp_reference": "20112MRS.FLWO.0000H",
             "2mrs_reference": "20112MRS.FLWO.0000H", "2mrs_id_match": True,
             "abs_velocity_delta_km_s": 0.4},
            {"2mpp_reference": "1999ApJS..121..287H",
             "2mrs_reference": "20112MRS.FLWO.0000H", "2mrs_id_match": True,
             "abs_velocity_delta_km_s": 25.0},
            {"2mpp_reference": "2003A&A...412...57P",
             "2mrs_reference": "", "2mrs_id_match": False,
             "abs_velocity_delta_km_s": 60.0},
        ])
        self.assertEqual(summary["selected_tully_member_count"], 3)
        self.assertEqual(summary["2mrs_main_table_id_match_count"], 2)
        self.assertEqual(summary["same_reference_code_count"], 1)
        self.assertEqual(summary["source_join_status_counts"], {
            "matched_id_different_reference_code": 1,
            "matched_id_same_reference_code": 1,
            "not_in_2mrs_main_table": 1,
        })
        delta_summary = summary["CF4_2mpp_abs_Vcmb_difference_km_s_by_source_status"]
        self.assertEqual(delta_summary["matched_id_same_reference_code"]["median"], 0.4)
        self.assertEqual(delta_summary["matched_id_different_reference_code"]["median"], 25.0)

    def test_tully_member_coverage_uses_distinct_parent_nests_and_full_nmb(self):
        def group_line(nest, nmb, pgc1):
            line = bytearray(b" " * 21)
            line[3:9] = f"{nest:6d}".encode()
            line[10:13] = f"{nmb:3d}".encode()
            line[14:21] = f"{pgc1:7d}".encode()
            return bytes(line).decode()

        def member_line(nest, pgc):
            line = bytearray(b" " * 17)
            line[3:9] = f"{nest:6d}".encode()
            line[10:17] = f"{pgc:7d}".encode()
            return bytes(line).decode()

        groups = [group_line(11, 4, 101), group_line(22, 2, 201)]
        members = [member_line(11, pgc) for pgc in (101, 102, 103, 104)]
        members += [member_line(22, pgc) for pgc in (201, 202)]
        edges = [(101, "101", 1, "a"), (102, "101", 2, "a"),
                 (201, "201", 3, "b"), (202, "201", 4, "b")]
        selected = [(101, "101", 1), (201, "201", 3), (202, "201", 4)]
        result = summarize_tully_member_coverage(edges, selected, groups, members)
        self.assertEqual(result["unique_Tully_parent_Nest_count"], 2)
        self.assertEqual(result["sum_Tully_Nmb_over_parent_Nests"], 6)
        self.assertEqual(result["secure_links_that_are_Tully_table4_members"], 3)
        self.assertEqual(result["aggregate_linked_fraction_of_Tully_Nmb"], 0.5)
        self.assertEqual(result["Nests_with_all_Tully_members_linked"], 1)
        self.assertEqual(result["per_Nest_fraction_bins"], {"(0,0.25]": 1, "1": 1})


if __name__ == "__main__":
    unittest.main()
