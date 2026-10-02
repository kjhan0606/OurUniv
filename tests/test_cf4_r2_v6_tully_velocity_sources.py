import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cf4_r2_v6_tully_velocity_sources import summarize_source_rows


class TullyVelocitySourceTests(unittest.TestCase):
    def test_source_reference_and_rounding_counts_are_separate(self):
        summary = summarize_source_rows([
            {"reference": "2012ApJS..199...26H", "abs_delta_km_s": 0.0},
            {"reference": "2012ApJS..199...26H", "abs_delta_km_s": 0.4},
            {"reference": "2012ApJS..199...26H", "abs_delta_km_s": 1.2},
            {"reference": "other", "abs_delta_km_s": 8.0},
        ])
        self.assertEqual(summary["linked_member_count"], 4)
        self.assertEqual(summary["Huchra_2012_2MRS_reference_count"], 3)
        self.assertEqual(summary["Huchra_2012_2MRS_equal_within_0p5_km_s_count"], 2)
        self.assertEqual(summary["by_reference"]["2012ApJS..199...26H"]["exact_equal_count"], 1)

    def test_missing_reference_is_preserved(self):
        summary = summarize_source_rows([
            {"reference": "", "abs_delta_km_s": 3.0},
        ])
        self.assertEqual(summary["reference_counts"], {"MISSING": 1})
        self.assertEqual(summary["by_reference"]["MISSING"]["count"], 1)

    def test_empty_source_rows_have_explicit_empty_statistics(self):
        summary = summarize_source_rows([])
        self.assertEqual(summary["linked_member_count"], 0)
        self.assertEqual(summary["reference_counts"], {})
        self.assertEqual(summary["absolute_CF4_individual_minus_2mpp_point_Vcmb_km_s"],
                         {"n": 0, "median": None, "p90": None, "maximum": None})


if __name__ == "__main__":
    unittest.main()
