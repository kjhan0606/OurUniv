import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cf4_r2_v6_tully_velocity_sources import summarize_source_rows


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


if __name__ == "__main__":
    unittest.main()
