"""Source-bound regression for the R2 grouped overlap observation audit."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cf4_r2_overlap_group_audit.py"
spec = importlib.util.spec_from_file_location("cf4_r2_overlap_group_audit", SCRIPT)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def test_frozen_overlap_is_not_17007_eligible_independent_redshifts():
    result = audit_module.audit()
    assert result["full_catalogue_all_nonunmatched_unique_targets"] == 17007
    assert result["N128_eligible_all_nonunmatched_unique_targets"] == 15211
    assert result["N128_calibration_all_nonunmatched_unique_targets"] == 3120
    assert result["N128_eligible_unique_targets_by_class"]["secure_joint_mark"] == 14878
    assert result["N128_eligible_secure_targets_missing_canonical_group"] == 4
    assert result["N128_eligible_secure_groups_with_multiple_matched_members"] > 0
    assert result["N128_eligible_secure_abs_CF4_group_V3k_minus_2Mpp_Vcmb_km_s"]["p90"] > 300
