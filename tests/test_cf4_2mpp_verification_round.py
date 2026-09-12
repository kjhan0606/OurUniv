from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "config/cf4_2mpp_verification_round_v1.json"


def test_sequential_verification_record_is_fail_closed():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["overall_status"] == "NO_GO_KF_EXPAND"
    checks = record["checks"]
    assert checks["1_jax_x64_value_gradient"]["status"] == "PASS"
    assert checks["2_preregistered_gh_boundary_stress"]["status"] == "FAIL"
    assert checks["3_selection_lf_bias_model_discrepancy"]["status"] == "PARTIAL_NO_GO"
    assert checks["4_development_and_untouched_validation_mocks"]["status"] == "NO_GO"
    assert checks["5_observational_resolution_frontier"]["status"] == "NO_GO"
    assert record["scope"]["gpfs_used"] is False
    assert record["scope"]["slurm_used"] is False
