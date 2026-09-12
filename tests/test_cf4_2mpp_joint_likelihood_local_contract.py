from __future__ import annotations

import json
from pathlib import Path

from cf4_2mpp_joint_likelihood_local import (
    SELECTION_EXPOSURE_UNITS,
    VELOCITY_CONVENTION,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/cf4_2mpp_joint_likelihood_local_contract_v1.json"


def test_local_contract_freezes_units_and_blocks_execution():
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert value["schema"] == "ouruniv-cf4-2mpp-joint-likelihood-local-contract-v1"
    assert value["stage"] == "KF-DESIGN"
    assert value["coordinate_and_velocity_units"]["coherent_and_FoG_displacement_cMpc_h"] == "h*v_r/(a*H(a))"
    assert value["coordinate_and_velocity_units"]["velocity"] == VELOCITY_CONVENTION
    assert value["selection_exposure"]["units"] == SELECTION_EXPOSURE_UNITS
    assert value["selection_exposure"]["shape"] == "(6,N,N,N) with one common cubic spatial grid"
    assert value["selection_exposure"]["normalization"] == "raw exposure; never rescaled to observed population totals"
    assert value["factor_ownership"]["independent_twompp_redshift_factor"] is False
    assert value["implementations"]["jax"]["value_and_gradient_must_match_numpy_oracle"] is True
    assert value["implementations"]["jax"]["entrypoint"] == "joint_log_likelihood_jax_checked"
    assert value["implementations"]["crossmatch_manifest"]["schema"] == "ouruniv-cf4-2mpp-secure-object-manifest-v1"
    assert value["implementations"]["crossmatch_manifest"]["exact_source_path_equality_required"] is True
    assert value["implementations"]["crossmatch_manifest"]["exact_group_count_from_manifest"] == 11610
    assert value["implementations"]["crossmatch_manifest"]["all_group_indices_must_be_used"] is True
    for key in ("real_catalog_inference", "KF_EXPAND", "Slurm_submission", "GPFS_read", "GPFS_write", "IC_PM_HOP_RAMSES"):
        assert value["authority"][key] is False


def test_local_contract_requires_high_order_quadrature_and_mock_calibration():
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert value["redshift_space_model"]["supported_orders"] == [3, 5, 7, 9, 11, 13, 15]
    assert "GH3 must be compared with GH7" in value["redshift_space_model"]["convergence_gate"]
    assert "independent development and untouched validation mock calibration" in value["required_gates_before_KF_EXPAND"]
    assert value["redshift_space_model"]["preregistered_case_id"] == "RSD_FOG_boundary_v1"
    assert value["redshift_space_model"]["preregistered_low_order"] == 3
    assert value["redshift_space_model"]["preregistered_high_orders"] == [7, 9]
    assert value["redshift_space_model"]["preregistered_relative_l1_tolerance"] == 0.005
