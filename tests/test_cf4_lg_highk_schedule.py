from pathlib import Path

import numpy as np
import pytest

from cf4_aggregate_evidence_oracle import geometry_key
from cf4_lg_highk_schedule import (
    build_joint_schedule,
    parent_l1_null,
    validate_bank,
)


def synthetic_bank() -> dict[str, np.ndarray]:
    count, parent_count = 8192, 256
    group = np.repeat(np.arange(4, dtype=np.int16), 2048)
    within = np.tile(np.arange(2048, dtype=np.int32), 4)
    midpoint = np.zeros((count, 3), dtype=np.float64)
    midpoint[:, 0] = (within % 9 - 4) * (2.0 / 3.0)
    axis = np.zeros((count, 3), dtype=np.float64)
    axis[:, 0] = 1.0
    keys = np.asarray([
        geometry_key(q, a) for q, a in zip(midpoint, axis)
    ], dtype=np.int16)
    conditional = np.full((count, parent_count), 1.0, dtype=np.float64)
    conditional[np.arange(count), (within + 17 * group) % parent_count] = 8.0
    conditional /= conditional.sum(axis=1, keepdims=True)
    weight = np.full(count, 1.0 / count)
    return {
        "cycle": np.asarray(16),
        "midpoint_mpc_h": midpoint,
        "axis": axis,
        "keys": keys,
        "group_id": group,
        "group_particle": within,
        "weight": weight,
        "parent_seed": np.arange(3193, 3449),
        "parent_conditional_probability": conditional,
        "P_parent": weight @ conditional,
    }


def test_real_design_is_fail_closed_before_schedule_or_fields():
    root = Path(__file__).parents[1]
    import json
    design = json.loads(
        (root / "config/cf4_lg_highk_conditioning_design_v1.json").read_text()
    )
    authorization = design["authorization"]
    assert authorization["real_bank_read_only_preflight_authorized"] is True
    assert authorization["real_schedule_generation_authorized"] is False
    assert authorization["conditional_field_technical_pilot_authorized"] is False
    assert authorization["PM_authorized"] is False
    assert design["probability_contract"]["single_parent_selection_forbidden"] is True
    assert "Do not multiply" in design["probability_contract"]["peak_evidence_policy"]
    assert design["resolution_ladder"]["zoom_pilot"]["particle_spacing_mpc_h"] < 0.3
    runner = (root / "scripts/run_cf4_lg_highk_preflight_v1.sbatch").read_text()
    assert "--output-root" not in runner
    assert "#SBATCH --mem=1G" in runner
    assert "OMP_NUM_THREADS=1" in runner
    assert "workers=-1" not in runner


def test_validate_and_schedule_are_deterministic_and_group_balanced():
    bank = synthetic_bank()
    validation = validate_bank(bank)
    assert validation["particle_count"] == 8192
    first, first_meta = build_joint_schedule(bank)
    second, second_meta = build_joint_schedule(bank)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert first_meta == second_meta
    np.testing.assert_array_equal(
        np.bincount(first["group_id"], minlength=4), [64, 64, 64, 64]
    )
    assert len(np.unique(first["fine_field_seed"])) == 256
    assert len(np.unique(first["likelihood_noise_seed"])) == 256
    assert np.isclose(first["posterior_weight"].sum(), 1.0)
    assert first_meta["peak_evidence_reapplied"] is False
    assert first_meta["single_parent_selected"] is False


def test_schedule_seed_changes_draw_but_not_prospective_seed_ranges():
    bank = synthetic_bank()
    first, _ = build_joint_schedule(bank, master_seed=11)
    second, _ = build_joint_schedule(bank, master_seed=12)
    assert np.any(first["bank_row"] != second["bank_row"])
    np.testing.assert_array_equal(first["fine_field_seed"], second["fine_field_seed"])
    np.testing.assert_array_equal(
        first["likelihood_noise_seed"], second["likelihood_noise_seed"]
    )


def test_parent_l1_null_is_deterministic():
    bank = synthetic_bank()
    _, metadata = build_joint_schedule(bank)
    first = parent_l1_null(
        bank, observed_l1=metadata["empirical_parent_L1"], draws=128, seed=44
    )
    second = parent_l1_null(
        bank, observed_l1=metadata["empirical_parent_L1"], draws=128, seed=44
    )
    assert first == second
    assert first["draws"] == 128
    assert first["q999"] >= first["q99"]
    assert 0.0 < first["tail_probability"] <= 1.0


def test_validate_rejects_parent_marginal_mismatch():
    bank = synthetic_bank()
    bank["P_parent"] = bank["P_parent"].copy()
    bank["P_parent"][0] += 1.0e-4
    with pytest.raises(ValueError, match="reconstruct parent marginal"):
        validate_bank(bank)
