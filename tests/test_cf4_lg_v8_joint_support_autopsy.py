import hashlib
import json
from pathlib import Path

import numpy as np

from cf4_lg_v8_joint_support_autopsy import (
    hard_p2_margins,
    hard_p2_pass_from_margins,
    json_default,
    logmeanexp,
    weight_summary,
)


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/p2_lg_v8_joint_support_autopsy.json"


def test_weight_summary_includes_explicit_zero_weight_rows():
    summary = weight_summary([-2.0, -2.0, -np.inf, -np.inf])
    assert summary["n_total"] == 4
    assert summary["n_nonzero"] == 2
    assert np.isclose(summary["effective_sample_size"], 2.0)
    np.testing.assert_allclose(
        summary["normalized_weights"], [0.5, 0.5, 0.0, 0.0]
    )


def test_logmeanexp_has_no_pair_count_bonus():
    assert np.isclose(logmeanexp([-3.0]), -3.0)
    assert np.isclose(logmeanexp([-3.0, -3.0, -3.0]), -3.0)


def test_hard_p2_margin_reconstructs_pass_and_failure():
    screen = {
        "pair_member_mass_range_msun_h": [5e11, 4e12],
        "pair_mass_ratio_max": 4.0,
        "pair_separation_range_mpc_h": [0.3, 1.2],
        "pair_midpoint_max_offset_mpc_h": 5.0,
        "isolation_radius_mpc_h": 3.0,
    }
    pair = {
        "masses_msun_h": [1.2e12, 1.0e12],
        "mass_ratio": 1.2,
        "separation_mpc_h": 0.7,
        "midpoint_offset_mpc_h": 2.0,
        "isolation_mpc_h": 5.0,
    }
    margins = hard_p2_margins(pair, screen)
    assert hard_p2_pass_from_margins(margins)
    pair["separation_mpc_h"] = 1.3
    margins = hard_p2_margins(pair, screen)
    assert margins["separation_upper_mpc_h"] < 0.0
    assert not hard_p2_pass_from_margins(margins)


def test_autopsy_contract_forbids_fresh_work():
    program = json.loads(PROGRAM.read_text())
    assert program["status"] == "frozen_before_detailed_v8_pairwise_attribution"
    assert program["decision"]["fresh_v9_authorized"] is False
    assert program["decision"]["RAMSES_authorized"] is False
    assert program["information_firewall"]["new_seed_or_forward_accessed"] is False


def test_json_default_converts_numpy_boolean_and_scalars():
    assert json.loads(json.dumps({"pass": np.bool_(True)}, default=json_default)) == {
        "pass": True
    }
    assert json.loads(
        json.dumps({"count": np.int64(3), "value": np.float64(0.5)},
                   default=json_default)
    ) == {"count": 3, "value": 0.5}


def test_autopsy_contract_pins_implementation_hash():
    program = json.loads(PROGRAM.read_text())
    implementation = REPO / program["implementation"]["path"]
    digest = hashlib.sha256(implementation.read_bytes()).hexdigest()
    assert digest == program["implementation"]["sha256"]
