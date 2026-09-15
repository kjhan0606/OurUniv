import json
from pathlib import Path

import numpy as np

from hong2021_v66_conditional_gradient_routing_audit import (
    classify,
    gradient_summary,
    selection_flags,
)


DOMAINS = ("TNG100", "SIMBA", "Swift")
REPO = Path(__file__).resolve().parents[1]


def test_v66_joint_gradient_summary_preserves_conditional_components() -> None:
    rows = []
    for position in range(4):
        for domain in DOMAINS:
            pair = np.ones(495, dtype=np.float64)
            pair[position] += 0.01
            rows.append(
                {
                    "domain": domain,
                    "joint_pair_gradient": pair.tolist(),
                    "joint_bounded_NLL_gradient": (0.5 * pair).tolist(),
                }
            )
    summary = gradient_summary(rows)
    routing, scale, compatible = selection_flags(
        summary, 0.0, [1.0] * 12, 1.0e-5
    )
    assert len(summary["aggregate_joint_pair_gradient"]) == 495
    assert summary["global_median_leave_one_out_cosine"] > 0.99
    assert routing is True
    assert scale is True
    assert compatible is True


def test_v66_classification_requires_all_predeclared_passes() -> None:
    selected = classify(True, True, True, True)
    conflict = classify(True, True, True, False)
    incoherent = classify(True, False, True, True)
    failed = classify(False, True, True, True)
    assert selected[2] is True and "final_output_layer" in selected[1]
    assert conflict[2] is False
    assert incoherent[2] is False
    assert failed[2] is False


def test_v66_result_stops_refit_and_selects_nonlocal_context_audit() -> None:
    record = json.loads((REPO / "config/hong2021_v66_result_record.json").read_text())
    assert record["audit"]["candidate_selected"] is False
    assert record["conditional_gradient_routing"][
        "conditional_routing_supported"
    ] is False
    assert record["conditional_gradient_routing"]["optimization_scale_pass"] is True
    assert record["selected_next_step"]["action"] == (
        "freeze and run a no-refit train-only target-free nonlocal context predictability audit"
    )
    assert record["firewall"]["training_or_refit_performed"] is False
    assert record["firewall"]["independent_gate_locked"] is True
