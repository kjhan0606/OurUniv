import numpy as np

from hong2021_v66_conditional_gradient_routing_audit import (
    classify,
    gradient_summary,
    selection_flags,
)


DOMAINS = ("TNG100", "SIMBA", "Swift")


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
