import numpy as np

from cf4_lowk_cross_mode_bridge_driver_audit import driver_audit


def test_driver_audit_reaches_only_independent_audit_gate():
    base = np.full(256, 1.0 / 256)
    bridge = np.broadcast_to(base, (3, 4, 256)).copy()
    control = bridge.copy()
    for checkpoint in range(3):
        for group in range(4):
            shift = group * 0.00005
            bridge[checkpoint, group, 0] += shift
            bridge[checkpoint, group, 1] -= shift
            control[checkpoint, group, 2] += 2 * shift
            control[checkpoint, group, 3] -= 2 * shift
    result = {
        "status": "complete_diagnostic",
        "particles_per_group": 2048,
        "checkpoints": [
            {
                "bridge_cycle": cycle,
                "matched_mh_sweeps": 2 * cycle,
                "bridge_minimum_exact_overlap": overlap,
            }
            for cycle, overlap in zip((4, 8, 16), (.05, .06, .07))
        ],
    }
    analysis = {
        "status": "complete_particle_matched_read_only_analysis",
        "science_evidence": {
            "all_bridge_checkpoints_pass_particle_matched_q999": True,
            "cross_mode_transport_demonstrated": True,
        },
    }
    audit = driver_audit(
        result, analysis, bridge, control, base, np.arange(3193, 3449),
        familywise_draws=100, stationarity_draws=100, seed=7,
    )
    assert audit["decision"]["driver_science_go_for_independent_audit"] is True
    assert audit["decision"]["parent_posterior_promotion_authorized"] is False
    assert audit["decision"]["seed_selection_authorized"] is False
