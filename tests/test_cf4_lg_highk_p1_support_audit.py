import numpy as np

from cf4_lg_highk_p1_support_audit import audit_support


def test_support_audit_forbids_zero_support_prefilter_but_retains_final_gate():
    seed = np.asarray([10, 11, 12])
    probability = np.asarray([0.7, 0.2, 0.1])
    schedule = np.asarray([10, 10, 11, 12])
    weight = np.full(4, 0.25)
    p1 = {"members": [
        {"seed": 10, "gates": {"Virgo": True, "Coma": True}, "pass": False},
        {"seed": 11, "gates": {"Virgo": False, "Coma": True}, "pass": False},
        {"seed": 12, "gates": {"Virgo": True, "Coma": True}, "pass": True},
    ]}
    result = audit_support(seed, probability, schedule[:3], weight[:3] / 0.75, p1)
    assert result["posterior_full_P1_mass"] == 0.1
    assert result["schedule_full_P1_count"] == 0
    decision = result["decision"]
    assert decision["legacy_parent_centered_P1_prefilter_authorized"] is False
    assert decision["forward_all_scheduled_states_before_pair_recentered_P1"] is True
    assert decision["pair_recentered_P1_final_environment_gate_retained"] is True


def test_support_audit_reports_gate_masses():
    seed = np.asarray([1, 2])
    probability = np.asarray([0.25, 0.75])
    schedule = np.asarray([1, 2])
    weight = np.asarray([0.5, 0.5])
    p1 = {"members": [
        {"seed": 1, "gates": {"A": True, "B": False}, "pass": False},
        {"seed": 2, "gates": {"A": False, "B": True}, "pass": False},
    ]}
    result = audit_support(seed, probability, schedule, weight, p1)
    assert result["posterior_gate_mass"] == {"A": 0.25, "B": 0.75}
    assert result["schedule_gate_mass"] == {"A": 0.5, "B": 0.5}
