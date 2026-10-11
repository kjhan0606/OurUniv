import json
from pathlib import Path

from cf4_lg_midpoint_proposal import verify_defensive_component
from cf4_lg_peak_cr import proposal_seed_rows


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/p2_lg_z0_forward_importance_v8.json"


def test_v8_bank_matches_audited_size_and_is_fresh():
    program = json.loads(PROGRAM.read_text())
    rows = proposal_seed_rows(program)
    assert program["status"] == "frozen_authorized_before_fresh_v8_generation"
    assert len(rows) == program["authorization"]["selected_bank_size"] == 256
    assert rows[0] == (5269, 6369, 7469, 8569)
    assert rows[-1] == (5524, 6624, 7724, 8824)
    assert all(seed > 5268 for seed, _, _, _ in rows)


def test_v8_proposal_has_the_audited_defensive_bound():
    program = json.loads(PROGRAM.read_text())
    peak = program["peak_constraints"]
    bound = verify_defensive_component(
        peak["protohalo_midpoint_prior"],
        peak["protohalo_midpoint_sampling_proposal"]["components"],
        0.5,
    )
    assert bound == peak["protohalo_midpoint_sampling_proposal"][
        "analytic_target_prior_over_proposal_bound"
    ]


def test_v8_stops_before_ramses_and_keeps_all_physical_gates():
    selection = json.loads(PROGRAM.read_text())["selection_policy"]
    assert selection["same_P2_physical_support_required"]
    assert selection["all_five_P1_gates_at_the_same_actual_pair_midpoint_required"]
    assert selection["stop_before_RAMSES_for_review"]
    assert selection["no_same_model_seed_extension"]


def test_v8_embeds_the_unchanged_v7_pair_model():
    v8 = json.loads(PROGRAM.read_text())
    v7 = json.loads((
        REPO / "config/p2_lg_z0_forward_likelihood_v7_development.json"
    ).read_text())
    assert v8["candidate_preselection"] == v7["candidate_preselection"]
    assert v8["z0_likelihood"] == v7["z0_likelihood"]
