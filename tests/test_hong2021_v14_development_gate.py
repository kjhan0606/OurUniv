from hong2021_v14_development_gate import select_candidate_rows
from hong2021_residual_evaluate import CENTERED_SCHEMAS
from hong2021_v14_edm import ENSEMBLE_SCHEMA


def test_development_selection_uses_earliest_all_domain_pass() -> None:
    rows = [
        {"step": 2000, "all_three_pass": False},
        {"step": 5000, "all_three_pass": True},
        {"step": 10000, "all_three_pass": True},
    ]
    assert select_candidate_rows(rows) == rows[1]
    assert select_candidate_rows(rows[:1]) is None


def test_v14_ensemble_uses_full_band_centered_evaluator_branch() -> None:
    assert ENSEMBLE_SCHEMA in CENTERED_SCHEMAS
