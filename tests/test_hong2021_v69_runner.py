from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v69_runner_is_no_gradient_refit_and_has_no_locked_path() -> None:
    text = (
        REPO / "scripts/hong2021_v69_pair_estimator_rank_convergence_lageunha.sh"
    ).read_text()
    lowered = text.lower()
    assert "hong2021_v69_pair_estimator_rank_convergence.py" in text
    assert "optimizer" not in lowered
    assert "development" not in lowered
    assert "eagle" not in lowered
    assert "complete_no_refit_train_only_pair_estimator_rank_convergence_audit" in text
