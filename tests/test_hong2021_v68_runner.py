from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v68_runner_is_no_refit_and_has_no_locked_data_path() -> None:
    text = (
        REPO / "scripts/hong2021_v68_population_pair_objective_audit_lageunha.sh"
    ).read_text()
    lowered = text.lower()
    assert "hong2021_v68_population_pair_objective_audit.py" in text
    assert "optimizer" not in lowered
    assert "development" not in lowered
    assert "eagle" not in lowered
    assert "complete_no_refit_train_only_population_pair_objective_audit" in text
