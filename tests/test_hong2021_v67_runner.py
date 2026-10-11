from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v67_runner_has_no_density_refit_or_locked_data_path() -> None:
    text = (
        REPO
        / "scripts/hong2021_v67_nonlocal_context_predictability_audit_lageunha.sh"
    ).read_text()
    lowered = text.lower()
    assert "hong2021_v67_nonlocal_context_predictability_audit.py" in text
    assert "optimizer" not in lowered
    assert "development" not in lowered
    assert "eagle" not in lowered
    assert "complete_train_only_target_free_nonlocal_context_predictability_audit" in text
