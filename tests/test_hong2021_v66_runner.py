from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v66_runner_is_no_refit_train_only_and_locked() -> None:
    text = (
        REPO
        / "scripts/hong2021_v66_conditional_gradient_routing_audit_lageunha.sh"
    ).read_text()
    lowered = text.lower()
    assert "hong2021_v66_conditional_gradient_routing_audit.py" in text
    assert "optimizer" not in lowered
    assert "development" not in lowered
    assert "eagle" not in lowered
    assert "complete_no_refit_train_only_conditional_gradient_routing_audit" in text
