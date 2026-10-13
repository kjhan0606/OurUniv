from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v65_runner_is_no_refit_train_only_and_locked() -> None:
    text = (
        REPO / "scripts/hong2021_v65_structure_factorization_audit_lageunha.sh"
    ).read_text()
    lowered = text.lower()
    assert "hong2021_v65_structure_factorization_audit.py" in text
    assert "optimizer" not in lowered
    assert "development" not in lowered
    assert "eagle" not in lowered
    assert "complete_no_refit_train_only_structure_factorization_audit" in text
