from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v64_runner_is_no_refit_and_has_no_development_or_eagle_path() -> None:
    text = (
        REPO / "scripts/hong2021_v64_sampler_alignment_audit_lageunha.sh"
    ).read_text()
    lowered = text.lower()
    assert "hong2021_v64_sampler_alignment_audit.py" in text
    assert "optimizer" not in lowered
    assert "development" not in lowered
    assert "eagle" not in lowered
    assert "complete_no_refit_train_only_sampler_alignment_audit" in text
