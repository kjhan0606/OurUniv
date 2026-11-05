from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_runner_stops_before_development_on_train_gate_failure() -> None:
    text = (REPO / "scripts/hong2021_v54_run_lageunha.sh").read_text()
    failure = text.index("complete_train_gate_failure")
    sampling = text.index('printf "%s\\n" sampling')
    assert failure < sampling
    assert "exit 0" in text[failure:sampling]
    assert "historical_EAGLE" not in text
