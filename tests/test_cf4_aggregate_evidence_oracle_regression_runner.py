import re
from pathlib import Path


INSIDE_SELECTION_SHA256 = (
    "d91f20f27fb2f43df781f79b2e9f89bd9c29f802a41a107ca69ab5d375955b13"
)
OUTSIDE_SELECTION_SHA256 = (
    "3b5ab76f91b008a8671b33a46f33f36f4788251bebe53e35ce784cabc031aefc"
)


def test_runner_uses_complete_frozen_selection_hashes():
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_cf4_aggregate_evidence_oracle_regression_lageunha.sh"
    ).read_text()
    matches = dict(re.findall(
        r'(inside|outside)_selection_sha\s*!=\s*"([0-9a-f]+)"',
        runner,
    ))

    assert matches == {
        "inside": INSIDE_SELECTION_SHA256,
        "outside": OUTSIDE_SELECTION_SHA256,
    }
    assert all(len(value) == 64 for value in matches.values())
