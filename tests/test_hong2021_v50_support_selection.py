import hashlib
from pathlib import Path

from hong2021_v50_support_selection import PROGRAM_SHA256, select_support


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_and_firewall() -> None:
    path = REPO / "config/hong2021_v50_support_selection_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"fit_or_optimizer": false' in text
    assert '"support_adjustment_after_scan": false' in text
    assert '"development_array_access": "forbidden"' in text
    assert '"independent_gate_locked": true' in text


def test_support_uses_frozen_fraction_for_wide_range() -> None:
    result = select_support(-3.0, 7.0)
    assert result == {
        "global_minimum": -3.0,
        "global_maximum": 7.0,
        "range": 10.0,
        "symmetric_margin": 0.5,
        "lower_support": -3.5,
        "upper_support": 7.5,
    }


def test_support_uses_frozen_minimum_margin_for_narrow_range() -> None:
    result = select_support(-1.0, 1.0)
    assert result["symmetric_margin"] == 0.25
    assert result["lower_support"] == -1.25
    assert result["upper_support"] == 1.25
