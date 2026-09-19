import hashlib
import json
from pathlib import Path

from hong2021_v63_preflight import PROGRAM_SHA256, _close, count_summary


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/hong2021_v63_conditional_moment_model_program.json"


def test_program_is_byte_bound_and_single_change_is_locked() -> None:
    assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == PROGRAM_SHA256
    row = json.loads(PROGRAM.read_text())
    assert row["single_model_change"]["coefficient"] == 0.1
    assert row["single_model_change"]["all_V56_terms_retained_exactly"] is True
    assert row["firewall"]["independent_gate_locked"] is True


def test_count_summary_detects_empty_objects() -> None:
    row = count_summary([0, 2, 4, 6])
    assert row["objects"] == 4
    assert row["zero_count_objects"] == 1
    assert row["minimum"] == 0
    assert row["maximum"] == 6
    assert row["total_selected_voxels"] == 12


def test_relative_or_absolute_comparison() -> None:
    assert _close(1.0, 1.0 + 5.0e-8)
    assert _close(1.0e8, 1.0e8 + 5.0)
    assert not _close(1.0, 1.001)
