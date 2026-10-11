import json
from pathlib import Path

from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v63_preflight import PROGRAM_SHA256
from hong2021_v63_train import PREFLIGHT_RECORD_SHA256
from hong2021_v63_train_gate import _rename_model_fields, mechanism_pass


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/hong2021_v63_conditional_moment_model_program.json"
RECORD = ROOT / "config/hong2021_v63_preflight_record.json"


def test_gate_is_locked_behind_train_mechanism_pass() -> None:
    program = json.loads(PROGRAM.read_text())
    record = json.loads(RECORD.read_text())
    assert sha256_file(PROGRAM) == PROGRAM_SHA256
    assert sha256_file(RECORD) == PREFLIGHT_RECORD_SHA256
    assert record["authorization"]["train_only_gate_required_before_development"] is True
    assert program["firewall"]["development_access_before_train_gate_pass"] == (
        "forbidden"
    )
    assert program["firewall"]["historical_EAGLE_access"] == "forbidden"


def test_gate_keeps_absolute_three_domain_criterion() -> None:
    ratios = {domain: 1.0 for domain in DOMAIN_ORDER}
    convergence = {domain: 0.001 for domain in DOMAIN_ORDER}
    assert mechanism_pass(ratios, convergence) is True
    ratios["Swift"] = 0.666
    assert mechanism_pass(ratios, convergence) is False


def test_inherited_rows_are_relabeled_for_v63_only() -> None:
    source = {
        "strata": {
            "q99_9_and_above": {
                "V56_over_truth_mean_delta_squared": 1.1,
                "V54_over_truth_mean_delta_squared": 7.0,
            }
        }
    }
    row = _rename_model_fields(source)["strata"]["q99_9_and_above"]
    assert row["V63_over_truth_mean_delta_squared"] == 1.1
    assert row["V54_over_truth_mean_delta_squared"] == 7.0
    assert "V56_over_truth_mean_delta_squared" not in row
