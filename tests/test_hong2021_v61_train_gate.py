import json
from pathlib import Path

from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v61_preflight import PROGRAM_SHA256
from hong2021_v61_train import PREFLIGHT_RECORD_SHA256
from hong2021_v61_train_gate import _rename_model_fields, mechanism_pass


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/hong2021_v61_reachable_support_model_program.json"
RECORD = ROOT / "config/hong2021_v61_preflight_record.json"


def test_gate_is_bound_to_passed_preflight_and_locked_before_development():
    program = json.loads(PROGRAM.read_text())
    record = json.loads(RECORD.read_text())
    assert sha256_file(PROGRAM) == PROGRAM_SHA256
    assert sha256_file(RECORD) == PREFLIGHT_RECORD_SHA256
    assert record["authorization"]["train_only_gate_required_before_development"] is True
    assert program["firewall"]["development_access_before_train_gate_pass"] == (
        "forbidden"
    )
    assert program["firewall"]["historical_EAGLE_access"] == "forbidden"


def test_gate_keeps_v56_absolute_three_domain_criterion():
    ratios = {domain: 1.0 for domain in DOMAIN_ORDER}
    convergence = {domain: 0.001 for domain in DOMAIN_ORDER}
    assert mechanism_pass(ratios, convergence) is True
    ratios["TNG100"] = 1.500001
    assert mechanism_pass(ratios, convergence) is False


def test_inherited_probe_rows_are_relabeled_without_changing_values():
    source = {
        "strata": {
            "q99_9_and_above": {
                "V56_over_truth_mean_delta_squared": 1.25,
                "V50_over_truth_mean_delta_squared": 7.0,
            }
        }
    }
    renamed = _rename_model_fields(source)
    row = renamed["strata"]["q99_9_and_above"]
    assert row["V61_over_truth_mean_delta_squared"] == 1.25
    assert row["V50_over_truth_mean_delta_squared"] == 7.0
    assert "V56_over_truth_mean_delta_squared" not in row
