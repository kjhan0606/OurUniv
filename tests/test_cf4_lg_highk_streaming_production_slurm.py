import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
PROGRAM = ROOT / "config/cf4_lg_highk_streaming_forward_program_v1.json"
PREPARE = ROOT / "scripts/run_cf4_lg_highk_streaming_pm_production_prepare_v1.sbatch"
PRODUCTION = ROOT / "scripts/run_cf4_lg_highk_streaming_pm_production_v1.sbatch"


def test_production_authorization_keeps_downstream_closed():
    program = json.loads(PROGRAM.read_text())
    auth = program["authorization"]
    assert auth["production_256_forward_execution"] is True
    assert auth["pair_recentered_P1_aggregation_execution"] is False
    assert auth["RAMSES"] is False


def test_terminal_rule_retains_full_loose_pair_denominator_and_grouped_support():
    rule = json.loads(PROGRAM.read_text())["terminal_decision_rule"]
    assert "log(number of all loose pairs" in rule["row_log_evidence"]
    assert rule["deduplicate_repeated_posterior_rows"] is False
    assert rule["minimum_jointly_eligible_rows"] == 8
    assert rule["minimum_normalized_row_weight_ESS"] == 8.0
    assert rule["maximum_single_normalized_row_weight"] == 0.25
    assert "not 27 posterior draws" in rule["covariance_phase_interpretation"]


def test_prepare_is_small_read_only_preflight_for_exact_manifest():
    text = PREPARE.read_text()
    assert "#SBATCH --mem=1G" in text
    assert "--prepare-production" in text
    assert "--mode production" in text
    assert "expected_cache_sha=" in text
    assert "--gres=" not in text


def test_production_array_and_measured_memory_contract_are_exact():
    text = PRODUCTION.read_text()
    assert "#SBATCH --array=0-15%4" in text
    assert "#SBATCH --mem=21G" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --partition=h200,h100,a100" in text
    assert '--batch-index "$SLURM_ARRAY_TASK_ID"' in text
    assert "pgrep" not in text
    assert "tripwire" not in text
    assert "syn101" not in text
