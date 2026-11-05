import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).parents[1]
PROGRAM = ROOT / "config/cf4_lg_highk_streaming_forward_program_v1.json"
PREPARE = ROOT / "scripts/run_cf4_lg_highk_streaming_pm_production_prepare_v1.sbatch"
PRODUCTION = ROOT / "scripts/run_cf4_lg_highk_streaming_pm_production_v1.sbatch"
RECOVERY = ROOT / "scripts/run_cf4_lg_highk_streaming_pm_production_recovery_v1.sbatch"
CHECK = ROOT / "scripts/check_cf4_lg_highk_streaming_pm_production_v1.py"
GPU_GUARD = ROOT / "scripts/cf4_lg_highk_recovery_gpu_guard.py"


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


def test_recovery_targets_only_failed_batches_on_h200_with_memory_guard():
    text = RECOVERY.read_text()
    assert "#SBATCH --partition=h200" in text
    assert "#SBATCH --array=1,2,4-15%1" in text
    assert "#SBATCH --mem=21G" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert '"$gpu_guard" --minimum-mib 115000' in text
    assert "expected_gpu_guard_sha=" in text
    assert "JAX_PLATFORMS=cuda" in text
    assert '--batch-index "$SLURM_ARRAY_TASK_ID"' in text
    assert "lg_highk_streaming_pm_production_v1_recovery_slurm-%A_%a.log" in text
    assert "lg_highk_streaming_pm_production_v1\n" in text
    assert "pgrep" not in text
    assert "tripwire" not in text
    assert "syn101" not in text
    expected_guard_sha = next(
        line.split("=", 1)[1] for line in text.splitlines()
        if line.startswith("readonly expected_gpu_guard_sha=")
    )
    assert hashlib.sha256(GPU_GUARD.read_bytes()).hexdigest() == expected_guard_sha


def test_recovery_preserves_frozen_runner_and_canonical_output_namespace():
    recovery = RECOVERY.read_text()
    production = PRODUCTION.read_text()
    shared_lines = (
        "readonly program=",
        "readonly cache=",
        "readonly output=",
        "readonly expected_program_sha=",
        "readonly expected_cache_sha=",
    )
    for prefix in shared_lines:
        assert next(line for line in recovery.splitlines() if line.startswith(prefix)) == next(
            line for line in production.splitlines() if line.startswith(prefix)
        )
    assert "cf4_lg_highk_streaming_forward.py\"]=0494bdf" in recovery
    assert "--prepare-production" not in recovery
    assert "rm " not in recovery
    assert "mv " not in recovery


def test_integrity_checker_is_read_only_and_supports_terminal_completeness_gate():
    text = CHECK.read_text()
    assert "_valid_completed_row" in text
    assert "validate_production_run_manifest" in text
    assert "validate_program_inputs" in text
    assert "validate_canonical_rows" in text
    assert "_validate_halo_catalogue" in text
    assert 'parser.add_argument("--require-complete", action="store_true")' in text
    assert "missing_batches" in text
    assert "write_text(" not in text
    assert "unlink(" not in text
    assert "replace(" not in text
    assert "rmtree(" not in text
