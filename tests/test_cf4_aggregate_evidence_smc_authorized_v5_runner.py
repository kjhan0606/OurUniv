from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_cf4_aggregate_evidence_smc_authorized_v5_lageunha.sh"
LAUNCHER = ROOT / "scripts/launch_cf4_aggregate_evidence_smc_authorized_v5_lageunha.sh"


def test_v5_runner_is_syntax_valid_isolated_and_execution_false_by_default():
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    text = RUNNER.read_text()
    assert "aggregate_evidence_smc_v5" in text
    assert "aggregate_evidence_smc_v5_run" in text
    assert "aggregate_evidence_smc_v5_receipts" in text
    assert "aggregate_evidence_smc_execution_grant_v5.json" in text
    assert "require_execution_authorization" in text
    assert "create_preflight_receipt" in text
    assert text.count("revalidate_preflight_receipt") >= 2
    assert "release.anchor" in text
    assert "v4" not in text


def test_v5_runner_records_snapshot_in_running_environment_and_complete_contract():
    text = RUNNER.read_text()
    assert "preflight_snapshot_sha256" in text
    assert "stage=aggregate_evidence_smc_authorized_v5" in text
    assert "failure_class=invalid_provenance_or_execution" in text
    assert "trap finish EXIT" in text
    assert "read_only_science_postcheck" in text
    assert "validate_published_bundle" not in text  # only the guarded helper owns the check
    assert "complete_pass_production_smc" in text
    assert "complete_scientific_fail_production_smc" in text
    assert "result_sha256" in text and "manifest_sha256" in text
    assert text.index("read_only_science_postcheck") < text.index("validated_complete=true")
    assert text.count("revalidate_preflight_receipt") >= 4


def test_v5_runner_has_no_unchecked_complete_or_completion_masquerade_path():
    text = RUNNER.read_text()
    completion = text[text.index("finish() {"):text.index("host=$(hostname)")]
    assert '[[ "${validated_complete:-false}" == true ]]' in completion
    assert "science_status=%s" in completion
    assert "outcome_kind=%s" in completion
    assert "failure_class=%s" in completion
    assert "result_sha256=%s" in completion and "manifest_sha256=%s" in completion
    assert "status=failed" in completion


def test_v5_launcher_refuses_before_any_remote_or_namespace_action():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text()
    assert text.index("require_execution_authorization") < text.index('if [[ -e "$data"')
    assert text.index("require_execution_authorization") < text.index("if ssh")
    assert "cf4-aggregate-evidence-smc-authorized-v5" in text
    assert "v4" not in text
