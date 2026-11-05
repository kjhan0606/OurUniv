import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_cf4_aggregate_evidence_smc_authorized_v3_lageunha.sh"
LAUNCHER = ROOT / "scripts/launch_cf4_aggregate_evidence_smc_authorized_v3_lageunha.sh"
REUSED_STATUS = ROOT / "scripts/status_cf4_aggregate_evidence_smc_production.sh"


def _bash_syntax(path):
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_authorized_runner_is_fixed_lageunha_fail_closed_and_resource_bounded():
    _bash_syntax(RUNNER)
    text = RUNNER.read_text()
    assert "readonly expected_host=lageunha" in text
    assert "d3213fa8fa2effe82dc6874911d21132dc088b4b" in text
    assert "f4e282cb1fe1e80a1184ead23d1fe5892a0c7c5e" in text
    assert "cf4_aggregate_evidence_smc_execution_authorization_program_v3.json" in text
    assert "cf4_aggregate_evidence_smc_execution_grant_v3.json" in text
    assert "available_kib < 41943040" in text
    assert "memory_kib < 67108864" in text
    assert 'CUDA_VISIBLE_DEVICES=""' in text
    assert "worker_processes=8" in text
    assert "threads_per_worker=1" in text
    assert "replicates_sequential=true" in text
    assert "flock -n 9" in text
    assert "trap finish EXIT" in text
    assert not re.search(r"\b(?:pgrep|ps)\b", text)
    assert not re.search(r"\b(?:while|until)\b", text)
    assert "sleep " not in text


def test_authorization_refusal_occurs_before_resource_check_or_reservation():
    text = RUNNER.read_text()
    authorization = text.index("require_execution_authorization(program)")
    disk_check = text.index("available_kib=$(df")
    reservation = text.index('started_at=$(date --iso-8601=seconds)')
    assert authorization < disk_check < reservation
    preauthorization = text[:authorization]
    assert 'mkdir "$state"' not in preauthorization
    assert 'mkdir "$data"' not in preauthorization
    assert 'exec 9>"$lock"' not in preauthorization
    assert ': >"$log"' not in preauthorization
    assert ': >"$environment"' not in preauthorization


def test_external_release_is_snapshotted_recorded_and_revalidated_fail_closed():
    text = RUNNER.read_text()
    assert (
        "readonly release=/gpfs/kjhan/CF4/recon/linear_cr/"
        "aggregate_evidence_smc_execution_authorization_v3_release.json"
    ) in text
    first_authorization = text.index("require_execution_authorization(program)")
    snapshot = text.index('release_sha=$(sha256sum "$release"')
    reservation = text.index('started_at=$(date --iso-8601=seconds)')
    core = text.index('nice -n 5 "$python"')
    removal_gate = text.index('if [[ ! -f "$release"', core)
    mutation_gate = text.index(
        '"$(sha256sum "$release" | awk \'{print $1}\')" != "$release_sha"',
        removal_gate,
    )
    second_authorization = text.index(
        "require_execution_authorization(program)", first_authorization + 1
    )
    assert first_authorization < snapshot < reservation < core
    assert core < removal_gate <= mutation_gate < second_authorization
    assert text.count("external_lineage_release=%s") == 3
    assert text.count("external_lineage_release_sha256=%s") == 3
    assert text.count("require_execution_authorization(program)") == 2


def test_release_removal_or_mutation_after_reservation_routes_to_failed_marker():
    text = RUNNER.read_text()
    trap_index = text.index("trap finish EXIT")
    core_index = text.index('nice -n 5 "$python"')
    gate_index = text.index('if [[ ! -f "$release"', core_index)
    failure_exit = text.index(
        'echo "authorization lineage or source changed during execution"',
        gate_index,
    )
    failed_marker = text.index('mv "$marker_tmp" "$failed"')
    assert trap_index < core_index < gate_index < failure_exit
    assert 'exit 65' in text[failure_exit:failure_exit + 120]
    assert failed_marker < trap_index


def test_authorized_runner_marker_lifecycle_remains_exclusive_and_no_follow_on():
    text = RUNNER.read_text()
    assert 'readonly running="$state/RUNNING"' in text
    assert 'readonly complete="$state/COMPLETE"' in text
    assert 'readonly failed="$state/FAILED"' in text
    assert 'if (( rc == 0 )) && [[ "${validated_complete:-false}" == true' in text
    assert 'validated_complete=${validated_complete,,}' in text
    assert 'mv "$marker_tmp" "$complete"' in text
    assert 'mv "$marker_tmp" "$failed"' in text
    assert 'rm -f "$running"' in text
    assert "complete_pass_production_smc" in text
    assert "complete_scientific_fail_production_smc" in text
    assert "automatic_retry_retune_scale_up_or_follow_on=false" in text
    assert "scancel" not in text and "sbatch" not in text


def test_launcher_refuses_locally_before_ssh_tmux_or_namespace_checks():
    _bash_syntax(LAUNCHER)
    text = LAUNCHER.read_text()
    authorization = text.index("require_execution_authorization(program)")
    namespace = text.index('if [[ -e "$data" || -e "$state" ]]')
    remote = text.index("if ssh -o BatchMode=yes")
    launch = text.index("tmux new-session -d")
    assert authorization < namespace < remote < launch
    assert "readonly host=lageunha" in text
    assert "readonly session=cf4-aggregate-evidence-smc-authorized-v3" in text
    assert "run_cf4_aggregate_evidence_smc_authorized_v3_lageunha.sh" in text
    assert not re.search(r"\b(?:pgrep|ps)\b", text)
    assert "sleep " not in text


def test_existing_status_checker_is_reused_unchanged_with_fake_namespaces(tmp_path):
    _bash_syntax(REUSED_STATUS)
    assert not (
        ROOT / "scripts/status_cf4_aggregate_evidence_smc_authorized_v3.sh"
    ).exists()
    environment = dict(os.environ)
    environment.update({
        "CF4_AGGREGATE_SMC_STATUS_STATE": str(tmp_path / "state"),
        "CF4_AGGREGATE_SMC_STATUS_DATA": str(tmp_path / "data"),
    })
    status = subprocess.run(
        ["bash", str(REUSED_STATUS)],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    assert status.returncode == 3
    assert "not_started_fail_closed" in status.stdout
    assert "scripts/status_cf4_aggregate_evidence_smc_production.sh" in (
        RUNNER.read_text()
    )

