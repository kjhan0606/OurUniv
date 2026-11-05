from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "scripts/run_cf4_aggregate_evidence_smc_authorized_v6_open_pilot_execution_lageunha.sh"
LAUNCH = ROOT / "scripts/launch_cf4_aggregate_evidence_smc_authorized_v6_open_pilot_execution_lageunha.sh"
STATUS = ROOT / "scripts/status_cf4_aggregate_evidence_smc_authorized_v6_open_pilot_execution.sh"


def test_runner_enforces_host_resources_environment_lineage_and_preflight():
    subprocess.run(["bash", "-n", str(RUN), str(LAUNCH), str(STATUS)], check=True)
    text = RUN.read_text()
    for value in ("OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1", 'CUDA_VISIBLE_DEVICES=""', "PYTHONNOUSERSITE=1", "PYTHONSAFEPATH=1", "MALLOC_ARENA_MAX=2", "MemAvailable", "67108864", "nproc", "ls-remote", "--untracked-files=all", "-P -m", "--preflight", "--run"):
        assert value in text
    assert "LC_ALL=C tr '[:upper:]' '[:lower:]'" in text
    assert not re.search(r"\b(?:sbatch|srun|ssh|tmux|pgrep|while|until)\b", text)
    assert "syn101" not in text and "v5" not in text and "v4" not in text


def test_launcher_local_and_status_marker_fail_closed():
    launch, status = LAUNCH.read_text(), STATUS.read_text()
    assert 'exec "$runner"' in launch
    assert not re.search(r"\b(?:ssh|tmux|sbatch|srun|pgrep|while|until)\b", launch)
    for value in ("RUNNING", "COMPLETE", "FAILED", "invalid_marker_count", "pilot_not_started_fail_closed", "forbidden_production_namespace_present", "manifest_sha256", "0o444"):
        assert value in status
    assert not re.search(r"\b(?:pgrep|while|until)\b", status)


def _temporary_status_script(tmp_path):
    roots = {name: tmp_path / name for name in ("receipts", "pilot", "data", "state")}
    text = STATUS.read_text()
    text = text.replace("/home/kjhan/miniconda3/envs/circle/bin/python", sys.executable)
    replacements = {
        "/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_receipts": roots["receipts"],
        "/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_disposable_pilot": roots["pilot"],
        "/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_run": roots["state"],
        "/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open": roots["data"],
    }
    for source, target in replacements.items():
        text = text.replace(source, str(target))
    script = tmp_path / "status.sh"
    script.write_text(text); script.chmod(0o755)
    return script, roots


def test_status_rejects_forbidden_namespace_and_invalid_marker_json(tmp_path):
    script, roots = _temporary_status_script(tmp_path)
    roots["data"].mkdir()
    result = subprocess.run([str(script)], text=True, capture_output=True)
    assert result.returncode == 65 and "forbidden_production_namespace_present" in result.stdout
    roots["data"].rmdir()
    marker = roots["receipts"] / ("a" * 64) / "pilot" / "RUNNING"
    marker.parent.mkdir(parents=True); marker.write_text("not-json"); marker.chmod(0o444)
    result = subprocess.run([str(script)], text=True, capture_output=True)
    assert result.returncode == 65 and "invalid_marker_json" in result.stdout


def test_status_validates_complete_manifest_mode_and_hash(tmp_path):
    script, roots = _temporary_status_script(tmp_path)
    authorization_id = "b" * 64
    receipt = roots["receipts"] / authorization_id / "pilot"
    output = roots["pilot"] / authorization_id
    receipt.mkdir(parents=True); output.mkdir(parents=True)
    manifest = {"status": "complete_disposable_pilot_schedule_only", "authorization_id": authorization_id, "schedule": {"sha256": "c" * 64}}
    encoded = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    for path in (receipt / "schedule_manifest.json", output / "schedule_manifest.json"):
        path.write_bytes(encoded); path.chmod(0o444)
    (receipt / "COMPLETE").write_text(json.dumps({"status": "complete_disposable_pilot_schedule_only", "schedule_sha256": "c" * 64, "manifest_sha256": digest})); (receipt / "COMPLETE").chmod(0o444)
    result = subprocess.run([str(script)], text=True, capture_output=True)
    assert result.returncode == 0 and "status=pilot_complete" in result.stdout
    (output / "schedule_manifest.json").chmod(0o600)
    result = subprocess.run([str(script)], text=True, capture_output=True)
    assert result.returncode == 65 and "invalid_complete_manifest" in result.stdout
