from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "scripts/run_cf4_aggregate_evidence_smc_authorized_v6_open_preflight_lageunha.sh"
LAUNCH = ROOT / "scripts/launch_cf4_aggregate_evidence_smc_authorized_v6_open_preflight_lageunha.sh"
STATUS = ROOT / "scripts/status_cf4_aggregate_evidence_smc_authorized_v6_open_preflight.sh"


def test_preflight_runner_is_normalized_read_only_and_pre_runtime():
    subprocess.run(["bash", "-n", str(RUN)], check=True)
    text = RUN.read_text()
    assert "LC_ALL=C tr '[:upper:]' '[:lower:]'" in text
    assert text.index("run_preflight_v6_open") < text.index("exit 65")
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert not re.search(r"\b(?:sbatch|srun|tmux|pgrep|ssh|mkdir|touch)\b", text)
    assert "syn101" not in text and "v5" not in text and "v4" not in text


def test_preflight_launcher_and_status_have_no_remote_poll_or_runtime_io():
    subprocess.run(["bash", "-n", str(LAUNCH), str(STATUS)], check=True)
    launch = LAUNCH.read_text()
    status = STATUS.read_text()
    assert "run_preflight_v6_open" in launch and "exit 65" in launch
    assert not re.search(r"\b(?:ssh|tmux|sbatch|srun|while|until|pgrep|mkdir|touch)\b", launch)
    assert "preflight_only_not_started_fail_closed" in status and "runtime_mutation=forbidden" in status
    assert not re.search(r"\b(?:while|until|pgrep|cat|mkdir|touch)\b", status)
