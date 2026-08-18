from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "scripts/run_cf4_aggregate_evidence_smc_authorized_v6_open_pilot_lageunha.sh"
LAUNCH = ROOT / "scripts/launch_cf4_aggregate_evidence_smc_authorized_v6_open_pilot_lageunha.sh"
STATUS = ROOT / "scripts/status_cf4_aggregate_evidence_smc_authorized_v6_open_pilot.sh"


def test_pilot_runner_has_lageunha_gate_and_unconditional_fail_closed_exit():
    subprocess.run(["bash", "-n", str(RUN)], check=True)
    text = RUN.read_text()
    assert "LC_ALL=C tr '[:upper:]' '[:lower:]'" in text
    assert text.index("run_authorized_v6_open_pilot") < text.rindex("exit 65")
    assert "set +e" in text and "gate_rc" in text
    assert not re.search(r"\b(?:sbatch|srun|ssh|tmux|pgrep|mkdir|touch|while|until)\b", text)
    assert "syn101" not in text and "v5" not in text and "v4" not in text


def test_pilot_launcher_and_status_are_non_mutating_and_non_polling():
    subprocess.run(["bash", "-n", str(LAUNCH), str(STATUS)], check=True)
    launch, status = LAUNCH.read_text(), STATUS.read_text()
    assert "run_authorized_v6_open_pilot" in launch and launch.rstrip().endswith("exit 65")
    assert not re.search(r"\b(?:sbatch|srun|ssh|tmux|pgrep|mkdir|touch|while|until)\b", launch)
    assert "pilot_execution_not_authorized_fail_closed" in status and "absent" in status
