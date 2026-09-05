from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_cf4_aggregate_evidence_smc_authorized_v6_lageunha.sh"
LAUNCHER = ROOT / "scripts/launch_cf4_aggregate_evidence_smc_authorized_v6_lageunha.sh"


def test_v6_runner_fails_closed_before_reservation_and_ascii_normalizes_hostname():
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    text = RUNNER.read_text()
    assert "LC_ALL=C tr '[:upper:]' '[:lower:]'" in text
    assert '[[ "$host_short_ascii_lower" != "$expected_host" ]]' in text
    authorization = text.index("require_execution_authorization")
    assert authorization < text.index('echo "unreachable v6 execution path"')
    assert "mkdir " not in text and "flock " not in text and "tmux " not in text
    assert not re.search(r"\bssh\b", text)
    assert "aggregate_evidence_smc_v5" not in text


def test_v6_launcher_gate_precedes_any_remote_or_output_action():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text()
    assert text.index("require_execution_authorization") < text.index('echo "unreachable v6 launch path"')
    assert not re.search(r"\b(?:ssh|tmux|mkdir)\b", text)
    assert "aggregate_evidence_smc_v5" not in text
