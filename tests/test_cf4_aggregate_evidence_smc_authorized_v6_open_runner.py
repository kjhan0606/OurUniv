from pathlib import Path
import re,subprocess
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'scripts/run_cf4_aggregate_evidence_smc_authorized_v6_open_lageunha.sh'
LAUNCH=ROOT/'scripts/launch_cf4_aggregate_evidence_smc_authorized_v6_open_lageunha.sh'
STATUS=ROOT/'scripts/status_cf4_aggregate_evidence_smc_authorized_v6_open.sh'
def test_open_runner_is_lageunha_normalized_and_pre_reservation_fail_closed():
 subprocess.run(['bash','-n',str(RUN)],check=True)
 t=RUN.read_text(); gate=t.index('require_execution_authorization')
 assert "LC_ALL=C tr '[:upper:]' '[:lower:]'" in t and gate<t.index('exit 65')
 assert not re.search(r'\b(?:sbatch|srun|tmux|pgrep|ssh)\b',t)
 assert 'mkdir ' not in t and 'v5' not in t and 'v4' not in t
def test_open_launcher_and_status_have_no_remote_or_poll_loop():
 subprocess.run(['bash','-n',str(LAUNCH),str(STATUS)],check=True)
 assert not re.search(r'\b(?:ssh|tmux|sbatch|srun|while|until|pgrep)\b',LAUNCH.read_text())
 status=STATUS.read_text(); assert 'not_started_fail_closed' in status and not re.search(r'\b(?:while|until|pgrep)\b',status)
