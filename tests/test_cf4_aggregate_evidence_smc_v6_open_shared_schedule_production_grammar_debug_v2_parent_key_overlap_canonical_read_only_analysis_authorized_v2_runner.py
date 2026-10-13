import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v2.sbatch"
STATUS = ROOT / "scripts/status_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v2.sh"
SOURCE = ROOT / "src/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v2.py"


def test_sbatch_exact_resources_and_dev_null():
    text = RUNNER.read_text()
    required = {
        "#!/bin/bash", "#SBATCH --job-name=cf4-parent-overlap-v1", "#SBATCH --partition=debug",
        "#SBATCH --nodelist=grammar-debug", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=4", "#SBATCH --mem=8G", "#SBATCH --time=01:00:00",
        "#SBATCH --no-requeue", "#SBATCH --signal=B:TERM@60",
        "#SBATCH --chdir=/home/kjhan/BACKUP/CF4", "#SBATCH --export=NONE",
        "#SBATCH --output=/dev/null", "#SBATCH --error=/dev/null",
    }
    assert required <= set(text.splitlines())
    assert "#SBATCH --kill-on-bad-exit" not in text
    assert text.startswith("#!/bin/bash\n")
    assert "syn101" not in text and "pgrep" not in text and "tmux" not in text
    assert "squeue" not in text and "while " not in text and "sleep " not in text


def test_runner_uses_only_absolute_python_and_known_child():
    text = RUNNER.read_text()
    assert "python_bin=/home/kjhan/miniconda3/envs/circle/bin/python3.11" in text
    assert "9ee5fb16ef60eb6a53af53ae6bd300a5ac8c01d81a8c961e7cdf1497efee3ccc" in text
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() in text
    assert text.count("/usr/bin/sha256sum --check --status -") == 2
    assert '"$python_bin" -P -m "$module_name"' in text
    assert "child_pid=$!" in text and 'kill -TERM "$child_pid"' in text and 'wait "$child_pid"' in text
    assert "OURUNIV_LIFECYCLE_MODE=receipt_supervisor_only" in text
    assert "success_0" not in text
    assert "timeout_124" in text and "killed_137" in text and "terminated_143" in text and "other_nonzero" in text
    assert "PATH=/usr/bin:/bin" in text and "PYTHONNOUSERSITE=1" in text and "PYTHONSAFEPATH=1" in text
    assert re.search(r"unset .*OURUNIV_CHILD_EXIT_CLASS.*OURUNIV_LIFECYCLE_MODE", text)


def test_child_rc0_exits_directly_without_supervisor_or_status():
    text = RUNNER.read_text()
    case = text[text.index('case "$child_rc" in'):]
    assert re.search(r"\n\s*0\) exit 0 ;;", case)
    zero_branch = case[case.index("0) exit 0 ;;"):case.index("124) child_class=timeout_124")]
    assert "run_supervisor" not in zero_branch and "receipt_status" not in zero_branch
    assert case.count('run_supervisor "$child_class"') == 1


def test_signal_path_cannot_invoke_terminal_supervisor():
    text = RUNNER.read_text()
    handler = text[text.index("handle_signal() {"):text.index("trap handle_signal")]
    assert "run_supervisor" not in handler
    assert "exit 143" in handler


def test_status_is_read_only_single_shot():
    text = STATUS.read_text()
    assert text.startswith("#!/bin/bash\n")
    assert "receipt_status_only" in text
    assert "/home/kjhan/miniconda3/envs/circle/bin/python3.11 -P -m" in text
    forbidden = ("pgrep", "squeue", "while ", "sleep ", "touch ", "mkdir ", "rm ", "mv ", "chmod ")
    assert all(token not in text for token in forbidden)


def test_scripts_do_not_submit_or_retry():
    combined = RUNNER.read_text() + STATUS.read_text()
    assert "sbatch " not in combined and "srun " not in combined
    assert "retry" not in combined.lower() and "follow_on" not in combined.lower()
