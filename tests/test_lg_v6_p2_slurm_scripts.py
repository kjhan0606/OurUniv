from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_resume_job_requests_only_large_gpu_partitions_and_backfill_resources() -> None:
    text = (REPO / "scripts/run_lg_v6_p2_resume_slurm.sh").read_text()
    assert "#SBATCH --partition=h200,h100,a100" in text
    assert "#SBATCH --cpus-per-task=16" in text
    assert "#SBATCH --mem=128G" in text
    assert "#SBATCH --time=04:00:00" in text
    assert "a40" not in text.lower()


def test_monitor_never_promotes_or_launches_ramses() -> None:
    text = (REPO / "scripts/watch_lg_v6_p2_slurm.sh").read_text()
    assert "sacct" in text
    assert "READY_FOR_PROMOTION_REVIEW" in text
    assert "no_recentered_survivor_stop_v6" in text
    assert "sbatch" not in text
    assert "RAMSES" not in text


def test_manual_tripwire_isolated_and_releases_original_on_failure() -> None:
    runner = (REPO / "scripts/run_lg_v6_p2_resume_slurm.sh").read_text()
    text = (REPO / "scripts/run_lg_v6_p2_manual_tripwire.sh").read_text()
    assert "CF4_V6_P2_DIR" in runner
    assert "scontrol hold" in text
    assert "scontrol release" in text
    assert "AllocTRES=" in text
    assert "kill -TERM -- -$remote_pgid" in text
    assert "manual_p2=$root/v3_bgc_lg_peak_p2_v6_manual_" in text
    assert "scancel \"$job_id\"" in text
    assert "RAMSES" not in text
