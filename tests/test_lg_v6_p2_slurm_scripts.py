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
