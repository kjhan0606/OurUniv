from pathlib import Path


ROOT = Path(__file__).parents[1]
CACHE = ROOT / "scripts/run_cf4_lg_highk_covariance_cache_v1.sbatch"
PILOT = ROOT / "scripts/run_cf4_lg_highk_streaming_pm_pilot_v1.sbatch"


def test_covariance_cache_job_has_measured_plus_twenty_percent_memory_contract():
    text = CACHE.read_text()
    assert "#SBATCH --mem=19G" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "--program \"$program\"" in text
    assert "--gres=" not in text


def test_integrated_pilot_is_slurm_gpu_only_and_streaming():
    text = PILOT.read_text()
    assert "#SBATCH --partition=h200,h100,a100" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --mem=48G" in text
    assert "--mode pilot" in text
    assert "--output-root \"$output\"" in text
    assert "syn101" not in text
    assert "pgrep" not in text
    assert "tripwire" not in text


def test_both_jobs_are_fail_fast_and_program_hash_pinned():
    for path in (CACHE, PILOT):
        text = path.read_text()
        assert "set -Eeuo pipefail" in text
        assert "expected_program_sha=" in text
        assert "#SBATCH --no-requeue" in text
        assert "#SBATCH --export=NONE" in text
