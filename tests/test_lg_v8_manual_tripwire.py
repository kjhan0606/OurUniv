from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v8_tripwire_holds_and_releases_the_original_job():
    script = (REPO / "scripts/run_lg_v8_manual_tripwire.sh").read_text()
    assert 'scontrol hold "$job_id"' in script
    assert 'scontrol release "$job_id"' in script
    assert "slurm_allocation_detected" in script
    assert "foreign_gpu_pids" in script
    assert 'kill -TERM -- -$remote_pgid' in script
    assert 'kill -KILL -- -$remote_pgid' in script


def test_v8_tripwire_preserves_interrupted_outputs_and_stops_before_ramses():
    script = (REPO / "scripts/run_lg_v8_manual_tripwire.sh").read_text()
    remote = (REPO / "scripts/run_lg_v8_manual_remote.sh").read_text()
    assert "quarantine_partial" in script
    assert "QUARANTINED_PARTIAL_RUN" in script
    assert "recoverable=true" in script
    assert "RAMSES_launched=false" in script
    assert "run_lg_v8_z0_importance_slurm.sh" in remote
