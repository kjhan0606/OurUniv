import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_phasec_syn06_gpu_sweep_v1.json"
RUNNER = ROOT / "scripts/run_cf4_phasec_syn06_gpu_sweep_v1.sbatch"
WORKER = ROOT / "scripts/run_cf4_phasec_syn06_gpu_sweep_worker_v1.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_program():
    return json.loads(PROGRAM.read_text())


def test_sweep_binds_two_distinct_syn06_gpus_and_exact_probe_seeds():
    program = load_program()
    assert program["schema"] == "ouruniv-cf4-phasec-syn06-physical-gpu-sweep-v1"
    gang = program["gang"]
    assert gang["node"] == "syn06"
    assert gang["GPU_count"] == gang["worker_count"] == 2
    assert gang["workers_concurrent"] is True
    assert gang["one_distinct_Slurm_GPU_per_worker"] is True
    assert [(row["index"], row["seed"]) for row in program["probe_assignments_per_GPU"]] == [
        (1, 2026083001),
        (6, 2026083006),
        (2, 2026083002),
    ]


def test_lineage_hashes_and_scope_firewall():
    program = load_program()
    for binding in program["lineage"].values():
        path = Path(binding["path"])
        assert path.is_file()
        assert sha256(path) == binding["sha256"]
    assert not any(program["scope_firewall"].values())
    for key in ("sampler", "actual_observational_data", "validation_seed", "Phase_D_or_later"):
        assert program["authorization"][key] is False


def test_runner_uses_gang_srun_and_20_percent_memory_headroom():
    program = load_program()
    execution = program["execution"]
    assert execution["requested_host_memory_MiB_per_worker"] >= 1.2 * execution[
        "expected_peak_host_memory_MiB_per_worker"
    ]
    assert execution["requested_total_host_memory_MiB"] == 2 * execution[
        "requested_host_memory_MiB_per_worker"
    ]
    runner = RUNNER.read_text()
    assert "#SBATCH --nodelist=syn06" in runner
    assert "#SBATCH --gres=gpu:2" in runner
    assert "#SBATCH --mem=18432M" in runner
    assert "srun --exclusive --exact" in runner
    assert "--mem=9216M --gres=gpu:1" in runner
    assert 'wait "$pid"' in runner
    assert '"$uuid_0" != "$uuid_1"' in runner
    assert "#SBATCH --array" not in runner


def test_worker_records_device_and_runs_only_bound_mock_gate_tasks():
    worker = WORKER.read_text()
    assert "capture-device" in worker
    assert "for task_index in 1 6 2" in worker
    assert '"$gate_implementation" run' in worker
    for text in (RUNNER.read_text(), worker):
        for forbidden in ("pgrep", "renameat2", "/tmp", "counts_train", "blackjax"):
            assert forbidden not in text
