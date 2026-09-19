import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_phasec_sampler_mechanics_acf_audit_v1.json"
SOURCE = ROOT / "src/cf4_phasec_sampler_mechanics_acf_audit.py"
RUNNER = ROOT / "scripts/run_cf4_phasec_sampler_mechanics_acf_audit_v1.sbatch"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_program() -> dict:
    return json.loads(PROGRAM.read_text())


def test_offline_audit_is_explicitly_draw_only_and_fail_closed():
    program = load_program()
    auth = program["authorization"]
    assert auth["offline_existing_draws_only"] is True
    for key in (
        "new_GPU_calculation",
        "actual_observational_field_inference",
        "actual_2Mpp_count_read",
        "actual_CF4_velocity_datum_used",
        "validation_or_Phase_D",
    ):
        assert auth[key] is False
    assert program["scope_firewall"]["remaining_mock_indices_released"] is False
    assert program["analysis"] == {
        "maximum_lag": 64,
        "projection": "stored convergence_projection_samples",
        "white_probe_indices": [0, 1, 31, 32, 1024, 4096, 16384, 32767],
        "harmonic_reference": "cos(step_size * integration_steps)",
        "classification_lag1_tolerance": 0.05,
        "classification_observed_ESS_gate": 100.0,
    }


def test_input_lineage_hashes_are_frozen():
    program = load_program()
    assert program["task_keys"] == ["task_0", "task_1"]
    for key in program["task_keys"]:
        binding = program["inputs"][key]
        assert sha256(Path(binding["task_root"]) / "result.json") == binding["result_sha256"]
        assert sha256(Path(binding["task_root"]) / "diagnostics.npz") == binding["diagnostics_sha256"]
    aggregate = program["inputs"]["aggregate"]
    assert sha256(Path(aggregate["path"])) == aggregate["sha256"]
    for binding in program["lineage"].values():
        assert not str(binding.get("sha256", "")).startswith("TO_BE_FILLED")
        assert sha256(Path(binding["path"])) == binding["sha256"]


def test_independent_acf_implementation_has_no_new_sampling_or_clipping():
    source = SOURCE.read_text()
    assert "np.fft.rfft" in source
    assert "integrated_ess" in source
    assert "np.cos(trajectory_lengths)" in source
    assert "new_GPU_run_required_by_this_audit" in source
    assert "np.clip(" not in source
    assert "pgrep" not in source
    assert "renameat2" not in source
    assert "/tmp" not in source


def test_slurm_runner_is_cpu_only_and_has_memory_headroom():
    execution = load_program()["execution"]
    assert execution["requested_host_memory_MiB"] >= 1.2 * execution[
        "expected_peak_host_memory_MiB"
    ]
    runner = RUNNER.read_text()
    assert "#SBATCH --partition=a40" in runner
    assert "#SBATCH --exclude=syn06" in runner
    assert "#SBATCH --mem=1024M" in runner
    assert "#SBATCH --gres=gpu" not in runner
    assert "--output \"$OUTPUT\"" in runner
    assert "scripts/tripwire/**" in runner
    assert "pgrep" not in runner
    assert "renameat2" not in runner
    assert "/tmp" not in runner


def test_loader_rejects_mutation_and_uses_a_single_schema():
    spec = importlib.util.spec_from_file_location("acf_audit", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    program, digest = module.load_program(PROGRAM)
    assert program["schema"] == "ouruniv-cf4-phasec-sampler-mechanics-acf-audit-v1"
    assert digest == sha256(PROGRAM)
