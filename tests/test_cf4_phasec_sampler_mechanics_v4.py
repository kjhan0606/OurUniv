import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_phasec_sampler_mechanics_pilot_v4.json"
SOURCE = ROOT / "src/cf4_phasec_sampler_mechanics_pilot_v4.py"
RUNNER = ROOT / "scripts/run_cf4_phasec_sampler_mechanics_v4.sbatch"
AGGREGATOR = ROOT / "scripts/aggregate_cf4_phasec_sampler_mechanics_v4.sbatch"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_program() -> dict:
    return json.loads(PROGRAM.read_text())


def test_v4_lineage_and_scope_are_frozen():
    program = load_program()
    assert set(program["lineage"]) == {
        "base_program",
        "replacement_program",
        "generator_gate_PASS_record",
        "syn06_GPU_sweep_record",
        "previous_sampler_program",
        "previous_sampler_implementation",
        "acf_audit_record",
        "V5_implementation",
        "fixed_implementation",
        "linear_implementation",
        "sampler_mechanics_implementation",
        "failed_v2_program",
        "failed_v2_implementation",
        "failed_v2_aggregate",
        "failed_v2_task_00_result",
        "failed_v2_task_01_result",
        "failed_v3_program",
        "failed_v3_implementation",
        "failed_v3_aggregate",
        "failed_v3_task_00_result",
        "failed_v3_task_01_result",
    }
    for binding in program["lineage"].values():
        path = Path(binding["path"])
        assert path.is_file()
        assert sha256(path) == binding["sha256"]
    for key in (
        "actual_observational_field_inference",
        "actual_2Mpp_count_read",
        "actual_CF4_velocity_datum_used",
        "validation_seed_access",
        "Phase_D_or_later",
    ):
        assert program["authorization"][key] is False
    assert not any(program["scope_firewall"].values())


def test_v4_keeps_science_and_truth_contract_but_repairs_mechanics():
    program = load_program()
    assert program["assignments"] == [
        {"task_index": 0, "mock_index": 0, "seed": 2026083000, "arm": "A"},
        {"task_index": 1, "mock_index": 6, "seed": 2026083006, "arm": "D"},
    ]
    assert program["truth_integrator"]["a_nbody_maxstep"] == 1 / 256
    mechanics = program["mechanics"]
    assert mechanics["latent_dimension"] == 32792
    assert mechanics["MAP"] == {
        "optimizer": "SciPy L-BFGS-B with exact JAX gradient",
        "maximum_iterations": 1536,
        "maximum_line_search_steps": 80,
        "objective_relative_tolerance": 1e-12,
        "projected_gradient_infinity_tolerance": 0.25,
        "require_optimizer_success": True,
        "finite_value_and_gradient_required": True,
    }
    sampler = mechanics["sampler"]
    assert sampler["integration_steps"] == 32
    assert sampler["warmup_steps"] == 1024
    assert sampler["posterior_draws_per_chain"] == 2048
    assert sampler["initial_step_size"] == 0.0005
    assert sampler["target_acceptance_rate"] == 0.9
    assert sampler["chain_initial_jitter_std"] == 0.05
    assert sampler["inverse_mass_matrix"] == "identity"
    gates = program["gates"]
    assert gates["warmup_divergence_fraction_max"] == 0.01
    assert gates["MAP_optimizer_success_required"] is True
    assert gates["warmup_energy_trace_must_be_stored"] is True
    assert gates["rank_normalized_split_Rhat_max"] == 1.05
    assert gates["bulk_ESS_min"] == 100.0
    assert gates["tail_ESS_min"] == 100.0


def test_v4_persists_warmup_diagnostics_and_has_fail_closed_checks():
    source = SOURCE.read_text()
    assert "warmup_energy" in source
    assert "warmup_divergence_fraction" in source
    assert "MAP_optimizer_success" in source
    assert "warmup_energy_trace_stored" in source
    assert "np.savez_compressed" in source
    assert "sampler_warmup_energy" in source
    assert "sampler_warmup_divergence_count" in source
    assert "warmup_energy_finite_by_block" in source
    assert "warmup_divergence_fraction_by_block" in source
    assert "initial value or gradient is nonfinite" in source
    assert "warmup_step_size" in source
    assert "warmup_step_size_trace_stored" in source
    assert "sampler_sampling_step_size" in source
    assert "sampling_step_size_trace_stored" in source
    assert "first-used step size does not match frozen initial step" in source
    inherited = (ROOT / "src/cf4_phasec_sampler_mechanics_pilot.py").read_text()
    assert '"evaluated_retained_draw_count"' in inherited
    assert "np.clip(" not in source
    assert "pgrep" not in source
    assert "renameat2" not in source
    assert "/tmp" not in source


def test_v4_runner_uses_quarantined_slurm_and_memory_headroom():
    execution = load_program()["execution"]
    assert execution["requested_host_memory_MiB_per_task"] >= 1.2 * execution[
        "expected_peak_host_memory_MiB_per_task"
    ]
    assert execution["aggregate_requested_host_memory_MiB"] >= 1.2 * execution[
        "aggregate_expected_peak_host_memory_MiB"
    ]
    for path in (RUNNER, AGGREGATOR):
        text = path.read_text()
        assert "#SBATCH --partition=a40" in text
        assert "#SBATCH --exclude=syn06" in text
        assert "GPU-906578dd-9007-fdbd-3c6a-a0c5821e24d6" in text
        assert '"$host_name" == syn05 || "$host_name" == syn07' in text
        assert "scripts/tripwire/**" in text
        assert "pgrep" not in text
        assert "renameat2" not in text
        assert "/tmp" not in text
    assert "#SBATCH --array=0-1" in RUNNER.read_text()
    assert "#SBATCH --mem=10240M" in RUNNER.read_text()
    assert "#SBATCH --mem=1024M" in AGGREGATOR.read_text()
    assert "--gres=gpu:1" in RUNNER.read_text()
    assert "--gres=gpu" not in AGGREGATOR.read_text()


def test_v4_loader_validates_the_frozen_program():
    spec = importlib.util.spec_from_file_location("sampler_v4", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    program, digest, base = module.load_program(PROGRAM)
    assert program["schema"] == "ouruniv-cf4-phasec-sampler-mechanics-pilot-v4"
    assert digest == sha256(PROGRAM)
    assert base["schema"] == "ouruniv-cf4-datum-bearing-z0-phasec-program-v1"


def _load_source_module():
    spec = importlib.util.spec_from_file_location("sampler_v4_mutation_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v4_validator_rejects_npz_result_consistency_mutation(tmp_path):
    module = _load_source_module()
    root = tmp_path / "task"
    root.mkdir()
    warmup_shape = (4, 1024)
    block_shape = (4, 8)
    sample_shape = (4, 2048)
    np.savez_compressed(
        root / "diagnostics.npz",
        sampler_is_divergent=np.zeros(sample_shape, dtype=bool),
        sampler_energy=np.ones(sample_shape, dtype=np.float64),
        sampler_warmup_energy=np.ones(warmup_shape, dtype=np.float64),
        sampler_warmup_is_divergent=np.zeros(warmup_shape, dtype=bool),
        sampler_warmup_acceptance_rate=np.ones(warmup_shape, dtype=np.float64),
        sampler_warmup_step_size=np.vstack(
            [np.asarray([0.0005] + [1.0] * (warmup_shape[1] - 1), dtype=np.float64)] * 4
        ),
        sampler_sampling_step_size=np.ones(4, dtype=np.float64),
        sampler_warmup_divergence_count=np.zeros(4, dtype=int),
        sampler_warmup_energy_finite_by_block=np.ones(block_shape, dtype=bool),
        sampler_warmup_divergence_fraction_by_block=np.zeros(block_shape, dtype=np.float64),
        sampler_warmup_initialization_value=np.ones(4, dtype=np.float64),
        sampler_warmup_initialization_gradient_RMS=np.ones(4, dtype=np.float64),
    )
    result = {
        "schema": module.TASK_SCHEMA,
        "pilot_pass": True,
        "sampler": {
            "warmup_divergence_count": [0, 0, 0, 0],
            "warmup_divergence_fraction": 0.0,
            "warmup_divergence_fraction_by_block": np.zeros(block_shape).tolist(),
            "warmup_energy_finite_by_block": np.ones(block_shape, dtype=bool).tolist(),
            "warmup_initialization_value": [1.0, 1.0, 1.0, 1.0],
            "warmup_initialization_gradient_RMS": [1.0, 1.0, 1.0, 1.0],
            "step_size": [1.0, 1.0, 1.0, 1.0],
            "sampling_step_size": [1.0, 1.0, 1.0, 1.0],
            "sampling_step_size_trace_stored": True,
            "warmup_step_size_first_used": [0.0005, 0.0005, 0.0005, 0.0005],
            "warmup_step_size_last_used": [1.0, 1.0, 1.0, 1.0],
            "warmup_step_size_trace_stored": True,
            "sampling_step_size_trace_stored": True,
            "warmup_energy_trace_stored": True,
            "sampling_divergence_fraction": 0.0,
        },
        "checks": {
            "all_warmup_energies_finite": True,
            "warmup_divergence_fraction": True,
            "warmup_energy_trace_stored": True,
            "divergence_fraction": True,
            "all_sampling_energies_finite": True,
            "warmup_step_size_trace_stored": True,
            "sampling_step_size_trace_stored": True,
        },
    }
    (root / "result.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    (root / "manifest.json").write_text(
        json.dumps(module.artifact_manifest(root, "ouruniv-cf4-phasec-sampler-mechanics-v4-task-manifest-v1"), sort_keys=True)
        + "\n"
    )
    (root / "COMPLETE").write_text(
        json.dumps(
            {
                "result_sha256": sha256(root / "result.json"),
                "manifest_sha256": sha256(root / "manifest.json"),
                "pilot_pass": True,
            },
            sort_keys=True,
        )
        + "\n"
    )
    module.validate_task(root)

    mutated = np.zeros(warmup_shape, dtype=bool)
    mutated[0, 0] = True
    with np.load(root / "diagnostics.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["sampler_warmup_is_divergent"] = mutated
    np.savez_compressed(root / "diagnostics.npz", **arrays)
    (root / "manifest.json").write_text(
        json.dumps(module.artifact_manifest(root, "ouruniv-cf4-phasec-sampler-mechanics-v4-task-manifest-v1"), sort_keys=True)
        + "\n"
    )
    (root / "COMPLETE").write_text(
        json.dumps(
            {
                "result_sha256": sha256(root / "result.json"),
                "manifest_sha256": sha256(root / "manifest.json"),
                "pilot_pass": True,
            },
            sort_keys=True,
        )
        + "\n"
    )
    import pytest

    with pytest.raises(module.SamplerMechanicsV4Error, match="divergence count"):
        module.validate_task(root)

    with np.load(root / "diagnostics.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["sampler_warmup_is_divergent"] = np.zeros(warmup_shape, dtype=bool)
    arrays["sampler_warmup_step_size"][0, 0] = 1.0
    np.savez_compressed(root / "diagnostics.npz", **arrays)
    (root / "manifest.json").write_text(
        json.dumps(module.artifact_manifest(root, "ouruniv-cf4-phasec-sampler-mechanics-v4-task-manifest-v1"), sort_keys=True)
        + "\n"
    )
    (root / "COMPLETE").write_text(
        json.dumps(
            {
                "result_sha256": sha256(root / "result.json"),
                "manifest_sha256": sha256(root / "manifest.json"),
                "pilot_pass": True,
            },
            sort_keys=True,
        )
        + "\n"
    )
    with pytest.raises(module.SamplerMechanicsV4Error, match="first-used step size"):
        module.validate_task(root)

    arrays["sampler_warmup_step_size"][0, 0] = 0.0005
    arrays["sampler_sampling_step_size"][0] = 2.0
    np.savez_compressed(root / "diagnostics.npz", **arrays)
    (root / "manifest.json").write_text(
        json.dumps(module.artifact_manifest(root, "ouruniv-cf4-phasec-sampler-mechanics-v4-task-manifest-v1"), sort_keys=True)
        + "\n"
    )
    (root / "COMPLETE").write_text(
        json.dumps(
            {
                "result_sha256": sha256(root / "result.json"),
                "manifest_sha256": sha256(root / "manifest.json"),
                "pilot_pass": True,
            },
            sort_keys=True,
        )
        + "\n"
    )
    with pytest.raises(module.SamplerMechanicsV4Error, match="sampling step size"):
        module.validate_task(root)
