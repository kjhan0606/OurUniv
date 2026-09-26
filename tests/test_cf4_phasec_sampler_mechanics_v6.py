import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_phasec_sampler_mechanics_pilot_v6.json"
SOURCE = ROOT / "src/cf4_phasec_sampler_mechanics_pilot_v6.py"
RUNNER = ROOT / "scripts/run_cf4_phasec_sampler_mechanics_v6.sbatch"
AGGREGATOR = ROOT / "scripts/aggregate_cf4_phasec_sampler_mechanics_v6.sbatch"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_program() -> dict:
    return json.loads(PROGRAM.read_text())


def test_v6_lineage_and_scope_are_frozen():
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
        "failed_v4_program",
        "failed_v4_implementation",
        "failed_v4_aggregate",
        "failed_v4_task_00_result",
        "failed_v4_task_01_result",
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


def test_v6_keeps_science_and_truth_contract_but_repairs_mechanics():
    program = load_program()
    assert program["assignments"] == [
        {"task_index": 0, "mock_index": 1, "seed": 2026083001, "arm": "A"},
        {"task_index": 1, "mock_index": 2, "seed": 2026083002, "arm": "B"},
        {"task_index": 2, "mock_index": 3, "seed": 2026083003, "arm": "B"},
        {"task_index": 3, "mock_index": 4, "seed": 2026083004, "arm": "C"},
        {"task_index": 4, "mock_index": 5, "seed": 2026083005, "arm": "C"},
        {"task_index": 5, "mock_index": 7, "seed": 2026083007, "arm": "D"},
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
    assert sampler["maximum_step_size"] == 0.04
    assert sampler["inverse_mass_matrix"] == "identity"
    gates = program["gates"]
    assert gates["warmup_divergence_fraction_max"] == 0.01
    assert gates["MAP_optimizer_success_required"] is True
    assert gates["warmup_energy_trace_must_be_stored"] is True
    assert gates["rank_normalized_split_Rhat_max"] == 1.05
    assert gates["bulk_ESS_min"] == 100.0
    assert gates["tail_ESS_min"] == 100.0


def test_v6_persists_warmup_diagnostics_and_has_fail_closed_checks():
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
    assert "warmup_raw_step_size" in source
    assert "warmup_used_step_size" in source
    assert "warmup_cap_applied" in source
    assert "sampling_raw_step_size" in source
    assert "sampling_used_step_size" in source
    assert "sampling_cap_applied" in source
    assert "raw/used step-size" in source
    inherited = (ROOT / "src/cf4_phasec_sampler_mechanics_pilot.py").read_text()
    assert '"evaluated_retained_draw_count"' in inherited
    assert "np.clip(" not in source
    assert "pgrep" not in source
    assert "renameat2" not in source
    assert "/tmp" not in source


def test_v6_runner_uses_quarantined_slurm_and_memory_headroom():
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
    assert "#SBATCH --array=0-5" in RUNNER.read_text()
    assert "#SBATCH --mem=10240M" in RUNNER.read_text()
    assert "#SBATCH --mem=1024M" in AGGREGATOR.read_text()
    assert "--gres=gpu:1" in RUNNER.read_text()
    assert "--gres=gpu" not in AGGREGATOR.read_text()


def test_v6_loader_validates_the_frozen_program():
    spec = importlib.util.spec_from_file_location("sampler_v6", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    program, digest, base = module.load_program(PROGRAM)
    assert program["schema"] == "ouruniv-cf4-phasec-sampler-mechanics-pilot-v6"
    assert digest == sha256(PROGRAM)
    assert base["schema"] == "ouruniv-cf4-datum-bearing-z0-phasec-program-v1"


def _load_source_module():
    spec = importlib.util.spec_from_file_location("sampler_v6_mutation_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v6_validator_rejects_raw_used_cap_and_binding_mutations(tmp_path):
    import pytest

    module = _load_source_module()
    root = tmp_path / "task"
    root.mkdir()
    warmup_shape, block_shape, sample_shape = (4, 1024), (4, 8), (4, 2048)
    warmup_raw = np.full(warmup_shape, 0.02, dtype=np.float64)
    warmup_raw[:, 0] = 0.0005
    warmup_raw[1, 33] = 0.06
    warmup_used = np.minimum(warmup_raw, 0.04)
    warmup_mask = warmup_raw > 0.04
    sampling_raw = np.asarray([0.01, 0.04, 0.06, 0.02], dtype=np.float64)
    sampling_used = np.minimum(sampling_raw, 0.04)
    sampling_mask = sampling_raw > 0.04
    arrays = {
        "sampler_is_divergent": np.zeros(sample_shape, dtype=bool),
        "sampler_energy": np.ones(sample_shape, dtype=np.float64),
        "sampler_warmup_energy": np.ones(warmup_shape, dtype=np.float64),
        "sampler_warmup_is_divergent": np.zeros(warmup_shape, dtype=bool),
        "sampler_warmup_acceptance_rate": np.ones(warmup_shape, dtype=np.float64),
        "sampler_warmup_raw_step_size": warmup_raw,
        "sampler_warmup_used_step_size": warmup_used,
        "sampler_warmup_cap_applied": warmup_mask,
        "sampler_sampling_raw_step_size": sampling_raw,
        "sampler_sampling_used_step_size": sampling_used,
        "sampler_sampling_cap_applied": sampling_mask,
        "sampler_warmup_divergence_count": np.zeros(4, dtype=int),
        "sampler_warmup_energy_finite_by_block": np.ones(block_shape, dtype=bool),
        "sampler_warmup_divergence_fraction_by_block": np.zeros(block_shape, dtype=np.float64),
        "sampler_warmup_initialization_value": np.ones(4, dtype=np.float64),
        "sampler_warmup_initialization_gradient_RMS": np.ones(4, dtype=np.float64),
    }
    result = {
        "schema": module.TASK_SCHEMA,
        "pilot_pass": True,
        "sampler": {
            "maximum_step_size": 0.04,
            "warmup_divergence_count": [0] * 4,
            "warmup_divergence_fraction": 0.0,
            "warmup_divergence_fraction_by_block": np.zeros(block_shape).tolist(),
            "warmup_energy_finite_by_block": np.ones(block_shape, dtype=bool).tolist(),
            "warmup_initialization_value": [1.0] * 4,
            "warmup_initialization_gradient_RMS": [1.0] * 4,
            "warmup_raw_step_size_first": warmup_raw[:, 0].tolist(),
            "warmup_raw_step_size_last": warmup_raw[:, -1].tolist(),
            "warmup_used_step_size_first": warmup_used[:, 0].tolist(),
            "warmup_used_step_size_last": warmup_used[:, -1].tolist(),
            "warmup_cap_applied_count": [0, 1, 0, 0],
            "warmup_raw_used_step_size_contract": True,
            "sampling_raw_step_size": sampling_raw.tolist(),
            "sampling_used_step_size": sampling_used.tolist(),
            "sampling_cap_applied": sampling_mask.tolist(),
            "sampling_cap_applied_count": 1,
            "sampling_raw_used_step_size_contract": True,
            "warmup_energy_trace_stored": True,
            "sampling_divergence_fraction": 0.0,
        },
        "checks": {
            "all_warmup_energies_finite": True,
            "warmup_divergence_fraction": True,
            "warmup_energy_trace_stored": True,
            "divergence_fraction": True,
            "all_sampling_energies_finite": True,
            "warmup_raw_used_step_size_contract": True,
            "sampling_raw_used_step_size_contract": True,
        },
    }

    def write_artifact(payload):
        np.savez_compressed(root / "diagnostics.npz", **payload)
        (root / "result.json").write_text(json.dumps(result, sort_keys=True) + "\n")
        (root / "manifest.json").write_text(
            json.dumps(
                module.artifact_manifest(root, "ouruniv-cf4-phasec-sampler-mechanics-v6-task-manifest-v1"),
                sort_keys=True,
            )
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

    def clone():
        return {name: np.array(value, copy=True) for name, value in arrays.items()}

    write_artifact(clone())
    module.validate_task(root)

    bad = clone()
    bad["sampler_warmup_used_step_size"][1, 33] = 0.06
    write_artifact(bad)
    with pytest.raises(module.SamplerMechanicsV5Error, match="raw/used|cap-applied"):
        module.validate_task(root)

    bad = clone()
    bad["sampler_warmup_cap_applied"][1, 33] = False
    write_artifact(bad)
    with pytest.raises(module.SamplerMechanicsV5Error, match="raw/used|cap-applied"):
        module.validate_task(root)

    bad = clone()
    bad["sampler_warmup_raw_step_size"][1, 33] = 0.03
    bad["sampler_warmup_used_step_size"][1, 33] = 0.03
    bad["sampler_warmup_cap_applied"][1, 33] = False
    write_artifact(bad)
    with pytest.raises(module.SamplerMechanicsV5Error, match="cap-applied counts"):
        module.validate_task(root)

    for value in (np.nan, np.inf, -0.01):
        bad = clone()
        bad["sampler_warmup_raw_step_size"][0, 12] = value
        write_artifact(bad)
        with pytest.raises(module.SamplerMechanicsV5Error, match="raw/used"):
            module.validate_task(root)

    bad = clone()
    bad["sampler_sampling_raw_step_size"][2] = 0.02
    bad["sampler_sampling_used_step_size"][2] = 0.02
    bad["sampler_sampling_cap_applied"][2] = False
    write_artifact(bad)
    with pytest.raises(module.SamplerMechanicsV5Error, match="sampling_raw_step_size"):
        module.validate_task(root)
