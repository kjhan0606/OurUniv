import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_phasec_sampler_mechanics_pilot_v1.json"
SOURCE = ROOT / "src/cf4_phasec_sampler_mechanics_pilot.py"
RUNNER = ROOT / "scripts/run_cf4_phasec_sampler_mechanics_v1.sbatch"
AGGREGATOR = ROOT / "scripts/aggregate_cf4_phasec_sampler_mechanics_v1.sbatch"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_program() -> dict:
    return json.loads(PROGRAM.read_text())


def load_module():
    spec = importlib.util.spec_from_file_location("sampler_mechanics", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lineage_is_complete_and_hash_frozen():
    program = load_program()
    assert set(program["lineage"]) == {
        "base_program",
        "replacement_program",
        "generator_gate_PASS_record",
        "syn06_GPU_sweep_record",
        "V5_implementation",
        "scan_generator_implementation",
        "fixed_implementation",
        "linear_implementation",
        "sampler_mechanics_implementation",
    }
    for binding in program["lineage"].values():
        path = Path(binding["path"])
        assert path.is_file()
        assert sha256(path) == binding["sha256"]
    generator = json.loads(
        Path(program["lineage"]["generator_gate_PASS_record"]["path"]).read_text()
    )
    assert generator["summary"]["all_seed_gate_pass"] is True
    assert generator["summary"]["passing_seed_count"] == 8


def test_only_predeclared_mock_mechanics_indices_are_authorized():
    program = load_program()
    assert program["assignments"] == [
        {"task_index": 0, "mock_index": 0, "seed": 2026083000, "arm": "A"},
        {"task_index": 1, "mock_index": 6, "seed": 2026083006, "arm": "D"},
    ]
    authorization = program["authorization"]
    for key in (
        "actual_observational_field_inference",
        "actual_2Mpp_count_read",
        "actual_CF4_velocity_datum_used",
        "validation_seed_access",
        "Phase_D_or_later",
    ):
        assert authorization[key] is False
    assert not any(program["scope_firewall"].values())
    assert program["decision_rule"]["remaining_mock_sampler_indices_after_PASS"] == [
        1,
        2,
        3,
        4,
        5,
        7,
    ]


def test_standardized_identity_hmc_contract_is_exact():
    mechanics = load_program()["mechanics"]
    assert mechanics["latent_dimension"] == 32792
    assert mechanics["parameterization"] == (
        "all 32792 coordinates are standard normal before the likelihood"
    )
    assert mechanics["MAP"] == {
        "optimizer": "SciPy L-BFGS-B with exact JAX gradient",
        "maximum_iterations": 256,
        "maximum_line_search_steps": 40,
        "objective_relative_tolerance": 1e-10,
        "finite_value_and_gradient_required": True,
    }
    sampler = mechanics["sampler"]
    assert sampler["chain_count"] == 4
    assert sampler["warmup_steps"] == 512
    assert sampler["posterior_draws_per_chain"] == 512
    assert sampler["integration_steps"] == 12
    assert sampler["inverse_mass_matrix"] == "identity"
    assert sampler["adaptation"] == "dual-averaging step-size only"
    source = SOURCE.read_text()
    assert "standard_to_physical" in source
    assert "identity_inverse_mass = jnp.ones" in source
    assert "dual_averaging_adaptation" in source
    assert "window_adaptation" not in source


def test_derived_state_and_convergence_gates_are_fail_closed():
    program = load_program()
    gates = program["gates"]
    assert gates == {
        "MAP_gradient_RMS_max": 0.25,
        "derived_count_intensity_max": 1000000.0,
        "rank_normalized_split_Rhat_max": 1.05,
        "bulk_ESS_min": 100.0,
        "tail_ESS_min": 100.0,
        "divergence_fraction_max": 0.01,
        "all_draws_and_energies_finite": True,
        "all_derived_count_intensities_finite_and_nonnegative": True,
        "clipping_allowed": False,
    }
    source = SOURCE.read_text()
    assert '"expected_retained_draw_count": 2048' in source
    assert '"posterior_predictive_RNG_called": False' in source
    assert "np.clip(" not in source
    assert "jnp.clip(" not in source
    assert '"remaining_mock_sampler_indices_allowed": (' in source
    assert 'if aggregate_pass else []' in source


def test_truth_integrator_is_the_generator_gate_production_member():
    integrator = load_program()["truth_integrator"]
    assert integrator == {
        "lpt_order": 2,
        "a_start": 0.015625,
        "a_stop": 1.0,
        "a_lpt_maxstep": 1 / 128,
        "a_nbody_maxstep": 1 / 256,
        "mesh_to_particle_ratio": 1,
        "float_dtype": "float64",
    }
    source = SOURCE.read_text()
    assert "jax.lax.scan" in source
    assert "nbody_step" in source


def test_program_loader_and_task_names_are_pure_and_frozen():
    module = load_module()
    controller, program_hash, base = module.load_program(PROGRAM)
    assert program_hash == sha256(PROGRAM)
    assert base["schema"] == "ouruniv-cf4-datum-bearing-z0-phasec-program-v1"
    assert module.task_name(controller["assignments"][0]) == (
        "mechanics_00_mock_00_seed_2026083000_arm_A"
    )
    assert module.task_name(controller["assignments"][1]) == (
        "mechanics_01_mock_06_seed_2026083006_arm_D"
    )


def test_slurm_hardware_quarantine_and_memory_headroom():
    program = load_program()
    quarantine = program["hardware_quarantine"]
    assert quarantine["excluded_node"] == "syn06"
    assert quarantine["expected_nodes"] == ["syn05", "syn07"]
    assert quarantine["known_failing_UUID"] == (
        "GPU-906578dd-9007-fdbd-3c6a-a0c5821e24d6"
    )
    execution = program["execution"]
    assert execution["requested_host_memory_MiB_per_task"] >= 1.2 * execution[
        "expected_peak_host_memory_MiB_per_task"
    ]
    assert execution["aggregate_requested_host_memory_MiB"] >= 1.2 * execution[
        "aggregate_expected_peak_host_memory_MiB"
    ]
    runner = RUNNER.read_text()
    aggregator = AGGREGATOR.read_text()
    assert "#SBATCH --array=0-1" in runner
    assert "#SBATCH --mem=10240M" in runner
    assert "#SBATCH --mem=1024M" in aggregator
    for text in (runner, aggregator):
        assert "#SBATCH --partition=a40" in text
        assert "#SBATCH --exclude=syn06" in text
        assert "GPU-906578dd-9007-fdbd-3c6a-a0c5821e24d6" in text
        assert '"$host_name" == syn05 || "$host_name" == syn07' in text
        assert "scripts/tripwire/**" in text
        for forbidden in ("pgrep", "renameat2", "/tmp"):
            assert forbidden not in text
    assert "capture-device" in runner
    assert '--device-record "$device_task_root/device.json"' in runner
    assert "--implementation-commit \"$EXPECTED_COMMIT\"" in runner
    assert "--implementation-commit \"$EXPECTED_COMMIT\"" in aggregator
