import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_datum_bearing_z0_phaseb_smoke_program_v1.json"
SOURCE = ROOT / "src/cf4_datum_bearing_z0_phaseb_smoke.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    return json.loads(PROGRAM.read_text())


def test_phase_b_scope_is_matched_mock_and_fail_closed():
    program = load()
    authorization = program["authorization"]
    assert authorization["Phase_B_matched_mock_smoke"] is True
    assert authorization["single_Slurm_GPU_submission"] is True
    for key in (
        "actual_observational_field_inference",
        "new_truth_seed",
        "validation_seed_access",
        "Phase_C_or_later",
        "automatic_follow_on",
        "IC_PM_HOP_RAMSES",
    ):
        assert authorization[key] is False
    assert program["mock"]["truth_seed_reused"] == 2026083000
    assert program["mock"]["new_truth_seed_count"] == 0


def test_actual_count_arrays_are_not_read_by_phase_b():
    source = SOURCE.read_text()
    assert 'datum["raw_selection_exposure"]' in source
    for forbidden in ('datum["counts_all"]', 'datum["counts_train"]', 'datum["counts_holdout"]'):
        assert forbidden not in source
    assert load()["input_bindings"]["Phase_A_datum"]["read_contract"].startswith(
        "read raw_selection_exposure only"
    )


def test_external_population_prior_is_not_observed_total_normalization():
    prior = load()["external_population_prior"]
    assert prior["published_cell_size_cMpc_h"] == 600.0 / 256.0
    assert prior["observed_population_totals_used_to_center_prior"] is False
    assert prior["ARES_example_nmean_equals_one_used_as_physical_prior"] is False
    assert len(prior["published_mean_count_per_original_voxel"]) == 6
    assert len(prior["published_bias"]) == 6
    assert min(prior["published_mean_count_per_original_voxel"]) > 0
    assert min(prior["published_bias"]) > 0


def test_spherical_rsd_gauge_selection_and_boundary_are_explicit():
    model = load()["model"]
    assert "mean(T*w)" in model["eta_gauge"]
    assert "observer-centred" in model["line_of_sight"]
    assert "plane parallel is forbidden" in model["line_of_sight"]
    assert "CIC" in model["deposition"]
    assert "after" in model["selection_application"]
    assert "12 cMpc/h" in model["boundary"]
    source = SOURCE.read_text()
    assert "radial_velocity / 100.0" in source
    assert "jax.vjp" in source


def test_numerical_gates_and_mechanics_are_frozen_before_execution():
    program = load()
    assert program["gates"] == {
        "radial_forward_relative_error_max": 5e-08,
        "rsd_mass_relative_error_max": 2e-13,
        "rsd_adjoint_relative_error_max": 2e-12,
        "directional_gradient_relative_error_max": 0.0002,
        "optimizer_relative_decrease_min": 1e-05,
        "boundary_max_displacement_cMpc_h_strict": 12.0,
        "prior_dominated_information_fraction_max": 0.01,
    }
    mechanics = program["mechanics"]
    assert mechanics["directional_probe_count"] == 3
    assert mechanics["HMC_transition_count"] == 4
    assert mechanics["HMC_semantics"].startswith("mechanics-only")


def test_resource_headroom_and_syntax_policy():
    execution = load()["execution"]
    assert execution["controller"] == "syntax_submit_and_audit_only"
    assert execution["backend"] == "Slurm_single_GPU"
    assert execution["requested_memory_MiB"] >= 1.2 * execution["expected_peak_memory_MiB"]
    assert execution["manual_numerical_execution_on_syntax_or_syn101_allowed"] is False
    assert execution["maximum_submissions"] == 1
    assert execution["monitoring_loop_allowed"] is False


def test_all_bound_inputs_and_sources_match_hashes():
    program = load()
    for section in ("input_bindings", "source_bindings"):
        for record in program[section].values():
            path = Path(record["path"])
            assert path.is_file()
            assert sha256(path) == record["sha256"]


def test_success_cannot_claim_posterior_or_resolution():
    semantics = load()["success_semantics"]
    assert semantics["Phase_B_technical_pass_only"] is True
    assert semantics["actual_present_day_density_posterior_created"] is False
    assert semantics["actual_present_day_velocity_posterior_created"] is False
    assert semantics["observational_frontier_or_resolution_claim"] is False
    assert semantics["target_0p3_cMpc_h_reached"] is False
    assert semantics["Phase_C_requires_separate_user_approval"] is True
    assert semantics["automatic_follow_on"] is False
