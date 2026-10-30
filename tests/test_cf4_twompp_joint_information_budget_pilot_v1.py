import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/cf4_twompp_joint_information_budget_pilot_v1.py"
PROGRAM = ROOT / "config/cf4_twompp_joint_information_budget_pilot_program_v1.json"
RUNNER = ROOT / "scripts/run_cf4_twompp_joint_information_budget_pilot_v1.sbatch"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_twompp_joint_info_v1", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _small_canonical_plan(grid: int) -> dict[str, np.ndarray]:
    q = np.rint(np.fft.fftfreq(grid) * grid).astype(np.int64)
    qx, qy, qz = np.meshgrid(q, q, q, indexing="ij")
    flat_q = [axis.ravel() for axis in (qx, qy, qz)]
    nyquist = -(grid // 2)

    def conjugate(values):
        opposite = -values
        return np.where(opposite == grid // 2, nyquist, opposite)

    cx, cy, cz = [conjugate(values) for values in flat_q]
    canonical = (
        (flat_q[0] < cx)
        | ((flat_q[0] == cx) & (flat_q[1] < cy))
        | (
            (flat_q[0] == cx)
            & (flat_q[1] == cy)
            & (flat_q[2] <= cz)
        )
    )
    canonical &= ~(
        (flat_q[0] == 0) & (flat_q[1] == 0) & (flat_q[2] == 0)
    )
    flat = np.flatnonzero(canonical)
    return {
        "flat_independent_field_indices": flat,
        "mode_merged_bin_index": np.zeros(flat.size, dtype=np.int64),
    }


def test_isolated_help_lists_two_environment_workflow() -> None:
    completed = subprocess.run(
        ["/home/kjhan/miniconda3/bin/python3.13", "-I", "-P", str(SOURCE), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "build-selection" in completed.stdout
    assert "run-pilot" in completed.stdout
    assert "validate-pilot" in completed.stdout


def test_program_binds_implementation_and_freezes_firewalls() -> None:
    program = json.loads(PROGRAM.read_text())
    implementation = program["bindings"]["implementation"]
    assert implementation["bytes"] == SOURCE.stat().st_size
    assert implementation["sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    authorization = program["authorization"]
    assert authorization["joint_information_budget_technical_pilot"] is True
    assert authorization["observed_population_totals_for_shot_noise_normalization"] is True
    for key in (
        "field_inference",
        "galaxy_positions_as_field_likelihood_datum",
        "truth_array_generation_or_deserialization",
        "exact_bias_RSD_FoG_selection_discrepancy_marginalization",
        "parent_posterior_promotion",
        "resolution_increase",
        "untouched_256_mock_validation",
        "IC_PM_HOP_RAMSES",
        "automatic_follow_on",
    ):
        assert authorization[key] is False


def test_program_uses_six_luminosity_populations_and_honest_scenarios() -> None:
    design = json.loads(PROGRAM.read_text())["design"]
    assert design["population_counts"] == [9617, 3463, 527, 15671, 6197, 1160]
    assert sum(design["population_counts"]) == 36635
    assert design["reference_bias_by_population"] == [1.3, 1.05, 0.85] * 2
    assert list(design["density_scenarios"]) == [
        "known_selection_reference_bias_ceiling",
        "normalization_marginalized_reference_bias",
        "normalization_marginalized_half_Fisher_sensitivity",
        "normalization_marginalized_quarter_Fisher_sensitivity",
    ]
    limitations = " ".join(design["limitations"])
    assert "not called marginalized" in limitations
    assert "12 cMpc/h" in limitations
    assert design["decision_rule"]["promotion_rule"].startswith(
        "No technical-pilot outcome promotes"
    )


def test_schechter_fraction_respects_full_and_empty_intersections() -> None:
    module = _load_module()
    distance = np.asarray([10.0, 100.0])
    full = module.schechter_fraction(
        distance, None, 99.0, -25.0, -21.0, -23.28, -0.94
    )
    empty = module.schechter_fraction(
        distance, None, -99.0, -25.0, -21.0, -23.28, -0.94
    )
    np.testing.assert_allclose(full, 1.0, rtol=0.0, atol=2e-14)
    np.testing.assert_array_equal(empty, 0.0)


def test_canonical_probe_is_hermitian_and_unit_power() -> None:
    module = _load_module()
    grid = 4
    plan = _small_canonical_plan(grid)
    flat = plan["flat_independent_field_indices"]
    real_probe, canonical = module.canonical_probe(grid, flat, 22)
    spectrum = np.fft.fftn(real_probe, norm="ortho")
    coordinates = np.column_stack(np.unravel_index(flat, (grid,) * 3))
    for coordinate in coordinates:
        conjugate = (-coordinate) % grid
        np.testing.assert_allclose(
            spectrum[tuple(conjugate)], np.conjugate(spectrum[tuple(coordinate)])
        )
    np.testing.assert_allclose(np.abs(canonical) ** 2, 1.0, atol=2e-15)


def test_zero_density_retention_exactly_reproduces_unit_prior_trace() -> None:
    module = _load_module()
    grid = 4
    plan = _small_canonical_plan(grid)
    transfer = np.ones((grid,) * 3)
    transfer[0, 0, 0] = 0.0
    metrics, _ = module.joint_trace_spectrum(
        transfer=transfer,
        velocity_precision=np.ones_like(transfer),
        plan=plan,
        expected_counts=np.ones((6, grid, grid, grid)),
        bias=np.asarray([1.3, 1.05, 0.85] * 2),
        bin_ids=np.asarray([0]),
        retention=0.0,
        marginalize_normalizations=False,
        probe_count=4,
        probe_seed=30,
        cg_rtol=1e-10,
        cg_maxiter=50,
    )
    np.testing.assert_allclose(metrics["posterior_prior_trace_fraction"], [1.0])
    np.testing.assert_allclose(
        metrics["recovered_information_fraction"], [0.0], rtol=0.0, atol=5e-15
    )


def test_positive_density_fisher_reduces_posterior_trace() -> None:
    module = _load_module()
    grid = 4
    plan = _small_canonical_plan(grid)
    transfer = np.ones((grid,) * 3)
    transfer[0, 0, 0] = 0.0
    metrics, _ = module.joint_trace_spectrum(
        transfer=transfer,
        velocity_precision=np.ones_like(transfer),
        plan=plan,
        expected_counts=np.ones((6, grid, grid, grid)),
        bias=np.asarray([1.3, 1.05, 0.85] * 2),
        bin_ids=np.asarray([0]),
        retention=1.0,
        marginalize_normalizations=False,
        probe_count=4,
        probe_seed=40,
        cg_rtol=1e-10,
        cg_maxiter=50,
    )
    assert 0.0 < metrics["posterior_prior_trace_fraction"][0] < 1.0
    assert 0.0 < metrics["recovered_information_fraction"][0] < 1.0


def test_runner_pins_resources_controller_and_two_python_environments() -> None:
    program = json.loads(PROGRAM.read_text())
    text = RUNNER.read_text()
    assert f"program_sha={hashlib.sha256(PROGRAM.read_bytes()).hexdigest()}" in text
    assert f"source_sha={hashlib.sha256(SOURCE.read_bytes()).hexdigest()}" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --mem=3072M" in text
    assert "#SBATCH --time=01:00:00" in text
    assert program["execution"]["requested_memory_MiB"] >= 1.2 * program["execution"][
        "expected_peak_memory_MiB"
    ]
    assert '"$SUBMISSION_CONTROLLER" == syntax' in text
    assert 'host_name" != syntax' in text and 'host_name" != syn101' in text
    assert "build-selection" in text and "run-pilot" in text and "validate-pilot" in text
    assert "scripts/tripwire/**" in text
    assert "renameat2" not in text
    assert "pgrep" not in text
    assert "/tmp" not in text
