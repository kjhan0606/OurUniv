import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/cf4_twompp_exact_joint_covariance_pilot_v1.py"
PROGRAM = ROOT / "config/cf4_twompp_exact_joint_covariance_pilot_program_v1.json"
RUNNER = ROOT / "scripts/run_cf4_twompp_exact_joint_covariance_pilot_v1.sbatch"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_exact_joint_covariance_v1", SOURCE)
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


def test_isolated_help_lists_frozen_workflow() -> None:
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "True",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [
            "/home/kjhan/miniconda3/envs/circle/bin/python3.11",
            "-I",
            "-P",
            str(SOURCE),
            "--help",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "validate-program" in completed.stdout
    assert "run-pilot" in completed.stdout
    assert "validate-pilot" in completed.stdout


def test_program_binds_inputs_and_freezes_both_gate_semantics() -> None:
    program = json.loads(PROGRAM.read_text())
    for binding in program["bindings"].values():
        path = Path(binding["path"])
        assert path.stat().st_size == binding["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    authorization = program["authorization"]
    assert authorization["single_geometry_exact_joint_covariance_pilot"] is True
    assert authorization["single_Slurm_submission"] is True
    for key in (
        "truth_field_array_generation_or_deserialization",
        "likelihood_datum_consumed_by_inference",
        "observational_CF4_field_inference",
        "galaxy_positions_consumed_as_field_likelihood_datum",
        "new_truth_or_validation_seed",
        "untouched_256_mock_validation",
        "parent_posterior_promotion",
        "resolution_increase",
        "IC_PM_HOP_RAMSES",
        "automatic_follow_on",
    ):
        assert authorization[key] is False
    gates = program["design"]["information_gates"]
    assert gates["material"]["information_numerical_95_lower_min_inclusive"] == 0.5
    assert gates["strong_stretch"]["information_numerical_95_lower_min_inclusive"] == 0.8
    assert math.isclose(
        gates["strong_stretch"]["expected_correlation_r_min_inclusive"],
        math.sqrt(0.8),
    )
    assert gates["strong_stretch"]["expected_residual_power_ratio_max_inclusive"] == 0.2


def test_woodbury_inverse_and_posterior_covariance_match_dense_identities() -> None:
    module = _load_module()
    rng = np.random.default_rng(12)
    A = rng.standard_normal((7, 5))
    B = rng.standard_normal((7, 2))
    qvar = np.asarray([0.7, 1.8])
    small = module.nuisance_woodbury_small_inverse(B, qvar)
    applied = np.column_stack(
        [module.apply_nuisance_r_inverse(np.eye(7)[:, i], B, small) for i in range(7)]
    )
    dense_r_inverse = np.linalg.inv(np.eye(7) + B @ np.diag(qvar) @ B.T)
    np.testing.assert_allclose(applied, dense_r_inverse, rtol=2e-13, atol=2e-13)
    precision = np.eye(5) + A.T @ applied @ A
    covariance_precision = np.linalg.inv(precision)
    covariance_data_space = np.eye(5) - A.T @ np.linalg.inv(
        np.eye(7) + A @ A.T + B @ np.diag(qvar) @ B.T
    ) @ A
    np.testing.assert_allclose(
        covariance_precision, covariance_data_space, rtol=3e-13, atol=3e-13
    )


def test_exact_velocity_only_trace_reproduces_unit_prior_on_small_identity() -> None:
    module = _load_module()
    grid = 4
    transfer = np.ones((grid,) * 3)
    transfer[0, 0, 0] = 0.0
    result, _ = module.exact_joint_trace_spectrum(
        exact_velocity_precision=lambda field: np.asarray(field),
        transfer=transfer,
        plan=_small_canonical_plan(grid),
        domain_bin_ids={"delta": np.asarray([0]), "theta": np.asarray([0])},
        isotropic_preconditioner_precision=np.ones_like(transfer),
        expected_counts=np.ones((6, grid, grid, grid)),
        bias=np.asarray([1.3, 1.05, 0.85] * 2),
        density_retention=0.0,
        marginalize_normalizations=False,
        probe_count=4,
        probe_seed=100,
        operator_test_seed=101,
        cg_rtol=1e-10,
        cg_maxiter=50,
        symmetry_max=1e-12,
    )
    for domain in ("delta", "theta"):
        np.testing.assert_allclose(
            result["domains"][domain]["posterior_prior_trace_fraction"], [1.0]
        )
        np.testing.assert_allclose(
            result["domains"][domain]["recovered_information_fraction"],
            [0.0],
            atol=5e-15,
        )


def test_positive_density_fisher_reduces_exact_small_trace() -> None:
    module = _load_module()
    grid = 4
    transfer = np.ones((grid,) * 3)
    transfer[0, 0, 0] = 0.0
    result, _ = module.exact_joint_trace_spectrum(
        exact_velocity_precision=lambda field: np.asarray(field),
        transfer=transfer,
        plan=_small_canonical_plan(grid),
        domain_bin_ids={"delta": np.asarray([0]), "theta": np.asarray([0])},
        isotropic_preconditioner_precision=np.ones_like(transfer),
        expected_counts=np.ones((6, grid, grid, grid)),
        bias=np.asarray([1.3, 1.05, 0.85] * 2),
        density_retention=1.0,
        marginalize_normalizations=True,
        probe_count=4,
        probe_seed=110,
        operator_test_seed=111,
        cg_rtol=1e-10,
        cg_maxiter=50,
        symmetry_max=1e-12,
    )
    for domain in ("delta", "theta"):
        trace = result["domains"][domain]["posterior_prior_trace_fraction"][0]
        information = result["domains"][domain]["recovered_information_fraction"][0]
        assert 0.0 < trace < 1.0
        assert 0.0 < information < 1.0


def test_runner_pins_controller_resources_hashes_and_non_destructive_contract() -> None:
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
    assert ".pilot.${SLURM_JOB_ID}.staging" in text
    assert "scripts/tripwire/**" in text
    assert "validate-program" in text and "run-pilot" in text and "validate-pilot" in text
    assert "renameat2" not in text
    assert "pgrep" not in text
    assert "/tmp" not in text
    assert re.search(r"(?m)^\s*rm(?:\s|$)", text) is None
