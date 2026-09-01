import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_phasec_pmwd_scan_diagnostic_v1.json"
SOURCE = ROOT / "src/cf4_phasec_pmwd_scan_diagnostic.py"
RUNNER = ROOT / "scripts/run_cf4_phasec_pmwd_scan_diagnostic_v1.sbatch"
AGGREGATOR = ROOT / "scripts/aggregate_cf4_phasec_pmwd_scan_diagnostic_v1.sbatch"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_program():
    from cf4_phasec_pmwd_scan_diagnostic import load_program

    return load_program(PROGRAM)[0]


def test_diagnostic_is_limited_to_the_two_observed_failure_seeds():
    program = load_program()
    assignments = program["assignments"]
    assert [row["seed"] for row in assignments] == [2026083002, 2026083007]
    assert assignments[0]["prior_monolithic_outcomes"] == {
        "1_over_64": "finite",
        "1_over_128": "nonfinite",
        "1_over_256": "nonfinite",
    }
    assert assignments[1]["prior_monolithic_outcomes"] == {
        "1_over_64": "nonfinite",
        "1_over_128": "finite",
        "1_over_256": "finite",
    }


def test_scan_integrator_and_step_localization_contract_are_frozen():
    program = load_program()
    integrator = program["integrator"]
    assert integrator["a_nbody_maxsteps"] == [1 / 64, 1 / 128, 1 / 256]
    assert "jax.lax.scan" in integrator["loop_implementation"]
    diagnostics = program["diagnostics"]
    assert diagnostics["record_every_step_particle_displacement_velocity_acceleration_finiteness"] is True
    assert diagnostics["record_first_nonfinite_step_and_component"] is True
    assert diagnostics["comparison_gates"] == {
        "density_cross_correlation_min": 0.999,
        "density_relative_L2_max": 0.03,
        "velocity_cross_correlation_min": 0.995,
        "velocity_relative_L2_max": 0.05,
    }


def test_lineage_hashes_and_scope_firewall():
    program = load_program()
    for binding in program["lineage"].values():
        path = Path(binding["path"])
        assert path.is_file()
        assert sha256(path) == binding["sha256"]
    assert not any(program["scope_firewall"].values())
    authorization = program["authorization"]
    for key in ("sampler", "actual_observational_data", "validation_seed", "Phase_D_or_later"):
        assert authorization[key] is False


def test_particle_metric_json_conversion():
    from cf4_phasec_pmwd_scan_diagnostic import particle_metric_rows

    metrics = np.asarray([[1.0, 0.0, 3.0], [0.0, 7.0, 4.0], [1.0, 0.0, 5.0]])
    rows = particle_metric_rows(metrics)
    assert [row["component"] for row in rows] == ["displacement", "velocity", "acceleration"]
    assert rows[0] == {
        "component": "displacement",
        "all_finite": True,
        "nonfinite_count": 0,
        "finite_max_abs": 3.0,
    }
    assert rows[1]["all_finite"] is False
    assert rows[1]["nonfinite_count"] == 7


def test_source_uses_scan_and_never_runs_inference_or_reads_observations():
    source = SOURCE.read_text()
    assert "jax.lax.scan" in source
    assert "nbody_step" in source
    for forbidden in (
        "raw_selection_exposure",
        "counts_train",
        "counts_holdout",
        "vobs",
        "prepare_catalog",
        "run_four_chains",
        "blackjax",
    ):
        assert forbidden not in source


def test_slurm_memory_headroom_and_controller_contract():
    execution = load_program()["execution"]
    assert execution["requested_host_memory_MiB_per_task"] >= 1.2 * execution[
        "expected_peak_host_memory_MiB_per_task"
    ]
    assert execution["aggregate_requested_host_memory_MiB"] >= 1.2 * execution[
        "aggregate_expected_peak_host_memory_MiB"
    ]
    runner = RUNNER.read_text()
    aggregator = AGGREGATOR.read_text()
    assert "#SBATCH --array=0-1" in runner
    assert "#SBATCH --mem=9216M" in runner
    assert "#SBATCH --mem=1024M" in aggregator
    for text in (runner, aggregator):
        assert '"$SUBMISSION_CONTROLLER" == syntax' in text
        assert "EXPECTED_UPSTREAM_COMMIT" in text
        assert "scripts/tripwire/**" in text
        assert "pgrep" not in text
        assert "renameat2" not in text
        assert "/tmp" not in text


def test_sampler_cannot_be_released_by_the_diagnostic_aggregate():
    decision = load_program()["decision_rule"]
    assert decision["automatic_sampler_release"] is False
    assert "all-eight-seed generator gate" in decision["next_if_pass"]
