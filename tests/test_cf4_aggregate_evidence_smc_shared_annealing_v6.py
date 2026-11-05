import json
from pathlib import Path

import numpy as np
import pytest

import cf4_aggregate_evidence_smc_shared_annealing_v6 as v6


class _Pilot:
    master_seed = 2026082301
    particle_count = 2048
    parent_seeds = tuple(range(3193, 3449))
    beta_history = (0.0, 0.25, 0.7, 1.0)

    def __init__(self):
        self.closed = False
        self.cache_disposed = False

    def close(self):
        self.closed = True
        self.cache_disposed = True


def _record(seed, schedule, *, beta=None):
    return v6.FreshProductionReplicate(
        master_seed=seed,
        beta_history=schedule.beta if beta is None else beta,
        evaluator_namespace="v6_fresh_production_evaluator",
        cache_namespace="v6_fresh_production_cache",
    )


def test_v6_design_keeps_fixed_science_and_all_execution_false():
    design_path = Path(v6.__file__).resolve().parents[1] / "config/cf4_aggregate_evidence_smc_shared_annealing_v6_design.json"
    design = json.loads(design_path.read_text())
    assert design["fixed_science_parameters"]["master_seeds"] == list(v6.V6_MASTER_SEEDS)
    assert design["fixed_science_parameters"]["particles_per_replicate"] == 2048
    assert all(value is False for key, value in design["authorization"].items() if key != "v6_design_and_implementation_authorized")
    assert "/gpfs" not in json.dumps(design)
    v6.validate_frozen_v6_parameters()


def test_pilot_schedule_is_shared_fixed_and_seed_zero_is_rerun_fresh():
    pilot = _Pilot()
    calls = []

    def production(seed, schedule):
        calls.append((seed, schedule.schedule_sha256, pilot.closed, pilot.cache_disposed))
        return _record(seed, schedule)

    schedule, records = v6.run_shared_annealing_plan(lambda: pilot, production)
    assert pilot.closed and pilot.cache_disposed
    assert [row[0] for row in calls] == list(v6.V6_MASTER_SEEDS)
    assert calls[0][0] == 2026082301 and all(row[2:] == (True, True) for row in calls)
    assert all(record.beta_history == schedule.beta for record in records)
    assert all(not record.pilot_cache_reused and not record.pilot_posterior_reused for record in records)


def test_stage_parity_mismatch_is_an_architecture_failure():
    def production(seed, schedule):
        beta = schedule.beta if seed != 2026082303 else (0.0, 0.3, 0.7, 1.0)
        return _record(seed, schedule, beta=beta)

    with pytest.raises(v6.ArchitectureFailure, match="shared_schedule_stage_parity"):
        v6.run_shared_annealing_plan(_Pilot, production)


def test_pilot_must_be_closed_and_disposed_before_production():
    class BadPilot(_Pilot):
        def close(self):
            self.closed = True

    with pytest.raises(v6.ArchitectureFailure, match="closed and disposed"):
        v6.run_shared_annealing_plan(BadPilot, lambda seed, schedule: _record(seed, schedule))


def test_null_calibration_is_deterministic_observed_failure_and_plausible_null_passes():
    p_pool = np.full(256, 1.0 / 256.0)
    observed = 0.5163215650597401
    first = v6.calibrate_max_l1_null(p_pool, observed)
    second = v6.calibrate_max_l1_null(p_pool, observed)
    assert first == second
    assert first.draws == 20_000 and first.seed == 2026081801
    assert first.tail_probability < 0.001

    rng = np.random.Generator(np.random.PCG64DXSM(77))
    plausible = rng.multinomial(2048, p_pool, size=4) / 2048.0
    diagnostics = v6.paired_incoherence_diagnostics(
        np.asarray([-6.7, -6.71, -6.69, -6.70]), plausible, p_pool,
        calibration_draws=512,
    )
    assert diagnostics["null_calibration"]["coherent_pass"] is True


def test_failed_channels_and_paired_incoherence_priority(monkeypatch):
    p_pool = np.zeros(256)
    p_pool[:2] = 0.5
    p_rep = np.tile(p_pool, (4, 1))
    p_rep[0, 0] += 0.3; p_rep[0, 1] -= 0.3
    p_rep[1, 0] -= 0.3; p_rep[1, 1] += 0.3
    monkeypatch.setattr(
        v6, "calibrate_max_l1_null",
        lambda *_args, **_kwargs: v6.L1NullCalibration(20_000, 2026081801, 0.3, 0.32, 0.00005),
    )
    value = v6.paired_incoherence_diagnostics(
        np.asarray([-6.8, -6.79, -6.57, -6.69]), p_rep, p_pool,
    )
    assert value["failed_channels"] == [
        "replicate_log_I_bar_range", "replicate_parent_probability_L1",
    ]
    assert value["primary_failure"] == "paired_incoherence"
    assert value["L1_diagnostic_threshold"] == 0.2
