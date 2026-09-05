import inspect
from types import SimpleNamespace

import numpy as np

import cf4_aggregate_evidence_smc_production as production


class FakeOracle:
    def __init__(self):
        self.registered = []
        self.sealed = False

    def register_terminal_history(self, master_seed, keys):
        assert keys.shape == (2048, 6)
        self.registered.append(master_seed)

    def seal_terminal_histories(self):
        assert self.registered == [
            2026082301, 2026082302, 2026082303, 2026082304
        ]
        self.sealed = True

    def terminal_parent_log_z(self, master_seed, keys):
        assert self.sealed
        assert master_seed in self.registered
        return np.zeros((2048, 256), dtype=np.float64)


def test_public_production_driver_fails_closed_before_any_oracle_call(monkeypatch):
    called = []
    monkeypatch.setattr(
        production,
        "_run_four_replicates_core_for_validation",
        lambda oracle: called.append(oracle),
    )
    with np.testing.assert_raises_regex(PermissionError, "not authorized"):
        production.run_four_production_replicates(FakeOracle())
    assert called == []
    assert set(inspect.signature(
        production.run_four_production_replicates
    ).parameters) == {"oracle"}


def test_private_validation_core_has_no_particle_seed_or_kernel_overrides(monkeypatch):
    calls = []

    def fake_replicate(master_seed, oracle):
        calls.append((master_seed, oracle))
        return SimpleNamespace(
            master_seed=master_seed,
            keys=np.zeros((2048, 6), dtype=np.int16),
            weights=np.full(2048, 1.0 / 2048.0),
            beta_history=np.asarray([0.0, 1.0]),
            move_history=[[{}, {}, {}, {}]],
            log_normalizer=0.0,
        )

    monkeypatch.setattr(production, "run_smc_replicate", fake_replicate)
    oracle = FakeOracle()
    result = production._run_four_replicates_core_for_validation(oracle)
    assert set(inspect.signature(
        production._run_four_replicates_core_for_validation
    ).parameters) == {"oracle"}
    assert [seed for seed, _ in calls] == [
        2026082301, 2026082302, 2026082303, 2026082304
    ]
    assert all(item is oracle for _, item in calls)
    assert result.replicate_parent_probability.shape == (4, 256)
    np.testing.assert_allclose(
        result.pooled_parent_probability,
        np.full(256, 1.0 / 256.0),
        atol=1e-15,
    )
    assert result.pooled_log_i_bar == 0.0
