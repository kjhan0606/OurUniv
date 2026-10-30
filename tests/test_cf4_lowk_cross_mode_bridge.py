import numpy as np
import pytest

from cf4_aggregate_evidence_smc import mh_rejuvenation_sweep
from cf4_lowk_cross_mode_bridge import (
    PopulationState,
    batched_mh_rejuvenation_sweep,
    run_beta_one_control,
    run_grouped_parallel_tempering_bridge,
    run_parallel_tempering_bridge,
)


class FakeOracle:
    def __init__(self):
        self.calls = 0

    def evaluate(self, midpoint_mpc_h, axis):
        self.calls += 1
        midpoint = np.asarray(midpoint_mpc_h)
        axes = np.asarray(axis)
        keys = np.column_stack((
            np.rint(midpoint).astype(np.int16),
            np.rint(3.0 * axes).astype(np.int16),
        ))
        log_z = -0.02 * np.sum(midpoint**2, axis=1)
        return keys, log_z


def state(offset, count=24):
    q = np.zeros((count, 3), dtype=np.float64)
    q[:, 0] = offset
    axis = np.zeros((count, 3), dtype=np.float64)
    axis[:, 0] = 1.0
    keys, log_z = FakeOracle().evaluate(q, axis)
    return PopulationState(q, axis, keys, log_z)


def test_bridge_is_deterministic_and_tracks_swaps_without_mutation():
    ladder = tuple(state(value) for value in (0.0, 0.5, 1.0))
    originals = tuple(row.midpoint_mpc_h.copy() for row in ladder)
    kwargs = dict(
        states=ladder,
        betas=np.asarray([0.0, 0.5, 1.0]),
        oracle=FakeOracle(),
        master_seed=77,
        checkpoints=(2, 4),
        sweeps_per_cycle=1,
    )
    left = run_parallel_tempering_bridge(**kwargs)
    right = run_parallel_tempering_bridge(**kwargs)
    assert [row.cycle for row in left] == [2, 4]
    for a, b in zip(left, right):
        assert np.array_equal(a.top.keys, b.top.keys)
        assert np.array_equal(a.top_origin_id, b.top_origin_id)
        assert np.array_equal(a.swap_proposal_count, np.asarray([48, 48]) * a.cycle / 2)
        assert np.all(a.swap_acceptance_count <= a.swap_proposal_count)
        assert not a.top.keys.flags.writeable
    for original, current in zip(originals, ladder):
        assert np.array_equal(original, current.midpoint_mpc_h)


def test_beta_one_control_uses_requested_sweep_checkpoints():
    result = run_beta_one_control(
        state=state(1.0), oracle=FakeOracle(), master_seed=9,
        checkpoints=(2, 5),
    )
    assert len(result) == 2
    assert all(row.keys.shape == (24, 6) for row in result)
    assert all(not row.keys.flags.writeable for row in result)


def test_batched_mh_matches_separate_kernels_with_one_oracle_call():
    states = (state(0.5), state(1.0))
    oracle = FakeOracle()
    actual = batched_mh_rejuvenation_sweep(
        states=states, betas=(0.4, 1.0), oracle=oracle,
        master_seeds=(11, 12), stages=(101, 102), sweep=3,
    )
    assert oracle.calls == 1
    expected = []
    for value, beta, seed, stage in zip(states, (.4, 1.), (11, 12), (101, 102)):
        separate = FakeOracle()
        q, a, k, z, _ = mh_rejuvenation_sweep(
            value.midpoint_mpc_h, value.axis, value.keys, value.log_z_bar,
            beta, separate, seed, stage, 3,
        )
        expected.append((q, a, k, z))
    for result, reference in zip(actual, expected):
        assert np.array_equal(result.midpoint_mpc_h, reference[0])
        assert np.array_equal(result.axis, reference[1])
        assert np.array_equal(result.keys, reference[2])
        assert np.array_equal(result.log_z_bar, reference[3])


def test_grouped_bridge_batches_all_groups_temperatures_and_controls():
    oracle = FakeOracle()
    ladders = tuple(
        (state(group), state(group + 0.5), state(group + 1.0))
        for group in range(2)
    )
    result = run_grouped_parallel_tempering_bridge(
        ladders=ladders, betas=np.asarray([0.0, 0.5, 1.0]), oracle=oracle,
        master_seeds=(21, 22), checkpoints=(1, 2), sweeps_per_cycle=1,
        lower_burnin_sweeps=2,
    )
    assert oracle.calls == 4  # two shared burn-in batches plus two shared cycle batches
    assert [row.cycle for row in result] == [1, 2]
    assert all(len(row.bridge_top) == 2 and len(row.control) == 2 for row in result)
    assert result[-1].swap_proposal_count.shape == (2, 2)


def test_grouped_bridge_preserves_legacy_rng_trajectory():
    ladder = (state(0.0), state(0.5), state(1.0))
    grouped = run_grouped_parallel_tempering_bridge(
        ladders=(ladder,), betas=np.asarray([0.0, 0.5, 1.0]),
        oracle=FakeOracle(), master_seeds=(77,), checkpoints=(1, 2),
        sweeps_per_cycle=1, lower_burnin_sweeps=0, namespace=4_000_000,
    )
    bridge = run_parallel_tempering_bridge(
        states=ladder, betas=np.asarray([0.0, 0.5, 1.0]),
        oracle=FakeOracle(), master_seed=77, checkpoints=(1, 2),
        sweeps_per_cycle=1, namespace=5_000_000,
    )
    control = run_beta_one_control(
        state=ladder[-1], oracle=FakeOracle(), master_seed=77,
        checkpoints=(1, 2), namespace=6_000_000,
    )
    for actual, expected_bridge, expected_control in zip(grouped, bridge, control):
        assert np.array_equal(actual.bridge_top[0].keys, expected_bridge.top.keys)
        assert np.array_equal(actual.bridge_top[0].log_z_bar, expected_bridge.top.log_z_bar)
        assert np.array_equal(actual.control[0].keys, expected_control.keys)
        assert np.array_equal(actual.control[0].log_z_bar, expected_control.log_z_bar)
        assert np.array_equal(actual.top_origin_id[0], expected_bridge.top_origin_id)
        assert np.array_equal(
            actual.swap_acceptance_count[0], expected_bridge.swap_acceptance_count
        )


@pytest.mark.parametrize("betas", [
    [0.1, 1.0], [0.0, 0.5], [0.0, 0.5, 0.5, 1.0], [0.0, 1.1, 1.0]
])
def test_bridge_rejects_invalid_ladders(betas):
    ladder = tuple(state(float(index)) for index in range(len(betas)))
    with pytest.raises(ValueError, match="temperature ladder"):
        run_parallel_tempering_bridge(
            states=ladder, betas=np.asarray(betas), oracle=FakeOracle(),
            master_seed=1, checkpoints=(1,),
        )
