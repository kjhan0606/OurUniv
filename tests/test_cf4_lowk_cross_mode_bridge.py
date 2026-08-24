import numpy as np
import pytest

from cf4_lowk_cross_mode_bridge import (
    PopulationState,
    run_beta_one_control,
    run_parallel_tempering_bridge,
)


class FakeOracle:
    def evaluate(self, midpoint_mpc_h, axis):
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
