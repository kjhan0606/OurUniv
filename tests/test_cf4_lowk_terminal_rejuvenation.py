import numpy as np
import pytest

from cf4_lowk_terminal_rejuvenation import continue_terminal_population


class FakeOracle:
    def evaluate(self, midpoint_mpc_h, axis):
        midpoint = np.asarray(midpoint_mpc_h)
        axes = np.asarray(axis)
        keys = np.column_stack((
            np.rint(midpoint).astype(np.int16),
            np.rint(3.0 * axes).astype(np.int16),
        ))
        log_z_bar = -0.01 * np.sum(midpoint**2, axis=1)
        return keys, log_z_bar


def terminal_fixture(count=32):
    midpoint = np.zeros((count, 3), dtype=np.float64)
    axis = np.zeros((count, 3), dtype=np.float64)
    axis[:, 0] = 1.0
    keys, log_z_bar = FakeOracle().evaluate(midpoint, axis)
    weights = np.arange(1, count + 1, dtype=np.float64)
    weights /= weights.sum()
    ancestors = np.arange(count, dtype=np.int64)
    return midpoint, axis, keys, log_z_bar, weights, ancestors


def test_terminal_continuation_is_deterministic_and_does_not_mutate_inputs():
    arrays = terminal_fixture()
    originals = tuple(value.copy() for value in arrays)
    kwargs = dict(
        master_seed=2026082301,
        midpoint_mpc_h=arrays[0],
        axis=arrays[1],
        keys=arrays[2],
        log_z_bar=arrays[3],
        weights=arrays[4],
        ancestor_labels=arrays[5],
        checkpoints=(2, 4),
        continuation_id=7,
    )
    left = continue_terminal_population(oracle=FakeOracle(), **kwargs)
    right = continue_terminal_population(oracle=FakeOracle(), **kwargs)

    assert [row.sweep for row in left] == [2, 4]
    for first, second in zip(left, right):
        assert np.array_equal(first.midpoint_mpc_h, second.midpoint_mpc_h)
        assert np.array_equal(first.axis, second.axis)
        assert np.array_equal(first.keys, second.keys)
        assert np.array_equal(first.log_z_bar, second.log_z_bar)
        assert np.array_equal(first.weights, np.full(32, 1.0 / 32))
        assert not first.weights.flags.writeable
        assert np.all(first.move_acceptance_count <= first.move_proposal_count)
    for original, current in zip(originals, arrays):
        assert np.array_equal(original, current)


@pytest.mark.parametrize("checkpoints", [(), (0,), (2, 2), (4, 2)])
def test_terminal_continuation_rejects_invalid_checkpoints(checkpoints):
    arrays = terminal_fixture(8)
    with pytest.raises(ValueError, match="checkpoints"):
        continue_terminal_population(
            master_seed=1,
            midpoint_mpc_h=arrays[0],
            axis=arrays[1],
            keys=arrays[2],
            log_z_bar=arrays[3],
            weights=arrays[4],
            ancestor_labels=arrays[5],
            oracle=FakeOracle(),
            checkpoints=checkpoints,
        )
