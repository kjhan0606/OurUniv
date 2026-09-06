import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cf4_chunked_hmc import make_chunks, checked_record


class ChunkedHMCTest(unittest.TestCase):
    def test_gaussian_recovery_and_frozen_sampling_step(self):
        settings = dict(divergence_threshold=1000., target_acceptance=.9,
                        initial_step_size=.01, maximum_step_size=.25, integration_steps=16)
        initialize, warm, sample, final = make_chunks(lambda x: -.5 * jnp.sum(x*x), 2, settings)
        state, adaptation = initialize(jnp.array([1., -1.]))
        (state, adaptation), _ = warm(state, adaptation, jax.random.split(jax.random.PRNGKey(51), 128))
        step = final(adaptation)
        state, records = sample(state, step, jax.random.split(jax.random.PRNGKey(52), 1024))
        positions, _, _, divergent, _, raw, used = checked_record(records, state)
        self.assertFalse(divergent.any())
        np.testing.assert_allclose(raw, float(step), atol=0)
        np.testing.assert_array_equal(raw, used)
        np.testing.assert_allclose(positions.mean(axis=0), 0, atol=.15)
        np.testing.assert_allclose(positions.var(axis=0), 1, atol=.2)


if __name__ == "__main__":
    unittest.main()
