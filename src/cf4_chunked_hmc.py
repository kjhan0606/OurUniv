"""Small chunked wrapper around the existing BlackJAX HMC/dual-averaging recipe."""
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.step_size import dual_averaging_adaptation


def make_chunks(logdensity, dimension, settings, *, record_steps=False):
    """Optional state-independent trajectory randomisation; same HMC target.

    Fixed-length callers keep their existing keys and seven-array trace. When
    requested, an eighth array records actual integration work, not an estimate.
    """
    kernel = blackjax.hmc.build_kernel(divergence_threshold=settings["divergence_threshold"])
    da_init, da_update, da_final = dual_averaging_adaptation(settings["target_acceptance"])
    inverse_mass = jnp.ones(dimension, dtype=jnp.float64)

    @jax.jit
    def initialize(position):
        return blackjax.hmc.init(position, logdensity), da_init(settings["initial_step_size"])

    step_range = settings.get("integration_steps_range")
    if step_range is not None and (len(step_range) != 2 or
            any(not isinstance(x, int) for x in step_range) or
            not 1 <= step_range[0] <= step_range[1]):
        raise ValueError("integration_steps_range must contain two ordered positive integers")

    def advance(key, state, step):
        count = settings["integration_steps"]
        if step_range is not None:
            key, length_key = jax.random.split(key)
            count = jax.random.randint(length_key, (), step_range[0], step_range[1]+1)
        state, info = kernel(key, state, logdensity, step, inverse_mass, count)
        return state, info, jnp.asarray(count)

    def info_record(state, info, raw, used, count):
        result = (state.position, state.logdensity, info.acceptance_rate, info.is_divergent, info.energy, raw, used)
        return result + (count,) if record_steps else result

    @jax.jit
    def warm_chunk(state, adaptation, keys):
        def transition(carry, key):
            state, adaptation = carry
            raw = jnp.exp(adaptation.log_step_size)
            used = jnp.minimum(raw, settings["maximum_step_size"])
            state, info, count = advance(key, state, used)
            return (state, da_update(adaptation, info.acceptance_rate)), info_record(state, info, raw, used, count)
        return jax.lax.scan(transition, (state, adaptation), keys)

    @jax.jit
    def sample_chunk(state, step, keys):
        def transition(state, key):
            state, info, count = advance(key, state, step)
            return state, info_record(state, info, step, step, count)
        return jax.lax.scan(transition, state, keys)

    def final_step(adaptation):
        return jnp.minimum(da_final(adaptation), settings["maximum_step_size"])

    return initialize, warm_chunk, sample_chunk, final_step


def checked_record(record, state):
    arrays = tuple(np.asarray(a) for a in record)
    if not all(np.isfinite(a).all() for a in arrays):
        raise FloatingPointError("non-finite HMC trace")
    if not np.isfinite(np.asarray(state.logdensity_grad)).all():
        raise FloatingPointError("non-finite HMC state gradient")
    return arrays
