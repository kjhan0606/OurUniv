"""Small chunked wrapper around the existing BlackJAX HMC/dual-averaging recipe."""
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.step_size import dual_averaging_adaptation


def make_chunks(logdensity, dimension, settings):
    kernel = blackjax.hmc.build_kernel(divergence_threshold=settings["divergence_threshold"])
    da_init, da_update, da_final = dual_averaging_adaptation(settings["target_acceptance"])
    inverse_mass = jnp.ones(dimension, dtype=jnp.float64)

    @jax.jit
    def initialize(position):
        return blackjax.hmc.init(position, logdensity), da_init(settings["initial_step_size"])

    def info_record(state, info, raw, used):
        return (state.position, state.logdensity, info.acceptance_rate, info.is_divergent, info.energy, raw, used)

    @jax.jit
    def warm_chunk(state, adaptation, keys):
        def transition(carry, key):
            state, adaptation = carry
            raw = jnp.exp(adaptation.log_step_size)
            used = jnp.minimum(raw, settings["maximum_step_size"])
            state, info = kernel(key, state, logdensity, used, inverse_mass, settings["integration_steps"])
            return (state, da_update(adaptation, info.acceptance_rate)), info_record(state, info, raw, used)
        return jax.lax.scan(transition, (state, adaptation), keys)

    @jax.jit
    def sample_chunk(state, step, keys):
        def transition(state, key):
            state, info = kernel(key, state, logdensity, step, inverse_mass, settings["integration_steps"])
            return state, info_record(state, info, step, step)
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
