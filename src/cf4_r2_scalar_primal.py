"""Select a scalar GL2 observation primal without changing its target law."""


def scalar_observation_terms(obs, rho, velocity, tracer, population, packs,
                             source, observation, order, *, backend):
    """Return the same ``(score, components)`` from a declared scalar backend.

    ``direct_value_only`` avoids constructing observation gradients. The
    ``matched_value_and_gradient_primal`` path preserves the historical
    floating-point primal used by exact-gradient runs. They implement the
    same data function, but compiled floating-point reductions can differ;
    callers must use one backend consistently within a scalar chain.
    """
    args = (rho, velocity, tracer, population, packs, source, observation, order)
    if backend == "direct_value_only":
        return obs.value(*args)
    if backend == "matched_value_and_gradient_primal":
        value_and_components, _unused_gradients = obs.derivative(*args)
        return value_and_components
    raise ValueError(f"unsupported scalar observation backend: {backend}")
