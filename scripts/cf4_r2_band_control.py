"""Common-state metric control; never returns a selected chain endpoint."""
import copy
import numpy as np
from scipy.fft import fftn
from cf4_r2_prior_split_hmc import split_hmc_step


def paired_control(oracle, metrics, q, value, gradient, rng, *, step, steps):
    """Same standard Gaussian and acceptance uniform; caller owns budget.

    Each arm starts from q and clones the supplied diagnostic RNG. The caller's
    RNG advances once, as one transition. The original chain RNG is not passed.
    """
    if set(metrics) != {'A','B'}:
        raise ValueError('two predeclared metric arms A/B required')
    n = metrics['A'].c.shape[0]
    if (metrics['B'].c.shape != metrics['A'].c.shape
            or not np.array_equal(metrics['A'].b, metrics['B'].b)):
        raise ValueError('aligned IC grids and unchanged nuisance metric required')
    k = np.fft.fftfreq(n)*n
    k2 = k[:,None,None]**2+k[None,:,None]**2+k[None,None,:]**2
    mask = (k2>16)&(k2<=64)
    def displacement(proposed, before):
        delta = (proposed-before)[:n**3].reshape((n,)*3)
        spectral = fftn(delta, norm='ortho', workers=1)
        return {'band_squared_jump':float(np.sum(np.abs(spectral[mask])**2))}
    original = copy.deepcopy(rng.bit_generator.state)
    rows = {}
    terminal_rng = None
    for arm in ('A','B'):
        clone = copy.deepcopy(rng)
        clone.bit_generator.state = copy.deepcopy(original)
        retained, _, _, info = split_hmc_step(
            oracle, metrics[arm], q, value, gradient, clone,
            step=step, steps=steps, proposal_readout=displacement)
        proposed = info['proposal_readout']['band_squared_jump']
        alpha = float(np.exp(info['log_acceptance']))
        rows[arm] = dict(**info, acceptance_probability=alpha,
            proposed_band_squared_jump_per_force=proposed/steps,
            weighted_band_squared_jump_per_force=alpha*proposed/steps,
            retained_band_squared_jump_per_force=
                displacement(retained,q)['band_squared_jump']/steps,
            discard_for_inference=True)
        # In-support arms consume identical counts of Gaussian and uniform draws.
        if terminal_rng is not None and clone.bit_generator.state != terminal_rng:
            raise ValueError('paired arms consumed inconsistent random streams')
        terminal_rng = copy.deepcopy(clone.bit_generator.state)
    rng.bit_generator.state = terminal_rng
    return rows
