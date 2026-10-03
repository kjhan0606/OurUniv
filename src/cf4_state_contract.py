"""Named particle-state contract for new diagnostics."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ParticleState:
    positions_cMpc_h: np.ndarray
    velocities_km_s: np.ndarray
    masses_Msun_h: np.ndarray
    tags: np.ndarray | None = None

def validate_state(state: ParticleState, *, box_cMpc_h: float) -> ParticleState:
    n = len(state.positions_cMpc_h)
    if state.positions_cMpc_h.shape != (n, 3) or state.velocities_km_s.shape != (n, 3): raise ValueError('positions/velocities must have shape (N,3)')
    if state.masses_Msun_h.shape != (n,): raise ValueError('masses must have shape (N,)')
    if not np.isfinite(state.positions_cMpc_h).all() or not np.isfinite(state.velocities_km_s).all() or not np.isfinite(state.masses_Msun_h).all(): raise ValueError('nonfinite state')
    if np.any(state.masses_Msun_h <= 0) or box_cMpc_h <= 0: raise ValueError('invalid mass or box')
    if state.tags is not None and state.tags.shape != (n,): raise ValueError('tags must have shape (N,)')
    return state
