"""Fixed optimizer coordinates; the statistical target stays in physical q."""
import numpy as np


class AffineObjective:
    """Evaluate f(origin + scale*z) - offset with its exact chain-rule gradient.

    This is an optimization adapter, not a new prior or posterior density.
    A constant objective offset avoids a large additive prior baseline in
    function-tolerance tests. Neither operation changes stationary points.
    """

    def __init__(self, objective, origin, scale, offset=0.0):
        self.objective = objective
        self.origin = np.asarray(origin, dtype=np.float64).copy()
        self.scale = np.broadcast_to(np.asarray(scale, dtype=np.float64),
                                     self.origin.shape).copy()
        self.offset = float(offset)
        if (self.origin.ndim != 1 or not np.isfinite(self.origin).all()
                or not np.isfinite(self.scale).all() or np.any(self.scale <= 0)
                or not np.isfinite(self.offset)):
            raise ValueError('finite origin and positive fixed coordinate scales required')

    def physical(self, z):
        z = np.asarray(z, dtype=np.float64)
        if z.shape != self.origin.shape:
            raise ValueError('optimizer coordinates have the wrong shape')
        return self.origin + self.scale * z

    def __call__(self, z):
        value, gradient = self.objective(self.physical(z))
        return float(value) - self.offset, np.asarray(gradient) * self.scale
