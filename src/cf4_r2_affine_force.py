"""Frozen first-order correction of a surrogate HMC potential.

U_proxy(q)=U_coarse(q)+d dot(q-q_anchor), where d is computed ONCE as
grad(U_fine)(q_anchor)-grad(U_coarse)(q_anchor). The Metropolis target remains
U_fine. Do not refresh the anchor per production trajectory: state-dependent
anchoring is not the fixed reversible proposal validated here.
"""
import numpy as np


class AffineCorrectedForce:
    def __init__(self,oracle,anchor,gradient_correction):
        self.anchor=np.array(anchor,dtype=float,copy=True)
        self.correction=np.array(gradient_correction,dtype=float,copy=True)
        if self.anchor.ndim!=1 or self.correction.shape!=self.anchor.shape or not (
                np.isfinite(self.anchor).all() and np.isfinite(self.correction).all()):
            raise ValueError('finite aligned canonical anchor and gradient correction required')
        self.anchor.setflags(write=False);self.correction.setflags(write=False)
        self.oracle=oracle

    def offset(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=self.anchor.shape or not np.isfinite(q).all():
            raise ValueError('finite matching canonical state required')
        return float(self.correction@(q-self.anchor))

    def __call__(self,q):
        offset=self.offset(q)
        value,gradient=self.oracle(q)
        if value==np.inf:return value,None
        gradient=np.asarray(gradient,dtype=float)
        if gradient.shape!=self.anchor.shape or not np.isfinite(gradient).all() or not np.isfinite(value):
            raise FloatingPointError('invalid coarse oracle at finite state')
        return float(value)+offset,gradient+self.correction
