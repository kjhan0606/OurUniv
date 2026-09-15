"""Survival-calibrated PM z=0 model and exact field-aware radial coordinates."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import digamma, polygamma

from cf4_pm_calibrated_z0 import PMCalibratedFieldModel


class SurvivalFieldModel(PMCalibratedFieldModel):
    def __init__(self, *args, selection_shells, survival_yes, survival_no, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection_shells = jnp.asarray(selection_shells)
        a, b = np.asarray(survival_yes)+1., np.asarray(survival_no)+1.
        if selection_shells.shape != (6, a.shape[1], self.n, self.n, self.n) or a.shape != b.shape:
            raise ValueError("invalid survival model shapes")
        np.testing.assert_allclose(np.sum(selection_shells, axis=1), self.response)
        self.survival_a, self.survival_b = jnp.asarray(a), jnp.asarray(b)
        self.survival_center = jnp.asarray(digamma(a)-digamma(b))
        self.survival_scale = jnp.asarray(np.sqrt(polygamma(1,a)+polygamma(1,b)))
        self.size += a.size
        self.count_train_fraction = .6

    def survival_logits(self, nuisance):
        return self.survival_center + self.survival_scale*nuisance[24:].reshape(self.survival_a.shape)

    def observe(self, rho, velocity, nuisance):
        intensity, radial = super().observe(rho, velocity, nuisance)
        probability = jax.nn.sigmoid(self.survival_logits(nuisance))
        thinned = jnp.einsum("ps,psijk->pijk", probability, self.selection_shells)
        return intensity*thinned/jnp.where(self.response > 0, self.response, 1), radial

    def nlp(self, vector, counts, radial_data):
        intensity, prediction = self.forward(vector)
        lam = self.count_train_fraction*intensity
        support = self.response > 0
        count_nll = jnp.sum(jnp.where(support, lam-counts*jnp.log(jnp.where(support, lam, 1)), 0))
        velocity_nll = .5*jnp.sum(jnp.where(self.train, (prediction-radial_data)**2/self.variance, 0))
        # Exact logit-Beta density, including p(1-p) Jacobian; linear scale is constant.
        nuisance = vector[self.field_size:]
        logits = self.survival_logits(nuisance)
        selection_nll = -jnp.sum(self.survival_a*jax.nn.log_sigmoid(logits)
                                + self.survival_b*jax.nn.log_sigmoid(-logits))
        return .5*jnp.sum(vector[:self.field_size+24]**2)+selection_nll+count_nll+velocity_nll


class RadialCoordinates:
    """q=mu(field,y_train)+chol(C)*epsilon, C=(I+A'WA)^-1.

The block triangular transformation has constant determinant, so evaluating
the original joint objective at the transformed vector is exact up to a
constant. Epsilon is independent standard Gaussian; its removal integrates
the four radial parameters analytically. Keeping it retains existing storage.
"""
    def __init__(self, model, radial):
        self.model, self.radial = model, jnp.asarray(radial)
        train = np.asarray(model.train)
        A = np.asarray(model.B)*np.asarray(model.qstd)[None,:]
        weight = 1/np.asarray(model.variance)[train]
        precision = np.eye(4)+(A[train].T*weight)@A[train]
        covariance = np.linalg.solve(precision, np.eye(4))
        self.factor = jnp.asarray(np.linalg.cholesky(covariance))
        self.operator = jnp.asarray(covariance@(A[train].T*weight))
        self.train_indices = np.flatnonzero(train)
        self.start = model.field_size+20

    def transform(self, vector):
        model = self.model
        _, _, velocity = model.fields(vector)
        base = model.radial_prediction(velocity, jnp.zeros(model.size-model.field_size))
        mean = self.operator@(self.radial[self.train_indices]-base[self.train_indices])
        q = mean+self.factor@vector[self.start:self.start+4]
        return vector.at[self.start:self.start+4].set(q), mean

    def nlp(self, vector, counts, radial):
        physical, _ = self.transform(vector)
        return self.model.nlp(physical, counts, radial)
