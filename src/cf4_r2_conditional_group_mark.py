"""Normalized Gaussian CF4 group mark conditioned on 2M++ member redshifts.

This is a numerical kernel, not a calibrated R2 observation model. A caller
must supply means from the *same evolved state* and a source-calibrated joint
covariance for (group Vcmb, group distance modulus, member Vcmb...). The
member redshifts are conditioned on, not scored again: this factor can only
be paired with a point-data model that actually owns those same redshifts.
Selection, group assignment and covariance calibration are separate work.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _vector(label: str, values, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f'{label} must be a finite vector of length {size}')
    return array


def conditional_group_mark_logpdf(
    observed_group_vcmb_km_s: float,
    observed_group_dm_mag: float,
    observed_member_vcmb_km_s,
    predicted_group_vcmb_km_s: float,
    predicted_group_dm_mag: float,
    predicted_member_vcmb_km_s,
    joint_covariance,
    member_recnos,
) -> float:
    """Return log p(Vgroup, DMgroup | individual member redshifts, state).

    `joint_covariance` is ordered (Vgroup, DMgroup, member Vcmb...). Entries
    may have mixed physical units, but must be a calibrated positive-definite
    covariance on those variables. `member_recnos` prevents accidental reuse
    of one point datum within a group. No arbitrary covariance floor is added.
    """

    recnos = tuple(int(value) for value in member_recnos)
    if len(recnos) != len(set(recnos)) or any(value <= 0 for value in recnos):
        raise ValueError('member recnos must be unique positive integers')
    n = len(recnos)
    observed_members = _vector('observed members', observed_member_vcmb_km_s, n)
    predicted_members = _vector('predicted members', predicted_member_vcmb_km_s, n)
    observed = _vector('observed group marks',
                       [observed_group_vcmb_km_s, observed_group_dm_mag], 2)
    predicted = _vector('predicted group marks',
                        [predicted_group_vcmb_km_s, predicted_group_dm_mag], 2)
    covariance = np.asarray(joint_covariance, dtype=np.float64)
    if covariance.shape != (n+2, n+2) or not np.all(np.isfinite(covariance)):
        raise ValueError('joint covariance has wrong shape or non-finite entries')
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12):
        raise ValueError('joint covariance is not symmetric')
    covariance = 0.5*(covariance+covariance.T)
    try:
        cho_factor(covariance, lower=True, check_finite=False)
        if n:
            member_factor = cho_factor(covariance[2:, 2:], lower=True,
                                       check_finite=False)
            cross = covariance[:2, 2:]
            member_residual = observed_members-predicted_members
            conditional_mean = predicted+cross @ cho_solve(
                member_factor, member_residual, check_finite=False)
            conditional_covariance = covariance[:2, :2] - cross @ cho_solve(
                member_factor, cross.T, check_finite=False)
        else:
            conditional_mean = predicted
            conditional_covariance = covariance
        factor = cho_factor(conditional_covariance, lower=True, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise ValueError('joint or conditional covariance is not positive definite') from error
    residual = observed-conditional_mean
    quadratic = float(residual @ cho_solve(factor, residual, check_finite=False))
    logdet = 2.0*float(np.sum(np.log(np.diag(factor[0]))))
    return -0.5*(quadratic+logdet+2.0*math.log(2.0*math.pi))
