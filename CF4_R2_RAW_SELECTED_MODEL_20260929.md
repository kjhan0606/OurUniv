# R2 raw selected-observable model: design in progress, not a promoted target

R1 -> **R2 ongoing** -> R3 same-field LG -> R4 precision evolution -> R5 zoom.
The short HMC bundle is complete; no sampler or gravity job is active at the
start of this design. Its apparent mixing improvement cannot cure selection
or shared-calibration assumptions. Do not restart it merely to stay busy.

## Required factorization (driver derivation)

Let C denote already-scored count data; u the retained point covariates; j a
latent source state; x the NEW raw FP/photometric measurements; B the mark
selection event; psi shared population/calibration parameters. Start with
candidate weights w_j(C,u,F), prior to the additional mark selection, and a
normalized raw-data law f_j(x|u,psi). If selection is b_j(x,u), the mark factor is

    p(x | C,u,B,F,psi)
      = sum_j w_j b_j(x,u) f_j(x|u,psi)
        / sum_j w_j Z_j(u,psi),
    Z_j = integral b_j(x,u) f_j(x|u,psi) dx.

This integrates to one over x. It scores count occurrence only once, provided
the weights actually come from the declared count/covariate model. Selected
candidate weights are proportional to w_j Z_j, while their selected data laws
are b_j f_j/Z_j. Both are required; using only the former is wrong. The current
eta likelihood ratios cannot stand in for f_j, which must be a normalized
density over observations. Shared psi is inferred jointly, never fitted to
these rows and then reused as an independent prior on the same rows.

Conditioning on optical/K flux to simplify an inverse regression changes
w_j to p(j|C,u,flux,F,psi); it cannot retain old flux-marginal candidate weights
without an independence argument. Conditioning on a measured quantity does
not condition on its inferred intrinsic luminosity at every candidate distance.
Known deterministic magnitude cuts are distinct from type, matching, richness
and one-mark/one-point graph incidence. The latter are not calibrated by a
normalizer for magnitude cuts alone. No new implementation claims otherwise.

## Source conventions now established

[Howlett2022 equations3,5,17 and section3.1.1](https://arxiv.org/html/2201.03112)
specify that replacing the group-redshift distance with a candidate distance
shifts log radius by minus eta. Aperture-corrected s differs from log measured
stellar dispersion; a raw-dispersion cut must be transformed consistently.
Their optical error prescription has perfectly anticorrelated radius and
surface-brightness errors and no photometric/spectroscopic error correlation.
This is a published modelling prescription, not an independently measured
full covariance matrix. The extra nonlinear-radius term used when fitting
at redshift distance must not duplicate explicitly modelled velocity scatter.

The local raw join does not yet carry aperture/plate/raw-dispersion columns.
They EXIST in the source header and can be joined without new observations.
The absence of a K-error column remains real. Optical/K magnitude systems and
selection corrections cannot be silently converted or assumed error-free.
Nor should source rejection/morphology selection be called a known rectangular
cut merely because the magnitude and dispersion bounds are known.

## Decision pending focused advice, not routine permission

One read-only Fable5 request is in
`config/cf4_r2_raw_selected_model_advice_20260929.md`. It concerns a substantive
observation-law revision, not another audit gate or reopening its previous
adverse advice. Driver will verify the proposed law and choose the minimum
implementable action. No heldout fit, source-prior broadening, optical refit,
new model fitting or large N256 calculation has occurred in this bundle yet.

Q-GOAL: a normalized observation model must improve actual CF4 constraints on
the SAME present field used later for MW/M31/M33; the labels remain ambiguous,
M33 unresolved. No truth IDs supply generated candidates. Q-LEAN: reuse real
raw columns and count machinery; do not manufacture a new generic calibration
framework, synthetic coverage claims about real data, or long simulations.

## Advice received; corrected implementation decision

Fable returned ADVISE PROCEED. Adopt the raw selected-observable direction,
but NOT its proposed factor as written:

- Correlated Gaussian half-space probabilities do not factor into independent
  Phi terms. For correlation1/2, the positive orthant probability is1/3, not1/4.
- A Gaussian optical/K population generally implies a different K marginal
  from the count model's Schechter law. Dividing just a marginal K Gaussian
  by a bin probability does not repair its correlated optical law or all
  selection normalizers. Preserve the existing count LF explicitly instead.
- Published FP coefficients fitted using overlapping galaxies are not an
  independent external calibration prior. A global population Gaussian does
  not reproduce every richness-corrected released skew PDF.
- Known cuts and a finite parameterization do not prove identifiability.
  Matching, one-link graph incidence, nonrectangular quality/rejection cuts,
  shared calibration and radial-kernel limitations remain uncalibrated.
- No K-error column means intrinsic and measurement scatter are not separately
  identified. It does not justify calling all cross-band error a single
  independent Gaussian with known covariance.

One bounded implementation bundle: implement the exact conditional-mark
algebra with the SAME Schechter within-bin law and a conditional optical
Gaussian, including its correlated two-cut normalizer. Treat corrected
catalogue K as the count operator's luminosity PROXY convention, not an
error-free physical luminosity measurement. For a candidate with allowed
proxy interval I_j, use q_j(M)=phi(M)/integral_I_j phi and
g_j(x|M)=N_3(x;mu+beta(M-M0)+eta_j e_r,Sigma+E).

The numerator is sum_j w_j q_j(M_observed,j) g_j(x|M_observed,j).
The denominator is sum_j w_j integral_I_j q_j(M) Pr(optical cuts|M,j)dM.
The observed-K-to-M translation has unit derivative; K count-bin occurrence
is already in w_j. This scores the within-bin K value jointly with raw optical
data, NOT an additional count or the published eta PDF. Group/type incidence
would enter w_j explicitly; setting it constant remains a working assumption.

First implement/test this necessary likelihood component plus actual source
cut/error geometry on the existing1414-row training cohort. No synthetic
test will be called calibration of actual selection. No new PM, frozen-field
parameter fit, live-target replacement, independent-prior claim, stationary
sampling or N256 in this bounded component bundle. Fitting shared parameters
and interfacing actual source weights are substantive next work, not implied
by a kernel test. Do not add many generic audit gates or require a new user
approval for the ordinary continuation.
