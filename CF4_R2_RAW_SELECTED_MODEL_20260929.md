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

408343/sourcebb68681 COMPLETED7s: six component tests passed. Correlated
orthant probabilities agree with the independent analytic formula to9.22e-8;
joint example density integrates to1.0000000000000013. All1414 actual rows
are inside the known magnitude/raw-sigma cuts. Reconstructing aperture-corrected
s from raw dispersion, angular radius and plate gives maximum discrepancy
5.02e-6dex, consistent with released precision. Host peak0.124GiB. Driver
viewed both rendered pages of `r2_raw_selected_component_v1/`'s Korean PDF.
This is component correctness/source geometry, NOT fitted survey calibration.

Continue immediately with the existing accepted408337 field: export its
UNCOLLAPSED true-K/source distance mixture and exact count-LF intersections
for the same1414 training links. No PM evolution, parameter fitting or new
field sampling. Reconstruct the OLD FP score from the sparse export before
using the cache for a replacement raw model, so data/weights cannot silently
change. The current accepted-state checkpoint, not the last rejected proposal,
is authoritative. H1002CPU8GiB/15min bound using the existing saved-field
readout, with ~6GiB estimated host demand plus headroom. No extra large field
snapshots; retain sparse positive-weight components only. This connects the
new component to the actual same field, not another synthetic universe.

408344/source7eaaf6a COMPLETED2m44s, no PM. The sparse cache contains219401
positive source/bin components for1414 rows (maximum332 per row). Recombining
them reproduces every old FP row to1.55e-15; accepted total-4.73896813107675.
The count LF at this endpoint is Mstar=-23.6011745, alpha=-.93715799; these
ACTUAL nuisance values must be used, not the defaults. Host MaxRSS4.73GiB.

Next bounded action: a fixed-field JOINT raw-population feasibility fit on
these SAME1414 rows, with the selected raw likelihood, not a refit treated
as an independent prior. Use a single conditional optical Gaussian whose
mean is affine in the count luminosity proxy and log(1+source richness),
plus a common positive-definite intrinsic covariance (15 shared parameters).
This is an explicit working population model; its adequacy/transfer is not
established. The r intercept absorbs the fixed-field distance-scale convention;
do NOT add an exactly redundant free zero or the old .004 prior here. Source
optical error prescription stays fixed. Morphology/graph incidence remains
constant as a declared unresolved assumption, not a calibrated quantity.

Fit at most32 L-BFGS updates/10 application minutes, H1002CPU8GiB/15min Slurm,
with differentiable correlated-cut quadrature and exact saved source weights.
No candidate truncation, fixed best-distance approximation or hidden K-bin
collapse. Preserve all training rows. Use CPU reference values and a directional
finite difference before fitting; compare quadrature at the terminal state.
These are direct necessities for a NEW likelihood implementation, not an
independent science-validation claim or new generic gate hierarchy.

Mean/covariance initializers are derived from these rows only as optimizer
starts, NOT data-derived priors. This fixed field already used published
summaries of the same observations, so the fit cannot independently validate
that field or establish absolute FP calibration. It prepares a replacement
raw observation component and exposes fit cost/degeneracy. No same-field
posterior, heldout score, source-wide selection solution or R2 completion is
implied. No automatic large HMC or resolution upgrade follows the fit.

408345/source0937c6f COMPLETED59s. The fixed-field15-parameter fit stopped
after29 iterations by relative objective reduction, with scaled gradient
norm6.14e-5. Total log-density improvement over the data-derived least-squares
initializer is only~0.0223nat: no dramatic scientific improvement is claimed.
Initial/final CPU-row comparisons are4.44e-15/1.60e-14; directional derivative
relative discrepancy2.32e-12. Cut order64->96 changes the total score by
0.000203nat (max row2.11e-6). Host peak2.39GiB, device temporary5.65GiB.
Driver viewed the rendered Korean PDF. These are feasible raw-likelihood
calculations and a conditional fit, not independent validation or calibration.

Immediately check the fitted model against the ACTUAL training observables:
draw selected optical/K marks from its generative law at the same saved source
mixtures, and compare observed ranks and distributions. Use the same Schechter
intervals, correlated optical errors, richness and cuts; never perturb the
actual field or draw a new cosmological simulation.1000 accepted mark draws
per row, fixed seed, max2e7 proposals, one5min/1CPU/1GiB Slurm allocation.
This is a training model-misspecification check, NOT heldout validation,
posterior predictive UQ, a p-value or proof of physical selection transfer.
It can expose a bad raw population/incidence law before expensive field
inference. No outcome-dependent row deletion or threshold adaptation.

408346/source6704743 COMPLETED26s:1,414,000 accepted mark draws from2,828,000
proposals. The four marginal training mean ranks are(.4968,.4961,.5013,.5067)
for(r_z,s,i,K); fractions below.1 are(.0955,.0912,.0926,.0884) and above.9
(.0919,.0955,.1033,.0891). Known optical-cut acceptance ranges.9725–1 per row.
Driver viewed both PDF pages, including three predetermined real-object
examples. No gross marginal failure appears, but matching marginals neither
proves the joint distribution nor calibrates actual physical selection, field
uncertainty, graph incidence or transfer. No p-value/heldout claim is made.

Driver refinement: lack of an external absolute anchor must not be equated
with impossibility of ANY conditional field posterior. A proper joint model
can infer population parameters and cosmic field together, with explicit
prior dependence and weak common-scale directions. [Dam2020 sections3.4/4](https://arxiv.org/html/2002.05898)
provides a primary-source precedent for shared FP calibration and velocity
inference; its linear/mocked setup is NOT validation of our nonlinear target.
Do not add a redundant zero coordinate or silently fix these15 parameters.
Plan and weak-prior proposal sent for focused advice in
`config/cf4_r2_joint_raw_field_advice_20260929.md`; no large calculation or
live target replacement is authorized by the advice request itself.

## Joint-model advice disposition and source-cell comparison

Fable returned ADVISE PROCEED, choosing source-cell quadrature BEFORE the live
raw-field adapter. Driver adopts that priority: the current sigma_LOS~38km/s
corresponds to .38cMpc/h, much smaller than a3cMpc/h source cell. This is a
testable numerical concern, NOT proof that observed field features are artifacts.

Driver corrects substantive advice errors: our density normalizes jointly over
optical x AND within-bin observed K, not x alone. A hypothetical d->lambda*d
has eta->eta-log10(lambda), not plus; with M->M-5log10(lambda), mean invariance
would require b0->b0+(e_r+5*bM)*log10(lambda). Fixed LF/bins, counts and field
dynamics break the asserted exact full-target gauge. A weak common-scale
direction remains plausible and must be measured. Richness is a FIXED observed
covariate, not another live-field gradient. Proper priors alone are not a
universal posterior-integrability proof. Training marginal agreement is not
physical-selection calibration. The suggested .01nat/row gate is arbitrary:
measure value/derivative convergence rather than manufacture a passing cutoff.

Next bounded comparison uses the SAME408337 field and15 fitted parameters,
first four source-label-ordered training rows per population (24 total), fixed
before scores. Compare1,2^3,4^3 Gauss-Legendre nodes per source cell; preserve
cell mass, velocity and angular completeness. Conservative fresh source-cell
support includes global coherent displacement,8sigma, TSC radius and cell
half-diagonal. Not merely subdividing surviving centre components. Read raw
mark values/coherent-velocity derivatives for24rows and count-key means for
the first row per population. Counts reuse boundary-fitted4x32 CDF integration;
the scalar gather must match full deposition and its derivative. Gaussian
tails/source aliases and cell-constant completeness remain declared limits.
No PM evolution, calibration refit, heldout scoring, new cosmology or ensemble.
One H100/2CPU/8GiB/20min job; memory headroom above prior~4.7GiB readout.
Outputs small JSON/NPZ and a Korean PDF with actual-object examples per test.
MW/M31 identities remain ambiguous and M33 unresolved; this changes the
observation link to the SAME NEW field, never seeds named objects or truths.

Finite R2 destination adopted: actual1–2cMpc/h conditional field maps and UQ,
multiple-chain mixing evidence, one untouched predictive evaluation and
quantified model/numerical/prior limitations. No claim that a universal
absolute-calibration solution is prerequisite for any conditional delivery;
also no3cMpc/h mechanics pilot labelled R2 completion. Large-chain/resolution
scope remains a separate evidence-based decision, not automatic advice authority.

408347/source5c29748 terminated FAILED9m57s AFTER all72 comparisons, because
the centre reproduction error1.615e-7 exceeded1e-7. All13 unit tests passed;
directional derivative discrepancy maximum2.86e-6 passed its2e-5 check. Results
are preserved, not relabelled PASS. Host2.74GiB. Source centres were promoted
from archive float32 to float64 by the subnode constructor; the old eta/true
radius used original geometry precision. This is a hypothesis for the small
endpoint difference, not a proven explanation until the next exact control.

Substantive measured effect: first-six count means change~5–12% centre->4^3,
whereas2^3->4^3 changes<.5% for these rows. Raw velocity directional scores can
change much more than the log densities themselves;4^3 convergence is not
yet demonstrated. Next bounded followup: reproduce ALL24 original centres in
their native geometry precision (same1e-7 tolerance), then8^3 on the six
prespecified count examples plus the largest2->4 raw-gradient discrepancy.
That last example is explicitly outcome-selected for a numerical stress test,
not heldout evaluation or a representative population estimate. Reuse old
1/2/4 results; no full rerun, new field or fit. Tighten conservative support
with EACH source cell's own fixed velocity bound, never its likelihood weight.
One H1002CPU8GiB20min job. Keep failed predecessor and precision-control result
separate; no post hoc loosening of the implementation-reproduction tolerance.

408348/source9c3868e COMPLETED6m10s. Original-precision centre reproduction
error1.27e-13 verifies that the predecessor's1.6e-7 discrepancy was geometry
precision, not likelihood inconsistency. No threshold was loosened. The seven
4^3->8^3 raw log-density differences have max.001080nat; velocity-direction
differences max.016929. These are local numerical controls, not a global error
bound. Six count means change by at most.0463% between4^3 and8^3, compared with
roughly5–12% centre->volume changes. Host2.98GiB. Two-page actual-object Korean
PDF rendered and viewed. R2 remains open; named LG roles still unresolved.

Driver decision: do not keep the point-source likelihood solely because its
own derivatives are consistent. Implement a streaming source-volume raw mark
adapter with LIVE tracer parameters, shared15 population parameters and field
mass/velocity dependence. Use4^3 as a development rule with the measured8^3
residual explicitly retained, not declared exact. First exercise the same
seven rows, no fit/PM/heldout: CPU endpoint, analytic velocity derivative and
joint directional finite difference. The count target is NOT yet replaced.
Proper weak Gaussian population priors from the advice prompt are explicit
modelling choices; no fitted covariance is made into an independent prior and
no redundant old FP zero is added. Stream small chunks with rematerialization
instead of a dense whole-cohort source-by-cut matrix. One H100/2CPU/8GiB/20min
readout pilot; measure actual cost before expanding to all1414 or chains.

408349/source69b52a1 COMPLETED8m03s: seven raw values reproduce to2.1e-14,
velocity derivatives to6.3e-14; joint26-coordinate directional discrepancies
max2.55e-8. Temporary GPU memory.829GiB. The dense streaming implementation
costs10.9–12.5s PER ROW/gradient: do not extrapolate it into costly whole-cohort
chains. Next implementation-only action packs individually contributing
(source subnode,true-K-bin) components, keeping allpositive components at the
current AND two local finite-difference states. Compare all26 gradient entries
against the dense reference with the same1e-7 tolerance. This is an explicit
state-refreshed local control, NEVER a frozen support cache for future HMC.
No likelihood/population/field prior changes, refit or large computation.
