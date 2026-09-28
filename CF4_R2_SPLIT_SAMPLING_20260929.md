# R2 uncertainty-side feasibility, not posterior delivery

## Execution update

408296/sourcef63eb13 is the ONE real-target feasibility pilot, submitted
after final count/FP values both reproduced to1e-7. H1002CPU24GiB,90minSlurm,
75min application;16 warmup+16 retained proposals,2 steps each, same1414 target.
It is not a posterior-delivery or large-chain authorization.

Correction:408224 ran the default GH/CDF16/32 readout because the submission
omitted CF4_R2_CDF_FINE=1. It was not the promised4x32/8x32 check.408276 PDF
then failed on the absent comparison key; its partial output is preserved.
408282/v14 reran ONLY the saved-field readout with the explicit fine option:
4m55s,exact fitted count score,4x32/8x32 delta+.0003436150,exposure relativeL1
2.0363e-7,scalar velocity-gradient discrepancy4.3061e-8. No PM or fit repeated.
408284 produced the corrected6-page Korean report in
`/gpfs/kjhan/CF4/z0_density/r2_single_mark_report_v2/R2_진행보고_예제그림.pdf`;
all6 rendered pages were visually inspected, with actual figures for every
reported test. Training count total46432.72vs47121 and radial shape residuals
are limitations, not proof of selection calibration.

408222 ended normally after32 accepted updates/42m57s, iteration limit rather
than convergence. Its final objective is144835.83910873727, FP log factor
-31.073932470664015; the saved-field readout408225 independently reproduces
that factor. FP shared zero+.0247701845dex=6.1925 priorSD, training correlation
-.003795 and residual mean+.0191765dex remain limitations, not calibrated
distance recovery.408224 count readout must finish before the proposed pilot.
408273 passed all7 mechanics/adapter tests, including checkpoint units and
rejection-preserving finite-target mechanics. The real-target adapter is now
implemented; no CF4 sampling has yet run. Future pilot reporting must not
mistake the last rejected force evaluation for the last accepted state.

The Korean six-page example-figure PDF is queued as408276 after readouts.
408226 was canceled while pending to correct SGZ velocity labeling (not LOS)
and add page rendering for visual verification. No science calculation was
canceled, and no result was removed.

## Original prospective plan and advisory disposition

R1 -> **R2 in progress** -> R3 same-state MW/M31/M33 -> R4 -> R5.
The1414-row working-target fit408222 is still running when this plan is
recorded. Its finite endpoint and readouts must be inspected before launch.
No R2, N256 or calibrated uncertainty product is claimed.

## Why this is needed, and what it does not repair

A MAP field cannot supply posterior spread. The old identity-mass HMC line
with ESS2–5 remains closed. A finite-dimensional prior-split HMC implementation
now exactly rotates the white Gaussian prior/kinetic Hamiltonian, then uses
nonlinear likelihood kicks and a FULL-target Metropolis correction. It keeps
rejected states and has fixed positive Fourier/nuisance proposal metrics.
It is not a replacement power spectrum, random high-k painting, Laplace
approximation or claim of dimension-independent mixing.

Small tests408227 passed4 controls;408228 passed5 including nonconstant Fourier
preconditioning and the100*tracer coordinate adapter.408237 is the augmented
6-test run covering fixed retained-step size and independent endpoint energy
agreement. No CF4 chain has run in these tests. Code:
`scripts/cf4_r2_prior_split_hmc.py`, `tests/test_cf4_r2_prior_split_hmc.py`.
Method background: [Beskos et al.2011](https://authors.library.caltech.edu/records/kbprr-5m424).
Only the finite-dimensional implementation tested here is asserted.

The scientific limits in `CF4_R2_FIELD_LEVERAGE_20260929.md` remain: selected
association, tracer/LOS model and common source-FP covariance are not calibrated.
No chain spread from this short pilot may become observational error bars,
R3 candidate selection or productionIC. MW/M31 remain ambiguous and M33
unresolved; their later observables constrain the SAME NEW physical state,
not truth-supplied components. Heldout stays untouched.

## Fable advice and driver assessment

Read-only Fable5 reviewed the preceding evidence record and split mechanics.
Verdict PROCEED as bounded feasibility, not production. It agreed that a fixed
SPD proposal metric can be approximate without changing the MH target, and
requested mode/zero displacements, energy errors, frozen metric arrays, and a
real-state canonical-coordinate seam check. Adopt these practical amendments.
Prompt: `config/cf4_r2_split_sampling_advice_20260929.md`.

Correct/reject these parts of its reasoning:

- Absolute FP/count log-score values do NOT bound their relative curvature
  or information. Their additive reference constants can change arbitrarily.
  Do not infer that FP cannot dominate some directions from -45 vs146000.
- The finite-grid high-k mass of the proposed inverse-Laplacian metric is
  not exactly1. Only its large-k limit is1;6000 at the fundamental is an
  explicitly borrowed stiffness guess, not a verified interpolated Hessian.
- Shared source covariance is not supplied to the current target. We have
  NOT established that all possible source data/refitting routes lack it.
  The sampling pilot does not resolve or waive that scientific work.
- Feasibility controls alone neither prove stationarity nor certify a physical
  posterior; no16-state ESS/coverage claim or arbitrary promotion threshold.

Q-GOAL: acquire a usable route from a fitted state to conditional ensemble
inference. Q-LEAN: one modest pilot, same field oracle, no Hessian sweep,
new gravity suite, likelihood change, heldout tuning or TNG dependency.

## Bounded prospective action

After408222 and its saved-field readouts, if the endpoint is finite and the
target reproduces, use that endpoint for ONE pilot:16 discarded warmup+16
fixed-step proposals,2 integration steps/proposal; startstep.1. Warmup alone
adapts to acceptance within[1e-6,.3], halves on three consecutive rejections;
the last warmup step is frozen. Record all states, including rejections, as
scalar state diagnostics and a rolling checkpoint, not large snapshot dumps.

Canonical variables are (whiteIC,white9tracers,whiteFPzero). The existing
optimizer uses100*white_tracer, so both coordinates AND gradients are mapped
with the exact chain rule; the prior is still independent unit Gaussian.
IC inverse mass C(k)=[1+5999/|integer_mode|^2]^-1, k!=0, C(0)=1. Transform the
saved SPD conditional secant metric from optimizer to canonical nuisance units
as D^-1 H_x D^-1 (D=diag(100,...,100,1)). Save both actual matrices. These are
proposal-efficiency guesses, never new priors or reported uncertainty.

Every force rebuilds candidate support; every proposal endpoint also uses the
independent value-only path for MH energy and checks derivative-primal agreement
at the unchanged1e-7 numerical tolerance. Only genuine infinite potential/zero
target support may be rejected normally. Finite-target bad derivatives or
capacity/undefined-model errors stop rather than silently restrict posterior.

H1002CPU24GiB;75min application/90min Slurm.64gradients at~44s are~47min;
startup compilation, endpoint value comparisons and output leave budget margin.
If time expires mid-proposal, retain the previous accepted state. No automatic
repeat or longer chain: assess acceptance, energy, actual jump sizes, IC power
drift and conditional-zero movement first. This fits inside the prospective
4GPU-hour bundle envelope together with the single joint fit and readouts;
it is not a bound on the cost of completing all R2.
