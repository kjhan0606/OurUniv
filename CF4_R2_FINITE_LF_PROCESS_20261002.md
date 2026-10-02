# R2 finite selected-LF process — 2026-10-02

Status at submission: implementation commit `2e6c8aa`; focused Slurm
regression job `410407` is pending. No fit, chain, held-out score, PMWD
evolution, map promotion or IC generation was run.

## Change and identity

The active count + conditional raw-FP target previously factored each
selected source contribution as

`rate * response_b * I_b(alpha)/I_R(alpha) * I_selected(alpha)/I_b(alpha)`.

Here `I_b` is the Schechter measure over one of five true-K bins, `I_R` is its
measure on the finite `[-25,-21]` reference window, and `I_selected` is the
intersection with the apparent-K selection and one observed absolute-K bin.
The two `I_b` factors cancel. The active target now evaluates the equivalent
finite expression directly:

`reference_rate * response_b * I_selected(alpha)/I_R(alpha)`.

The selected intersection is finite because every observed absolute-K bin is
bounded. Its integral and the reference denominator therefore remain finite
for alpha at, above or below `-1`; the target no longer needs to define an
unobserved all-faint galaxy total. Twelve-point Gauss-Legendre integration is
streamed over the 30 population/bin intersections to bound quadrature
workspace. The old all-tail helpers and default transfer remain intact for
historical and other callers.

The active nuisance coordinate is now `alpha = -1 + 0.5*u7`, with the
existing `u7 ~ Normal(0,1)` prior. This broad `Normal(-1,0.5^2)` is a
development regularizer centered on the same-survey LF reference, not an
independent calibration. It places the paper's alternative `alpha=-0.73`
within about `0.54` prior standard deviations; the previous positive-only
transform put it about three standard-normal units away. The Mstar coordinate
and reference-rate center are unchanged. Sampling remains prohibited until
selection/association and independent-or-joint calibration are defensible;
prior sensitivity is still required.

Existing saved tracer vectors use the old transform. To preserve their old
physical alpha when explicitly replaying a state, convert
`u7_new = 2*(alpha_old+1) = 0.12*exp(0.5*u7_old)`. Reusing the old vector
unchanged would change the physical state and prior.

## Scope and decision limits

- R2 model/normalization repair only. It does not calibrate LF shape, bias,
  selection, source association, group covariance, RSD/FoG or the full
  likelihood, and does not make a posterior or held-out prediction.
- Q-GOAL: this removes a mathematically unnecessary unbounded-tail restriction
  in the active z=0 observation target; it does not identify MW/M31 or resolve
  M33. Those observables must later constrain all three latent roles on the
  same NEW field at `<=0.3 cMpc/h`; native identities remain evaluation-only.
- Q-LEAN: one algebraic cancellation, a finite selected-window quadrature and
  focused value/gradient regressions; no new simulation, chain, data sweep or
  extra likelihood factor.
- R2 remains NO-GO for posterior promotion at N256/1.5 cMpc/h. MW/M31 remain
  ambiguous and M33 unresolved. The actual z=0 posterior is still the first
  science delivery; R3–R5 remain downstream.

## Validation

The single Slurm job `410407` runs the focused LF-transfer, source-mass,
volume-count and raw-support test files on the H200/H100/A100 partitions,
requesting one GPU, two CPUs, 12 GiB host memory and 30 minutes. The intended
checks are legacy-factorization value/gradient equality for `alpha>-1`,
finite values/derivatives across `alpha=-1`, count-volume equivalence, and
support wiring. Results and any limitations are appended after completion.
