# R2 block elliptical-slice feasibility pilot — 2026-10-21

## Decision and scientific scope

Driver disposition: **proceed with one bounded feasibility sweep only**. Astra's
read-only advisory was CONDITIONAL: blockwise elliptical slice sampling (ESS)
is a materially different, exactly likelihood-corrected route from the closed
HMC/LIS branch, but a single sweep is not posterior evidence. The current
curvature screen remains inconclusive; CIC knot crossings are a possible
gradient-regularity issue, not an established cause.

The pilot retains the unchanged conditional-v6 target: 47,121 training counts
(37,951 keys) and 1,414 linked raw-FP marks, at N256 / 384 cMpc/h
(1.5 cMpc/h nominal field spacing), with all 24 standard-normal nuisance
coordinates and their priors. The likelihood score is the full count plus
linked-mark score, evaluated with fresh support at every candidate. No heldout
values, truth identities, new data, gravity/RAMSES run, map, sample or field
readout is used or produced.

The anchor is the explicitly discarded `next_index=60` HMC transport state,
replayed through the saved same-target gradient contract. It is a numerical
starting point only, not a posterior draw. The target API returns matched
GL2 value-and-gradient primals; ESS uses only their direct count and mark
score terms, not the gradient. For each ESS proposal the existing
`obs.derivative` primal is retained to avoid the separately compiled
value-only route that differed by 0.005473 nat on a prior saved state; its
observation-side AD gradients are discarded. The PMWD pullback is skipped,
and the evolved density/velocity field is reused for nuisance-only proposals.
Fresh tracer-dependent support is still rebuilt and checked at every proposal.
The saved-state anchor replay remains a full value-and-gradient check.

## Frozen transition

All coordinates have independent standard-normal priors. For a fixed,
orthogonal projector `P_b` and a fresh `nu ~ N(0,I)`, update only block `b`
along

`q(theta) = (I-P_b)q + (P_b q) cos(theta) + (P_b nu) sin(theta)`.

The slice threshold uses the **complete count-plus-mark likelihood score**.
Every candidate uses the exact unchanged target; no coarse force, truncated
likelihood, or surrogate decides acceptance. A genuine zero-likelihood point
is `-inf` and is rejected by bracket shrinkage. NaN, support-capacity,
memory, or execution errors stop the pilot; they are not converted to zero
likelihood. If the predeclared bracket cap is exhausted, that block is
incomplete, not a completed transition.

One fixed systematic sweep has three blocks:

1. Real IC Fourier modes with integer radial `|k| <= 64`.
2. Their full orthogonal complement in the N256 IC field.
3. All 24 nuisance coordinates.

The radial split is a fixed engineering partition motivated by the N128 count
grid scale. It is **not** a claim that marks or nonlinear evolution contain
no information above `k=64`; every block is scored by the same full likelihood.
Real-valued iid auxiliaries are projected with a conjugate-symmetric rFFT
mask; no unconstrained complex draw or norm renormalization is used. Each
conditional block kernel preserves the full target. The sequential sweep is
invariant, though it is not generally itself reversible.

## Hard resource and interpretation limits

- At most 8 proposal likelihood evaluations per block; 24 total, plus the
  one required saved-state replay (25 target evaluations maximum).
- 3-hour application deadline inside a 3.5-hour Slurm allocation, one
  typed GPU, 48 GiB host memory request, and a fail-closed 38.4 GiB RSS guard
  (20% retained headroom).
- Fresh 16,384-cell support blocks. Existing target/device/workspace guards
  remain unchanged. No automatic retry, second sweep, angle/window tuning,
  Fourier-boundary ladder, or support-ceiling increase.
- H200 is checked first (`--gres=gpu:H200:1`); if occupied, select one
  compatible H100 or A100 typed request and record all three availability
  counts and the chosen partition/GRES. No manual node execution.
- Save only scalar target-evaluation and per-block timing/bracket/displacement
  diagnostics, existing low-band readouts and peak RSS. Do not write a field
  or promote the last state.
- Record whether each scalar likelihood proposal reused the cached PMWD field;
  cache keys are exact IC-coordinate equality, never approximate state
  matching.

A completed sweep establishes only that this exact block transition can be
executed under the cap. ESS has no MH rejection rate; the first slice point
inside the bracket is accepted by construction. One accepted sweep cannot
establish stationarity, mixing, effective sample size, calibrated uncertainty,
heldout prediction, a z=0 posterior, or a full-survey calibrated CF4 result.
If it fails the budget or target guards, stop this branch rather than tune it
in place. If it completes, require a separately justified multi-state
assessment before any inference claim.

## Q-GOAL / Q-LEAN and LG handoff

**Q-GOAL:** this is one bounded method-feasibility step toward the first
conditional z=0 density/velocity posterior in the approved latent-IC/evolved-
state route. It does not establish mark-presence ignorability, full-survey
selection/group inclusion, calibrated uncertainty, or heldout prediction.
MW/M31 remain role-ambiguous and M33 unresolved; their observables must later
constrain those same roles on the same NEW evolved LG field at
`<=0.3 cMpc/h`. Native truth IDs remain calibration/evaluation labels only,
never seeds or candidate selectors.

**Q-LEAN:** reuse the existing exact target, saved-state replay and Slurm
infrastructure; one three-block sweep, no sampler/metric ladder, source
census, heldout read, independent gravity run, field rendering, or R3 entry.

R2 remains **NO-GO** until a defensible field posterior, uncertainty and
untouched conditional heldout prediction are delivered. R3 has not started.
