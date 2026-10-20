# R2 local curvature-range screen — 2026-10-21

## Decision and scope

Status: bounded diagnostic design and implementation adopted after driver
verification and an Astra CONDITIONAL GO; targeted tests pass. The single
Slurm calculation has not yet been submitted. This is a matrix-free local
falsification at one saved state, **not** an N256
likelihood-informed-subspace (LIS) sampler, posterior, field delivery, or
authorization for an automatic follow-up.

The active estimand is unchanged: the conditional v6 training target
(47,121 count rows and 1,414 linked selected marks) on the 384 cMpc/h box at
nominal 1.5 cMpc/h field sampling. Mark-presence ignorability, full-survey
selection/inclusion, held-out prediction, shared covariance and calibrated
uncertainty remain unresolved. Do not describe this as a full-survey
calibrated CF4 posterior.

## Existing evidence and driver check

The terminal exact-GL2 checkpoints at `next_index=38, 40, 60` have identical
sampler fingerprints: N=256, box=384 cMpc/h, matched order-2 value/gradient,
the same frozen v6 split and association ledger, and the same proposal metric.
Their saved full gradients have shape `N^3 + 24`. Source-commit comparison
found only restart-allowlist changes and an optional field-readout change in
the target-building path; the GL2 target itself is unchanged. The checkpoints
are dependent, explicitly discarded transport states—not posterior samples.

Driver-computed pairwise total-gradient and conditional negative-log-
likelihood-gradient cosines are:

| Checkpoint pair | IC total | IC likelihood-NLL | nuisance total | nuisance likelihood-NLL |
| --- | ---: | ---: | ---: | ---: |
| 38 ↔ 40 | 0.347671 | -0.141293 | -0.715852 | -0.715688 |
| 40 ↔ 60 | 0.005299 | 0.000728 | 0.808974 | 0.808913 |
| 38 ↔ 60 | 0.011534 | 0.155095 | -0.409425 | -0.408965 |

Here the likelihood-NLL gradient is `g_target - q`, subtracting the known
unit-Gaussian prior gradient in canonical coordinates. These cosines quantify
strong state dependence and near-orthogonal IC scores across states40↔60;
they do not estimate curvature rank, stationarity, mixing or a posterior
covariance. The older low-k A/B report
compares endpoints whose low-k coefficients are close but whose complete q
states differ; it is not an identical-q oracle discrepancy. It is likewise
not a local Hessian measurement.

The active target already uses standard-normal canonical coordinates for the
IC and all 24 nuisance coordinates. Therefore the local negative-log-
likelihood Hessian action is measured as
`[(g_target(q+h v)-g_target(q-h v))/(2h)] - v` for an IC-only unit direction
v. The HMC momentum mass is not used as prior whitening. Nuisance-coordinate
gradient responses are retained to expose IC–nuisance coupling, while the
screen does not reduce or integrate out nuisance coordinates.

## Frozen single-anchor screen

- Anchor: the terminal `next_index=60` state from job 421553. Replay its exact
  value, terms and full gradient first; stop on any mismatch.
- Reuse the existing N256 GL2 conditional-v6 oracle. Rebuild shifted support
  for every distinct candidate using 16,384-cell chunks. No held-out count or
  FP values are read.
- Draw six deterministic, mutually orthonormal Gaussian directions in the
  16,777,216-dimensional IC-white block (seed 20261021); nuisance inputs stay
  fixed. The first four form a rank-at-most-four response range; the last two
  are untouched validation directions.
- Use centered finite differences at h=0.1 in canonical prior coordinates.
  Repeat sketch direction 1 and validation direction 1 at h/2=0.05.
- Hard cap: 17 exact target gradients total (one checkpoint replay, twelve
  +/- evaluations at h, four +/- half-step evaluations). Stop before starting
  a pair that cannot fit both evaluations; never extend or increase rank
  automatically.
- Local screen thresholds are predeclared: each repeated IC and joint action
  changes by at most 5% between h and h/2; both validation IC responses have
  at most 10% of squared energy outside the four-response range; each IC
  response exceeds ten times the replay/numerical noise proxy. Failure of
  numerical stability is inconclusive. A validation failure falsifies this
  rank-4 local screen. A pass only justifies a separately costed proposal
  design; it does not authorize sampling.
- Preserve Hessian indefiniteness; form neither a positive Fisher substitute
  nor a dense Hessian. Keep at most about 4 GiB added host workspace. Record
  the response norms, small sketch singular values, support-change indicators,
  all 24 nuisance cross-responses, and every target evaluation. Do not write
  probe fields or promote a best perturbation to a candidate.

The code reuses `cf4_r2_conditional_map.main` with its existing replay guard
and `sampling_driver` hook. Only the unnecessary per-evaluation present-field
readout and best-parameter-file output are disabled for this diagnostic; the
target, force order, priors, support rule and likelihood are unchanged.
Six new pure-math tests pass. The existing map-budget, linked-radius and
count-exposure test files also pass in the Slurm `circle` Python environment;
`py_compile`, Slurm-script `bash -n`, and `git diff --check` pass. The login
node's default Python lacks JAX, so GL2 wiring tests were run with the
project's `circle` environment; no target evaluation was run locally.

## Cost and execution limits

Job 421553 used 184 forces in 68,308 seconds, about 371 seconds per force,
with an 18.78 GiB host peak. Seventeen exact gradients therefore suggest
about 105 minutes of force work, not a guaranteed runtime. Adopt Astra's
3-hour application cap and 3.5-hour Slurm cap, one GPU and 48 GiB host memory.
The script stops if process peak RSS reaches 38.4 GiB, retaining 20% of the
requested memory. No login-node target evaluation or manual GPU execution.

Submission procedure: inspect H200 first and, if idle, use `h200` with
`--gres=gpu:H200:1`; otherwise inspect the permitted H100 and A100 modes and
submit one compatible typed-GRES job. Record the selected partition and GRES
in the Slurm environment and result JSON. The job has not yet been submitted.

## Audit and goal alignment

Astra's 2026-10-21 mid-course advisory is CONDITIONAL GO for exactly this local
curvature-range falsification, and NO-GO for an N256 LIS sampler. The driver
adopts that boundary after checking the checkpoint fingerprints, source
diffs, canonical prior and saved-array shapes. The advisory is not an R2 exit
audit. No external result can promote a posterior here.

**Q-GOAL:** this measures whether a small local likelihood-curvature range
could be useful for the same conditional z=0 density/velocity inference. It
does not produce that field or close R2. MW/M31 remain role-ambiguous and M33
unresolved; their observables must constrain those same roles on the same NEW
evolved LG field at `<=0.3 cMpc/h`. Native truth identities remain
calibration/evaluation labels only, never candidate seeds or selectors.

**Q-LEAN:** one saved anchor, one exact oracle, 17 gradients maximum, no
second backend, no denominator search, no held-out score, no sampler ladder,
and no standalone PM or RAMSES run. A favorable result does not auto-launch
another calculation.

R2 remains **NO-GO**; no held-out prediction, calibrated UQ or z=0 posterior
has been delivered.
