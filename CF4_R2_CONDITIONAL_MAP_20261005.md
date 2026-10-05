# R2 conditional v6 field diagnostic — 2026-10-05

## Decision and question

The user approved proceeding after the selected-mark denominator audit. The
scientific question is deliberately narrow: can the current conditional v6
training target produce a coherent change in an N256 present-day field from a
fixed LCDM initial state? This is not a repair or validation of the preceding
40-transition chain's nonstationarity. A maximum-a-posteriori point is not a
posterior draw and may suppress weakly constrained modes; the result cannot
establish posterior uncertainty, convergence, a calibrated CF4 posterior, or
an IC/seed for production.

Fable's requested read-only plan audit could not run because the Claude CLI
reported its usage limit. The prescribed Astra backup review returned
**CONDITIONAL GO**. The driver independently checked the active code path and
exact-GL2 memory/result record, and adopts its corrections below. Astra's
review did not independently recount catalogue rows or inspect heldout values;
the batch preflight checks the frozen cohort sizes and source ledger without
loading heldout measurements.

## Frozen conditional target

- N256 latent IC, 384 cMpc/h box, 1.5 cMpc/h field-grid spacing.
- Exactly 47,121 v6 training 2M++ count points on 37,951 training keys,
  including the empty exposed-cell contribution in the Poisson expectation.
- Exactly 1,414 secure linked FP marks, conditional on each observed 2M++
  point and on the mark being present. The active numerator/denominator uses
  that row's linked-point radius. Count occurrence is scored once; do not add
  another `f_n` or a group-incidence term.
- Existing LCDM Gaussian IC prior and 24 standard-normal nuisance priors,
  each applied once. All nuisance coordinates start at the current coordinate
  origin; this is not a continuation of an older fitted nuisance state.
- Rebuild the existing shifted-source support at every distinct optimizer
  candidate. Its current 8-sigma truncation remains an approximation.
- Use matched fine GL2 value-and-gradient evaluations for every optimizer
  candidate, including line-search trials. No lower-order force surrogate.

The heldout count/FP measurement values are not loaded or scored. The code
reads only heldout geometry required to preserve the frozen training exposure
mask. The conditional-on-mark-presence assumption does **not** establish that
mark availability is field-independent; full survey/group selection, redshift
success and tracer-bias calibration remain unresolved. The N128 count grid and
angular selection are retained and replicated onto N256 source nodes, so the
1.5 cMpc/h field sampling is not evidence that observations resolve 1.5
cMpc/h structure.

## Execution and resource bounds

One typed-H100 Slurm allocation, selected after one resource check found H200
occupied and no idle permitted H200/H100/A100 GPU. The directly measured
exact-GL2 N256 peak is 30.91 GiB on a 69.81-GiB device. The preceding joint
pilot host peak was 15.26 GiB; three L-BFGS correction pairs add approximately
0.75 GiB for the 16,777,240-dimensional double vector, so the 48-GiB host
request retains more than 20% headroom. The job may queue normally.

Hard caps: eight exact target value-gradient evaluations total (including the
initial point and rejected line-search trials), four optimizer iterations,
two line-search attempts per iteration, three L-BFGS correction pairs, 100
minutes application time, and 120 minutes Slurm time. The best finite state
is checkpointed as objective evaluations proceed. Exhausting any cap is an
incomplete MAP attempt unless the strict optimizer and gradient conditions
both pass. There is no automatic retry or follow-on fit.

Each objective evaluation necessarily evolves its candidate IC through the
PMWD forward model and its adjoint. This bundle launches no separate RAMSES or
standalone gravity simulation. Physical diagonal velocity variance is saved
from the final same-state forward readout only; it is not an input to this
likelihood, posterior velocity uncertainty, or the phenomenological tracer
LOS width.

## Goal alignment and limits

Q-GOAL: this is one actual-data conditional z=0 field diagnostic on the
approved latent-IC route. It does not reach the eventual LG requirement.
MW/M31 remain role-ambiguous and M33 unresolved; their observables must later
constrain those same roles on the same NEW evolved LG field at `<=0.3`
cMpc/h, with native truth identities reserved for calibration/evaluation.

Q-LEAN: one bounded exact-target optimization is proportionate after the
source-definition audits. No new data, source census, mock ensemble, heldout
score, sampler tuning, or extra simulation is included. Report count, FP, IC
prior and nuisance-prior changes separately: a lower total objective driven
by prior shrinkage is not automatically improved reconstruction.

Status at submission: pending. Final interpretation and artifacts are added
below after the single allocation terminates.
