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

The first attempt used one typed-H100 Slurm allocation based on the resource
snapshot at that time. The v2 retry preflight showed 5 of 8 H200 devices free
on syn104. Before v3 submission, a fresh `scontrol show node syn104` showed
only 1 of 8 H200 devices allocated, so 7 were unallocated; H100/A100 counts
were not rechecked for that submission. The retry therefore uses
`--partition=h200 --gres=gpu:H200:1`, and the executable checks that it
actually received H200 plus sufficient device-memory headroom.
The directly measured exact-GL2 N256 peak is 30.91 GiB on a 69.81-GiB device
from the earlier H100 profile; the runtime guard checks the selected H200's
actual reported device limit before evaluating the target. The preceding joint
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

## First allocation failure and bounded recovery

The first typed-H100 allocation, job `413590`, stopped before any objective or
heldout measurement value was evaluated. The state-local support builder
required a padded source-cell width above its inherited 32,768-cell workspace
ceiling. Slurm did not OOM: batch MaxRSS was 5.61 GiB and the process-recorded
host peak was 10.44 GiB of the requested 48 GiB. No likelihood evaluation was
completed and no science result was produced.

Recovery raises only this fail-closed workspace ceiling to 131,072 cells for
this diagnostic. The candidate list, exact geometric filter, positive
source/bin retention, eight-sigma rule, 40-million raw-component ceiling,
likelihood, quadrature and priors are unchanged; excess support still raises
an error and is never clipped. The chosen cap permits at most 1,048,576
quadrature subnodes in one order-2 source row before positive-component
filtering. Generic callers retain the prior 32,768-cell default. A unit test
checks that the old ceiling rejects rather than truncates, and that the larger
ceiling preserves the full rounded width. This is a mechanical recovery, not
a relaxation of a science criterion. Retry output is isolated at
`/gpfs/kjhan/CF4/z0_density/r2_conditional_map_20261005_v3/`.

Execution recovery history: job `413597` exited in 12 seconds at the source
commit preflight because the submitted expected-SHA string was mistyped; it
never initialized the output directory or evaluated data. Job `413598` used
the correct commit and H200, passed the frozen-cohort checks, and completed
one exact target value/gradient evaluation at the initializer. It then failed
because the persistence callback accepted two arguments while the objective
passed three. The just-computed objective, gradient and decomposition were
not persisted, the optimizer did not move from the initializer, and no field
result is recoverable from that process. Its machine report records the
callback exception but not the in-memory evaluation. Heldout measurement
values remained unloaded. The callback signature is now tested, and the outer
failure handler also records completed evaluation rows and best objective if
a later error occurs. H200 reports a 104.85-GiB device limit, passing the
existing runtime memory-headroom guard. Retry v3 keeps all scientific inputs
and bounds fixed; only persistence/error reporting is repaired. The v3 JSON
initially carried stale H100/A100 free-GPU counts and the v2 H200 count; this
administrative metadata was corrected to the recorded v3 pre-submit snapshot
(7 H200 free, other-mode counts unavailable) after completion. No score or
field value was changed.

## Conditional MAP attempt result — job 413612

The H200 job completed normally in 1:23:15 with 6 exact value/gradient
evaluations in 4 optimizer iterations. Evaluation 6 is the best objective,
`8214941.8818`, down `29860.8550` (`0.362%`) from the initializer. The prior
sentence's originally transcribed `8219182.53` was evaluation 5, not the best;
the result JSON is authoritative. The best decomposition was
IC prior NLL `8072142.03`, nuisance prior NLL `1.21`, count log likelihood
`-144588.28`, and conditional FP log likelihood `+1789.64`. The RMS density
change from the initializer was `1.411`; componentwise velocity changes were
`[51.70, 40.46, 54.28] km/s`. The best gradient infinity norm was
`13524.69`, versus the required `1e-4`, and the solver stopped at the
four-iteration cap. The code therefore reports
`CONDITIONAL_MAP_ATTEMPT_INCOMPLETE`, not a converged MAP.

This demonstrates that a finite optimizer path can evaluate the declared
training objective and lower it modestly; it does **not** establish a stable
or calibrated z=0 density/velocity posterior. The incomplete best field is
retained only as a diagnostic artifact, not promoted as an IC, seed, posterior
sample or production map. Heldout values remain unloaded. Selection and
mark-availability calibration, cell-exposure quadrature stability, MW/M31
role ambiguity and M33 non-identification remain open. R2 stays NO-GO.

Status: job 413612 completed; interpretation and limitations are recorded
above. No automatic follow-on fit was launched.
