# R2 joint optimization recovery — 2026-10-10

R1 physical connection -> **R2 present field (incomplete)** -> R3 LG joint
constraints -> R4 precision evolution -> R5 zoom ICs. This bundle keeps the
full R2 objective: CF4+galaxy-conditioned z=0 density/velocity samples at
1.5 cMpc/h in 384 cMpc/h, held-out prediction, uncertainty and calibration.
A conditional MAP warm start alone cannot satisfy that objective.

## Evidence and decision

Slurm 418601 completed in 10m04s, exit 0, on H200. The authoritative result
is `/gpfs/kjhan/CF4/z0_density/r2_conditional_full_gradient78_20261010/result.json`.
Objective 8208466.323127546; IC gradient RMS 1.792859 and infinity norm
12.509119; tracer RMS 69.570895 and infinity norm 110.387543; population RMS
168.900254 and infinity norm 354.618602. ICs were fixed and heldout outcomes
were not read. These are not a converged MAP, posterior or R2 completion.

The scalar line/secant sequence has coupled-coordinate rebound. A small
derivative at an earlier state does not remain small after another coordinate
moves. Comparing the IC maximum alone with the nuisance maximum is not a
block-priority criterion: there are 16777216 IC coordinates and only 24
nuisances. The IC prior contributes millions to the objective because of its
dimension; a small fractional change against that baseline alone is likewise
not a scientific progress metric. The previous Astra B review requested one
missing full gradient, not indefinite scalar optimization.

Close that scalar orchestration series. Preserve its source/results. Do not
append another experiment-specific branch to the 48643-line diagnostic script.
Reuse the existing matched GL2 objective and ordinary L-BFGS-B in a single
allocation. The scientific destination and likelihood are unchanged.

## Concrete computation

Start from saved gradient78 `q`. Re-evaluate once and compare the existing
five objective terms and full gradient; initialization is included in the
18-evaluation ceiling. Use fixed affine coordinates `q=q_start+D*z`, with
unit IC scale and positive nuisance scales recorded in the result. The
gradient is `D*g_q`. Evaluate the same prior/likelihood in original coordinates;
there is no new statistical term or physical bound. A fixed objective offset
and `ftol=0` prevent the large additive prior baseline from producing a
relative-function stopping claim.

Run L-BFGS-B with at most 17 iterations, history 10, line-search allowance 10,
and **18 total exact value/gradient calls**, including rejected trials and
initialization. The wrapper enforces this limit independently of SciPy's
`maxfun`. Five-hour application deadline; six-hour Slurm limit leaves room
for an in-flight evaluation and final field readout. Refresh support for
every changed state with the same existing approximate 8-sigma support rule.

Save improving finite parameters and separately save optimizer-accepted
parameters. Report whether the best readout came from an accepted iteration
or a trial. Save original-coordinate block gradients, each likelihood/prior
change, IC displacement, and same-state density/velocity/physical dispersion.
Never equate physical dispersion with posterior uncertainty. A transformed
gradient tolerance or finite-budget stop is not canonical convergence.

One Slurm GPU, 2 CPU threads, 48 GiB host memory. Prior measured host peak
16.64 GiB plus about 2.5 GiB for ten L-BFGS curvature pairs leaves more than
20% headroom; actual optimizer copies may add memory within this envelope.
Require device memory above 1.2 times the measured 30.91 GiB peak. Check
H200 first and request `gpu:H200:1` when available; otherwise compatible
H100/A100. Expected artifacts are approximately 1 GiB, no RAMSES dump.
Tests execute in the same allocation before the fit. No login-node numerics.

## What follows

Inspect accepted progress, separate data-score and prior changes, field
movement and cost. A useful optimizer becomes initialization for a specified
posterior sampler; it never supplies uncertainty itself. If the bounded run
stalls, use its accepted/rejected direction evidence to address conditioning
or target cost; do not restart coordinate-by-coordinate jobs or copy a tiny
iteration cap indefinitely.

Observation-model work remains necessary: mark-presence ignorability,
association/shared-group dependence, tracer response and quadrature error
need explicit assumptions and calibrated sensitivity/prediction. Preserving
an unresolved label forever is not a solution, but inventing an inclusion
factor is not justified. Sampling and independent mock/heldout validation
remain required for the full R2 goal. Heldout outcomes stay closed in this
optimization bundle; its predictive protocol must be fixed before opening.

Q-GOAL: unfreeze the actual present-field inference within the ratified
latent-IC joint route. MW/M31 remain role-ambiguous and M33 unresolved at R2;
R3 must identify components from the same NEW evolved field and connect
their observables at LG <=0.3 cMpc/h. Truth identities are evaluation-only.
Q-LEAN: one compact adapter and the existing runner replace many scalar jobs;
no new observation terms, external catalogue search or validation framework.
R2 remains open; no exit email or R3 entry follows from this bundle.

## Mid-course review and driver disposition

Astra returned CONDITIONAL GO for this bounded initialization, explicitly
withholding R2 exit approval. The unchanged response is in
`config/cf4_r2_joint_optimization_astra_20261010.md`.

Driver checked gradient78, the previous Astra B response, the selected-mark
denominator audit and the active target. Adopt the joint experiment and close
the scalar ladder. Adopt the suggested fixed vector rather than blanket
nuisance scaling, and increase maxls to 10 within the SAME 18-call budget:

```
tracer: [.002,.005,.005,.005,.010,.005,.030,.010,.010]
population: [.0005,.003,.001,.001,.003,.002,.003,.003,.003,
             .010,.010,.020,.003,.003,.020]
```

These are provisional optimizer scales reflecting historical stiffness plus
group fallbacks, not measured full curvature or posterior covariance. No
additional GPU scale probe is needed before this bounded experiment.

Correct the overly broad blocker wording: the active 1414-row cohort has one
mark per distinct Tempel parent. Full CF4 incidence and multi-mark covariance
are not prerequisites to optimizing THIS conditional subset. Availability
ignorability, shared calibration and tracer response still require explicit
assumptions and validation. The full R2 objective is not reduced to this
subset diagnostic. No claim that all CF4 selection has been calibrated follows.

The driver checks the implementation; the review evaluated design, not the
subsequently edited diff. Tests cover transformed coupled gradients/optimum,
budget/callback behavior and the existing observation wiring, in the same
Slurm allocation before optimization. No separate exit audit was requested.

## Execution

Source `4c7ef5133fda4fdf6aa542632c2f9da931694399` committed and pushed.
Job **418606** started 2026-10-10 23:59:02 KST on syn104 through the
`h200` Slurm partition, `gpu:H200:1`, 2 CPUs, 48 GiB, six-hour limit.
Immediately before submission H200 had 8 configured/7 allocated GPUs;
H100/A100 were not needed for mode selection. No manual node execution.
Output: `/gpfs/kjhan/CF4/z0_density/r2_joint_warm_start_20261010/`.
Logs: `/gpfs/kjhan/CF4/logs/cf4_R2_joint_418606.{out,err}`.
Submission/running state is not a test pass or an optimization outcome.

At 00:00 KST Oct11, Slurm reports RUNNING (elapsed1m15s). All 15 focused
tests passed (2 affine, 5 budget, 3 linked-radius, 5 exposure). The main
application recorded STARTED with the frozen source and 18-call cap.
No completed replay/accepted joint update was yet recorded at that check.
# Terminal result: job 418606

Slurm reports FAILED after 01:12:42, exit 1:0. Eight finite evaluations
were saved; evaluations 3, 5 and 6 were accepted. Evaluations 7 and 8
were improving objective trials but were not accepted. Checkpoint replay
passed. The subsequent candidate required 191040 padded source cells,
exceeding the configured 131072 workspace ceiling; no support was truncated.
This is a numerical workspace failure, not evidence of posterior convergence.

The best evaluated objective was 7493930.197595556 versus initial
8208466.323127545. Its IC prior NLL fell to 7223516.112365052, but count
log likelihood worsened to -276345.4036981737 from -142299.81701650965.
Do not equate objective decrease with improved observed structure or choose
this unaccepted trial as a posterior draw. The last accepted state is saved
separately in accepted_parameters.npz.

Batch MaxRSS was 82750756K (about 78.92 GiB), greater than the 48 GiB
request; the previous application peak of 36.22 GiB was stale at failure.
A blind ceiling increase or identical restart is not justified. Inspect
candidate-list materialization and shape-dependent packing before the next
allocation; preserve all source contributions and the unchanged likelihood.
Any rerun must correct the host memory estimate and request at least 20%
margin. R2 posterior, uncertainty, held-out prediction and calibration remain
undelivered.

Recovery implementation: query/filter candidates one observation at a time;
optional support_chunk_cells evaluates all locally selected candidates in
bounded blocks, without truncation. Default legacy ceiling behavior remains.
The option is forwarded through ResolutionObservationTarget and the MAP
entry point, but is not enabled in the production launcher. The total retained
component ceiling remains: blocking the weight workspace does not bound all
packed components or compiled adjoint memory.

Grammar CPU Slurm job 1171037 completed, exit 0, 20 seconds, MaxRSS 588420K.
All six raw-volume tests passed, including real-weight component identity
between split/unsplit packing, velocity-dependent support refresh, and a
65-candidate/64-cell-workspace case preserving all 325 bin components.
GPU test 418678 was canceled while pending due to QOSMaxGRESPerUser.
These are packing tests, not a full N256 target/gradient or memory validation.

N256 recovery check submitted as syntax job 418697, source commit 60d12b4,
H100 typed GRES gpu:H100:1, 2 CPUs, 96 GiB, 75-minute Slurm limit and
one exact evaluation. H200 was fully allocated; syn09 had two unallocated
H100 GPUs at submission. Queue reports QOSMaxGRESPerUser, so availability
does not imply immediate eligibility. Do not duplicate or manually launch it.
Output: /gpfs/kjhan/CF4/z0_density/r2_chunked_replay_20261011.
The check uses 16384-cell support blocks and replays gradient78, retaining
the same conditional target and checking all five terms and full gradient.
It does not assess the failed broad candidate's peak or prove sampler mixing.
No posterior, held-out score, uncertainty or MW/M31/M33 identification is
delivered by this check. Same NEW field and ambiguous/unresolved LG roles
remain requirements of the subsequent science route.

Job 418697 completed successfully (exit 0:0) in 11m37s on H100 syn09.
The unchanged-target checkpoint replay passed, including the full canonical
gradient and five objective terms. Objective: 8208466.323127571; exactly one
target evaluation, no IC update. Application host peak was 11.096 GiB;
Slurm batch MaxRSS was 9627408 KiB. This verifies chunked target equivalence
at gradient78 only, not the failed broad candidate's memory or posterior
mixing. Next recovery requirement is broad-support state validation before
any bounded optimization/sampler continuation; R2 remains incomplete.

Recovery provenance clarification: neither saved parameter file contains the
candidate that triggered the 191040-cell exception. `accepted_parameters.npz`
is evaluation 6; `best_parameters.npz` is unaccepted evaluation 8, whose
support width was 77568 and whose objective is recorded. The persistence
callback runs only after a finite completed evaluation. Thus replaying either
file cannot be advertised as reproducing the failing candidate. Evaluation 8
can serve as a broader recorded-state replay with component-value checks;
the exact failing state requires deterministic trajectory reproduction or
explicit pre-evaluation candidate capture. Do not loosen checkpoint lineage
checks merely to accept a failed run's arbitrary saved file.

Broad recorded-trial replay 418811 completed on H200 syn104 in 12m09s,
exit 0:0. All five objective terms matched evaluation 8; objective
7493930.197595556. Support retained 36505268 components, largest candidate
77550 cells, using 16384-cell blocks. Application peak 10.675 GiB;
Slurm batch MaxRSS 10231300 KiB. No optimizer transition occurred.
This is not a replay of the unsaved failing candidate and not a posterior.
The output inherited incorrect resource/nuisance labels: host request was
96 GiB (not 48), H200 was checked with 2 unallocated GPUs, and nuisances
were saved trial-8 coordinates (not zero). Preserve the raw result; correct
future metadata from Slurm rather than hardcoded claims. No full-gradient
reference exists for this trial, although a finite full gradient was computed.

Driver next-route decision after recovery: do not restart the identical MAP
experiment. The numerical recovery enables the unchanged conditional target,
but optimizing the 16M-dimensional Gaussian prior toward its mode is not a
posterior sampling strategy. Return to the reusable prior-split HMC kernel,
with exact GL2 acceptance and fresh chunked support; initially use matched
GL2 forces to avoid confounding force approximation with integrator failure.
Use gradient78 as a declared diagnostic initializer, not unaccepted trial 8
as a selected posterior sample. Before a larger chain: wire the exact current
v6 cohort/conditioning, use a fixed metric, retain rejected states, record
canonical prior/likelihood terms, test reversibility and Gaussian reference
behavior in a small Slurm CPU test, then run a bounded transition pilot.
Freeze the heldout scoring protocol before opening heldout values. Successful
transitions alone do not establish mixing, uncertainty, observational
resolution or selection calibration. These remain R2 delivery requirements.
Q-GOAL: same NEW evolved field; MW/M31 ambiguous, M33 unresolved, no truth-ID
candidate selection; their LG observables and <=0.3 cMpc/h refinement remain
later requirements. Q-LEAN: reuse kernels; no new scalar probes or separate
RAMSES simulations. Obtain warranted Astra advice before a large chain.

HMC mechanics regression rerun on grammar Slurm job 1171217, debug
partition, 2 CPUs, 8 GiB, 10-minute cap: all 10 existing tests passed
(2.761 seconds test time). Coverage includes prior-flow conservation,
full-map reversibility/volume preservation, Gaussian mean/covariance,
rejection retention and RNG checkpoint replay. These small algorithm tests
do not validate the active CF4 target wiring, N256 integration error or
scientific chain mixing. No numerical test was run on the login node.

Active-v6 HMC wiring test 1171218 passed on grammar debug (one small
mocked rejection-retention test). Next bounded pilot: replay gradient78,
then two fixed-step .02, one-step matched-GL2 HMC proposals; three full
target calls total, chunk size 16384, fixed existing inverse-Laplacian IC
metric and provisional nuisance scales squared as inverse mass. These are
not posterior curvature measurements. Preserve the retained current state
after every proposal, including rejection; do not select the best evaluated
candidate. One H200 GPU, 2 CPUs, 96 GiB, 75-minute Slurm / 60-minute app cap.
No heldout values, UQ claim, MAP continuation or standalone simulation.

### Validation continuation boundary

Driver inspection of the active adapter confirms that it loads only v6
training counts plus heldout geometry, not heldout count values. Preserve
that separation during metric/step tuning. Before opening heldout values,
freeze the predictive scoring protocol and the sampled-target fingerprint.
Evaluate predictions over retained post-warmup states (including rejection
repeats), not the best objective candidate. Report count and conditional FP
scores separately; conditional linked-mark performance cannot establish full
CF4 inclusion/survival calibration. The old N128 survival holdout screen uses
a different cohort and unconditional field fixture and is not a v6 posterior
validation result. Reuse its numerical utilities only after checking the
active observation contract. Mean/variance accumulation alone does not prove
mixing or supply Monte Carlo uncertainty. MW/M31 roles remain ambiguous and
M33 unresolved in this R2 field; no truth labels may choose a generated state.
