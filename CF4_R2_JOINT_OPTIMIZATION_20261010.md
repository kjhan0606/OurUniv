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
