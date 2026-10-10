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
