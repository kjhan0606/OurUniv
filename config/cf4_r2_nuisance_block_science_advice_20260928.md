# R2 scientific decision after the one bounded MAP extension

Read-only scientific/implementation-plan advice. Do not edit files, build,
submit jobs, run simulations, send mail, or invoke other agents. Return a
usable concise verdict with evidence and conditions, not a diagnostic ladder.

External-review trigger: the completed bounded fit still is nonstationary,
with a 4.09-prior-SD shared FP zero and structured radial count residuals;
decide whether another substantive computation is justified and what it must
do differently. This does not reinstate routine per-step external audits.

Read CF4_R2_FIRST_FIELD_ADVICE_20260928.md and the LAST sections of
CF4_R2_FITTED_TANGENT_20260928.md. The active goal/order is in
CF4_END_TO_END_REPLAN_20260913.md. If inspecting implementation, use
scripts/cf4_r2_v6_partial_map.py and src/cf4_r2_linked_singleton_target.py.
State which evidence/files you actually inspected. Do not claim a code audit
if you only review this summary. No need to read the entire long master log.

## Goal, scope and established evidence

CF4 plus galaxy data -> joint latent-IC/current-state inference -> first deliver
actual z=0 density/mean velocity/posterior uncertainty, surroundings1–2cMpc/h;
later same-state LG<=.3, precision evolution and phase-consistent zoom IC.
R1 done; R2 unfinished; R3–R5 not promoted. Native truth IDs may label tests,
never seed/select generated MW/M31/M33 candidates. At this coarse NEW state
MW/M31 roles remain ambiguous and M33 unresolved; their observables must
constrain that same NEW state later, not independently supplied peaks.

Current N128/384 (3cMpc/h) development target uses47121 training counts and
429 strict ungrouped linked FP marks, NOT all CF4.985 grouped links excluded.
Gaussian LCDM white-IC prior unchanged;9 white tracer parameters and one FP
zero. Optimizer coordinates are IC white,100*white_tracer,white_zero; different
blocks' max gradients are NOT invariant physical-uncertainty comparisons.
No heldout observations have been scored. Identity-mass ESS2–5 HMC is closed.
Do not prescribe arbitrary power restoration, truth seeds, new TNG, N256
or large simulations as an automatic next move.

408119 completed1h51m08s,64 accepted joint updates,66 full gradients,70
score-only trials, stop=iteration limit. F159462.614936->151825.923500;
counts−156978.833864->−146150.539323; FP−43.947229->−25.563497. Last gain53.69.
One verified scalar rate MAP initialization preceded the joint fit; predicted
and actual gain137.92777589 agree~1.5e-12; not marginalization/profiling.
Final largest gradient42.6823 is the true-K[-25,-23.6667] bias, optimizer
coordinate2097154; IC max3.65466. Final nuisance gradients in optimizer units:
[9.30557,-.70674,-42.68229,-18.45683,-1.21986,-.000512,3.42135,-1.21624,
9.71923,-1.81574]. Largest bias itself is1.12825, not an established extreme
bias failure. A full gradient costs~75s on H200. Source/target unchanged.

Shared white FP zero1.66710->4.08892; actual zero.0066684->.0163557dex under
the SAME SD.004 prior. Thus FP score recovery cannot be attributed solely to
the changed field. This is a tension, not proof of bad implementation or a
reason to widen the prior. No frozen-field zero decomposition has been run.

408150 saved-field integration check finished5m51s.4x32/8x32 log-score
delta.001199; exposureL1=1.469e-6,max occupied logdelta.000724. Scalar velocity
derivative discrepancy1.27e-8. Full composed terminal adjoint also passed
normalized discrepancy1.62e-6. Thus simple derivative/integral failure is not
currently established. Mean training count47586 vs47121, but residual shape
persists: predicted/observed4979.7/4417 at48–60cMpc/h,1366.6/1801 at168–180.
This is training comparison, not heldout/PPC or calibration.

408121 finite-gradient curvature feasibility is still RUNNING. First scaled
rate-axis curvature19.0345070956 matches analytic19.0345070942. Additional
epsilon and two IC directions are pending. Do not invent their outcomes or
claim an SPD Hessian/Laplace from three directions. Driver will assess them.

## Proposed next substantive action for your assessment, not authorization

Do NOT append another identical64-step fit. Consider one bounded conditional
nuisance optimization at the saved NEW field: optimize all10 calibration/
tracer coordinates against the SAME count+FP factors+same priors; freeze IC
only for this optimizer block, not as a posterior approximation. Reuse the
saved/evaluated rho/velocity and refresh FP source support for changing LOS
width; differentiate only nuisance arguments to avoid repeated PM evolution
and unnecessary field VJPs. One full-field objective/gradient at the resulting
state measures whether the residual simply moves into IC. No nuisance mean
becomes a new prior or a permanently fixed parameter in subsequent inference.

Up to20 accepted block iterations/20min application,30min Slurm,H2002CPU12GiB
estimate<=10GiB host. First establish cost of the nuisance-only derivative.
Reuse the existing bounded optimizer with a correctly handled empty IC block,
one small coupled-quadratic regression and actual full-target reproduction.
No new framework, mock stream or parameter sweep. Keep all Gaussian priors,
selection windows, likelihood weights, true parameter domains and field phase.
Then driver decides the next inference method rather than attaching automatic
unbounded alternating cycles. This delivers a conditional optimization result,
NOT a calibrated posterior or a solution to selection/covariance.

## Please answer

1. Q-GOAL: Is this proposed block a useful, proportionate step toward the
   actual z=0 posterior and later same-state MW/M31/M33, or are we prolonging
   the wrong approach? Explicitly retain unresolved M33/role identification.
2. Q-LEAN: Is this the smallest useful action given the observed gradient,
   FP-zero shift, radial residuals and75s full-gradient cost? Prefer a sharper
   alternative if the proposal adds mainly diagnostics without resolving a
   bottleneck. No indefinite same-target optimization or audit ladders.
3. Does the evidence establish a block-conditioning issue, or only motivate
   testing it? How should the shared-zero tension be interpreted without
   pretending this is calibrated CF4 inference? Distinguish identifiability,
   model mismatch, nonconvergence, and implementation error.
4. Give ONE next decision after that action (or your replacement), with a
   concrete stopping/replanning condition. Do not silently convert MAP into
   posterior, a few HVPs into covariance, or a plug-in heldout score into PPC.
5. Verdict PROCEED / CONDITIONAL PROCEED / REPLAN, with any essential changes.

Current selection/bias/FoG calibration, association and shared source-fit
covariance are unresolved. Those are scientific limits, not excuses to claim
that this partial target's numerical optimization is full R2 completion.
