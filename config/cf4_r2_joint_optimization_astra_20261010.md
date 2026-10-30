**Mid-course verdict: CONDITIONAL GO for one compact, jointly preconditioned optimization of unchanged conditional v6, as initialization only. Stop the scalar line/secant ladder. This is not R2 exit approval.**

### Findings

The scientific objective is valuable: a dynamically compatible ensemble connects observed density and velocities to possible histories without promising a unique past. The latent-IC forward route remains the best-supported practical route in this repository; the independent fine-field generators did not establish the required physical distribution. However, nominal 1.5 cMpc/h cells do not establish observational resolution, and the complete calibrated posterior plus LG zoom ensemble is not currently demonstrated. See [the approved replan](/home/kjhan/BACKUP/CF4/CF4_END_TO_END_REPLAN_20260913.md:63).

A conditional field initialization is computationally feasible. Job [418601](/gpfs/kjhan/CF4/z0_density/r2_conditional_full_gradient78_20261010/result.json) evaluated the full gradient in 593 seconds. It confirms:

- Objective: 8,208,466.3231.
- IC gradient RMS/infinity norm: 1.7929/12.5091.
- Tracer infinity norm: 110.3875; population infinity norm: 354.6186.
- No IC movement, optimizer call, posterior sample, or heldout scoring.

These are not convergence results. Choosing the next coordinate by raw infinity norm neglects both curvature and the dimensionality of the IC block. Repeated cross-coordinate rebounds demonstrate that the scalar gates do not measure joint stationarity.

The earlier [Astra B verdict](/home/kjhan/BACKUP/CF4/config/cf4_r2_pop9_reduced_astra_20261007.md) requested **one full gradient because the IC pullback was missing**. That concern is resolved. It did not require indefinite nuisance-coordinate polishing.

### Recommended bounded calculation

Reuse the existing [matched-GL2 objective](/home/kjhan/BACKUP/CF4/scripts/cf4_r2_conditional_map.py:326) and [resolution target](/home/kjhan/BACKUP/CF4/src/cf4_r2_resolution_target.py:49), restarting from gradient 78.

Use `q = q78 + D*z`, `D_IC = 1`, with `grad_z = D*grad_q`. No Jacobian-dependent objective term is needed for this constant optimization reparameterization.

A practical **provisional**, fixed scale vector is:

```text
tracer D[0:9]:
[.002, .005, .005, .005, .010, .005, .030, .010, .010]

population D[0:15]:
[.0005, .003, .001, .001, .003, .002, .003, .003, .003,
 .010, .010, .020, .003, .003, .020]
```

These rounded scales reflect the recorded stiffness differences: population 0/2/3 are much stiffer than several log-Cholesky and LOS coordinates. Unmeasured coordinates use conservative group fallbacks. This is an optimizer starting metric, **not** measured posterior curvature or covariance. Historical secants are state-dependent; do not call this optimal preconditioning or perform new GPU probes to perfect it. Blanket nuisance `D=1e-3` risks unnecessarily immobilizing broader directions.

Essential conditions:

1. One replay must reproduce gradient 78’s objective components and canonical gradient; count it within **18 total exact evaluations**, including rejected trials. Use the existing hard wrapper cap.
2. Keep exact GL2 value/gradient, refreshed support, unchanged data/priors, and fail-closed support/memory limits. No clipping, likelihood floors, physical parameter bounds, or selection inventions.
3. Standard L-BFGS-B with `maxcor=10` is proportionate. Prefer `maxls=10` within the same 18-evaluation ceiling; a five-trial startup line search can itself be limiting. Do not interpret line-search exhaustion as scientific impossibility.
4. One allocation: **five-hour application/six-hour Slurm cap**, 48 GiB host, typed GPU with measured headroom. Preserve accepted states separately from best evaluated trials.
5. Report original-coordinate block gradients, objective components, IC displacement and same-state density/velocity change. `ftol=0` and a fixed objective offset are acceptable; transformed-gradient success alone is not canonical convergence.

A useful outcome is a reproducible joint descent and a clearly labelled conditional field initialization, with an honest cost/stopping assessment. Prior-dominated descent is not automatically better reconstruction. Failure or budget exhaustion ends this experiment and prompts one substantive route decision—not another scalar ladder or automatic identical restart.

### What actually blocks posterior claims

The [denominator audit](/home/kjhan/BACKUP/CF4/CF4_R2_SELECTED_MARK_DENOMINATOR_AUDIT_20261005.md:5) supports the narrower 1,414-linked-mark estimand, not full CF4 inclusion calibration.

For this conditional subset, a complete CF4 group-incidence model and multi-mark within-Tempel covariance are **not prerequisites to this optimization**: the active cohort has one mark per distinct Tempel parent. However, conditioning on presence does not prove missingness is ignorable. Mark availability, linked-point conditioning, count selection, tracer/RSD-FoG response, shared calibration and numerical approximation remain explicit model assumptions requiring targeted sensitivity and predictive checks.

External calibration of every nuisance is not inherently necessary: some can be inferred jointly. But selected-only catalogues cannot identify an unrestricted missing selection mechanism. Full-survey claims require an explicit defensible observation law, additional relevant information or declared assumptions—not more optimizer iterations. Posterior uncertainty additionally requires demonstrated sampling/mixing, numerical/model sensitivity and a frozen predictive protocol before heldout use.

### Q-GOAL / Q-LEAN and historical claims

**Q-GOAL: conditional pass.** The joint experiment advances the same-field z=0 route. It supplies neither posterior uncertainty nor production ICs. MW/M31 remain role-ambiguous and M33 unresolved. Their observables must eventually constrain roles identified from the **same NEW evolved field at ≤0.3 cMpc/h**; native truth identities are calibration/evaluation only, never candidate seeds or selectors.

**Q-LEAN: replace the execution pattern.** The 48,643-line diagnostic script, 7,028-line test file and 895-KB design log are disproportionate to the present numerical question. Preserve their history, but stop extending their job-specific branches. Keep reusable target, replay, support, checkpoint and budget checks. Defer wholesale refactoring and further source/mock/precision ladders.

The [Grok handover](/home/kjhan/BACKUP/CF4/HANDOVER_GROK_20260920.md:22) materially overstates posterior, calibration and holdout completion. Historical coarse/proxy products and IC preflights do not certify the active v6 posterior or an LG-conditioned zoom ensemble. The current authoritative record contradicts those promotion claims.

**R2 remains open and NO-GO for a calibrated posterior. No exit email or R3 is approved.** This is design advice based on the inspected target and original runner, not certification of the driver’s subsequent implementation diff.
