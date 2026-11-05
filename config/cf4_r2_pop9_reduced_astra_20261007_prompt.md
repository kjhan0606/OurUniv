# Mid-course Astra audit — population-9 gate, 2026-10-07

This is not an R2 exit audit. Do not approve closing R2. Do not authorize the exit email or R3. Read only. Do not modify files, submit jobs, or add a likelihood or prior term.

## Q-GOAL

Does one next measurement on the same conditional v6 training target stay aligned with the CF4 Local Group reconstruction? The first science delivery remains the actual z=0 posterior. The current product is an unfinished conditional MAP attempt. It is not a posterior, not a held-out result, and not a production initial condition.

MW and M31 roles are ambiguous. M33 is unresolved. Their observables must constrain those roles on the same new evolved Local Group field at <=0.3 cMpc/h. Native truth identities may label calibration or evaluation only. They must not seed or select generated-field candidates. Do not assume known components inside g(F).

## Q-LEAN

The population-9 coordinate has met its predeclared tenfold gate. An 18-evaluation IC warm start is a large calculation: each fixed-IC derivative costs about 15 minutes, so 18 evaluations are several hours on one GPU. Is that proportionate now, or does the evidence support only a smaller measurement, or neither?

## Recorded state

Result: `/gpfs/kjhan/CF4/z0_density/r2_conditional_pop9_secant2_20261007/result.json`. Job 414980, commit `199694891a199bbd63d8b35333a8e92985041a5d`, `h200` / `gpu:H200:1` on syn104, 30:46, exit 0. Status `CONDITIONAL_POP9_LINE_REDUCED`. `longer_warm_start_authorized` is false. Held-out count values and FP marks were not loaded. The IC coordinates were fixed. The optimizer was not called.

The same objective as job 413612 was used: standard-normal prior minus the count and conditional-FP scores. No new term. Population coordinate 9 is the log of the first intrinsic-FP covariance diagonal. Tracer 0 is held at 0.23014459265495463.

Verification reproduced job 414961: population coordinate 9 = -0.5340042606603805, objective 8214295.992951905, derivative +179.29929799083575. The secant point is population coordinate 9 = -0.592468447322586, objective 8214289.311950613, derivative +46.11400943672288. Tracer-0 derivative -410.4774282041946. Conditional-FP log likelihood 1999.5313480197265. The IC prior, tracer prior, and count term did not change across that step.

The tenfold gate uses job 414930 evaluation 0: absolute derivative 699.9968692786738, so the gate is 69.99968692786738, and the conditional-FP log likelihood must not fall below 1789.6374659353557. Both conditions pass. The sign is still positive. A linear extrapolation from the last two positive derivatives suggests a further move near 0.02 and about half a nat. That extrapolation is not a measurement. The predeclared search does not take a third secant.

Tracer 0 already met its own tenfold gate in job 414921: 13524.69411623714 fell to -410.477428204. Its remaining objective was estimated near half a nat and was not pursued. Population coordinate 9 was then 699.9968692786738. It is now +46.11400943672288.

The IC-block gradient was last summarized at the original saved best, before these nuisance moves: rms 2.005 and infinity norm 14.95, in `/gpfs/kjhan/CF4/z0_density/r2_conditional_optimizer_diagnostic_20261006/result.json`. It has not been recomputed at the present nuisance point. The other 22 nuisance gradients have not been remeasured here. Design record: `CF4_R2_CONDITIONAL_OPTIMIZER_DESIGN_20261006.md`. Active plan paragraph: `CF4_MASTER_PLAN.md`.

## Question

Which one next action does the evidence support?

A. Submit one IC-only warm start now, at most 18 exact evaluations, with all 24 nuisance coordinates held, the same objective, no held-out scores, and no new term.

B. First remeasure one fixed-IC derivative at this nuisance point, including the IC block and all 24 nuisance components, and do not start an IC update until that gradient exists.

C. Do not start either measurement. The recorded gates are met, the state is not stationary, and neither measurement is proportionate.

State the verdict as A, B, or C. Cite the result paths you checked. Name any recorded number that does not match those files. Do not close R2.
