# R2 conditional optimizer design — 2026-10-06

## Question

Job 413612 ran the frozen conditional v6 target for four L-BFGS-B iterations and six exact evaluations, then stopped because `maxiter=4`. The objective fell by 29,860.855 (0.362%). The best gradient infinity norm was 13,524.69, against the declared absolute gate of `1e-4`. That gate did not stop the run. The same cap is not repeated.

This bundle does not fit the target again. It checks whether the saved initializer and best state reproduce the recorded terms, where the best-state gradient lives, and whether that gradient matches one directional finite difference. A longer warm start stays unauthorized until those results exist. R2 remains NO-GO.

## What the saved path already shows

The artifact is `/gpfs/kjhan/CF4/z0_density/r2_conditional_map_20261005_v3/result.json`. Six evaluations took 4,974 seconds. Host peak was 16.64 GiB of the requested 48 GiB.

The objective is about 8.245e6, of which the IC prior is 8.076e6. From the initializer to evaluation 3 the objective fell by 23,763, almost entirely in the count term. From evaluation 3 to evaluation 6 it fell by 6,098: IC prior 3,619, count 2,039, conditional FP 461, and a nuisance-prior cost of about 1.2. A lower total is not by itself a better reconstruction. White-noise mean square moved only from 0.96272 to 0.96227.

Gradient RMS went from 16.96 to 2.96 and then back up to 4.61 while the objective kept falling. The infinity norm fell from 6.47e4 to 6.09e3 and then rose to 1.35e4. Evaluation 2 was one rejected line-search trial. Infinity-norm decrease is not a monotone convergence certificate here. `gtol=1e-4` is about 1e8 times smaller than the best infinity norm, and `ftol=1e-9` is far below the last accepted relative decrease.

The saved artifact has scalar terms and `best_parameters.npz`. It does not store the gradient vector. Count and FP score gradients can be separated with the existing `component_derivative` on `ResolutionObservationTarget`. The objective algebra now lives in `assemble_conditional_objective`; the MAP runner and this diagnostic both call it. No prior or likelihood term is added.

## Diagnostic

Script: `scripts/cf4_r2_conditional_optimizer_diagnostic.py`. It does not call the optimizer, read heldout measurements, or change the GL2 order-2 operator.

1. Replay the initializer and the saved best state. Compare IC prior, tracer prior, population prior, count, and conditional FP with evaluations 1 and 6.
2. Stop before component attribution and the finite difference if any term differs by more than `1e-3` absolute or the total objective differs by more than `1e-8` relative. Do not loosen that gate after seeing the replay.
3. If both states pass, record RMS and infinity norm for the IC block, the 9 tracer coordinates, and the 15 population coordinates, including the coordinate that holds the infinity norm.
4. Split the best-state score gradient into count and conditional FP. Their sum, subtracted from the parameters, must rebuild the joint gradient to relative L2 `1e-4` and maximum absolute error `1e-3` times the joint infinity norm. A component adjoint runs only when its compiled peak has 20% device headroom. Otherwise the block summary remains and the component split is skipped.
5. One finite difference at the best state, along the joint gradient, with `eps=1e-4`. Relative error above `1e-2` means this gradient is not a basis for a longer fit. One direction does not certify every coordinate.

Application budget 70 minutes, Slurm 90 minutes, one GPU, 48 GiB host. A stage starts only when the remaining application budget can hold it. The result directory is `/gpfs/kjhan/CF4/z0_density/r2_conditional_optimizer_diagnostic_20261006/`.

Resource choice at design time: syn104 had 7 of 8 H200 devices allocated and one unallocated. syn08 and syn09 each had all 5 H100 devices allocated. syn101 had all 8 A100 devices allocated and syn102 was down. The selected mode is `h200` with `--gres=gpu:H200:1`. The submit-time free counts are recorded in the result.

## What a passed diagnostic does not authorize

`longer_warm_start_authorized` is false for every status this script can write. A later warm start, if the replay and finite difference pass and the gradient is not confined to one nuisance or one likelihood term, would start from `best_parameters.npz`, keep this objective, and use `maxcor=10`, `maxls=5`, and 18 exact evaluations. That run is a large calculation and needs an external review before submission. It is not part of this bundle.

MW/M31 roles remain ambiguous and M33 unresolved. Their observables still have to constrain those roles on the same new evolved LG field at `<=0.3` cMpc/h. Native truth identities stay calibration and evaluation only. R3 does not start from this diagnostic.

## CPU checks

`tests/test_cf4_r2_conditional_optimizer_diagnostic.py` checks the prior-minus-score algebra, the reproduction gate, block ownership, the finite-difference limit, component-sum agreement, and the status labels. The N256 replay itself is the Slurm job.

## Result — job 414485

The job completed on `h200` / `gpu:H200:1` at syn104 in 59:35, exit 0. Application time was 3,532 seconds. Host peak was 13.0 GiB of the requested 48 GiB. The device limit was 104.85 GiB. Both saved states reproduced the recorded terms with empty failure lists, inside the `1e-3` absolute and `1e-8` relative gates. No heldout values were loaded. The optimizer was not called. `longer_warm_start_authorized` is false. Status: `CONDITIONAL_OPTIMIZER_DIAGNOSTIC_INCOMPLETE_BUDGET`. R2 remains NO-GO.

The best-state joint gradient is not an IC-field spike. Its infinity norm, 13,524.69, is tracer coordinate 0. That coordinate enters the count rate as `log_reference_rate_per_cell = log(sum of reference LF fractions) + 2*tracer[0]`. The count score gradient carries it: tracer infinity norm 13,524.39, while the count gradient on all 15 population coordinates is 0. The IC block at the same state has RMS 2.005 and infinity norm 14.95. The population block has RMS 399.2 and infinity norm 700.00 at population coordinate 9, which is `p[0]` in the FP unpacking, the log of the first intrinsic-covariance Cholesky diagonal. The conditional-FP score gradient owns that coordinate; its IC RMS is 0.00363. Adding the two score gradients and subtracting them from the parameters rebuilds the joint gradient to relative L2 `5.7e-16`.

The finite difference did not run. After the two replays and two component adjoints, 668 seconds remained, short of the 720-second value-stage threshold. That is a budget stop, not a failed derivative check and not evidence that the 13,524 gradient is an IC mode.

## Finite difference — job 414657

The completion ran on `h200` / `gpu:H200:1` at syn104 in 21:32, exit 0. Application time was 1,239 seconds and host peak was 13.4 GiB. The best state again reproduced evaluation 6. The joint infinity norm remained tracer coordinate 0 at 13,524.69, with IC RMS 2.005. One step of `1e-4` along the gradient gave an observed directional derivative 18,891.49 against the analytic value 18,883.23. The relative error is `4.37e-4`, inside the `1e-2` gate. Status: `CONDITIONAL_OPTIMIZER_FD_PASSED`. `longer_warm_start_authorized` remains false. The gradient is a real derivative of this objective, and it is still the count-amplitude coordinate.

## Nuisance-block probe — job 414744

The fixed-IC probe ran on `h200` / `gpu:H200:1` at syn104 in 16:59, exit 0. The field evolution took 43 seconds. One derivative of the 24 nuisance coordinates then took 934 seconds, above the predeclared 480-second probe limit, so no L-BFGS iteration started. Status: `CONDITIONAL_NUISANCE_BLOCK_TIMING_STOP`. The saved best objective and tracer-0 gradient, 13,524.69 at tracer value 0.299, were reproduced. `longer_warm_start_authorized` remains false. The recorded field named `population9` is nuisance index 9, which is population coordinate 0, not population coordinate 9. Its gradient there is -68.0. It is not the conditional-FP infinity norm of 700.

A 10-evaluation nuisance optimization at this cost would be several hours. The 480-second gate is not raised after the fact. The scaled step `1/|g|` would also move tracer 0 by only about `7e-5` per unit optimizer step, which is not a useful amplitude trial when each evaluation costs a quarter hour.

## Tracer-0 line — job 414887

The line search ran on `h200` / `gpu:H200:1` at syn104 in 58:09, exit 0. Four evaluations, host peak 12.8 GiB. Derivative compilation took 119 seconds and each later evaluation took about 798 seconds, so the earlier 934-second probe was mostly execution. Support shape did not change. The baseline reproduced evaluation 6. Status: `CONDITIONAL_TRACER0_LINE_IMPROVED`. `longer_warm_start_authorized` remains false.

Tracer 0 moved from 0.299378 to 0.199378. The objective fell by 343.192, of which the count term contributed 343.167 and the tracer prior 0.025. The IC prior and conditional FP were unchanged. The tracer-0 derivative changed from +13,524.69 to -6,010.14. A further step to 0.099378 raised the objective by 1,084 above the baseline, and the half step to 0.149378 also stayed above the accepted point. That half step continued below 0.199378; the sign change lies between 0.199378 and 0.299378, so the half step did not enter the bracket. Population coordinate 9 kept its derivative at 699.997 because that coordinate was not moved.

The absolute derivative fell from 13,525 to 6,010, which is not a tenfold reduction. The same objective therefore has a one-coordinate root inside that interval, and the accepted point is an improvement rather than a stationary point.

## Secant — job 414921

The secant ran on `h200` / `gpu:H200:1` at syn104 in 31:27, exit 0. It reproduced the accepted line point, tracer 0 = 0.199378 and derivative -6,010.139, then evaluated tracer 0 = 0.230145. The objective there is 8,214,499.038, which is 442.844 below the original saved best and 99.653 below the accepted line point. The count term contributed 442.826 of the drop from the saved best, and the tracer prior 0.018. The IC prior and conditional FP did not change. The tracer-0 derivative is -410.477, 33 times smaller than 13,524.69 and inside the tenfold gate. Status: `CONDITIONAL_TRACER0_LINE_AMPLITUDE_REDUCED`. The derivative is still negative, so the root lies slightly above 0.230145. A linear estimate puts the remaining move near 0.002 and the remaining objective near half a nat. That extrapolation is not a measurement, and another tracer-0 evaluation is not the next job. Population coordinate 9 remains at derivative 699.997. `longer_warm_start_authorized` remains false. This is not a joint stationary point.

## Next: population coordinate 9

Hold the IC fixed and hold tracer 0 at 0.230145. Verify that secant point, then move only population coordinate 9, the log of the first intrinsic-FP covariance diagonal. Its derivative is positive, so the first step is -0.1. A sign change stops the search. A worse objective gets one midpoint between the accepted point and the failed point, then stops. At most three trials follow the verification. A tenfold drop with a conditional-FP term that does not worsen is `CONDITIONAL_POP9_LINE_REDUCED`. This is not a joint warm start. Application budget 70 minutes, Slurm 80 minutes. H200 was fully allocated at the first submission, both H100 nodes were fully allocated, and syn102 had free A100 devices, so the job uses `a100` / `gpu:A100:1`. The existing device-peak guard still requires 20 percent headroom. Job 414929 reached the A100 and passed that guard, then failed in 46 seconds because `support` was called with the population array in the order position. No population-9 evaluation was completed. The retry passes integer order 2, which is the call used by the tracer-0 jobs.

## Population-9 line — job 414930

The retry ran on `a100` / `gpu:A100:1` at syn102 in 01:14:01, exit 0. Application time was 4,434 seconds and host peak was 12.64 GiB. The device limit was 59.44 GiB, above the 20 percent headroom required for the 30.91 GiB exact-GL2 peak. Three evaluations, then the 70-minute application budget stopped the third trial. Status: `CONDITIONAL_POP9_LINE_IMPROVED`. `longer_warm_start_authorized` remains false. This is not a joint stationary point.

The baseline reproduced the secant point: tracer 0 = 0.23014459265495463, population coordinate 9 = -0.12125573632738024, objective 8,214,499.037674768, and derivative 699.99687. Two steps of -0.1 reached population coordinate 9 = -0.32125573632738025. The objective there is 8,214,374.458668761, which is 124.579 below the secant point. The conditional-FP log likelihood rose from 1,789.637 to 1,914.261. The population prior rose from 0.057388 to 0.101639. The IC prior, tracer prior, count term, and tracer-0 derivative -410.477 stayed unchanged, so this coordinate is the conditional-FP term. The population-9 derivative fell from 699.997 to 525.563. That is not a tenfold drop, and the sign did not change. The slope between the last two points is about 1,024 per unit. A Newton step from -0.321256 is -0.513346, to -0.834602, inside the absolute cap of 1. A linear estimate of the remaining objective is about 135 nats. That estimate is not a measurement. The IC pullback was not recomputed at the moved population coordinate.

## Next: one damped Newton step

Hold the IC fixed and hold tracer 0 at 0.23014459265495463. The other nuisance coordinates stay at the saved best except population coordinate 9. Read job 414930 and require `CONDITIONAL_POP9_LINE_IMPROVED`. Verify evaluation 2 to relative objective `1e-8` and gradient agreement `1e-4` times the recorded absolute derivatives. Take one Newton step from evaluations 1 and 2, with the absolute step capped at 1. If that objective is not lower, take one midpoint between the verified point and the Newton point, then stop. Do not take another outward step. The tenfold gate still uses evaluation 0, absolute derivative 699.997, and requires the conditional-FP log likelihood not to fall. The status labels remain `CONDITIONAL_POP9_LINE_REDUCED`, `CONDITIONAL_POP9_LINE_IMPROVED`, and `CONDITIONAL_POP9_LINE_NO_IMPROVEMENT`. None of them authorizes a joint warm start. Application budget 70 minutes, Slurm 80 minutes. Select one idle H200 if one is free; otherwise one compatible H100 or A100. The device-peak guard is unchanged. An 18-evaluation IC fit remains a large calculation and needs an Astra audit before submission.

## Population-9 Newton — job 414956

The Newton step ran on `h200` / `gpu:H200:1` at syn104 in 30:36, exit 0. Application time was 1,791 seconds and host peak was 12.77 GiB. The device limit was 104.85 GiB. It verified population coordinate 9 = -0.321256, then evaluated the capped step at -0.834602. Status: `CONDITIONAL_POP9_LINE_IMPROVED`. `longer_warm_start_authorized` remains false. Heldout values were not loaded. This is not a joint stationary point.

The verification reproduced job 414930's last point. The objective there is 8,214,374.458668737 and the derivative is +525.563. At -0.834602 the objective is 8,214,365.068462489, which is 9.390 lower. The conditional-FP log likelihood rose from 1,914.261 to 1,923.948, and the population prior rose from 0.102 to 0.398. The IC prior, tracer prior, count term, and tracer-0 derivative -410.477 stayed unchanged. The population-9 derivative changed from +525.563 to -742.580. The sign bracket is therefore (-0.834602, -0.321256). The absolute derivative grew, so the tenfold gate, 69.9997, was not met. The slope fitted to the two -0.1 steps did not persist: the average slope across this jump is about 2,470 per unit, not 1,024. The earlier estimate of about 135 remaining nats is not a measurement and is not used.

## Next: one secant inside the population-9 bracket

Hold the IC fixed and hold tracer 0 at 0.23014459265495463. Require job 414956 to be `CONDITIONAL_POP9_LINE_IMPROVED`, with a positive derivative at -0.321256 and a negative derivative at -0.834602. Verify the Newton point, then evaluate one secant at population coordinate 9 = -0.534004. That value lies inside the bracket and closer to the positive-derivative end. Do not step outside the bracket and do not take a midpoint. The tenfold gate remains evaluation 0 of job 414930: absolute derivative 699.997, and the conditional-FP log likelihood must not fall below 1,789.637. The same three status labels apply. None authorizes a joint warm start. Application budget 70 minutes, Slurm 80 minutes. Select one idle H200 if one is free; otherwise one compatible H100 or A100. An 18-evaluation IC fit remains a large calculation and needs an Astra audit before submission.

## Population-9 secant — job 414961

The secant ran on `h200` / `gpu:H200:1` at syn104 in 30:37, exit 0. Application time was 1,805 seconds and host peak was 12.77 GiB. The device limit was 104.85 GiB. It verified the Newton point, then evaluated population coordinate 9 = -0.534004. Status: `CONDITIONAL_POP9_LINE_IMPROVED`. `longer_warm_start_authorized` remains false. Heldout values were not loaded. This is not a joint stationary point.

The verification reproduced the Newton point. At -0.534004 the objective is 8,214,295.992951905, which is 69.076 below the Newton point and 203.045 below the population-9 line start. The conditional-FP log likelihood rose from 1,923.948 to 1,992.817. The population prior fell from 0.398 to 0.193. The IC prior, tracer prior, count term, and tracer-0 derivative -410.477 stayed unchanged. The population-9 derivative is +179.299. Its absolute value is above the tenfold gate of 70.000. The tightened sign bracket is (-0.834602, -0.534004). A linear estimate puts the root about 0.058 beyond -0.534004 and the remaining objective near 5 nats. That estimate is not a measurement.

## Next: one secant inside the tightened bracket

Hold the IC fixed and hold tracer 0 at 0.23014459265495463. Require job 414961 to be `CONDITIONAL_POP9_LINE_IMPROVED`. Verify its secant point, derivative +179.299 at -0.534004, then evaluate one secant at -0.592468. The Newton endpoint, derivative -742.580 at -0.834602, stays the other side of the bracket. Do not step outside the bracket and do not take a midpoint. The tenfold gate is unchanged. If this evaluation also misses that gate, stop this coordinate. Do not queue a third secant. None of the status labels authorizes a joint warm start. Application budget 70 minutes, Slurm 80 minutes. Select one idle H200 if one is free; otherwise one compatible H100 or A100. An 18-evaluation IC fit remains a large calculation and needs an Astra audit before submission.

## Population-9 second secant — job 414980

The second secant ran on `h200` / `gpu:H200:1` at syn104 in 30:46, exit 0. Application time was 1,809 seconds and host peak was 12.84 GiB. The device limit was 104.85 GiB. It verified population coordinate 9 = -0.534004, then evaluated -0.592468. Status: `CONDITIONAL_POP9_LINE_REDUCED`. `longer_warm_start_authorized` remains false. Heldout values were not loaded. This is not a joint stationary point and not an R2 exit.

The verification reproduced job 414961. At -0.592468 the objective is 8,214,289.311950613, which is 6.681 below the first secant and 209.726 below the population-9 line start. The conditional-FP log likelihood rose from 1,992.817 to 1,999.531. The population prior rose from 0.193 to 0.226. The IC prior, tracer prior, count term, and tracer-0 derivative -410.477 stayed unchanged. The population-9 derivative is +46.114. That is below the tenfold gate of 70.000, and the conditional-FP log likelihood remains above its line-start value of 1,789.637. The sign did not change. A linear estimate from the last two positive derivatives puts the remaining move near 0.02 and the remaining objective near half a nat. That estimate is not a measurement. The predeclared search stops. No third secant is queued.

The IC pullback has not been recomputed at this nuisance point. The last IC-block summary, rms 2.005 and infinity norm 14.95, belongs to the original saved best. An 18-evaluation IC update is still a large calculation. It is not submitted. The next action is a mid-course Astra audit of that choice. It is not an R2 exit audit.

## Astra disposition — 2026-10-07

Astra returned verdict B in `config/cf4_r2_pop9_reduced_astra_20261007.md`. The driver adopts B. One full gradient is measured at the reduced point, with the IC pullback and all 24 nuisance components saved. The 18-evaluation IC update is not submitted. Option A is not supported: the coordinate gates do not show that the IC block at this nuisance point is the useful direction. Option C is not supported: the missing IC pullback is one evaluation, and the saved line results do not contain it. This audit does not close R2.

Two corrections are adopted. The population-9 line code builds all 24 nuisance derivatives and stores tracer 0 and population 9 only. It also passes a zero IC score gradient, so those jobs did not compute the IC pullback. The coordinate itself is the shifted log Cholesky entry `log(L00) = log(0.3) + population_white[9]`, with scale 1. The covariance diagonal is `L00` squared. Earlier wording that called it the log covariance diagonal was imprecise. The measured coordinate and its derivative are unchanged. Slurm accounting confirms job 414980 completed on syn104 in 30:46 with exit 0. Astra could not open the accounting socket.

## Next: one full gradient

Hold the IC and the reduced nuisance point: tracer 0 = 0.23014459265495463 and population coordinate 9 = -0.592468447322586. Require job 414980 to be `CONDITIONAL_POP9_LINE_REDUCED`. Evaluate the same objective once, using the field pullback rather than a zero IC score gradient. The objective must match to relative `1e-8`, and the tracer-0 and population-9 derivatives must match to `1e-4` times their recorded absolute values. Save the block summary and the 24 nuisance derivatives in the result, and the full gradient array beside it. Do not step, do not call the optimizer, and do not read held-out values. `longer_warm_start_authorized` stays false. Application budget 70 minutes, Slurm 80 minutes. Select one idle H200 if one is free; otherwise one compatible H100 or A100. This measurement does not authorize an IC update.

## Full gradient — job 415065

The gradient ran on `h200` / `gpu:H200:1` at syn104 in 17:30, exit 0. Application time was 1,018 seconds and host peak was 9.89 GiB. The device limit was 104.85 GiB. Status: `CONDITIONAL_FULL_GRADIENT_RECORDED`. The objective, tracer-0 derivative, and population-9 derivative reproduced job 414980. `longer_warm_start_authorized` remains false. Heldout values were not loaded. The optimizer was not called. The full gradient is saved beside the result. This is not a joint stationary point and not an R2 exit.

The IC block has rms 1.936337 and infinity norm 14.104463. That is the same scale as the original saved best. The joint infinity norm is tracer coordinate 2 at +7246.380965. The next largest nuisance derivatives are population 3 at +1911.610424, tracer 7 at +1482.055, and population 0 at -1352.095. Tracer 0 remains -410.477428 and population 9 remains +46.114009. An IC update is not the next measurement: its block is two orders of magnitude below the nuisance infinity norm.

Tracer 2 equals 1.201751235764841. It is the white coordinate of true-K bias 1, `exp(0.5*tracer[2])` = 1.823715, for the intrinsic bin (-25, -23.6666666666667). The same source masses enter the count and conditional-FP scores, so the tenfold gate uses their sum. Astra's requested gradient now exists. This one-coordinate step is the standing nuisance route, not another large calculation, so it does not go back to Astra.

## Next: tracer coordinate 2

Hold the IC, tracer 0, and population coordinate 9 at the recorded point. Verify job 415065, then move only tracer 2. Its derivative is positive, so the first step is -0.1. A sign change stops the search. A worse objective gets one midpoint between the accepted point and the failed point, then stops. At most three trials follow the verification. A tenfold drop, from 7246.381 to 724.638, with a combined count plus conditional-FP log likelihood that does not fall, is `CONDITIONAL_TRACER2_LINE_REDUCED`. An improved objective without that drop is `CONDITIONAL_TRACER2_LINE_IMPROVED`. Neither authorizes an IC update or a joint warm start. Application budget 70 minutes, Slurm 80 minutes. Select one idle H200 if one is free; otherwise one compatible H100 or A100.

## Tracer-2 line — job 415072

The line ran on `h200` / `gpu:H200:1` at syn104 in 57:59, exit 0. Application time was 3,441 seconds and host peak was 12.81 GiB. The device limit was 104.85 GiB. It verified tracer 2 = 1.201751235764841, then took three steps of -0.1, to 0.9017512357648408. Status: `CONDITIONAL_TRACER2_LINE_IMPROVED`. The sign did not change, and the run stopped at the three-trial cap rather than on the budget. `longer_warm_start_authorized` remains false. Heldout values were not loaded. This is not a joint stationary point.

The objective at the last point is 8,212,987.330120384, which is 1,301.982 below the verified point. The count log likelihood rose by 1,296.245 and the conditional-FP log likelihood rose by 5.422. The tracer prior fell by 0.316. The IC prior and population prior did not change. The tracer-2 derivative fell from +7246.381 to +1578.418. That remains above the tenfold gate of 724.638. The combined likelihood rose, so the missed gate is the derivative magnitude.

The same steps moved the tracer-0 derivative from -410.477 to -5855.131. Population 9 stayed near +45.849. A linear estimate from the last two tracer-2 derivatives puts another descent step near -0.088 and about 70 nats, while the tracer-0 derivative changed by about -1,840 per 0.1. Those estimates are not measurements. They are the reason this coordinate is not continued: another tracer-2 step would be aimed at a still-positive derivative while enlarging the tracer-0 derivative. The line stored only three nuisance derivatives. The other 21, and the IC pullback, were not recorded at the moved point.

## Next: one full gradient at the improved tracer-2 point

Hold the IC fixed. Use the lowest objective from job 415072: tracer 0 = 0.23014459265495463, tracer 2 = 0.9017512357648408, and population coordinate 9 = -0.592468447322586. Require `CONDITIONAL_TRACER2_LINE_IMPROVED`. Evaluate the same objective once, with the field pullback. The objective must match to relative `1e-8`, and the tracer-0, tracer-2, and population-9 derivatives must match to `1e-4` times their recorded absolute values. Save the block summary, all 24 nuisance derivatives, and the full gradient array. Do not take another tracer-2 step, do not call the optimizer, and do not read held-out values. `longer_warm_start_authorized` stays false. This is one evaluation, the same measurement already used after the population-9 reduction. It is not an 18-evaluation IC fit and does not go back to Astra. Application budget 70 minutes, Slurm 80 minutes. Select one idle H200 if one is free; otherwise one compatible H100 or A100.

## Full gradient after the tracer-2 move — job 415090

The gradient ran on `h200` / `gpu:H200:1` at syn104 in 16:27, exit 0. Application time was 957 seconds and host peak was 9.90 GiB. The device limit was 104.85 GiB. Status: `CONDITIONAL_FULL_GRADIENT_RECORDED`. It reproduced the last tracer-2 line point: objective 8,212,987.330120384, tracer-0 derivative -5855.130694, tracer-2 derivative +1578.417819, and population-9 derivative +45.849414. `longer_warm_start_authorized` remains false. Heldout values were not loaded. The optimizer was not called.

The IC block has rms 1.868253 and infinity norm 12.631119. The joint infinity norm is tracer 0. Population coordinate 3 remains +1912.610, and the other large population derivatives changed by about 1 or less from the gradient recorded before the tracer-2 move. Tracer 7 moved from +1482.055 to -516.042, tracer 3 from +294.516 to -220.239, and tracer 8 from -497.808 to +214.432. The population block did not absorb the tracer-2 step. An IC update is still not the next measurement.

## Next: revisit tracer 0 at this point

Hold the IC, tracer 2 = 0.9017512357648408, and population coordinate 9 = -0.592468447322586. Require job 415090 to be `CONDITIONAL_FULL_GRADIENT_RECORDED`. Verify that point, then move only tracer 0. Its derivative is negative, so the first step is +0.1. A sign change stops the search. A worse objective gets one midpoint, then stops. At most three trials follow the verification. Each trial records the tracer-2 derivative as well. A tenfold drop, from 5855.131 to 585.513, with a count log likelihood that does not fall, uses the existing tracer-0 status labels. Neither an improvement nor a reduction authorizes an IC update. The earlier tracer-0 search is not being reopened at the old point; this point is the one created by the tracer-2 move. Application budget 70 minutes, Slurm 80 minutes. Select one idle H200 if one is free; otherwise one compatible H100 or A100.

## Tracer-0 revisit — job 415094

The revisit ran on `h200` / `gpu:H200:1` at syn104 in 30:25, exit 0. Application time was 1,798 seconds and host peak was 12.76 GiB. The device limit was 104.85 GiB. It verified tracer 0 = 0.23014459265495463, then took one step of +0.1. Status: `CONDITIONAL_TRACER0_LINE_NO_IMPROVEMENT`. The sign changed, so the search stopped before a midpoint. `longer_warm_start_authorized` remains false. Heldout values were not loaded. This is not a joint stationary point.

At tracer 0 = 0.33014459265495466 the objective is 8,213,347.680982311, which is 360.351 worse. The count log likelihood fell by 360.323. The conditional-FP term stayed at 2,004.9531130185724, and the IC and population priors did not change. The tracer-0 derivative changed from -5855.131 to +13714.015. The tracer-2 derivative rose from +1578.418 to +3599.493. The bracket is (0.23014459265495463, 0.33014459265495466). The accepted point remains the verified endpoint because the outward step is worse.

## Next: one secant inside the tracer-0 bracket

Hold the IC, tracer 2 = 0.9017512357648408, and population coordinate 9 = -0.592468447322586. Require job 415094 to be `CONDITIONAL_TRACER0_LINE_NO_IMPROVEMENT`. Verify the negative-derivative endpoint, then evaluate one secant at tracer 0 = 0.26006480881995014. Do not step outside the bracket and do not take a midpoint. Record the tracer-2 derivative. The tenfold gate remains absolute derivative 585.513, and the count log likelihood must not fall below -142849.21378243115. The same tracer-0 status labels apply. None authorizes an IC update. Application budget 70 minutes, Slurm 80 minutes. Select one idle H200 if one is free; otherwise one compatible H100 or A100.

## Tracer-0 secant — job 415106

The secant ran on `h200` / `gpu:H200:1` at syn104 in 30:33, exit 0. Application time was 1,806 seconds and host peak was 12.78 GiB. The device limit was 104.85 GiB. It verified tracer 0 = 0.23014459265495463, then evaluated 0.26006480881995014. Status: `CONDITIONAL_TRACER0_LINE_AMPLITUDE_REDUCED`. `longer_warm_start_authorized` remains false. Heldout values were not loaded. This is not a joint stationary point.

The objective is 8,212,892.871417244, which is 94.459 below the verified point. The count log likelihood rose by 94.466. The conditional-FP term stayed at 2,004.9531130185724. The tracer prior rose by 0.007. The tracer-0 derivative is -404.55082315761325, inside the tenfold gate of 585.513. The sign is still negative. A linear estimate from the verified point and this secant puts the remaining move near 0.002 and the remaining objective near half a nat. That estimate is not a measurement. The tracer-0 coordinate stops, as it did after the first secant met the same kind of gate. The tracer-2 derivative rose from +1578.418 to +2141.346. Population 9 stayed at +45.849. The other nuisance derivatives and the IC pullback were not recorded at this moved tracer 0.

## Next: one full gradient at the reduced tracer-0 point

Hold the IC fixed. Use the secant point from job 415106: tracer 0 = 0.26006480881995014, tracer 2 = 0.9017512357648408, and population coordinate 9 = -0.592468447322586. Require `CONDITIONAL_TRACER0_LINE_AMPLITUDE_REDUCED`. Evaluate the same objective once, with the field pullback. The objective must match to relative `1e-8`, and the tracer-0, tracer-2, and population-9 derivatives must match to `1e-4` times their recorded absolute values. Save the block summary, all 24 nuisance derivatives, and the full gradient array. Do not take another tracer step, do not call the optimizer, and do not read held-out values. `longer_warm_start_authorized` stays false. This one evaluation does not authorize an IC update and does not go back to Astra. Application budget 70 minutes, Slurm 80 minutes. Select one idle H200 if one is free; otherwise one compatible H100 or A100.

## Full gradient after the tracer-0 secant — job 415110

The gradient ran on `h200` / `gpu:H200:1` at syn104 in 16:26, exit 0, from commit 172f139. Application time was 964 seconds and host peak was 9.90 GiB. The device limit was 104.85 GiB. Status: `CONDITIONAL_FULL_GRADIENT_RECORDED`. It reproduced the secant point: objective 8,212,892.871417244, tracer 0 = 0.26006480881995014, tracer 2 = 0.9017512357648408, population coordinate 9 = -0.592468447322586, tracer-0 derivative -404.55082315761325, tracer-2 derivative +2141.3461173787564, and population-9 derivative +45.84941387404232. `longer_warm_start_authorized` remains false. Heldout values were not loaded. The optimizer was not called.

The IC block has rms 1.894683 and infinity norm 12.766063. The joint infinity norm is tracer 2. Every population derivative changed by at most 6.82e-13 from job 415090. The tracer-0 step of +0.029920 moved tracer 0 by +5450.580, tracer 2 by +562.928, tracer 3 by +674.317, tracer 7 from -516.042 to +790.060, and tracer 8 from +214.432 to -185.978. The largest nuisance derivatives are tracer 2 at +2141.346, population 3 at +1912.610, population 0 at -1352.818, population 2 at -864.756, and tracer 7 at +790.060. An IC update is still not the next measurement.

The tracer-0 column, from the secant verify point to this secant, and the last tracer-2 step, from 1.001751 to 0.901751 at tracer 0 = 0.230145, give a 2×2 curvature with positive diagonal and positive determinant. The two cross derivatives, 18,382.66 and 18,814.31, differ by 2.3 percent. One Newton step from those columns is +0.016033 in tracer 0 and -0.136883 in tracer 2, inside the component cap of 0.2. The local quadratic model drops about 150 nats. That model is not a measurement. A tracer-2-only continuation would aim at the still-positive tracer-2 derivative while driving the tracer-0 derivative back through the gate of 585.513, about -1,840 per 0.1 from the last measured tracer-2 segment. The next measurement is therefore one joint step.

## Next: one capped step in tracer 0 and tracer 2

Hold the IC and population coordinate 9. Require job 415110 to be `CONDITIONAL_FULL_GRADIENT_RECORDED`. Build the tracer-0 curvature column from the two evaluations in job 415106, which share tracer 2. Build the tracer-2 column from the last two evaluations in job 415072, which share tracer 0 = 0.23014459265495463 and end at the current tracer 2. The archives must agree at those shared points. Verify the job 415110 point to relative objective `1e-8` and to `1e-4` times the absolute tracer-0, tracer-2, and population-9 derivatives. Then evaluate one nuisance-only point at the Newton step, passing integer support order 2 and a zero IC score gradient. The component cap is 0.2. If that objective is worse, evaluate one midpoint of tracer 0 and tracer 2 and stop. There is no second Newton step. A tenfold drop in both absolute derivatives, from 404.551 and 2141.346, with a count plus conditional-FP log likelihood that does not fall, is `CONDITIONAL_TRACER_PAIR_REDUCED`. An improved objective without both drops is `CONDITIONAL_TRACER_PAIR_IMPROVED`. A missed objective is `CONDITIONAL_TRACER_PAIR_NO_IMPROVEMENT`. None of these authorizes an IC update. The step uses the same objective and derivatives already in hand, so it does not go back to Astra. The 18-evaluation IC fit remains a separate large calculation. Application budget 70 minutes, Slurm 80 minutes. Select one idle H200 if one is free; otherwise one compatible H100 or A100.

## Tracer pair — job 415116

The pair ran on `h200` / `gpu:H200:1` at syn104 in 30:24, exit 0, from commit dd5b312. Application time was 1,810 seconds and host peak was 12.79 GiB. The device limit was 104.85 GiB. It verified the job 415110 point, then evaluated tracer 0 = 0.2760982361105669 and tracer 2 = 0.7648686304974732. Status: `CONDITIONAL_TRACER_PAIR_IMPROVED`. The objective improved, so no midpoint was evaluated. `longer_warm_start_authorized` remains false. Heldout values were not loaded. The optimizer was not called. This is not a joint stationary point.

The objective is 8,212,729.856747742, which is 163.015 below the verified point. The count log likelihood rose by 161.743 and the conditional-FP log likelihood rose by 1.162. The tracer prior fell by 0.110. The IC prior and the population prior did not change. The tracer-0 derivative is -29.728881333040174, inside the tenfold gate of 40.455, and its sign is still negative. The tracer-2 derivative is +237.64288261095007, above the tenfold gate of 214.135, and its sign is still positive. Population 9 stayed at +45.738. The local quadratic model had put the drop near 150 nats. That model is not this measurement.

The three stored derivatives do not locate the joint infinity norm. At the previous point population 3 was +1912.610, but this job did not remeasure that component, the other nuisance derivatives, or the IC pullback. Another joint Newton step is not formed from these two evaluations, because tracer 0 and tracer 2 moved together. A tracer-2-only nudge large enough to bring 237.643 under 214.135 would also move the tracer-0 derivative through the cross coupling measured earlier, about 1.8e4, and back outside the tracer-0 gate. The pair coordinate stops at this improved point.

## Next: one full gradient at the improved tracer-pair point

Hold the IC fixed. Use the Newton point from job 415116: tracer 0 = 0.2760982361105669, tracer 2 = 0.7648686304974732, and population coordinate 9 = -0.592468447322586. Require `CONDITIONAL_TRACER_PAIR_IMPROVED`. Evaluate the same objective once, with the field pullback. The objective must match to relative `1e-8`, and the tracer-0, tracer-2, and population-9 derivatives must match to `1e-4` times their recorded absolute values. Save the block summary, all 24 nuisance derivatives, and the full gradient array. Do not take another tracer step, do not call the optimizer, and do not read held-out values. `longer_warm_start_authorized` stays false. This one evaluation does not authorize an IC update and does not go back to Astra. Application budget 70 minutes, Slurm 80 minutes. Select one idle H200 if one is free; otherwise one compatible H100 or A100.

## Line search executed by job 414887

The same objective was kept. The saved IC and the other 23 nuisance coordinates stay fixed. Tracer 0, whose negative-log-likelihood derivative is positive, is moved by a predeclared -0.1. If that point is worse, one half step of -0.05 is tried and the search stops. If it is better, another -0.1 is taken from the accepted point. At most three trials follow the baseline. The derivative is compiled once; a support-shape change stops the search. The baseline must reproduce evaluation 6. A tenfold drop in the absolute tracer-0 gradient with a count term that does not worsen is `CONDITIONAL_TRACER0_LINE_AMPLITUDE_REDUCED`. An improved objective without that drop is `CONDITIONAL_TRACER0_LINE_IMPROVED`. Otherwise the line did not improve the objective. None of these authorizes a joint warm start. Application budget 70 minutes, Slurm 80 minutes. This remains a routine one-coordinate check, not the 18-evaluation IC fit that needs an Astra audit.
