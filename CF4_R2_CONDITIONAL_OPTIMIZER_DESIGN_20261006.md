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

## Line search executed by job 414887

The same objective was kept. The saved IC and the other 23 nuisance coordinates stay fixed. Tracer 0, whose negative-log-likelihood derivative is positive, is moved by a predeclared -0.1. If that point is worse, one half step of -0.05 is tried and the search stops. If it is better, another -0.1 is taken from the accepted point. At most three trials follow the baseline. The derivative is compiled once; a support-shape change stops the search. The baseline must reproduce evaluation 6. A tenfold drop in the absolute tracer-0 gradient with a count term that does not worsen is `CONDITIONAL_TRACER0_LINE_AMPLITUDE_REDUCED`. An improved objective without that drop is `CONDITIONAL_TRACER0_LINE_IMPROVED`. Otherwise the line did not improve the objective. None of these authorizes a joint warm start. Application budget 70 minutes, Slurm 80 minutes. This remains a routine one-coordinate check, not the 18-evaluation IC fit that needs an Astra audit.
