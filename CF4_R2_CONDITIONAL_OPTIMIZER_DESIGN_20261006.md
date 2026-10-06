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

## Next, still not a warm start

The infinity norm is one count-amplitude coordinate, so the planned IC warm start is not the next fit. The missing check is the one directional finite difference at the saved best state. It is a routine completion of this diagnostic: one exact gradient and one stepped value, application budget 40 minutes, Slurm 45 minutes, no component adjoints and no optimizer. At the submission check, syn104 had all 8 H200 devices allocated, both H100 nodes had all 5 devices allocated, and neither A100 node had a free device. The queued job still uses `h200` / `gpu:H200:1`, the device on which job 414485 measured this target. If it misses the relative `1e-2` gate, no rescaling and no longer fit follow. If it passes, the next change is a same-objective nuisance scaling or block order whose stationary point is unchanged. That short test comes before any 16,777,216-coordinate warm start. A warm start of 18 evaluations remains a large calculation and needs an Astra audit before submission. This finite-difference completion does not.
