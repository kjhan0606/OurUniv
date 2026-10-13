# Corrected-target N256 exact-GL2 sampler check — 2026-10-04

## Scope

This is sampler-mechanics evidence for the conditional R2 target, not a z=0
posterior, map, LG identification or held-out test. The field is N=256 in a
384 cMpc/h box (1.5 cMpc/h cells); the frozen count observations remain on
N128. It uses all47,121 v6 training counts and1,414 reconciled conditional
FP rows, each conditioned on its securely linked 2M++ point radius
(22.301–179.868 cMpc/h; ownership ledger commit
`c6d9d104dc7d1629d5c86cfea4a2819a479f9381`). Held-out outcomes were not read.

The force and Metropolis target are both exact GL2. Metric mass parameter is
6000, step size0.08, with one split-HMC integration step per proposal. Job
412367 began from the corrected-target checkpoint left unchanged by the four
rejected GL1/GL2 proposals in412356. Job412371 continued its accepted
checkpoint and restored the saved RNG state. Both used typed H100 Slurm with
48GiB host-memory requests; there was no manual node execution.

## Trace

| Transition | ΔH | MH probability | Accepted | IC-white jump RMS | Seconds | Current white mean-square |
|---:|---:|---:|:---:|---:|---:|---:|
| 1 | -2.3695 | 1.000 | yes | 0.066073 | 446.0 | 0.96275310 |
| 2 | -2.5889 | 1.000 | yes | 0.066078 | 453.0 | 0.96285613 |
| 3 | -2.1183 | 1.000 | yes | 0.066077 | 401.1 | 0.96289120 |
| 4 | -1.9890 | 1.000 | yes | 0.066069 | 388.9 | 0.96297624 |
| 5 | -2.5367 | 1.000 | yes | 0.066071 | 386.4 | 0.96306639 |
| 6 | -1.2421 | 1.000 | yes | 0.066055 | 387.7 | 0.96311799 |
| 7 | -1.7738 | 1.000 | yes | 0.066100 | 395.4 | 0.96319540 |
| 8 | -1.2083 | 1.000 | yes | 0.066083 | 388.9 | 0.96329203 |

All8/8 proposals were accepted. Mean ΔH was-1.978, all values were negative,
and the IC-white mean-square rose from the N256 initializer's0.962720 to
0.963292. Per-transition costs averaged406s (range386–453s). The two Slurm
segments took16:32 and55:50; the largest observed host RSS was12.46GiB
(13,065,988KiB). The estimated exact-gradient device peak was30.91GiB against
69.81GiB available, satisfying the20% margin.

## Same-chain continuation: transitions 9–24

Typed-H100 job412389 resumed the accepted checkpoint and RNG state from
transition8 and completed16 more one-step proposals with no target, metric or
step changes. It exited0 in1:54:32; MaxRSS was13,798,156KiB under the48GiB
request. The device estimate remained30.91GiB/69.81GiB. Every proposal was
accepted, but this single development chain does not estimate a calibrated
acceptance rate.

| Chain transition | ΔH | Exact target energy | Accepted | IC-white mean-square | Seconds |
|---:|---:|---:|:---:|---:|---:|
| 9 | -1.4758 | 8,220,444.546 | yes | 0.96332482 | 469.3 |
| 10 | -1.4490 | 8,220,964.872 | yes | 0.96340168 | 390.8 |
| 11 | -0.9422 | 8,221,345.493 | yes | 0.96345586 | 394.9 |
| 12 | -1.1477 | 8,221,967.910 | yes | 0.96354184 | 392.0 |
| 13 | -1.0276 | 8,222,224.741 | yes | 0.96358418 | 391.7 |
| 14 | -0.8842 | 8,222,806.616 | yes | 0.96366511 | 398.6 |
| 15 | -1.1478 | 8,223,538.821 | yes | 0.96376584 | 389.3 |
| 16 | -0.5274 | 8,224,184.053 | yes | 0.96385296 | 388.6 |
| 17 | -0.7789 | 8,224,801.037 | yes | 0.96393742 | 393.9 |
| 18 | -0.3728 | 8,225,788.437 | yes | 0.96406212 | 387.3 |
| 19 | -0.3776 | 8,226,485.367 | yes | 0.96415451 | 385.1 |
| 20 | -0.1622 | 8,226,990.879 | yes | 0.96422274 | 390.9 |
| 21 | -0.7389 | 8,227,513.195 | yes | 0.96429484 | 387.1 |
| 22 | -0.5603 | 8,227,389.589 | yes | 0.96429064 | 384.3 |
| 23 | -0.6320 | 8,227,870.802 | yes | 0.96435892 | 385.4 |
| 24 | -0.4468 | 8,228,578.904 | yes | 0.96445265 | 400.6 |

The16-step segment mean ΔH was-0.791945 (range-1.475789 to-0.162199), and
mean transition time was399.1s. Its target energy rose about8,305.7 nat from
the starting state; transition22 moved down slightly from21, so the path was
not strictly monotone. Across all24 transitions, all24 were accepted, mean
ΔH was-1.187405, and white mean-square moved from the initializer's0.962720
to0.964453. Continued energy and white-power drift mean this is not evidence
of stationarity or a posterior sample.

Job412389 predates target-component tracing, so its saved rows lack an exact
prior/count/raw-mark decomposition. The next continuation adds per-evaluation
and per-transition values for prior energy, training-count log score and
conditional raw-mark log score, checking that the latter two sum to the joint
observation score before computing the Hamiltonian target. This is diagnostic
instrumentation only; it does not change the target.

## Component-traced continuation and fail-closed stop

Typed-H100 job412443 restored the job412389 checkpoint/RNG and completed13 of
16 requested one-step exact-GL2 transitions before a numerical consistency
guard stopped the run. It exited FAILED after1:41:26; MaxRSS was13,417,552KiB
under the48GiB request. All13 completed proposals were accepted. Their mean
ΔH was-0.357929 (range-0.707945 to-0.037444); exact target energy rose from
8,228,578.904 to8,234,475.760 nat, while IC-white mean-square moved from
0.964453 to0.965234. This is still one drifting warm-up chain, not a posterior
or stationarity result.

For these13 accepted states, prior energy rose by6,557.653 nat, the count
log-score improved by660.082 nat (therefore lowering negative log target by
that amount), and the conditional raw-mark log-score improved by0.715 nat.
Thus the net target-energy increase was5,896.856 nat and was dominated by the
prior-energy rise, partially offset by the count term. The raw-mark
contribution was small along this path. This is component attribution for
these accepted warm-up moves only, not a scientific decomposition of
posterior information or a convergence diagnosis.

The next proposal was not accepted or rejected: it stopped before the MH
uniform draw. At the same candidate, the value-and-gradient call gave target
energy8,234,648.382990 nat and the separate value-only compilation gave
8,234,648.377517 nat, a0.005473 nat difference entirely in the training-count
score; prior and raw-mark terms were identical. The1e-7 guard correctly
prevented silently treating numerically distinct compiled primals as
identical. Inspection showed that this was the exact-GL2 chain, where force
and target already use the same value-and-gradient function; recomputing that
same scalar via a separately compiled value-only path was redundant. The
driver now uses the endpoint value returned with the exact GL2 gradient for
the MH Hamiltonian, without loosening the independent-path guard used when
force and target are genuinely distinct.

The accepted checkpoint/RNG remains at transition13. Because the failed
proposal consumed its momentum before the primal guard but the MH uniform is
drawn only afterward, restoring that accepted RNG state replays the same
proposal and then continues the stream consistently. Resume only the three
remaining transitions, with no changed target, metric or step. Do not count
the aborted candidate as a chain transition.

Typed-H100 job412509 did that exact replay and completed transitions14–16
(the last three of the requested segment) in24:02; MaxRSS was10,654,056KiB
under48GiB. The first replayed candidate exactly matches the saved failed
candidate's value-only primal, and all3 transitions were accepted. This
confirms checkpoint/RNG continuity and removes the duplicate value-only
compilation from the exact-GL2 endpoint; the independent primal guard remains
for cases where force and target are genuinely different.

| Segment transition | ΔH | Prior energy | Count log score | Raw-mark log score | IC-white mean-square |
|---:|---:|---:|---:|---:|---:|
| 1 | -0.364885 | 8,090,797.9 | -144,078.037 | 5,979.788 | 0.96449719 |
| 2 | -0.609558 | 8,090,965.1 | -144,000.541 | 5,979.985 | 0.96451712 |
| 3 | -0.707945 | 8,092,221.6 | -143,925.738 | 5,979.931 | 0.96466691 |
| 4 | -0.124709 | 8,092,586.7 | -143,866.225 | 5,980.159 | 0.96471043 |
| 5 | -0.220051 | 8,092,922.1 | -143,830.205 | 5,980.640 | 0.96475041 |
| 6 | -0.481058 | 8,093,516.8 | -143,777.941 | 5,980.440 | 0.96482130 |
| 7 | -0.433041 | 8,093,860.8 | -143,734.807 | 5,980.567 | 0.96486232 |
| 8 | -0.173460 | 8,094,236.5 | -143,700.877 | 5,980.734 | 0.96490711 |
| 9 | -0.333515 | 8,095,014.3 | -143,648.528 | 5,981.182 | 0.96499982 |
| 10 | -0.302627 | 8,095,907.5 | -143,608.004 | 5,981.064 | 0.96510630 |
| 11 | -0.393075 | 8,096,192.9 | -143,561.187 | 5,980.822 | 0.96514033 |
| 12 | -0.037444 | 8,096,601.1 | -143,517.481 | 5,980.735 | 0.96518899 |
| 13 | -0.471706 | 8,096,981.9 | -143,473.936 | 5,980.106 | 0.96523439 |
| 14 | +0.069113 | 8,097,184.7 | -143,443.334 | 5,979.652 | 0.96525856 |
| 15 | -0.085214 | 8,097,848.0 | -143,420.045 | 5,979.953 | 0.96533763 |
| 16 | -0.177238 | 8,098,899.2 | -143,379.656 | 5,980.021 | 0.96546294 |

Across all16, mean ΔH was-0.302901 (range-0.707945 to+0.069113), mean
proposal time376.7s, and all16 proposals were accepted. Relative to the
transition24 starting state, the target energy rose7,719.903 nat. Its
components changed by +8,474.895 prior energy, +754.362 count log-score and
+0.630 raw-mark log-score; because target energy is prior minus both scores,
the count and raw terms partially offset the prior rise. White mean-square
rose0.964453->0.965463. This continues the drift seen earlier in the chain,
not a stationarity or posterior pass.

## Interpretation and next action

The result establishes that the corrected conditional target has a finite
exact-GL2 gradient and can make local transitions at this checkpoint. The
original independent value-only endpoint added about96s per move; after
removing that redundant compilation, the replayed exact-force proposals cost
about4.9 minutes each. Across40 transitions the white power and target
energy continue to evolve, so this trace does not establish equilibrium,
mixing, effective sample size or uncertainty. In particular, “40/40
accepted” must not be quoted as a calibrated acceptance rate.

Typed-H100 job412532 completed the planned same-state path comparison in
43:24 (MaxRSS13,315,880KiB under48GiB); all10 focused HMC tests passed. It
re-evaluated the transition40 checkpoint at exactly the saved energy and used
one seeded common momentum, step0.08 and metric mass6000. The paths did not
modify q, p, the accepted-chain RNG/checkpoint, or any heldout outcome.
The complete machine-readable result is
[`result.json`](/gpfs/kjhan/CF4/z0_density/r2_n256_gl2_trajectory_length_124_20261005/result.json).

| Integration steps | ΔH | Clipped acceptance probability | IC-white jump RMS | Elapsed seconds |
|---:|---:|---:|---:|---:|
| 1 | -0.039157 | 1.000 | 0.066083 | 298.9 |
| 2 | -0.379040 | 1.000 | 0.132088 | 578.6 |
| 4 | -2.673129 | 1.000 | 0.263556 | 1175.7 |

Movement per second is approximately2.21e-4,2.28e-4,2.24e-4 white-coordinate
RMS, respectively: longer paths produced proportionally more displacement at
proportionally more cost in this single momentum. Meanwhile the magnitude of
ΔH rose, and the prior component increased by689.698,2,524.917,9,548.832nat;
count log-score improved29.001,146.714,607.984nat, while raw-mark score
changed only-1.501,-2.065,-0.474nat. The negative ΔH values make all three
clipped probabilities1, but do not estimate a chain acceptance rate. This is
not evidence of stationarity or posterior movement efficiency. It does not
justify four-step transitions or more warm-up on the present partial target.

The generated JSON's generic `matched_settings.integration_steps` was
incorrectly left at the old default8 although its explicit path list and all
three rows correctly report1,2,4. Preserve that raw artifact; the report
writer is corrected for future runs, and this note is the authoritative
interpretation of job412532.

Next return to the unresolved R2 observation-law blocker: a defensible
shared-group count/CF4-mark factor with calibrated group inclusion and
multi-member covariance. Reuse the frozen v6 source graph and existing
Tully/CF4 identity work; do not repeat catalogue censuses, invent covariance,
run gravity, fit, score heldout data, or extend this chain. First establish
whether already available independent sources can identify the missing terms;
otherwise specify the exact public calibration input or physically validated
mock needed before further code or sampling.

**Q-GOAL:** exact-target dynamics are necessary for the current N256/1.5 z=0
posterior route but are not the density map or zoom IC. MW/M31 remain
role-ambiguous and M33 unresolved; their observables must eventually constrain
the same NEW evolved LG field at `<=0.3 cMpc/h`. Native truth identities remain
calibration/evaluation-only.

**Q-LEAN:** the one-state/one-momentum 1/2/4 comparison was bounded and left
the chain untouched. No new likelihood factor, survey census, simulation,
held-out score or long chain. R2 remains NO-GO: calibrated source/group selection,
shared-member covariance, stationarity, uncertainty, held-out prediction and
the z=0 map remain unresolved.
