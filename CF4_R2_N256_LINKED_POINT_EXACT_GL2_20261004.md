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

## Interpretation and next action

The result establishes that the corrected conditional target has a finite
exact-GL2 gradient and can make accepted local transitions at this checkpoint
at a measured cost of about6.7 minutes each. The repeated negative ΔH and
evolving white power/target energy are consistent with continued movement
away from the initializer;24 transitions cannot establish equilibrium,
mixing, effective sample size or uncertainty. In particular, “24/24 accepted”
must not be quoted as a calibrated acceptance rate.

Continue the same checkpointed chain for16 additional one-step transitions at
the unchanged target, metric and step. Record prior energy and both log-score
components separately to diagnose which term drives the energy change. This is
still a bounded warm-up/mechanics continuation, not posterior production. No
GL1 force, parameter ladder, new gravity run or held-out access is part of this
diagnostic.

**Q-GOAL:** exact-target dynamics are necessary for the current N256/1.5 z=0
posterior route but are not the density map or zoom IC. MW/M31 remain
role-ambiguous and M33 unresolved; their observables must eventually constrain
the same NEW evolved LG field at `<=0.3 cMpc/h`. Native truth identities remain
calibration/evaluation-only.

**Q-LEAN:** exact-force continuation with target-term decomposition only; no
new likelihood factor, survey census, simulation, held-out score or sampler
ladder. R2 remains NO-GO: calibrated source/group selection,
shared-member covariance, stationarity, uncertainty, held-out prediction and
the z=0 map remain unresolved.
