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

## Interpretation and next action

The result establishes that the corrected conditional target has a finite
exact-GL2 gradient and can make accepted local transitions at this checkpoint
at a measured cost of about6.8 minutes each. The repeated negative ΔH and
small monotonic white-power change are consistent with continued movement
away from the initializer; eight transitions cannot establish equilibrium,
mixing, effective sample size or uncertainty. In particular, “8/8 accepted”
must not be quoted as a calibrated acceptance rate.

Continue the same checkpointed chain for16 additional one-step transitions at
the unchanged target, metric and step. Record exact target energy at every
state so that relaxation versus a stationary plateau can be checked before
any production-length request. This is still a bounded mechanics continuation,
not posterior production. No GL1 force, parameter ladder, new gravity run or
held-out access is authorized by this test.

**Q-GOAL:** exact-target dynamics are necessary for the current N256/1.5 z=0
posterior route but are not the density map or zoom IC. MW/M31 remain
role-ambiguous and M33 unresolved; their observables must eventually constrain
the same NEW evolved LG field at `<=0.3 cMpc/h`. Native truth identities remain
calibration/evaluation-only.

**Q-LEAN:** one exact-force eight-transition trace and one next bounded
continuation; no new likelihood factor, survey census, simulation, held-out
score or sampler ladder. R2 remains NO-GO: calibrated source/group selection,
shared-member covariance, stationarity, uncertainty, held-out prediction and
the z=0 map remain unresolved.
