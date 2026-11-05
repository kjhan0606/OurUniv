# R2 joint field-MCSE stress result — 2026-11-05

## Decision

Accept the result as a **single fixed negative-control risk bound only**.
Do not treat it as production field-gate calibration, sampler qualification,
posterior evidence, held-out validation, or R2 exit. R2 remains NO-GO.

## Execution and provenance

- Syntax Slurm job450919, partition `a10`, node `syn04`, CPU-only.
- Source commit: `c2053cf3f6433f0e7f4746f655698389f7521175`.
- Runtime11:04, exit0, MaxRSS262,012KiB under2GiB requested.
- Five startup suites passed:9+6+17+7+4=43 tests.
- Frozen design: 16^3 map, four independent synthetic chains,512 states per
  chain,256 paired replicates, batch lengths32/64, full-box and fixed147-cell
  central reporting region. No target/CF4 values, held-out outcomes, or
  posterior fields were accessed.
- Raw artifacts and logs:
  `/gpfs/kjhan/CF4/z0_density/r2_joint_field_mcse_stress_20261105/450919/`
  and `/gpfs/kjhan/CF4/logs/cf4_r2_joint_mcse_450919.{out,err}`.

The completed result is source-pinned and all256 replicate indices are unique
and present. The serialized paired events agree with the aggregate counts;
the expected cells are finite/defined. Astra's read-only mid-course audit
independently verified the plan, source, oracle, records, logs, and accounting.
Its verdict is **PASS for this fixed negative control; NO-GO for production
gate qualification and R2 exit**.

## Oracle and false-acceptance result

For the predeclared slow velocity-y component (`phi=0.9`), the streamed
250,000-history conditional oracle gives a finite-sample pooled mean-error
ratio of `0.1090723771`. Its one-sided97.5% Student-t/delta-method lower bound
is `0.1090667306`, above the0.10 violation threshold. Relative oracle MCSE is
`2.64e-5`; zero-occupancy probability under the specified occupancy model is
`6.91e-156`, and none of the250,000 histories had zero occupied samples. The
oracle interval is approximate, not an exact finite-sample or joint guarantee.

There were0 false acceptances in256 replicates. The exact one-sided
Clopper-Pearson upper bounds are:

- 1.1634% at95% confidence, meeting the stated5% risk criterion for this
  alternative;
- 1.4306% at97.5%, the binomial part of the nominal Bonferroni95% statement.

The complete gate passed0/256; MCSE-only passed0/256; cross-chain-only passed
0/256. At both batch lengths, velocity-y mean precision failed in both
reporting regions on all256 replicates. Thus the result bounds false
acceptance under one known-inadequate law, but says nothing about the gate's
acceptance rate for adequate fields or its ability to discriminate in a
production posterior. The asymmetric oracle/binomial confidence construction
is only nominal familywise95% because the oracle interval uses an approximate
Student-t/delta-method bound.

## Cross-chain interpretation

The cross-chain comparator is the spread of **individual chain means**, not
the Monte-Carlo standard error of the pooled field mean. For four comparable,
independent chains, the former is approximately `sqrt(4)=2` times the latter.
The finite-sample pooled velocity-x/z reference ratios are each0.05342, while
their median cross-chain mean-disagreement p95 ratios are about0.174 in the
full box and0.170 in the central region. Their cross-chain mean-map checks
therefore fail even though the pooled references are below0.10. This is
consistent with the statistic's scale and does not indicate nonstationarity.
The0.10 cross-chain cutoff is a separate, stronger per-chain-agreement
condition than a0.10 pooled-precision requirement. Its scientific purpose
and sampling-cost implications need explicit reconciliation before any
positive-control case or production qualification. No threshold relaxation
or positive-control job is authorized by this result.

## Follow-up correction and disposition

Astra found a nonblocking result-label edge case: zero false accepts with too
few replicates could be labelled `RISK_BOUND_MET` even when its exact upper
bound exceeded5%. The production result is unaffected because its256-trial
bound passes. The status helper was corrected and regression-tested in
commit `b8970cb5dfe6f2c6abd738d5f790e6bdc243a0af`; Syntax job450966 passed all
10 focused tests in7s. Job450919 was not rerun.

The selected-mark parent/availability/group-linkage law remains unresolved;
better sampler or MCSE behavior cannot supply that missing denominator. Do
not read held-out outcomes or launch the multi-chain target. Use existing
evidence rather than repeat mock or catalogue searches. No email was sent.

**Q-GOAL:** this supplies one narrow uncertainty-method result for the
384cMpc/h, N256/1.5cMpc/h conditional z=0 inference, not the requested
density/velocity posterior, observation calibration, uncertainty product,
or held-out prediction. MW/M31 remain role-ambiguous and M33 unresolved; their
observables must ultimately constrain those same roles on the same NEW
evolved LG field at `<=0.3cMpc/h`, with native truth IDs reserved for
calibration/evaluation.

**Q-LEAN:** the11-minute fixed case was proportionate. Reuse it; do not turn
it into a batch/threshold/correlation ladder. First state whether the
cross-chain check is intended to constrain pooled MCSE or impose a stricter
per-chain agreement rule. The decisive separate scientific dependency is
source-matched selection information; R2 remains NO-GO.
