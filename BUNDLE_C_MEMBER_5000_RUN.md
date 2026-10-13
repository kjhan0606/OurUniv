# Single-field5000 continuation: execution record

2026-09-11 user requests5000 updates. Frozen submitted plan:
`BUNDLE_C_MEMBER_5000_PLAN.md`. This means cumulative steps301..5000 from the
300-update checkpoint,4700 new updates, not5000 additional and not a new fit.
No automatic multi-field/actual-data/field-prior/IC task follows this endpoint.

## Fable decision and driver disposition

Fable5 completed normally58623ms, CONDITIONAL GO. Main model claude-fable-5,
auxiliary Haiku routing, no fallback auditor. Exact response
`config/cf4_member_5000_fable5_v1.response.json`. Q-GOAL/Q-LEAN accepted for this
limited one-field question; both outcome/generalization claims are restricted.

Only blocking condition: before update301, missing/inconsistent Adam moments,
step counts or failed step300 reproduction must HARD ABORT, never initialize a
fresh optimizer. Implemented by restore_member_fit(), strict source/config
checks and comparison to the saved native step300 metrics. No recovery/fallback
branch exists. All optimizer states must have the expected300 updates. The
one new regression checks the resumed next update and Adam moments against an
uninterrupted update, plus missing/mismatched-state rejection.

Budget shortfall is INCONCLUSIVE_BUDGET; other exceptions/incomplete work are
INCOMPLETE_EXECUTION (unavailable native target retains its specific label).
Fixed5000 result is PASS_SINGLE_FIELD_LEARNING_ONLY or
NO_GO_SINGLE_FIELD_LEARNING_AT_5000, not a general impossibility verdict.
No best-epoch selection or changes to mass/L1/overlap criteria.

## Implementation

Reused model/runner/tests/Slurm script; configuration
`config/cf4_member_single5000_v3.json`. Existing architecture, losses, learning
rate, clip, source data and precision unchanged. No true member inputs at new
field inference. Native fixture0 alone receives optimizer updates. Other
fixtures are used only by the existing source/baseline/availability preparation;
no new-field evaluation or13-field learning is claimed here.

Six focused/reused tests run inside the same allocation (previous5 plus full
optimizer-continuation regression). Restore check also runs on the actual
source before training. Source snapshot at300; small post-update reports at
1000/2000/3000/4000; final5000 maps/checkpoint/metrics. Pre-update history and
post-update snapshot values must not be confused. Intermediate reports do not
stop or pass the experiment. New history has4700 records, with cumulative step
numbers301..5000; source300 history is retained separately.

Resource request unchanged:1GPU,2CPUs,6GiB host (conservative peak5GiB+20%),
90min Slurm/70min learning cap. Expected45–60min on similar hardware, no queued
time included. New output:
`/gpfs/kjhan/CF4/z0_density/bundle_c_v1/member_mass_single5000_v3/`.
Original source `member_mass_repair_v2/checkpoint_final.pt` stays untouched.

Static py_compile/bash-n/diff checks pass. Numerical tests and execution are
pending submission; not a successful run or scientific result yet.

## Submission

Implementation787497f committed/pushed. Slurm **342086**, cf4_C_member_5000,
submitted2026-09-11 23:09:46 KST and started23:09:47 on **syn05** through the
scheduler. Allocated1GPU/2CPUs/6GiB,90min limit. Initial RUNNING at6s; tests
and actual checkpoint restoration must be confirmed from subsequent outputs.
Logs `/gpfs/kjhan/CF4/logs/cf4_C_member_5000_342086.{out,err}`.
Source budget300 remains completed; target5000 cumulative,4700 new updates.
No second science job or polling daemon is launched.

## Completed342086

COMPLETED/exit0,2026-09-11 23:48:12 KST after38m25s. Six tests pass; source300
state/metrics restored,4700 new updates,5000 cumulative, no multi-field fit or
development evaluation. PASS_SINGLE_FIELD_LEARNING_ONLY, all endpoint criteria.
MW/M31/M33 mass ratios1.008794/1.006867/1.016018, overlaps.988665/.998279/1,
map-L1 .031465/.010308/.016018. Final loss.0077098 versus.468869 at300.
Conservation2.05e-8, host/GPU peaks3.336/2.244GiB. One-field fitting may be
memorization; no observed LG or posterior. User now approves13-field learning
and3-field evaluation; next frozen plan/run `BUNDLE_C_MEMBER_MULTI_{PLAN,RUN}.md`.
