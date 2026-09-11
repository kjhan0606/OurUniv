# Member-mass repair v2: one approved bounded fit

2026-09-11 user approves the final recommendation in
`MEMBER_REPAIR_COMPARISON.md`. Frozen submitted plan:
`BUNDLE_C_MEMBER_REPAIR_PLAN.md`. Fable response:
`config/cf4_member_repair_plan_fable5_v1.response.json` (normal126851ms,
main model claude-fable-5; auxiliary Haiku routing, not the plan auditor).

## Plan decision and disposition

Fable CONDITIONAL GO. Q-GOAL/Q-LEAN pass. One blocking condition: assert all
four integrated native target masses strictly positive in ALL13 training and
ALL3 development fixtures BEFORE fitting. Implemented in the reused prepare()
using the existing load_case() validation; save target_availability.json. The
three development fixtures are discarded after availability checking; their
values never enter initialization, optimizer, thresholds or model selection.
No additional audit micro-stage or fallback auditor is needed for this check.

Driver correction to the audit: v1's spatial baseline was only an evaluation
comparator, NOT a training initializer/input, so it cannot be blamed as a
training cause. Retain the plan's uncertainty about the cause. Memory and ETA
remain estimates, not Fable-certified performance. Positive remainder by itself
is not scientific evidence of recovery. No M33 waiver or new-field posterior
claim follows from a screen or final pass.

## Implementation and fixed scope

- Reuse `src/cf4_member_mass_readout.py`,
  `scripts/cf4_bundle_c_member_mass.py`, existing tests/Slurm runner.
- `config/cf4_member_mass_repair_v2.json`: new output member_mass_repair_v2.
  Existing v1 config, weights/checkpoints/data remain untouched.
- Flat four-way U-Net530804 parameters. Positive TRAINING integrated-fraction
  head initialization, stable log-mass-squared plus spatial-KL loss.
- One300-step fixture0 segment. Pass only when all three member mass ratios
  [.8,1.2], overlaps>=.8, L1<=.3 and conservation<=1e-6/positive remainder.
  Save step0/300 metrics, screen checkpoint and native/predicted/baseline maps.
  Stop with INCONCLUSIVE_SINGLE_FIELD_LEARNING on failure. No development
  score can rescue a failed screen;300 steps do not prove impossibility.
- On pass SAME fit/optimizer continues1700 updates with13 fields and48 signed
  symmetries. Original final criteria plus stricter M33 versus baseline/zero
  and host median overlap>=.5. All native roles explicit; no truth at inference.
- Five focused/reused numerical regressions inside the Slurm job: existing
  allocation/backward and signed observer features, new mass/shape sparse/empty
  support and initialization/subsequent-backbone-learning tests, reused coarse
  restriction symmetry test. Static py_compile/bash-n/diff check locally only.
- One GPU/2CPU/6GiB host/90min, expected host peak5GiB+20%; same20GiB GPU sizing
  with existing2.20GiB measured v1 peak, not a new memory benchmark. No more
  than2000 updates or70min learning; no automatic retries/follow-up bundle.

This is native simulated component mass-map feasibility, not actual CF4/LG
inference, member COM velocities, a joint member/field posterior or fine ICs.
The member-to-SAME-field observational conditioning direction remains absent.

## Execution

Implementation complete; py_compile, bash -n and git diff --check pass.
Numerical regressions have NOT run locally and no scientific result exists
yet. Commit and submit once with source pin; actual job state is recorded below.

Implementation `8dbaeae` committed/pushed. Slurm **342013**, job name
cf4_C_member_repair, started2026-09-11 18:26:28 KST on **syn05** (scheduler
allocation, NOT manual execution). Initial RUNNING at8s; allocated1GPU,
2CPUs and6GiB,90min. No claim of numerical pass from this startup state.

Outputs `/gpfs/kjhan/CF4/z0_density/bundle_c_v1/member_mass_repair_v2/`;
logs `/gpfs/kjhan/CF4/logs/cf4_C_member_repair_342013.{out,err}`.
One job contains the tests, target-availability assertion,300-step screen and
conditional same-fit continuation. No monitoring/process-scan daemon or
additional science job is launched. Next bundle still requires approval.
