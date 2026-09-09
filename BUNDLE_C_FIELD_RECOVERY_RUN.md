# C field recovery — approved one-job execution

2026-09-10. User approved implementation and one bounded Slurm comparison after
Fable5 plan corrections. Plan BUNDLE_C_FIELD_RECOVERY_PLAN.md; audit disposition
BUNDLE_C_FIELD_RECOVERY_AUDIT.md. No extra science/audit bundle is opened.

## Implementation and frozen details

- `src/cf4_flow_energy.py`: two independent full draws, detached CPU traces,
  all ancestor/node/mask/continuous log-probability scores, one node graph at
  a time. Original FP64 conservative decoder and strict-FP32 network unchanged.
- `scripts/cf4_bundle_c_field_recovery.py`: original v2 preparation, checkpoint
  and isolated optimizer copies; bounded8-pair screen, matched1000-step branches,
  original16 development draws, fixed one-step/rollout comparisons and maps.
- Config `config/cf4_field_recovery_v1.json` freezes seeds/modes/resource caps.
  Data order is freshly and identically seeded for both branches; this is not
  exact restoration of v2's unsaved RNG state. Original scale exposure and
  full-context record shape are retained. No model architecture changes.
- Structural normalization uses all66 unordered native-training cube pairs,
  independently for each of4 score modes and3 fixed feature groups. Eight
  8-cell blocks at n/4,3n/4; original log-power and velocity/dispersion metrics.
  Zero-scale groups explicitly unavailable. No heldout feature/loss tuning.
- One lambda from the planned pilot gradient-norm ratio; original clip10 on
  combined gradients. Baseline is a per-mode EMA(decay.95), initialized0 and
  updated ONLY after the current gradient. Full rollout every8th repair step.
- Both screen halves score .75/.375/.1875 transitions and a full rollout,
  with7/24,7/24,7/24,1/8 weights. The full graph sequence must fit20 GiB GPU
  estimated peak before training. Host request24 GiB includes20% headroom.
- Unchanged original all16-draw morphology rule is required for development
  adoption. ES superiority additionally requires lower retained .1875 high-band
  log-ratio RMS in BOTH teacher and rollout comparisons (same fixed2 cases x2
  draws as337496). All scales/cases are reported, not tuned or hidden. This is
  a small reused-development comparison, not a formal superiority test.
- Existing native roundtrip, trained inverse/Jacobian and precision checks are
  reused. Added tests: mixed atom/Beta-law integrated ES gradient with both
  parameter directions and fixed baseline; complete14-node two-scale gradient
  against monolithic differentiation plus conservation; fixed feature scaling.
  All numerical tests execute inside the same Slurm allocation, not on syntax.

## Submission and limits

Runner `scripts/run_cf4_bundle_c_field_recovery.sbatch`: one GPU,2 CPUs,
24 GiB host memory,4h wall cap; partitions a40,a100,h100,h200, exclude syn06.
Preparation/screen/learning cap3h; remaining1h reserved for evaluation. No
claim that the new high-dimensional score estimator will finish or learn.
Source commit and job ID will be recorded after submission. The runner refuses
changes against the submitted source and creates a new output directory.

Submitted as Slurm **338194**, pushed source **43692a1**. Initial scheduler
check: PENDING(Resources); no node assigned, tests/training not started and
no run logs yet. Queue delay is outside the4h allocation cap. This is a
submission report, not a numerical-test pass or completed comparison.

Output `/gpfs/kjhan/CF4/z0_density/bundle_c_v1/field_recovery_v1`;
logs `/gpfs/kjhan/CF4/logs/cf4_C_field_recovery_JOBID.{out,err}`.
Two final checkpoints, original16 full development fields per branch and small
tables/projections; target<6 GiB. No raw data pass/new simulation/IC launch.

Expected screen failures terminate without training or a new seed/weight. A
screen pass means only no gross pathology detected, not usable signal-to-noise.
Nonfinite/OOM/time stops do not permit a scientific winner. Successful execution
still requires all scientific criteria; no automatic next bundle on completion.

No actual CF4/LG weighting or member assignment occurs here. M33 can remain
unresolved/shared-cell; same-field member/selection and global environment laws
are still missing. This trial cannot deliver an observed high-resolution LG
posterior or authorize the auditor's prior-fine-IC fallback.

## Completed learning, evaluation dtype correction

338194 ran on syn08/H100 NVL,2026-09-10 00:50:14--02:04:45 KST (1h14m31s).
All7 tests passed; the bounded screen passed its operational checks. BOTH
branches finished1000 updates and saved final checkpoints. Evaluation then
failed before morphology measurement: `trained_inverse` passed the FP64 output
of `encode_tree` through the training-record adapter, which preserved its
dtype. The FP32 convolution rejected the resulting double input. This is a
driver implementation defect, not a numerical inverse failure, OOM, failed
learning criterion or evidence against the scientific model.

User requests fixing the evaluation error. The fix reuses `device_record`,
which converts network z/context to FP32, masks to int64 and validity to bool;
native physical moments/decoder stay FP64. One regression calls the actual
evaluation function on FP64 native input with nonidentity FP32 layers and
checks finite passing inverse, input dtype, no native mutation or gradients.
Eight tests run in the evaluation allocation. Original thresholds unchanged.

`--evaluate-only` reads the two completed checkpoints without creating an
optimizer, preparing training records, recomputing the screen or taking any
optimizer updates. New outputs go to
`/gpfs/kjhan/CF4/z0_density/bundle_c_v1/field_recovery_evaluation_v1`.
The failed result/logs and original checkpoints remain untouched. Original
16-draw evaluation and paired teacher/rollout metrics are reused for both
branches, followed by the same frozen terminal decision. No new trial/audit.

Runner `scripts/run_cf4_bundle_c_field_evaluation.sbatch`: one GPU,2 CPUs,
9 GiB host memory,1h cap (application3500s), same partitions/exclusion. Prior
whole-run sampled MaxRSS7078212 KiB (~6.75 GiB), plus20% gives~8.10 GiB;9 GiB
rounded request is conservative for this evaluation-only job, which retains
only2 training cubes and no prepared training dataset. Peak GPU allocation
in338194 was2621341184 bytes (~2.44 GiB). No new memory/storage probe.
Submission and actual test/evaluation outcomes are recorded separately below.

First correction submitted as338388, source62d1d0f. Slurm assigned syn101
(scheduled allocation, NOT manual execution). It stopped in the new test's
cleanup: unittest.mock.patch tried deleting a PyTorch backend descriptor that
only supports get/set. No scientific evaluation/output directory was created.
Use the already-tested explicit save/configure/restore pattern instead. This
is a test-harness defect; preserve338388 logs and retry the same evaluation
without retraining, criteria changes or a new scientific experiment.

## Evaluation completed — code corrected, scientific criterion still fails

Corrected source18b84f8, Slurm **338389**, syn101 A100-SXM4-80GB through Slurm
(not manual). COMPLETED exit0,2026-09-10 08:14:20--08:19:14 KST,4m54s. All
**8 tests passed**, including the actual FP64-native/FP32-model regression.
Both saved models pass the unchanged inverse/Jacobian checks:
control1.53e-6/2.86e-6, repair1.56e-6/3.34e-6 (limits1e-4/1e-3).
Additional training updates **0**. Original checkpoints and failed-run outputs
remain preserved. Sampled host MaxRSS2187624 KiB; GPU peak294758912 bytes.

Retained .1875 diagnostics, identical2 cases x2 draws in each mode:

| Quantity | NLL control | NLL + structural ES |
| --- | ---: | ---: |
| True-parent high-band P/native mean | 0.654972 | 0.664406 |
| Full-rollout high-band P/native mean | 0.436332 | 0.462160 |
| True-parent high-band log-ratio RMS | 0.550199 | 0.521605 |
| Full-rollout high-band log-ratio RMS | 0.873808 | 0.801201 |
| Original development draws passing ALL criteria | 0/16 | 0/16 |

The ES branch modestly improves these one-step/rollout summaries, but both
branches fail the unchanged development criteria. Decision:
**CLOSE_THIS_REPAIR_LINE_BOTH_FAIL**. This is scientific non-adoption, NOT an
evaluation software failure. The original evaluator now completed successfully.
Close this bounded objective/current-architecture repair; no extra steps,
coefficients, seeds, member diagnostic, alternative prior or IC run submitted.
No actual CF4/LG spatial posterior is delivered; this does not prove that all
learned priors or the final scientific goal are impossible.

Full results, all scales/cases and comparison image:
`field_recovery_evaluation_v1/result.json`, per-branch `*_development.json`,
`*_paired_generation.json` and `density_comparison.png`, below the output root
given above. Next bundle remains subject to a new plan and user approval.
