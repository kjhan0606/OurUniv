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
