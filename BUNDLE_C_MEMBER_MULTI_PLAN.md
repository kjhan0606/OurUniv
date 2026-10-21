# Member readout: thirteen-field continuation and fixed development evaluation

2026-09-12. User approves proceeding after the5000-step single-field result.
ONE continuation of that model, followed by evaluation; no new model family,
simulation, actual CF4 inference or automatic next science bundle.

## Evidence and purpose

342086 COMPLETED38m25s,6 tests passed,5000 cumulative updates on fixture0.
Final MW/M31/M33 mass ratios1.00879/1.00687/1.01602, overlaps.98866/.99828/1,
map-L1 .03147/.01031/.01602. PASS_SINGLE_FIELD_LEARNING_ONLY. This demonstrates
fitting a native decomposition, potentially memorization, not new-field
identification or an actual LG posterior. Original300-step failure is retained.

Q-GOAL: now test field-to-member prediction across multiple native fields and
on fixed spatially disjoint development fields, needed to assess whether this
readout can inform later LG observation conditioning. Final science route
remains CF4+galaxy+LG -> present density/velocity posterior -> LCDM IC -> forward
LG zoom. LG<=.3/environment1–2 cMpc/h. This pilot still does NOT deliver that
posterior, member COM velocities or inverse member-to-field conditioning.

## Fixed fit and information boundary

- Restore source `member_mass_single5000_v3/checkpoint_final.pt` with full
  AdamW state at5000, source787497f/job342086, status PASS_SINGLE_FIELD_LEARNING_ONLY.
  Hard-abort on source/state mismatch; reproduce native fixture0 endpoint
  metrics before any new optimizer step. No fresh-optimizer fallback.
- SAME530804-parameter U-Net/flat allocation, full128^3/.1875 support, ten
  total-moment/coarse-observer features, mass/shape loss, AdamW1e-4/.01 decay,
  gradient clip10 and strict FP32. Native role labels remain fractional targets.
- Add13000 updates (cumulative5001..18000), exactly1000 complete shuffled
  passes over the13 training fields [0,2,3,4,5,7,8,9,10,11,12,14,15]. One field
  per update; equal exposure. Each gets one of48 signed rotations/reflections
  with aligned physical velocity/dispersion/observer channels. Fresh declared
  augmentation seed912013; no old stochastic sequence claimed. Fixture0's
  prior5000 exposures are explicit. No LR sweep/schedule change/new seeds.
-13000 is a bounded engineering choice for shared-model learning across13
  fields, not proof1000 exposures per field is adequate. User has prioritized
  a larger useful fit over repeated tiny budget extensions. No pilot-stop at
  300/1000 updates or fixed-input re-screen during this multi-field fit.
- Native development [16,17,20] never enter initialization/gradient/LR/best-
  epoch selection. They were previously used by project development and are
  NOT an independent final test set. Training fields can overlap/shared-ID;
  continue recording that, do not count augmentation as independent universes.
  Reuse geometry exclusion and all16 positive-target availability assertions.
- At NEW-field inference accept total moments and coarse observer only, never
  native member IDs/masses/positions. MW/M31/M33 hypotheses all stay explicit;
  shared-cell/unresolved M33 has no waived criterion or separate-peak rule.

## Evaluation and claims

Reused endpoint evaluation ONLY: canonical13 training fields plus3 development
fields, old TRAINING spatial-mean-fraction baseline, maps and mass/L1/overlap/
centroid metrics. Report training aggregate both with/without pretrained
fixture0, using the same final13 rows, no new inference run. Pretraining alone
must not dominate the claim. Do not choose an intermediate best checkpoint.

Unchanged v2 final acceptance: mean training L1 each of4 roles<=.8 baseline;
development M33 each improves baseline AND <.8*min(baseline_L1,1); MW/M31
median L1 no worse than baseline AND median overlap>=.5. Report zero-mass
null L1=1 and all member mass/centroid errors. Conservation<=1e-6/nonnegative.
All pass -> PASS_MASS_MAP_FEASIBILITY_ONLY; otherwise NO_GO_MEMBER_MASS_READOUT
for this candidate. Short budget -> INCONCLUSIVE_BUDGET, runtime error ->
INCOMPLETE_EXECUTION, never scientific failure from an incomplete evaluation.

Mean-field OOD ablation is secondary input-sensitivity evidence only, not
an observational influence/calibration test. Deterministic mean maps do not
represent assignment multimodality. Missing joint F,S law, COM/stellar offsets,
selection p(E|F,O), global/fine priors and actual observations must eventually
let LG constraints change the SAME total field, not just its role labels.
Neither outcome here supplies that missing link. Close the fit and report;
no automatic additional training, diffusion repair, observed LG or IC job.

## Q-LEAN and resources

Reuse runner/model/loss/six existing tests, maps and baseline. Small changes:
permit approved source PASS status, carry restored screen success, freeze
equal-exposure augmentation sequence and label cumulative/new updates. One
focused schedule regression, no new gate/audit framework or diagnostic series.
Per-update history plus ordinary progress logs; final checkpoint/evaluation.

One Slurm GPU/2CPUs/6GiB host (conservative peak5GiB+20%),3h allocation,
a40,a100,h100,h200 excluding syn06.13000*.48s~104min training, estimate~2h
with overhead, not guaranteed ETA. Learning150min, application175min, allocation
180min; incomplete budget classified explicitly. Existing peak3.336GiB host/
2.244GiB GPU, no larger network/cache; retained20GiB GPU envelope. New output
`member_mass_multifield_v4`; all earlier runs remain preserved. Syntax uses
Slurm for numerical tests/learning. No filesystem/process-scan diagnostics.
