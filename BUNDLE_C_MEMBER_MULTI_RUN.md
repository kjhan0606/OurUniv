# Thirteen-field member continuation v4

2026-09-12 user approves next learning/evaluation bundle. Submitted plan
`BUNDLE_C_MEMBER_MULTI_PLAN.md` is frozen; this disposition specifies the final
audited implementation. One model continuation, not an actual CF4/LG product.

## Fable plan decision

Normal98118ms, main claude-fable-5, auxiliary Haiku routing, CONDITIONAL GO.
Exact request/response `config/cf4_member_multi_fable5_v1.*`. No fallback audit.
Q-GOAL/Q-LEAN accept a bounded multiple-field mass-readout attempt; no joint
posterior, observed-field or universal identifiability claim is authorized.

- E1 incorporated: PRIMARY training gate uses the12 fields excluding pretrained
  fixture0. Require each role's mean L1<=.8 of the corresponding baseline mean.
  The original13-field mean/decision remain separately reported, not used to
  rescue a failure. Development criteria stay unchanged. This checks learning
  across additional training fields, not generalization by itself.
- E2 resolved from EXISTING accounting, not a new memory experiment. Both the
  recent single-field jobs AND v1 prepared all13 native unaugmented cases in
  CPU memory:13*15*128^3*4 bytes=1.5234375GiB (10 features,1 total,4 targets).
  Augmentation is applied to ONE CUDA case per update; no13-fold augmented
  GPU cache. Existing v1 multi-field/48-symmetry run341713 measured host3.99GiB,
  and342086 with the same13-case cache measured3.336GiB. Retain conservative
  host peak5GiB+20%=6GiB. Fable's suggestion that another1–1.5GiB cache is
  newly added to the3.336GiB measurement is incorrect; it was already included.
- E3 already covered: MemberMassTests.test_features_and_observer_symmetries
  checks features(augment(native,i)) against the actual transform() for ALL48
  cases, including signed velocity/observer and permuted directional sigma.
  The reused coarse-restriction symmetry test also runs. This path existed in
  v1, not a new untested augmentation operator. No redundant transform test.
- Per-development M33 shared-cell context is added from arrays already loaded:
  native occupied-cell count, fraction of M33 mass in M31-occupied cells and
  whether native peak cells coincide. These disclose grid co-occupancy; none
  waives M33 criteria or proves physical resolution/identifiability.

The audit's assertion that this exact readout is necessary before *any*
observation-conditioning design is too strong: it is the selected project
route being tested, not a universal mathematical prerequisite. Median host
criteria tolerate one bad development field, so all three rows/maps are kept.

## Implementation and numerical contract

`config/cf4_member_multifield_v4.json`; reuse existing model/runner/Slurm script.
Restore full5000-step model+Adam from342086/source787497f, hard-abort if source
PASS/state/fixture0 metrics are not reproduced. Train cumulative5001..18000:
13000 NEW updates, exactly1000 per each13 fields, shuffled cycles/48 symmetries
with fresh declared augmentation seed912013. Fixture0 also has5000 prior
updates. No new initialization/LR schedule/model/loss and no best-epoch choice.

Seven tests inside the allocation: previous6 and a deterministic equal-cycle/
symmetry-range/reproducibility schedule regression. Full native restore checked
before learning. Source5000 snapshot, per-update history,250-update progress
records, final checkpoint and one13-training/3-development evaluation. Tests
and numerical learning do not run on the syntax login node.

Final fixed criteria: training12 each-role mean L1<=.8 baseline; each retained
M33 L1<baseline AND <.8*min(baseline,1); each host median L1<=baseline and median
overlap>=.5; conservation<=1e-6/nonnegative. All rows include mass/centroid,
overlap and zero-mass null. PASS_MASS_MAP_FEASIBILITY_ONLY or NO_GO for this
candidate; incomplete budgets/runtime labelled separately. Reused development
fields are not independent validation. No new actual observations enter this
fit. Mean-field ablation is only OOD sensitivity, not observation influence.

One Slurm GPU/2CPU/6GiB/3h, partitions a40,a100,h100,h200 excluding syn06.
Learning150min/application175min;~2h expected, augmentation/node variation may
cause budget shortfall. New output member_mass_multifield_v4, preserve all old
outputs. No automatic follow-up density-prior repair, further learning or ICs.
The missing joint member/field law, velocities/stellar offsets, selection and
actual LG-to-SAME-field conditioning remain deferred.

Static py_compile/bash-n/diff checks pass. Numerical tests run only in the
allocation; no numerical pass/result or submission is claimed yet.

## Submission

Implementation20253e1 committed/pushed. Slurm **342129**, cf4_C_member_multi,
submitted2026-09-12 00:30:11 KST. Initial **PENDING(Resources)**, no node or
numerical test/learning result yet. Request1GPU/2CPU/6GiB,3h as planned.
Logs `/gpfs/kjhan/CF4/logs/cf4_C_member_multi_342129.{out,err}`;
outputs `/gpfs/kjhan/CF4/z0_density/bundle_c_v1/member_mass_multifield_v4/`.
Source-pinned allocation will perform tests, restore5000 and run13000 new
updates plus endpoint evaluation. No manual run or polling daemon launched.

## Completed result and subsequent inspection

342129 COMPLETED/exit0 on syn05,2026-09-12 00:30:28–02:15:21 KST,1h44m53s.
Seven tests pass; full5000-step restoration reproduced the source metrics.
13000 new updates completed, exactly1000 per each13 fields,18000 cumulative.
Host/GPU reserved peaks4.064/2.150GiB. NO_GO_MEMBER_MASS_READOUT: primary12-
training mean L1 MW/M31/M33=.994/.878/1.630; all three role criteria fail.
Development16/17/20 M33 L1=1.063/1.155/1.858; all fail. Pretrained fixture0
now2.816/.957/1.066 versus.0315/.0103/.0160 before continuation.

Read-only follow-up found98.2% of624 identical field/symmetry combinations
improve first-to-last recorded loss (means4.714->1.733); these are different
training times, not a controlled causal experiment. Quality failure is not
absence of learning. Development20 M33 mass ratio.9474/overlap.04453 demonstrates
spatial misallocation, not only normalization. Random native M31/M33 selection
is absent from field-only inputs; a population-level assignment ambiguity is
not proof of duplicate-input conflicts in the fixed13-field training set.
Source labels, augmentation and finite optimization effects remain distinct.
User approves the bounded frozen-model diagnosis/redesign described in
`BUNDLE_C_MEMBER_DIAGNOSIS_PLAN.md`. No additional training authorized.
