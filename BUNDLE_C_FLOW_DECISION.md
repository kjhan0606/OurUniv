# C frozen-flow decision bundle

User approved questions1–4 jointly, followed by driver decisions5–6,2026-09-09.
No new training, architecture, amplitude tuning, raw snapshot pass or actual
CF4/LG inference. This is one bounded analysis job using saved v1/v2 models,
existing metrics and cached native fields, not another validation framework.

## Starting evidence

337279 completed10:43:18–10:52:48 KST on A10080GB,9m30s. Four regressions and
the strict-FP32 checkpoint/context gate pass. All6000 corrected training steps
and16 evaluated draws completed. All16 still fail overall development gates;
fine upper-four-band power ratios0.235–0.593 versus native. TF32 inversion
failure is fixed but morphology is not. No actual LG map or production release.

## One calculation covering questions1–4

- Reuse original32 evaluation rows (16 per model) and saved draw0 maps for
  before/after comparison. Fixed two heldout cubes, same native normalization,
  projection depth1.5 and color scale; no outcome-selected nice examples.
- Use first TWO original training cubes and BOTH heldout cubes. Preserve the
  within-fit disjoint train/test split, but two train cubes may overlap and
  all are historically reused one-box development, not independent validation.
- Evaluate fixed native conditional likelihood at all six physical scales,
  all seven binary nodes, for both final checkpoints. Sum per-node mean NLL
  per octet; record categorical and continuous contributions. Both models use
  strict FP32/full context for controlled comparison. Units are split-coordinate
  density, not physical-field Lebesgue density. Do not compare unnormalized
  total NLL across voxel counts or claim a convergence curve from two final
  models trained differently.
- For each model/cube use TWO fixed draws. Fine1.5→.75→.375→.1875: at every
  stage compare an ancestral rollout to a one-step refinement of the true
  native parent. Save/restore CUDA RNG so the two paths use identical random
  state at that step. First step must agree exactly. Both are scored against
  the SAME original1.5 parent at a given stage, making velocity/boundary
  definitions comparable. Compare power, connectivity, velocity and physical
  dispersion using existing metrics; never treat field phase differences from
  one native truth as proof of conditional distribution failure.
- Also compare final environment12→1.5 rollout with native3→1.5 one-step
  generation. Do not compute the existing eight-band spectrum at intermediate
  N4/N8 (empty bands). All six scales still receive native NLL evaluation.
- Plot fixed draw0 maps and scale-dependent power error alongside native NLL.
  High-band RMS(log Pgenerated/Pnative) is a descriptive error, NOT a new
  selection gate. Original acceptance rules remain unchanged. Two stochastic
  draws and two training cases do not supply calibrated population errors.

Frozen seeds:99803+100*case+draw; training adds10000, environment adds1000.
No optimizer, no gradient updates, no additional saved full3D fields. Outputs
`flow_decision_v1`: request/status/result JSON, density_slabs.h5 and three PNGs.
Estimate<20 MiB artifacts, host8000 MiB+20%=9600 MiB requested, one GPU/two
CPUs,1h cap. Cached native source only; numerical work is on Slurm. Existing
four regression tests plus inline pairing/conservation/finite-score checks.

## Driver interpretation after completion: questions5–6

1. Improvement: report each metric separately, original acceptance results and
   strict-precision controlled comparison. Context, scored-cell exposure and
   training precision changed together; cannot attribute improvement uniquely.
2. If both train and heldout are poor, a heldout-only generalization failure
   is not sufficient explanation. This does NOT isolate optimization from
   model capacity. Native field distributions differ; an NLL gap alone cannot
   establish overfitting. No intermediate validation checkpoints are available.
3. Teacher-parent failure identifies a problem already within one refinement;
   substantially worse rollouts indicate additional accumulation/context shift.
   Both may occur. Do not prescribe rollout-training merely because steps exist.
4. Compare paired v1→v2 native NLL and structure changes at the same case/scale.
   Lower NLL with worse structure supports a practical objective/quality
   mismatch for this fit, not a proof that likelihood learning cannot work.
5. Decide keep/one justified repair/close the present candidate. Do not launch
   any such next fit without user approval. Missing causal evidence must be
   stated instead of automatically lengthening training or switching families.
6. Tie the decision to actual CF4/LG: q_S member/field observational readout,
   normalized global environment law and native CF4/count/LG joint conditioning
   remain absent. Existing mark posterior cannot be treated as an independent
   extra likelihood; fixed native halos cannot be attached to changed fields.
   If these cannot be supplied, prettier TNG draws do not deliver the goal.

Completion produces evidence for a driver report, not an automatic scientific
GO or downstream fit. Next bundle remains subject to user approval.

Submitted337496, source4a21709 committed/pushed. Logs:
`/gpfs/kjhan/CF4/logs/cf4_C_flow_decision_337496.{out,err}`. Evidence directory:
`/gpfs/kjhan/CF4/z0_density/bundle_c_v1/flow_decision_v1`.
Read result.json and inspect all three figures before final driver judgment.

## Completed evidence and driver decision2026-09-09

337496 completed16:35:09–16:37:52 KST,2m43s, A10080GB; four regressions
passed. All128 diagnostic generation comparisons and48 fixed native likelihood
records completed; three figures inspected. No optimizer/fit was run.

At .1875, v2 heldout high-band mean power/native is0.650 with the true .375
parent, versus0.390 after1.5→.1875 rollout. Training counterparts are0.676 and
0.297. Means cover two cubes/two draws/four upper bands, not confidence limits.
The heldout v2 rollout declines0.766→0.571→0.390 at .75/.375/.1875; teacher
parent values0.766→0.729→0.650. Within-step underproduction and additional
rollout deterioration both occur; heldout-only overfitting is not enough to
explain the observed training failures.

At .1875, mean native NLL improves39.912→39.222 heldout and40.763→39.295
training. However training rollout high-band log-error worsens1.184→1.230;
heldout improves1.051→0.985 only modestly. Different models/contexts/precision
and only final checkpoints do not establish convergence, a unique causal
defect, or impossibility of likelihood learning. Native-parent visual success
partly inherits the supplied fine .375 skeleton; it is not recovered LG detail.

DECISION: retain v1/v2 as failed development baselines; actual-field adoption
ON HOLD. User accepted this hold and approved continuing the redesign.
No autonomous longer fit, amplitude patch or diffusion-family switch. The next
design must address one-step and rollout errors AND supply an honest member/
field LG observation interface. See BUNDLE_C_STRUCTURE_LG_REDESIGN.md. This
analysis bundle is closed; C and the actual high-resolution posterior are not.
