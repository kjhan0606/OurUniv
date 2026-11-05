# Frozen-checkpoint cause separation, no additional field training

2026-09-10. User said 'next proceed' after receiving the completed diffusion
failure and recommendation to distinguish implementation from learning failure.
Scope: driver code inspection plus ONE short Slurm frozen-checkpoint diagnostic;
no optimizer update, new model, replacement fit or automatic scientific follow-up.
Fable5 plan audit before numerical execution; accepted conditions incorporated
in the same diagnostic, not a new audit ladder.

Evidence:338402 completed30,000 updates and8 regression tests, but all8 retained
fine draws are invalid. Every rollout fails at the first1.5->.75 refinement;
native-parent checks at all three scales fail too. Actual failure is in the
existing split-fraction decoder: continuous fractions round to zero in FP64.
This is NOT8 measured bad-morphology fields. The automatic NO_GO_MORPHOLOGY label
contains generation/support failures. Final300-update continuous loss1.000238,
categorical.002355. No current model promotion or actual observed LG product.

Questions and fixed experiment:

1. On the first historical training cube[13,8,10], read only existing native
   moments. Compare raw and EMA30k checkpoints at native-parent .75 and.1875
   with fixed timesteps[1,25,50,75,100], one fixed noise/mask realization each.
   Reuse training normalization and encoding. Report input ranges, epsilon
   prediction RMS/correlation/MSE, zero-predictor MSE and the elementary
   noisy/sqrt(1-alpha_bar) predictor. Near t100 the latter should already be
   informative without a learned cosmological prior. This is an algebraic
   reference, NOT a proposed replacement generator or tuned model.
2. Compare final weights to initialization at the exact original seed and
   optimizer step counts. Report stem, early/late spatial blocks, output
   noise/category heads, final norm, and raw-versus-EMA differences. Compute
   continuous-only and categorical-only gradients on the .75/t50 fixed case,
   with NO step. A deterministic checkpointed/noncheckpointed backward
   comparison on that same small parent lattice detects gradient plumbing
   defects. No full-scale training benchmark or learning-curve extension.
3. Trace ONE original first failed heldout case[33,0,0], seed99803, .75 scale,
   using the unchanged EMA sampler. At t100/75/50/25/1 report noisy/prediction
   ranges. Inspect final raw versus legal active fraction coordinates and the
   first failing binary node; quantify saturation rather than clipping/flooring
   it away. Existing outputs/checkpoints untouched, diagnostics in a new folder.

Do not infer a unique cause merely from loss~1. Distinguish a frozen/incorrect
training path, learned noise-input insensitivity, normalization problems, and
reverse-chain amplification. Initialization-comparison and input gradients are
diagnostic, not proof that one architectural change would learn cosmology.
No required model adoption thresholds are weakened. No neural parameter update.

Q-GOAL: removes an immediate software/learning uncertainty before interpreting
this failed present-field prior; does not solve the CF4->z0->IC/zoom chain.
MW/M31/M33 are still unassigned on new fields; no native identity seeds, known
member masks, same-field q_S or LG likelihood are added. Shared-cell M33 remains
possible. Selection and environment inference remain missing. Prior audit's
ordering is retained: member learnability must be addressed BEFORE any more
field-model training; this frozen diagnostic is not another training attempt.

Q-LEAN: one existing checkpoint, two fixed scale cases plus one existing failed
seed, one compact report, reuse model/encoder/sampler and tests. No new generic
instrumentation/monitoring, filesystem or process-scan tests, simulations or raw
snapshot passes.1 GPU,2 CPUs,6GiB host RAM (engineering peak5GiB incl.524MiB
checkpoint/native arrays/model copies+20%),20min cap. Observed preceding host
peak4.14GiB and GPU reservation3.62GiB provide scale references, not guaranteed
new peaks; no whole training slab or optimizer is resident on GPU here.
Keep budget bounded and save partial diagnostics on error. No automatic rerun
or subsequent training. Commit/push results and report the next justified action.
