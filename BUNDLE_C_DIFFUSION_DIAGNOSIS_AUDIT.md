# Frozen-checkpoint diagnosis: accepted Fable5 plan audit

2026-09-10. Submitted plan is preserved unchanged in
`BUNDLE_C_DIFFUSION_DIAGNOSIS.md`; request and full response are
`config/cf4_diffusion_diagnosis_fable5_v1.{txt,response.json}`.
Fable5 returned CONDITIONAL GO in58.6s, with Q-GOAL and Q-LEAN valid.
This zero-update post-mortem does not bypass member learnability before any
further field training. One20min Slurm allocation only; no automatic training.

Conditions incorporated in `scripts/cf4_diffusion_frozen_diagnosis.py`:

1. Historical training saves checkpoints only every10k steps and did not
   record an initialization hash. Thus a reconstructed seed initialization
   cannot be independently verified. OMIT the initial-versus-final comparison
   from scientific evidence; report UNAVAILABLE, not a verified update.
   Retain actual raw/EMA differences, saved optimizer counters and frozen
   gradients. This obeys 'validate initialization BEFORE trusting Check2'
   without inventing missing evidence or another run.
2. Assert exact location/spread equality between original training
   `representation.json` and checkpoint. Reuse original `native_record` for
   input preparation and pass those SAME arrays to the unchanged sampler.
3. Add per-channel empirical-majority categorical baseline, fixed probability
   .99 for that class and .01/3 for others, under the identical masked loss.
   Report majority accuracy and model loss. This in-probe baseline is not an
   independently fitted predictor or a proposed replacement cosmology model.
4. Flag exact-zero continuous-head gradients separately from small gradients;
   report missing gradients, checkpoint/plain tolerance agreement and relative
   differences. Exact zero alone does not prove detachment: it requires graph
   evidence and must not be confused with convergence or a zero activation.
5. Explicitly classify338402's8 retained draws as GENERATION/SUPPORT failures,
   not valid fields with a measured bad power spectrum. Preserve raw outputs.

Flush after weights/normalization, each checkpoint's fixed denoising checks,
and backward comparison; record errors and existing partial diagnostics.
Fraction saturation concerns mass/internal channels0,4,5,6, not the relative
velocity tanh channels. No saturation clipping, latent floor or model patch.

If healthy plumbing but failed denoising is found, this does not uniquely
identify capacity, encoding or optimization as the root cause. Report that
uncertainty. The next scientific bundle still requires a concrete proposal
and user approval; further field training first requires member learnability.

## Submitted, not yet evaluated

Implementation3240750 committed and pushed; static Python compilation,
shell syntax and git whitespace checks pass. Slurm **338746**, submitted
2026-09-10 18:41:37 KST. Initial scheduler state PENDING(Priority), no
allocation or numerical diagnosis yet. ReqTRES1 GPU/2 CPUs/6GiB,20min;
partitions a40,a100,h100,h200, excluded syn06. Backfill at18:41:38 estimates
20:35:50 KST, not a start promise. SchedNodeList syn101 is only a possible
ordinary Slurm allocation; no manual execution. Numerical output will be
`/gpfs/kjhan/CF4/z0_density/bundle_c_v1/diffusion_frozen_diagnosis_v1/result.json`.
No separate monitoring daemon or automatic scientific follow-up was created.

## Completed and interpreted2026-09-11

338746 completed2026-09-10 18:42:44 KST,36s Slurm elapsed,19.699s application,
exit0, zero optimizer updates, host peak2.357GiB/GPU reserved3.061GiB.
Raw and EMA predictions have epsilon RMS about.005 versus target1, almost
zero noise correlation and MSE about1. At t100 the direct algebraic reference
MSE is2.36–2.37e-7. The tested model has failed to learn useful denoising even
on native training input, not merely to extrapolate through a reverse chain.
The unchanged seed99803 chain amplifies latent RMS1.001 ->772 ->1429 ->1871
->2032, then the first mass split has2707/4096 active fractions failing support.

Training-record/checkpoint normalization is identical. No missing parameter
gradients; continuous-head gradient RMS5.35e-4 is nonzero. Plain/checkpointed
continuous relative gradient difference9.90e-10; categorical2.77e-8. Thus no
gradient disconnection/checkpointing mismatch is detected in this fixed case.
Stem continuous gradient RMS4.43e-8 is small, but does not isolate architecture,
loss competition or optimization as the cause. Both raw and EMA fail. Original
step0 weight evidence is unavailable; saved optimizer counters are all30000.
The probe's categorical majority fraction99.923–100% makes low category loss
weak evidence of learning. None of these checks establishes cosmic morphology.

Close this frozen diagnosis. No field generated/promoted and no automatic
training extension. User now requests a modification PROPOSAL; see
`BUNDLE_C_REPAIR_PROPOSAL.md` and its Fable5 audit/disposition when available.
