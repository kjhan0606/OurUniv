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
