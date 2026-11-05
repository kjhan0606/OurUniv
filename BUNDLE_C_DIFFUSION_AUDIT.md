# Spatial diffusion replacement — Fable5 plan audit disposition

2026-09-10. Design/audit completed; no numerical job or new model submitted.

## Provenance and decision

Primary Fable5 CLI completed normally, exit0, is_error=false,121178ms.
modelUsage includes claude-fable-5 plus an auxiliary Haiku entry. No fallback.
The auditor reviewed supplied plan/evidence only; no independent code/data
inspection, numerical testing or literature verification was performed by it.
Preserved request, exact submitted plan, original response and empty stderr:
`config/cf4_diffusion_fable5_plan_audit_v1.*`.

Verdict **CONDITIONAL GO**. Q-GOAL accepts ONE bounded field-prior replacement
as relevant to the absent fine structure. Q-LEAN accepts the reused data,
conservation/evaluation code, changed-code tests and one-job scope. It does
not certify the new learner or the actual CF4/LG posterior connection.

## Five conditions and disposition

1. **Sizing before submission:** save actual model/state counts and estimates
   for checkpointed activations, noisy modalities, FP64 buffers and cache plus
   update/evaluation time. Compare with measured worst-scale in-job32 updates.
   Proposed48GiB host and43.2GiB GPU allowance are estimates, not measurements.
   This is a compact run config, not a monitoring or filesystem framework.
2. **Frozen legal-branch mapping:** commit a complete versioned specification;
   native-encode identity must pass before optimization. Per-scale/channel
   correction frequencies expose training/support mismatch. The auditor's
   wording 'before the training job starts' is implemented before the first
   optimizer update INSIDE the same Slurm job. Numerical tests never run on
   syntax's login node; no separate audit/preflight ladder is introduced.
3. **No discarded failures:** a numerically invalid fixed draw is a morphology
   failure with reason/count; no redraw or success-conditioned prior claim.
4. **Budget-incomplete status:** <30,000 updates at the time cap means
   INCONCLUSIVE_BUDGET, no adoption or family-level impossibility claim.
   Numerical/operational failure remains separately named. All partial metrics
   remain reported, not erased. The audit's 'no evidentiary weight' is read as
   no family-level verdict, not denial that partial trends contain information.
5. **No serial field-model detour:** on failure/inconclusive outcome, address
   native-data member-state learnability/identifiability before another field
   training trial. No automatic extension, new family or new simulation.
   This is ordering for the next approved design, not authorization to launch
   an unspecified q_S probe. Closed peak/proxy/budget variants stay closed.

These conditions are incorporated in BUNDLE_C_DIFFUSION_PLAN.md. No second
audit is needed merely to confirm these accepted implementation requirements.

## Scientific qualifications retained by the driver

The audit's 'inconclusive is the likeliest outcome' is a qualitative forecast,
not a measured probability. Published140 A100-hours are not our convergence
estimate. The proposed24h cap intentionally limits risk and may be insufficient.

A joint discrete/continuous reverse chain and deterministic physical decoder
define a sampler (including explicit failure outcomes), not a tractable field
likelihood or verified cosmological posterior. Many-to-one canonicalization
can distort learned morphology. A trajectory-space target is conceivable;
efficient observation-driven inference with it remains unproved.

The argument for field-first is a prioritization decision, not a theorem that
q_S cannot be studied until generated fields improve: native fields already
exist. The audit acknowledges that parallel logical possibility. One bounded
attempt is accepted; repeated field-only improvements are not the destination.

MW/M31/M33 remain UNIDENTIFIED on new generated fields. A shared-cell M33 is
allowed, but the seven diagonal cell moments do not determine a unique bound
subhalo or subcell center. Latent decompositions need a calibrated joint law,
not independent catalogue labels. A negative finite-data learner/probe cannot
prove all statistical member information absent. Positive selected fixtures
do not supply the selection factor. Environment/selection/sky placement,
stellar-to-COM discrepancies and actual same-field LG-on/off information remain
missing. No actual fine LG inference or IC/zoom follows from a morphology pass.

## Approval handoff

Design bundle complete. Recommend the specified implementation plus ONE Slurm
diffusion experiment:1 GPU,4 CPUs,48GiB host RAM,24h allocation cap;30,000
updates or22h learning cap, then fixed evaluation. Existing training box,
translated/rotated/reflected crops, strict spatial holdout within the fit.
Architecture and memory must be sized before submission as above. Await user
approval at this new implementation boundary. No training is currently launched
by this recommendation and no new scientific result has been produced.
