Read-only science advice. Do not edit files, build, submit jobs, run simulations,
send messages/email, use Bash, spawn agents, scan filesystems or inspect processes.
Use only the exact project files listed below when additional evidence is needed.

This is an important science/cost decision after the first numerically usable
actual-data field fit, NOT a routine stage approval. Answer in Korean, <=1200 words.
Driver independently checks and decides; your verdict is advice, not authority.

Goal: conditional local-universe z=0 density/velocity and uncertainty first,
then same-state MW/M31/M33 constraints at <=0.3 cMpc/h and phase-consistent zoom
IC. Ambient target1–2 cMpc/h. Approved latent-IC joint history/present-state route.
R1→R2 surrounding field→R3 LG→R4 accurate evolution→R5 zoom IC.

Current field is N128/384=3 cMpc/h, DEVELOPMENT, not target resolution. The
v6 frozen observation graph supplies47,121 2M++ training counts and429 strict
ungrouped one-point/one-FP-row links.985 Tempel-grouped links are excluded, not
assumed independent. Buffered exposures and untouched heldout counts/FP exist.
One FP zero prior SD.004 dex; remaining source covariance/association and
survey-matched tracer bias/FoG calibration are unresolved. Existing TNG
dispersion evidence is not a transferable survey calibration. No new TNG or email.

Numerics: finite GH radial-cut jumps were replaced by unnormalized shell-CDF
integration,4 nodes x32 physical intervals, explicit8sigma tail, periodic images.
At fitted v1,4x32/8x32 count-score difference.0008623, exposure L1=1.407e-6,
max occupied log difference.001044, scalar derivative error1.93e-6. Live support
is rebuilt every fit/trial; no likelihood floor or altered cosmological prior.
Initial PM+observation finite-difference checks passed, but do not claim this
certifies all possible states or all model physics.

408032 took35m59s,17 accepted updates: objective231875.999→175535.453;
count log likelihood−231771.137→−174395.939; conditional FP log ratio.0859→−50.9406;
max optimizer-coordinate gradient756.69→129.91. IC prior penalty104.948→1088.293.
The initial white_IC=.01*fixed unranked draw is ONLY an optimizer initialization,
not a prior realization; Gaussian prior/LCDM spectrum unchanged. MAP missing
prior small-scale variance is NOT by itself evidence of an incorrect prior.
Final rho mean1, range.0093–497.55, rate multiplier2.508, predicted training
count40625 vs47121. Strong bands and flows in the map are not named-object IDs.
No stationary solution, posterior uncertainty or heldout prediction yet.

Continuation408084 is currently running,32-update/70min cap, same target and
accepted v1 state, fresh L-BFGS history. At24 further accepted updates:
objective161665.592, count−159708.499, FP−46.9565, gradient_inf34.64. Typical
full gradient evaluation~75s; score-only trials also cost time.5 startup
optimizer/moment tests passed. No N256 is submitted. Terminal field/readout
will also save physical particle velocity variance, distinct from uncertainty
and the phenomenological LOS width. No inference from its mere completion.

Possible next cost decision: another <=3 H200 GPU-hour bounded same-target
optimization/curvature-feasibility bundle, NOT N256, long HMC or arbitrary
high-k amplitude correction. Is that scientifically justified, or should a
specific model/information limitation take precedence? Prior identity-mass
HMC with ESS2–5 remains closed. Do not demand unavailable full-catalog mocks
as a generic answer without explaining an actionable alternative or real impasse.

Please assess:
1. Q-GOAL: Does this direction support the actual R2/LG/zoom goal? Distinguish
   useful partial-target progress from what remains indispensable for R2.
2. Q-LEAN: What is essential versus deferrable? Recommend at most3 substantive
   next bundles, no elaborate gate framework or another broad diagnostic sweep.
3. What can/cannot be concluded from count improvement and FP degradation?
   What minimum next evidence decides whether more MAP iterations are useful?
4. What uncertainty approximation is defensible/affordable here, and when?
   Do not label a nonstationary MAP or optimizer inverse Hessian calibrated posterior.
5. MW/M31/M33 must be identified from the SAME NEW state in R3; their roles are
   currently ambiguous, especially unresolved M33. Their actual observables must
   constrain that state; native truth IDs may never seed/select generated candidates.
6. End with PASS / CONDITIONAL PASS / REJECT / NO VERDICT on the proposed bounded
   next work, reasons and the most important concrete action. No fabricated numbers.

Exact optional evidence:
CF4_R2_FITTED_TANGENT_20260928.md
CF4_R2_SDSS_MOCK_DISPOSITION_20260928.md
CF4_END_TO_END_REPLAN_20260913.md
scripts/cf4_r2_v6_partial_map.py
src/cf4_r2_linked_singleton_target.py
src/cf4_r2_linked_singleton_jax.py
src/cf4_r2_marked_tracer_jax.py
