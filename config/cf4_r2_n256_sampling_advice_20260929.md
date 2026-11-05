Read-only advisory audit. Do not edit files, build, submit jobs, run simulations,
send messages/mail, or inspect unrelated projects. Cwd is OurUniv/CF4:
/home/kjhan/BACKUP/CF4. This is advice before a potentially large two-chain GPU
allocation, NOT a new routine gate or a request to repeat completed tests.

Read CF4_MASTER_PLAN.md and CF4_R2_DELIVERY_ROUTE_20260929.md for the approved
route and current evidence; CF4_END_TO_END_REPLAN_20260913.md if necessary.
Relevant implementation:
src/cf4_r2_resolution_target.py
src/cf4_r2_raw_volume_target.py
src/cf4_r2_chunked_volume_count.py
scripts/cf4_r2_n256_joint_pilot.py
scripts/cf4_r2_corrected_split_hmc.py
scripts/cf4_r2_prior_split_hmc.py
Read only these small actual result JSONs as needed (do not load big NPZs):
/gpfs/kjhan/CF4/z0_density/r2_raw_joint_pilot_v1/result.json
/gpfs/kjhan/CF4/z0_density/r2_n256_source_profile_v3/result.json
/gpfs/kjhan/CF4/z0_density/r2_n256_dynamics_profile_v1/result.json
/gpfs/kjhan/CF4/z0_density/r2_n256_joint_pilot_v1/result.json
The last job408412 is running; missing proposals are PENDING, not failed or
passed. Do not wait/poll it. Give conditional advice using actual evidence.

Final goal: CF4/galaxy/LG-conditioned present-state density and velocity
posterior with LCDM-consistent history and phase-consistent zoom ICs. The
environment needs1–2cMpc/h grid spacing; LG ultimately<=.3cMpc/h. Approved R2
implementation:384cMpc/h box,N256/1.5. First R2 science delivery may explicitly
be conditional on1414 secure raw FP/K marks plus47121 training counts, not
all-CF4 or externally calibrated absolute distances. Actual z=0 posterior is
first delivery. R3 later identifies MW/M31/M33 FROM THE SAME NEW FIELD; MW/M31
role ambiguity and unresolved M33 are retained, never truth-ID seeded.
Their observables must constrain that same field. N256 particle mass is
2.90e11Msun/h, not resolved LG halos. No new TNG/download requirement.

Evidence and limits:
- N128 new raw target pilot completed8 proposals in1h43m40s,7accepted,
  3/4fixed-step accepted. Force GL2, target GL4, LOS4x8, optical axis1/256.
  Priors are IC and24 independent white nuisance priors once; all9tracer and
  15population variables live. No old eta factor or independent FP zero.
  The raw-population prior is a declared model, not external calibration.
- White power drifts.6493->.6981: NOT equilibrium samples. The current L2
  trajectories are short. Fixed split metric is only a proposal guess:
  IC inverse Laplacian with fundamental mass6000,nuisance inverse mass1e-5.
  Coarse-force proposals receive fine-target Metropolis corrections; the
  coarse-force energy is not substituted for the fine target. Fine and
  coarse caches/accepted fields survive rejection. Mechanics tests passed.
- No heldout scores were read. V6 point/group/key split already fixed;
  do not resplit. Observed count keys/exposure remain128/3cMpc/h even when
  source/field is256/1.5. Completeness remains the piecewise-constant original
  source angular model. This limits information resolution; grid spacing
  is not an assertion that all1.5-cMpc/h modes are data determined.
- N256 source cost: full native count gradient532s with2^3 nodes per child
  source; device20.4GiB/host3.3. Too-small batches caused previous timeouts.
  Keep tested full-grid likelihood; experimental compressed backend is NOT
  promoted or required. No further optimization gate ladder.
- Actual N256 dynamics+probe adjoint7s after compilation, device19.4GiB.
  Phase-preserving white prolongation is exact under the declared restriction.
  Added high modes are prior initial values, not recovered information.
- Running N256 pilot uses force GL1,target GL2 per1.5cell, count mass factor
  1/8, unchanged observed grid/LOS/optical rule. Four bounded transitions,
  2discarded warmup+2fixed-step. Tracks inherited128-band power separately
  because newly drawn high modes disguise low-band drift. Source GL4
  sensitivity at the SAME N256 grid is still required, not yet done.

Driver's proposed next finite bundle IF the current pilot works:
1. Resume the SAME N256 target for two independently randomized chains.
   Use longer trajectories, chosen from measured pilot movement/cost, and
   discard adequate warmup before freezing adaptation. Starting from a
   single MAP-like parent is not overdispersed initialization; propose a
   realistic alternative using available states/prior draws, no truth IDs.
   A first bounded allocation of roughly2x24GPU-hours is contemplated, NOT
   submitted. Let measured oracle cost determine the proposal budget.
2. Save IC checkpoints/thin realizations and streaming present-field means,
   variance, physical velocity dispersion separately from posterior velocity
   uncertainty; record nuisance/low-mode/power traces including rejections.
   Avoid a huge particle/trajectory archive. Assess mixing and MC error;
   high acceptance alone is not enough. Extend only where evidence requires.
3. On actual retained states: same-grid finer quadrature and weak common-scale
   prior sensitivity, using reweighting ONLY if overlap/ESS supports it.
   Produce honest information-support maps. Freeze model before ONE untouched
   v6 predictive assessment. Selection/association remains conditional unless
   genuinely calibrated, not declared calibrated by matching training marginals.

Please answer <=1500 words, prioritized and concrete:
A. Q-GOAL: does this advance the final reconstruction/zoom goal and provide a
   legitimate conditional R2 delivery without claiming LG or all-CF4 success?
B. Q-LEAN: what is truly required vs deferrable? Avoid extra validation Python,
   repeated audits, blanket quadrature upgrades or a new proxy/data ladder.
C. Recommend the minimum useful longer-chain experiment, initialization,
   trajectory/adaptation policy, diagnostics and stopping/evidence criteria.
   Flag any mathematical error in the proposed current target/sampler wiring
   with file/function evidence. Do not confuse an approximate coarse FORCE
   with a change in the Metropolis target. Do not invent universal acceptance
   or runtime thresholds. Explain limits of two chains with related starts.
D. Address observed3 vs source1.5 grid honestly: is conditional delivery
   defensible with an information-resolution map? If you require changing
   observation resolution, explain the specific scientific necessity and how
   to preserve the already-frozen split, rather than simply demanding it.
E. Explicitly keep MW/M31/M33 same-new-field identification/observable linkage,
   ambiguity and unresolved members in the R2->R3 handoff.
Verdict: ADVISE PROCEED / CONDITIONAL / ADVISE AGAINST / NO VERDICT. Separate
present evidence from conditional predictions about running408412.
