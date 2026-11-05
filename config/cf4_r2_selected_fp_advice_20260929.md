# Focused R2 science advice: selection of the distance-indicator marks

Read-only audit. Do not edit files, build, submit jobs, run simulations, or
send mail. Work only in /home/kjhan/BACKUP/CF4. This is NOT LRD_JWST.

Goal: actual CF4-conditioned present density/velocity posterior, environment
1–2cMpc/h, later same-state MW/M31/M33 and <=0.3cMpc/h LG/phase-consistent zoom
IC. R2 currently has a conditional N128/384 (3cMpc/h) working target, not a
calibrated posterior. MW/M31 ambiguous and M33 unresolved; no truth-ID seeds.
Do not weaken the final science goal or introduce another long gate ladder.

The current short HMC mechanics pilot408337 is unchanged-target and running.
Do not recommend killing it simply because this calibration is incomplete;
do not promote it to a calibrated posterior either. We are planning the next
substantive calibration correction, NOT asking for another sampler tuning plan.

Read these three scoped files:
- CF4_R2_LINKED_POINT_MARK_OWNERSHIP_20260927.md
- scripts/cf4_r2_linked_fp_sparse_train.py (cohort loader/selection)
- src/cf4_r2_linked_singleton_jax.py
Optional narrowly relevant: src/cf4_r2_fp_distance.py;
CF4_R2_ZERO_ANCHOR_SOURCE_20260929.md. No repository-wide survey.

Primary-source observation verified by driver: Howlett2022 sec5.2.2 warns
that selecting a brighter magnitude subset can bias subsequent use and calls
for refitting FP/distances with the changed magnitude limit. Sec5.3 corrected
columns use richness-specific fits. Source URL:
https://arxiv.org/html/2201.03112#S5.SS2.SSS2
This is an r-band warning, NOT direct proof of bias for our K-selected crossmatch.

Current sample retains one FP row and one securely linked2M++ point per group,
no anchor, graph-closed training:1414 links. Thus the scored marks are a
selected subset of the public SDSS FP catalogue. Source supplies corrected
flat-eta-prior PDF moments/shape, with its original fn correction. The count
factor already describes K-selected2M++ populations. Its normalized conditional
mark mixture uses candidate weights from that count process, while setting
the association term to0 (constant). A common .004dex relative zero is free;
absolute/shared-fit calibration incomplete. Do not just widen it until fit improves.

Driver question: does the existing count-conditioned latent-distance weighting
already suffice for mark selection, or does an observed-mark-dependent matching
law still require p(m|eta,A) proportional to p(A|m,eta) p(m|eta) with its own
normalizer? Distinguish (a) source fn already used once, (b) count selection,
(c) additional optical/NIR/type/group matching. Don't assert any new factor
must be multiplied without writing the corresponding joint/conditional law.

Give one lean implementable recommendation based on AVAILABLE columns
(public r,s,i/errors, optical magnitudes, redshifts, richness, source PDF;
2M++ link IDs and photometry exist). No TNG or author emails. Specifically:
1. State which conditional independence would make current marks valid, and
   whether it is established. Separate a demonstrated error from an open risk.
2. Minimum defensible repair/calibration using existing public data or source
   mocks, and what CANNOT be identified from selected summaries alone.
3. Would an FP refit on the K-selected intersection by itself double-condition
   the count likelihood? Is a joint color/FP/inclusion model necessary?
4. One bounded, discriminating next action; do not propose many speculative
   diagnostics, arbitrary magnitude cuts, scalar corrections, or holdout tuning.

Q-GOAL: contribution to an actual CF4-informed z=0 posterior and later same-state
LG identification. Q-LEAN: essential versus deferrable and concrete information
gain. Verdict format ADVISE PROCEED / ADVISE MODIFY / ADVISE STOP / NO VERDICT.
No claim that these requests authorize a large new calculation.
