# One focused science-design consultation: an implementable selected FP law

Read-only audit. Do not edit files, build, submit jobs, run simulations or send
mail. Work only in /home/kjhan/BACKUP/CF4. No repository-wide survey.
Return <=1000 words; concrete equations and a single next implementation.
Verdict ADVISE PROCEED / ADVISE MODIFY / ADVISE STOP / NO VERDICT.

R2's real CF4-conditioned z=0 density/velocity posterior at1–2cMpc/h is not
finished. Current N128/384=3cMpc/h working partial target is not production.
Later SAME generated field must supply MW/M31/M33 candidates and LG<=0.3
with phase-consistent zoom IC. MW/M31 ambiguous; M33 unresolved. Native truth
identities cannot seed/select generated candidates. No TNG requirement, mail,
new independent simulations, heldout tuning or high-k rescaling.

Read ONLY:
- CF4_R2_SELECTED_FP_ADVICE_20260929.md (important counterexample/corrections)
- scripts/cf4_r2_raw_fp_inputs.py (actual inputs and missing information)
- src/cf4_r2_linked_singleton_jax.py (current conditional mark law)
- CF4_R2_LINKED_POINT_MARK_OWNERSHIP_20260927.md (ownership, historical counts)
Optional: src/cf4_r2_fp_distance.py (source eta likelihood, not raw-data PDF).

Current single-mark target:47121 count points and1414 links, one FP and one
securely matched2M++ point per source group, no anchors, frozen training only.
The raw join now has8910? NO: exactly8901 CF4-linked training parent rows,
1414 selected links. Public source r,s,i,er,es,ei, optical magnitudes/errors,
extinction/k corrections, source richness, group/individual z; linked Ksmag.
Raw r is redshift-distance-based log size, r_true=r_z-eta (check conventions).
No K error column or per-row full optical error covariance supplied by join;
no external shared FP-fit covariance. These are limits, not proof unavailable.
The official mock lacks K and reconstructed group selection. Old marginal
coverage does not calibrate this joint selection.

We need avoid endless gate-building. The selection warning is not proof of
actual bias. Existing count x conditional-mark target has known approximations;
no longer HMC or N256 until an explicit defensible calibration route is chosen.
Keep .004dex source relative zero versus absolute calibration distinct.

Please independently derive ONE minimal defensible model that uses these
raw observations, distinguishing its computable implementation from empirical
calibration not identifiable in this data. Consider (do not assume) either:
(a) a normalized joint optical-FP/K marked population with finite shared
hyperparameters, replacing the published eta factors, not multiplying them;
(b) a genuinely valid conditional regression that removes known photometric
selection without discarding latent-distance population conditioning.
State conditional independences, integration measure, numerator/selection
normalizer, and how the count occurrence is scored once. A claim that apparent
photometry may merely be conditioned away must derive the remaining candidate
distance weights. If fitting shared parameters jointly, do not use their fit
as an independent prior on the SAME rows. Identify exactly what external
information or explicit assumption is indispensable, if any.

Do NOT repeat: infer latent-distance selection from observed P(link|K,z),
association-only weighting with unchanged selected-mark PDF, arbitrary scalar
offset/temperature/broadening or another brightness-cut demonstration. An
inverse FP by itself does not fix latent distance weights or sigma selection.
Nor append radial terms to TSC counts without a common observation law.

Is it possible to construct and FIT a selected conditional model from this
sample with known deterministic cuts, or is absolute population/calibration
unidentified? Distinguish identifiability needed for a conditional posterior
from proof of physical transfer. Give one bounded next ACTION that moves
science, not a list of future audits. It may honestly advise a different
tractable factorization using current data, but explain science/selection cost.

Q-GOAL: does this advance an actual CF4-conditioned present field and later
same-field LG, instead of only proving our code reproduces itself?
Q-LEAN: indispensable implementation versus deferrable checks; no huge generic
inference framework or long new simulations. No approval for large compute.
