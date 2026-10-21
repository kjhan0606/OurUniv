# One focused correction to the just-returned R2 group advice

Read-only scientific advice. Do not edit, execute jobs, build or invoke agents.
The previous advice is assessed in CF4_R2_FIELD_LEVERAGE_20260929.md, last section.
Read scripts/cf4_r2_linked_fp_sparse_train.py:load_train_singletons and
src/cf4_r2_linked_singleton_jax.py. This is a substantive likelihood-design
question within the current bundle, not a routine code audit.

Facts: the985 excluded Tempel-grouped links ALREADY have precisely ONE observed
FP row and ONE securely linked count point per source group (zero non-FP
anchors), selected by load_train_singletons. Other physical group members can
exist in the source/point catalogue, but this proposed mark set does not add
their distances or independently score their group-mean redshift. The current
429 and these985 have disjoint source-group IDs. Your previous claim that985
means985 multi-FP averages with enhanced per-system precision is unsupported.
FP score movement also does not measure its fraction of posterior information.

Source paper https://arxiv.org/html/2201.03112 sec2.2–2.3 defines eta from
d(z_group); the source's group redshift enters its FP size and eta numerator.
It is not by itself a second independent velocity observation when we evaluate
the published FP PDF at eta_pred=log10(d(z_group)/d_latent)+zero.
The same measured numerator cancels in eta_observed-eta_pred. Richness-specific
FP means/std/alpha are in logdist_corr*. Common fitted-FP/absolute calibration
uncertainties remain unresolved for ALL rows, including the existing429.

Driver question: Is the blanket exclusion of ALL grouped galaxies actually
required for this one-FP-per-group conditional mark model? Could the985 be
included using the existing conditional latent-source mixture, with their
published group-distance numerator, but conditioning on the single linked
point's observed individual redshift (owned by the count/point model), without
inventing independent group-redshift data or a multi-member distance average?
This would be explicitly a broader CONDITIONAL WORKING likelihood, fixed source
FP fit and uncalibrated association/shared-source covariance; not a complete
physical group model or production R2. The SAME global LOS parameter must
remain in counts and marks; no narrower group kernel or ad hoc extra weight.

If yes, specify the essential assumptions and smallest regression needed
before one bounded1414-row joint fit. If no, identify the precise mathematical
missing factor/covariance that cannot be absorbed by treating groupz as the
fixed distance-indicator reference. Do not merely cite a docstring prohibition.
Distinguish one observed FP per physical group from multiple FP members of one
group. Do not pretend a one-row marginal alone solves cross-group FP fit errors,
selection or tracer/environment-dependent residuals. No heldout or new TNG.

Q-GOAL: actual CF4-conditioned current-state posterior before same-NEW-field
MW/M31/M33 (roles ambiguous,M33 unresolved at N128). Q-LEAN: avoid building a
full host/member hierarchy if this selected marginal does not require it;
also avoid bypassing a real shared-data constraint just to add more rows.
Answer in <=800 words with concrete mathematics and a usable recommendation.
