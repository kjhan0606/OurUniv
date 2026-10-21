# Astra mid-course review: CF4 mark holdout extension — 2026-10-21

The response below is preserved unchanged. Driver disposition follows it.

## Astra response

## Mid-course verdict

**CONDITIONAL GO for a bounded held-out mark extension; no held-out measurement read or scoring yet. R2 remains NO-GO.** This is not an exit audit.

The current evaluator is materially incomplete **as validation of the joint count+FP conditional model**: it checks only 2M++ counts, not the selected FP information constraining distance/velocity. Its count-only implementation remains useful and correctly labeled.

### The 197-row cohort

The reported 197 role-1 groups are a defensible **candidate conditional selected-mark cohort**, not yet 197 certified evaluable rows. I accept that count as driver-supplied metadata evidence; I did not inspect its measurements.

There is one additional structural requirement beyond membership eligibility: `scripts/cf4_r2_raw_field_profile.py::load_inputs` applies an ownership-ledger screen excluding unresolved associations/collisions. The existing reconciliation and ledger are training-specific. Apply the **same scientific rule**, made role-aware, to the held-out candidates before calling this the active-v6 evaluation cohort.

Graph closure supports leakage control under the recorded associations. It neither establishes physical membership nor makes mark availability ignorable. Successful prediction here could support the frozen model’s selected-mark predictions; it cannot calibrate missing eligibility, targeting, redshift-success, FP-quality, linkage or group-inclusion probabilities.

### Sufficient work now, without measurement access

Freeze one compact manifest and implementation contract covering:

1. Explicit v6 split/hash and ordered PGC, source-group and linked-recno identities. Do not inherit the loader’s historical v5 default.
2. Exactly one mark, one secure linked point, no anchor, allowed membership state, and the active no-unresolved-association/collision rule. Verify source-group, PGC and point uniqueness.
3. Role-1 consistency across each candidate’s existing graph component, with no train/buffer overlap through the recorded parent/group/source relationships. Distinct source labels alone do not prove distinct parent groups. Reuse existing identity evidence; no new catalogue search is needed.
4. Training-only provenance of fitted population parameters, centering and preprocessing. Preserve any published/shared-calibration dependence as an explicit limitation rather than claiming independent end-to-end validation.
5. Exact measurement schema, transform, factor ownership and predictive aggregation, tested using synthetic or training fixtures.

These are implementation checks, not additional science gates. If an existing identity connection reveals real leakage, report its consequence; do not silently repair the split around an already-fitted posterior.

### Missing measurement assets

The driver’s **0/197 coverage** finding means the present three training assets cannot evaluate these marks. A bounded, identity-restricted extraction from the pinned raw selected sources is proportionate. It is **not blocked by the missing full-survey inclusion law for this narrower conditional diagnostic**.

However, raw `r/s/i`, within-bin K, uncertainties and inputs used to reconstruct cuts include scored measurements. Calling them “covariates” does not exempt them from the holdout embargo. Write/test the extraction recipe now; materialize and validate those rows only in the declared post-fit evaluation phase. ID-only source coverage can be checked earlier. Missing, duplicate or invalid rows must be reported, not silently removed according to their eventual scores.

Do **not** recover frozen field-dependent mixture weights from the training asset. Rebuild the held-out source support and weights from each retained draw.

### What requires valid posterior draws

Execute the frozen extraction, row-level numerical checks, predictions and scores only after a valid posterior exists—not after the one-sweep ESS feasibility pilot.

Reuse the actual raw-mark operator: shared population parameters, optical covariance/cut normalization, within-bin K density, linked **2M++ point radius** for source conditioning, and the CF4 row’s separate redshift reference. Preserve coordinate-density/Jacobian conventions; do not substitute the historical eta likelihood or add another `f_n`, group-redshift observation or count occurrence.

For each retained draw, evaluate both components on the **same evolved field and shared nuisance state**:

`ell_joint(draw) = ell_count_holdout(draw) + sum(ell_selected_mark(draw))`

Then apply one log-mean-exp over draws, retaining repeated states and legitimate zero-probability states. Separately marginalized count and mark scores cannot simply be added to obtain that joint predictive score. This remains the frozen conditional-v6 law, with its stated coarsening/association assumptions—not a newly established full survey law.

If reporting marks **conditional on the observed held-out counts**, distinguish that quantity explicitly: under the same joint model its log score is joint predictive minus count predictive, not generally the unweighted mark-only mixture. Joint-score Monte Carlo reliability must also be assessed; merely receiving “valid draws” does not ensure a stable high-dimensional predictive estimate.

### Minimum next action; Q-GOAL and Q-LEAN

**Add the metadata manifest, role-aware structural reconciliation and synthetic/training-tested evaluator contract now as a bounded follow-up alongside the unchanged pinned ESS pilot. Defer measurement extraction and actual scoring until valid post-fit evaluation.** Do not modify or extend that pilot, launch another inference run, reopen parent-table searches or create a new survey requirement because of this finding.

**Q-GOAL:** Yes. This closes a real evaluation omission on the route to the first conditional z=0 density/velocity posterior at 1.5 cMpc/h in the 384-cMpc/h box. Neither this plumbing nor eventual selected-mark predictive success certifies full-survey calibration or field uncertainty.

**Q-LEAN:** Yes, with the bounded separation above. It reuses the frozen observation operator and existing sources without extra simulations or arbitrary gate ladders.

MW/M31 remain role-ambiguous and M33 unresolved. Their observables must ultimately constrain those same roles on the same **NEW evolved LG field at ≤0.3 cMpc/h**. Native truth identities remain calibration/evaluation labels only, never candidate seeds or selectors. No R2 closure, exit email or R3 entry is supported here.

## Driver disposition

Adopt the CONDITIONAL GO for the bounded implementation work. The metadata-only join previously found 197 role-1 single-mark candidates under the active membership rule, while all three existing materialized measurement assets cover 0/197. Astra additionally identified the need to apply the training ledger’s no-unresolved-edge/collision condition to role 1 and to verify parent/GID/CF4-role closure. We will implement and test this contract without reading held-out measurements, counts, positions, radii, or field states. The post-fit extraction recipe will be frozen but not executed now. The active ESS job and its source pin remain unchanged. No field, posterior, predictive score, R2 approval, exit email, or R3 transition is claimed.
