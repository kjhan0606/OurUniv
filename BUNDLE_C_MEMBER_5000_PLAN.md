# User-requested 5000-update single-field continuation

2026-09-11. User explicitly requests5000 updates after discussion of increasing
the single-field learning budget. Interpret5000 as CUMULATIVE on fixture0:
resume the saved300-update model/Adam state, add4700 updates. No reset and no
automatic13-field continuation, additional seeds, models or later bundle.

342013 finished normally in3m09s, all5 tests passed. Same-field initial/final
loss7.140842 -> .468869. Final MW/M31/M33 mass ratios2.064/1.146/.957,
overlaps.915/.838/.789 and map-L1 1.235/.469/.379. All final map-L1 criteria
failed and multi-field training was NOT entered. Late logged loss still
declined, but M33 map error fluctuated: length insufficiency is a hypothesis,
not an established cause. This is a newly authorized larger budget, not a
claim that the original300-step screen passed or a repeated automatic retry.

## Exact continuation

- Source `member_mass_repair_v2/checkpoint_final.pt`, source commit8dbaeae,
  exactly300 completed updates; associated result must identify job342013,
  status INCONCLUSIVE_SINGLE_FIELD_LEARNING and fixture0.
- Restore SAME530804-parameter flat-head U-Net and complete AdamW state,
  including moments/step counts. Validate source/config compatibility and
  restore a matching step300 snapshot before any optimizer update. Reuse a
  small optimizer-continuation regression in the same Slurm allocation.
- Unchanged total-field features, .1875/128^3 support, labels, observer,
  log-mass-squared+shape-KL loss, equal roles, AdamW1e-4/weight-decay.01,
  gradient clip10 and strict FP32. No augmentation in this single-field fit.
- No model/optimizer reseeding of the saved weights. The source has no RNG
  state, but the continued learning path has no stochastic augmentation or
  dropout. Do not claim bitwise continuation across GPU/library changes.
- Train cumulative steps301..5000, bounded by the same70min learning/90min
  Slurm envelope. No plateau-based early stopping or early scientific pass.
  Save small post-update scalar snapshots at1000/2000/3000/4000/5000 plus
  source300; intermediate points are progress reports, NOT extra gates.
  Keep final checkpoint and native/predicted/baseline maps; no checkpoint sweep
  or best-epoch selection. Existing history remains per-update/pre-update loss.
- Judge ONLY the fixed5000 endpoint by unchanged screen criteria: each member
  mass ratio[.8,1.2], overlap>=.8 and normalized map-L1<=.3, nonnegative maps,
  positive remainder and conservation<=1e-6. PASS_SINGLE_FIELD_LEARNING_ONLY
  or NO_GO_SINGLE_FIELD_LEARNING_AT_5000. Incomplete budget/execution is labelled
  separately, never a completed scientific failure. Preserve the earlier result.

## Goal, scope and resource accounting

Q-GOAL: determine whether this fixed field-to-member model can learn a native
decomposition with an adequate *tested* budget. Pass is one-field fitting,
possibly memorization, NOT generalization, observed LG or a joint posterior.
No claim5000 is theoretically adequate for all models. Its relation to the
final CF4+galaxy+LG -> z0 posterior -> LCDM IC -> forward LG-zoom goal is limited
but concrete: assess a possible mass readout for local observation constraints.
LG numerical target<=.3, environment1–2 cMpc/h remain unchanged.

NEW-field interface accepts only total moments/observer, never native member
IDs/masses/centers as inputs. MW/M31/M33 fractional/shared-cell targets remain
explicit; no M33 waiver or distinct-peak rule. This one-field fit cannot test
identification on new fields. Stochastic assignment, velocities/stellar offsets,
selection, field priors and observation-conditioning of the SAME total field
are deferred and unresolved. A successful readout does not supply that inverse
conditioning link. No actual-data inference/density-prior training/IC/simulation.

Q-LEAN: reuse runner/model/loss/metrics/plots and16-fixture availability check;
only restore-state support, one focused continuation test and five small scalar
reports. No new diagnostic infrastructure or per-checkpoint audit ladder.

One Slurm GPU/2CPUs/6GiB host/90min; a40,a100,h100,h200 excluding syn06.
Previous host/GPU peaks3.324/2.244GiB. Retain conservative host estimate5GiB
+20%=6GiB, existing20GiB GPU envelope.4700 extra updates at measured~.48s
suggest~38min learning; allow45–60min estimate with node/IO variation, not ETA.
Separate output `member_mass_single5000_v3`; all v1/v2 results retained.
After this one endpoint report the decision and wait at the next bundle.
