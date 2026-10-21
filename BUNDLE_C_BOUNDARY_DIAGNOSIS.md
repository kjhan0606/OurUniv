# Saved-field boundary diagnosis — driver-reviewed

2026-09-13, user-approved. No external review: bounded routine diagnosis,
no discovery claimed, goal change or large calculation. Keep original NO-GO.

347108 generated16 valid conservative fields; boundary-gradient ratio fails
16/16, power5/16, connected fraction2/16. The existing boundary metric is a
ratio of mean squared density differences at coarse faces versus other faces,
then divided by the same statistic in ONE native realization. It is sensitive
to rare peaks and their positions; a failure alone does not prove seams.

One CPU-only Slurm job,2CPU/2GiB/5min (estimated peak1.5GiB+20% rounded2).
Use all eight STORED first draws and matched native cubes, never select good
cases or regenerate the missing second draws. Reproduce their old ratios.
For each axis, compare all face phases at periods2/4/8 on a common interior:
faces8..55, transverse cells8..55, equal phase counts. Report raw density
squared gradients AND log1p(rho/mean-rho) squared gradients; the latter is a
diagnostic view, not altered physical data or a replacement acceptance gate.
Save per-plane energy and largest1/10-edge contributions to distinguish
rare-peak sensitivity from widespread grid locking. Linear-ramp and
coarse-staircase controls check phase indexing and seam sensitivity.

Deliver JSON and one phase-profile plot. No neural fit, new field draws,
amplitude adjustment, posterior, significance/independence claim or automated
promotion. The eight source regions can overlap. Driver distinguishes:
code mismatch; peak-sensitive native denominator; generated-only periodic
roughness; or mixed/inconclusive evidence. Original criteria/results stay.
Do not infer a unique model cause from this descriptive comparison.

Q-GOAL: distinguish an actual fine-field defect from a screening weakness
before spending more compute toward the z=0 LG posterior. Q-LEAN: reuse
stored physical fields and fixed metrics; one short script, two controls,
one run. MW/M31/M33 identification and observation likelihood remain the
frozen field-only position law, not native catalogue-seeded candidates.
This diagnosis does not resolve member masses/COM or unresolved M33, and
does not apply actual LG conditioning to failed fields. Next scientific
decision follows the evidence, not automatic longer training.

## Result and driver disposition

Slurm347159 COMPLETED/exit0 on syn07 at2026-09-13 01:55:29 KST,8s wall,
3.884s application, peak host.153GiB. Both controls pass, all eight saved
first draws processed, all24 original axis-ratios reproduced to1e-12 relative.
Source ce3528ed64f14ee26de51ee89b00d8a22f4233d9; submission mistakenly used
the symbolic EXPECTED_COMMIT=HEAD, so result metadata says HEAD. Driver read
the identical full HEAD immediately before submission and after completion;
no intervening edits/commits. Record this provenance limitation rather than
rewriting outputs or repeating science. Future submissions use literal hashes.

Mixed result, not an arithmetic bug or an across-the-board seam catastrophe:

- Balanced raw boundary/internal ratios vary widely even in native fields:
  .0478–3.9879 across24 case/axes, versus generated.0406–5.2574. Original
  full64 native ratios also vary .1458–4.1901. A ratio to one native realization
  mixes stochastic peak geometry with possible grid effects.
- Only10 out of13824 interior coarse-boundary cell-pair differences contribute
  median85.23% of native squared-gradient sum and87.29% of generated sum.
  Thus raw rho-squared is an extremely peak-concentrated seam screen. This is
  NOT evidence that density peaks themselves are scientifically irrelevant.
- In the log1p(rho/mean-rho) diagnostic, median coarse-boundary/internal ratio
  is native.9937 versus generated1.1414. Generated/native comparison exceeds1
  in22/24 axes (range.9058–1.2648); overlapping fields/axes are not independent,
  so this is descriptive evidence, not a significance calculation.
- Period2 log phase medians are native[1.009,.991], generated[1.061,.939];
  period4/8 similarly show excess at refinement faces. This supports modest
  hierarchical grid locking, not its unique architectural/training cause.
- First stored projection524295 shows recognizable coarse structure and some
  fine structure, with velocity/dispersion differences. It is one fixed example,
  not proof of morphology or actual-LG reconstruction.

Driver closes this bounded diagnosis. Keep original NO-GO and all power/
connected-fraction failures. Do not replace the failed gate with log-density
ratios and declare success, smooth/clip samples, or resume training blindly.
The next model decision should target cross-parent fine structure and the
partition of coarse velocity variance into fine bulk versus internal motions,
while separating peak-sensitive realization scatter from grid-locked errors.
No new learner or actual-data conditioning has been launched by this diagnosis.
Use the existing seven-moment native/generated fields and learned model, not
another standalone MW/M31/M33 identity diagnostic or a direct-IC route change.

Artifacts: /gpfs/kjhan/CF4/z0_density/bundle_c_v1/stable_field_boundary_v1/
{tests,profiles,result}.json and phase_profiles.png. Original stable_field_v1
and stable_field_eval_v2 preserved. Routine diagnostic findings evaluated by
driver under the new selective external-review policy; no external call.
