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
