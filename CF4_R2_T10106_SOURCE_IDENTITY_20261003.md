# R2 — T10106 source-catalogue identity control (2026-10-03)

## Driver review and bounded objective

The v6 source-only graph census (Slurm 410928, source `e9c06d3`) passed its two
focused tests and completed in four seconds. It found 6,028 training FP source
groups: 1,833 with one direct secure 2M++ point, 272 with multiple, and 3,923
with none. The score-blind least-complexity eligible multi-link control is
Tempel source group `T10106`: two direct count members (`recno` 52802/52824),
two FP rows (`PGC` 54049/54054), no cross-method anchor, and two distinct
population/voxel keys. Both count members are population 3 at 125.5622 and
121.6602 cMpc/h. This establishes graph topology only; there are no
likelihood scores, mark values, field state or posterior result.

Driver audit, applying the user's rule that a duplicated or consecutive Fable
review is handled by the driver:

- Q-GOAL: yes. Catalogue identity for this fixed training control is needed
  before selecting the correct shared-group factor that links CF4 marks to
  count-owned 2M++ points on the same field.
- Q-LEAN: one score-blind cross-catalogue identity join is proportionate. It
  reads only IDs, with no CF4 distance marks, heldout
  values, likelihood, PM state, fit or sampler.
- MW/M31 remain role-ambiguous and M33 unresolved. This source join cannot
  identify any generated component; eventual MW/M31/M33 observables must
  constrain those roles on the same NEW evolved field at LG resolution, with
  truth identities reserved for calibration/evaluation.

The audit distinguishes CF4 `1PGC`, Tempel `source_group`, and 2M++ `GID`
namespaces. Equal or different labels do not establish physical membership.
The calculation is frozen to the selected control in
`r2_v6_multimember_graph_census_20261003_v2/result.json` and the source hashes
listed in the output JSON.

## Execution

The source-only implementation is `scripts/cf4_r2_t10106_source_identity.py`.
It checks the two secure member crossmatches and reports their CF4 `1PGC` and
2M++ `GID` relations independently. It does not use group richness, group
velocities, distance moduli, FP eta or any likelihood/field data. Two focused
regressions protect against conflating catalogue namespaces and hiding
duplicate/missing member identities.

The runner requests one typed H100 GPU, one CPU, 2 GiB host memory (more than
20% above the sub-0.5-GiB estimated working set), and a 10-minute cap. H200,
H100 and A100 modes were checked before submission; H100 was selected. This is
a source-identity check, not a GPU calculation; it remains on Slurm per the
project's execution policy.

## Interpretation boundary

The result may establish whether these two catalogues assign the linked
points to a common catalogue group, but cannot establish physical common
membership, group inclusion probability, member-redshift covariance,
CF4 shared-mark covariance, or a calibrated same-field likelihood. If the
catalogue partitions disagree, preserve both assignments as observed
association uncertainty; do not force a common latent group. If they agree,
that is still only a bookkeeping bridge, not a physical group law. No
posterior promotion follows either result.
