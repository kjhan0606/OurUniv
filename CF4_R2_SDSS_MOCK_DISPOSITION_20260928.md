# R2 SDSS-PV mock disposition — 2026-09-28

## Decision

Do **not** download the full 10.6 GB `mocks.tar.gz`. After Slurm407247,
perform one pre-registered, no-retention streamed subset of at most15 mock
members because the current linked FP factor explicitly uses a fixed
`sigma_los_km_s=100` stochastic source displacement in
`scripts/cf4_r2_linked_fp_sparse_train.py`. The subset reports source-PDF
coverage and compares mock central/satellite host-relative velocities with
that provisional scale; host-relative galaxy velocities are not automatically
identical to the model's source-displacement kernel. It is not a
group-selection calibration or R2 exit. This is the driver's amended
adoption of Fable's advisory.

## What the source can and cannot calibrate

Howlett et al.'s 2,048 mocks encode SDSS-PV FP measurement/selection and
host/member truth, so they can test the individual FP measurement PDF and
host-relative velocity behavior for that selected SDSS sample. The paper
explicitly omits the redshift-success rate and describes the mock HOD as
approximate and tuned to the SDSS clustering monopole. These mocks do not
provide recovered Tempel memberships or the parent-before-selection table,
nor a matched CF4/2M++ graph. They therefore cannot calibrate the missing
Tempel group-inclusion law, CF4 method inclusion, or shared CF4/2M++/FP
redshift covariance. Host truth is not observed group membership.

The archive is one 10.6 GB compressed tar, not independently downloadable
catalogue files. The existing bounded bridge streamed one fixed member
(5,304,320 compressed bytes), retaining the 33,881-row fixture. The separate
source-PDF evaluation reports nominal 68/90/95-percent containment of
68.540/90.325/95.257% on that one correlated mock box. That supports the
source measurement-PDF as a development component, but not the factor's
fixed 100 km/s stochastic displacement across independent volumes. The
bridge reports a selected-satellite host-relative LOS RMS of 397.6 km/s in
that one fixture; it is not itself a replacement calibration for the linked
2M++/FP factor.

## Q-GOAL and Q-LEAN

- **Q-GOAL:** the mock host/member observables could support a future
  individual-to-host velocity term, but no mock term constrains the actual
  CF4+2M++ evolved field or identifies the missing observed Tempel/CF4
  selection graph. They do not advance the same-field MW/M31/M33 posterior;
  MW/M31 roles remain ambiguous and M33 unresolved when unsupported.
- **Q-LEAN:** reject a full archive download and reject using mock host IDs as
  Tempel membership or an inclusion denominator. The small subset is limited
  to the existing 100 km/s per-link stochastic displacement: first archive
  realization's eight observer catalogues plus one catalogue from each of
  the next seven distinct realizations in tar order. Selection uses archive
  names/order only, never eta scores or v6 heldout marks. Stream at most
  512 MiB compressed; if those 15 members cannot be reached within the cap,
  stop without fallback. Retain per-member summaries only, not mock rows.
  Evaluate source-PDF 68/90/95 coverage by redshift and host-mass strata and
  central/satellite host-relative velocity offsets, including the empirical
  fraction within 100/200 km/s. A coverage screen within two binomial
  standard errors, with no distance/host-mass trend beyond two standard
  errors across at least two independent boxes, supports only the individual
  FP eta-PDF component. It does not by itself validate the 100 km/s kernel:
  if the host-offset distribution contradicts a single Gaussian scale, keep
  the factor partial and specify a central/satellite mixture or discrepancy
  model. This cannot calibrate 2M++ tracer FoG or group COM scatter.

The joint data ownership contract already exists in
`CF4_R2_COARSENED_MARKED_COUNTS_20260926.md` and
`CF4_R2_LINKED_POINT_MARK_OWNERSHIP_20260927.md`. Do not write a duplicate
plan: its unresolved terms remain `p(U|C,F,S)`, `p(A|C,U,F,S)` and
`p(G|A,C,U,F,S)`, including field-dependent group inclusion/association and
the shared group/member redshift-distance law. Any later factor must be
normalized and score each observed count exactly once.

## R2 status and next work

R2 remains **NO-GO** for a production z=0 posterior. First complete the
bounded v6 membership and dynamic-support diagnostic already submitted as
Slurm407247; then perform only the bounded15-member calibration above. After
that, work on the missing actual-data group/association ownership and its
heldout prediction; neither mock result closes those terms. No IC or
high-resolution map is promoted by this memo.

References: [Howlett et al. 2022](https://academic.oup.com/mnras/article/515/1/953/6611706),
[official SDSS-PV v1.1 data and mock release](https://zenodo.org/records/6824749).
