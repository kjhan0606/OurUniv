# R2 — raw velocity-source overlap for Tully members (2026-10-03)

## Driver plan review

The preceding identity bridge established that 441 of 828 frozen v6 secure
CF4/2M++ PGC links are members of the archived Tully-2015 Nest assigned to
their CF4 `1PGC` parent. EDD defines the CF4 group `Vcmb` as an average over
the 2MASS K<11.75 Tully-2015 group members. The next narrow question is whether
the 2M++ row for those same galaxies cites the 2MRS source used to build that
membership catalogue, and how closely the two catalogs' *raw individual*
CMB-frame `Vcmb` values agree. This does not use Tully's adjusted member
`Vcmba` or group velocities.

- Q-GOAL: yes. This identifies a possible duplicated observed-redshift
  contribution between the 2M++ count process and CF4 group-velocity source.
  It is source-overlap evidence only, not a constraint on the reconstructed
  density/velocity field.
- Q-LEAN: one selected source-reference cross-tab over 441 verified members;
  no full-catalog refit, covariance fitting, likelihood, posterior, heldout
  access, or simulation.
- The local CF4 inputs remain the canonical hash-pinned 2026-07-07 VizieR
  J/ApJ/944/94 snapshot. The currently listed EDD table has four additional
  rows globally; that mutable endpoint is not a gate for this reproducible
  cohort. No TLS verification bypass is allowed.
- MW/M31 roles remain ambiguous and M33 unresolved. The eventual LG
  observables must constrain those roles on the same NEW field at
  <=0.3 cMpc/h; native truth identities may label evaluation only.

This is driver-reviewed: a Fable-directed audit would be consecutive with the
immediately preceding driver audit, so the user assigns it to the driver.
No Astra call is made for this routine source cross-tab.

## Data and calculation contract

The selected cohort is fixed at 272 training groups/828 secure PGC links,
then restricted to the 441 PGCs whose table-4 Nest matches the Tully parent
Nest reached through CF4 `1PGC`. Compare only the ordinary CMB-frame CF4
individual `Vcmb` and 2M++ point `Vcmb`; group values and Tully adjusted
`Vcmba` are excluded. Use the 2M++ `Ref` bibcode as a source label, with
`2012ApJS..199...26H` identified as Huchra et al. (2012), the 2MRS catalogue
referenced by the Tully-2015 ReadMe. A matching bibcode and close velocities
are evidence for likely source overlap, not proof of identical spectra,
individual measurement-error covariance, or a group-mean covariance law.

Sources: [EDD CF4 individual distances](https://edd.ifa.hawaii.edu/describe_columns.php?table=kallcf4),
[EDD CF4 All Groups](https://edd.ifa.hawaii.edu/describe_columns.php?table=kcf4allgroup),
[2M++ CDS ReadMe](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/MNRAS/416/2840?format=html),
and [Tully-2015 CDS ReadMe](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/AJ/149/171?format=html).

One typed-H100 Slurm job runs three focused tests and the pinned source join:
one CPU, 2 GiB memory (more than 20% above the <1-GiB estimate), 10-minute
limit. H200/H100/A100 are checked; H100 is selected with the required typed
GRES. The source snapshot and upstream cohort/result hashes are checked before
writing a fresh, non-overwriting result directory.

## Result

Pending. The result may refine the source-overlap description but cannot
promote R2, estimate a calibrated covariance, or authorize a posterior/IC.
