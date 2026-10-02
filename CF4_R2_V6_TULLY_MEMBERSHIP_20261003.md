# R2 — v6/Tully-2015 exact member crosswalk (2026-10-03)

## Driver plan review

The v6 relation-stratified readout found that group velocities cannot be
treated as independent member redshifts, but the local `1PGC`/`GID` labels do
not reveal whether the linked galaxies actually belong to the group catalogue
that supplied CF4's systemic velocity. The EDD defines CF4 group `Vcmb` from
Tully-2015 2MASS group members, while the archived Tully tables provide both
`Nest` assignments and exact member PGCs. This bundle checks only that missing
identity bridge on the already frozen 272 training groups and 828 secure
links.

- Q-GOAL: yes. Exact member ownership is needed before the same observed
  redshift may appear in both the 2M++ count process and CF4's group-velocity
  source term.
- Q-LEAN: one source-ID join against the archived member list is sufficient.
  It reads no velocity values, marks, heldout measurements, likelihood, or
  field state and does no covariance fit or simulation.
- MW/M31 remain role-ambiguous and M33 unresolved. Tully catalogue identities
  are source-process labels only; they cannot seed generated LG candidates.
  Later LG observables must constrain the same NEW field at <=0.3 cMpc/h.

The audit is driver-run: this source-only continuation immediately follows a
driver audit, so a Fable/Astra request would be duplicate/consecutive under
the user's reviewer-routing rule. Previous all-sample source results are
retained; this query does not repeat their redshift summaries.

## Calculation contract

The script reconstructs only the frozen v6 group labels and secure links,
using the hash-pinned identity census, FP source-group IDs, count-point
manifest, and crossmatch. It then joins each secure PGC against Tully-2015
table 4's `(PGC, Nest)` membership, cross-checks table 5's combined-catalog
`Nest`, and maps each CF4 `1PGC` to the table-3 parent `Nest`. Missing and
multiple Tully associations remain explicit. Summaries are stratified by the
previous CF4/2M++ catalogue relation classes.

No `Vcmb`/`Vcmba` values are parsed or compared; the aim is member ownership,
not another velocity-offset statistic. Table/file hashes pin the archived
source. The official EDD definitions are [CF4 All Groups](https://edd.ifa.hawaii.edu/describe_columns.php?table=kcf4allgroup)
and [CF4 FP Groups](https://edd.ifa.hawaii.edu/describe_columns.php?table=kcf4fpgroup);
the archived membership columns are documented by the [Tully-2015 CDS
ReadMe](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/AJ/149/171?format=html).

One typed-H100 Slurm job runs three focused regressions and the source join:
one CPU, 2 GiB host memory (over 20% above the conservative <1-GiB estimate),
10-minute cap. H200/H100/A100 availability is checked; H100 is the selected
typed GRES. A pass establishes catalogue member overlap only. It cannot
calibrate the later EDD revision, selection, or a shared covariance law.

## Result

Pending. R2 remains NO-GO for production posterior/IC promotion. Any
unresolved or version-mismatched membership becomes an explicit source-model
requirement, not an inferred covariance or a reason to drop observations.
