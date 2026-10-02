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
driver audit, so the Fable-directed review would be duplicate/consecutive.
Under the user's routing rule, the driver performs it (Astra remains the
normal substitute when a distinct Fable audit is warranted). Previous
all-sample source results are retained; this query does not repeat their
redshift summaries.

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

Slurm job411000 COMPLETED/exit0 on typed H100 in5s; all3 focused regressions
passed; MaxRSS3,520K (3.44MiB). The source join found 441 of828 secure PGCs in
both Tully tables4 and5; all441 have matching Nest assignments in those
tables and agree with the Tully parent Nest reached through the CF4 `1PGC`.
The other387 are absent from the archived Tully member table; none is forced
into a group. There were no member/table conflicts or ambiguous joins.

Across272 CF4 groups, table4 member-Nest relation classes are shared62,
partly-unassigned163, and all-unassigned47. The CF4-`1PGC` parent-Nest classes
are shared253, distinct6, partly-unassigned2, and all-unassigned11. Within the
167 groups whose CF4 and 2M++ catalogues both share a group assignment, 44
groups have all matched Tully members in one Nest, 96 are partly unassigned,
and27 have no assigned matched Tully member; the CF4-parent Nest is shared in
164 of these groups and absent in3. These are identity/linkage facts about
the archived Tully-2015 catalogue, not physical membership truth or proof
that its velocity values are the exact current EDD inputs.

The frame check found a material semantic mismatch for any direct velocity
comparison: Tully-2015 table4 exposes adjusted `Vcmba`, not a raw member
`Vcmb`. Its ReadMe Note G3 points to the CF2 cosmological adjustment. Tully
et al. (2013), Eq.15, adjusts negative peculiar velocities as a function of
distance uncertainty; the archived table does not provide the raw member
velocity and all inputs needed to invert that adjustment. Therefore the
Tully crosswalk cannot supply a raw-redshift covariance estimate, and no
`Vcmba`/`Vcmb` subtraction or ad-hoc inverse is authorized. Sources:
[Tully et al. 2013, CF2 §V.1](https://arxiv.org/abs/1307.7213) and the
[Tully-2015 CDS ReadMe](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/AJ/149/171?format=html).

Current-source version check: the local CF4 group/individual inputs are
hash-pinned in `data/PROVENANCE.md` to a 2026-07-07 VizieR download of
J/ApJ/944/94 (`cf4_groups.csv`: 38,053 rows; `cf4_galaxies.csv`: 55,877
rows). The official EDD description for CF4 All Groups reports its corrected
2023-05-02 table and defines group `Vcmb` as the average over all Tully-2015
2MASS K<11.75 members; the current EDD table listing reports 38,057 entries.
See the [official EDD All Groups definition](https://edd.ifa.hawaii.edu/describe_columns.php?table=kcf4allgroup),
[EDD group-velocity definition](https://edd.ifa.hawaii.edu/describe_columns.php?table=kcf4allvel),
and [current table selector](https://edd.ifa.hawaii.edu/dfirst.php).
That four-row count difference means the local snapshot is not yet proven
row-identical to the currently listed EDD table. The online EDD column page
was readable through indexed official documentation, but a direct syntax
request failed TLS certificate verification; no verification bypass was
used. This does not invalidate the pinned July snapshot, but selected-row
version identity remains open.

Driver decision: the exact-ID bridge is accepted for archive-level source
ownership only. Q-GOAL: yes, it disambiguates a possible shared redshift
source term but does not yet constrain a generated field. Q-LEAN: yes, one
join and a primary-source frame check; no velocity refit, covariance fit,
likelihood, posterior, heldout access, or simulation. R2 remains NO-GO for
production posterior/IC promotion. The next bounded action is to establish
the selected-272-row correspondence between the pinned July VizieR group
snapshot and the current EDD table; do not download/reprocess the full catalog
for a four-row global count discrepancy. Also keep the distinction between
current group `Vcmb` and archived member `Vcmba`; no inferred transform is
allowed. If the selected EDD rows cannot be obtained with valid TLS, retain
the July snapshot as the explicit analysis version and keep the current-EDD
equivalence unresolved. MW/M31 remain role-ambiguous and M33 unresolved; their
observables must ultimately constrain those roles on the same NEW field at
<=0.3 cMpc/h, and native truth identities remain labels for evaluation only.
