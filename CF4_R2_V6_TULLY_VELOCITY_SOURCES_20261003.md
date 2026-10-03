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
`Vcmba` are excluded. Separately, exact-join each selected 2M++ `Name` to the
official Huchra et al. 2MRS table-3 `ID`, then compare its 2M++ `Ref` with the
2MRS `r_cz` adopted-redshift reference. A same ID and bibcode establishes
catalogued object/publication overlap, not that the same spectrum or exact
measurement was used. A different bibcode also cannot rule out indirect
reuse through a compilation. Huchra table-3 `cz` is solar-system-barycentric,
whereas 2M++ `Vcmb` is CMB-frame; do not compare those numbers without an
explicit frame conversion.

Sources: [EDD CF4 individual distances](https://edd.ifa.hawaii.edu/describe_columns.php?table=kallcf4),
[EDD CF4 All Groups](https://edd.ifa.hawaii.edu/describe_columns.php?table=kcf4allgroup),
[2M++ CDS ReadMe](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/MNRAS/416/2840?format=html),
the [Tully-2015 CDS ReadMe](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/AJ/149/171?format=html),
and [Huchra et al. (2012) 2MRS CDS ReadMe](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJS/199/26?format=html&tex=true).

The Huchra table is a 3.37-MB compressed, 44,599-row fixed-width catalog. Its
ignored local copy is hash-pinned in `data/PROVENANCE.md`. The focused parser
tests and one selected-cohort join are bounded metadata work; no GPU,
simulation, field, likelihood, or heldout values are needed. The runner can
also be executed as a typed-H100 Slurm job if this check later grows beyond
that bounded scope. The source snapshot and upstream cohort/result hashes are
checked before writing a fresh, non-overwriting result directory.

## Result

The initial classifier result from job411020 used the Huchra catalogue
bibcode, not the `20112MRS.*` source codes stored in 2M++. Its zero-overlap
interpretation is withdrawn; the preserved output is
`/gpfs/kjhan/CF4/z0_density/r2_v6_tully_velocity_sources_20261003_v1/result.json`.
Corrected job411025 found 3 explicit 2MRS source codes among 441 matched
Tully-member rows (two `20112MRS.FLWO.0000H`, one `20112MRS.ZMA..0000H`). The
other 438 references were correctly left unresolved at that point. Across all
441, absolute CF4-individual versus 2M++-point raw CMB-frame `Vcmb` differences
have median14, p90 64, maximum273 km/s, with 9 equal to stored precision. The
three explicit-code rows have median16, p90 59.2, maximum70 km/s; this is not a
measurement-error or covariance estimate.

The official Huchra et al. table-3 ID/reference join then matched all441 2M++
`Name` values exactly to a 2MRS `ID`. Fifteen rows have identical 2M++ `Ref` and
2MRS `r_cz` bibcodes; 426 have different codes. Thirteen of the fifteen exact
citation matches were outside the original explicit-`20112MRS.*` class. The
same-reference subgroup's CF4–2M++ absolute `Vcmb` difference has median35,
p90 111.8, maximum129 km/s (n=15); the different-reference subgroup has
median13, p90 62.5, maximum273 km/s (n=426). Thus a shared bibcode is not an
identical-measurement guarantee, and a different bibcode does not prove
independence. These small selected-cohort descriptions do not estimate
cross-covariance.

The first parser attempt stopped on official row43534, where CDS omits
trailing blank fixed-width redshift fields for a row with no `cz`. It wrote no
result. The parser now right-pads those absent tail fields, still rejects a
missing ID and duplicate IDs, and its 10 focused tests pass. The initial
read-only metadata join is preserved at
`/gpfs/kjhan/CF4/z0_density/r2_v6_tully_velocity_sources_20261003_v4/result.json`
(source `ec24ed0`). The completed extension at source `4403e47` took1.213s
using a bounded local metadata join; no GPU, heldout values, likelihood, field
state, PM evolution or simulation was used. Its result SHA256 is
`7293abe57c9609431961a4b89f0343eeb2eabda10dccf9108dc6feb5ffe8eff5` at
`/gpfs/kjhan/CF4/z0_density/r2_v6_tully_velocity_sources_20261003_v6/result.json`.

That extension also closes member-coverage bookkeeping. The828 selected
secure links map to215 unique Tully parent Nests (26 links have no unique
parent); those Nests have1,381 published table-4 members in total. The441
selected links that occur in table4 represent31.93% of that summed `Nmb`,
not the completeness of the full 2M++ catalogue. There are no `Nmb`/listed
member-count mismatches; only60/215 Nests have every listed member represented
by this selected link subset. Those60 consist of42 `Nmb=1` and18 `Nmb=2`
Nests. The absolute CF4 group-minus-2M++ member-mean residual is:

| Published Tully `Nmb` | Fully linked Nests | Median | p90 | Maximum |
|---:|---:|---:|---:|---:|
| 1 | 42 | 13 km/s | 72.2 km/s | 127 km/s |
| 2 | 18 | 13.25 km/s | 48.55 km/s | 68 km/s |

For the18 two-member Nests, the 2M++ member-velocity standard deviation has
median91.92, p90358.93 and maximum508.41 km/s. The singleton rows have zero
within-group scatter by construction. This small, selected, fully linked
subset is descriptive only; it does not identify a covariance or selection
law.

One explicit namespace warning matters for the predeclared control: Tempel
group `T10106` contains PGC54049 and PGC54054, while its CF4 parent
`1PGC=53982` maps to Tully Nest100181. Only PGC54054 is shared between the
Tempel pair and that Tully Nest's eight published members. Nest100181 is not
among the60 fully linked Nests. Thus Tempel, CF4 `1PGC`, Tully `Nest`, and
2M++ `GID` remain distinct membership namespaces even when a parent/control
label agrees.

Driver source audit of the current mechanics found the remaining science
barrier directly in code: `MomentObservationTarget` sums the per-row
`raw_field_logpdf` factors, and `cf4_r2_raw_live_mark.py` explicitly labels
its single-mark association assumption. The separate shared-group kernels
require caller-supplied calibrated covariance and group-inclusion inputs;
they are not wired into that moment target. Therefore this source join does
not close the multi-member CF4/2M++ observation law. Do not add a second
redshift score or infer covariance from these residuals. The next R2 work is
the existing shared-group likelihood/calibration branch, using the frozen
source graph and a defensible covariance/inclusion input; no posterior or
production field is authorized by these mechanics alone.

Driver decision: close this catalog-reference subroute; do not grow a citation
alias or velocity-difference ladder. Exact IDs establish that these441 objects
occur in both catalogues, and15 shared citations identify publication overlap
only. Neither the selected individual residuals nor the18 fully linked
two-member group means establish identical spectra, measurement covariance,
complete group selection, or a production group likelihood. Huchra `cz` is
solar-system-barycentric and was not compared numerically with CMB-frame 2M++
`Vcmb`. Q-GOAL: this is prerequisite source ownership, not a field constraint.
Q-LEAN: the bounded join/coverage check is now closed; no more citation or
residual sweep is warranted. R2 remains NO-GO for posterior/IC promotion.
MW/M31 remain role-ambiguous and M33 unresolved; their observables must
constrain the same NEW field at LG<=0.3 cMpc/h, with native truth identities
evaluation-only.
