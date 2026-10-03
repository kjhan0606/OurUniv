# R2 Tully–2MRS–2M++ full-member identity crosswalk — 2026-10-03

## Decision

The bounded source-identity census is complete. It establishes a large shared
object graph between the archived Tully 2015 member list and the local 2M++
catalog, while preserving several inconsistencies between Tully Tables 3, 4,
and 5. It does **not** identify a density field, calibrate shared-redshift
covariance, or authorize a posterior or IC. R2 remains **NO-GO**.

## Inputs and method

The source schemas and record counts are pinned to the official CDS catalogs:
[Tully 2015](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/AJ/149/171?format=html&tex=true),
[Huchra et al. 2012 / 2MRS](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJS/199/26?format=html&tex=true),
and [Lavaux & Hudson 2011 / 2M++](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/MNRAS/416/2840?format=html&tex=true).
Tully Table 5 contains 43,038 member-position rows; Table 4 contains 43,038
group-member rows; Table 3 summarizes 25,474 Nests. Huchra Table 3 has 44,599
2MRS rows. The 2M++ catalog has 72,973 rows, of which the official ReadMe
defines 69,160 as real galaxies and 3,813 as synthetic Zone-of-Avoidance rows;
the latter were excluded from identity matching.

The v2 run performed two mutual-unique, spherical position checks at 3 arcsec:

1. Tully Table 5 Galactic coordinates to Huchra Table 3 Galactic coordinates.
2. Huchra Table 3 equatorial coordinates to real 2M++ source coordinates.

An exact Huchra `ID`–2M++ `Name` match is the direct identity link. A
position-only 2M++ result is recorded as a candidate and is **not** promoted
to identity. The same rule applies to the 42 alias candidates below. Only
object IDs, positions, the 2M++ `Ref` fake-row flag, Tully membership keys,
and v6 split-role labels entered this computation; redshift, magnitude,
CF4-group/FP outcomes, field values, and likelihood scores were not used.

## Findings

### Cross-catalog identities

- **42,568/43,038** Tully Table 5 members have a reciprocal unique 2MRS
  positional match within 3 arcsec. Their separation median is 0.385 arcsec,
  p90 0.833 arcsec, maximum 2.994 arcsec. There are 468 with no 2MRS
  candidate and 2 with multiple candidates; those two are not forced into an
  identity.
- **35,743** Tully members have an exact 2MASS ID/2M++ Name link to a real
  2M++ row. This direct-name set maps to 27,529 v6 training, 4,374 heldout,
  1,216 buffer, and 2,624 outside-v6-parent rows. These are member identities,
  not independent observations or effective sample sizes.
- Of the **6,825** Tully→2MRS matches without an exact real-2M++ Name,
  42 have a reciprocal unique 2M++ position-only candidate and 6,783 have no
  candidate within 3 arcsec. The 42 candidates (31 training, 5 heldout,
  2 buffer, 4 outside the v6 parent) remain unassigned; split roles are
  reported only as candidate labels. There are no ambiguous 2MRS→2M++
  position matches among them.
- Of the exact-name links, 35,727 also have a unique 3-arcsec Huchra↔2M++
  positional match; 16 exact-name links lack that independent positional
  confirmation. They remain visible as a quality flag rather than being
  silently discarded or treated as a position-confirmed pair.
- Tully PGC 33946 links by the same 2MASS ID through Huchra to 2M++ `recno`
  56598, and this ID match is position-consistent; its v6 role is training.
  This resolves **object identity only**, not the discrepant published
  velocity measurement/source question documented separately in
  `CF4_R2_TULLY_RAW_HV_RECONCILIATION_20261003.md`.

### The archived Tully tables are not internally interchangeable

The official Table 4 and Table 5 member associations differ for five PGC
identities:

| PGC | Table 4 Nest | Table 5 Nest | Recorded issue |
|---:|---:|---:|---|
| 9067 | 200013 | absent from Table 5 | Table-4-only member |
| 40621 | 100002 | 0 | Table 5 ungrouped; Table 4 assigns a Nest |
| 41618 | 100002 | 0 | Table 5 ungrouped; Table 4 assigns a Nest |
| 42447 | 100002 | 0 | Table 5 ungrouped; Table 4 assigns a Nest |
| 212964 | absent from Table 4 | 200013 | Table-5-only member |

The Table 3 `Nmb` also disagrees with the archived member-row counts: Nest
200006 has 49 Table 4 rows versus `Nmb=47`; Table 5 has 194 rows for Nest
100002 versus `Nmb=197`, and 49 for Nest 200006 versus `Nmb=47`. No table was
chosen as a silent correction source. In the existing CF4–2M++ crossmatch,
PGC 9067 is only a non-mutual 18.74-arcsec review candidate to the same 2M++
row directly named by PGC 212964; this does not justify merging the PGCs or
repairing the archived membership. Nest 200013 is not among the selected v6
training parents.

### Relevance to the frozen v6 cohort

Across all Table 5 Nests, seven have exact-name-linked 2M++ members in both
training and heldout roles. A score-blind cross-tab of the frozen 272-group,
828-link training cohort reached 228 distinct CF4 `1PGC` parents and 215
nonzero Tully parent Nests; 13 distinct parent IDs have no Tully Table 3
parent entry. **None of the 215 reached Nests is among the seven with both
training and heldout members.** Therefore this archive’s known direct
Tully-member graph does not require rebuilding the current frozen v6 split
for train/heldout separation. This finding is limited to the archived Tully
Table 3 parent mapping and the current v6 cohort; it does not certify other
group definitions, future cohorts, or covariance.

## Driver review and next action

**Q-GOAL:** Yes, this closes a necessary object-ownership question for a
future same-field joint likelihood; it adds no direct constraint on the new
z=0 field. The present field posterior is still undelivered. MW/M31 remain
role-ambiguous and M33 remains unresolved. Their eventual observables must
constrain that same newly inferred/evolved field at LG resolution
`<=0.3 cMpc/h`; native identities may label calibration/evaluation only, not
select generated-field candidates.

**Q-LEAN:** Proportionate. The unexpected exact-name gap justified one
position-only alias check, not a matching-radius sweep, redshift comparison,
gravity rerun, or posterior gate. Keep the 42 positional candidates, 16
position-unconfirmed exact-name rows, and five Table 4/Table 5 member
disagreements explicit; do not hand-correct or merge them.

Next, incorporate the direct identity graph into the selected v6
shared-group-factor ownership/covariance design, while keeping the 13
unmapped parent IDs and Tully table anomalies unresolved. Before any fit,
specify the factor’s independent unit, member-level sharing, selection, and
group-redshift covariance. Preserve the untouched heldout outcomes. Then
evaluate one bounded training-only likelihood pilot and separately assess
heldout prediction. Do not claim a field posterior or move to R3 until this
target is stationary, its shared-data law is defensible, and the untouched
heldout test passes. No new PM evolution or production IC is authorized by
this source census.

## Artifacts and audit trail

- Source implementation: commits `4523510` and `64f7587` on
  `agent/freeze-zoom-pipeline`; both pushed to `origin`.
- Focused suite: 5 tests passed, including the pinned Table 4/Table 5
  discrepancy and 2M++ real/ZoA row-count contracts.
- Typed-H100 Slurm jobs 411497 (initial exact-name census) and 411502
  (alias-candidate follow-up) each completed in 8 seconds with 2 GB requested.
- Authoritative v2 result:
  `/gpfs/kjhan/CF4/z0_density/r2_tully_full_member_2mpp_overlap_20261003_v2/result.json`.
- Full member map:
  `/gpfs/kjhan/CF4/z0_density/r2_tully_full_member_2mpp_overlap_20261003_v2/tully_member_2mpp_identity.csv`
  SHA256 `5ce539bcbcdd279ffa859177100eb9dace1b8998bee8ff94bb87fcd352755645`.
- This was a consecutive driver-led source audit under the user’s duplicate/
  consecutive-review rule; no second external audit was requested. Neither
  run fitted a field, read heldout outcomes, ran PM evolution, or promoted R2.
