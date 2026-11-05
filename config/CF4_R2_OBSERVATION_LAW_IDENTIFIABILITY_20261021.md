# R2 observation-law identifiability reconciliation — 2026-10-21

## Purpose and scope

This is a driver-led synthesis of already reviewed, hash-pinned v6 and source
audits, following the completed band-metric diagnostic. It does not repeat a
catalogue census, read heldout values, fit a field, or change the active
estimand. Astra's 2026-10-21 mid-course advice was to make this boundary
explicit before any further fit or simulation.

The 1,414-mark component discussed below is specifically the active
SDSS-PV/CF4 linked training cohort, not all eight distance-indicator methods
represented in the full CF4 compilation.

The active science route jointly infers latent initial conditions and their
forward-evolved present-day state/history. The z=0 density/velocity posterior
is the first science delivery; the eventual Local Group zoom must reach
`<=0.3 cMpc/h`. This reconciliation produces no field or posterior. The
band-metric spending rule has failed and that sampler proposal is closed; more
sampler tuning cannot identify missing observations.

## Identifiability table

| Observation component | What is present in the frozen analysis | What the current model can identify | What remains unidentified / minimum added information |
| --- | --- | --- | --- |
| 2M++ count process | 47,121 v6 training count rows (37,951 frozen training keys) and a graph-closed, field-dependent expected-count term. The count occurrence is scored once with the model's explicit source/radial terms. | A conditional count-field score under the adopted count catalogue, graph, exposure, and tracer assumptions. | A complete CF4 group-entry or FP-mark-availability law. The count factor does not supply a CF4 parent/failure denominator or calibrate every source-type, group, and link inclusion process. |
| Linked SDSS-PV FP marks | 1,414 v6 training marks from the SDSS-PV/CF4 linked cohort, each with one secure training-count link, one mark, one distinct Tempel parent/singleton and no unresolved association in this selected subset. The raw FP factor uses the published object-level cuts and the same linked-count source measure in numerator and denominator. | The explicitly conditional FP-mark contribution for these observed linked marks, conditional on their linked count points and marks being present. | Whether mark availability is ignorable, plus the probability that an eligible CF4 source obtains a usable distance mark. Selected marks alone cannot estimate the missingness/selection law. |
| FP measurement-selection normalization | Published `f_n` normalizes an individual SDSS-PV FP measurement likelihood over that object's magnitude and velocity-dispersion cuts; the published distance PDF already includes this correction. | The documented object-level FP measurement correction for the selected measurement and stated cuts. | `f_n` is not a Tempel/CF4 group-inclusion probability, a redshift-success model, or a parent list of groups with no distance measurement. Do not multiply a second `f_n` into the current raw mark factor. |
| Source eligibility, targeting, redshift success | The SDSS-PV catalogue contains 34,059 FP rows built from DR14; its source history spans SDSS target/sample releases, not one homogeneous CF4 parent. Reviewed SDSS-PV/Tempel joins cover observed rows, not a complete source-specific target-and-failure parent. Of 34,059 rows, 418 lack a Tempel counterpart; among 10,020 R2-eligible rows, 75 are source-ungrouped—not known non-detections. The nearest-random DR7 mask proxy had a 43.390% false-positive rate on native `in_mask=0` and was rejected. | Structural facts about selected rows and a partial historical targeting-provenance bridge. | Eligibility, attempt, redshift-success and later FP/CF4-linkage probabilities. Minimum useful addition: a parent matched to the actual source-selection stage, with eligibility, targeting/attempt, success/failure, FP-quality and catalogue-linkage states for detections and non-detections. In the exact frozen 1,414-mark cohort, all best-object IDs occur in `SpecObjAll`; the tested `Target`/`TargetInfo` route returns zero. A separate join to SDSS-I/II tiling tables matches 1,400 selected objects, and 1,385 selected spectra have an exact `SpecObjAll.tile = sdssTilingInfo.tile` match. This is historical targeting provenance for selected spectra, not a denominator or a complete DR14/SDSS-PV parent; see the bounded evidence record below. Do not treat selected `SpecObjAll` rows alone as a target/failure parent. |
| Group/member association and inclusion | The v6 graph has 6,028 training FP source groups: 1,833 with one direct secure count link, 272 with multiple links, and 3,923 with none. Among the 272 multi-link groups, catalogue assignments are not universally shared: 167 shared in both catalogues, 32 partly unassigned 2M++ IDs, 54 all unassigned 2M++ IDs, 11 distinct 2M++ groups, and 8 distinct CF4 `1PGC` groups. | Frozen catalogue ownership/crosswalk structure; not physical group membership. The 1,414 selected mark subset avoids a multi-mark parent factor only for that subset. | The population of CF4/Tempel groups that could have entered but did not, and a calibrated rule relating CF4 `1PGC`, Tempel, Tully `Nest`, and 2M++ `GID` membership. Catalogue-absent or unassigned rows are not failures. |
| Member redshifts and shared covariance | Tully member rows overlap only partially with the wider v6 graph: 441/828 secure PGC links are represented; those selected links are 31.93% of summed published `Nmb`, and only 60 reached Nests are fully represented. The checked velocity offsets vary by relation class. Within-Tempel repeated-mark covariance is structurally absent from the selected 1,414-mark factor because its marks occupy distinct parents; that does not remove group/count, source, or zero-point uncertainty elsewhere. | Selected-source identity and descriptive velocity aggregation/residual summaries. | A calibrated joint uncertainty for group `Vcmb`, group distance modulus and member velocities, including covariance and selection, where the broader shared-group likelihood is used. The existing shared-group mark primitive does not acquire this calibration from selected residuals. |
| Validation and uncertainty | The untouched v6 heldout split is preserved. The band-control endpoints are explicitly discarded. | Mechanics/conditional diagnostics only. | A stationary field posterior, calibrated posterior uncertainty, and heldout predictive evaluation under a defensible observation law. Training fit or selected-sample prediction cannot validate missing-group inclusion. |
| MW / M31 / M33 | No resolved same-field LG identification has been produced by this bundle. | Nothing new about component identity or LG structure. | MW/M31 remain role-ambiguous and M33 unresolved. Their observables must ultimately constrain those same roles on the same NEW evolved LG field at `<=0.3 cMpc/h`; native truth IDs may label calibration/evaluation only, never seed or select candidates. |

## Driver disposition and next boundary

The existing evidence supports a selected, linked-mark conditional component;
it does not support fitting an empirical CF4 group-survival/inclusion law from
selected-only rows. It also does not establish that conditioning on mark
availability makes that availability ignorable. Do not revive the historical
disjoint-count survival defect as a correction to the current inclusive-v6
selection law. Do not invent a survival/bias coefficient, infer missing rows
from unmatched catalogues, or use SDSS-PV selected-host residuals as a
transferable CF4/2M++ law. COSMOSIM/MDPL2 and SMDPL were already closed as
current calibration sources for the documented access/truth reasons; no
large retrieval is reopened here.

For a *full-survey calibrated posterior*, the minimum route is source-specific
parent/failure metadata plus a defensible group/member and shared-covariance
calibration that matches the relevant CF4 component. For a narrower
conditional diagnostic, retain the 47,121 count process and 1,414 selected
linked marks as explicitly conditioned data; any availability-bias
sensitivity assumptions must be externally justified and reported as
assumptions, not learned from the selected rows. The current evidence does not
justify claiming either a full-survey posterior or calibrated uncertainty.

The bounded DR14 key-coverage probe is complete. The frozen
`training_photometry.npz` supplied 1,414 unique PGCs; the local SDSS-PV join
maps these to 1,414 unique `bestObjID`s. The earlier eight-request,
read-only `Target`/`TargetInfo` test returned no links and closes only that
direct candidate key route. A bounded alternative query, in eight batches of
at most 180 IDs, joined `SpecObjAll.targetObjID` to
`sdssTiledTargetAll.targetID` and `sdssTilingInfo.targetID`. All 1,414
selected `sciencePrimary` rows returned; 1,400 distinct selected objects
matched historical tiling records and 14 did not. The joined table contains
1,417 unique target/run/tile records across 75 tile runs; 17 matched objects
have more than one run. A second eight-batch query added the actual
`SpecObjAll.tile`: 1,385/1,414 selected spectra have at least one same-target
`SpecObjAll.tile = sdssTilingInfo.tile` match, 15 have tiling history but no
exact tile match, and 14 have no historical target record. This strengthens
the interpretation as SDSS-I/II provenance for most of the *selected*
spectra, but not as a denominator. Official DR14 table descriptions define
`sdssTiledTargetAll` as the SDSS-I/II tiling target list with `targetID` equal
to the CAS DR7 object ID, and `sdssTilingInfo` as a record per target/run.
The DR14 `TiMask` view defines `TILED=0x1`, `DECOLLIDED=0x2`,
`COVERED=0x4`, `REMOVED=0x8`, and `CULLED=0x10`; observed values are retained
without treating any mask as a redshift-success flag. The selected-cohort
tiling join does not show the eligible objects that failed targeting,
spectroscopy, FP quality, or later CF4/Tempel linkage, and cannot calibrate
those probabilities. No catalogue or target parent was downloaded. The
query template, exact cohort binding, per-batch response hashes, and
aggregate counts are in
[`cf4_r2_dr14_tiling_cohort_probe_20261021.md`](cf4_r2_dr14_tiling_cohort_probe_20261021.md).

On 2026-10-21 the current public SDSS-PV archive was checked at Zenodo v1.1.0.
Its file manifest contains the selected PV catalogue, random catalogue, mock
description, and the previously audited mock ensemble; it does not publish
the Table-1 supersets or a row-level preselection/failed-target parent. The
Howlett et al. paper states that the exact queries for the catalogue and its
supersets are available upon reasonable request. This establishes a possible
author-supplied route, not a currently accessible calibration input. No
author was contacted, consistent with the no-email instruction, and the large
mock archive was not downloaded again. Sources: [Zenodo v1.1.0 manifest](https://zenodo.org/records/6824749),
[Howlett et al. data-availability statement](https://academic.oup.com/mnras/article/515/1/953/6611706).

A direct read of the primary Howlett et al. paper further confirms why the
public release cannot close this denominator: §2.5 says the composite DR8–DR14
sample had no publicly available angular mask that tracked its imaging,
targeting, tiling and veto history; §3.2 says the mock construction does not
account for redshift-success rate. The paper's SDSS-PV selection model and
mock suite remain useful for the published FP/counting regimes, but they are
not an attempt-complete source-specific parent for the CF4-linked marks. This
is a bounded literature check, not a new data or mock analysis; it supports
closing the public-parent search and retaining the explicitly conditional
estimand. Source: [Howlett et al. (2022), §§2.5, 3.2 and Data Availability](https://academic.oup.com/mnras/article/515/1/953/6611706).

The bounded run-4 result identifies only a post-target-selection historical
tiling frame: run-entry flags and some later spectrum-record stages for one
deterministically selected SDSS-I/II run. It is not representative of the 75
runs linked to the selected cohort. The 2,273 target-table join misses remain
in-frame with unknown eligibility, and the 24,880 clean-warning/non-null-z
same-tile primary rows are observation-record proxies, not complete attempt
outcomes or FP-usable distances. The calibration does not transfer to the
active 1,414 linked-mark likelihood: it does not identify pre-target
photometric eligibility, FP/morphology quality, or downstream CF4/Tempel
linkage. Accordingly, close exploratory coverage queries. Reopen this source
only if an independently defined, source-matched parent with eligibility,
attempt/outcome, FP-quality and catalogue-linkage states becomes available.
Otherwise retain the conditional linked-mark estimand and treat any
availability-bias correction as an externally justified sensitivity
assumption, not something fit from this frame. Do not infer failures from
missing joins. Do not launch another sampler, metric ladder, gravity/RAMSES
run, or mock archive download in this reconciliation. Q-GOAL: this provenance
work constrains one observation stage but produces no z=0 map/posterior. The
active route is joint latent-IC/evolved-state inference, with the z=0
posterior first. Q-LEAN: the narrow feasibility question is answered; no
further query ladder, survey census or gate suite is justified. MW/M31 remain
role-ambiguous and M33 unresolved; their observables must ultimately constrain
those same roles on the same NEW evolved LG field at `<=0.3 cMpc/h`, with
native truth IDs used only for calibration/evaluation.
R2 remains NO-GO.

## Scope of the only currently identifiable field target

If R2 later delivers a field posterior without new parent/failure data, its
estimand is the posterior under the frozen **conditional v6 model** for the
47,121 training count rows and the 1,414 observed linked SDSS-PV marks—not a
full-survey calibrated CF4 posterior. Conditioning on observed mark presence
does not by itself make the conditional mark distribution immune to
availability bias. Keep the published object-level FP measurement cuts
separate from the unmodeled parent eligibility, redshift success, FP-quality,
Tempel/CF4 linkage and group-inclusion processes; do not fit or silently set
those missing probabilities from selected rows. A held-out predictive check
within this selected cohort can assess only that conditional model and cannot
validate the missing parent/failure process. Published calibration dependence
and unresolved shared count/mark covariance remain explicit limitations.

Driver disposition after Astra's 2026-10-21 read-only advice: retain this
conditional estimand, but do not start an N256 likelihood-informed-subspace
implementation or pilot without training-only basis-rank, setup-cost and
full-target-correction evidence. The existing exact target/kernel remains the
reference; there is no posterior promotion or R2 exit. Q-GOAL: this is still
only a route toward the first z=0 density/velocity posterior in the same
latent-IC/evolved-field model. MW/M31 are role-ambiguous and M33 unresolved;
their observables must ultimately constrain those same roles on the same NEW
evolved LG field at `<=0.3 cMpc/h`, with native truth IDs used only for
calibration/evaluation. Q-LEAN: preserve this boundary and settled evidence;
no new sampler architecture, data census, diagnostic ladder, or gravity run.
Reopen sampler development only with a costed, capped, falsifiable proposal
that corrects both informed and complementary blocks against the full target.

## Evidence already in the repository

- [`CF4_R2_SELECTED_MARK_DENOMINATOR_AUDIT_20261005.md`](../CF4_R2_SELECTED_MARK_DENOMINATOR_AUDIT_20261005.md)
- [`CF4_R2_SDSS_MOCK_STREAM_PLAN_20261005.md`](../CF4_R2_SDSS_MOCK_STREAM_PLAN_20261005.md)
- [`CF4_R2_COSMOSIM_MDPL2_ACCESS_20261005.md`](../CF4_R2_COSMOSIM_MDPL2_ACCESS_20261005.md)
- [`CF4_R2_V6_GROUP_IDENTITY_CENSUS_20261003.md`](../CF4_R2_V6_GROUP_IDENTITY_CENSUS_20261003.md)
- [`CF4_R2_V6_REDSHIFT_OVERLAP_20261003.md`](../CF4_R2_V6_REDSHIFT_OVERLAP_20261003.md)
- [`CF4_R2_V6_TULLY_MEMBERSHIP_20261003.md`](../CF4_R2_V6_TULLY_MEMBERSHIP_20261003.md)
- [`CF4_R2_V6_TULLY_VELOCITY_SOURCES_20261003.md`](../CF4_R2_V6_TULLY_VELOCITY_SOURCES_20261003.md)
- [`config/cf4_r2_equilibration_astra_20261011.md`](cf4_r2_equilibration_astra_20261011.md) (pair-3 result and review record)
- [Howlett et al. (2022), SDSS peculiar-velocity catalogue](https://doi.org/10.1093/mnras/stac1681)
- [Official SDSS DR14 table descriptions](https://skyserver.sdss.org/dr14/en/help/docs/tabledesc.aspx) and [schema indexes](https://skyserver.sdss.org/dr14/en/help/docs/indices.aspx)
