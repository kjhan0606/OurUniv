# DR14 selected-cohort / SDSS-I-II tiling provenance probe — 2026-10-21

## Scope

This is an exact, selected-only provenance check. It is not a catalogue
release, parent-sample construction, selection-function fit, or R2 science
result. All SkyServer requests were read-only `GET` SQL requests, limited to
180 frozen IDs per request. No table dump or raw response body was retained;
the compact aggregates, SQL fingerprints and response SHA-256 values below
are retained so the observed query boundary is auditable. SkyServer does not
guarantee row ordering, so response hashes may change on an otherwise
equivalent rerun.

## Frozen cohort binding

- Selected PGCs: `training_photometry.npz`, 1,414 unique `selected_PGC` values
  (`/gpfs/kjhan/CF4/z0_density/r2_raw_fp_inputs_v1/training_photometry.npz`),
  file SHA-256 `e470e2a741952ea864eb70d613df4d95c864a24e2c453bc4f0dcd5649b2318ce`.
- PGC-to-SDSS-PV object mapping: `SDSS_PV_public.dat`
  (`/gpfs/kjhan/CF4/external/sdss_pv_6824749/SDSS_PV_public.dat`),
  file SHA-256 `a48cfd9a9a3c27ff908f98972b2e26536e514b6635f9ed14c656dff3e5075485`.
- The join yielded 1,414 unique selected `bestObjID`s. SHA-256 of their
  numerically sorted newline-delimited IDs:
  `54f94706536c454d70526672e3f74db38a60b8964ff26902e59f0d00132d9489`.
- Each query round used those IDs in the frozen cohort order, split into
  offsets 0, 180, 360, 540, 720, 900, 1080 and 1260 (last batch 154 IDs).

## Query round A: target and tiling rows

SQL template (the final `IN` values are the corresponding at-most-180 frozen
IDs):

```sql
SELECT s.bestObjID,s.targetObjID,t.targetID,t.primTarget,t.secTarget,
       t.tiMaskAll,t.tiPriority,ti.tileRun,ti.tile,ti.tiMask
FROM SpecObjAll s
LEFT JOIN sdssTiledTargetAll t ON t.targetID=s.targetObjID
LEFT JOIN sdssTilingInfo ti ON ti.targetID=t.targetID
WHERE s.sciencePrimary=1 AND s.bestObjID IN (...)
```

All eight HTTP requests returned 200. The result was 1,414 distinct
`sciencePrimary` best-object rows and 1,431 joined rows. Every selected row
had a `targetObjID`; 1,400 selected rows matched `sdssTiledTargetAll` and
`sdssTilingInfo`, while 14 matched neither. There were 1,410 distinct
`targetObjID`s (the 14 unmatched selected rows correspond to 10 distinct
IDs), 1,400 distinct matched `targetID`s, and 1,417 distinct target/run/tile
records across 75 tile runs. Seventeen matched selected objects had more
than one tiling run.

Raw `tiMask` frequency per target/run record was `{2: 3, 4: 11, 5: 53,
7: 1350}`. Raw `tiMaskAll` frequency over distinct matched targets was
`{5: 51, 7: 1349}`. These are provenance counts, not rates among eligible
parent targets or redshift-success outcomes.

| Batch offset | IDs | joined rows | SQL SHA-256 | response SHA-256 |
|---:|---:|---:|---|---|
| 0 | 180 | 181 | `4159d95dbd09606080ba74d38a99af9b35d1a5da8df1c85f5bff52317941a66f` | unavailable: captured value was only 63 hex characters; not reconstructed |
| 180 | 180 | 180 | `1c33f95ebfd84a87591915bcef432a371c922c17d8c564321b540dc6efc894ee` | `bb5d1bc41c7f5ed7a162056e1281f4bf135df54e95e2116d87f3943d92796f21` |
| 360 | 180 | 182 | `61fa6e840978720be5c779ea8972ff705d28a29f3cb441916214076125dd952a` | `395e9778efbd5729df7e05a1687d2dd33bd4532ea949ffe450a183b4157adf1f` |
| 540 | 180 | 182 | `71e03eba4ae33e555ce9535c7baa5e5bc17a39b1c308dd6ae0f28dfba8fc0f53` | `77e0fdcff65ccb27208a69d871711ba5ccac658d9bd9507e95b74893f29f779c` |
| 720 | 180 | 185 | `ea86c254941d8958f27ded2233161c7d6201a2c34c24155c96f298f8fcdedd7f` | `1a503ffaac7006cd6902563422e50f58d65ab53962721d3260d87017181cc133` |
| 900 | 180 | 183 | `ea1c44f303a8fdf01f6c77aa39aedaaffb3cf07341d27c035195d7272f72728a` | `167e28b7b2f8e971e46fbcce9204ef1654c955062fc3accef3a325be1b3799c4` |
| 1080 | 180 | 182 | `ee51d1dcb728ddd82cb0ba99d5d16857c1d62502daeab7f2ec0cca57c34c30eb` | `644b065bebe8f3744da78f5c59a79b00180b509ed4acf4b6dfae436b9acb1aa7` |
| 1260 | 154 | 156 | `3517f43bb9347ba82f7b1b740eadc02c06e4c694aee0308d9fa88e28f13fb775` | `941f51648834b96c74b4acfdf70c50aab9c556290f6fdef0858f12ca211b6183` |

## Query round B: actual spectrum tile equality

This same frozen cohort and split were queried, adding `s.specObjID`,
`s.tile AS specTile`, `ti.tile AS tilingTile`, `ti.tileRun`, and `ti.tiMask`.
The join keys and `sciencePrimary` restriction were unchanged. All eight
requests returned HTTP 200 and 1,431 joined rows. All 1,414 distinct
`bestObjID`/`specObjID` rows were present. For 1,385 selected spectra, at
least one tiling record for the same `targetID` had `specTile = tilingTile`;
15 selected spectra had historical target/tiling rows but no exact tile
match; 14 had no historical tiling target record. This is strong evidence of
SDSS-I/II tile provenance for most of the *selected* spectra, but still does
not expose the missing eligible population or all downstream failure stages.

| Batch offset | IDs | joined rows | SQL SHA-256 | response SHA-256 |
|---:|---:|---:|---|---|
| 0 | 180 | 181 | `95314a9392e3a4b6e37a396c471434d61d3abd59340b971b08d30f3f5c01cda1` | `ccea7f9b6faad2c56bb1da5c756087121d68734c26eb465c86b1b08db488dd81` |
| 180 | 180 | 180 | `39ecbdd9630d88d12a45b5f05b0add784456ab90ddb54693f516e6a98cbc8d60` | unverified: captured value shares a 13-hex suffix with its SQL fingerprint; raw output unavailable |
| 360 | 180 | 182 | `2c61b014e47630d10a8729f8f2bea6fdc2e45bd1542166ca83aebef95bbc51dd` | `ec59b941bb5f0d2c8b4db20e6bd645e1a18222b79af9f7e27074d91696aece65` |
| 540 | 180 | 182 | `a025fb96139496b7c525077cd6df329415e497aeeadd3100fd219418cd2210d3` | `51da2abda5767b2f1eec38d9b6d09f56f25721ef12a0f3d1e72df675f97452fb` |
| 720 | 180 | 185 | `2e386a8ab6229019d767882df2dce9668002e2b6e94b6ee6f5418a3b05d1bb34` | `45135a05f1d25d3f34f66aa0d2717f492cfe5f1af68ea7710fcfc56bb705d08c` |
| 900 | 180 | 183 | `d6e7b4122a3649f6f66e1f7b9f047d14b5275f535e4718462bebae8a1bf83e44` | `72a24adbb9b750ff433198effdd160c46de6ccef978c5fd20d78e41eefe6f28b` |
| 1080 | 180 | 182 | `3e63f2fbb9f820d2f89b7b631a2c97eb081e5a3c5eee7f83d6f145462a7a4399` | `57e1c7cc59ac3f14322fb2e86d3a966080078df51ec0f9f9e6a9c88a2809c642` |
| 1260 | 154 | 156 | `1f578236c8a8f9a7f6dc4c994a3f8d19f47d70a7c3222a10db45b06c6f67a439` | `d1b3ce054d22542a6a7757bfc879251570048e03f4ce9e63158ee330504a3bde` |

## Mask meanings and interpretation boundary

The read-only DR14 `TiMask` view returned these rows:

| Name | Value | DR14 description |
|---|---:|---|
| `AR_TMASK_TILED` | `0x00000001` | Assigned to tile |
| `AR_TMASK_DECOLLIDED` | `0x00000002` | Object in its collision group which may be assigned a fiber |
| `AR_TMASK_COVERED` | `0x00000004` | In area covered by a tile |
| `AR_TMASK_REMOVED` | `0x00000008` | Removed from tile due to collision with plate center, quality, or light trap hole |
| `AR_TMASK_CULLED` | `0x00000010` | Culled from this tiling run |

Thus `tiMask=7` sets the first three bits; it is **not** a redshift-success or
FP-mark-availability flag. The official DR14 table descriptions define
`sdssTiledTargetAll` as SDSS-I/II targets run through tiling, with `targetID`
the CAS DR7 object ID; `sdssTilingInfo` records each time a target entered a
tiling run. See [DR14 table descriptions](https://skyserver.sdss.org/dr14/en/help/docs/tabledesc.aspx),
[DR14 indexes](https://skyserver.sdss.org/dr14/en/help/docs/indices.aspx), and
the [DR14 `TiMask` view](https://skyserver.sdss.org/dr14/en/help/docs/tabledesc.aspx?name=TiMask).

## Run-4 parent-feasibility aggregate

The run was chosen independently of the frozen CF4 cohort: the first indexed
`sdssTileAll` pair was `tileRun=4,tile=73`. A first tile-73-only aggregate was
discarded because conditioning on an assigned tile would select the frame.
The retained aggregate instead began with every indexed `sdssTilingInfo`
target in `tileRun=4` (28,633 distinct `targetID`s), grouped by raw `tiMask`.

The exact SQL text was not retained, so the original statement cannot now be
replayed or independently checked from this record. The recorded join
contract is:

```sql
FROM sdssTilingInfo AS ti
LEFT JOIN sdssTiledTargetAll AS t
  ON t.targetID = ti.targetID
LEFT JOIN SpecObjAll AS s
  ON s.targetObjID = ti.targetID
WHERE ti.tileRun = 4
GROUP BY ti.tiMask
```

The spectrum join is directly from `ti.targetID`, not through nullable
`t.targetID`; otherwise the measured any-spectrum count (27,023) would not
exceed the matched `sdssTiledTargetAll` count (26,360). The reported target
totals use distinct target IDs, because spectrum and tiling joins can fan out.
This is a faithful query-structure reconstruction from the contemporaneous
run record, **not** the byte-identical SQL submitted to SkyServer. Its saved
SQL fingerprint is useful as a provenance label but does not recover the
missing statement or response body.

| `tiMask` | run targets | target-table matches | any spectrum | same-tile spectrum | same-tile science-primary | same-tile primary, `zWarning=0`, non-null `z` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 23 | 14 | 3 | 0 | 0 | 0 |
| 2 | 254 | 142 | 131 | 0 | 0 | 0 |
| 4 | 1,636 | 1,480 | 256 | 0 | 0 | 0 |
| 5 | 881 | 859 | 879 | 879 | 873 | 867 |
| 6 | 60 | 54 | 9 | 0 | 0 | 0 |
| 7 | 25,779 | 23,811 | 25,745 | 25,742 | 24,214 | 24,013 |
| **Total** | **28,633** | **26,360** | **27,023** | **26,621** | **25,087** | **24,880** |

The 2,273 target-table misses remain in the run frame with unknown selection
attributes; they are not failures. The 24,880 count is a same-tile primary
spectrum record with a clean warning field and non-null redshift, not an
attempt-complete denominator, an independently vetted redshift, or an
FP-usable distance. The first indexed run is a deterministic feasibility
example, not a representative sample of the 75 runs linked to the selected
cohort. No fraction from this one run is transferred to the CF4/SDSS-PV
likelihood.

The retained run-level SQL fingerprint is
`44a1c31c51b701c7f4262ee3d02a85a5bc0e484c71bd8fa168dc31870c14f969`; the
captured response fingerprint is
`8ac81d428ec50925e0c6b2524f7e6e02e78bc33569455f0b72500b4726bb8c6f`. No raw
catalogue rows or response body were saved.

## Decision

Accept a partial provenance bridge for 1,385 selected spectra whose DR14
science-primary `tile` exactly matches a historical SDSS-I/II tiling record.
Do not call this an eligibility denominator, redshift-success calibration,
SDSS-PV parent sample, FP-quality selection law, Tempel/CF4 linkage law, or
shared-group covariance calibration. The DR14-produced SDSS-PV sample has
mixed release/target histories, whereas these tables document only SDSS-I/II
tiling. No further exploratory query is warranted now. Reopen the parent
route only if an independently defined source-matched parent becomes
available with eligibility, attempt/outcome, FP-quality, and CF4/Tempel
linkage states. This is not permission for a bulk archive download or another
sampler run.

Q-GOAL: the selected-cohort bridge plus one whole-run aggregate clarifies a
single historical post-target-selection stage, but yields no z=0
map/posterior. The approved route remains joint latent-IC/evolved-state
inference, with the z=0 posterior first. MW/M31 are role-ambiguous and M33
unresolved; their observables must constrain those same roles on the same
NEW evolved LG field at `<=0.3 cMpc/h`, with native truth IDs used only for
calibration/evaluation.

Q-LEAN: two exact-cohort, bounded SQL rounds and one run-bounded aggregate
answered the narrow provenance/feasibility question. Stop exploratory
queries. The run-4 frame lacks pre-target eligibility and downstream FP/CF4
outcomes; its 2,273 target-table mismatches are unknown, not failures. It is
not representative of all linked runs and supplies no transferable
selection correction. No survey census, catalogue download, simulation, or
new gate is justified. R2 remains NO-GO.
