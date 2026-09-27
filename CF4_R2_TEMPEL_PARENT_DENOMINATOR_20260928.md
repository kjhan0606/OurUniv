# R2 Tempel parent denominator/source check — 2026-09-28

## Status

Slurm 407002 completed successfully (5m14s; MaxRSS 306,948 KiB). This is a
source-identity and denominator-mechanics result only. It is **not** a
calibrated parent-selection law, a field fit, an R2 posterior, or a density /
velocity map. No gravity run or simulation was launched.

## What the source join establishes

- Tempel tables contain 584,449 member rows, 88,662 multi-member groups, and
  297,204 individually keyed `GroupID=0` singleton parents.
- The full SDSS PV v1.1 table has 34,059 rows. Exact photometric `objID` joins
  match 33,641 rows to Tempel; 418 have no Tempel counterpart. Among exact
  matches there are zero Tempel `GroupID` or richness conflicts.
- The current 10,020 R2-eligible FP rows have 9,945 exact Tempel links and 75
  source-ungrouped rows.
- An independent Tempel-member↔2M++ position/redshift association (within
  10 arcsec and 300 km/s, reciprocal one-to-one only; no FP identity/flag used)
  yields 17,714 member pairs. Of 13,246 Tempel parents with a linked member,
  2,720 map to one local 2M++ GID and 42 to multiple GIDs; 10,484 have no
  linked GID. This is an observed catalogue bridge, not a physical isolation
  label or a latent field-selection probability.

The JSON's two displayed row counts are wrong because the driver used
`len(dict)` (number of columns): it reports 7 Tempel rows and 8 FP rows.
The parser's row-count gates and matching totals establish 584,449 and 34,059
respectively. The calculation is not rerun for this reporting-only defect;
the script is corrected for future runs. The result JSON is preserved
unchanged so the original run remains auditable.

## Footprint finding and interpretation

The random-neighbour p99 radius (186.4 arcsec) classifies 98.998% of native
`in_mask=1` rows as inside, but also classifies 43.390% of native
`in_mask=0` rows as inside. Thus the random catalogue's nearest-neighbour
support is **rejected as a mask classifier**. The reported 68,103
“common-z-window” parent count and every parent rate derived from that proxy
are contaminated and must not enter an inclusion/propensity law.

The source contract is more specific than “no SDSS mask exists.” Howlett et al.
state that a complete angular mask for their mixed DR8+later-release PV data
was not publicly available; instead they flagged rows against a pre-existing
NYU-VAGC DR7 MANGLE mask, retaining all PV rows and exposing `in_mask` in the
catalogue. That DR7 mask can define the published `in_mask` convention, but it
does not describe the full DR8–DR14 targeting/tiling/bright-star selection.
The cited NYU-VAGC `lss_combmask.dr72.ply` endpoint redirects to HTTPS but
fails TLS certificate validation from this node; no verification bypass was
used and no replacement mask was inferred from the random catalogue. The
public KIAS 0.025-degree raster is a different, supplemented product and is
not silently substituted for the exact polygon mask.

The published random catalogue was used only as a failed diagnostic, not as
an exposure law. A bounded exact-name search found no trusted mirror of the
NYU `lss_combmask.dr72.ply`; the original host redirects HTTP to HTTPS and
strict TLS validation fails. No certificate bypass or substitute mask was
used. No random-neighbour threshold/pixel-size sweep was run.

## Split and scientific limits

Six Tempel parents contain linked members assigned to both v5 train and
heldout roles across the full source graph. Five of those six fall inside the
now-withdrawn proxy-mask/redshift window; the sixth is outside that proxy
window. Therefore v5 is not closed for any model that includes this expanded
Tempel-parent graph. Existing v5 results remain valid only for their original
narrower graph; do not relabel them as leakage-free Tempel holdout results.
The run examined heldout role/selection counts descriptively to identify this
mixing, but did not score heldout distance values or field predictions.

The source denominator is not the missing `S_group(d,F)` law. It does not
identify true-distance inclusion, selected-group roles, group/member
covariance, or tracer bias. The 418 full-source and 75 eligible rows absent
from Tempel are outside this source grouping, not proof of physical isolation
or non-detection. `GroupID=0` denotes individually keyed singleton catalogue
entries; multiple linked GIDs remain ambiguous.

Q-GOAL: this adds observed-parent identity and zero/one/multiple association
information needed for the CF4/LG present-state likelihood, but contains no
new-field constraint. MW and M31 remain ambiguous latent roles; M33 remains
unresolved. On every generated state, those same-state observables must be
evaluated against the same evolved field. Native identities may label
calibration/evaluation only, never seed or select a field candidate.

Q-LEAN: one bounded source-join/control was proportionate. The failed mask
proxy is sufficient to reject that classifier; further threshold sweeps,
large mocks, gravity runs, or sampler work are not justified by this result.

## Next R2 action

The exact NYU polygon artifact was not recovered, so close this denominator
route at the descriptive source-graph result rather than manufacturing a
common-footprint rate. This does not close R2. Independently, v5 is not safe
for any future likelihood that uses the expanded Tempel/2M++ graph; the next
bounded step constructs a new component-closed split from the frozen source
identities, buffering whole mixed train/heldout components and their count
keys. Even an exact DR7-mask reproduction would not calibrate the full
DR8–DR14 selection or complete the R2 posterior; the selected-group,
true-distance selection, tracer/bias and shared-covariance terms still need a
defensible joint observation model.

No email was sent.

## Sources

- Howlett et al. (2022), §2.5–2.6, [SDSS peculiar velocity catalogue](https://doi.org/10.1093/mnras/stac1681).
- [Official SDSS-PV v1.1 data/random/mock record](https://zenodo.org/records/6824749).
- Tempel et al. (2017), [catalogue and ReadMe](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/602/A100).
- The SDSS-PV article cites the [NYU-VAGC DR7 source](http://sdss.physics.nyu.edu/vagc/) and identifies the DR7 MANGLE polygons used for `in_mask`.
