# R2/5 — source membership and within-cell observation connection

User requests the next implementation bundle. Reuse the10,020 FP rows and
the established group-distance integrator; do not repeat provisional-sigma
sweeps or start a gravity run. Acquire only the needed columns of the official
Tempel2017 SDSS galaxy/group tables, then join the SDSS FP objects by exact
photometric objID. Preserve integer IDs without floating-point conversion.
Retain full velocity membership of the linked groups, group dispersions and
measurement errors as reusable inputs. Check membership/richness identities
before assuming these data implement the FP source grouping.

Source: https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/A+A/602/A100?format=html
The catalogue contains584,449 galaxies and88,662 multi-member groups. Raw
CMB redshifts/IDs/dispersion are used; published distances are NOT imported
because their cosmology differs from our fixed basis. Photometric IDs are
recommended by the corrected SDSS FP v1.1 source.

Important inference limit: a published group dispersion estimated from those
same member redshifts is not independent calibration data. Save raw members
so their mean/scatter can enter a single joint law; do not multiply a
dispersion-derived prior and the same velocity sample as independent evidence.
Singletons and source-group ambiguities must remain unresolved, not receive
invented measured dispersions. A source membership bridge does not certify
Gaussian virial velocities, the COM discrepancy, or group selection.

Implement the missing conditional-position factor algebra:
`p(points|counts,F) = product_cells n_c! product_i lambda(x_i,F)/Lambda_c(F)`
for unordered within-cell configurations. Combined with Poisson counts this
recovers the corresponding point-process density. Lambda must integrate the
same intensity used at the points. Test normalization/factorials and cell
amplitude cancellation; shape changes need not cancel. Evaluate the literal
existing selection without changing its known zero-support point: no floor,
mask repair, deletion or claim that positive voxel exposure fixes point support.

Q-GOAL: replace arbitrary group inputs with source-linked data and complete
the mathematical count/position interface needed by actual z=0 inference.
Q-LEAN: two bounded downloads, streaming rows, existing tests plus one targeted
test, one small Slurm allocation; no generic gates, new simulation or sampler.
H200/H100/A100 checked, typed H200 selected;1 CPU,4GiB (<=3.3GiB estimate
plus20%),15-minute cap, each remote table bounded below100MB.

MW/M31/M33 remain the R3 task: candidates must come from each NEW evolved
state, with MW/M31 role ambiguity and unresolved M33 retained, and observed
positions/masses/velocities constraining that SAME field. Observed catalogue
IDs in this bundle must never seed/select generated LG identities.

## Execution and driver assessment

Slurm406012 completed in2m11s; batch MaxRSS72,364KiB. Four observation
tests passed. Results:
`/gpfs/kjhan/CF4/z0_density/r2_tempel_source_connection_v1/result.json`
and `source_group_members.npz` in the same directory.

- Official tables contain the expected88,662 groups and584,449 members.
- Of10,020 FP rows,9,945 match exact photometric IDs;75 unmatched PGCs are
  explicitly retained in the result. No group-ID or richness disagreement
  occurs among matched rows. This is a partial, not complete, source bridge.
- Full membership for4,422 source groups totals30,113 galaxies;2,399 FP
  rows are source singletons. Singleton status and unmatched IDs are separate
  categories, not mutually exclusive or a basis for dropping observations.
- Published radial dispersion median156.455 km/s,10–90% range36.722–355.893.
  Recomputed/published median ratio1.000000294. This confirms the source
  definition numerically; it is NOT independent covariance calibration.
- SDSS FP source-group versus Tempel catalogue cz absolute differences are
  median9.042,90th percentile15.754,max23.150 km/s. Preserve each source's
  convention; do not replace the FP eta numerator with the Tempel mean.
- Count/conditional-position factorization passes its algebra test. The
  actual literal selection still gives zero support to recno67100; its
  conditional log density is undefined/rejected, not repaired by a floor.
  All57,238 count points remain preserved.

Accept the data connection and conditional-factor implementation. Do not
promote a posterior, claim calibrated group covariance/selected-group prior,
or launch another gravity run. No new gravity calculation occurred.

Next science work: resolve the75 unmatched identities from source evidence;
specify a single source-aware group/member and selection law, using these raw
members once rather than multiplying their published dispersion as independent
evidence; establish a justified point-selection treatment for recno67100.
Tempel's FoF/refinement selects velocity membership (including escape-velocity
clipping), and small-group dispersions are uncertain. An iid Gaussian scatter
prior inferred from this selected sample is therefore not automatically valid.
Reference: https://arxiv.org/html/1704.04477 . The unresolved selected-group
distance prior and COM/model discrepancy must remain explicit in any bounded
inference pilot; source-data acquisition alone does not solve them.
