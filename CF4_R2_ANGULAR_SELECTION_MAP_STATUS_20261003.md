# R2 angular-selection input status — 2026-10-03

Status: the missing-file blocker is closed; selection calibration and the
R2 posterior are not.

## Source and completed checks

The project already has the two full-pixel ARES 2M++ completeness maps in the
pinned repository snapshot `6cf608ed8c70474c15c8ca76cce675742e291c71`:

- `completeness_11_5.fits.gz`, SHA256
  `1eb561b93d6cc95095350d20ce61131a1c44afcdf385dec033817f0dc5a87ace`.
- `completeness_12_5.fits.gz`, SHA256
  `20af269339c04eb002556afe88ef3ade19f43c08c6a146e6a17fdd29af4f80a9`.

Both are HEALPix NSIDE=512 RING maps, with values across empty-sky pixels. The
source-bound V4 technical-input result
`/gpfs/kjhan/CF4/kf_design/twompp_disjoint_tracer_v4/pilot/result.json`
passes its declared pointwise metadata gate for 69,160 catalogue rows. Median
absolute map/catalogue differences are zero; p95 differences are 0.00308 and
0.00500 for the 11.5 and 12.5 maps. This is a technical provenance/coordinate
check, not field inference. The 2M++ paper documents the two magnitude-limited
angular completeness maps, and the ARES/BORG analysis uses those maps in its
selection response ([Lavaux & Hudson 2011](https://academic.oup.com/mnras/article/416/4/2840/975884),
[Lavaux & Jasche 2016](https://academic.oup.com/mnras/article/455/3/3169/2892571)).

The existing common-cosmology N128 selection integral already applies the
maps over 3 cMpc/h voxels: 29,100 observed count keys have positive integrated
exposure. Its recorded order-six integral has not passed an angular
quadrature-convergence check and is explicitly not ready for a quantitative
joint likelihood.

## What remains unresolved

The raw catalogue does not contain the full maps; its `c11.5/c12.5` columns
are pointwise redshift-incompleteness marks. Across the point-mark audit,
442 pointwise map/catalogue discrepancies remain. One faint-sample object
(2M++ recno 67100) has map support zero but positive catalogue mark 0.5 and
positive *cell-integrated* exposure 0.09919. A separate locality diagnostic
found only 98/442 disagreements compatible with a one-pixel neighbour;
automatic nearest-neighbour substitution or smoothing is not justified. The
point-process route therefore remains NO-GO under the literal map, while the
coarsened count cells retain positive integrated support. The eventual
shared-latent conditional must state which observation representation it uses
and preserve this distinction.

This closes “no angular map at empty-sky locations” as an availability claim,
not “selection is calibrated.” Radial/K/evolution/extinction and luminosity
terms, angular integration convergence, source/map discrepancy treatment,
population bias and count-model discrepancy, group covariance, and the shared
count/mark conditional remain unresolved. No data were changed and no new
calculation or posterior was run.

Q-GOAL: removes a stale data-availability blocker for the same CF4-conditioned
z=0 field target; it does not establish that field.

Q-LEAN: reuse the already pinned maps and N128 integral; do not redownload,
rebuild a proxy mask, or repeat the full integral merely to reconfirm presence.

MW/M31 remain role-ambiguous and M33 unresolved. Their observables must
ultimately constrain those same latent roles in the NEW evolved field at LG
resolution `<=0.3 cMpc/h`; native truth IDs remain calibration/evaluation-only.
