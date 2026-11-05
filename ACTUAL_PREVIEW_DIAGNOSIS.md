# Actual CF4 + 2M++ preview: bounded diagnosis

2026-09-07. User requested diagnosis, not a new inference. Bundle A remains
active; no Bundle B launch. Preserve the failed fit and its original inputs.

## Evidence and disposition

Actual fit333872 and aggregation333990 completed. The result is
`NO_GO_SAMPLER_NOT_VALIDATED`, max Rhat1.32510, min bulk ESS11.0578,
with zero sampling divergences. This is a model-stress product, not a
calibrated present-day density/velocity posterior or an LG reconstruction.

Saved-chain diagnostic334240 (source9a23c0b) completed in36s, exit0,
peak RSS958896 KiB. It used64 predetermined retained fields from each of
four chains, with no new fit, mock, chain repair or held-out retuning.
Result: `/gpfs/kjhan/CF4/z0_density/actual_preview_diagnostic_v1/result.json`.

## 1. Confirmed imported magnitude-convention mismatch

`src/cf4_twompp_disjoint_tracer_pilot.py:distance_and_absolute_magnitude`
uses Astropy physical-Mpc distance modulus to construct absolute magnitudes.
`src/cf4_twompp_joint_information_budget_pilot_v1.py:_cosmology_distance_table`
likewise returns physical-Mpc luminosity distances to `schechter_fraction`.
The numerical absolute-bin edges and Schechter Mstar were imported unchanged
from the 2M++/ARES setup: edges -25 to -21, Mstar=-23.28, h=.6711.

The primary ARES source at `/gpfs/kjhan/CF4/software/ares-6cf608ed` demonstrates
the different convention. In `libLSS/data/schechter_completeness.hpp`,
`buildCompletenessFromSchechterFunction` converts d from Mpc/h to Mpc only
for the redshift lookup (`comph2com(d)`), then passes the original d into
`computeSchechterCompleteness`. The latter applies `5*log10(d_lum)+25`;
`libLSS/physics/cosmo.hpp:d2dlum` is simply `(1+z)*d`.

Thus its numerical magnitude is M_h = M_phys - 5 log10(h), about
M_phys +0.866 mag at h=.6711. Our unchanged bin edges select different
physical populations; the unchanged LF Mstar is also in the wrong convention.
Catalog and selection both using physical Mpc does NOT resolve the mismatch
with those imported constants. Either use the Mpc/h magnitude convention
throughout, or transform all edges and Mstar consistently to physical units.
Do not change only one of catalog binning and selection.

This is a confirmed implementation/input-model mismatch. Its quantitative
contribution to the observed residuals and convergence failure has not yet
been isolated. No corrected artifact has been generated in this diagnosis.
The source population definitions, cosmology and empirical tracer parameters
are described in [Lavaux & Jasche, Unmasking the Masked Universe](https://arxiv.org/html/1509.05040),
sections2,4,5.1 and Table1. The code's bright/faint ordering of Table1 values
is correctly reordered; a table-order swap is not the identified bug.

## 2. H0 failure follows field mixing, not conditional noise alone

For each saved field, compute the exact Gaussian conditional of the three
bulk components and H0 offset using the existing likelihood and priors.
The same-objective quadratic identity and held-out exclusion checks pass.

- H0 conditional SD:0.08345 km/s/Mpc.
- Correlation of sampled H0 with its field-dependent conditional mean:.96285.
- Across-field conditional-mean variance / conditional-noise variance:14.20.
- Chain-mean H0 offsets:-2.004,-1.659,-2.179,-2.112 km/s/Mpc.
- Field-dependent H0 conditional-mean Rhat1.3628, bulk ESS10.51 on the
  diagnostic subsample; use as supporting evidence, not a full-chain replacement.
- Whitened conditional offsets: means [.052,.057,-.168,.008],
  SDs [1.033,1.005,.951,1.047] for bulk x/y/z and H0.

These results support a slow field/nuisance direction. Resampling H0 while
holding the saved fields fixed cannot certify or repair field-marginal mixing.
An exact marginalization of this four-parameter Gaussian block during field
inference, with conditional reconstruction afterward, is a reasonable proposed
sampler adjustment, not yet implemented or proven sufficient.

## 3. Count shifts are partly normalization, with remaining selection risks

The model normalizes rho**bias by its full-box mean. Removing that log moment
from the fitted log-rate coefficient gives median effective shifts
[.921,-.159,.100,.763,-.187,-.298] across the six populations.
The two brightest populations still imply factors about2.5 and2.1 relative
to the corresponding bare prior coefficients. This explains part, not all,
of the large fitted normalization shifts; it does not prove correct bias or
selection. These are unconverged-chain descriptive estimates.

The catalog explicitly removes CF4-matched candidate objects, whereas the
raw selection calculation uses the original angular completeness and
Schechter radial fraction without an explicit CF4-removal selection factor.
Those removals change the tracer population. Their spatial/luminosity effect
remains unquantified; disjoint objects alone do not establish independence
or justify transferring full-catalog empirical density/bias priors unchanged.

Radially binned held-out residuals have population-dependent structure, e.g.
population0 at90-120 cMpc/h has533 observed versus457.7 predicted galaxies;
population3 in the same shell has1043 versus1151.3. These correlated,
unconverged predictive summaries do not isolate a cause or supply significance
levels. No tuning on these held-out counts is authorized by this diagnosis.

## Focused next proposal, not executed

1. Correct and explicitly bind magnitude/LF/population conventions; reconcile
   the retained-sample selection and transferred tracer priors. Preserve old
   products and do not force agreement by matching observed totals.
2. Address the field/H0 coupling in the sampler, reusing the Gaussian-block
   algebra and minimal targeted tests rather than a new validation framework.
3. Obtain authorization for one corrected actual-data refit; inspect mixing
   and predictive fields before considering scientific promotion or Bundle B.

No claim that these changes guarantee success, no further prior-family/mock
repair series, no resolution promotion, and no GPFS/filesystem investigation.
