# 2M++-like MDPL2 mocks: R2 applicability decision — 2026-10-06

## Candidate and scope

Hollinger & Hudson (2024) is a newly identified physically based source for a
narrow part of the R2 observation-law problem. Their mock 2M++ catalogues use
the z=0 MDPL2 simulation and the SAG/SAGE semi-analytic galaxy catalogues,
assign K-band luminosities by abundance matching to the 2M++ luminosity
function, impose the 2M++ magnitude/depth and Zone-of-Avoidance conditions,
and use incompleteness weights. The study constructs 15 non-overlapping
200 h^-1 Mpc spheres and compares a 4 h^-1 Mpc-smoothed tracer density field
with the simulated peculiar velocities. It reports an `fσ8` estimate biased
high by a factor `1.04 ± 0.01` for that analysis; this is not a correction
for our Poisson field likelihood.

The underlying MDPL2 halo and SAG/SAGE catalogues are publicly queryable
through [COSMOSIM's MDPL2 tables](https://www.cosmosim.org/metadata/mdpl2/)
and the article states that its underlying data are available from COSMOSIM.
The service documentation says full access requires registration and an API
token. A separate [MultiDark-Galaxies release](https://skiesanduniverses.org/Products/MockCatalogues/MDGALAXIES/)
lists SAG/SAGE/Galacticus tables above a stated stellar-mass completeness cut
for 40 snapshots; an HTTP HEAD request from the current syntax runtime
returned403, so availability from this runtime is not established. No data
were downloaded, no registration was attempted, and no token was requested or
used.

## Driver applicability decision

This is a **conditional count/tracer-bias validation candidate**, not a
drop-in R2 calibration and not authority to insert the published 4% number.
It can potentially test how a physically evolved galaxy population responds
to a 2M++-like K-band selection and mask, with halo/galaxy truth available at
z=0. It does not supply:

- the CF4 distance-indicator parent samples and their success/failure or
  survival probabilities;
- recovered CF4/Tempel group identities under the active association process;
- a joint covariance for group redshift, group distance and member velocities;
- an already-run realization of our active v6/N256 observation operator; or
- a demonstrated N256-resolution matter-truth mesh suitable for field-level
  coverage tests.

The paper's target is a smoothed velocity-versus-density reconstruction,
not our latent-IC Poisson posterior. It can inform a bounded 2M++ count-law
mock only after checking the actual query tables, columns, query/result size,
redshift completeness prescription and access conditions. It cannot close
the CF4 group-inclusion/shared-mark-covariance branch. Published 2M++
luminosity-function or bias inputs are also not independent if reused as
priors after being estimated from the same 2M++ catalogue; that dependence
must remain explicit.

## Next bounded action and stop conditions

Before any retrieval, obtain authorized COSMOSIM access (registration and
token) or verify an accessible public mirror, then inspect only metadata and
row counts for the z=0 MDPL2 SAG/SAGE and halo products. Do not submit a large
query or download catalogues until the minimum columns, storage estimate and
whether a suitable matter-truth field exists are known. If those checks pass,
the first numerical use is one 2M++-selection-matched, data-split mock of the
existing count operator with the nonuniform expected-count integral included;
it must be labeled count/tracer-bias validation only. Stop if the truth field
or selection variables needed for that comparison are absent. No new model
correction, field fit, heldout score, posterior sampling or gravity run is
authorized by this source audit alone.

## Project-goal limits

**Q-GOAL:** the candidate could validate one component of the actual CF4-
conditioned z=0 inference—the 2M++ count/tracer relation—but produces no
CF4-conditioned density/velocity map and does not yet demonstrate the active
1.5 cMpc/h target. MW/M31 remain ambiguous latent roles and M33 unresolved;
their observables must ultimately constrain those same roles on the same NEW
evolved LG field at `<=0.3 cMpc/h`, with native truth IDs restricted to
calibration/evaluation.

**Q-LEAN:** one source/access/field-truth eligibility check precedes any
calculation. No repeat SDSS-PV mocks, no repeated source census, no bias
coefficient transfer, no sampler extension, and no simulation are warranted
here. R2 remains NO-GO until the conditional observation law is defensible and
a stationary field posterior, uncertainty, untouched heldout prediction and
the z=0 map are delivered.

## Sources

- Hollinger & Hudson (2024), [arXiv:2312.03904](https://arxiv.org/abs/2312.03904),
  especially §§2–3 and Data Availability Statement.
- [COSMOSIM MDPL2 metadata and table list](https://www.cosmosim.org/metadata/mdpl2/)
  and [TAP authentication guide](https://www.cosmosim.org/cms/tech-docs/access-via-tap/).
- [MultiDark-Galaxies public release](https://skiesanduniverses.org/Products/MockCatalogues/MDGALAXIES/).
