# CF4TF mock source applicability — 2026-10-06

## Finding

Qin et al. describe 2,000 CF4TF mock catalogues made from 250 L-PICOLA
boxes, eight observers per box. The simulations are 1,800 Mpc/h with
2560^3 particles and a minimum halo mass of about 5e11 Msun/h. The mocks are
useful for CF4TF flow-estimator, cosmic-variance and selected-catalogue
measurement studies.

They do not expose the denominator needed to calibrate CF4 group inclusion or
redshift success. The published algorithm derives a *relative* sky
completeness from CF4TF/2M++ HEALPix counts, normalizes it to the mean of two
ALFALFA patches, caps it at one, then downsamples to the observed smoothed
CF4TF redshift distribution in each patch. The paper explicitly says this is
not an absolute completeness. Distance-ratio errors are drawn from observed
redshift-bin errors and centered on truth under the assumption that the data's
Malmquist correction is already adequate. Thus these mocks test a selected
CF4TF catalogue under the adopted procedure; they do not independently
recover its parent/sample-success law, the active 2M++ count operator, or the
CF4/Tempel group/member association and shared-redshift covariance.

The article says its mock catalogues are available on reasonable request to
the corresponding author; it provides no public catalogue/repository link.
The later combined-CF4 mock paper likewise says the CF4TF/6dFGSv/combined
simulated catalogues can be shared on request, while identifying the SDSS-PV
mock set as the publicly hosted component. The existing SDSS-PV mock ensemble
has already been evaluated in this project. The user has prohibited email;
no author was contacted, no account was created, and no files were requested
or downloaded.

## Driver disposition

Close this public-source branch as **not presently accessible for calibration**.
If the files become available through an authorized non-email route, their
potential role is limited to a CF4TF selected-mark/flow diagnostic until the
parent selection and group-ownership fields are demonstrated. Do not transfer
their mock residuals or covariance to the full heterogeneous CF4 likelihood.

R2 remains **NO-GO** for a calibrated full posterior. The saved conditional
MAP stop was inspected read-only in its runner and result JSON. The runner
explicitly set `maxiter=4` and `maxfun=8`; SciPy reports `TOTAL NO. OF
ITERATIONS REACHED LIMIT` after four iterations and six finite exact
value/gradient evaluations. The best objective falls by 29,860.855 (0.362%),
but its gradient infinity norm is still 13,524.69, far above the predeclared
`1e-4` convergence condition. No exception, nonfinite evaluation or failed
checkpoint is present. Therefore the observed stop is explained by the
deliberately small iteration cap; this does not establish optimizer adequacy,
target correctness, a converged MAP or a field posterior. Do not replay the
same tiny budget. Any continuation needs a justified scalable optimization
design and fixed stopping/resource criteria; the CF4 selection limits remain
unchanged. Heldout outcomes were not read.

**Q-GOAL:** a CF4TF-only mock could eventually validate one selected distance-
mark component of the same z=0 inference; it cannot deliver the 1.5 cMpc/h
field or the LG result. MW/M31 roles remain ambiguous and M33 unresolved;
their observables must ultimately constrain those roles on the same NEW
evolved field at `<=0.3 cMpc/h`, with native truth IDs used only for
calibration/evaluation.

**Q-LEAN:** one paper/method/data-availability check closes this branch. No
repeated repository census, contact attempt, new mock generation, field fit,
or simulation is justified by this source alone.

## Sources

- Qin et al., [Cosmic Flow Measurement and Mock Sampling Algorithm of
  Cosmicflows-4 Tully-Fisher Catalogue](https://arxiv.org/html/2109.14808v1),
  §§III.1–III.2 and Conclusion.
- Howlett et al., [Evaluating bulk flow estimators for CosmicFlows–4
  measurements](https://academic.oup.com/mnras/article/526/2/3051/7296158),
  §2 and Data Availability.
