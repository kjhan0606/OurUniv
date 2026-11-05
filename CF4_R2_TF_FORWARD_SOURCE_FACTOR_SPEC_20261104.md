# R2 individual-TF forward-factor feasibility — 2026-11-04

## Decision

**Conditional GO for this source/factor specification; NO-GO for changing the
active target or launching a fit.** A selection-normalized individual-TF
likelihood is a plausible alternative observation component. It is not yet a
calibrated substitute for the active SDSS-PV FP term, and adding it to the
current count+FP target would leave the FP-selection limitation intact.

Keep the active v6 count+raw-FP target unchanged. If this route is advanced,
evaluate it as a separately named count+TF target (replacing, not silently
stacking on top of, the current FP mark component). That is a material target
revision and needs its own factor and calibration decision. This note makes no
posterior, held-out, field, or selection-calibration claim.

## Training-only source reconciliation

A read-only key/role join used the current v6 split and only role-0 training
CF4 groups. It did not read any v6 held-out count, mark, velocity, or TF
outcome values, and did not use the older TF holdout flags or moduli.

- The current v6 split has **16,323 training CF4 groups**. Their member table
  contains25,005 unique PGC rows; the public Kourkchi et al. 2020 individual
  TF table1 matches **7,531** of those PGCs.
- Those training matches have 5,448 raw/corrected SDSS-i entries, 3,666 raw
  W1 entries, and parseable inclination-corrected HI linewidth, HI-flux proxy
  (`m21`), and heliocentric HI velocity. The CF4 individual-galaxy table gives
  finite RA, Dec, and CMB-frame `Vcmb` for all7,531 matched training PGCs.
- The7,531 measurements occupy6,660 current-v6 training CF4 groups. **419
  groups contain multiple matched TF galaxies** (maximum48); a product of
  independent single-galaxy velocity factors would therefore need an explicit
  shared group/distance/velocity covariance treatment.
- The existing exact CF4–2M++ crossmatch has3,089 secure linked TF PGCs among
  these training rows. Of these,2,860 map to v6 training count points, none
  maps to the v6 held-out or buffer point roles, and229 are outside the frozen
  point-role array. The other4,442 have no secure link. These are linkage
  facts, not an estimate of TF inclusion probabilities; unmatched does not
  mean unobserved or failed.

The local source schema distinguishes `imag`, `icmag`, and `i*mag`, as well
as raw/corrected W1 magnitudes. They are not interchangeable. `logWmxi` is the
inclination-corrected width; `m21` is computed from HI flux. `Rsrc` records the
attenuation-correction source, **not the HI survey**; `QSflag`/`QWflag` are
photometry grades, not targeting or failed-observation states. The public
individual table does not provide an object-level HI source/attempt/failure
crosswalk.

## Candidate factor and ownership

For training TF object `i`, let `X_i` contain its conditioned observed
position/redshift, linewidth, flux and genuinely available survey/quality
variables. A forward conditional mark factor has the form

\[
p(m_i\mid X_i,F,S_i=1,\theta_{TF})
=\frac{S_i(m_i,X_i)\,q_{TF}(m_i\mid X_i,F,\theta_{TF})}
       {\int S_i(m',X_i)\,q_{TF}(m'\mid X_i,F,\theta_{TF})\,dm'} .
\]

The TFR mean and scatter belong in `q_TF`; its predicted distance/velocity
must come from the **same** latent-IC evolved field used by the count model.
For a TF galaxy whose redshift-space point is also in the 2M++ catalogue,
the count process owns that point's observed occurrence/redshift once; the TF
factor conditions on its observed redshift and sky position and adds only the
photometric mark. Do not multiply a second redshift likelihood. Different
group/member redshifts require an explicit shared measurement model.

The TF-only `DMtf` Gaussian group summary is not the proposed datum: use the
raw individual observations once, with explicit ownership of passbands,
members, zero point, slope/curvature, intrinsic scatter, measurement errors,
and group-level shared motions. Do not import full-sample TFR fits as
independent priors when they were estimated using these galaxies. Do not treat
the 2M++ crossmatch subset as the only TF cohort without modeling the extra
matching selection.

## What the literature does—and does not—supply

Kourkchi et al. selected the published CF4 TF sample through morphology,
inclination, HI-quality, anomaly, photometry-quality, and velocity criteria.
Its HI measurements combine ADHI, ALFALFA, Springob/Cornell and pre-digital
sources with different sky coverage; the public table lacks a per-object HI
survey-source field. [Kourkchi et al. 2020, §§II.1–II.2](https://arxiv.org/html/2009.00733)

Boubel et al. provide a useful forward conditional `P(m | width, flux,
redshift, sky, TF parameters, velocity model)` with the magnitude-dependent
selection inside the normalization. Their derivation lets selection factors
independent of magnitude cancel, but explicitly excludes sky-position and
true-distance dependence from its selection model and warns this can fail for
heterogeneous samples. Their velocity predictor is a Carrick/2M++ redshift-
space model, not the project's nonlinear evolved latent field. [Boubel et al.
2024, §§2 and4](https://arxiv.org/html/2301.12648v2)

The later CF4 H0 analysis uses an empirical magnitude-completeness curve
derived from the selected magnitude counts against an assumed Euclidean
unselected count law. Its parameter-variation check is for an H0 estimate;
it does not establish selection calibration or coverage for a high-dimensional
field posterior, nor automatically transfer to this current-v6 training
cohort. [Boubel et al. 2024, §2.1](https://arxiv.org/html/2408.03660v1#S2.SS1)

Thus conditioning can remove an absolute selection factor depending only on
already-conditioned variables. It does **not** remove magnitude-dependent
selection unless the normalized `S_i` is supported; nor does it establish
that group/method availability, sky-dependent HI capture, quality selection,
or latent-distance/environment dependence is ignorable. The available table
alone cannot prove those assumptions.

## Minimum unresolved items before any implementation

1. Bind the exact magnitude column and linewidth/error convention to the
   paper's fitted observables; preserve the distinct raw, corrected, and
   dust-corrected variants.
2. Specify a source-/sky-aware selection law or explicitly bound the transfer
   uncertainty. Do not misinterpret `Rsrc` or image-quality grades as HI
   survey provenance/attempt outcomes.
3. Define the field-to-observed-redshift mapping. The Boubel predictor cannot
   simply be evaluated at `cz/H0` as though that were real-space position;
   either derive a compatible kernel for the evolved field or justify and
   test the approximation.
4. Handle the419 multi-member groups with shared distance/velocity and
   source-calibration covariance; singleton-only filtering is not an
   automatically ignorable shortcut.
5. Keep the current v6 held-out split protected and recheck all datum/identity
   ownership against the current graph before any target revision. The
   September group-level TF/FP disjointness result is not by itself a
   current-v6 independence or leakage certificate.

Until these are resolved, do not implement a TF factor, run another sampler,
or promote a field. Continue the separate MC-error/resource-contract work for
the explicitly conditional v6 target; any resulting field must be labeled
conditional on the frozen count+selected-FP observation model, not a
full-survey calibrated CF4 posterior.

**Q-GOAL:** a correctly factored TF distance/velocity mark could constrain
the same first z=0 field, but this feasibility check produces no posterior,
uncertainty, or held-out prediction. MW/M31 remain role-ambiguous and M33
unresolved. Their observables must ultimately constrain those same roles on
the same NEW evolved LG field at `<=0.3 cMpc/h`; truth identities remain
calibration/evaluation-only.

**Q-LEAN:** this is one current-v6 training-only join and one factor decision.
No new catalogue sweep, simulation, sampler screen, held-out read, or
instrumentation campaign is justified.
