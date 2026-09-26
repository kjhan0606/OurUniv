# R2/5 — source distance-factor implementation bundle

Approved autonomous continuation, 2026-09-26. First delivery remains the
actual z=0 posterior, from the latent-IC joint present-state/history route.

This bundle acquires the small corrected SDSS FP catalogue, joins its PGC
identities to actual CF4 FP members, implements its selection-corrected
distance likelihood shape, and connects it to the existing N128 PM velocity
state with numerical derivative tests. No gravity run, seed selection,
N256 production, new TNG dependency or large mock archive is needed.

## Source contract and scope

- Howlett et al. 2022, https://arxiv.org/abs/2201.03112, sections 2.3 and 5:
  eta uses the **group** redshift. The prior is flat in eta over [-1.5,1.5].
  The per-galaxy selection normalizer is already included. Hence evaluating
  the supplied PDF shape gives likelihood ratios within that support,
  conditional on the published FP fit and its skew-normal approximation.
  Do not add another selection weight,
  BGc transform, or parameter-coordinate Jacobian.
- Official v1.1 catalogue and schema: https://zenodo.org/records/6824749.
  Preferred `logdist_corr` columns use group-richness-dependent FP fits.
  Mean/std are **not** skew-normal location/scale. The schema explicitly
  requires zero-point calibration at group level, not individual level.
- The implemented field connection is a cold, single-stream control using
  multiplicative redshifts, not a calibrated FoG/group-velocity model.
  Holdout is closed over the observed matched CF4/Tempel group graph. No fit or
  choice is based on its score. The published FP fit itself used the source
  sample: these split scores are NOT independent end-to-end heldout validation.
  The amplitude derivative is a numerical
  check, never a power-spectrum amplitude adjustment.
- Unknown shared FP/zero-point uncertainty, source selection beyond the
  published conditional model, overlap with 2M++ redshifts, and other CF4
  methods remain explicit. Do not multiply this control blindly into the
  2M++ score or call it the complete CF4 posterior.
- TF source assessment follows the FP source bridge; do not treat SDSS FP
  calibration as TF/6dFGSv calibration. Full SDSS mocks (10.6 GB) and randoms
  (284 MB) are not downloaded for this small implementation control.

Q-GOAL: replaces an ungrounded distance-mark proxy with source-defined,
field-dependent FP information in the R2 observation model. Q-LEAN: reuse
one existing PM state; two focused tests, one bounded allocation, no new
generic gates or repetitive group-residual fits.

MW/M31/M33: this R2 FP source does not resolve them. R3 candidates must be
identified from each NEW evolved field, with MW/M31 role ambiguity and
unresolved M33 retained. Their observed distances, velocities and masses
must constrain that same field. Catalogue identities here identify observed
measurements only, never seed/select generated LG components.

Execution: H200 typed GRES, 2 CPUs, 8 GiB (estimated peak <=6.5 GiB plus
margin), 15-minute limit. Unit checks and source/state connection in the same
allocation. Results are written once to `r2_sdss_fp_source_link_v1`.

## TF source assessment

Kourkchi et al. 2020, https://arxiv.org/html/2009.00733, sections III.2–III.4
and table 4: final TF distances already incorporate residual Malmquist,
colour/band, and Hubble-parameter regularizations. Passbands share linewidth
and inclination errors; treating them as independent measurements or applying
the residual-bias polynomial again is incorrect. A TF extension must preserve
the final-product correction history and shared calibration uncertainty.
The published FP skew-normal conditional cannot simply be copied to TF.

Job405986 stopped at the test launcher because the established JAX environment
does not contain pytest (no science calculation ran). The driver removed that
unnecessary test-runner dependency and the astropy dependency, reusing the
project's Galactic-to-supergalactic rotation. Job405988 is the corrected run.

Job405988 passed both focused tests, but the actual-field undamped distance
iteration failed the residual/support check. No rows were removed and no
scientific result was promoted. Job405990 uses 128 half-damped iterations,
unchanged observations/likelihood, plus explicit failure counts. Convergence
alone does not establish single-stream uniqueness or physical FoG calibration.

The 128-iteration run405990 gives 10,020 eligible source rows; no nonfinite
or out-of-box roots, but three residuals exceed 0.01 km/s (maximum0.378).
Retain the same tolerance and all rows; extend to512 iterations to verify
numerical convergence rather than weakening the check.

Job405991 with512 iterations leaves one slowly converging row at0.0136 km/s.
Stop increasing the iteration budget: implement a local Newton refinement
after the128-iteration warm start, using the actual row-wise redshift
derivative. The all-row residual check and field-gradient test are unchanged.

## Completed implementation result

H200 job **405997 COMPLETED/exit0**,75 seconds, batch MaxRSS1,777,476 KiB.
Both focused tests pass. Official34,059 source entries yield10,072 unique
native CF4 FP matches and10,020 eligible rows in5,450 CF4 groups (fixed
redshift-radius range15–180 cMpc/h). Group-graph closure promotes127 rows
to holdout; the eligible split is7,555/2,465. No object was dropped for
root failure: all10,020 converge, maximum residual1.66e-10 km/s.

The same-state source-factor derivative with respect to a diagnostic
velocity amplitude is-35.22357127; centred finite difference gives
-35.22356983. This verifies differentiability, not an amplitude fit or
an instruction to rescale the physical power spectrum. Log ratios versus
zero peculiar velocity are-9.7312/-5.1478 on the two splits; the underlying
PM phase is unconditional, so these are not reconstruction-quality claims.

Artifacts:
- `/gpfs/kjhan/CF4/z0_density/r2_sdss_fp_source_link_v1/result.json`
- `/gpfs/kjhan/CF4/z0_density/r2_sdss_fp_source_link_v1/source_link.npz`

Driver decision: accept the **source-distance implementation and same-state
numerical connection**. Do not promote it to a complete joint likelihood or
R2 posterior. Repeated group-redshift residual fitting and cold-root tuning
are closed. The next substantive bundle should marginalize source-group
distance/velocity and shared FP zero point, connecting the *conditional*
distance factor to the existing 2M++ redshift-bearing observation model
without counting redshifts twice. Retain selection and group-assignment
uncertainty; do not infer a whole-CF4 selection denominator from this subset.
TF/6dFGSv must have their own source contracts before inclusion. No new
large simulation, mock download, N256 production or LG identity claim is
justified by these numerical checks alone.
