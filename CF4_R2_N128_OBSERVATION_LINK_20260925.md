# R2 native N128 observation/PM link — 2026-09-25

## Connected inputs and result

The actual 2M++ row partition and 1.5 cMpc/h ARES selection already existed
in `bundle_c_v1/native_data_v2` and `selection_1p5_v2`. The earlier statement
that only N32 selection existed was wrong. Slurm 404203 summed the native
sparse counts and **volume-averaged**, rather than interpolated, the
six-population/six-shell selection to N128 (3 cMpc/h). It exactly retains
33,376 selected galaxies: 24,998 train and 8,378 held out, in 29,106
occupied population/cells. All occupied cells have positive selection.
The saved 11,504 disjoint parent marks give Beta survival calibration;
posterior-mean survival ranges 0.345–0.829 over 36 population/shell bins.
Its uncertainty remains a nuisance, not a fixed exact correction.

Slurm 404204 evolved one **unconditional** LCDM PMWD realization with
2,097,152 particles in the 384 cMpc/h box and deposited an N128 density and
mean-velocity state. The PM field has 5,769 exactly empty CIC cells; the
tracer response was changed to use `rho**bias` at the fixed published biases
instead of `exp(bias*log(rho))`, without inventing a density floor. This is
a genuinely evolved N128 PM state, **not** CF4/2M++ conditioned or an
independent higher-fidelity accuracy reference.

Slurm 404205 applied N128 selection/survival mean and six-population spherical
RSD/FoG response to that PM density/velocity. With the formerly arbitrary
diffuse component set to **zero**, the 29,106 occupied cells had zero
nonfinite and zero zero-intensity predictions; the minimum occupied intensity
was `8.05e-5` at the expressly placeholder unselected rate of one per cell.
The 3% comparison also had zero failures. These data do not require or
estimate a 3% component. **Withdraw the fixed 3% choice.** A future state
that predicts zero intensity for a positive observed count has zero Poisson
probability (or needs a separately justified response model), never a finite
score supplied by the JAX numerical log guard.

Slurm 404208 interpolated the **same** PM velocity to all 19,313 actual CF4
radial rows (15,346 train, 3,967 held out). Thus both observation channels
now read the same N128 evolved state. Its phase is random, so velocity
residuals are not a failed CF4 fit or useful posterior predictive score.
No joint likelihood or IC gradient was claimed.

Machine outputs are in `/gpfs/kjhan/CF4/z0_density/` under
`r2_native_128_observations_v1`, `r2_pm128_unconditional_v1`,
`r2_pm128_observed_support_v1`, and `r2_pm128_cf4_velocity_link_v1`.

## Calibration disposition and next science step

The selection survival probabilities have an independent calibration sample.
The 3% diffuse fraction does **not**: those marks identify survivor status,
not unclustered contaminants. In the linear regime the response
`(1-f)rho**b + f` has contrast proportional to `(1-f)b`; galaxy clustering
largely confounds `f` and `b`. Fitting `f` on one unconditional PM sky and the
actual galaxy sky would absorb phase mismatch. Therefore a physical `f`
calibration from these inputs is **not identified**; setting `f=0` removes an
unsupported parameter but does not prove true contamination is zero.
Published bias values remain priors at a different voxel scale, not a
calibrated N128 bias. FoG/model discrepancy and selection errors remain.

The N128 PM control uses the archived R1 cosmology (`h=.746`,
`Omega_m=.31`), whereas the original ARES/2M++ selection program states
`h=.6711`, `Omega_m=.3175`. This is acceptable for a support/wiring probe
only. A quantitative likelihood must freeze one coherent cosmology and
regenerate or justify its distance/luminosity selection and PM transfer.
The native N256 selection is a development integral with documented narrow
footprint error, not a precision calibration certificate.

Next: freeze that cosmology/selection contract, replace the placeholder rate
with a sourced prior, include disjoint survival uncertainty, and obtain
matched-phase PM/higher-fidelity or independent galaxy mocks for bias/FoG/
discrepancy calibration. Then run a **small joint CF4+count IC-gradient and
held-out test** before actual N128 posterior sampling. No N256 posterior or
LG-on claim follows from wiring. MW/M31/M33 remain latent roles of a future
generated state, unresolved M33 explicit, with no native truth IDs in
inference. Q-GOAL: real observations are connected to one evolved global
state. Q-LEAN: native data and one bounded PM state suffice to reject an
unsupported 3% knob; no TNG/download or extra gate is required.
