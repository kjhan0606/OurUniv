# R2 disjoint-tracer selection defect — 2026-09-26

## Bounded result

The N128 saved-state screen used the **same** unconditional PM field and
fixed bias/FoG response as the preceding IC-gradient control. It did not run
another simulation or select an IC. Slurm405237 (15 s) read 11,502
calibration-only parent survival marks and tested angular exchangeability
within each 2M++ population/radial shell. Across 22 active strata, the
eight-supergalactic-octant Pearson sum was 384.49 versus conditional
permutation median 147.56 and 95th percentile 176.01. The Monte Carlo
`p=0.000976` is the resolution floor of 1,024 null draws, not an exact
tail probability. The test conditions on each stratum's total successes;
its implementation and saved-mark totals were independently checked by
Fable5. It rejects the *population-by-shell only* exchangeability model,
not the existence of a valid selection model.

The same screen compressed the Gamma-Poisson likelihood to occupied cells
and reproduced the previous JAX count log likelihood at the Beta posterior
mean to <0.001 nat. For 2,048 draws from the independent 36-Beta mark
posterior, however, the training and joint likelihood importance ESS were
only 2.76 and 1.64. Consequently the reported Beta-integrated holdout
log-predictive `-56586.42` is **WITHDRAWN / NO VERDICT**. Its two-half
agreement is not evidence of convergence; neither this number nor the
mean-survival fixed-field score may choose a physical observation model.
The prior-draw failure does not prove exact marginalization impossible.

Slurm405241 (7 s) used the bound original CF4–2M++ crossmatch file and the
same 11,502 marks. Of 3,183 failed survival marks, **3,120 (98.0%)** are
CF4-crossmatched 2M++ galaxies; only 63 failures are not crossmatched. The
crossmatch indicator alone has octant statistic 362.24 (`p` at the same
0.000976 floor). Among the 8,382 noncrossmatched calibration parents the
residual survival statistic is 88.44 versus a null 95th percentile 87.89,
permutation `p=0.0478`. Thus CF4 overlap dominates this particular mark's
angular failure. Residual sky/environment/metadata effects are **not**
certified absent. In the largest original stratum (population3/shell5), the
observed crossmatch fractions range from 11/236 to 25/71 across octants;
the octants are diagnostic partitions, not a proposed discontinuous sky law.

Machine outputs:
`/gpfs/kjhan/CF4/z0_density/r2_survival_holdout_screen_v1/result.json`
and `/gpfs/kjhan/CF4/z0_density/r2_survival_cause_attribution_v1/result.json`.
The crossmatch source binding is SHA256
`64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf`.

## Driver decision after Fable5 advice

Fable5 audited the bounded test as CONDITIONAL PASS and the holdout estimate
as NO VERDICT. A second read-only route consultation advised abandoning the
disjoint count thinning and including 2M++ galaxies also observed by CF4,
while treating their CF4 distances as conditional marks. The driver accepts
that the current scalar survival law must **not** enter a production R2
posterior and that a marked-point-process factorization is the cleaner
*candidate* route. It is not yet an approved likelihood or count-catalogue
replacement.

Reasons for this narrower disposition:

- The existing CF4 factor is a BGc-transformed, grouped radial-velocity
  Gaussian likelihood at a distance-derived position. The 2M++ count cell
  is built from its own observed `Vcmb`. For overlapping objects, a simple
  `p(counts|field) p(radial|field)` product is not automatically the exact
  `p(cz, distance|field)` chain rule: shared redshift, BGc reference-pool
  dependence, group-to-individual matching, and ambiguous crossmatch classes
  require explicit treatment or quantified approximation.
- The bound crossmatch lists 17,007 unique 2M++ targets over the full
  source catalogue, not necessarily 17,007 eligible rows in the current
  N128 sample. Fable's instruction to restore "all 17,007" cannot be
  applied literally to this filtered catalogue. Secure versus conflict,
  collision and review classes also cannot be silently reclassified.
- Comparing heldout log scores from two different row sets at one random PM
  phase would not validate either model. The previous Beta importance estimate
  is invalid, so repeating it unmodified is not a useful repair. Fable's
  proposed extra mock score experiment is deferred until the joint datum
  factorization and matching semantics are fixed; it is not a prerequisite
  to recording this diagnostic.

Next essential **single science bundle**: derive and implement the
overlap-aware joint datum at the observed-galaxy level, including the exact
redshift/position convention, grouped CF4 mark construction, secure/ambiguous
match handling and train/calibration/holdout split. On a bounded mock and
one saved field, verify that the same redshift is not used as two independent
measurements and that the count-selection support is correct. Only then
prepare a *new, preserved* inclusive-count diagnostic catalogue and compare
heldout predictions under a stable nuisance integration method. Do not
retroactively replace the disjoint products, tune the new model on the
heldout rows, or launch a long N128/N256 posterior yet. Bias, FoG, rate
shape, LF/completeness and velocity discrepancy still need calibration.

Q-GOAL: this removes a concrete selection/overlap bias from the route toward
the actual CF4+galaxy z=0 posterior; it does not supply that posterior.
Q-LEAN: two short Slurm diagnostics using preserved data, followed by one
necessary joint-datum design rather than an octant correction field, another
random-phase search or a simulation. No native halo identity was used.
MW/M31/M33 remain latent generated-field roles, including ambiguous shared
components and unresolved M33; their observables must later constrain the
same inferred state, not this selection diagnostic.
