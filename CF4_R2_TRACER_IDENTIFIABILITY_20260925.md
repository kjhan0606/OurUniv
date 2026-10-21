# R2 tracer decision — 2026-09-25

## Result from actual-data saved chains

Syntax Slurm404160 read only the four completed actual CF4+2M++ N32/384
chains (2048 nuisance projections per chain), their corrected count datum,
and the independent 20% survival-mark calibration. It did not refit or run
gravity. The machine-readable result is
`/gpfs/kjhan/CF4/z0_density/r2_tracer_identifiability_v1/job_404160.json`.
All six standardized bias coordinates have posterior means 2.417–4.947 and
pooled SD 0.109–0.225 against an N(0,1) prior. The model transform is
`bias_p = published_bias_p * exp(0.25 * z_p)`, so these correspond to
1.83–3.45 times the imported bias centres **within this old 12 cMpc/h
model**. Four chain means agree closely. The same model's posterior is
therefore informative about *its own* bias parameters, but does not measure
physical scale-dependent galaxy bias or calibrate an N256 PM tracer law.
Bias–rate correlations are at most 0.228 in absolute value and the largest
bias–survival correlation is 0.051. All supported population/radius shells
have survival calibration marks (11,504 parent marks total). These checks do
not rule out a common matter-field/response/selection mismatch. The old
likelihood, smoothing, field prior, count sample and imported bias definition
are jointly implicated; the cause is not isolated by nuisance projections.

The old model takes a positive lognormal z=0 field, biases it, then deposits
cell-centred tracer mass through TSC and two-node FoG. That response is not
the same as an R2 particle/continuous-tracer operator. The imported ARES
prior was defined at 2.34375 cMpc/h, not 12 or 1.5. Do not transfer the
posterior values or tighten the prior around them. The strong displacement
is a **model-stress warning**, not proof of a unique code defect or novel
galaxy physics. A small width conditional on an incorrect model is not an
accuracy certificate.

## External advice and driver disposition

Read-only Claude Opus advisory returned CONDITIONAL PASS for recording the
identifiability result and REJECT for using it as an R2 bias measurement. Its
Q-GOAL/Q-LEAN advice favours one bounded existing-PM count-variance comparison
instead of a TNG download, a new old-model refit, or an N256 HMC. Adopt the
no-transfer/no-refit point. Amend its proposed input: the saved R1 particle
state in Slurm354568 occupies only a 12 cMpc/h box, so it cannot be aligned
with the actual 384 cMpc/h survey or used to calibrate its counts. Its claim
that the TSC/FoG response caused the full bias shift is a plausible
hypothesis, not a measured attribution. The existing native PM fields in
`cf4_z6_native_physics_plan_v1.json` do cover 384 cMpc/h but are N32/12
coarse products; they can test the old-scale response only, not N256/1.5.

The earlier 64-particle synthetic same-state control had one occupied cell
with zero predicted intensity under a velocity perturbation. Its JAX
numerical log guard generated a finite but **invalid** count score. The
corrected Slurm404169 output stores that count score as `null`; the local
derivative at the supported state remains finite. No likelihood floor has
been introduced.

## Next bounded science action

Use the existing 384 cMpc/h native PM coarse fields and corrected actual
2M++ count/selection arrays to compare count-in-cell fluctuations by
population and radial shell **at 12 cMpc/h**, including the selection window,
shot noise and disjoint survival uncertainty. This is a statistic-of-fields
calibration diagnostic, not a sky-by-sky fit to an unconstrained realization.
If that comparison cannot distinguish plausible bias/response choices, report
non-identifiability rather than retune a prior. A 1.5 cMpc/h calibration
requires an appropriately resolved 384 cMpc/h PM state or a justified
alternative; none of the archived 12-box R1 and N32/384 fields supplies it.
Do not launch N256 actual-data sampling until the positive-support tracer
response, high-resolution calibration and state-dependent evaluation cost
are measured. Existing 2M++ and CF4 data enter once each, with frozen
cross-match and held-out ownership.

Q-GOAL: this is R2 environment work toward actual z=0 density/velocity and
eventual LG zoom ICs. It does not identify MW/M31/M33, resolve ambiguous
roles or certify M33; those remain latent on the same future evolved state.
Q-LEAN: no TNG, no new simulation or old-model refit in this decision.

## Native-PM/actual-count bounded comparison — Slurm404189

The predeclared 384 cMpc/h, N32 comparison used all six archived native PM
coarse fields, recentered their density from the 3 to 6 cMpc/h count-cell
origin by the established conservative shift, and applied the actual ARES
exposure weighted by the disjoint calibration-mark Beta survival mean.
For each population/window it compared the observed second factorial count
moment (Poisson self-pairs removed) against the same selection-window moment
of `rho^b` on each independent-phase PM field, conditioning both on that
population's observed total. Output:
`/gpfs/kjhan/CF4/z0_density/r2_native_count_fluctuation_v1/job_404189.json`.

For the full 5–180 cMpc/h radial window, observed excesses are
`[1.498, 1.162, 0.873, 1.908, 1.169, 1.118]`. Six-PM-field means with
published biases are `[1.096, 0.429, 0.403, 1.266, 0.385, 0.236]`;
reusing the old actual-fit biases instead yields
`[57.0, 69.5, 30.5, 21.1, 24.7, 17.8]`. The last contrast strongly
reinforces **NO TRANSFER** of old-model bias to a different matter operator.
The published-bias shortfall is not itself a fitted replacement bias: these
six PM fields are unconstrained to the real sky, the comparison omits RSD/FoG,
uses a coarse matter-density power-law tracer, includes neither field cosmic
variance nor selection-calibration uncertainty in an error bar, and does not
test held-out likelihood or a 1.5 cMpc/h state. The six native fields also
overlap the old development corpus, so this is not independent validation.

Decision: close this N32 factorial-moment diagnostic without tuning bias to
it. Move to a positive-support **continuous tracer** tied to a 384 cMpc/h PM
state, with spherical RSD/FoG and the same actual selection/cross-match
contract. Its N128/3 cost/gradient and selection/bias uncertainty must be
measured before N256/1.5 inference. A Gaussian-smoothed matter density or
other occupancy support must be specified as physical resolution/discrepancy,
not an undocumented likelihood floor. MW/M31/M33 remain unresolved latent
roles, not something to infer from this coarse count statistic.
