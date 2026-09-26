# R2/5 — joint calibration and selected-radial-shape bundle

Reuse the saved unconditional N128/384 state and the original10,020 FP and
114 linked non-FP measurements. No new gravity, TNG, group-scatter sweep or
independent prior made from the preceding fit. Deliver one conditional joint
nuisance sampler, not a density/velocity posterior.

## Frozen model and calculation

For each group, integrate the product of its FP and non-FP marks over ONE
distance using the existing correlated redshift kernel. Share one FP eta
zero point across all groups and one modulus offset per non-FP method.
Infer these five offsets simultaneously with a sixth parameter kappa:

`w_g(d) = dd d^2 rho_F(d) exp[kappa log(d/100)]`.

Use this same weight in both conditional numerator and denominator. Kappa
is an effective SELECTED radial intensity shape, not a probability, measured
group inclusion, or separately identified tracer bias. Its amplitude cancels
when conditioning on redshift. Bias1, the density interpolation and the
150/100/50km/s member/COM/catalogue covariance remain provisional. This does
not provide the missing joint count/point law or solve source-richness fit
covariance, group interlopers or environment-dependent selection.

Use the closed5,330 train /1,491 heldout group split. Do not read preceding
fitted offsets into initialization or priors. Raw non-FP errors are used once;
neither the combined CF4 DM nor the fitted0.1785mag relative scatter enters.
Retain full skew-normal FP source likelihoods, not Gaussian group moments.

The DEVELOPMENT Gaussian prior is zero-centred with SDs:
FP0.004dex; each non-FP method1mag; kappa2. The FP width is conditional on the
published relative CF3 calibration; it excludes CF3 absolute-scale uncertainty
and cross-survey covariance. The other widths are explicit regularization,
NOT externally calibrated uncertainties. The interface accepts a full prior
Cholesky factor, but this run uses a diagonal development prior. Marginalizing
it does NOT solve the omitted source-fit covariance. Do not add the published
cosmic-variance zero-point width to a future latent-field model automatically.
Source context: https://arxiv.org/html/2201.03112, sections5.3–5.4.

Cache fixed geometry/kernel at257 and513 distance nodes from the existing
source driver. This is a conditional speed cache only, not a replacement for
the differentiable same-field operator in eventual IC inference. Run one
six-dimensional MAP fit for preconditioning, then4 HMC chains,128 warmup and
256 retained draws each, seed20260927, randomized3–7 leapfrog steps,
target acceptance.85. MAP covariance is a proposal coordinate transform,
NOT a Laplace posterior approximation. Stop if the observed Hessian is not
positive definite; do not clip its eigenvalues. Preserve bounded partial
chains on a720-second sampler deadline, without declaring convergence.

Check32 predeclared retained draws with513 nodes; report the range of log
target corrections, not just absolute offsets. Report heldout conditional
mark-factor averaging over the same64 predeclared draws with importance
concentration diagnostics. This uses a fixed data reference and shared source
fits, so is not absolute predictive density or independent validation. No
holdout fitting, acceptance-threshold tuning, or automatic follow-on run.
The bounded numerical screen is split Rhat<1.05, split ESS>100 for all six
coordinates, zero retained divergences, and sampled quadrature-correction
range<0.05nat. These are pilot diagnostics, not scientific acceptance or a
whole-posterior accuracy guarantee; retain their unrounded values.

Two focused tests suffice: the full shared/correlated calibration target
against a Gaussian algebraic reference, and selected-shape normalization,
holdout exclusion and joint vector differentiation. Reuse the existing HMC
wrapper and source distance integrals; add no monitoring framework.

Slurm:1 H200/H100/A100 compatible GPU,2CPU,6GiB host memory (planning peak
<=5GiB plus20%; measured RSS will be reported),16min. All numeric work runs
there, not on Syntax. Preserve existing outputs and unrelated files.

Q-GOAL: remove fixed empirical calibration estimates from the actual-data
conditional mark model, while making selection-shape uncertainty explicit
for eventual joint field inference. R2 present-state samples are still owed;
this nuisance-only result cannot substitute for them.
Q-LEAN: one existing state, one model, one bounded fit/sample job, no new
simulation, catalogue download or generic validation infrastructure. Routine
driver review applies; no external audit is needed for this implementation.

MW/M31/M33: these parameters do not identify LG components. R3 must identify
candidates from each NEW evolved field/state, retain MW/M31 role ambiguity
and unresolved M33, and constrain that SAME field with their observables.
No native truth labels, true centres or best-seed selection enter this bundle.
Remaining route: R2 actual z=0 posterior -> R3 LG<=0.3cMpc/h -> R4 accurate
forward validation -> R5 phase-consistent zoom ICs.

## Execution record

First Slurm406219 stopped after1m06s/exit1. Two new algebra/ownership tests
and the reused same-distance non-FP test passed; both fixed-state caches were
saved. The new executable/module shared a basename without prioritizing src,
so Python imported the executable recursively at startup. This is a driver
import-path error, not scientific/model failure; no fit or samples started.
Fix src precedence explicitly and rerun with --reuse-cache, preserving the
successful cache and failed logs. No regeneration of observations or gravity.

Retry H200406244 COMPLETED1m27s/exit0; all three focused/reused tests pass.
Sampler/fit/readout runtime44.76s; batch MaxRSS1,690,296KiB (~1.61GiB),
within the6GiB reservation. Original cache406219 is retained unchanged.
Outputs:
`/gpfs/kjhan/CF4/z0_density/r2_joint_calibration_v1/{result.json,chains.npz,retained_checks.npz}`.
The geometry and its source manifest are in `r2_joint_calibration_cache_v1`.

MAP converged in8 iterations/13 evaluations, maximum white-coordinate
gradient3.03e-6. The observed precision is positive definite without clipping.
Four chains retained256 draws each, total1,024, after128 warmup each.
Mean acceptance0.8684, retained divergences0, maximum split Rhat1.003044,
minimum split ESS719.88. These are ordinary split diagnostics, not
rank-normalized or an independent proof of global mixing.

| Conditional parameter | Mean | SD | Development prior SD |
| --- | ---: | ---: | ---: |
| Shared FP zero (dex) | -0.010688 | 0.001045 | 0.004 |
| SBF offset (mag) | -0.08505 | 0.06002 | 1 |
| SN II offset (mag) | -0.20434 | 0.23017 | 1 |
| SN Ia offset (mag) | -0.01986 | 0.01663 | 1 |
| TF offset (mag) | -0.02837 | 0.17538 | 1 |
| Selected radial kappa | 0.39885 | 1.86396 | 2 |

These are NOT calibration measurements to transfer to the full inference.
They condition on a fixed RANDOM-PHASE field and an uncalibrated velocity/
selection law, with source-shape covariance omitted. Their narrower errors
than the prior bridge REML fit do not demonstrate improved physical accuracy:
that calculation had a different distance/scatter model. No old residual-fit
offsets or scatter enter this target or its initialization.

At32 predeclared retained points, Q513-minus-Q257 training log-factor
corrections span0.00060639–0.00097298nat, range0.00036659; heldout correction
range0.00112003. Accept the bounded numerical implementation, not a uniform
posterior-error bound or production model. The64-draw integrated heldout
factor has importance ESS5.5267 and maximum weight0.3803. Its reference-based
logmean1870.763 is NOT an absolute predictive density and is NOT accepted as
a reliable predictive-performance estimate. No holdout refit or expansion
was performed to improve this number.

## Driver decision and next substantive delivery

Accept the shared, correlated-prior-capable nuisance interface and its one
bounded conditional sampler. Joint offsets are no longer fixed empirical
inputs. Selection-shape uncertainty is propagated rather than silently fixed,
but kappa SD1.864 versus prior2 shows little conditional information about
this shape. Its marginalization does not identify selection separately from
the selected population or certify the physical FoG/source-fit model.

Close this fixed-field calibration control. Do not run more frozen-field
offset/scatter sweeps or use these samples as an independent prior for their
own data. The next substantive implementation must connect field updates and
these SAME raw marks/nuisances, with a stated selected-group law. The still
missing inclusive count/within-cell point/redshift factorization must not be
papered over by multiplying the old inclusive counts with these conditional
marks. Until that law and its remaining source/covariance uncertainty are
handled, report any field fit as a declared partial/development model, not
the complete CF4+galaxy R2 posterior. No N256 production launch or new gravity
calculation follows from this nuisance-only PASS.

R2 actual z=0 posterior, R3 MW/M31/M33 identification and LG resolution,
R4 precise forward validation and R5 usable zoom ICs remain outstanding.
