# R2 conditional FP component: bounded discrimination, 2026-11-07

This implements the first small observation-factor study of the user-approved
route revision, not a production target, calibrated selection model or R2 exit.
The full goal remains the CF4+galaxy-conditioned z=0 density/velocity posterior
in the 384 cMpc/h box on a 1.5 grid, with uncertainty/calibration/held-out checks.

## Factor and ownership

For measured X=(r_z,s,i), define U=(2*r_z+i, .04*r_z+s) and T=r_z. At fixed
redshift/aperture metadata, U corresponds to apparent magnitude and raw
dispersion; (U,T) is an invertible transformation with unit absolute Jacobian.
For a Gaussian component N(mu,C), the conditional mean is
mu_T + C_TU C_UU^-1 (U-mu_U); variance is its Schur complement. Implementation
uses the transformed Cholesky factor, without a variance floor. Optical error
covariance is included before conditioning on measured U. This does not make
selection on unknown true covariates ignorable.

For latent components j, use weights proportional to w_j p(U|j,F), then mix
their T|U laws. Retaining the previous w_j after conditioning is incorrect.
Other conditioned quantities, e.g. observed K, require their own correctly
owned factors. FP-only source weights are not automatically 2M++ weights.
A common zero point in the same mean direction as eta remains confounded;
per-row Fisher information with fixed population parameters is not calibrated
distance uncertainty or independent information from multiple group members.

Known cuts on U cancel only when inclusion is independent of T and remaining
latent variables given the actually conditioned observations. The SDSS
magnitude/raw-dispersion cuts can be represented by U; sky/redshift can be
conditioned. Colour, morphology, concentration, axial ratio, image quality and
FP-outlier rejection are not all captured by U. The source header provides
colour ingredients, axis ratio and a morphology label, but a label is not a
complete rejection/selection law. The existing training archive does not carry
all primary selection variables or the iterative FP rejection history.
See [Howlett et al., sections 2 and 4.2](https://arxiv.org/html/2201.03112).
No normalization in this component corrects those unresolved stages.

## Fixed study, not a new data census

Reuse training_photometry.npz: 8,901 existing CF4-linked training source rows
and its 1,414 linked-mark subset. The larger set is NOT the complete native
SDSS catalogue or a failed-target denominator. Validate selected error
covariances against training_cut_geometry.npz. Reuse the old final_fit.npz
only as a fixed covariance/direction sensitivity reference, not a calibration
prior, selected field seed or new result. No parameters are fitted here.

Check joint/conditional density identity, normalization, latent-node
reweighting, local distance response/Fisher decomposition, and explicit
shared-zero-point degeneracy. Report information fractions for all training
rows without favourable filtering, under the development prior covariance
and the legacy fitted covariance. The optional direction e_r+5*beta is a
low-redshift fixed-covariance illustration of M changing with eta, not the
complete field derivative. Do not sum these row informations as independent.

Use one fixed negative control a(e)=logistic(gamma*e) for gamma=0,.25,.5,1.
These strengths are sensitivity axes, not empirically allowed selection
bounds. Selection on the residual shifts its conditional mean and can mimic
distance. Passing the Gaussian mechanics cannot qualify the real source.

CPU Slurm on Syntax a10: one core, 10 minutes, 512 MiB requested; expected
peak <=400 MiB for the small arrays, SciPy and illustrated report, leaving
>=20% headroom. No GPU, PM, RAMSES, sampler, held-out score or large raw output.
The two existing/new focused test files run before the study. Save one JSON,
one PNG and one Korean PDF with a figure for each of the four scientific
checks. Measure actual peak RSS and wall time. No polling/process-scan loop.

Continue to source-specific selection/covariance work only if conditioning
retains useful signal. Do not promote this component to the field target
until its source weights, selection assumptions, group covariance and
identity-level train/held-out separation are defensible. Failure redirects
the observation-factor design, not another sampler sweep.

Q-GOAL: same NEW evolved field and eventual MW/M31/M33 <=0.3 LG constraints;
MW/M31 roles remain ambiguous, M33 unresolved, and truth identities remain
calibration/evaluation only. Q-LEAN: one reusable factor and one bounded study;
no new selection census, simulation or generic gate framework.

## Executed result — Syntax CPU job 453825

Source commit `78a1e00d3e57a19d3eb8a1f3533a5993cfd031df`; a10/syn01,
one CPU, 512 MiB, COMPLETED/0:0 in 18 seconds. Study wall time was 4.65 s,
process-reported peak RSS 211.75 MiB. Slurm's short-job RSS sample was only
3,356 KiB and is not used as the actual Python peak. Six new component tests
and seven existing raw-FP regression tests passed. The source/data checks,
13-test scope and illustrative PDF do not constitute survey calibration.

Artifacts are in
`/gpfs/kjhan/CF4/z0_density/r2_conditional_fp_component_20261107/453825/`:
`result.json`, `conditional_fp_examples.png`, `R2_조건부_FP_예제보고.pdf`.
The JSON pins all three input file hashes. The rendered PNG was inspected;
all four scientific checks have an example figure and the limitations are
visible. The legacy fitted field/population was neither refitted nor promoted.

Median local conditional/joint Fisher information fractions:

| Fixed reference/direction | Existing training rows (8,901) | Linked subset (1,414) |
| --- | ---: | ---: |
| Development prior covariance, radius-only eta shift | 0.19970 | 0.19983 |
| Legacy fitted covariance, radius-only eta shift at fixed M | 0.37056 | 0.37042 |
| Legacy covariance, illustrative low-z M response | 0.27231 | 0.27192 |

These fractions hold the population parameters/covariance fixed. They do NOT
measure information retained by a marginalized field posterior, independent
distance precision, or how well the old population transfers to the larger
source set. The 8,901 rows occupy 6,028 training source groups; they cannot be
treated as 8,901 independent group measurements. There was no zero-information
row for the tested directions; the unit test also checks a direction with
zero conditional information, without forcing a positive floor.

The two-node conditional mixture integrates to 1.0000000000. Covariate
conditioning changes its illustrative weights from (0.5,0.5) to
(0.105883,0.894117); ignoring that change shifts the mean from the correct
0.811558 to 0.780038. The bounded logistic residual-selection control at
gamma=0.5 shifts the standardized mean by 0.236044; fixed-covariate-only
selection leaves the normalized distribution unchanged. Gamma is an
illustrative sensitivity value, not a calibrated selection prior. In the
radius-only control, a common eta shift and a shared FP zero point acting
in that same mean direction have rank-one information. This does not assert
that all such degeneracies survive an additional independent calibration or
the full joint field/count model.

Driver decision: the candidate is not empty of distance signal under these
reference geometries, so continue the observation-model redesign. Selection
calibration is still false, the live v6 target is unchanged, and R2 is not
complete. Next address source-specific conditional selection/population
calibration and the native FP-only/shared-group source kernel; do not use
these fixed-parameter Fisher fractions to authorize a production chain.
No new field, held-out prediction, PM/RAMSES run or external exit audit occurred.

## Source-law decision and shared-group connection

The immediate continuation reuses the existing group-distance/shared-zero
machinery as the ownership reference, but does not call its normalized scores
with a new unconditional zero-point prior. The raw conditional FP component
has covariate evidence that depends on those same shared variables.

| Source stage | Treatment in the candidate | Remaining scientific condition |
| --- | --- | --- |
| Observed magnitude and raw dispersion cuts | Fixed by measured U; their indicator is constant at that U | Cancellation also requires no remaining response/latent-dependent selection; noisy measured U is not true U |
| Observed sky/redshift window | Can condition on these observations | Redshift-to-distance uncertainty and source spatial weights remain; observed n(z) is not a true-distance prior |
| Colour, axial ratio, morphology and image/spectral quality | Additional measured covariates may describe some stages | U alone does not capture them; not all original quality variables/history are present. A selected morphological population needs its own response model |
| Iterative FP residual rejection | Response-dependent; does NOT cancel | A final fit/threshold alone does not reconstruct all earlier rejection boundaries; no residual-selection normalization is yet supported |
| CF4/Tempel/count linkage and single-mark restriction | Artificial count intersection is being reconsidered, not silently removed | Preserve identity-level split and occurrence/redshift ownership; missing links are not failures or independent sources |
| Group and source-wide calibration | Integrate shared variables jointly | Their covariance, selected-population transport and group assignments need evidence; numerical integration does not measure them |

Source for selection stages: [Howlett et al., sections 2 and 4.2](https://arxiv.org/html/2201.03112).
Its relative CF3 zero-point error is not an independent absolute-scale prior;
the existing `CF4_R2_ZERO_ANCHOR_SOURCE_20260929.md` already explains the
overlap/cosmic-variance issue. Do not repeat that investigation or adopt a
new convenient scatter width from the toy examples below.

For group g, let j be a **joint local node** (e.g. common distance and group
offset), b a source-wide calibration node, and w_gj(b) the supplied normalized
source weights conditional on already-owned redshift/other observations K.
At fixed nodes, the new functions evaluate

    N_g(b) = sum_j w_gj(b) product_i p(U_i|j,b) p(T_i|U_i,j,b)
    D_g(b) = sum_j w_gj(b) product_i p(U_i|j,b)
    p(T_all|U_all,K,F) = sum_b q_b product_g N_g(b) /
                         sum_b q_b product_g D_g(b).

This assumes independent member measurement errors **conditional on the
shared nodes**. It does not allow unknown correlated errors to be discarded.
Source-weight amplitudes are normalized separately at each (g,b); otherwise
arbitrary source normalization would manufacture a calibration constraint.
The equivalent conditional mixture has weights proportional to
`q_b product_g D_g(b)`, not the old q_b. Returning the joint and covariate
integrals separately keeps information ownership explicit. For a full field
posterior, dropping p(U|F) also requires declaring a defensible conditional-
design model or supplying p(F|U); this algebra does not equate that inference
to the full joint observation model.

`grouped_log_integrals` and `shared_conditional_logpdf` extend the existing
small NumPy reference; no second source-fit or group framework is introduced.
Four focused checks compare to an independently constructed nine-dimensional
Gaussian with shared offsets, exactly enumerate shared/local nodes, test
arbitrary source-amplitude and member-order invariance, and reject malformed
or wholly unsupported inputs. The bounded CPU study also reports existing
training group multiplicities and illustrated shared-error/conditioning
examples. Examples are explicitly NOT new calibration priors or field fits.

This connection removes an implementation gap, not the unresolved source law.
The existing live v6 target and frozen sample remain unchanged. Next work must
establish a supported source population/response-selection law and distance
weights, not run another parameter sweep on assumed calibration. R2 remains
incomplete; MW/M31 role ambiguity and unresolved M33 must still be constrained
on the same NEW evolved field in R3. Q-LEAN: one extension to the existing
component and bounded CPU study, no new simulation or general gate framework.

### Executed shared-group result — Syntax CPU job 453839

Source `f680d54a12a085797daddd1f94a926875f574067`; a10/syn01, one CPU,
512 MiB, COMPLETED/0:0 in 10 seconds. All 17 focused tests passed (10
conditional-FP, including four new shared-node checks, plus seven raw-FP
regressions). Study time 5.00 s; process-measured peak RSS 258.41 MiB, below
the predeclared 400 MiB estimate and 512 MiB allocation. The short-job Slurm
RSS sample again missed the process peak and is not used as peak memory.

The existing 8,901 training rows contain 6,028 source groups: 4,983 single-row
groups and 1,045 multi-row groups containing 3,918 rows. These are catalogue
identities, not verified physical memberships. The wider cohort therefore
cannot be used as 8,901 independent group distances. This is NOT evidence of
that error in the old 1,414-mark cohort, which deliberately has distinct
source parents; the new integration enables reconsidering that restriction.

The shared-offset integral agrees with an independent analytic 9D Gaussian
reference (joint, covariate and conditional densities, absolute log tolerance
1e-10). In the explicit discrete example, shared-node weights change from
(0.25, 0.50, 0.25) to (0.06209, 0.46289, 0.47502). Both the correct and stale-
weight response curves integrate to one, but their total-variation distance
is 0.03946: normalization alone would miss this different probability law.
In the separate dimensionless n=8 sharing example, the mean standard error
is 0.61237, versus 0.39528 when a common error is incorrectly made independent.
Neither example measures a real-survey bias, uncertainty or calibration prior.

Artifacts: `/gpfs/kjhan/CF4/z0_density/r2_conditional_fp_component_20261107/453839/`,
including `result.json`, `shared_group_examples.png` and
`R2_그룹공유보정_예제보고.pdf`. The figure was visually inspected; each of the
four reported checks has an example, with actual-data versus illustrative
content explicitly separated. The original four-panel component report was
also regenerated. Input hashes and source commit are recorded in the JSON.

Driver decision: adopt this numerical connection for continued source-model
development. It does not calibrate response selection, shared covariance,
group membership or native FP-only distance weights. No field was fitted,
held-out outcome read, production chain submitted, or sample/target changed.
No additional sampler/mock sweep is warranted by these tests. The next
science task remains specifying and confronting the actual selected-source
response and source-distance law; R2 and the full field deliverable stay open.
