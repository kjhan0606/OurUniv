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
