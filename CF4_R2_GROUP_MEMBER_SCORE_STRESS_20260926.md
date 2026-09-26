# R2 grouped/member partial-likelihood stress — 2026-09-26

## Purpose and fixed scope

This is the single bounded follow-up to the overlap audit, **not** a
validation mock of the actual 2M++/CF4 likelihood. Source-bound eligible
secure matches supply group multiplicities; one preserved unconditional
N128/384 PM density state supplies fixed spatial heterogeneity. A synthetic
scalar field-direction count proxy and one CF4-like Gaussian mark per group
are generated 4,096 times for 512 fixed sampled groups. The experiment asks
whether a product's candidate Hessian agrees with its score variance when
group/member dependence or distance-mark selection is omitted. It cannot
measure the true magnitude of those dependences in the sky.

Four arms are predeclared: independent reference, shared group/member
velocity affecting both count rate and one distance mark, selection that
depends on the distance residual and local field, and a negative control
that improperly repeats a single CF4 group mark for each matched member.
The output reports score variance/H and mean score separately; a nonzero
mean or ratio above one is evidence of miscalibration **in that synthetic
arm only**. The 300-km/s member scatter, rate coupling and mark coupling are
stress values, not inferred FoG/bias priors. No observed count/CF4 likelihood
is fit, no IC is selected, and no additional gravity evolution runs.

Q-GOAL: this tests a specific assumption behind a possible R2 present-field
likelihood before spending resources on posterior sampling. It does not
constrain MW/M31/M33 yet. Their identities remain latent generated-field
roles, including shared-component and unresolved-M33 branches; their
observables must later constrain that same evolved state, not native labels.
Q-LEAN: one short fixed-source mock and no new validation framework, field
generator or all-sky rate fit. A passing independent reference would verify
the score calculation, **not** the real-world conditional-independence
assumption; a stress failure would show why the composite product needs
explicit calibration or a fuller point/mark model.

Script `scripts/cf4_r2_group_member_score_stress.py` and Slurm wrapper
`scripts/run_cf4_r2_group_member_score_stress.sbatch` use a hash-bound
crossmatch and fixed preserved state paths. H200/H100/A100 typed
test submissions passed; the earliest mode was A100 with `gpu:A100:1`.
Job **405282** was submitted with one CPU, 3 GiB and 10-minute cap. Its
initial run FAILED at startup (exit1, zero-second elapsed) before random
draws: the stress-direction builder incorrectly required every saved PM
density cell to be positive. The preserved source state explicitly has5,769
zero-density cells. This is a mock-feature bug, not an invalid PM state or
CF4 observation failure. The same intended comparison is retried with
`log1p(rho)`, preserving the failed log and without changing source data,
random seeds, arms or scientific decision criteria.

Retry **405294 COMPLETED/exit0** in one second. Its fixed 512-group sample
contains439 single-member and73 multi-member groups, including one with45
secure matched members. Results for candidate score variance divided by
candidate expected curvature `J/H`:

| Synthetic arm | J/H | Mean score | Interpretation |
| --- | ---: | ---: | --- |
| independent reference | 1.032 | +0.090 | finite-Monte-Carlo check of the implemented score |
| shared group/member velocity | 1.399 | -0.590 | materially understated variance for this stress coupling |
| distance-dependent group selection | 1.294 | -42.934 | nonzero score centre as well as understated variance |
| erroneous per-member duplicate group mark | 5.681 | -0.308 | negative control strongly detects duplicate CF4 information |

The linked arm has count/mark score covariance125.31 versus9.59 in the
independent arm. The selected arm retained about66.85% of marks. Its synthetic
mark-selection rule deliberately depends on the distance residual **and**
the local field; this violates the missing-at-random premise behind the
partial-likelihood advice. The duplicate arm is emphatically *not* the
proposed implementation—it represents the wrong per-member CF4 factor that
we already forbade.
In the linked arm the marginal count process is overdispersed relative to
the candidate Poisson model (its count-only variance/H is1.306); therefore
the total1.399 is **not** an isolated estimate of overlap cross-covariance.

Driver decision: the mock passes its **structural stress purpose** and refutes
any blanket claim that merely dropping shared-redshift/selection factors
guarantees an unbiased, correctly calibrated posterior. It does **not**
establish that actual CF4 selection has the synthetic dependence, estimate
the real `J/H`, or reject every possible conditioned BGc approximation.
Therefore the inclusive diagnostic remains **NO-GO for production R2
posterior**. Do not apply a synthetic `1.399` covariance correction to real
data or run a long N128/N256 sampler. The next useful science work is a
source-aware calibration or independent survey mock that actually generates
individual 2M++ redshifts, CF4 grouping/distance selection, and grouped marks
from one field, with the 442 mark/map anomalies explicitly handled. If that
cannot validate the conditioned product, implement a normalized joint
point/mark model. Repeating abstract Gaussian score arms would add no needed
information.

## Source group-redshift construction check

The follow-up source-bound Slurm405317 **COMPLETED/exit0** in two seconds.
Among38,023 CF4 groups with listed individual distance contributors and finite
group V3k, published `Ngal` matches the count of `cf4_galaxies.csv` rows
sharing `1PGC` for **every** group. Thus the catalogue's distance-contributor
membership is explicit and usable for a future observation model; this does
not establish the full physical membership of each galaxy group. But group
`V3k` is **not** the simple arithmetic
mean or median of those members' `Vcmb`: absolute group-minus-member-mean
offsets have median206/p90 620 km/s over all groups, and median28/p90 196
km/s among10,389 groups with an eligible secure 2M++ match. For the2,589
secure-matched groups with multiple CF4 individuals, the mean-offset median
is76.7 km/s. These are source comparisons, not group-redshift noise
estimates or proof of a particular group-velocity recipe. The next mock
must keep the published group redshift as a distinct observed quantity with
its dependence on member redshifts specified or stress-tested; replacing it
by the raw member average would fabricate the source-data process.
The [CF4 catalogue paper](https://inspirehep.net/files/0509066626daba3d8e951586c658a0a8)
describes group systemic velocity and group distances built from member
distance contributions; it does not justify replacing the published group
velocity by an unweighted mean of the listed individual velocities.
The [official EDD CF4 column definitions](https://edd.ifa.hawaii.edu/describe_columns.php?table=kcf4allgroup)
make the distinction stronger: `Ngal`-like distance contributions and the
group CMB-frame velocity have different source populations. EDD describes
the latter as averaged from the Tully (2015) 2MASS K<11.75 group catalogue,
**not** from just the CF4 galaxies with measured distance moduli. Thus the
`Ngal` equality above cannot close the redshift-generation part of a mock.
The local [2M++ catalogue ReadMe](data/2mpp_ReadMe.txt) separately identifies
its `GID` as a Lavaux--Hudson group-finder identifier; it must not be equated
with CF4's `1PGC` or with the Tully (2015) group membership. A short
source-only bridge therefore measures the actual overlap and velocity
agreement without assuming those group catalogues coincide. The frozen
script is `scripts/cf4_r2_group_catalog_bridge.py`; Syntax Slurm405399 was
submitted on A100 (H200/H100/A100 typed preflights passed) with one CPU,
3 GiB and 10 minutes. It remained PENDING(Resources) and was cancelled
before execution at the user's direction. The same script was resubmitted
as H200 Slurm405476 with `--partition=h200 --gres=gpu:H200:1`, one CPU,
3 GiB and 10 minutes; Slurm confirms the typed H200 GRES. Its initial state
is PENDING(Priority). Its result,
when available, is not a survey mock or an accepted group-redshift law.

The R2 decision is narrower than the earlier mock suggestion: do not simulate
published CF4 group velocity by averaging CF4 distance contributors. A joint
survey mock requires the applicable velocity-member group catalogue and its
selection/grouping process, or a separately validated conditional law. An
alternative is to model the individual CF4 distance marks jointly with the
matched 2M++ redshifts through shared latent distances and group/method
errors. That alternative also needs explicit selection and covariance; it
has not been implemented or validated here. MW/M31/M33 are not inferred from
this source audit, and any later role model must identify ambiguous or
unresolved members from the same newly generated field without native IDs.
