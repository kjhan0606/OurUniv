# R2/5 — 2M++ selection-coordinate repair boundary

Status: bounded source-model comparison completed in Slurm **406565**; it is
not an R2 posterior or a calibrated galaxy likelihood. No email/contact was
sent. Startup attempts 406560/406561/406562 are preserved: unavailable
`pytest`, a reversed test expectation, and the wrong Python environment for
`astropy`, respectively. None produced a science comparison.

## What the current target assumes

The actual catalogue code assigns each of six populations using published
apparent `Ksmag` and an absolute K magnitude computed from *observed* `Vcmb`
(`scripts/cf4_r2_common_catalogue.py` and
`src/cf4_twompp_disjoint_tracer_pilot.py`). The current forward count model
instead displaces a source by coherent RSD/FoG, deposits its fixed population
mass, and multiplies the result by the six-population exposure tabulated on
the *observed-space* output voxel (`src/cf4_r2_continuous_tracer.py`). This
does not move a source between observed absolute-K bins, and in general
cannot represent several real-space radii contributing to one redshift-space
cell. The imported six rates/biases are also conditional on the earlier
2M++ population and selection model, not calibrated intrinsic rates for a
new marked process.

The primary 2MRS field-level analysis by
[Nusser (2026)](https://arxiv.org/html/2606.08593v2) evaluates its flux
selection at model real-space distance before RSD deposition; its catalogue
and population definition are not identical to ours. The original
[2M++ ARES/BORG study](https://arxiv.org/html/1509.05040) uses the two
apparent-K samples and three absolute-K bins. Neither paper licenses a blind
swap of our six exposure maps or its published rate/bias priors.

## Source-consistent candidate

Let `r` be true comoving distance, `s` the model observed-redshift distance,
`M` intrinsic K absolute magnitude in the adopted h convention, and
`mu(r)` the corresponding luminosity-distance modulus. The forward marks are

    m_corrected = M + mu(r) + Delta_K(z_true,z_observed)
    M_observed = m_corrected - mu(s).

Here `Delta_K` is the change in the published redshift-dependent 2M++ K
correction from the true to observed redshift; fixed-sightline extinction
cancels. This is a mean correction model, not object-specific photometry.

For every source, classify `m_corrected` into the published bright/faint apparent bins
and `M_observed` into the three observed absolute bins; apply observed-radius
support, angular completeness and source survival to that selected mark.
Deposit the source at `s`. Integrate the intrinsic luminosity distribution
over the joint inequalities, including true magnitudes outside the observed
`-25<M_observed<-21` interval because RSD can move them into it. This is a
**marked** point-process specification, not six independent radial weights.
The selection and group membership of any CF4-matched objects must be handled
jointly rather than multiplied again as an independent CF4 group mark.

`src/cf4_r2_observed_magnitude_transfer.py` implements exact conditional
bin-transfer integrals **under an assumed Schechter LF**, including
bright/faint intrinsic tails. The [2M++ source paper, §2.2 and §2.6](https://arxiv.org/html/1105.6107)
defines a redshift-dependent correction to the published K magnitude. Its
Table 2 `|b|>10, K<11.5` row, whose `alpha=-0.94, M*=-23.28` are imported by
the [ARES/BORG 2M++ analysis](https://arxiv.org/html/1509.05040), was fitted
over `-25<M_K<-17`—not just the six observed populations' `-25<M_K<-21`.
Thus `-21<M_K<-17` is within that fit's magnitude interval, whereas
`M_K<-25` and `M_K>-17` require extrapolation. The source also notes an LF
inflection near `-21` and bright-end Schechter departure, so fit-domain
membership alone does not calibrate these migration probabilities for our
survey/field model. The code now requires an explicit
K-correction shift; the source's mean redshift-dependent formula is applied
in the fixed-distance comparison. Galaxy-specific photometric/aperture
deviations remain outside this model. It deliberately does not supply intrinsic
luminosity-dependent bias, overall
rate, survey angular map, observed radial cut, CF4 inclusion/covariance, or
source redshift/FoG calibration. None of these can be filled by relabeling the
old six `nbar` values. Its r=s limit must reproduce the existing six LF
fractions, while r!=s can move absolute-magnitude labels.

One fixed-radius test uses the adopted cosmology at true radii 10, 30, 90,
175 cMpc/h and observed shifts -3, 0, +3 cMpc/h. It isolates selection
geometry and is **not** an estimate of the actual survey's velocity law or
posterior bias. Initial Slurm406563 completed in12s with3/3 tests but set the
redshift-dependent K-correction shift to zero. Source-corrected Slurm406565
completed in3s with4/4 tests; its preserved result is
`/gpfs/kjhan/CF4/z0_density/r2_selection_coordinate_control_v2.json`.
The superseded no-correction result remains at
`/gpfs/kjhan/CF4/z0_density/r2_selection_coordinate_control_v1.json`.
The zero-RSD source/voxel comparison agrees exactly at all four radii. At
30 cMpc/h with s-r=+3, the paper's mean K-correction changes by+0.00269mag;
10.34% of *selected LF measure* comes from central intrinsic bins that change
observed absolute-K bin. Another8.79% comes from the **faint-side** intrinsic
range outside the six observed bins, `M>-21`; at this radius and apparent
cut, that contribution lies inside the source LF's `-25<M<-17` fitted interval,
but its exact fraction is still LF-model dependent. Restricting
the denominator to selected in-range intrinsic sources, central-bin migration
is 10.34/(100-8.79)=11.34%. The six-bin LF selection vector differs from
post-RSD voxel selection by8.64% in L1, normalized by the selected LF
measure. The additional redshift-dependent K-correction changes that vector
by0.112% in L1. At10 cMpc/h, s-r=-3 gives47.21% central-bin migration and
52.85% normalized L1 difference; its0.145% bright-side tail relies on
`M<-25` extrapolation. These are fixed-displacement sensitivity
numbers under an assumed LF, **not** actual
2M++ frequencies, a field-level bias estimate or a full selected intensity
comparison. Velocity distribution, angular completeness, radial support,
intrinsic bias and CF4 overlap are deliberately absent.

## Decision before another field fit

Do not run another long HMC chain on the present partial target. First fit or
externally calibrate the *intrinsic* luminosity-dependent rate/bias and
source redshift/group-selection law (including FoG and the CF4 overlap), then
compare a full source-consistent mock/heldout prediction against the frozen
post-RSD exposure approximation. Only after a target is fixed should a
materially different blocked/preconditioned sampler be tested with independent
starts; no N256 posterior or uncertainty map is licensed by this diagnostic.
The existing identity-mass HMC geometry line is closed.

## Source-marked count connection (bounded next control)

`src/cf4_r2_marked_tracer_jax.py` now implements a differentiable source-side
alternative: five intrinsic absolute-K masses at each real-space count voxel,
the published mean corrected-K shift and apparent/observed-absolute binning
at each coherent+stochastic LOS destination, fixed-sightline angular
completeness and observed-radius support, then deposition into six observed
count populations. It does **not** multiply the archived observed-voxel
exposure afterward. Intrinsic rates, within-bin LF shape, nonlinear bias,
FoG, survey inclusion and CF4 group overlap remain assumptions or unknowns;
the old six published `nbar`/bias values are not reused as intrinsic values.

The two small numerical tests in typed-A100 Slurm406567 passed: the JAX mark
transfer agrees with the NumPy reference including corrected K, and source
selection followed by deposition preserves the expected selected mass and
has a finite, matched LOS-velocity derivative. This is kernel wiring only.

The next one-state N128 check uses the preserved **unconditional** PM state,
conservative native-node mass/momentum readout at count-voxel centres, and
eight-subcell angular averages from the existing public ARES masks. It sets
one arbitrary unit intrinsic count per source cell and b=1, partitioned by
the assumed Schechter LF; these are deliberately not fitted parameters or
new priors. Evaluate forward cost and whether the 45,776 actually occupied
population-cells have positive support. Do not score or fit a posterior,
launch another HMC chain, infer bias from this state, or certify angular
quadrature from one support result. Typed H100/4 CPU/24 GiB/30 min is a
bounded upper resource request (estimated host peak <=20 GiB plus 20%).

Q-GOAL: this tests whether a source-consistent K/RSD count operator can be
connected to the same kind of evolved matter/velocity state needed for R2.
Q-LEAN: one saved-state forward evaluation, no new gravity simulation or
validation ladder. MW/M31/M33 are **not** identified on this coarse state;
at R3 their candidate roles (including unresolved M33) must be inferred
from each NEW evolved state and their observables condition that same state,
without native-truth IDs selecting candidates.

The first H100 submission **406568 FAILED before calculation** because the
JAX environment lacks `healpy`; the separate geometry environment has it.
The same-scope retry **406569 COMPLETED/exit0** in 61 s. It prepared the
fixed public-mask/cosmology tables using Python 3.13, then ran the source-
marked forward operator in the GPU JAX environment. The result is
`/gpfs/kjhan/CF4/z0_density/r2_marked_source_n128_v2/result.json`, with
geometry in `r2_marked_source_geometry_v1/geometry.npz`. All **45,776**
occupied population-cells have positive intensity (minimum `1.07025e-5` in
the arbitrary unit-rate model); compiled forward time was **42.13 s** and
host peak **2.71 GiB**. The six predicted totals are deliberately *not*
compared to 57,238 observed galaxies as a fit: the unit intrinsic rate and
b=1 were declared mechanics inputs. This establishes forward feasibility
and support for one saved state, **not** a CF4+2M++ likelihood calibration,
heldout prediction, mixing, an N256 map or posterior uncertainty. In
particular, eight angular subcell samples are not certified at mask edges,
the selected-group/point law is unresolved, and no new gravity evolution ran.

Next substantive R2 work is to make the intrinsic rate, luminosity-dependent
bias and LOS scatter *joint uncertain parameters* of the source-selected
count law while preserving full-count/conditional-CF4 ownership. Any
training-only fit must test heldout counts/CF4 marks and simulation-based
coverage, and a distinct preconditioned/blocked IC sampler must then reach
stationarity before N256. The passed forward screen alone does not justify
another identity-mass HMC path-length trial.

Q-GOAL: correct the observational mapping that informs the same evolved IC
and its z=0 density/velocity state. Q-LEAN: one analytic transfer and one
fixed comparison, without a new gravity simulation, training series or
expanded gate framework. MW/M31/M33 remain latent identities inferred from
each **new** evolved state at R3, with role ambiguity and unresolved M33
retained; no native-truth component IDs or this coarse selection test may
select an LG candidate. Their observables must condition that same state.
