# R2/5 — 2M++ selection-coordinate repair boundary

Status: bounded source-model comparison completed in Slurm **406563**; it is
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

    m = M + mu(r)
    M_observed = m - mu(s) = M + mu(r) - mu(s).

For every source, classify `m` into the published bright/faint apparent bins
and `M_observed` into the three observed absolute bins; apply observed-radius
support, angular completeness and source survival to that selected mark.
Deposit the source at `s`. Integrate the intrinsic luminosity distribution
over the joint inequalities, including true magnitudes outside the observed
`-25<M_observed<-21` interval because RSD can move them into it. This is a
**marked** point-process specification, not six independent radial weights.
The selection and group membership of any CF4-matched objects must be handled
jointly rather than multiplied again as an independent CF4 group mark.

`src/cf4_r2_observed_magnitude_transfer.py` implements the exact Schechter-LF
conditional bin-transfer geometry, including bright/faint intrinsic tails.
It deliberately does not supply intrinsic luminosity-dependent bias, overall
rate, survey angular map, observed radial cut, CF4 inclusion/covariance, or
source redshift/FoG calibration. None of these can be filled by relabeling the
old six `nbar` values. Its r=s limit must reproduce the existing six LF
fractions, while r!=s can move absolute-magnitude labels.

One fixed-radius test uses the adopted cosmology at true radii 10, 30, 90,
175 cMpc/h and observed shifts -3, 0, +3 cMpc/h. It isolates selection
geometry and is **not** an estimate of the actual survey's velocity law or
posterior bias. Slurm406563 completed in12s, three focused regressions pass,
and its preserved result is
`/gpfs/kjhan/CF4/z0_density/r2_selection_coordinate_control_v1.json`.
The zero-RSD source/voxel comparison agrees exactly at all four radii. At
30 cMpc/h with s-r=+3, 10.47% of *selected LF measure* comes from intrinsic
central bins that change observed absolute-K bin; another8.90% comes from
intrinsically outside `-25<M<-21`. The six-bin LF selection vector differs
from post-RSD voxel selection by8.74% in L1, normalized by the selected LF
measure. At10 cMpc/h, s-r=-3 gives47.41% central-bin migration and53.12%
normalized L1 difference. These are fixed-displacement sensitivity numbers,
not actual 2M++ frequencies, a field-level bias estimate or a full selected
intensity comparison. Velocity distribution, angular completeness, radial
support, intrinsic bias and CF4 overlap are deliberately absent.

## Decision before another field fit

Do not run another long HMC chain on the present partial target. First fit or
externally calibrate the *intrinsic* luminosity-dependent rate/bias and
source redshift/group-selection law (including FoG and the CF4 overlap), then
compare a full source-consistent mock/heldout prediction against the frozen
post-RSD exposure approximation. Only after a target is fixed should a
materially different blocked/preconditioned sampler be tested with independent
starts; no N256 posterior or uncertainty map is licensed by this diagnostic.
The existing identity-mass HMC geometry line is closed.

Q-GOAL: correct the observational mapping that informs the same evolved IC
and its z=0 density/velocity state. Q-LEAN: one analytic transfer and one
fixed comparison, without a new gravity simulation, training series or
expanded gate framework. MW/M31/M33 remain latent identities inferred from
each **new** evolved state at R3, with role ambiguity and unresolved M33
retained; no native-truth component IDs or this coarse selection test may
select an LG candidate. Their observables must condition that same state.
