# Relative FP calibration versus absolute anchor: source-bound follow-up

R2 remains incomplete. The live408296 sampler is a mechanics experiment on
the unchanged .004dex conditional working target; do not edit its target.

## Source facts, not fitted error inflation

[Howlett et al.2022, section5.4](https://arxiv.org/html/2201.03112#S5.SS4)
distinguishes relative CF3 calibration(.004dex), a cosmic-variance component
(.016dex), and an excluded absolute CF3-anchor uncertainty(.0116dex for the
example H0=75+/-2). It cautions against adding cosmic variance twice when
theory already models it, and specifies a nonzero relative calibration mean
in eq25. The public column description identifies `logdist_corr` as the
richness-dependent fit; that label alone does not establish whether the
release already applies the eq25 offset.
[Release description](https://zenodo.org/records/6824749/files/data_description.pdf).

## Driver inference and limits

The absolute-scale component is not represented by merely calling the .004
coordinate a complete zero-point error. This was already listed as a limit;
the explicit source magnitude supplies a concrete route to addressing it,
rather than an arbitrary .01/.02 prior sweep. But neither its applicability
to our fixed-h=.746 reference nor the release's central offset can be silently
assumed. Audit the reference/offset convention before changing data or mean.

If separate relative and anchor Gaussian errors were justified as independent,
their sum could be represented by one Gaussian with variance equal to their
variance sum. This is algebra, not evidence of their independence, not a
CF4-final calibration, and not permission to infer two separate offsets from
this one-indicator subset. At identical physical offset, changing this prior
would change the target; old samples could not be called new-model samples.

Do NOT blindly add .016: the latent IC prior already models in-box cosmic
variance, while finite-box missing modes would need their own window/covariance
treatment. Do NOT combine another CF4 H0 systematic on top of the same
anchor without tracing overlap. No new latent bulk flow, cosmology change,
prior widening, field fit, resampling or heldout evaluation is implemented here.

Source assembly currently imports `logdist_corr` directly. Whether a published
central correction remains to be applied needs direct release/usage evidence,
not a correction guessed from the current residual. Current MW/M31 identities
remain ambiguous, M33 unresolved; future constraints act on the same NEW field.

## Usage cross-check

The [Lai et al. author code, pinned SDSS branch](https://github.com/YanxiangL/Peculiar_velocity_fitting/blob/80f3b5e4f7233971b6fed00f64ba10be4961feb8/fit_pec_vel_wide_angle_SDSS_zero_point_data_local.py)
reads `logdist_corr` directly and declares a .004-squared zero variance.
Its reformat catalogue and our release give identical means for the two
explicitly checked public rows:PGC1233903(.108613),PGC2180626(.027767).
This is a two-row usage check, not a full catalogue validation or proof of
how the source applied its calibration. It does NOT support adding a fresh
central shift blindly. No author code was executed or copied into our target.

Our h=.746 and the source anchor example use different reference numbers.
Whether a small reference-scale mean shift is required must follow the actual
eta/physical-distance convention, not merely inserting log10(75/74.6) by eye.
Keep the reference conversion distinct from uncertainty in that reference.

## Fable advice and a further source correction

Read-only Fable5 returned ADVISE MODIFY. It supports one common measurement-
scale nuisance at fixed cosmology, not per-row error inflation, and the
Gaussian collapse only with explicit independent/same-reference assumptions.
It requests the actual anchor magnitude and mean convention before changing
the target. Adopt that separation; the current mechanics run need not stop.
Its suggestion that a full-column match to another author's file would by
itself settle how the ORIGINAL release calibrated its mean is too strong:
matching files establish usage/provenance, not a derivation of the convention.
No full-catalogue/heldout-mark comparison is launched just for that assertion.

Important additional source check: [CF3 sectionIX and summary](https://arxiv.org/html/1605.01765#S9)
obtains75+/-2 from a restriction on the velocity monopole, rather than purely
from independent distance anchors. SectionVII separates a Cepheid/TRGB
zero-point systematic estimate(.05mag) from TF/SN links, high-z Hubble fitting
and cosmological corrections. Therefore the illustrative .0116dex must not
automatically become an independent Gaussian anchor prior in this field model.

Driver: the .05mag contribution would convert algebraically to .01dex, but
it is not by itself the covariance of the whole CF3->SDSS ladder. Its common
versus differential components and link overlap need explicit treatment.
Nor should the total published H0 error be copied wholesale as a distance-
scale prior. Keep the live .004 conditional baseline unchanged; the source
follow-up has narrowed what must be modelled, not supplied a complete new
calibration or an authority to broaden the prior until the fit looks good.
