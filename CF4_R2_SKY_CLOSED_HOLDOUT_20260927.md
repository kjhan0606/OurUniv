# R2/5 — sky-closed count and group-mark holdout

The **actual R2 density/velocity posterior is still NO-GO**. This bundle
establishes a reusable observation split; it does not infer a field, calibrate
a galaxy model or validate a sampler. No email was sent.

The older N128 live-field pilot used all 57,238 eligible 2M++ points, so its
scores cannot become heldout predictions after the fact. The old pointwise
hash split also allows a CF4 training group to carry the redshift of a heldout
matched 2M++ member. The new split holds out all observed N128 voxels in
observer-centred supergalactic octant **5** (the earlier LF-only diagnostic
used octant 3). The observer lies exactly on a voxel boundary at 192 cMpc/h;
the window is a spatial mask, **not** a random catalogue thinning fraction.
The same six-bin predicted intensity must be integrated over *only* the
training voxels for a fit, or only the heldout voxels for a prediction.

Slurm **406580 COMPLETED/exit0** in one second. The frozen
[`result.json`](/gpfs/kjhan/CF4/z0_density/r2_sky_closed_split_v2/result.json)
and `split.npz` preserve source-bound identities and masks:

| Object | Training | Heldout sky | Boundary buffer |
| --- | ---: | ---: | ---: |
| 2M++ observed points | 53,838 | 3,400 | 0 |
| CF4 native groups | 17,413 | 1,877 | 23 |
| FP source groups | 6,820 | 0 | 1 |

There are 262,144 heldout N128 voxels. All 910 CF4 crossmatch edges attached
to heldout 2M++ points withhold their CF4 group mark from training. CF4↔FP
source-group closure then withholds linked FP marks. No training CF4 group has
a heldout point edge, and no training FP row is linked to a withheld CF4 group.
The six-population train+heldout sparse counts reproduce the frozen full
parent counts exactly. Boundary-crossing CF4/FP groups are excluded from
both mark training and mark validation. Octant 5 does not intersect the
selected FP group sky sample; hence **this split has no FP sky validation**.
That is a limitation, not a reason to relabel the FP training rows as test
data. The v1 split calculation 406577 was retained, but v2 tightened the
boundary-buffer classification before use; future code should read v2 only.

The source-selected Poisson kernel accepts an exact per-voxel sky window.
It includes empty *selected* voxels in the integral, rejects an observed
count key outside the window, and never rescales the intrinsic rate by a
global 7/8 fraction. Two non-JAX split tests passed in 406580. The first
JAX test allocation 406578 failed immediately because the dedicated JAX
environment lacks pytest; it did not test the numerical change. A bounded
`unittest` rerun in the correct environment **406579 COMPLETED/exit0**:
all six marked-operator tests passed in 1m44s. Its peak MaxRSS was
3.52 GiB under a 4 GiB request, so this test job should request at least
5 GiB if repeated; the split-only job was much smaller.

This is not a pristine independent survey test: the same actual catalogue
and source laws have informed previous development, and octants are spatially
correlated. It *does* prevent the forthcoming fit from directly consuming
these 3,400 count observations and their connected CF4/FP group marks.
Historical CF4 and row-hash holdout labels remain available for their old
diagnostics; the new fit must not silently mix split definitions. The LF
conditional score already uses observed-K bin frequencies; it must not be
multiplied by the six-bin count score over those same observations.

Next: fit a **single** source-selected marked-count and CF4 conditional-mark
model on these training roles, carrying LF shape, intrinsic rate, luminosity
bias and FoG/selection uncertainty on the same evolved IC/current field.
Then score the untouched sky count/CF4 marks and independent survey mocks,
while diagnosing sampler stationarity before N256. FP validation needs a
separate, graph-closed source footprint test; it is not supplied by this sky
octant. A source-calibrated CF4 group-selection/conditional-distance law is
still missing. These are substantive scientific barriers, not code gates.

Q-GOAL: an honest CF4+2M++ heldout prediction is required for the first
science delivery, the z=0 posterior. Q-LEAN: one small source-bound split
and an exact window in the existing Poisson kernel; no gravity run, new
posterior claim, broad filesystem work or external audit. MW/M31/M33 remain
ambiguous latent roles to be identified from each **new** evolved field in
R3; unresolved M33 remains an allowed outcome and native truth identities
cannot seed candidates. Their observables must constrain that same field.
