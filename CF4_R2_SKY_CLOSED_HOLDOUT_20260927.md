# R2/5 — sky-closed count and group-mark holdout

The **actual R2 density/velocity posterior is still NO-GO**. This bundle
establishes a reusable observation split; it does not infer a field, calibrate
a galaxy model or validate a sampler. No email was sent.

## Current split: geometry-selected octant 2, graph-closed v5

The initial octant-5 split below was **not used in a field fit**. It had no
FP sky-validation groups. Before examining any field/likelihood residuals,
typed-H100 **406581 COMPLETED/exit0** applied the frozen footprint-only rule:
exclude the LF diagnostic's octant 3, require minimum point/CF4/complete-FP/
matched-TF coverage, and maximize the weakest coverage relative to one eighth
of that sample. Eligible octants were 2, 6, 7; **octant 2** won. This choice
uses only positions/source-group membership, not observed marks or scores.
The prior conditional-LF training used octant 2, so its fitted LF shape must
not be imported as an independent prior for this holdout.

The first octant-2 closure (406582) missed the 114 raw non-FP distance anchors
inside FP source groups. The first same-field job **406583 was cancelled by
the driver after 24 seconds** before a score was produced. The corrected
anchor-inclusive closure **406584 COMPLETED/exit0**, with 3/3 focused tests.
The final direct FP-member graph check **406586 COMPLETED/exit0** also included
3,062 secure 2M++→FP-member edges, 395 from heldout points. It changed
**zero** CF4 or FP roles versus v4; thus the one-prior-IC control begun on
v4 has identical training/validation membership. The v5 roles are:

| v4 object | Training | Heldout sky | Boundary buffer |
| --- | ---: | ---: | ---: |
| 2M++ observed points | 47,542 | 9,696 | 0 |
| CF4 native groups | 16,359 | 2,867 | 87 |
| FP source groups | 6,034 | 764 | 23 |

The 2,827 crossmatch edges of heldout points, all 114 cross-method anchor
edges, and the 3,062 direct FP-member edges participate in the closure.
No training CF4 group attaches to a
heldout count point, and no training FP group contains a withheld CF4 or
anchor mark. The exact full-parent count projection still holds. For any
**new** field fit, use
`/gpfs/kjhan/CF4/z0_density/r2_sky_closed_split_v5/split.npz` and its
`result.json`, never v1–v4. A one-prior-IC source-count+FP numerical control
under the role-identical v4, typed-H100 **406585**, completed in 4m32s.
It checked the modified actual-data target, not an actual-data posterior.

The control evolves one fresh, unconditioned N128/384 Gaussian IC and evaluates
source-selected five-true-K/six-observed-K 2M++ counts and the 6,034 training
FP groups on the **same** PM density/velocity. The 9,696 sky-heldout count
points are excluded from the target; their raw score is recorded only as an
untrained-state readout. Both training and heldout occupied intensity minima
are positive (9.93e-6 and 1.14e-5). The target's IC directional derivative
agrees with finite difference to relative **9.10e-6**. First compiled value
and gradient took 162.38s; whole job 268.21s; peak host 5.40 GiB under a
32-GiB request. The absolute train count/FP/prior and untrained heldout scores
are in the [control result](/gpfs/kjhan/CF4/z0_density/r2_source_sky_joint_control_v2/result.json).
They are **not** heldout prediction, model evidence, a fitted rate/bias/FoG,
stationary field samples or uncertainty. TF-only groups are not yet included.
The earlier old-operator identity-mass HMC nonstationarity remains decisive;
this successful gradient must not be used to revive its chains.

## Earlier octant-5 development split (preserved, not for fitting)

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
That was a limitation, not a reason to relabel the FP training rows as test
data. The v1 split calculation 406577 and v2–v4 are preserved as development
history. The current v5 above replaces them for prospective R2 fitting.

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
while diagnosing sampler stationarity before N256. The v5 split now supplies
764 FP sky-validation groups, but their shared selection/covariance still
needs calibration. A source-calibrated CF4 group-selection/conditional-
distance law is missing. These are substantive scientific barriers, not code
gates.

The current positive power-law/Poisson count response is a development model.
The [Manticore-Local field-level 2M++ analysis](https://arxiv.org/html/2505.10682)
uses a more flexible nonlinear bias relation and an overdispersed generalized
Poisson likelihood; these are motivated alternatives for a *bounded*
training/heldout and independent-mock comparison, not parameters or posterior
samples to copy into our different box, cosmology and CF4 joint law.

Q-GOAL: an honest CF4+2M++ heldout prediction is required for the first
science delivery, the z=0 posterior; the same-state source-selected gradient
is the minimum numerical bridge to fitting that target. Q-LEAN: one
geometry-only choice, graph-closed split and one bounded prior-state forward/
adjoint calculation; no new simulation, posterior claim, broad filesystem
work or external audit. MW/M31/M33 remain
ambiguous latent roles to be identified from each **new** evolved field in
R3; unresolved M33 remains an allowed outcome and native truth identities
cannot seed candidates. Their observables must constrain that same field.
