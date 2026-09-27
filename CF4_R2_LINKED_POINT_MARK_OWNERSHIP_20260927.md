# R2/5 — ownership of linked 2M++ points and FP distance marks

The present-day CF4+galaxy density/velocity posterior remains **NO-GO**. This
is a source-bound factorization decision, not a field fit, selection
calibration, or permission to restart the nonstationary N128 sampler. No email
or external contact was made.

## Frozen-source graph, without reading heldout distance marks

The v5 graph-closed split, point manifest, FP source assembly, and secure
`data/cf4_2mpp_crossmatch_v1.csv` links give 3,062 distinct counted-point →
FP-source-group edges. All 3,062 counted points have degree one; 2,413 FP
groups have at least one direct edge. The complete group-degree histogram is
`{1:2099, 2:200, 3:54, 4:27, 5:11, 6:6, 7:4, 8:2, 9:1,
10:2, 12:3, 18:1, 19:1, 25:1, 31:1}`. The graph-closed roles are:

| FP source-group role | One direct point | Multiple direct points | No direct point |
| --- | ---: | ---: | ---: |
| Training | 1,836 | 273 | 3,925 |
| Sky heldout | 253 | 40 | 471 |
| Buffer | 10 | 1 | 12 |

These counts use **direct secure FP-member links**, not all CF4 group/2M++
associations. They do not imply the unlinked groups are physically outside
2M++. The linked points occupy 2,790 distinct observed population/voxel keys;
an observed key need not contain exactly one galaxy. No direct edge crosses
from a heldout point to a training FP group or conversely. These are source
identity and split checks only; no FP eta, distance mark, field score, or
heldout residual was used to choose a model.

## One coherent candidate law

For observed population/voxel key `k`, let `j` index true source cell, true-K
bin and LOS-displacement state. The current source-selected count operator
provides the Poisson mean

`Lambda_k(F) = sum_j lambda_j(F) T_{j→k}(F)`.

For one **identified counted point** in key `k`, its conditional source-state
probability is proportional to `lambda_j T_{j→k}`, not to a separate generic
`d² rho^b` group prior. If its measured individual redshift `z` is retained
as a *conditioned covariate*, a normalized conditional FP mark factor must
instead use

`p(m | k,z,A,F) = [sum_j lambda_j T_{j→k} a(A|j,k,F)
                       r(z|j,k,A,F) L_FP(m|j,z,A,F)]
                 / [sum_j lambda_j T_{j→k} a(A|j,k,F)
                       r(z|j,k,A,F)]`.

Here `A` is the observed association/selection graph, `a` its conditional
occurrence probability (or an explicitly stated field-independent design
assumption), `r` is the conditional individual-redshift density, and `L_FP`
is the **existing source-corrected**
FP PDF, not a second selection correction. A redshift-only eligibility cut
constant at fixed observed `z` cancels; latent-distance-dependent inclusion
does not. The count operator integrates over finite Gaussian-Hermite LOS
nodes, so its continuous-redshift representation needs a separate quadrature
accuracy check. The formula is the ownership contract, not a calibrated
all-group likelihood.

For a multi-point FP group, multiplying single-point expressions would treat
its shared distance/velocity/FP calibration as independent. One group-COM
latent with possible member offsets and member-redshift covariance must sit
**inside one group integral**;
the observed count occurrences remain scored once by the Poisson factor.
Groups without a direct counted point need a separate selected-group
conditional law and its inclusion/model-discrepancy uncertainty. Source FP
`f_n` correction, group incidence, and 2M++ selection are distinct. The
public SDSS random `n(z_obs)` cannot be substituted for the missing
true-distance group inclusion probability.

The old count × normalized FP-mark product is a declared **partial/composite
target**: because the mark factor is normalized conditional on its supplied
redshifts, overlap alone does not prove literal duplication of a count.
However, its `d² rho^b` distance weights are not demonstrated to equal the
count-conditioned source distribution above, and it omits a calibrated
`p(U,A|C,F)` for individual redshifts/positions and group incidence. Thus
neither the product nor the one-to-one repair alone is a production joint law.

## First implemented component and one actual-source control

`src/cf4_r2_marked_tracer_jax.py` now exposes the five-true-K-by-source
contribution to a selected observed count key, using the **same**
K/selection/RSD transfer as the full count operator. A periodic TSC gather is
the transpose of its deposit stencil. A separate continuous Gaussian-LOS
radial-key density, including both signed branches through the observer,
provides the source mixture at one measured individual redshift. Its
integral equals the count-key law only after adequate LOS quadrature and
negligible periodic-image aliases; the existing GH3 count rule is an
approximation, not an exact continuous-redshift marginal near cuts. A
normalized one-link FP likelihood-ratio kernel accepts an explicit
association log probability instead of silently assuming a second count
likelihood or calibrated FP inclusion.

Typed-H100 Slurm **406603** passed eight source/operator tests; **406605**
passed nine, adding a smooth-case continuous-radial integral and conditional
normalization control. The source-bound one-group **406607 COMPLETED/exit0**
in4m22s (nine tests included; batch MaxRSS4,202,616 KiB). Before viewing FP
scores, the first lexical source-group ID among the1,002 training groups
with exactly one directly counted point, one FP row, no anchor and point
distance30–120 cMpc/h was frozen: `P1085367`, point index26,507,
population1, voxel(48,86,72), observed radius83.9233 cMpc/h. On the saved
**unconditional** N128/384 PM state, the full count mean and summed source
contribution at its observed key agree:

`0.012438174302271256` versus `0.012438174302271251`.

Its continuous observed-radius key density is0.0023156943 per cMpc/h. With
the association factor fixed to a field-independent constant and FP/global
offsets fixed at zero **only for this mechanics control**, the conditional
one-FP log-likelihood ratio is-0.3097834203. The derivative with respect to
a common source-velocity multiplier is-0.4686775166 versus centred finite
difference-0.4686775148 (relative discrepancy1.80e-9). These are one-state
numerical values, **not** a likelihood comparison, calibrated association,
posterior information gain, or evidence that the observed group is well fit.
The FP group redshift may differ from its linked individual 2M++ redshift;
their source covariance/selection is not yet in this factor. Heldout FP
marks were not scored. The small test's continuous-vs-GH3 integral agreement
within2% does not establish this accuracy for actual N128 keys near survey
or K boundaries. Job artifacts:
`/gpfs/kjhan/CF4/z0_density/r2_linked_fp_one_state_v1/result.json`.
The post-control code added the second signed LOS branch through the
observer; **406610 COMPLETED/exit0**, nine tests passed including an explicit
non-negligible opposite-branch analytic check. Job406607 preceded that
addition, but its point lies at83.923 cMpc/h with a 1-cMpc/h LOS width,
so the omitted opposite-branch term underflows to zero there. Preserve the
original result and this source-version distinction; no changed science
score is claimed from the branch test.

The reference calculation still traverses all2,097,152 source cells for
one point. A fixed-state single-point timing is not an all-group posterior
cost estimate; applying it naïvely to1,836 links at every HMC step would be
inappropriate. A source-to-key sparse/ray neighbourhood with a check against
the full count intensity, or an equivalent exact collective operator, is
needed before scalable live-field inference. The missing group association,
selected-group incidence and shared redshift/FP covariance must still be
calibrated or bounded from source-backed mocks and heldout prediction.

## Next in-scope implementation

Make the linked factor scalable across the1,836 training one-link groups,
checking both its support and GH3-vs-continuous marginal error on the frozen
training split; do not select by FP score. Specify or bound the conditional
association/point-redshift law and shared zero/FP covariance before fitting
a live IC. Extend to the273 training
multi-link groups using one shared group latent, and address the 3,925
unlinked training groups with an explicit selected-group model before an
all-method field fit. Sky-heldout FP marks remain untouched for prospective
assessment. No identity-mass HMC extension or N256 promotion while the
target and group-selection covariance are unsettled.

Q-GOAL: ties the available galaxy density and FP distance information to the
**same** evolved field without claiming unobserved high-k phases. Q-LEAN:
uses frozen source products and one exact conditional factor first, not a
new gravity run, full mock download, audit ladder, or email. MW/M31/M33 remain
latent identities on each **new** evolved field, including MW/M31 ambiguity
and unresolved M33; their observables must constrain that same field. Native
truth IDs may only label calibration/evaluation, never select candidates.
