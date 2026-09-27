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
does not. The count operator integrates over Gaussian-Hermite LOS nodes and
does **not yet expose** the continuous `r` or the source-to-key contributions
needed by this expression. This formula therefore specifies the ownership
contract, not an implemented likelihood.

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

## Next in-scope implementation

Expose source-to-observed-key contributions from the **same** K/selection/RSD
count transfer and a continuous individual-redshift density whose integral
reproduces that transfer. First connect the 1,836 training one-link groups
without altering the other groups' scientific label. Test count/conditional
normalization and finite IC derivative on a small fixed-state control, then
assess sensitivity on the frozen training split. Extend to the 273 training
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
