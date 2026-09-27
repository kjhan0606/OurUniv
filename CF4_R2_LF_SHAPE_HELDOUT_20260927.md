# R2/5 — source LF shape and marked-count parameter boundary

Status: one predeclared sky-heldout luminosity-function check completed in
typed-H100 Slurm **406570**; the follow-up LF-gradient regression **406573**
completed in 1m39s, with three tests passing.
This is not a calibrated 2M++ count likelihood or a CF4-conditioned field.
No email or author contact was used.

The [2M++ source paper, Table 2](https://arxiv.org/html/1105.6107) reports
different Schechter shapes for different fitted samples. Its default CMB
`-25<M_h<-21`, `5000<cz<20000 km/s` row has `alpha=-0.73`,
`Mstar=-23.17`, whereas the imported `|b|>10, K<11.5`, low-redshift,
`-25<M_h<-17` row has `alpha=-0.94`, `Mstar=-23.28` and no tabulated
`nstar`. Neither shape and normalization is an independent prior for our
same-catalogue count likelihood. In particular, the source notes a faint
shape inflection near `M_h=-21`; the imported single Schechter law is not
automatically a calibrated true-luminosity law for all five intrinsic bins.

The one-split check used the existing 57,238 eligible 2M++ parent rows and
the frozen CF4/PM cosmology, restricted to 49,489 rows in the source's default
redshift/absolute-K window. Octant 3 in the observer-centred supergalactic
box (8,368 rows) was untouched by the two-parameter maximum-likelihood fit
to the other 41,121 rows. At each observed redshift and apparent-K class,
the Schechter density was normalized on the intersection of the class and
`-25<M_h<-21` boundaries. This conditional calculation uses the paper's
two-class angular-completeness assumption under which a class-constant map
factor cancels. It does not model real-space RSD, luminosity-dependent bias,
field-dependent selection, CF4 overlap or the overall rate.

| Shape | Heldout total conditional log density | Heldout PIT KS D |
| --- | ---: | ---: |
| Imported low-z bright `(-0.94,-23.28)` | -294.4833 | 0.01732 |
| Paper default `(-0.73,-23.17)` | -291.8857 | 0.01259 |
| Fit on seven octants `(-0.77164,-23.16801)` | -294.4240 | 0.01632 |

The paper-default shape improves this one heldout total by only 2.60 log
units over the imported shape across 8,368 correlated galaxies. This is too
small and too dependent on one sky split/source catalogue to freeze a new LF
or reject the old one. The training-fitted shape is not a production prior:
using it as a prior and then scoring the same galaxies again would reuse data.
The actual result is
`/gpfs/kjhan/CF4/z0_density/r2_lf_conditional_skyholdout_v1/result.json`.
Two focused integration/interval tests passed in the same 9-second job.

The source-selected GPU operator now accepts `alpha` and `Mstar` and exposes
the corresponding five intrinsic-bin LF fractions so the source masses and
redshift-space bin transfer use the *same* shape. The first additional
gradient test in **406571** found nonfinite `Mstar` reverse derivatives at
unbounded intrinsic-bin edges. Replacing the inactive infinite arguments to
incomplete gamma with finite safe values made the same three tests pass in
**406572**. The installed JAX 0.10.1 has a shape-parameter gradient rule;
406573 additionally checked `alpha` against a finite difference and passed;
406574 checked the imported near-`alpha=-1` reference and passed. The same
module now builds five luminosity-dependent, unit-spatial-mean density
responses under one intrinsic log-rate, and evaluates the selected six-bin
Poisson factor from sparse occupied keys plus the **full** empty-voxel
intensity sum. Typed-A100 **406576** completed in 2m04s with all five focused
regressions passing, including the rate/bias mass and sparse-Poisson checks. No
hidden positive-intensity floor or observed-count renormalization is added.
This is numerical wiring, not a prior or inferred LF posterior.

Next R2 action: use the observed-K information exactly once in a joint
source-selected count/mark law, carry LF shape and intrinsic rate/bias/FoG
uncertainty into the **same evolved IC/current field**, and establish
training/heldout and independent mock coverage before N256 sampling. A
conditional K score over *apparent class* already includes absolute-bin
frequencies and must **not** be multiplied by the six-bin count score; that
would double count those data. The existing CF4 selected-group/point law and
identity-mass HMC nonstationarity remain separate barriers; no development
maps are promoted.

The older live count pilot consumed **all** 57,238 eligible galaxies. It
therefore has no independent 2M++ count holdout, even though frozen row split
flags exist. A future field fit must use a declared count training subset and
its matching thinning intensity, retain a genuinely unscored count holdout,
and close CF4 group/point links across that split so a training group mark
cannot leak a heldout member's redshift. This is essential before claiming
heldout prediction, not a reason to discard the pilot's numerical checks.

Q-GOAL: test the actual survey's luminosity-shape assumption that controls
which same-field sources enter each observed count bin. Q-LEAN: one 9-second
source-only split and reused GPU regressions, no gravity run, new sampling
series or validation framework. MW/M31/M33 remain latent roles identified
from each **new** evolved state at R3, with unresolved M33 retained; this
LF check neither supplies identities nor constrains the LG directly.
