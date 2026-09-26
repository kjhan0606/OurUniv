# C: environment-conditioned multiscale spatial model — proposed implementation

Designed after user-approved diagnosis336263,2026-09-08. DESIGN ONLY; no new
training or inference authorized by the diagnosis approval. Request approval
for the bounded implementation pilot below. C remains open, D unentered.

Subsequent user approval2026-09-08 authorizes the one implementation/GPU pilot.
Frozen implementation details BUNDLE_C_FLOW_PILOT_RUN.md supersede the tentative
interior chart/architecture mechanics below: equivalent binary-tree moment
coordinates explicitly handle atoms and inactive states. Larger scope remains
unchanged; no actual-data or global inference is authorized by this pilot.

## Decision and evidence

Choose ONE conditional3D multiscale normalizing-flow candidate, not another
stationary Gaussian-copula fit. This is a recommendation, not a conclusion
that this ML family has been validated for CF4. Diagnosis shows negligible
density roundtrip loss but large generated-field errors and a separate lossy
scalar-variance representation. Implement explicit cross-scale/environment
conditioning and lossless moment coordinates together. Do not train on the
old five-channel scalar-sigma target.

[Dai & Seljak2024](https://arxiv.org/html/2306.04689v2) demonstrate hierarchical
conditional wavelet flows and field probabilities for TWO-DIMENSIONAL weak
lensing. That supports the multiscale probability factorization, not our3D
total-matter/halo/velocity application or small-data calibration. Select flows
for explicit conditional density evaluation and posterior use, not guaranteed
sample quality. Their paper also distinguishes realistic samples from
well-calibrated parameter posteriors.

A3D conditional diffusion alternative has cosmological precedent:
[Rouhiainen et al.2024](https://arxiv.org/html/2311.05217v2) model TNG300 GAS
density on a264^3 mesh in a205 cMpc/h cube, conditioned on a separate LR DMO
simulation, with adjacent-volume conditioning. It is not a demonstrated
.1875 total-matter/velocity or LG-observation model. Reported training was70h
on two A100s. Do not reproduce that programme or launch another diffusion
comparison here. These papers inform architecture, not our likelihood.

## Target probability and actual-data connection

Use present-day states throughout. B is a12-grid environment state, C the
1.5-grid environment, F the fine LG total-matter field, S a disjoint member/
remainder decomposition. H(S,F) is its calibrated observational readout
(positions, member COM velocities, masses/proxies); nuisances are eta.
A proposed target is

    p(B) q_E(C | B) q_F(F | C) q_S(S | F,C) p(eta)
      * L_CF4(C,F,eta) * L_counts(C,F,eta) * L_LG(H(S,F),eta).

All q terms must be normalized generative laws, including boundary/occupancy
branches. The observation-conditioned JOINT distribution couples LG and
surrounding matter, not a saved mark posterior plus an independent fine field.
LG likelihood applied to q_S allows observations to update F. No three fixed
truth halos. An equivalent factorization conditioned on explicit H is allowed
only with its normalization and consistency with S specified, not assumed.

The existing12-grid actual-data posterior is a diagnostic initialization or
proposal, NOT p(B). Use the original present-field prior and native likelihoods
once, or an explicitly corrected proposal ratio. Never multiply that posterior
by the same data likelihood again. Its diagnostic calibration limits remain;
a fine conditional does not scientifically certify its large scales.

q_E is currently MISSING: native observations/selection at1.5 are input tables,
not a1.5 posterior. Coarse-to-fine changes must use native CF4/count positions
and responses, not a likelihood frozen on B. Existing grid origins/selection/
units remain binding. Frozen CF4 rows do not directly sample LG R2; LG
observations provide that information. Maintain the384 domain/environment;
a24-cube model alone is not that domain.

End-to-end product: LG-on/off posterior draws with the SAME joint prior,
surroundings1.5 and LG.1875 numerical cells, physical sigma distinct from
posterior uncertainty, and observation-space checks. No IC/zoom delivery
until the present-field stage and subsequent dynamics validation are adequate.

## Representation and architecture

Use factor2 refinement: environment12→6→3→1.5; LG1.5→.75→.375→.1875.
This is a numerical hierarchy, not six approvals or six model experiments.
A shared scale-conditioned implementation has physical scale as explicit
input; do not assume density statistics are scale invariant.

Each parent splits into eight child extensive7-moment states. Keep child
mass, three mean velocities and THREE diagonal physical variances; zero-mass
cells explicit. Scalar sigma=sqrt(trace variance/3) is derived. No arbitrary
rotations of a diagonal tensor: axis permutations/sign flips must transform
vectors/diagonals appropriately, or cross moments must be acquired later.

For an interior parent with child weights w_i>0, sum w=1, parent mean mu and
per-axis variance V, use seven mass-simplex coordinates. For each axis:

    r = t / sqrt(1 + dot(t,t)), t in R^7
    u_i = sqrt(V/w_i) (U r)_i
    child_mean_i = mu + u_i
    child_variance_i = V (1 - dot(r,r)) a_i / w_i

U is an8x7 orthonormal basis perpendicular to sqrt(w); a is an eight-way
simplex. There are49 continuous degrees of freedom per positive octet:
7 mass +3*(7 mean +7 intrinsic-variance allocation). Parent M/P/Qdiag are
conserved without discarding directional fine variances. Native inverse
exists on the interior. Train ON these conservative coordinates, with decode
inside the model, not unconstrained channels plus an untrained renormalization.
Include the coordinate Jacobian when claiming a physical-coordinate density;
latent inference must not invent an unconstrained56-dimensional density on
this49-dimensional constraint surface.

Empty children, zero internal dispersion and cold parents are discrete/lower-
dimensional branches, not values for an arbitrary floor. Model valid masks
explicitly, flow only on active degrees of freedom, deterministic cold/empty
limits. Basis uses active weights only. If native roundtrip cannot handle
these cases cleanly, stop before training; never drop failing physical cells.

Compact3D convolutional coupling networks condition on neighbouring coarse
density/mean velocity/three variances and scale, not just one cell's total.
Spatially coupled layers span multiple parents to learn surrounding structure
orientation. One fixed architecture (proposal: six coupling blocks,32/64
channels), conditional negative log probability training, no heldout-driven
P(k) correction. Probability fit is not physical validation. Halo information
enters through q_S; q_F and q_S jointly determine LG-conditioned structure.

For q_S, use native-calibrated nonnegative four-way component allocations
(MW member, M31 member, M33 member, everything else), whose extensive moments
sum to F. Generalize the conservative representation to active components.
Labels are disjoint SUBFIND members, not inclusive M200c spheres. Reuse native
profiles and COM/mass proxy calibrations; these are not extra observations.
Member mass/COM derive from S. Inclusive host mass is a distinct total-field/
profile readout with discrepancy, not bound mass. Never copy catalogue identity
onto changed fields. This statistical decomposition does not prove generated
objects are bound or dynamically viable; later dynamics testing is essential.

## Data, feasibility and bounded next implementation

Reuse cached400^3 seven moments and native member products; no raw snapshot
pass or new simulation. General q_E/q_F training samples the available native
volume, not only13 LG-selected cubes. Member calibration uses93 components
and the native catalogue. The one75-cube source has limited independent LG
systems/long modes: overlapping crops, rotations and51971 correlated triples
cannot be counted as independent universes.

Before extracting crops reserve geometry-only disjoint validation region(s),
excluding all prior training and diagnosis voxels including context. Check
geometry in the data job, not filesystem properties. If no untouched region
with required context exists, label it reused-development, not independent
validation; seek a specific extra source only if scientific promotion needs it.
No new heldout label through rotation. Halo-ID and field-voxel leakage differ.

Next approval covers ONE implementation pilot: lossless coordinates plus
conditional flow, one fixed fit on cached data, retained mock density/velocity
samples, spatial diagnostics and measured resources. Native roundtrip precedes
fitting in the same job; no generic audit framework or scalar-sigma-only release.
Test that conditioning changes structure, not only totals. Reuse P(k)/hot-region
statistics and add mean-velocity/directional-sigma and parent-boundary checks;
inspect individual fields, not only means/loss curves. Used diagnosis cases
are stress cases, not fresh validation. Freeze new criteria/split before fitting.
No automatic hyperparameter search or time/seed extensions on failure.

Proposed pilot cap: one Slurm GPU,4h total, estimated host peak20 GiB+20%=
24 GiB request; GPU target<=20 GiB plus20% headroom=24 GiB. Design estimates,
not measured performance. Bounded patches/batch1 and an initial within-job
timing/memory segment count against the same cap. No global384-domain inference
or promise of convergence within4h. Measure cost before pricing real inference.

q_S and full-domain q_E posterior integration remain indispensable, NOT solved
by the first q_E/q_F pilot. Global evaluation must define ONE normalized law:
implementation tiles can share context, but do not multiply overlapping crop
probabilities or independently stitch maps. Coherent global coupling evaluation
or normalized conditional outpainting is needed before deployment. Local pilot
success does not establish full-domain feasibility.

After pilot success AND explicit q_S/global-likelihood integration, release
actual CF4/count+LG-on/off density/velocity maps, not more oracle-only demos.
If the required joint law or resources cannot be supplied, report the missing
component rather than paint a map. New simulations, additional prior families,
D and production inference are deferred. Present approval ends with this design.
