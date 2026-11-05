# R2 CF4 group-selection / conditional-mark bundle — 2026-09-26

## Why this bundle is on the critical path

The first science delivery remains an actual CF4+2M++ conditioned z=0
density/velocity posterior from the same latent LCDM IC state. The previous
bundle established only the coarsened N128 2M++ count factor. It did not
identify the CF4 distance-group inclusion probability, the CF4 group
velocity conditional on overlapping 2M++ observations, or a distance-modulus
mark likelihood. Multiplying inclusive counts by the old grouped BGc score
would therefore still be unsupported.

This bundle combines three decisions in one bounded source calculation:

1. Construct the frozen *observed* CF4-to-2M++ GID bridge using eligible
   secure edges, preserving one-to-many and many-to-one cases.
2. On mutually one-to-one observed pairs, test one train-fitted, heldout
   Student-t group-redshift conditional using the 2M++ group velocity and its
   published dispersion/richness. This is a restricted empirical control,
   not a new group identity rule or a field-linked velocity likelihood.
3. Compare CF4 distance-method composition and determine whether unlinked
   2M++ GIDs really provide a denominator for *CF4 distance inclusion*.
   They do not merely by existing: the catalogues define groups differently,
   crossmatch coverage is incomplete, and CF4 methods have heterogeneous
   selection. An observed association rate must not be relabelled as a
   distance-measurement probability.

Q-GOAL: these are the overlap/selection dependencies blocking the R2 joint
observation law and hence the first present-field posterior. Q-LEAN: one
source-bound Slurm job, one prespecified velocity candidate, the existing
native CF4 holdout, no new gravity run, sampler, seed search, repeated
hyperparameter trial or general-purpose gate framework. The numerical N128
count score is not being repeated.

R3 connection is explicit even though this calculation is R2: MW/M31/M33
hypotheses must be generated from **each new evolved state** without native
truth IDs. MW/M31 role ambiguity and unresolved M33 remain branches; observed
positions/distances/masses/velocities must constrain that same state. Neither
a 2M++ GID nor a CF4 group ID is an oracle MW/M31/M33 component label.

## Source meaning and planned decision

The [CF4 source paper](https://arxiv.org/pdf/2209.11238) describes a
heterogeneous distance-indicator assembly. Its grouped distance modulus
combines measured-distance members, while the group velocity uses all known
group velocity members. The paper also documents method-dependent selection
and Malmquist corrections (notably FP). The [2M++ source paper](https://arxiv.org/pdf/1105.6107)
defines a separate magnitude/redshift-selected catalogue and grouping.
Consequently a 2M++ GID is not a CF4/Tully group identifier or a universal
unobserved-group parent sample. In particular, a CF4 selected-only train/
holdout split cannot identify the missing CF4 group-inclusion denominator.

The frozen trial is `delta Vcmb = CF4 group Vcmb - 2M++ group Vcmb`, with
fixed Student-t df=4 and scale
`sqrt(fitted_floor^2 + (2M++ group sigma/sqrt(Rich))^2)`. The floor and
location use *only* the native CF4 training split. The heldout 90% and 95%
coverage screen is within two binomial standard errors of nominal. Passing
would support only this restricted conditional's predictive width, not its
field dependence, CF4 selection, distance-mark calibration, independent-sky
coverage, or the full joint likelihood. Failure closes this candidate without
retuning on the holdout.

Execution: Syntax typed `h200`/`gpu:H200:1`, 1 CPU, 4 GiB, 10-minute cap;
source `scripts/cf4_r2_mutual_group_conditional.py`. The initial Slurm405888
and downstream 405893 results are preserved as **superseded**: the initial
mutual-pair graph was restricted to native CF4 nodes and missed links to
non-native CF4 groups. Corrected source jobs **405895 and dependent 405896**
both COMPLETED/exit0. The v2 [source bridge and velocity result](/gpfs/kjhan/CF4/z0_density/r2_mutual_group_conditional_v2/result.json)
and [frozen stratification](/gpfs/kjhan/CF4/z0_density/r2_mutual_group_stratified_v2/result.json)
are authoritative; no output was overwritten.

## Results and interpretation

The corrected source-bound result contains 3,912 eligible 2M++ GIDs. The
secure eligible links reach 2,549 native CF4 groups; 158 of those touch
multiple GIDs. Conversely 340 mapped GIDs touch more than one **any** CF4
group. Requiring one-to-one correspondence in **both** directions across the
full canonical CF4 graph leaves 1,759 native pairs. Of the 14,878 secure eligible
edges, 8,824 lack a 2M++ GID at all. Another 1,430 eligible GIDs have no
secure link to **any** canonical CF4 group; they are not known CF4 distance
non-detections. These numbers rule
out simply dividing matched by all 2M++ GIDs to obtain a CF4 group-selection
probability.

The source-defined method mix differs: among 19,313 native CF4 groups the
largest exact patterns are FP-only 9,149 and TF-only 8,503, whereas the
mutual-pair subset has FP-only 891, TF-only 387 and FP+TF 252. It is therefore
not a representative surrogate for all distance methods. The subset median
published `e_DMzp` is 0.36 mag; the existence of a quoted error is not a
calibration of selection or shared systematic errors.

The one prespecified Student-t trial used 1,389 training and 370 heldout
mutual pairs. Its fitted location is +1.76 km/s, scale floor 43.65 km/s.
Heldout nominal 90%/95% interval coverages are **87.57%/93.51%**; both lie
within their prespecified two-binomial-SE screens (3.12/2.27 percentage
points), but are low and the sample is a correlated, non-independent sky
subset. This is only a weak predictive-width result for one restricted
observed pairing, not validation of the group velocity law or a field-linked
mark factor. The earlier single-member individual-redshift conditional
failure cannot be overwritten by this different subset.

A frozen-parameter method/depth decomposition completed in typed-H200 Slurm
**405896/exit0** from the saved 1,759 pairs; it did not refit or select a new
conditional model. The corrected stratified result
shows that the apparent aggregate pass hides a depth-dependent failure:
among 121 heldout groups with `Vcmb >= 10,000 km/s`, nominal 90% coverage is
**80.99%**, 9.01 percentage points low versus a 5.45-point two-SE width;
nominal 68% coverage is 57.85%, also more than two SE low. On the training
split, the 691 FP-only groups have 83.50% coverage for nominal 90%, 6.50
points low versus 2.28-point two-SE width. These correlated, multiple
descriptive checks do not furnish a formal independent-volume hypothesis
test, but they reject using the aggregate screen to call the velocity
conditional calibrated. **Close this single trial; do not inflate its scale
on the holdout or run an unconstrained model sweep.**

The bundle decision is **NO-GO for this restricted velocity conditional as a
calibrated law, and NO-GO for a complete R2 likelihood or N256 inference**.
CF4 method-dependent distance-mark selection/inclusion and same-state
group/member dependence remain uncalibrated.
The next defensible route must either obtain method-specific parent/selection
information or define a narrower, explicitly conditional selected-sample
target and bound its selection bias. The latter would be a diagnostic
approximation, not the full actual-CF4 posterior.

The next *implementation* bundle should start with the dominant FP-only and
TF-only group strata, locate their published selection-corrected distance
products and parent/forward-selection definitions, and determine which
per-object normalization survives CF4 grouping. It should then build one
same-field conditional distance-modulus factor with method/shared-zero-point
uncertainty and test its predictions on frozen heldout sky/depth strata or
source-consistent survey mocks. If the required parent/selection products are
not available, state that failure and limit the inference target explicitly;
do not treat a selected-only CF4 split or unmatched 2M++ GIDs as the missing
denominator. Only after that factor and the overlapping redshift dependence
are defensible should N128 sampling be considered, followed by N256.
