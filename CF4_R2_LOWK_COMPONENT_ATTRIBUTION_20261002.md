# R2 low-k likelihood-component attribution — 2026-10-02

## Decision and audit

The preceding saved-state projection (job410046) finds that the partial exact
N256 target's likelihood gradient locally drives all three measured axial
fundamental amplitudes farther outward at both accepted endpoints. It is not
evidence that calibrated CF4 observations physically require those modes.
Fable5's read-only audit returned **CONDITIONAL PASS** for the sign/normalization
and for one component attribution. The driver independently checked the target
and source, adopts the bounded attribution with Fable's conditions, and does
not adopt its speculative random-phase probability as a p-value.

The audit also emphasizes that the likelihood force varies materially between
these two nearly identical low-k states; their six signed derivatives are not
independent confirmations. The three q phases are all near `pi`, producing a
descriptive observer-centred latent cosine pattern under the existing
corner-origin convention. It is not a density feature or a calibrated
significance. For a proper target, the ensemble identity is
`E[q_i * dU/dq_i]=1`; hence no single-state zero-gradient threshold is used.
The prior result JSON field called `likelihood_gradient` is in fact
`grad(-log_likelihood)`.

Driver adopts the audit's required checks: replay the same exact energy,
reconstruct the saved total gradient from both terms, report complex mode
vectors/radial and phase derivatives, all-mode IC gradient RMS, and each of
the24 nuisance-gradient coordinates. The component breakdown is conditional
on each checkpoint's fixed nuisance values, so it cannot by itself establish
which observational data are physically correct. No held-out values are read.

Additional driver interpretation: the phases of the three q modes are all
close to `pi` (A offsets about3,16,7 degrees; B about3,16,7 degrees), so their
cosine parts align near the centred observer under the existing grid-origin
convention. This is a descriptive latent-white-field phase pattern, not a
physical density peak. The A/B states share low-k lineage; their gradients
are materially state-dependent, so six negative radial derivatives are not
six independent confirmations. One endpoint's force cannot be treated as the
mean force under the posterior.

## Smallest next calculation

The saved exact-GL2 artifacts contain only `canonical`, its **summed**
`fine_gradient`, target energy and chain metadata; they contain neither the
z=0 density/velocity nor separate component gradients. Attribution therefore
requires one deterministic PMWD forward replay at each of the SAME two saved
IC states, then separate adjoints for the exact active score components:

1. the existing selected 2M++ voxel-count factor;
2. the existing raw selected CF4 FP-distance-mark factor.

The active exact N256 target does not include the other CF4 TF/SNIa/SBF
methods; they remain outside this attribution. The two shared tracer nuisance
blocks must be reported, including their conditional-on-fixed-state
interpretation. Held-out observations stay excluded. No chain transition,
optimizer, likelihood change, new gravity history, structure finder, or map
production is included. This replay recomputes the same deterministic IC-to-z=0
forward state only because that field was not saved; it is not a new physical
realization or an independent cosmological simulation.

For each of the same two checkpoints, require all of the following in the
same allocation:

- component scores sum to the prior stored exact score/energy at that state;
- the two component score gradients add back to the saved exact total gradient
  over all IC coordinates and all24 white nuisance coordinates (normalized
  L2 and max-relative discrepancy each <=1e-6, fixed before execution);
- report each component's complex gradients in the three modes and its
  all-mode IC-gradient RMS, plus all24 nuisance-gradient coordinates;
- report radial and transverse directions separately, using the exact
  conjugate-pair convention, and keep the mode-level result conditional on
  fixed nuisance coordinates;
- record a complete result for both A and B, or label the calculation
  incomplete without inference.

The outcome-to-action mapping is fixed in advance: a term dominates only if
its full complex gradients dominate the other term and reproduce the summed
direction across all three modes at both correlated endpoints. Large opposing
count/mark gradients indicate joint tension; comparable outward terms indicate
a shared forward/selection geometry issue. Material endpoint-to-endpoint
attribution changes mean state dependence and do not justify a single-term
correction. No outcome licenses posterior promotion, held-out use, prior
changes, or another chain.

## Scope and cost

Q-GOAL: isolate which factor in the current R2 partial observation law creates
the outward low-k score force before changing the model or spending on samples.
This is directly upstream of the required actual z=0 density/velocity delivery;
it does not itself identify the LG.

Q-LEAN: use only checkpoints410010/410011 and the already frozen training
inputs. One bounded same-state decomposition, two PMWD replays and two
component adjoints per endpoint; no extra modes/checkpoints/held-out score or
sampler sweep. Prior GL2 gradient evaluations took about14–19 minutes each;
the planned single H200 allocation is capped at2h30, with32 GiB host memory
(over20% above the measured20.61-GiB peak). If the complete attribution cannot
fit the cap, stop and report incomplete rather than silently extending it.

MW/M31 remain role-ambiguous and M33 unresolved. Their eventual observations
must constrain those latent roles on the same NEW field. Native truth
identities remain evaluation/calibration only and never seed or select a
candidate. This R2 force attribution performs no R3 identification.

First execution410050 (`db6bc13`, H200/syn104) failed after22m23s on a code
unpack error: the derivative API returns four blocks `(rho, velocity, tracer,
population)`, but the driver unpacked three. This is an implementation failure,
not a physics result. Before failure, the deterministic A replay reproduced its
saved exact energy to7.45e-9 and returned count/FP scores-135382.44/+5980.75;
no component gradient was retained, so no attribution conclusion follows.
Preserve its result/logs under
`/gpfs/kjhan/CF4/z0_density/r2_n256_lowk_component_attribution_20261002_v1/`.

The driver now maps all four derivative blocks through the PMWD pullback and
adds two small tests for block layout and rejection of incomplete tuples. The
retry preserved the same checkpoints, training factors, numerical tolerances
and2h20 application cap; the target and requested physics did not change. No
new simulation, chain transition or heldout score was added.

Retry job410059 (`739fc12`) completed on H200/syn104 in1h09m42s, exit0; batch
MaxRSS13,304,076K. Both focused tests passed. Output:
`/gpfs/kjhan/CF4/z0_density/r2_n256_lowk_component_attribution_20261002_v2/`.
Both saved-state exact-energy replays matched: A absolute error7.45e-9,
B1.68e-8. Independent component score errors were0 (count A/B),9.1e-13 (FP A)
and0 (FP B). At both accepted endpoints, count+FP reconstructed the stored
full exact target gradient: relative L23.78e-15(A),3.72e-15(B); relative
max2.99e-15/2.46e-15. Thus the 4-block mapping, pullback, sign and component
sum pass the same-state algebraic check.

For the three fundamental IC-white conjugate-pair modes, NLL radial derivative
sums were:

| Endpoint | 2M++ count NLL | Selected raw CF4 FP-mark NLL | Combined likelihood NLL |
|---|---:|---:|---:|
| A | -8425.34 | +36.87 | -8388.47 |
| B | -2611.38 | +92.29 | -2519.09 |

The negative values mean the partial likelihood NLL locally decreases as the
tested latent mode amplitudes grow, against the Gaussian prior's inward force.
The count factor accounts for nearly all of that outward low-k force at both
states; selected raw FP marks are small and oppose it. Count IC-gradient
all-mode RMS was .3212/.2410 (A/B), versus FP .00463/.00568. This is a genuine
diagnosis of which implemented factor drives the force in these two states,
not evidence that the underlying observations physically support extreme IC
modes. These states share one short-chain lineage, are not independent
replicates, and nuisance values are held fixed. Nuisance gradient norms are
large/state-dependent (count 1951/2071; FP 2737/4939, mostly population
parameters), so the IC-force attribution does not establish calibrated
joint likelihood behavior.

The active target is partial: graph-closed 2M++ voxel counts plus a selected
raw CF4 FP-distance-mark subset; TF/SNIa/SBF terms are absent. No heldout data,
chain extension, target change, map promotion or gravity re-evolution occurred.
R2 remains incomplete at N256/1.5 cMpc/h; this does not meet the <=0.3 LG
resolution goal. MW/M31 role ambiguity remains and M33 is unresolved; their
observables must ultimately constrain roles on this same newly inferred field.
## Fable5 terminal advisory and driver disposition

Fable5 returned **CONDITIONAL PASS** on the implementation and interpretation.
It independently rederived the score/NLL sign, conjugate-pair factor,
four-block pullback and prior-once reconstruction; the saved-gradient
reconstruction and independent component scores provide strong numerical
cross-checks. It agrees the graph-closed count factor, not selected raw FP
marks, supplies nearly all of the low-k radial force at these two fixed states.
It cautions that this is not evidence that the observations physically support
the large latent modes: count strength changes by about3.2x between the two
nearby states and nuisance gradients are large. Rate-coordinate residuals
imply total normalization mismatch alone is unlikely to explain the shape
force. It also corrected the wording: raw FP opposes count in five of six
mode-state pairs, but weakly reinforces it in A's `(1,0,0)` mode; “small”
describes the three-mode sum, not every individual mode.

The driver verified the writer's exact target gradient in
`scripts/cf4_r2_force_pair_comparison.py`: `fine_gradient = q - grad(logL)`;
therefore `fine_gradient-q` is the full likelihood NLL gradient. The active
observer is `[192,192,192]` in a384 cMpc/h periodic box. The saved component
IC-gradients are summaries, not full arrays; the next geometric localization
will therefore analyze the **combined count-dominated** likelihood gradient,
not mislabel it as pure count. FP contributes only about1.4%/2.4% of the
count all-mode IC-gradient RMS at A/B, but it is nonzero.

The driver adopts Fable's recommended smallest next step before auditing or
editing the count law: from the same two saved q/gradient checkpoints, compute
the full-field and low-k-shell radial NLL derivatives through8 fundamental
harmonics (`0 < |n| <= 8`, grouped by exact integer `n_x^2+n_y^2+n_z^2`);
phase-correct the Fourier coefficients for the centered observer and
compare shellwise monopole against dipole/quadrupole angular components. The
alternating phase template follows the verified half-box observer convention.
This needs FFT/mode algebra only: no PMWD replay, heldout read, chain, target
change, or gravity evolution. Predeclared routing: a dominant radial monopole
points to radial selection/LF/K-correction and volume normalization (including
the unexposed outer volume); anisotropy points to angular completeness/exposure;
an incoherent remainder points to low-intensity occupied cells/bias behavior.
No null p-value or physical peak claim will be made. Q-GOAL: a radial count
artifact would contaminate local flows/voids and therefore the LG environment;
localizing it advances R2 before MW/M31/M33 are identified on the same NEW
field. Q-LEAN: two existing states, inexpensive saved-gradient FFT only; no
replay or repeated sampler work. MW/M31 remain ambiguous and M33 unresolved;
their observables still must constrain the same generated field at <=0.3
cMpc/h. The first unit run exposed a false-anisotropy risk from grouping
different |k| values in rounded shells; the implementation now uses exact
|k|^2 shells, and all3 focused tests pass. Fable could not open the original
checkpoint-writer file, so the
driver checked it directly; its remaining extra whole-field/radial-profile
estimates will be treated as hypotheses until this next computation reproduces
them. Advisory is not authority; substantive recommendations were independently
checked and their limits recorded here.

## Saved-gradient geometry outcome

The focused follow-up job410076 used only the saved IC-white q and exact target
gradient at A/B; it performed zero PMWD replays, chain transitions, heldout
scores or new simulations. H100/syn08 completed in10s/exit0 with3 focused
tests passing; batch MaxRSS3,516K. The full-field radial derivatives reproduce
Fable's prior rough estimates: A -4128.52 and B -777.69. By exact integer
`|n|^2` Fourier shells, the likelihood NLL radial derivative over
`0<|n|<=8` is -4972.07(A)/-2102.56(B). The `|n|^2=1` fundamental shell alone
is -8388.47/-2519.09; the remaining shells inside the band add+3416.40/+416.53,
and modes outside the band add+843.54/+1324.87. Thus the anomalous radial force
is concentrated in the fundamental triplet and is partly opposed by the other
coordinates.

The low-k **score-force Fourier energy** decomposes into centered monopole,
dipole, quadrupole and higher/unmodelled shares:

| Region | Endpoint | Monopole l=0 | Dipole l=1 | Quadrupole l=2 | Higher / unmodelled |
|---|---|---:|---:|---:|---:|
| `|n|<=8` | A | 43.5% | 14.7% | 16.7% | 25.1% |
| `|n|<=8` | B | 5.9% | 18.7% | 27.5% | 47.9% |
| Fundamental `|n|^2=1` | A | 78.1% | 11.9% | 10.0% | ~0%* |
| Fundamental `|n|^2=1` | B | 33.1% | 50.6% | 16.3% | ~0%* |

`*` The six axial modes on this shell are exactly spanned by l=0/1/2; this
zero residual is algebraic completeness, not model fit evidence. Fit ranks are
stored per shell. Fundamental derivative values match job410059 within4e-11.
The result is **not** a robust observer-centred monopole pattern: A's
fundamental has a strong monopole share, but B's is more dipolar; over all
low-k modes the monopole share changes from43.5% to5.9%. A/B remain correlated
states, so this is evidence of state-dependent force geometry, not independent
replication or an identified sky dipole. These coefficients describe the
latent Fourier likelihood force, not z=0 mass overdensity or observed-sky
multipoles.

The driver therefore rejects a radial-selection-only diagnosis and does not
edit the count observation law yet. Fable5's second read-only review returned
CONDITIONAL PASS. It verified the reported algebra and emphasized that the
fundamental shell's six axial modes are exactly spanned by l=0/1/2, so its
zero residual degrees of freedom are algebraic, not evidence for that model.
The fundamental monopole cannot diagnose radial shape: its angular kernel is
positive across the survey radius. A's/B's dipole share is comparatively stable
while the monopole collapses, and the dipole rotates; this is state dependence,
not a robust centered peak. Fable's earlier speculative “central overdensity
0.5–1” illustration has no reproducible source in the artifacts and is
withdrawn from inference and planning. Generic latent Fourier anisotropy does
not diagnose an angular-exposure defect; no mask multipole projection is
warranted.

The driver independently checked the count-law context: the intrinsic
population bias response is normalized to unit mean over the full periodic
source box (`intrinsic_biased_source_masses`), whereas the active count
likelihood integrates only the graph-closed training exposure (empty exposed
cells included). Therefore the next discriminating readout is in data space,
not another latent Fourier decomposition. It will use the SAME accepted A/B q
states and exactly reproduce each saved GL2 count score, then aggregate
observed/predicted training counts by observed-key-cell radius x population,
geometric octant among training-exposed cells, and model-intensity quantile
within radius/population. A radial trend shared across regions may implicate
selection/LF; a sky residual conditional on radius/population/intensity may
implicate angular exposure; a residual by predicted intensity may implicate
the bias/low-intensity regime. These are routing clues, not automatic model
edits or significance tests.

Because the accepted checkpoints contain canonical q and total gradient but
not rho/velocity, recovering the same count prediction requires two
deterministic PMWD forward replays (one per q); this is not a new independent
gravity realization, chain, adjoint, posterior prediction, or heldout score.
Only training counts and the frozen heldout/buffer geometry needed to form
the training exposure are read; heldout count/mark values remain unopened.
The existing component score is an exact reproducibility gate (absolute score
error <=1e-7). Q-GOAL: diagnose the partial likelihood driving the actual R2
z=0 environment before it contaminates LG history. Q-LEAN: two fixed states,
two forward-only evaluations, no sampling or target changes. MW/M31 remain
role-ambiguous; M33 unresolved; all three must eventually be constrained by
observables on the same NEW field at<=0.3 cMpc/h. R2 remains incomplete at
1.5 cMpc/h.

## Same-state training-count residual and Fable5 audit

Job410095 (`5ecaaa7`, H200/syn104) completed in5m54s/exit0, MaxRSS2.44GiB.
It performed two deterministic PMWD forwards from the same accepted A/B q
checkpoints; no new realization, chain step, PM adjoint, heldout score or law
edit. Both exact fine count scores reproduce the component-attribution scores
to2.91e-11. Observed training counts are47,121; expected counts are46,810.54
(A) and47,061.09(B), so the total observed/expected ratios are1.0066/1.0013.
Overall rate normalization at these states is close, but the in-sample
population/radius residual is structured. Pooled ratios (A/B) are1.071/1.048
at0–36,1.007/1.000 at36–72,.980/.976 at72–108,1.012/1.003 at108–144, and
1.047/1.053 at144–180 cMpc/h. The180–192 edge bin has38 observed versus
68.47/65.84 predicted; do not treat that small edge bin as a general radial
trend.

Population conditioning shows why the pooled radial profile is incomplete.
Using the code-defined six populations (`3*apparent_bin+absolute_K_bin`),
state A has O/E=.886 for population0 over0–132 cMpc/h,1.057 for population1
over0–96,1.070 for population2 over0–60, and1.088 for population4 over
108–168. Population0 rises broadly from O/E=.663 at12–24 to1.109 at168–180;
it is not perfectly monotonic. The corresponding B broad ratios are.872,1.052,
1.043 and1.079. These discrepancies are training residuals at two correlated,
nonstationary states, not data-only evidence for a luminosity-function defect.

As a descriptive octant margin after rescaling each population/radius stratum
to its own total, `+-+` has O/E1.091(A)/1.078(B) and `-++` .949/.947; the
other exposed sectors are near unity. Sector `-+-` has zero training exposure
because it is held out. The stability across A/B is not replication: both
states share one short lineage, and a fixed-field residual can arise from
non-equilibration as well as exposure or bias mismatch.

Quantile-label correction: an initial driver summary pooled the script's local
intensity-bin indices as if they were five globally comparable quintiles. That
pooling is invalid when tied cut values are merged. I retract those pooled
Q0–Q4 ratios. Only29 of96 population/radius strata contain five distinct bins;
the resulting restricted subgroup is not a calibration test, so no global
intensity-quantile conclusion is retained. The script reports each shell's
cut values, and its unit tests now also assert a known cell's exact radius,
octant and high-intensity-bin assignment.

Independent operator review also found a concrete boundary limitation in the
current observation law: observed catalogue rows enter count cells by NGP
`floor(position/dx)` (`cf4_r2_common_catalogue.py`), while predictions apply
the radial cut before a TSC deposit whose support extends1.5 cells. Thus the
180–192 cMpc/h predicted tail can include TSC leakage absent under the same
NGP binning of the data. This is a plausible explanation for some edge-bin
deficit, not a measured causal attribution; no kernel or cut was changed.
It needs a matched mock or a consistent voxel-integrated observation operator
before any correction.

Fable5 read-only audit returned **CONDITIONAL PASS**. It checked the fine count
score/operator identity, training-only geometry and representative radial
arithmetic; the driver independently checked both endpoints and all quoted
aggregates. It correctly warned that the previous pooled quantile indices
were not global quintiles and that the radial pool hides population-by-distance
structure. Its optional suggestion to add `OWNER.txt` was not adopted: no
project or injected cluster instruction requires that duplicate marker; the
Slurm result already stores job ID and source commit.

Driver disposition: adopt Fable's single next test, a prior-regularized
9-coordinate tracer-nuisance conditional profile at frozen field A. The
checkpoint stores q but not rho/velocity, so one deterministic PMWD forward is
needed to reconstruct the exact same field; there will be no PM adjoint,
sampling, heldout values, or observation-law edit. Reproduce the active GL2
count value/gradient at the initial nuisance state, then compare the existing
population-by-radius table before/after a bounded profile. If the profile
converges and reduces these residuals, that diagnoses nuisance
non-equilibration only; it does not validate the model. If structured
residuals remain, external bias/survival and RSD/FoG calibration remain
necessary before sampling. Stop extra octant/quantile slicing.

First execution410100 failed after18m19s on a driver tuple-unpacking error at
the first optimizer trial: JAX `value_and_grad(has_aux=True)` returns the
nested pair `((objective, auxiliary), gradient)`, not three top-level values.
Before the failure, the deterministic A field replay took17s; the initial
score matched by2.91e-11, saved tracer-gradient relative error was5.26e-16,
and compiled device estimate20.43GiB/104.85GiB. **Zero optimizer trials
completed**, so this is solely an implementation failure and gives no
nuisance-profile or science conclusion. The nested return is now handled by
one tested unpacker; retry410101 uses the v2 output directory and the same
data, field, target, prior and eight-evaluation cap.

Q-GOAL: this tests whether a nuisance starting point is the immediate R2
obstacle to the actual z=0 field; it does not supply LG information. Q-LEAN:
one fixed endpoint, nine variables, existing radial/population bins, bounded
optimizer calls; no heldout use or target change. MW/M31 remain ambiguous and
M33 unresolved; later observables for all three must constrain the same NEW
field at<=0.3 cMpc/h, with native truth IDs used only for calibration/evaluation.

### Retry410101 failure and guarded retry

Retry410101 terminated after18m04s/exit1 with host MaxRSS3.82GiB. The saved
fixed-field PMWD forward and initial exact score/tracer-gradient checks passed;
the first optimizer evaluation returned, but the driver's finite-value guard
then attempted to concatenate scalar/vector outputs and the two-dimensional
population-by-radius expectation table via `np.r_`, raising a NumPy dimension
error. Zero optimizer trials were recorded, so neither this run nor its initial
training residual is a profile result. The v2 output/log are preserved.

The guard now tests finiteness of each returned array independently, and a
focused regression covers both a finite 2-D table and a NaN in that table.
All four focused profile tests pass. Same field/data/likelihood/prior and
eight-evaluation/time caps are retained; the next Slurm attempt writes to a
new v3 directory. This is a driver repair only, not evidence about tracer
nuisance adequacy or the count law. Q-GOAL/Q-LEAN and same-new-field
MW/M31/M33 constraints are unchanged.

### Completed v3 profile and independent Fable5 result audit

Job410105 (`de86df4`, H200/syn104) completed in1h09m13s/exit0, MaxRSS3.89GiB.
It reached exactly8 evaluations and correctly reports
`FROZEN_FIELD_TRACER_PROFILE_BOUNDED_NOT_CONVERGED`; it is not a converged
conditional optimum. At the saved A start, v3 reports score absolute error
0.0 and tracer-gradient relative error8.906e-16. The earlier audit request
incorrectly copied2.91e-11 and2.47e-15 from prior runs; these are not the v3
reproduction values. No heldout outcomes were read/scored, PMWD adjoints or
chain transitions occurred, and the count law was not changed.

The initial conditional objective135387.0284 falls to135218.3741
(-168.6543nat): count log score improves168.3521nat and prior NLL falls only
0.3022nat. Training population-by-radius L1 is.064100→.058544 (8.7% relative
reduction), expected count46,810.537→47,395.376 against47,121 observed. The
saved terminal gradient norm is595.803, so the eight-evaluation cap was reached
well before convergence. Correction to the earlier Pearson label:328.062→291.938
uses all59 positive-mean bins, not the expected>=5 subset. At expected>=5 the
result is53 bins and326.197→290.009. Population0's positive-mean statistic is
144.019→144.773 over16 bins; at expected>=5 it is142.229→142.920 over15 bins,
with8,859 observed against initial expectation9,837.428.
Population0's radial deficit is essentially unchanged, while some smaller
population-conditioned residuals move. These are descriptive conditional
in-sample statistics on one nonstationary field, not a Poisson test or
calibration result.

Fable5 independently audited the result read-only and returned **CONDITIONAL
PASS**. It caught the audit prompt's stale reproduction numbers; the driver
verified the actual v3 values above. It also identified that the old radial
summary clips every cell at/above180 into its final slot while labelling that
slot180–192. The historical JSON is preserved; reporting code now labels it
`>=180` with no finite upper edge. From the saved component-attribution JSON,
the initial raw-FP NLL gradient in the same nine tracer coordinates has norm
7.370, versus1952.673 for count plus prior (0.377%); adding it changes the
initial joint gradient norm to1954.578. This does not make the count-only
profile a joint optimum, but rules out a large opposing raw-FP force at this
particular start.

Driver adopts Fable's principal conclusion: no more fixed-field nuisance
profile. The objective has diminishing gains, rate normalization accounts for
about85% of the remaining gradient norm, and the population0 split signature
survives. The exact rate-coordinate gradient inferred from the source law and
final expected total is about+549; this is a derived component, not a stored
full gradient vector. A converged conditional optimum on this one field would
not close R2 or justify sampling.

### Predeclared R2 test plan: TSC-prediction / NGP-count operator closure

Implement one forward-only operator control at the saved endpoint-A initial
nuisances (not the profiled coordinates). Recover the same saved N256 field
once, compute expected training voxel counts with the active TSC deposit and
with a diagnostic periodic NGP deposit matching the actual catalogue's
`floor(position/dx)`, then draw one fixed-seed Poisson voxel-count mock from
the NGP mean on training graph-closed exposure only. Both means use the same
source-volume GL2 integral, existing analytic redshift-derived absolute-K
transfer, angular completeness, RSD/FoG quadrature and radial cut; only the
final voxel assignment changes. Sampling from the integrated NGP mean is the
minimal binned Poisson-process mock, not a row-level mock archive: it isolates
TSC-versus-NGP sensitivity but cannot calibrate source-group survival, the
luminosity law, CF4 marks or the actual-data selection. Exact saved TSC score
and radial/population means are mandatory gates before interpreting the mock.
The predeclared report compares the same radial x population bins, the
population0 share within observed bright populations0–2 at selected shells,
and the correctly labelled `>=180` tail. It reports a Pearson/Poisson
reference as a descriptive one-realization check only; no pass promotes a
posterior. No adjoint, fit, chain, heldout outcome, or observation-law edit.

Q-GOAL: isolates whether the known voxel operator mismatch contributes to the
R2 actual z=0 field residual before R3. Q-LEAN: one fixed field, two forward
operators and one seeded voxel-count draw; analytically integrate K labels
instead of building/archiving individual fake galaxies. MW/M31 remain
ambiguous and M33 unresolved; later observables must constrain those roles on
the same NEW field at<=0.3 cMpc/h, with truth IDs used only for calibration or
evaluation. R2 remains incomplete at1.5 cMpc/h; this is not LG identification,
calibration, heldout validation, or production inference.

### Completed closure result and independent Fable5 audit

Job410117 completed/exit0 on H200/syn104 in10m28s; Slurm batch MaxRSS was
3,521,324K. Its12 existing count-operator regression tests and3 diagnostic
tests passed. The exact active-TSC reference gate passed: saved-score absolute
error2.91e-11 and max radial x population mean error5.21e-11. At the same
endpoint-A field and initial tracer coordinates, TSC and diagnostic NGP
exposed means are46,810.537 and46,820.573 (difference10.036); the maximum
single aggregate mean difference is28.854. One seeded NGP Poisson voxel-count
mock has47,103 objects; aggregate Pearson checks are71.715/53 bins under NGP
and72.099/53 under TSC, both inside the approximate one-draw95% interval.
This verifies mechanics only; the one draw is upper-tail and is not validation.

Driver independently recomputed actual TRAINING-only radial x population
aggregates (96 bins, not the full voxel likelihood): the un-factorialized binned
score is292,733.987 (TSC) vs292,746.405 (NGP), delta+12.418nat; L1 is3020.466
vs2903.874; expected>=5 Pearson is326.197/53 bins vs305.005/53. The apparent
NGP improvement is concentrated: population3 `>=180` has38 observed vs66.685
(TSC) and37.830 (NGP), accounting for12.34 of the21.19 Pearson decrease.
Excluding that one bin gives313.858 vs305.004 over52 bins. Population0 has
8,859 observed vs9,837.428/9,836.761 expected (TSC/NGP); its expected>=5
Pearson worsens142.229→144.012. Observed bright-population0 shares in
36–96cMpc/h are.116,.152,.218,.350,.560; active-TSC shares are
.144,.183,.272,.397,.596, and NGP moves them by at most.0033. Thus NGP can
explain much of the single outer-tail deficit, not the bright population
split. Actual-data voxel score under NGP was not computed; do not generalize
the coarse-bin result to the voxel likelihood.

Fable5 read-only result audit returned **CONDITIONAL PASS** and independently
reproduced the coarse aggregate arithmetic. Driver verified the catalogue's
`floor(pos/(384./128))` NGP rule in `scripts/cf4_r2_common_catalogue.py` and
the `deposition` pass-through through `cf4_r2_chunked_volume_count.py`. The
threshold correction above is applied to this note and the active master plan;
historical JSON is preserved. These same-data, nonstationary aggregate
statistics are bin counts, not inferential degrees of freedom. Q-GOAL: yes,
this removes a plausible operator explanation without promoting a z=0
posterior. Q-LEAN: yes, one field/two operators/one seed was proportionate.

### Follow-up: field-free K/LF transfer shape

The read-only audit's next suggestion was implemented in
`scripts/cf4_r2_uniform_transfer_shape.py`; it records JSON at
`/gpfs/kjhan/CF4/z0_density/r2_n256_uniform_transfer_shape_20261002_v1/result.json`.
Using the five observed training
bright-share shells36–96cMpc/h, a uniform-density/zero-velocity field,
24-point radial GL weighted by r^2, and a61-by-61 grid on the existing standard
normal alpha/mstar white coordinates, the saved initial pair
(alpha_white=.103,mstar_white=-1.608) predicts pop0 shares
.153,.213,.299,.427,.621, versus observed
.116,.152,.218,.350,.560. A nearby grid pair(.2,.1) predicts
.104,.154,.230,.353,.557 (conditional-binomial deviance6.35, prior NLL.025).
This is only transfer-shape capacity on reused training counts, not a fit or
calibration. Luminosity-bias coordinates cannot affect a uniform field because
the full-box-normalized response rho^beta is identically1 at rho=1. In the
actual fixed-field TSC table the shares are instead
.144,.183,.272,.397,.596. Hence field/bias weighting materially shifts the
split, and the field-free calculation alone cannot decide whether the
prior-centred LF point helps on the actual field.

### Completed v1 prior-centre fixed-field sensitivity and Fable5 audit

Job410125 completed/exit0 on H200/syn104 in3m06s; batch MaxRSS3,409,652K.
With only alpha_white/mstar_white set to0, the active-TSC marked-count mean
falls46,810.537→37,944.431 against47,121 training observations. The five
bright-population0 shares move closer but cross to the low side: observed
.116,.152,.218,.350,.560; baseline.144,.183,.272,.397,.596; prior-centre
.097,.128,.205,.323,.530. The unprofiled count score falls1,895.595nat, but
this comparison is dominated by the19.5% mean deficit. The v1 candidate full
table was not saved, so its L1/Pearson cannot be rate-adjusted retrospectively.
Result: `/gpfs/kjhan/CF4/z0_density/r2_n256_lf_prior_center_sensitivity_20261002_v1/result.json`.
This job evaluated marked counts only; raw-FP was not reevaluated.

Fable5's read-only audit returned **CONDITIONAL PASS**. Driver verified the
source-law scaling: `tracer_masses` adds2*u0 to log rate,
`intrinsic_biased_source_masses` multiplies masses linearly by exp(log-rate),
and the sparse Poisson factor uses training-count sum47,121 and the exposed-mean
integral. The Poisson-MLE total-rate multiplier gives s=1.241842,
delta_u0=.108298 for the candidate, and s=1.006632 for baseline. On a
like-for-like rate-profiled comparison, candidate count score is-136,248.385
vs-135,381.415 baseline (delta-866.970nat); evaluating the Gaussian prior at
these MLE rate points, the count-plus-prior objective is worse by865.707. This
is not a joint MAP over the rate prior. These are
one-field, training-only conditional sensitivities, not a posterior or
independent prediction. The prior-centre shares bracket rather than match the
observed values; all five density-bias coordinates and sigma_los were frozen.
Candidate v1 L1/Pearson are not rate-profiled statistics. Fable confirmed
Q-GOAL and Q-LEAN alignment and same-new-field role constraints.

### Completed v2 prior-centre fixed-field sensitivity and Fable5 audit

Job410128 completed/exit0 on H200/syn104 in5m24s (batch MaxRSS3,507,112K).
Its TSC scalar-rate identity passed (map error3.55e-15; direct-vs-analytic
score error0), and four focused sensitivity tests passed. Fable5's read-only
result audit returned CONDITIONAL PASS. Driver independently reconstructed
the6x16 training-only observed, baseline and candidate tables from the saved
profile, closure and v2 outputs; the historical v2 JSON is not overwritten.
The audit found two reporting defects: `baseline_rate_profiled` held the
unprofiled closure table, and aggregate score delta compared the candidate
profiled table against the unprofiled baseline while the full score used both
profiled scores. The JSON generator/test are fixed; future outputs store the
observed table and clearly separate unprofiled/profiled baseline metrics.

Corrected derived baseline metrics after profiling its rate are mean47,121,
radialxpopulation L1 2,964.260 and expected>=5 Pearson321.977/53 bins. The
prior-centre candidate has the same profiled total but L1 7,380.534 and
Pearson2,109.859/53. Its population0 share is closer to observed in all five
36–96cMpc/h shells, but crosses to the low side; it is not a count fit. The
candidate is mainly an Mstar change: baseline Mstar≈-23.602, alpha≈-.9368;
candidate -23.28/-.94. With all five bias coordinates and sigma_los fixed,
count-score delta is-866.970nat, consistently decomposed as radialxpopulation
aggregate-803.099 and within-bin allocation-63.871. Aggregate contributions
by population are pop0-294.774, pop1-4.643, pop2-52.636, pop3-190.900,
pop4-96.424, pop5-163.721nat. Pop0 contributes-337.650nat beyond96cMpc/h;
pop3 contributes-197.310nat beyond96, especially144–180. These are in-sample
fixed-field sensitivities, not a significance test, physical-law rejection,
posterior or independent prediction. The count-plus-prior objective is still
865.707nat worse at count-only MLE rates; this is not a joint MAP and does not
exclude fitting other tracer coordinates or correcting the observation law.

Corrected aggregate score change by radial shell, summing all six populations
(nat): 0–12:-0.518; 12–24:+4.507; 24–36:-13.621; 36–48:-58.036;
48–60:-68.953; 60–72:-16.941; 72–84:-52.170; 84–96:-47.347;
96–108:-38.601; 108–120:-52.903; 120–132:-45.919; 132–144:-68.309;
144–156:-87.254; 156–168:-130.854; 168–180:-133.704; clipped `>=180`
tail:+7.522. These sum to-803.099nat. This localizes where this fixed-field
candidate loses under the current aggregate count law; it does not identify a
radius-dependent physical-law error.

Source audit confirmed six observed populations (two apparent-K samples x
three absolute-K bins), true/observed modulus plus redshift correction and RSD
before TSC deposition, and five true-K bias responses normalized over the
whole periodic box, not per source chunk. No simple radius-wiring defect is
established. External survival/bias and RSD/FoG calibration remain missing;
do not launch another LF scan, broad optimizer, chain or simulation. Next R2
work must source and validate radius-dependent selection/transfer and
calibration inputs before another fit.

The same audit exposed an exact-tie LF transfer gradient discrepancy: before
the patch, values matched but the exact-boundary shift gradient was-2.53536485
vs-1.58305506. A whole-call direct fallback was rejected because inactive
out-of-table nodes can make its predicate true on production chunks. The final
fix selects by original magnitude bounds and averages tied survival values,
matching the nested direct max/min subgradient without that all-chunk slow
path. `test_cf4_r2_lf_transfer_reuse.py` passes3/3, including apparent and
three-way ties, JIT, shift, Mstar and alpha gradients; the inactive-clamped-
source shell-CDF test, four observed-magnitude tests and five sensitivity tests
also pass. Job410128 is forward-only and unaffected.
No blanket historical gradient rerun is justified; odd-order GH/zero-velocity
gradient results remain tie-sensitive unless their saved-state weighted tie
contribution is shown zero.

Q-GOAL: yes, this narrows the R2 count-law issue and repairs an R2 gradient
convention without claiming MW/M31/M33 identification. Q-LEAN: one bounded
fixed-field replay and exact table correction plus a compact gradient fix; no
scan or posterior chain. Held-out outcome values were not read. R2 remains
incomplete and NO-GO at1.5cMpc/h; MW/M31 are role-ambiguous and M33 unresolved,
to be constrained on the same NEW field at<=0.3cMpc/h. Native truth IDs remain
calibration/evaluation only.
