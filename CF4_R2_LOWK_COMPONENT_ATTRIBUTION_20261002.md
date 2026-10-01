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
