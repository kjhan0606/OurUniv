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

Execution: job410050 was submitted from commit`db6bc13` and started on H200
node syn104. It requests1 GPU,4 CPUs,32 GiB host memory, and2h30 wall time;
the application guard is2h20. Output:
`/gpfs/kjhan/CF4/z0_density/r2_n256_lowk_component_attribution_20261002_v1/`.
The terminal outcome, numerical resource use, and driver disposition will be
added here; a running job is not a scientific result.
