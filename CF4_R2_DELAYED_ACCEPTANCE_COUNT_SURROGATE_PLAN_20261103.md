# R2 bounded delayed-acceptance count-surrogate screen — 2026-11-03

## Decision

Run one fixed-anchor, training-only delayed-acceptance HMC feasibility screen
after the projection, metric, support and detailed-balance regressions pass.
This is an engineering test, not an R2 posterior, a field product, or authority
to start R3. The exact target remains the unchanged N256/384 cMpc/h
conditional-v6 count + linked-FP + prior target. No held-out values are loaded.

The completed job447829 is not resubmitted. It established two accepted HMC
transport transitions but no stationarity or posterior uncertainty. The new
question is narrowly whether a cheaper, support-safe N128 count surrogate can
produce any exact-corrected movement in the fine target.

## Proposal and exact correction

Let `q_f` be the N256 standard-normal IC coordinates and `theta=(tau,p)` the
nine tracer and fifteen FP-population standard-normal nuisances. Let `R` be the
tested unitary full-spectrum restriction from 256 to 128. The update changes
`z=R q_f` and `tau` only; `q_perp=(I-R*R)q_f` and `p` stay fixed. Reconstruct
the fine coordinate as

```text
q_f' = q_f + R* (z' - R q_f).
```

The stage-one conditional surrogate is the standard-normal prior on the
updated coordinates plus the N128 training-count score. It uses the same
frozen training keys, counts, exposure, source-selection map, cosmology, and
tracer parameterization. A fixed `epsilon=1e-30` expected counts is added
inside `log(lambda + epsilon)` for occupied training keys only; the expected
rate integral remains unmodified. This is only a proposal surrogate: no floor
enters the fine target or the reported scientific likelihood. Negative or
nonfinite intensities/fields are fatal numerical errors, not MH rejections.

Stage one uses the existing fixed prior-split HMC integrator, with the
N128-restricted Fourier metric and the first nine nuisance metric entries,
step `0.11996093316338646`, eight steps, and no adaptation. The surrogate
potential includes the same Gaussian prior on updated coordinates. Its HMC
Metropolis decision is made before any fine evaluation.

For a stage-one-accepted proposal, evaluate the unchanged N256 direct
value-only target and apply

```text
alpha2 = min(1, exp(-[(U_f(q')-U_s(q')) - (U_f(q)-U_s(q))])) .
```

The current fine scalar is cached from one exact anchor replay. Rejection at
either stage retains the original anchor; stage-one rejection performs no
fine evaluation. Treat only the exact target's declared empty-support errors
as zero fine density and stage-two rejection. Memory, nonfinite, and unrelated
support-construction failures abort the screen.

The kernel is invariant for the fine target conditional on the frozen
complement and population nuisances, provided the checks below pass. It does
not update those frozen coordinates and is not an ergodic full-posterior
sampler. Eight independent proposals from one discarded anchor are not chain
draws.

## Pre-submission checks and fixed caps

Focused tests must establish real-coordinate `R R*=I`, adjoint consistency,
norm decomposition, Nyquist handling, exact preservation of `q_perp`, the
restricted proposal metric, and synthetic two-stage detailed balance. The
surrogate gradient must be the derivative of the exact floored expression.
Recompute the fine scalar at the recorded state60 anchor and require agreement
with the saved exact-gradient target value within `1e-7` nat before proposals.
Report the number of N128 training-event rates at or below the fixed floor at
the anchor and at every tested endpoint.

The screen is fixed at eight independent proposals, each from that same anchor,
with one HMC length/step and no retuning. Hard limits: nine N256 scalar values
(one anchor plus at most eight accepted-stage-one candidates), four aggregate
GPU hours including setup/compilation, a four-hour-thirty-minute Slurm wall,
24 GiB host RAM (at least20% over the measured16.25 GiB HMC host peak), and
the existing exact-GL2 GPU allocation guard of at least39,826,427,463 bytes.
Stop with a preserved partial result if any cap is reached; no retry, scale
ladder, metric changes, held-out read, extra simulation, or field promotion.
Use Slurm only, check H200 first, then H100/A100, use the matching typed GRES,
and record the selected partition/GRES and occupancy snapshot.

Record for each proposal: seed, stage-one acceptance and energy error,
stage-two probability/decision when reached, `Delta(U_f-U_s)`, prior/count/FP
decomposition, floored-event count, retained low-mode/tracer displacement
(zero on either rejection), exact-target calls, and elapsed time. Also record
the aggregate setup and cost. A few acceptances mean feasibility only; zero or
negligible corrected movement closes this fixed surrogate configuration, not
all delayed-acceptance methods.

## Astra advice and driver disposition

Astra's read-only review returned **CONDITIONAL GO for this one bounded
screen**, subject to the listed mapping and numerical tests. It verified the
conditional detailed-balance argument when the real-coordinate restriction is
coisometric, the complement/population coordinates are fixed, stage-one HMC is
reversible, and the exact correction uses cached fine values. It warned that a
positive event-intensity floor does not protect against negative/nonfinite
rates or invalid PM outputs; those must abort. It also requested the exact
restriction of the N256 metric and explicit identification of the
prior-split integrator. The driver adopts these conditions. The independently
defined N128 inverse-Laplacian symbol will be used only after regression proves
it equals `R C256 R*` including Nyquist classes; otherwise submission is
blocked.

### Q-GOAL

This tests a cost lever for the first actual conditional z=0 field posterior
at1.5 cMpc/h. It delivers no posterior, uncertainty, held-out prediction, or
observation/sampler calibration. MW and M31 remain role-ambiguous; M33 remains
unresolved. Their observables must ultimately constrain those same roles,
including unresolved/shared candidates, on the same NEW evolved LG field at
`<=0.3 cMpc/h`. Native truth identities are calibration/evaluation-only, not
generated-field seeds or selectors.

### Q-LEAN

One fixed support-safe count surrogate, one conditional Fourier block, eight
fixed-anchor proposals, and at most nine exact scalar evaluations test a
specific cost hypothesis. There is no parameter search, held-out read,
covariance campaign, separate gravity run, or posterior claim. R2 remains
open/NO-GO and R3 has not started.
