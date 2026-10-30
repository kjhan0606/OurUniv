# R2 sampler redesign — Gaussian-reference ESS (design draft, 2026-10-28)

**Status: the Gaussian-reference ESS remains design-only; no production
sampler or sampler pilot is approved.** The separately bounded one-pullback
feasibility diagnostic was submitted once as job 439311, but its metadata
guard failed before any VJP; the corrected validator is regression-tested.
The value-only score-cost probe 439023 completed on syntax with exit 0 in
25:19 (MaxRSS 10.64 GiB). At the same anchor, the matched scalar
evaluation took 400.46 s and direct value-only evaluations took 178.01, 142.38,
and 144.50 s (mean 154.97 s; 0.387x time, about 2.58x faster). The direct and
matched objective agree at float64 precision; one FP term differs by one ULP
(`9.09e-13` nat). Aggregate q0/repeat scores are bitwise equal, but the
original strict termwise repeat flag remains false and its output is
preserved as `VALUE_ONLY_REPEAT_MISMATCH_NOT_POSTERIOR`. Driver disposition:
roundoff-level termwise non-bitwise equality is immaterial at this scale, and
the direct-value evaluator is suitable for the next bounded feasibility
diagnostic. This establishes evaluator cost only, not low-k movement or
sampler efficacy.

## Goal and fixed target

Keep the full N256 / 384 cMpc/h conditional-v6 target, not a reduced field:

\[
\pi(q\mid D_{\rm train})\propto
\mathcal N(q;0,I)\exp\{\ell_{\rm count}(q)+\ell_{\rm FP}(q)\},
\]

with 47,121 training count occurrences (37,951 keys), 1,414 observed linked
raw-FP marks, and all 24 nuisance coordinates. The FP term remains conditional
on the linked count point and observed mark presence. That conditional
estimand is not a full-survey selection calibration: selected marks do not
identify missing eligibility, targeting, redshift-success, FP-quality or
catalogue-linkage probabilities. Do not fit those probabilities from selected
rows or imply that the availability bias is ignorable.

“Exact” transition below means invariant to this frozen numerical target
(including its GL2 rule and existing approximate 8-sigma support truncation),
not exact integration of the underlying continuous survey likelihood.

The exact full target—not the proposal approximation—must decide every
transition. Held-out values and native truth identities are excluded from
basis construction. No field or last state is promoted; R2 stays NO-GO until
valid posterior draws, uncertainty and held-out prediction exist.

## Single candidate to assess

Use a positive-semidefinite, training-only positive-row Gauss–Newton
surrogate as a **proposal geometry**, not as a new likelihood, a full
Poisson-Fisher claim, or a claim about posterior rank. Let \(\lambda_c(q)\)
be predicted intensities for every population on every exposed training
cell, including zero-count cells. At one fixed anchor \(q_0\), define
\(A_0=\{i:e_i^{\rm train}=1,\lambda_{c,i}(q_0)>0\}\) and
\(J_{A_0}=\partial\lambda_{c,A_0}/\partial q\). The proposed geometry is

\[
F_+(q_0)=J_{A_0}^T
\operatorname{diag}(1/\lambda_{c,A_0}(q_0))J_{A_0}\succeq0.
\]

This deliberately omits zero-intensity rows **from proposal construction
only**. Such a row must not be called information-free: if
\(\lambda(t)=t^2\), then \(\lambda(0)=\lambda'(0)=0\), while the
positive-side limit of \((\lambda')^2/\lambda\) is4. Astra confirmed that
proving a global structural-zero mask would be a disproportionate extra
project. Instead, freeze \(A_0\) at the training-only anchor, add no
intensity floor, and leave every train-exposed row—including zero and newly
activated rows—in the unchanged exact Poisson target and its residual. This
can reduce proposal efficiency but cannot change the target. Any occupied
training key with zero/nonfinite intensity, or any negative/nonfinite
predicted intensity, fails closed. If positive-row weights or the resulting
metric are nonfinite or exceed a predeclared numerical cap, stop rather than
insert an intensity floor. Do not read raw finite-sketch eigenvalues as a
calibrated proposal spectrum.
PMWD's N-body function is a `custom_vjp`, so direct JVPs through evolution
are not supported. A VJP-only randomized sketch may avoid that
limitation: for fixed independent \(z_k\sim\mathcal N(0,I)\) over the
positive anchor rows \(A_0\), compute

\[
s_k=J_{A_0}^T W^{1/2}z_k,\qquad
\widehat F_+=\frac1r\sum_{k=1}^r s_ks_k^T,\qquad
W=\operatorname{diag}(1/\lambda_{c,A_0}(q_0)).
\]

Each \(s_k\) is one reverse pullback and
\(\mathbb E[\widehat F_+]=F_+(q_0)\). Do not treat the nonzero eigenvalues
of this finite sketch as proposal-spectrum estimates. For example, if
\(F_+=I_d\) with \(d\gg r\), the rank-r sketch has nonzero eigenvalues of
order \(d/r\), despite being unbiased as a matrix estimator. Use discovery
sketches only to propose a subspace \(U\). Use a separate, independent,
training-only validation sketch to estimate the projected information
\(U^T F_+(q_0)U\) and residual positive-row trace outside \(U\); report uncertainty and
batch-to-batch stability, and fail closed if they do not stabilize under the
fixed sketch cap. These checks can validate proposal geometry, not prove that
the discovered subspace is the posterior's complete informed subspace. The
small Gram matrix is useful for a candidate range, not a license to read its
sample eigenvalues as the true spectrum. This avoids a dense \(J\), not
repeated full backward passes; sketch variance and actual setup cost remain
unknown. First-order reverse sensitivities still require a
stability check because the particle-to-grid CIC readout is piecewise
differentiable. The previous local finite-difference curvature screen
remains closed and is not repeated.

The count intensity does not depend on the 15 population nuisance
coordinates \(p\), so the count Fisher is identically zero there. Do not leave
that stiff block at prior scale: add a separate 15-D proposal block from the
raw-FP log score at the same training anchor,

\[
H_p=\nabla_p^2\ell_{\rm FP}(q_0),\qquad
F_p=\operatorname{PSD}(-\operatorname{sym}H_p),\qquad
C_p=(I+F_p)^{-1}.
\]

Here the field, tracer coordinates and support packs are fixed while taking
the p-only Hessian; source inspection confirms p enters the raw-FP factor but
not support construction. Eigenvalue clipping is proposal geometry only; the
full raw-FP score remains in the exact residual. It does not remove field–p
cross-correlation, which is an explicit mixing risk to test, not a solved
claim. The reference is block diagonal between \((\text{IC},t)\) and p; the
orthogonal count-field complement retains identity covariance.

If the independent validation criteria pass, one possible full-rank
\((\text{IC},t)\) proposal covariance is
\[
C_{qt}=U(I+F_{U,+})^{-1}U^T+(I-UU^T),
\]
where \(F_{U,+}\) is the independently validated positive-semidefinite
projected positive-row geometry on the discovery subspace; the orthogonal
complement retains prior scale. The final values of rank, \(F_{U,+}\),
uncertainty and pass thresholds are not
yet specified. Combine this with \(C_p\) as a fixed block-diagonal \(C\).
Form one full-support Gaussian reference \(g=\mathcal N(\mu,C)\) and the
single fixed center \(\mu=q_0+C\nabla\log\pi(q_0)\). The anchor and its
gradient are proposal construction only, never posterior draws. Do not
re-estimate \(\mu\) or \(C\) from each current state. Use one joint elliptical
slice update in fixed \(g\)-whitened coordinates so the informed subspace,
complement and nuisance block all participate; if separate blocks are ever
introduced, define them in those coordinates or use exact conditional
Gaussians. Then use the residual

\[
\log r(q)=\log\mathcal N(q;0,I)+\ell_{\rm count}(q)+\ell_{\rm FP}(q)
          -\log g(q).
\]

Because \(C\) is full rank and fixed, and every slice evaluation uses the
complete unchanged target, this proposal preserves the frozen numerical
target regardless of how imperfect the Fisher geometry is. Poor geometry or
omitted field–p covariance can still make it unusably slow; full rank alone
does not guarantee useful movement. It cannot be reported as success merely
because the kernel is invariant on paper.

## Turning-point advice and driver disposition

Astra's read-only advice found the fixed-reference ESS argument sound only
with the exact residual above. A state-dependent plug-in reference is not
covered. Astra independently flagged the finite-sketch spectral inflation
example and required independent training-only validation of the proposed
subspace, including information outside it. On zero-intensity rows, do not
equate \(\lambda=0,J=0\) with zero Fisher information: a quadratic activation
can have a finite nonzero one-sided limit. Driver disposition: use a fixed
positive-row surrogate geometry only, while preserving all rows in the exact
target/residual; do not launch a structural-zero census. The count Fisher's
exact zero in the population coordinates still requires the separate raw-FP
curvature proposal and leaves field–population coupling unresolved. These
are design constraints, not approval to implement or submit the sampler.
The one-pullback VJP count and feasibility outcome remain unmeasured; discovery
and validation sketch stability, useful movement criteria, and sampler-level
cost remain uncosted. The single diagnostic has a fixed 2 h application / 2.5 h
Slurm limit and 20 GiB host request. If the pullback is unstable or exceeds its
cap, close this candidate rather than increasing the sketch ladder.

## Fixed-reference ESS mechanics check — 2026-10-29

Added `src/cf4_r2_gaussian_reference_slice.py`, a generic full-space ESS
primitive that evaluates the exact residual `log_target + 0.5*||x||^2` in
whitened coordinates for a fixed full-rank linear Gaussian reference. Slice
heights are centered at the current state and the squared-norm difference is
evaluated as `(x_new-x) dot (x_new+x) / 2`, avoiding a needless subtraction of
two large reference norms; the accepted physical state is reused from its
complete-target evaluation rather than recolored. The small CPU test
`tests/test_cf4_r2_gaussian_reference_slice.py` passed three checks, including
a 27,000-draw retained sample from a correlated Gaussian target with a
different fixed Gaussian reference. This verifies only the toy-kernel
algebra/statistics; it does not validate an N256 transform, CF4 target wiring,
PMWD derivatives, sampler efficiency, posterior uncertainty, or any field.
The production basis and reference must remain fixed and be validated
separately before this primitive is used in R2.

The next one-pullback diagnostic is drafted in
`scripts/cf4_r2_intensity_vjp_feasibility.py` with a typed-GPU Slurm runner.
It differentiates the active N256 count intensity jointly in white-IC and the
nine tracer coordinates, using one training-only positive-row random
cotangent; the 15 population coordinates are correctly absent because count
intensity does not depend on them. The full operator computes the volume
intensity map, but heldout outcomes are not loaded and heldout rows are zero
in the cotangent and excluded from metrics. It has one pullback, no intensity
floor, rejects nonfinite row weights and a rank-one outer product that cannot
be represented in float64, and requests20-GiB host memory with a 2-hour
application timeout and a 2.5-hour Slurm cap. The memory request is above
20% headroom over the prior measured 15.26 GiB joint-target peak. Job 439023 has
finished; its scalar timing is not used as a proxy for this reverse pass. The
driver has tightened its inputs and
forward wiring. The VJP now calls the shared `count_intensity_from_field`
used by the count-predictive path, rather than duplicating native-to-count
deposition and volume intensity. It loads only frozen source geometry and
split/exposure metadata; it no longer calls the profile loader that also
reads training FP-mark data. It still computes full-volume intensities, but
only training-exposed positive rows enter the cotangent, and no held-out
outcomes or held-out metric are loaded. The three Gaussian-reference
mechanics tests, five count-forward wiring tests, and twenty synthetic
count/mark holdout-plumbing tests pass; Python compilation, SBATCH shell
syntax, and whitespace checks pass. These are source-level checks only:
N256 execution and PMWD reverse memory remain untested. The first submission
of this one-pullback diagnostic, job 439311, ended after 20 seconds at the
anchor metadata guard: it required a top-level `likelihood_changed` key that
the retained anchor does not write. The target rules, source revision,
checkpoint replay, target orders and training/held-out metadata do match. No
N256 forward intensity map or VJP ran. The guard now uses those recorded
contract fields; two regression tests pass, and `_validated_anchor()` accepts
the actual retained state and its (16,777,240)-coordinate vector. Resubmit
once only after committing this correction and taking a fresh H200-first
snapshot.

The exact `squeue`/`sacct` query on syntax established job 439023 as terminal
`COMPLETED`, exit 0, elapsed 25:19, MaxRSS 10.64 GiB. Its ID was invalid on
grammar, confirming that accounting from the current grammar shell is not
authoritative for syntax jobs.

## Evidence required before implementation or submission

1. Job 439023 completed the corrected cost probe. Direct value-only evaluation
   was about 2.58x faster than the matched scalar route in this bounded
   same-anchor comparison; the one-ULP FP term difference is preserved and
   classified as roundoff-level. Use one fixed scalar backend within a future
   chain; do not infer low-k mixing from this speed result.
2. Do not use job 439023's scalar-value timing as a proxy for an intensity VJP:
   the active map has \(N_{IC}=256^3=16,777,216\) white-field coordinates
   and the count predictor returns \(6\times128^3=12,582,912\) intensities.
   Job 439311 failed at the anchor metadata gate before running the drafted
   one-shot feasibility calculation. The corrected gate now
   uses the fixed anchor to construct only the exact positive-row mask and one
   deterministic random cotangent, then performs one reverse pullback through
   the actual N256 intensity map (no JVP, held-out values, or dense \(J\)). It
   records numerical finiteness, wall time, and peak host/device memory. The raw
   float64 output and one input-gradient vector alone are about 96 MiB and
   128 MiB, respectively; these lower bounds do not include the PMWD reverse
   tape or likelihood workspaces. This one-shot gate is capped at 2 h
   application, 2.5 h Slurm and 20 GiB host RAM. If the pullback is unstable
   or over budget, stop this proposal without escalating the sketch.
   Only if it passes should separate, fixed-budget training-only discovery
   and validation sketches be costed. Compute and inspect the fixed 15x15
   \(F_p\), including its finite/symmetric spectrum and anchor cost.
   Predeclare separate discovery and validation VJP caps,
   projected-spectrum uncertainty, residual-information and subspace-stability
   criteria, and memory/time limits. Do not reuse a heldout value, truth label,
   or candidate selector; do not add a rank/seed ladder.
3. Specify \(U,\Lambda,F_p,C,\mu\) and every block move before a pilot. The
   generic fixed-reference residual identity now has a passing correlated
   Gaussian toy test, but the eventual R2 wiring must independently prove the
   same identity for the complete unchanged CF4 target, including nuisance
   and complementary directions. No bracket, rank, or scale ladder.
4. Cost the full exact target correction and declare one walltime/evaluation
   cap plus useful low-mode and p-block movement criteria. If the basis is
   unstable, p curvature is unusable, setup exceeds the cap, or exact-corrected
   movement remains negligible, close this candidate; do not tune around the
   result.

If these items cannot be met from existing training-only assets, do not submit
this sampler. The 1.5 cMpc/h spacing remains grid sampling, not proven
observational resolution.

## Goal checks

- **Q-GOAL:** this is aimed at the first conditional z=0 density/velocity
  posterior on the approved latent-IC/evolved-state route. The eventual LG
  requirement remains MW/M31 role resolution and unresolved M33 constrained
  by observables on the same NEW evolved LG field at `<=0.3 cMpc/h`; truth IDs
  are calibration/evaluation only.
- **Q-LEAN:** one proposal family and one capped falsification, reusing the
  current target and scalar-evaluator work. No new gravity simulation,
  source census, held-out scoring before draws, or diagnostic ladder.

## 2026-10-30 VJP result and bounded concentration screen

The corrected feasibility job439314 completed on H200 / `gpu:H200:1` (exit0,
4:31, batch MaxRSS5.95GiB). The one training-only positive-row VJP is finite:
white-IC norm957.57, tracer norm230.06, joint norm984.82; device peak43.78GiB.
Among 11,009,805 training-exposed cells, 1,993,850 have positive intensity.
The minimum is1.9562e-37, so the largest `1/sqrt(lambda)` is2.26094e18.
This amplification alone does not establish row influence because the
Jacobian also enters the contribution. No posterior, heldout score or field
was produced. Full run record:
`/gpfs/kjhan/CF4/z0_density/r2_intensity_vjp_feasibility_20261029/439314/result.json`.

Astra recommended one bounded fixed-anchor tail/repeat screen, not a ladder;
the verbatim review is in
`CF4_R2_VJP_TAIL_SCREEN_ASTRA_REVIEW_20261030.md`. The prepared implementation
uses the same anchor/operator and positive-row convention, fixed lowest-1%
intensity tail, two fixed seeds, six full/tail/rest VJPs and one repeated
original-seed full VJP. Its predeclared decomposition/repeat tolerance is
1e-8; tail share itself is not a failure gate. It has one-hour application,
75-minute Slurm and20GiB host-memory caps, no floor/row deletion, and no
heldout access. Five pure NumPy tests, compilation, shell syntax and
whitespace checks pass. This is a numerical diagnostic, not a sketch, sampler,
posterior or R2 completion; at this record it has not yet been submitted.
