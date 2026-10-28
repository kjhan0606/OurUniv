# R2 sampler redesign — Gaussian-reference ESS (design draft, 2026-10-28)

**Status: review/design only. Not code approval, a production sampler, or a GPU
submission.** The previously submitted value-only score-cost probe 439009 was
cancelled before allocation because the startup source-HEAD guard would have
rejected the later documentation commit. It performed no calculation. A
corrected copy of that already-capped probe may be considered separately after
pinning the current source revision and checking typed GPU availability.

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

Use a positive-semidefinite, training-only Gauss–Newton/Fisher approximation
to the observed-count prediction map as a **proposal geometry**, not as a new
likelihood or a claim about posterior rank. Let \(\lambda_c(q)\) be predicted
intensities for every population on every exposed training cell, including
zero-count cells, and \(J=\partial\lambda_c/\partial q\). The sketch uses the
training exposure mask, never held-out count values. A matrix-free sketch of

\[
F_c=J^T\operatorname{diag}(1/\lambda_c)J\succeq0
\]

could provide a candidate low-rank basis for the IC/tracer block. Do not read
raw finite-sketch eigenvalues as a calibrated proposal spectrum.
PMWD's N-body function is a `custom_vjp`, so direct JVPs through evolution
are not supported. A VJP-only randomized Fisher sketch may avoid that
limitation: for fixed independent \(z_k\sim\mathcal N(0,I)\) over exposed
training-cell/population outputs, compute

\[
s_k=J^T W^{1/2}z_k,\qquad \widehat F_c=\frac1r\sum_{k=1}^r s_ks_k^T,
\qquad W=\operatorname{diag}(1/\lambda_c).
\]

Each \(s_k\) is one reverse pullback and
\(\mathbb E[\widehat F_c]=F_c\). Do not treat the nonzero eigenvalues of this
finite sketch as Fisher eigenvalue estimates. For example, if
\(F_c=I_d\) with \(d\gg r\), the rank-r sketch has nonzero eigenvalues of
order \(d/r\), despite being unbiased as a matrix estimator. Use discovery
sketches only to propose a subspace \(U\). Use a separate, independent,
training-only validation sketch to estimate the projected information
\(U^T F_c U\) and residual trace outside \(U\); report uncertainty and
batch-to-batch stability, and fail closed if they do not stabilize under the
fixed sketch cap. These checks can validate proposal geometry, not prove that
the discovered subspace is the posterior's complete informed subspace. The
small Gram matrix is useful for a candidate range, not a license to read its
sample eigenvalues as the true spectrum. This avoids a dense \(J\), not
repeated full backward passes; sketch variance and actual setup cost remain
unknown. No intensity floor is allowed in this proposal metric: nonpositive exposed
intensities must be handled with a separately justified zero-row rule or the
sketch fails closed. First-order reverse sensitivities still require a
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
C_{qt}=U(I+F_U)^{-1}U^T+(I-UU^T),
\]
where \(F_U\) is the validated positive-semidefinite projected information
on the discovery subspace; the orthogonal complement retains prior scale.
The final values of rank, \(F_U\), uncertainty and pass thresholds are not
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
subspace, including information outside it. The count Fisher's exact zero in
the population coordinates requires the separate raw-FP curvature proposal;
this still does not solve field–population coupling. Driver disposition:
adopt these as design constraints, not as approval to implement or submit.
The VJP count, projected-spectrum stability, useful movement criteria, and
memory/time caps remain uncosted. If they cannot be fixed within a small
predeclared budget, close this candidate.

## Evidence required before implementation or submission

1. The prior job 439009 is cancelled without evaluation because its pinned
   source revision did not match the current documentation-bearing HEAD. If
   the corrected, already-approved cost probe is submitted, compare the two
   scalar backends at the same anchor and use one deterministic backend
   consistently within any future chain; do not infer low-k mixing from its
   speed result.
2. Verify that the actual N256 training-count intensity map exposes the
   reverse pullback needed for separate randomized \(F_c\) discovery and
   validation sketches, without JVP, held-out count values, or a dense \(J\).
   Compute and inspect the fixed 15x15 \(F_p\), including its finite/symmetric
   spectrum and anchor cost. Predeclare separate discovery and validation VJP
   caps, projected-spectrum uncertainty, residual-information and
   subspace-stability criteria, and memory/time limits. Do not reuse a heldout
   value, truth label, or candidate selector; do not add a rank/seed ladder.
3. Specify \(U,\Lambda,F_p,C,\mu\) and every block move before a pilot. State
   the residual-target identity in code and test it against known correlated
   Gaussian targets, including nuisance and complement directions. No
   bracket, rank, or scale ladder.
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
