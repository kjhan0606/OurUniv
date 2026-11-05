# R2 fundamental-mode score/gradient projection — 2026-10-02

## Question and stage

R2 is still the actual-data present-day density/velocity field inference; it is
not complete. The accepted N256 chain endpoints show unusually large first-axis
latent IC-white Fourier modes. These are coefficients of the canonical input
field, **not** evolved density peaks or identified clusters. The observer is at
the centre of the 384 cMpc/h box. This bounded check asks whether the exact
saved target gradient locally pulls those three modes toward lower amplitude,
balances the Gaussian prior radially, or pulls farther outward.

## Advice and driver disposition

Fable5's read-only advice correctly standardizes the real and imaginary parts
of the orthonormally transformed white field (prior SD `1/sqrt(2)` per part),
and treats the extreme amplitudes as a model-stress signal rather than an
established data misfit. It recommends projecting the saved full gradient at
both predetermined accepted endpoints and deferring more chains.

The driver adopts that small projection, but does **not** adopt a numerical
gradient threshold inferred from `H~6000`. Source inspection confirms that
`fine_gradient = grad(0.5*q.q - log_likelihood)`; the likelihood gradient is
`fine_gradient - q`. The parameter6000 belongs to the sampler's proposed
inverse-Laplacian momentum-mass metric, whose own source comment explicitly
says it is not a fitted Hessian or prior covariance. It therefore cannot be
used as local target curvature to turn a gradient magnitude into a predicted
field displacement. The report will give the signed complex coefficients,
prior/total/likelihood gradient decomposition, and radial directional
derivatives without a post-hoc pass/fail cutoff.

For one non-self-conjugate mode `k`, with `q_k` and total target gradient
`g_k`, the stored diagnostics use the real-field conjugate pair convention:

- Gaussian prior potential contribution: `|q_k|^2`.
- Radial derivative for `q_k -> a*q_k`: `2 Re(conj(q_k) g_k)`.
- Its prior and likelihood pieces are `2|q_k|^2` and
  `2 Re(conj(q_k) (g_k-q_k))`, respectively.

A small radial derivative is only local radial balance in these selected
modes. It is not whole-field stationarity, posterior convergence, ESS,
calibration, or a z=0 map. A positive derivative means local downhill motion
reduces the selected amplitude; a negative derivative means it increases it.
The A/B starts share low-k lineage and are not independent evidence. No
cosmological forward evolution or held-out score is part of this test.

## Bounded execution

Read only the saved exact-GL2 checkpoints from the matched metric6000 controls
410010/410011. Compute orthonormal FFT coefficients of `canonical` and its
saved `fine_gradient` at `(1,0,0)`, `(0,1,0)`, and `(0,0,1)` for each endpoint.
Use one short Slurm allocation, one GPU for the FFT, 1 CPU, 4 GiB host memory,
and a10 partition; write one JSON result under the existing GPFS project
output root. No CF4 rows, held-out data, target/prior edits, density painting,
PM evolution, sampling, or raw checkpoint changes.

Q-GOAL: this is a small R2 diagnosis of whether the data-conditioned objective
supports the observed low-k excursion, before spending on any new chain. It
does not advance the eventual LG-on/off comparison directly.

Q-LEAN: two existing checkpoints and six Fourier modes, with only four FFTs
and no new forward model, simulation, or tuning ladder. Defer per-likelihood
component gradients unless the total projection warrants that additional
attribution.

MW/M31 remain ambiguous and M33 unresolved. Their eventual observables must
constrain roles on the same newly inferred/evolved field; native truth IDs may
label calibration/evaluation only, never seed or select candidates. This
diagnostic does not claim any MW/M31/M33 identification.

Submitted as Slurm job `410046` (a10, 1 GPU, 1 CPU, 4 GiB, 10-minute cap).
Output: `/gpfs/kjhan/CF4/z0_density/r2_n256_lowk_gradient_projection_20261002_v1/`.

## Outcome

Job410046 completed/exit0 in14 seconds. For all six endpoint/mode pairs, the
total negative-log-target gradient has a negative radial derivative. The
likelihood-gradient contribution is outward and larger than the inward prior
contribution:

| Endpoint | Three-mode prior potential (nats) | Prior radial derivative | Likelihood radial derivative | Total radial derivative |
| --- | ---: | ---: | ---: | ---: |
| A | 98.8767 | +197.7535 | -8388.4693 | -8190.7159 |
| B | 99.6086 | +199.2173 | -2519.0880 | -2319.8707 |

The three modes are each strongly non-prior-like (largest standardized real
components about `-9.3 sigma`); A's complex-mode gradients are mostly
anti-aligned radially with its q coefficients, while B has a larger transverse
component. The local exact **partial-target** gradient therefore does not
pull these selected amplitudes back toward the Gaussian prior at either saved
endpoint; locally it favors still larger amplitudes. This supports a target-
stress / likelihood-force explanation over the specific hypothesis that the
current exact target simply relaxes the extreme modes. It does **not** show
that the physical CF4 data support them: the observation/selection model is
still partial and uncalibrated. Nor does one derivative per endpoint establish
trajectory, equilibrium or posterior behavior. A/B share low-k lineage, and
the check covers only the three axial fundamental modes, not the full low-k
band.

Driver disposition: do not extend HMC or alter the Gaussian prior from this
result. The large outward partial-likelihood force is an important model-stress
finding. The smallest decision-useful follow-up is to attribute that same
saved-state low-k likelihood force to the existing count and CF4 distance-mark
components (including their shared nuisance terms), at the same two endpoints;
no new field, forward evolution, or held-out data. This can identify whether a
single factor dominates or whether the conflict is joint. Only after that
attribution should the driver choose a targeted observation-law correction
or close this target as scientifically unusable. Ask Fable5 for advisory
assessment of this next scope because the result is scientifically important;
do not treat the advice as authority or as validation of the current target.

No held-out score was read. R2 remains incomplete; R3 MW/M31/M33 role
identification, R4 precision evolution, and R5 zoom IC delivery have not
advanced. Preserve the fundamental-mode fields as latent IC-white coordinates,
not density structures.

## Fable5 follow-up audit and driver assessment

Fable5 returned **CONDITIONAL PASS**. It independently verified the target
gradient sign, prior conjugate-pair normalization and radial-derivative
arithmetic. It also notes the force is strongly state-dependent: nearly
identical low-k q coefficients at A/B have substantially different gradient
magnitudes and directions. Thus the six negative derivatives are correlated
observations of one phase pattern, not six confirmations, and one state does
not estimate a posterior mean force.

Fable identified an additional descriptive feature: all three coefficients
have phase near `pi`, which under the centered observer / origin convention
places their latent cosine maximum near the observer (about3,16,7 degrees of
phase offset). The driver confirms the coefficients and observer convention,
but does not attach a random-phase p-value to this selected pattern. It remains
a feature of q, not a z=0 mass peak. For a normalizable posterior, the
stationarity identity is `E[q_i * dU/dq_i]=1` for each real coordinate; zero is
not an ensemble radial-derivative target, and no single-state threshold is
appropriate. The JSON key `likelihood_gradient` above denotes the gradient of
the negative log-likelihood, not `grad(log L)`.

Fable judged one exact same-state count-vs-raw-mark attribution Q-GOAL aligned
and Q-LEAN proportionate, conditional on reproducing the saved score/energy
and total gradient, reporting complex per-mode vectors (including radial and
phase directions), all-mode IC-gradient RMS, and all24 nuisance gradients.
The driver adopts those conditions. The same deterministic N256 PMWD forward
must be replayed once per endpoint because only q and the summed gradient were
saved; this is a target-evaluation replay, not another independent or
production simulation. Fable's per-adjoint cost estimate is advisory, based on
prior GL2 timings; the single job is capped and will report incomplete if it
cannot finish both endpoints. See
`CF4_R2_LOWK_COMPONENT_ATTRIBUTION_20261002.md` for the frozen next action.
