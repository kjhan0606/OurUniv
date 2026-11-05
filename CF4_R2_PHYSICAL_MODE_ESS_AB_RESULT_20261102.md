# R2 fixed physical-mode ESS A/B result — 2026-11-02

## Authoritative disposition

The corrected bounded screen is Syntax Slurm job 447541, run from source
commit `ce38d4cf4e76b361fa600a719562a3b5823824ca` on `syn09`, H100
(`gpu:H100:1`). It completed with Slurm exit 0 in 28:01. The application
status is `PHYSICAL_MODE_ESS_AB_INCOMPLETE_STOP`, not a sampler pass. Preserve
the raw machine output unchanged at
`/gpfs/kjhan/CF4/z0_density/r2_physical_mode_ess_ab_20261102_retry1/447541/result.json`.

The matched exact-GL2 checkpoint replay passed. The direct scalar anchor
reproduced the aggregate anchor score exactly; the raw FP component differs
from the matched record by only `9.09e-13` nat from reduction roundoff. The
job made eight exact target evaluations: one replay, one direct scalar anchor,
four Arm-A proposals, and two Arm-B proposals.

- Arm A, the 16,777,216-coordinate all-IC block, reached its four-evaluation
  bracket cap without an accepted transition.
- Arm B's six-dimensional fundamental (`k^2=1`) block reached its
  two-evaluation cap without an accepted transition. The `k^2=4–8`, IC
  complement and nuisance blocks were never attempted.
- Therefore the A/B comparison and movement ratios are undefined. This closes
  this predeclared one-anchor, fixed-cap candidate only; it is not evidence
  that ESS, physical Fourier updates, or R2 inference are generally
  impossible. Do not extend its bracket, change seeds, or repeat the same
  candidate.

Peak process RSS was 12,407,372 KiB (application report 12.22 GiB); peak GPU
memory was 33,056,607,488 bytes out of a 74,962,354,176-byte allocation limit.
Resource limits did not cause the bracket failures. The job read zero heldout
count values, zero heldout FP marks, and zero native truth IDs; it produced no
posterior sample, uncertainty, or field. No state was promoted and R3 did not
start.

## Interpretation corrections for inherited JSON fields

The common conditional-map runner left several optimizer fields in the raw
JSON. They are stale boilerplate, not facts about job 447541:

- `estimand` says “MAP diagnostic”; this job performed no optimization. It was
  one discarded-anchor ESS transport diagnostic under the frozen conditional
  v6 target.
- `optimizer: L-BFGS-B`, `optimizer_coordinates`, and
  `longer_warm_start_authorized` do not describe this diagnostic and must not
  be used to infer a MAP, a transformed-coordinate run, or authorization for
  a longer sampler.
- The typed-resource snapshot reports three free H200 GPUs. That did **not**
  mean H200 was idle: `syn104` had five of eight H200s allocated. H100 was
  selected consistently with the H200-first policy because H200 was occupied;
  H100 had three free GPUs and both A100 nodes were full. The correct snapshot
  is the pre-submit Slurm allocation record in the master plan, not an
  inference from the free count alone.

All six proposal candidates had worse count log likelihood than the anchor,
by `21,626` to `129,452` nat. The deterioration was mainly in the count term.
These candidates were rejected and are not posterior states. This is
consistent with prior-scale elliptical proposals being poorly matched to the
conditional likelihood, but six rejected candidates cannot identify the
unique cause or quantify an acceptance rate.

The first submission, job 447520, is a separate preserved implementation
failure: it passed checkpoint replay and scalar-anchor equality, then raised a
`NameError` before entering either arm. It is not combined with the scientific
bracket result.

## Independent mid-course review and next direction

Astra's read-only review accepts the failure disposition and recommends
continuing R2 with a materially different, bounded blockwise MALA augmentation
to the existing full-space HMC kernel. A proposal must use projected
gradients of the complete negative log target, a fixed positive-definite
metric, the exact forward/reverse Gaussian proposal-density ratio, and the
complete target difference. The existing HMC transition remains in any later
sampler composition; this screen does not establish MALA efficacy or HMC
stationarity.

Before any larger sampler assessment, the engineering test must verify the
proposal mathematics and projector wiring on synthetic targets, then report
accepted and rejection-inclusive physical-mode movement at multiple existing
discarded states, full-target evaluation cost versus historical HMC, and
whether movement is only downhill transient relaxation. Endpoints remain
discarded. No further ESS bracket or rank/geometry search is authorized by
this result.

The full read-only response is preserved in
`CF4_R2_PHYSICAL_MODE_ESS_AB_ASTRA_REVIEW_20261102.md`. The driver checked the
response against the raw job result and master-plan HMC records: both arms
stopped at the stated bracket caps; the `k^2=4..8`, complement and nuisance
blocks were not reached in Arm B; no movement ratio exists; and the rejected
candidates' count-score losses do not uniquely identify a failure mechanism.
I adopt the candidate closure, no-retry rule and MALA design direction. I do
not interpret the MALA suggestion as proof that the new proposal will mix or
as permission for a posterior run.

One band label in the advisory shorthand required correction before the next
design: the failed ESS plan's 66-coordinate `k^2=4..8` band is not the HMC
drift band recorded in `config/cf4_r2_equilibration_astra_20261011.md`.
That evidence defines `4<|k|<=8`, or `16<k^2<=64` (1,852 real coordinates
at N256). Arm B never reached either band, so this correction is not a
reinterpretation of its result. The driver adopts the MALA method but uses
the actual measured HMC band in the next frozen plan. Read-only projection of
the two saved full gradients gives this shell RMS 10.891962 at state38 and
10.853915 at state60; `k^2=1` gives 110.180225 and 103.125104. These two
states are dependent transport states, not independent chains.

**Q-GOAL:** sampler work is only a prerequisite for the first conditional-v6
N256/1.5 cMpc/h z=0 density/velocity posterior. Heldout prediction,
posterior-conditional uncertainty, sampler convergence, and observation-model
calibration are separate requirements. MW/M31 remain role-ambiguous and M33
unresolved; their observables must later constrain those roles on the same
NEW evolved LG field at `<=0.3 cMpc/h`. Native truth identities are
calibration/evaluation-only.

**Q-LEAN:** stop this fixed-cap ESS candidate, preserve raw output, correct
inherited metadata interpretation, and test only the materially different
full-target-corrected MALA idea. No new RAMSES run, heldout access, truth-ID
selection, field output, or R3 entry.
