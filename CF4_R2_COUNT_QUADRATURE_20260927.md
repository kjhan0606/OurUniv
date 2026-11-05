# R2 count-kernel LOS quadrature check — 2026-09-27

## Scope and decision

Compare the existing one-dimensional Gauss–Hermite LOS rule (GH3) against
higher-order rules at every occupied **training** count key in frozen split
v5. This is a numerical-integration control on one saved, **unconditional**
N128/384 cMpc/h PM state—not a CF4-conditioned field, FP-mark comparison,
held-out prediction, fitted model, or posterior. The 47,542 training points
occupy 38,194 population/voxel keys. Held-out count keys and all FP marks were
excluded.

The single-node accumulation implementation avoids static unrolling of all
LOS nodes in one XLA graph. On A100, job406640 passed all9 focused regressions
and completed GH3/9/15 in3m54s. Job406656 repeated the rule with per-key
diagnostics and completed in3m55s; all9 regressions passed. It requested10
GiB; Slurm MaxRSS was about2.04GiB and the runner's own peak estimate1.45GiB.
GH21 job406668 used the frozen v4 means and completed in56s, MaxRSS1.33GiB
under a2GiB request. The original output arrays/results remain in
`/gpfs/kjhan/CF4/z0_density/r2_count_quadrature_v{2,4,5}/`.

## Results at this fixed state

| Rule vs GH21 | Median key relative change | 95th percentile | 99th percentile | Maximum | Train Poisson log-likelihood delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| GH3 | 0.0952% | 0.6164% | 1.3082% | 37.76% | −4.615 |
| GH9 | 0.0246% | 0.1221% | 0.5733% | 27.14% | −1.895 |
| GH15 | 0.0106% | 0.0475% | 0.3148% | 20.03% | +0.028 |

GH15 and GH21 therefore agree closely in the aggregate training-count score
for this one state. The remaining large *relative* errors are very sparse:
the largest discrepancies are all population3 keys with observed count1 and
small predicted means, whose voxel-center radii are near or just outside the
180 cMpc/h selection edge (roughly179–183 cMpc/h). The GH15-vs-GH21 maximum
has a per-key observed-count log-term change of about0.18, while the full
training log-likelihood changes by only0.028. This is consistent with a
selection-boundary/low-rate tail effect, not proof of a field-independent
bound or exact continuous-redshift convergence.

Implementation history is preserved: job406616 exited before calculation
because a short commit hash failed the source pin; job406617 reached GH3/9
but its statically unrolled GH15 graph requested a55.34GiB GPU allocation and
OOMed. The dynamic single-node path fixed that memory failure. Job406649
completed the GH calculations but failed while serializing NumPy integer
coordinates; job406656 repeated them and saved JSON-safe worst-key details.
These are implementation failures, not science passes.

## Scientific interpretation and next work

- **Q-GOAL:** this tightens one piece of the same-field galaxy observation
  operator needed for the CF4/LG present-state posterior. It does not recover
  density/velocity information, identify the LG, or advance the resolution:
  the tested state is unconditional N128 (3 cMpc/h cells), not the requested
  N256 (1.5 cMpc/h) field and not the LG's <=0.3 cMpc/h zoom posterior.
- **Q-LEAN:** one fixed-state count-map comparison and one GH21 extension; no
  new simulation, training, seed scan, or validation framework. Stop the
  generic quadrature sweep here. Retain the boundary-key limitation, and use
  the exact per-key results only to focus the next same-source comparison.
- The count-map comparison is **not** the still-missing test of the normalized
  continuous individual-redshift/FP mark factor against the count-conditioned
  source law at actual training keys. That source-key factor also lacks a
  calibrated association/point-redshift law, shared multi-member FP-group
  covariance, and selected-group incidence for the3,925 unlinked training
  groups. Do not begin a live-field sampler or call the product a joint
  posterior until those ownership/selection terms are addressed.
- MW/M31/M33 remain latent and ambiguous on any new field, particularly
  unresolved M33. Native identities did not enter this calculation; their
  observable constraints still must act on the same evolved field. This
  result supplies no MW/M31/M33 identification or calibration.

**Status:** R2 present-field posterior remains **NO-GO**. Next deliverable is
the same-source continuous-radius mark/count comparison for frozen training
links, including a tractable exact/supported source-key method; held-out marks
remain untouched. Then specify calibrated one-link association/redshift
ownership, extend to shared-latent multi-link groups, and give unlinked
selected groups an explicit law before field fit or posterior assessment.

Source was committed and pushed through `62c0f64` on
`agent/freeze-zoom-pipeline`. No email or external contact was made.
